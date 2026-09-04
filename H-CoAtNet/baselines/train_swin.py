import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support
from roboflow import Roboflow



# Configuration

# SECURITY: Use env var ROBOFLOW_API_KEY
API_KEY = os.getenv("ROBOFLOW_API_KEY", "")  # env only, no hardcode
TARGET_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
RESULTS_DIR = __import__("pathlib").Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def seed_everything(seed=42):
    import random, numpy as np, os
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
IN_CHANS = 3



# Swin Transformer Components (from scratch)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, H_pad, W_pad, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            shifted_x = x
            attn_mask = None

        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            H_down, W_down = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, H_down, W_down
        return x, H, W, x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=5,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2 ** i),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    downsample=i < self.num_layers - 1
                )
            )
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        for layer in self.layers:
            x_out, H_out, W_out, x, H, W = layer(x, H, W)
        x = self.norm(x_out).transpose(1, 2)
        return self.head(self.avgpool(x).flatten(1))



# Training, Evaluation, and Plotting

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    return avg_loss, accuracy, all_targets, all_preds



def compute_ece(probs, y_true, n_bins=15):
    import numpy as np
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc_bin = (pred == np.array(y_true))
    for i in range(n_bins):
        mask = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i+1])
        if mask.sum() > 0:
            ece += np.abs(acc_bin[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece)

def evaluate_with_probs(model, loader, criterion, desc="Evaluating"):
    model.eval()
    total_loss, all_preds, all_targets, all_probs = 0.0, [], [], []
    import torch.nn.functional as F
    from tqdm import tqdm
    import numpy as np
    import torch
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(loader) if len(loader)>0 else 0.0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if all_preds else 0.0
    return avg_loss, accuracy, all_targets, all_preds, np.array(all_probs)

def plot_curves(history):
    metrics = ['loss', 'acc']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        # Test held-out: not plotted during training (TRIPOD-AI)
        plt.title(f'Model {metric.capitalize()} Curves')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{metric}_curves.png', dpi=300)
        plt.show()



# Main Execution Logic

def main():
    global SEED
    _ap = argparse.ArgumentParser(); _ap.add_argument("--seed", type=int, default=SEED); _a,_ = _ap.parse_known_args(); SEED = _a.seed
    SUFFIX = "" if SEED==42 else f"_seed{SEED}"
    seed_everything(SEED)
    print(f"Using device: {DEVICE} | Seed: {SEED}")
    if API_KEY == "API_KEY_HERE":
        print("[WARNING]  Set ROBOFLOW_API_KEY env var")

    # 1. Download Dataset
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
    dataset = project.version(1).download("folder")
    DATASET_DIR = dataset.location

    # 2. Setup DataLoaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "valid"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 3. Model Initialization
    model = SwinTransformer(num_classes=num_classes).to(DEVICE)

    try:
        print("\n--- Model Summary ---")
        summary(model, input_size=(BATCH_SIZE, IN_CHANS, TARGET_SIZE[0], TARGET_SIZE[1]))
    except Exception as e:
        print(f"Could not show model summary due to: {e}")

    # 4. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE, desc="Validating")
        # test held-out: evaluated once after training (TRIPOD-AI)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
                
        print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Losses: Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_swin_from_scratch.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    # 5. Final Evaluation
    print("\n" + "="*60 + "\n--- Final Evaluation (Test Held-Out, Once) ---\n" + "="*60)
    if os.path.exists('best_swin_from_scratch.pth'):
        model.load_state_dict(torch.load('best_swin_from_scratch.pth'))
        # A* Final Test (Held-Out, Once) with full metrics -- device-aware
        model.eval()
        import torch.nn.functional as F
        y_true, y_pred, y_probs_list = [], [], []
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Final Test (Held-Out)"):
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, pred = outputs.max(1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_probs_list.extend(probs.cpu().numpy())
        import numpy as np
        y_probs = np.array(y_probs_list)
        final_test_acc = (np.array(y_true) == np.array(y_pred)).mean()
        print(f"Final Test Accuracy: {final_test_acc:.4f} (n={len(y_true)})")
        # A* metrics: Balanced Acc, Kappa, MCC, ECE, AUROC/AUPRC
        try:
            from collections import Counter
            import json
            from sklearn.metrics import balanced_accuracy_score
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            ece = compute_ece(y_probs, y_true)
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
            from sklearn.preprocessing import label_binarize
            prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            try:
                y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
                auroc = roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr')
                auprc = average_precision_score(y_bin, y_probs, average='macro')
            except:
                auroc, auprc = None, None
            print(f"  Balanced Acc: {bal_acc:.4f} | Macro F1: {f1_m:.4f} | Kappa: {kappa:.4f} | MCC: {mcc:.4f} | ECE: {ece:.4f} | AUROC: {auroc}")
            from sklearn.metrics import classification_report
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
            report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
            results = {"model": "Swin", "seed": SEED, "test": {"accuracy": float(final_test_acc), "balanced_accuracy": float(bal_acc), "macro": {"precision": float(prec_m), "recall": float(rec_m), "f1": float(f1_m)}, "weighted": {"precision": float(prec_w), "recall": float(rec_w), "f1": float(f1_w)}, "kappa": float(kappa), "mcc": float(mcc), "ece": float(ece), "auroc_macro": float(auroc) if auroc is not None else None, "auprc_macro": float(auprc) if auprc is not None else None, "n": int(len(y_true)), "support_per_class": {str(class_names[i]): int(Counter(y_true)[i]) for i in range(len(class_names))}, "y_true": list(map(int, y_true)), "y_pred": list(map(int, y_pred))}, "per_class": report, "classes": class_names}
            import pathlib
            pathlib.Path("results").mkdir(exist_ok=True)
            with open(f"results/results_swin{SUFFIX}.json", "w") as jf:
                jf.write(json.dumps(results, indent=2))
            print(f"  Saved results/results_swin.json")
        except Exception as e:
            print(f"  Metrics save failed: {e}")
            import traceback; traceback.print_exc()
            print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Swin Transformer (Test Held-Out)')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'confusion_matrix_swin.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        try:
            import sys; sys.path.insert(0, "tools")
            try:
                from in_train_figures import save_in_train_figures
            except ImportError:
                sys.path.insert(0, "H-CoAtNet/tools")
                from in_train_figures import save_in_train_figures
            save_in_train_figures(y_true, y_probs, class_names, f"swin{SUFFIX}")
        except Exception as e:
            print(f"  [Fig] in-train figures skip: {e}")

        plot_curves(history)
    else:
        print("No best model was saved. Skipping final evaluation.")


if __name__ == '__main__':
    main()
