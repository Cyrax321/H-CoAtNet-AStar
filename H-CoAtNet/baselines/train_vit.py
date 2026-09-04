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



# ViT Components

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, N, D)
        B, N, D = x.shape
        
        # (B, N, 3*D) -> (B, N, 3, heads, head_dim) -> (3, B, heads, N, head_dim)
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = (attn_weights @ v)
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)
        return output


class MLP(nn.Module):
    """
    A simple two-layer MLP for the Transformer encoder.
    """
    def __init__(self, in_features, hidden_features, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    A single block of the Transformer Encoder.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches, flattens them, and projects them linearly.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        return self.head(cls_output)



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

def plot_curves(history, model_name="ViT"):
    metrics = ['loss', 'acc']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        # Test held-out: not plotted during training (TRIPOD-AI)
        plt.title(f'{model_name} {metric.capitalize()} Curves')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_{metric}_curves.png', dpi=300)
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

    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 3. Model Initialization (ViT-Tiny/Small configuration)
    model = VisionTransformer(
        img_size=TARGET_SIZE[0],
        patch_size=16,
        in_chans=IN_CHANS,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        dropout=0.1
    ).to(DEVICE)

    try:
        print("\n--- Model Summary ---")
        summary(model, input_size=(BATCH_SIZE, IN_CHANS, TARGET_SIZE[0], TARGET_SIZE[1]), device=str(DEVICE))
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
            torch.save(model.state_dict(), 'best_vit_from_scratch.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    # 5. Final Evaluation
    print("\n" + "="*60 + "\n--- Final Evaluation (Test Held-Out, Once) ---\n" + "="*60)
    if os.path.exists('best_vit_from_scratch.pth'):
        model.load_state_dict(torch.load('best_vit_from_scratch.pth'))
        # A* Final Test (Held-Out, Once) -- device-aware
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
            results = {"model": "ViT", "seed": SEED, "test": {"accuracy": float(final_test_acc), "balanced_accuracy": float(bal_acc), "macro": {"precision": float(prec_m), "recall": float(rec_m), "f1": float(f1_m)}, "weighted": {"precision": float(prec_w), "recall": float(rec_w), "f1": float(f1_w)}, "kappa": float(kappa), "mcc": float(mcc), "ece": float(ece), "auroc_macro": float(auroc) if auroc is not None else None, "auprc_macro": float(auprc) if auprc is not None else None, "n": int(len(y_true)), "support_per_class": {str(class_names[i]): int(Counter(y_true)[i]) for i in range(len(class_names))}, "y_true": list(map(int, y_true)), "y_pred": list(map(int, y_pred))}, "per_class": report, "classes": class_names}
            import pathlib
            pathlib.Path("results").mkdir(exist_ok=True)
            with open(f"results/results_vit{SUFFIX}.json", "w") as jf:
                jf.write(json.dumps(results, indent=2))
            print(f"  Saved results/results_vit.json")
        except Exception as e:
            print(f"  Metrics save failed: {e}")
            import traceback; traceback.print_exc()
            print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - ViT (Test Held-Out)')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'confusion_matrix_vit.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        try:
            import sys; sys.path.insert(0, "tools")
            try:
                from in_train_figures import save_in_train_figures
            except ImportError:
                sys.path.insert(0, "H-CoAtNet/tools")
                from in_train_figures import save_in_train_figures
            save_in_train_figures(y_true, y_probs, class_names, f"vit{SUFFIX}")
        except Exception as e:
            print(f"  [Fig] in-train figures skip: {e}")

        plot_curves(history)
        import json as _js; _hp = __import__('pathlib').Path('histories'); _hp.mkdir(exist_ok=True); _hp.joinpath(f'history_vit{SUFFIX}.json').write_text(_js.dumps({'model': 'vit', 'history': history}, indent=2))
    else:
        print("No best model was saved.")


if __name__ == '__main__':
    main()
