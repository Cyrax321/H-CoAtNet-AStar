import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from timm import create_model
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support
from roboflow import Roboflow



# Configuration

# SECURITY: Use env var ROBOFLOW_API_KEY
API_KEY = "gXuxxWEMFJ8nK73o7pN7"  # Roboflow API key (hardcoded for Colab per user request)
TARGET_SIZE = (224, 224)
BATCH_SIZE = 24
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
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



# GFT Model Definition

class GALA(nn.Module):
    """
    Gradient Attention Learning Alignment (GALA) block.
    """
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., temperature=0.1, ema_alpha=0.9):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.ema_alpha = ema_alpha
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.smoother = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        nn.init.ones_(self.smoother.weight)
        self.register_buffer('ema_grad', None)

    def compute_gradient_attention(self, attn_weights):
        mean_attn = attn_weights
        grad = F.pad(mean_attn, (1, 1), mode='replicate')
        grad = (grad[:, :, 2:] - grad[:, :, :-2]) / 2.0
        importance_scores = grad.abs().sum(dim=1)
        return importance_scores

    def forward(self, x):
        B, N, C = x.shape
        x_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        
        importance_scores = self.compute_gradient_attention(attn_weights)
        importance_scores = self.smoother(importance_scores.unsqueeze(1)).squeeze(1)

        if self.training:
            # Check if ema_grad is uninitialized or if the batch size has changed
            if self.ema_grad is None or self.ema_grad.shape[0] != B:
                self.ema_grad = torch.zeros_like(importance_scores)
            
            # Update EMA
            self.ema_grad.copy_(self.ema_alpha * self.ema_grad + (1 - self.ema_alpha) * importance_scores.detach())
            final_scores = 0.5 * (importance_scores + self.ema_grad)
        else:
            final_scores = importance_scores

        final_scores = (final_scores - final_scores.mean(dim=-1, keepdim=True)) / (
            final_scores.std(dim=-1, keepdim=True) + 1e-6)
        final_scores = F.softmax(final_scores / self.temperature, dim=-1)
        
        return x_out, final_scores


class GFT(nn.Module):
    """
    Gradient Focal Transformer (GFT)
    """
    def __init__(self, base_model='vit_tiny_patch16_224', num_classes=5, pretrained=True):
        super().__init__()
        self.base_vit = create_model(base_model, pretrained=pretrained)
        embed_dim = self.base_vit.embed_dim
        num_patches = self.base_vit.patch_embed.num_patches
        
        self.patch_embed = self.base_vit.patch_embed
        self.cls_token = self.base_vit.cls_token
        self.pos_embed = self.base_vit.pos_embed
        self.base_encoder = nn.Sequential(*self.base_vit.blocks[:8])
        
        self.pruning_schedule = [
            int(num_patches * 0.75),
            int(num_patches * 0.50),
            int(num_patches * 0.25)
        ]
        
        num_heads_from_vit = self.base_vit.blocks[0].attn.num_heads
        self.gala_stages = nn.ModuleList([
            GALA(dim=embed_dim, num_heads=num_heads_from_vit) for _ in range(3)
        ])
        
        self.norm = self.base_vit.norm
        self.head = nn.Linear(embed_dim, num_classes)
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def select_patches(self, tokens, pos_embed, importance_scores, k):
        _, top_k_indices = torch.topk(importance_scores, k=k, dim=1)
        top_k_indices = top_k_indices.sort(dim=1)[0]
        
        batch_idx = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
        pruned_tokens = tokens[batch_idx, top_k_indices]
        pruned_pos_embed = pos_embed[batch_idx, top_k_indices]
        
        return pruned_tokens, pruned_pos_embed

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.base_encoder(x)

        cls_token, patch_tokens = x[:, :1], x[:, 1:]
        patch_pos_embed = self.pos_embed[:, 1:, :]

        for i in range(len(self.gala_stages)):
            current_tokens = torch.cat((cls_token, patch_tokens), dim=1)
            updated_tokens, importance_scores = self.gala_stages[i](current_tokens)
            
            cls_token = updated_tokens[:, :1]
            patch_tokens = updated_tokens[:, 1:]
            patch_importance = importance_scores[:, 1:]
            
            k = self.pruning_schedule[i]
            patch_tokens, patch_pos_embed = self.select_patches(
                patch_tokens, 
                patch_pos_embed.expand(B, -1, -1),
                patch_importance, 
                k
            )

        final_tokens = torch.cat((cls_token, patch_tokens), dim=1)
        final_tokens = self.norm(final_tokens)
        return self.head(final_tokens[:, 0])



# Training, Evaluation, and Plotting

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if len(all_preds) > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, loader, criterion, desc="Evaluating"):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if len(all_preds) > 0 else 0.0
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
        plt.title(f'Model {metric.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{metric}_curves.png', dpi=300)
        plt.show()



# Main Execution Logic

def main():
    seed_everything(SEED)
    print(f"Using device: {DEVICE} | Seed: {SEED}")
    if API_KEY == "API_KEY_HERE":
        print("⚠️  Set ROBOFLOW_API_KEY env var")

    # 1. Download Dataset
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
    dataset = project.version(1).download("folder", overwrite=False)
    DATASET_DIR = dataset.location

    # 2. Setup Transforms and Loaders
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
    validation_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "valid"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 3. Model Initialization
    model = GFT(num_classes=num_classes, pretrained=True).to(DEVICE)
    
    # Class Weights
    counts = np.bincount(train_dataset.targets)
    class_weights = torch.tensor(
        [len(train_dataset) / (c * num_classes + 1e-6) for c in counts], 
        dtype=torch.float
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    try:
        print("\n--- Model Summary ---")
        summary(model, input_size=(BATCH_SIZE, 3, *TARGET_SIZE))
    except Exception as e:
        print(f"Could not show model summary: {e}")

    # 4. Training Loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, validation_loader, criterion, desc="Validating")
        # test held-out: no evaluation during training

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
                
        print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Losses: Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gft_model.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    # 5. Final Evaluation
    print("\n" + "="*60 + "\n--- Final Evaluation (Test Held-Out, Once) ---\n" + "="*60)
    if os.path.exists('best_gft_model.pth'):
        model.load_state_dict(torch.load('best_gft_model.pth'))
        _, final_test_acc, y_true, y_pred, y_probs = evaluate_with_probs(model, test_loader, criterion, desc="Final Test (Held-Out)")
        print(f"Final Test Accuracy: {final_test_acc:.4f} (n={len(y_true)})")
        # A* metrics
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
            report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
            results = {"model": "GFT", "seed": SEED, "test": {"accuracy": float(final_test_acc), "balanced_accuracy": float(bal_acc), "macro": {"precision": float(prec_m), "recall": float(rec_m), "f1": float(f1_m)}, "weighted": {"precision": float(prec_w), "recall": float(rec_w), "f1": float(f1_w)}, "kappa": float(kappa), "mcc": float(mcc), "ece": float(ece), "auroc_macro": float(auroc) if auroc else None, "auprc_macro": float(auprc) if auprc else None, "n": int(len(y_true)), "support_per_class": {str(class_names[i]): int(Counter(y_true)[i]) for i in range(len(class_names))}, "y_true": list(map(int, y_true)), "y_pred": list(map(int, y_pred)) }, "per_class": report, "classes": class_names}
            import pathlib
            Path("results").mkdir(exist_ok=True)
            with open(f"results/results_gft.json", "w") as jf:
                jf.write(json.dumps(results, indent=2))
            with open(f"results/results_final_gft.json", "w") as jf:
                jf.write(json.dumps(results, indent=2))
            print(f"  Saved results/results_gft.json")
        except Exception as e:
            print(f"  Metrics save failed: {e}")
            import traceback; traceback.print_exc()
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print(f"Final Test Accuracy: {final_test_acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - GFT Model')
        plt.savefig(RESULTS_DIR / 'confusion_matrix_gft.png', dpi=300)
        plt.show()

        plot_curves(history)
    else:
        print("No best model was saved. Skipping final evaluation.")


if __name__ == '__main__':
    main()
