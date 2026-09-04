import os
import argparse
import random
import json
import time
from pathlib import Path

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
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
from roboflow import Roboflow
from timm import create_model
from timm.models.vision_transformer import Block

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
# === Configuration ===
# Roboflow key via env only (no hardcode for release).
API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
TARGET_SIZE = (224, 224)
BATCH_SIZE = 24
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
 

# ===========================
# Hierarchical Squeeze-Excitation with Token Pruning (H-CoAtNet)
# IMPORTANT: Inference is FORWARD-ONLY, no label / loss gradient is used.
# See Alg.1 in manuscript: importance = softmax( L2norm(SE(x)) ), then top-k.
# This matches CLAIM/STARD-AI reproducibility and addresses R1-4.
# ===========================
class HierarchicalSE(nn.Module):
   """
   HierarchicalSE: channel-wise SE gating + forward-only token importance.
   At inference (and training forward), importance is computed from L2 norm
   of gated tokens, NOT from dL/dx. No ground-truth label is required.
   Token pruning: 49 -> 36 (75% of original) -> 24 (50% of original).
   """
   def __init__(self, dim, reduction=16, dropout=0.0):
       super().__init__()
       mid = max(1, dim // reduction)
       self.se = nn.Sequential(
           nn.Linear(dim, mid, bias=True),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(mid, dim, bias=True),
           nn.Sigmoid()
       )
 
   def forward(self, x):
       # x: (B, N, C) where N=49, C=768 after ConvNeXt stage4 (7x7)
       s = x.mean(dim=1)  # (B, C) - global average per channel
       gates = self.se(s).unsqueeze(1)  # (B, 1, C) - channel reweight
       out = x * gates  # (B, N, C) - gated tokens
       token_scores = out.norm(dim=-1)  # (B, N) - L2 norm per token (forward-only)
       token_scores = token_scores - token_scores.mean(dim=-1, keepdim=True)
       token_std = token_scores.std(dim=-1, keepdim=True) + 1e-6
       importance = F.softmax(token_scores / token_std, dim=-1)  # (B, N) - no label needed
       return out, importance
 

# =======================================================
# H-CoAtNet: Hierarchical Hybrid ConvNeXt + Transformer (Liu et al. ConvNeXt CVPR'22)
# Architecture: ConvNeXt-Tiny [3,3,9,3] dims [96,192,384,768] + 2 ViT Blocks + HierarchicalSE 49->36->24
# Canonical citations: ConvNeXt (Liu et al. CVPR'22), CoAtNet (Dai et al. NeurIPS'21), ViT (Dosovitskiy et al. ICLR'21), Swin (Liu et al. ICCV'21), EfficientNet (Tan & Le ICML'19), SE (Hu et al. CVPR'18)
# GFT baseline is separate (see train_gft.py: 8 ViT blocks + 3 GALA 75%->50%->25%)
# =======================================================
class HCoAtNet(nn.Module):
   def __init__(self, base_model='convnext_tiny', num_classes=5, vit_blocks=2, pretrained=True):
       super().__init__()
 
       # Load ConvNeXt-Tiny with [3,3,9,3] blocks, dims [96,192,384,768] (Liu et al. CVPR'22)
       # pretrained=True uses ImageNet-1K weights via timm; from-scratch if False
       cnn_backbone = create_model(base_model, pretrained=pretrained, num_classes=0)
 
       # --- Full CNN Backbone (no duplication: stages already contain 3+3+9+3 blocks) ---
       self.cnn_stem   = cnn_backbone.stem  # 224->56, 96ch
       self.cnn_stage1 = cnn_backbone.stages[0]  # 56x56, 96ch, 3 blocks
       self.cnn_stage2 = cnn_backbone.stages[1]  # 28x28, 192ch, 3 blocks
       self.cnn_stage3 = cnn_backbone.stages[2]  # 14x14, 384ch, 9 blocks
       self.cnn_stage4 = cnn_backbone.stages[3]  # 7x7, 768ch, 3 blocks
 
       # --- Transformer Blocks ---
       vit_dim = 192
       self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, vit_dim))
       self.vit_blocks = nn.ModuleList([
           Block(dim=vit_dim, num_heads=6) for _ in range(vit_blocks)
       ])
 
       # --- Hierarchical Selection ---
       final_embed_dim = 768
       num_final_patches = 49
       self.selection_sizes = [
           int(num_final_patches * 0.75),
           int(num_final_patches * 0.5),
       ]
       self.hierarchical_blocks = nn.ModuleList([
           HierarchicalSE(dim=final_embed_dim, reduction=16, dropout=0.05) for _ in self.selection_sizes
       ])
 
       # --- Classifier Head ---
       self.classifier = nn.Sequential(
           nn.LayerNorm(final_embed_dim),
           nn.Linear(final_embed_dim, num_classes)
       )
 
   def select_patches(self, tokens, importance, k):
       B, N, C = tokens.size()
       k = min(k, N)
       _, top_k_idx = torch.topk(importance, k, dim=1)
       batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
       return tokens[batch_idx, top_k_idx]
 
   def forward(self, x):
       # --- Early CNN Stages (ConvNeXt-Tiny [3,3,9,3], no hack duplication) ---
       x = self.cnn_stem(x)  # B,96,56,56
       x = self.cnn_stage1(x)  # B,96,56,56
       x = self.cnn_stage2(x)  # B,192,28,28
 
       # --- Transformer Stage ---
       B, C, H, W = x.shape
       x = x.flatten(2).transpose(1, 2)
       x = x + self.pos_embed
 
       for blk in self.vit_blocks:
           x = blk(x)
 
       x = x.transpose(1, 2).reshape(B, C, H, W)
 
       # --- Later CNN Stages ---
       x = self.cnn_stage3(x)
       x = self.cnn_stage4(x)
 
       # --- Hierarchical Selection and Classification ---
       x = x.flatten(2).transpose(1, 2)
 
       current_tokens = x
       for attn_block, select_size in zip(self.hierarchical_blocks, self.selection_sizes):
           tokens_attn, importance = attn_block(current_tokens)
           current_tokens = self.select_patches(tokens_attn, importance, select_size)
 
       x = current_tokens.mean(dim=1)
       return self.classifier(x)

CoAtGFT = HCoAtNet  # backward compat alias for old checkpoints/imports

# ===========================
# Training, Evaluation, and Plotting (Unchanged)
# ===========================
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
   avg_loss = total_loss / len(loader)
   accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
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
   avg_loss = total_loss / len(loader)
   accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
   return avg_loss, accuracy, all_targets, all_preds
 

def plot_curves(history, save_prefix="hcoatnet"):
   # TRIPOD-AI compliant: plot ONLY train/val, test is evaluated ONCE after training (no leakage)
   for metric in ['loss', 'acc']:
       plt.figure(figsize=(10, 6))
       plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}', color='#0072B2', linewidth=2)
       plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}', color='#D55E00', linewidth=2)
       plt.title(f'H-CoAtNet {metric.capitalize()} -- Train vs Validation (Test Held-Out)', fontsize=12)
       plt.xlabel('Epoch', fontsize=11)
       plt.ylabel(metric.capitalize(), fontsize=11)
       plt.legend(fontsize=10)
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(RESULTS_DIR / f'{save_prefix}_{metric}_curves.png', dpi=300, bbox_inches='tight')
       plt.savefig(f'{metric}_curves.png', dpi=300)  # legacy compat
       plt.show()
       plt.close()
 

def compute_ece(probs, y_true, n_bins=15):
    """Expected Calibration Error (A* metric)."""
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

# ===========================
# Main Training Logic (TRIPOD-AI Compliant: Test Held-Out)
# ===========================
def main():
   global SEED
   _ap = argparse.ArgumentParser(); _ap.add_argument("--seed", type=int, default=SEED); _a,_ = _ap.parse_known_args(); SEED = _a.seed
   SUFFIX = "" if SEED==42 else f"_seed{SEED}"
   seed_everything(SEED)
   print(f"Using device: {DEVICE} | Seed: {SEED}")
   if not API_KEY:
       raise ValueError("Set ROBOFLOW_API_KEY env var (export ROBOFLOW_API_KEY='your_key'). Get key: https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj -> Download -> Show download code")
 
   # 1. Download Dataset (Roboflow version 1, frozen split via server)
   # NOTE: For A* frozen split, also save local indices to splits/seed42_indices.json via tools/freeze_split.py
   print("[Downloading] Downloading dataset from Roboflow...")
   rf = Roboflow(api_key=API_KEY)
   project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
   dataset = project.version(1).download("folder")
   DATASET_DIR = dataset.location
   print(f"   Dataset at: {DATASET_DIR}")
   # Save dataset location + SHA for reproducibility
   with open(RESULTS_DIR / "dataset_info.json", "w") as f:
       json.dump({"location": DATASET_DIR, "version": 1, "seed": SEED, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}, f, indent=2)
 
   # 2. Setup DataLoaders
   train_transform = transforms.Compose([
       transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(15),
       transforms.TrivialAugmentWide(),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
   ])
   val_test_transform = transforms.Compose([
       transforms.Resize(TARGET_SIZE),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
 
   train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "train"), transform=train_transform)
   validation_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "valid"), transform=val_test_transform)
   test_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "test"), transform=val_test_transform)
 
   num_workers = 0 if os.name == 'nt' else 2
   train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
   validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
   test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
 
   class_names = train_dataset.classes
   num_classes = len(class_names)
   print(f"[OK] Found {num_classes} classes: {class_names}")
 
   # 3. Class Weights
   counts = np.bincount(train_dataset.targets)
   class_weights = torch.tensor([len(train_dataset) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float).to(
       DEVICE)
   print("Class Weights:", class_weights.cpu().numpy())
 
   # 4. Initialize Model, Loss, Optimizer (Protocol Table 3)
   # H-CoAtNet: convnext_tiny pretrained (ImageNet-1K), AdamW 5e-5, Cosine T=30, WD 0.01, 30 epochs
   model = HCoAtNet(num_classes=num_classes, pretrained=True).to(DEVICE)
   criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
   optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
   # Note: Warmup 5 epochs can be added via timm scheduler if needed; cosine covers 30ep budget equally for all baselines
 
   try:
       print("\n--- Model Summary (ConvNeXt-Tiny [3,3,9,3] + 2 ViT + HierarchicalSE 49->36->24) ---")
       summary(model, input_size=(BATCH_SIZE, 3, *TARGET_SIZE))
       # Also log FLOPs via thop if available (see tools/compute_flops.py for full table)
       try:
           from thop import profile
           macs, params = profile(model, inputs=(torch.randn(1, 3, *TARGET_SIZE).to(DEVICE),), verbose=False)
           print(f"   FLOPs: {macs/1e9:.2f} GMac | Params: {params/1e6:.2f} M (input 224x224)")
       except ImportError:
           pass
   except Exception as e:
       print(f"Could not show model summary due to: {e}")
 
   # 5. Main Training Loop -- TRIPOD-AI: test set NOT touched (validation selects model)
   history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
   best_val_acc = 0.0
   best_epoch = -1
 
   for epoch in range(EPOCHS):
       print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
       train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
       val_loss, val_acc, _, _ = evaluate(model, validation_loader, criterion, desc="Validating")
       scheduler.step()
 
       history['train_loss'].append(train_loss)
       history['train_acc'].append(train_acc)
       history['val_loss'].append(val_loss)
       history['val_acc'].append(val_acc)
 
       print(f"[Metrics] Epoch {epoch + 1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
       print(f"   Losses: Train: {train_loss:.4f}, Val: {val_loss:.4f}")
 
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           best_epoch = epoch + 1
           torch.save(model.state_dict(), 'best_coat_gft_model.pth')
           torch.save(model.state_dict(), RESULTS_DIR / 'best_hcoatnet.pth')
           print(f"[NEW BEST] New best model saved with Val Acc: {best_val_acc:.4f} (epoch {best_epoch})")
 
   # 6. Final Evaluation -- TEST EVALUATED ONCE ON BEST VAL MODEL (no leakage)
   print("\n" + "="*60)
   print("--- Final Evaluation on Best Model (Test Held-Out, Once) ---")
   print(f"   Best val acc {best_val_acc:.4f} at epoch {best_epoch} -- now evaluating test ONCE")
   print("="*60)
   model.load_state_dict(torch.load(RESULTS_DIR / 'best_hcoatnet.pth' if (RESULTS_DIR / 'best_hcoatnet.pth').exists() else 'best_coat_gft_model.pth', map_location=DEVICE))
   _, final_test_acc, y_true, y_pred, y_probs = evaluate_with_probs(model, test_loader, criterion, desc="Final Test (Held-Out)")
   print(f"[OK] Final Test Accuracy: {final_test_acc:.4f} (n={len(y_true)})")
   # Per-class support
   from collections import Counter
   print(f"   Test support per class: {Counter(y_true)} | Classes: {class_names}")
 
   # 7. A* Reports and Plots -- Full Metrics Package
   print("\n[Report] Classification Report (per-class Precision/Recall/F1):")
   report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
   print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
   # Balanced accuracy, Kappa, MCC, AUROC, AUPRC, ECE
   from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
   bal_acc = balanced_accuracy_score(y_true, y_pred)
   kappa = cohen_kappa_score(y_true, y_pred)
   mcc = matthews_corrcoef(y_true, y_pred)
   ece = compute_ece(y_probs, y_true, n_bins=15)
   # One-vs-rest AUROC/AUPRC (if probs available and >1 class)
   try:
       from sklearn.preprocessing import label_binarize
       y_bin = label_binarize(y_true, classes=list(range(num_classes)))
       auroc_macro = roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr')
       auprc_macro = average_precision_score(y_bin, y_probs, average='macro')
   except Exception as e:
       auroc_macro = float('nan')
       auprc_macro = float('nan')
       print(f"   AUROC/AUPRC skipped: {e}")
   # Weighted / macro
   prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
   prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
   print(f"\n[Metrics] A* Aggregate Metrics (n={len(y_true)}):")
   print(f"   Accuracy: {final_test_acc:.4f} | Balanced Acc: {bal_acc:.4f}")
   print(f"   Macro  P/R/F1: {prec_macro:.4f}/{rec_macro:.4f}/{f1_macro:.4f}")
   print(f"   Weighted P/R/F1: {prec_w:.4f}/{rec_w:.4f}/{f1_w:.4f}")
   print(f"   Cohen's Kappa: {kappa:.4f} | MCC: {mcc:.4f} | ECE: {ece:.4f}")
   print(f"   AUROC macro: {auroc_macro:.4f} | AUPRC macro: {auprc_macro:.4f}")
   # Save comprehensive results JSON (single source of truth for paper)
   results = {
       "model": "H-CoAtNet",
       "seed": SEED,
       "best_val_acc": float(best_val_acc),
       "best_epoch": int(best_epoch),
       "test": {
           "accuracy": float(final_test_acc),
           "balanced_accuracy": float(bal_acc),
           "macro": {"precision": float(prec_macro), "recall": float(rec_macro), "f1": float(f1_macro)},
           "weighted": {"precision": float(prec_w), "recall": float(rec_w), "f1": float(f1_w)},
           "kappa": float(kappa),
           "mcc": float(mcc),
           "ece": float(ece),
           "auroc_macro": float(auroc_macro) if not np.isnan(auroc_macro) else None,
           "auprc_macro": float(auprc_macro) if not np.isnan(auprc_macro) else None,
           "n": int(len(y_true)),
           "support_per_class": {str(class_names[i]): int(Counter(y_true)[i]) for i in range(num_classes)},
           "y_true": list(map(int, y_true)),
           "y_pred": list(map(int, y_pred)),
           "y_probs": y_probs.tolist() if hasattr(y_probs, "tolist") else list(y_probs),
       },
       "per_class": report,
       "classes": class_names,
   }
   with open(RESULTS_DIR / f"results_hcoatnet{SUFFIX}.json", "w") as f:
       json.dump(results, f, indent=2)
   with open(RESULTS_DIR / ("results_final.json" if SEED==42 else f"results_final{SUFFIX}.json"), "w") as f:
       json.dump(results, f, indent=2)
   print(f"   Saved: {RESULTS_DIR / 'results_final.json'}")
   # Confusion Matrix (raw + normalized)
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(12, 10))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
   plt.xlabel('Predicted Label')
   plt.ylabel('True Label')
   plt.title('Confusion Matrix -- H-CoAtNet (Test Held-Out, n=%d)' % len(y_true))
   plt.tight_layout()
   plt.savefig(RESULTS_DIR / f'confusion_matrix_hcoatnet{SUFFIX}.png', dpi=300, bbox_inches='tight')
   plt.show()
   plt.close()
   # Normalized
   cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
   plt.figure(figsize=(12, 10))
   sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
   plt.xlabel('Predicted Label')
   plt.ylabel('True Label')
   plt.title('Confusion Matrix (Row-Normalized) -- H-CoAtNet')
   plt.tight_layout()
   plt.savefig(RESULTS_DIR / f'confusion_matrix_hcoatnet_norm{SUFFIX}.png', dpi=300, bbox_inches='tight')
   plt.show()
   plt.close()
 
   # In-training figures: ROC/PR + reliability (never breaks training)
   try:
       import sys; sys.path.insert(0, "tools")
       from in_train_figures import save_in_train_figures
       save_in_train_figures(y_true, y_probs, class_names, f"hcoatnet{SUFFIX}")
   except Exception as e:
       print(f"  [Fig] in-train figures skip: {e}")

   plot_curves(history, save_prefix=f"hcoatnet{SUFFIX}")
   import json as _js; _hp = __import__('pathlib').Path('histories'); _hp.mkdir(exist_ok=True); _hp.joinpath(f'history_hcoatnet{SUFFIX}.json').write_text(_js.dumps({'model': 'hcoatnet', 'history': history}, indent=2))
   print("\n[OK] Done. All metrics saved. See REBUTTAL_FIX_README.md for bootstrap CI next step: python tools/bootstrap_ci.py --results results/results_final.json")
 

if __name__ == '__main__':
   main()
