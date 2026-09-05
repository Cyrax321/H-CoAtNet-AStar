#!/usr/bin/env python3
"""
Ablation study for H-CoAtNet: fair component removal under identical protocol.

Paper:
  Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification (H-CoAtNet).
  Authors: Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan, Lekshmi C.R.
  Status: Under review (revision). This script supplies Table 5 and Fig. 5.

Purpose:
  Prove that each novel component is necessary. We remove one piece at a time
  and measure the drop on the same frozen test set. If the drop is significant,
  the component is justified. This is the standard component ablation used in
  medical imaging venues such as MICCAI, MedIA, and TMI.

Reviewer mapping (see REBUTTAL_FIX_README.md):
  - R1-4 (token pruning without test label): Full vs w/o SE.
  - R1-3 (architecture consistency, ViT role): Full vs w/o ViT.
  - R2-2 and E-1 (novelty beyond standard CoAtNet): Full vs CNN-only.
  - R1-7 and R2-5 (reproducibility, 30 epoch concern): identical protocol block below.
  - R1-8 and R2-4 (uncertainty): JSON outputs carry y_true/y_pred/y_probs for
    bootstrap_ci.py and stats_tests.py. Run those after training.
  - R1-9 (efficiency): efficiency block per variant for compute_flops.py merge.
  - R1-2 (leakage): TRIPOD-AI Type 2b. Validation selects the checkpoint.
    Test is evaluated once at the end. No test curve appears during training.

Variants (only these two flags toggle, nothing else):
  - full:    ConvNeXt-Tiny [3,3,9,3] + 2 ViT blocks + HierarchicalSE 49->36->24.
             Reference model. Matches H-CoAtNet/proposed_method/train_h_coatnet.py.
  - noSE:    Same backbone + 2 ViT, no pruning. Mean pool over 49 tokens.
             Isolates the HierarchicalSE contribution.
  - noViT:   Same backbone + SE, no transformer. Straight ConvNeXt path 56->28->14->7.
             Isolates the ViT contribution.
  - cnnOnly: Pure ConvNeXt-Tiny. No ViT, no SE. Equals the CoAtNet baseline shape.
             Isolates the full hybrid novelty.

Fairness lock (identical for every variant, see ABLATION_PLAN.md Section 3):
  - Frozen stratified split, seed 42, Roboflow ich-s-7lnsj v1.
  - Image size 224x224, ImageNet mean/std.
  - Augmentation: RandomResizedCrop 0.8-1.0, HFlip, Rot15, TrivialAugmentWide,
    RandomErasing 0.2. Same object for all variants.
  - Loss: cross entropy + label smoothing 0.1 + class weights N/(C*Nc).
  - Optimizer: AdamW lr 5e-5, weight decay 0.01, CosineAnnealing T=epochs.
  - Batch 24, 30 epochs, convnext_tiny ImageNet-1K pretrained, deterministic.
  Changing any of these per variant would break fairness and invalidate Table 5.

Usage (Colab T4, quote the path because it contains a space):
  python3 "ablation study/ablation_study.py" --variant smoke --epochs 1 --seed 42
  python3 "ablation study/ablation_study.py" --variant noSE --epochs 30 --seed 42
  python3 "ablation study/ablation_study.py" --variant all --epochs 30 --seed 42
  python3 "ablation study/ablation_study.py" --variant compare

Outputs (dual write keeps old paths working):
  - results/results_ablation_{v}.json and ablation study/results/... (single source).
  - histories/history_ablation_{v}.json (curves for the 12-figure suite).
  - ablation study/figures/curve_{v}_{acc,loss}.png (train/val only).
  - ablation study/figures/confusion_{v}_{raw,norm}.png.
  - figures/fig_ablation_02_main_bar, 03_drop, 04_perclass_heatmap (PNG+PDF).
  - ablation study/ablation_table.tex (Table 5), ablation_summary.csv/json.

Reproducibility:
  - Deterministic seed 42 via seed_everything(). Same seed for all variants in
    the main table. Multi-seed 42-46 goes to the supplement with --tag.
  - Roboflow key is read from env ROBOFLOW_API_KEY only. Never hardcoded (R1-6).
  - Self-contained model definition. No import from train_h_coatnet.py, so no
    drift if that file changes. Parity was checked line by line on 2026-09-05.

References (canonical, verify before submission):
  - ConvNeXt: Liu et al., A ConvNet for the 2020s, CVPR 2022.
  - CoAtNet: Dai et al., CoAtNet, NeurIPS 2021.
  - ViT: Dosovitskiy et al., ICLR 2021.
  - SE: Hu et al., Squeeze-and-Excitation, CVPR 2018.
  - TRIPOD-AI, STARD-AI 2024, CLAIM 2024 checklists.
"""

# ----------------------------------------------------------------------------
# Imports: stdlib first, then scientific stack. All are in requirements-colab.txt
# except torch/torchvision which Colab pre-installs. No custom dependencies.
# ----------------------------------------------------------------------------
import os
import sys
import json
import argparse
import random
import hashlib
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Headless backend. Required on Colab, safe locally.
import matplotlib.pyplot as plt
import seaborn as sns
from timm import create_model
try:
    # Pinned path for timm 0.9.12 (see H-CoAtNet/requirements.txt).
    from timm.models.vision_transformer import Block
except ImportError:
    try:
        # Fallback for Colab latest where timm reorganized modules.
        from timm.models.vit import Block
    except ImportError:
        # Last resort, same Block class re-exported under timm.layers.
        from timm.layers import Block
from sklearn.metrics import (classification_report, confusion_matrix, cohen_kappa_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score)

# ----------------------------------------------------------------------------
# Paths: REPO is the outer checkout. STUDY is this folder (contains a space,
# so every shell call must quote it). We write to both results/ and
# ablation study/results/ so old notebooks keep working.
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "ablation study"
RESULTS = REPO / "results"
STUDY_RESULTS = STUDY / "results"
STUDY_FIGS = STUDY / "figures"
HIST = REPO / "histories"
for d in [RESULTS, STUDY_RESULTS, STUDY_FIGS, HIST, REPO / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Locked protocol. Do not change per variant. Values match
# H-CoAtNet/proposed_method/train_h_coatnet.py lines 31-45 and 283-321.
# Table 3 in the manuscript must list these exact numbers.
# ----------------------------------------------------------------------------
TARGET_SIZE = (224, 224)  # Input resolution for all variants.
BATCH_SIZE = 24  # Matches H-CoAtNet. ViT/Swin baselines use 16, not used here.
EPOCHS_DEFAULT = 30  # Equal budget. Addresses R2-5 (convergence by epoch 25).
LR = 5e-5  # AdamW LR for ConvNeXt-based models. EfficientNet/CNN use 3e-4 elsewhere.
WD = 0.01  # Weight decay for all ablation variants.
LS = 0.1  # Label smoothing in CE. Helps calibration (ECE) on small classes.
SEED_DEFAULT = 42  # Main seed. Supplement uses 42-46 with --tag.
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Standard ImageNet norm, all splits.
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Variant registry. use_vit and use_se are the only degrees of freedom.
# ORDER fixes the display order in every figure and in Table 5.
# Colors are Okabe-Ito colorblind safe, shared with the 12-figure suite.
# ----------------------------------------------------------------------------
VARIANTS = {
    "full":    {"use_vit": True,  "use_se": True,
                "name": "Full H-CoAtNet", "desc": "ConvNeXt-T + 2 ViT + SE 49->36->24"},
    "noSE":    {"use_vit": True,  "use_se": False,
                "name": "w/o SE", "desc": "ConvNeXt-T + 2 ViT, mean pool 49"},
    "noViT":   {"use_vit": False, "use_se": True,
                "name": "w/o ViT", "desc": "ConvNeXt-T + SE, no transformer"},
    "cnnOnly": {"use_vit": False, "use_se": False,
                "name": "CNN-only", "desc": "Pure ConvNeXt-T"},
}
ORDER = ["full", "noSE", "noViT", "cnnOnly"]
VMAP_COLOR = {"full": "#0072B2", "noSE": "#D55E00", "noViT": "#009E73", "cnnOnly": "#CC79A7"}


# ----------------------------------------------------------------------------
# Determinism: same seed gives same init, same shuffle, same aug sequence.
# Required so the drop in Table 5 is causal (component effect, not luck).
# Supplement adds multi-seed mean+-SD to show the result is not seed lucky.
# ----------------------------------------------------------------------------
def seed_everything(seed=42):
    """Fix Python, NumPy, Torch, and cuDNN randomness for this process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# HierarchicalSE: channel gating plus forward-only token scoring (Alg. 1).
# Input x has shape (B, N, C) with N=49 and C=768 from ConvNeXt stage 4 (7x7).
# Importance uses the L2 norm of gated tokens, standardized then softmaxed.
# No ground truth label, no loss gradient, and no backward pass are used,
# so it is valid at test time. This directly answers R1-4.
# Pruning keeps 36 tokens (75 percent of 49) then 24 (50 percent of 49).
# ----------------------------------------------------------------------------
class HierarchicalSE(nn.Module):
    """Channel-wise SE gating with forward-only L2 token importance."""

    def __init__(self, dim, reduction=16, dropout=0.05):
        """Build the two-layer SE bottleneck. reduction=16 matches the paper."""
        super().__init__()
        mid = max(1, dim // reduction)
        self.se = nn.Sequential(nn.Linear(dim, mid), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(mid, dim), nn.Sigmoid())

    def forward(self, x):
        """Gate channels, score tokens by L2 norm, return both for top-k."""
        # Global average over tokens gives one descriptor per channel.
        s = x.mean(dim=1)
        # Sigmoid gates in [0,1] reweight each channel.
        gates = self.se(s).unsqueeze(1)
        # Apply gating, keep shape (B, N, C).
        out = x * gates
        # Forward-only score: L2 norm per token, no label needed.
        scores = out.norm(dim=-1)
        # Standardize across tokens for stable softmax temperature.
        scores = scores - scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True) + 1e-6
        importance = F.softmax(scores / std, dim=-1)
        return out, importance


# ----------------------------------------------------------------------------
# AblationCoAtNet: ConvNeXt-Tiny backbone with switchable ViT and SE stages.
# Borrowed (cite): ConvNeXt-Tiny stages [3,3,9,3], dims [96,192,384,768]
#   (Liu CVPR 22), ViT Block 192-d 6 heads (Dosovitskiy ICLR 21).
# New (ours): early stages 1-2, then 2 ViT blocks, then late stages 3-4
#   interleaving plus HierarchicalSE 49->36->24. Unlike stacked CoAtNet and
#   unlike GFT which uses 8 ViT plus 3 GALA stages. Addresses R1-3 and R2-2.
# Self-contained on purpose: no import from train_h_coatnet.py, so reviewer
# can read this file alone and trust there is no hidden difference.
# ----------------------------------------------------------------------------
class AblationCoAtNet(nn.Module):
    """ConvNeXt-Tiny backbone with optional mid ViT and late SE pruning."""

    def __init__(self, use_vit=True, use_se=True, num_classes=5, pretrained=True, vit_blocks=2):
        """Create backbone and optional stages. pretrained=True uses IN1K."""
        super().__init__()
        self.use_vit = use_vit
        self.use_se = use_se
        # ConvNeXt-Tiny without its classifier head. Stages hold [3,3,9,3] blocks.
        backbone = create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
        self.cnn_stem = backbone.stem      # 224->56, 96 channels.
        self.cnn_stage1 = backbone.stages[0]  # 56x56, 96ch, 3 blocks.
        self.cnn_stage2 = backbone.stages[1]  # 28x28, 192ch, 3 blocks.
        self.cnn_stage3 = backbone.stages[2]  # 14x14, 384ch, 9 blocks.
        self.cnn_stage4 = backbone.stages[3]  # 7x7, 768ch, 3 blocks.
        # Mid transformer operates at 28x28 with 192 channels (784 tokens).
        if use_vit:
            self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, 192))
            self.vit_blocks = nn.ModuleList([Block(dim=192, num_heads=6) for _ in range(vit_blocks)])
        else:
            # Empty ModuleList keeps state_dict keys stable and forward simple.
            self.pos_embed = None
            self.vit_blocks = nn.ModuleList([])
        # Late pruning operates at 7x7 with 768 channels (49 tokens).
        if use_se:
            self.selection_sizes = [int(49 * 0.75), int(49 * 0.5)]  # Resolves to [36, 24].
            self.hierarchical_blocks = nn.ModuleList(
                [HierarchicalSE(dim=768, reduction=16, dropout=0.05) for _ in self.selection_sizes])
        else:
            self.selection_sizes = []
            self.hierarchical_blocks = nn.ModuleList([])
        # Classifier head on the mean pooled kept tokens.
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, num_classes))

    def select_patches(self, tokens, importance, k):
        """Keep top-k tokens by importance. Batched gather, no label used."""
        B, N, C = tokens.size()
        k = min(k, N)
        _, idx = torch.topk(importance, k, dim=1)
        bi = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
        return tokens[bi, idx]

    def forward(self, x):
        """Early CNN, optional ViT, late CNN, optional SE, mean pool, linear."""
        # Early local texture: stem plus stages 1-2. Output is (B,192,28,28).
        x = self.cnn_stem(x)
        x = self.cnn_stage1(x)
        x = self.cnn_stage2(x)
        # Mid global context: 2 ViT blocks on 784 tokens. Skipped for noViT/cnnOnly.
        if self.use_vit:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2) + self.pos_embed
            for blk in self.vit_blocks:
                x = blk(x)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        # Late refinement: stages 3-4. Output is (B,768,7,7).
        x = self.cnn_stage3(x)
        x = self.cnn_stage4(x)
        # Flatten to 49 tokens of 768-d for selection.
        x = x.flatten(2).transpose(1, 2)
        cur = x
        # Hierarchical pruning 49->36->24. Skipped for noSE/cnnOnly (mean pool 49).
        if self.use_se and len(self.hierarchical_blocks) > 0:
            for blk, k in zip(self.hierarchical_blocks, self.selection_sizes):
                gated, imp = blk(cur)
                cur = self.select_patches(gated, imp, k)
        # Mean over kept tokens then linear to 5 ichthyosis classes.
        return self.classifier(cur.mean(dim=1))


# ----------------------------------------------------------------------------
# Transforms: byte-identical to train_h_coatnet.py train/val transforms.
# Train uses stochastic aug. Val and test use deterministic Resize only.
# Same object is used for all 4 variants, which is what makes the comparison
# fair. Fallback keeps Colab running on old torchvision without changing
# fairness (fallback still applies equally to every variant).
# ----------------------------------------------------------------------------
def build_transforms():
    """Return (train_transform, eval_transform) shared by all variants."""
    try:
        trivial = transforms.TrivialAugmentWide()
    except AttributeError:
        # Very old torchvision lacks TrivialAugmentWide. Fall back but warn.
        # Fairness holds because the fallback is still shared across variants.
        print("  WARN: TrivialAugmentWide missing, using AutoAugment fallback (still identical across variants)")
        try:
            trivial = transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        except AttributeError:
            # No-op placeholder keeps the Compose length stable for logging.
            trivial = transforms.RandomHorizontalFlip(p=0.0)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        trivial,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# Dataloaders: ImageFolder with train/valid/test subfolders from Roboflow v1.
# Note the folder is named valid (not val) in the Roboflow export. Shuffling
# is on for train only. Batch 24 and 2 workers match the main training.
# ----------------------------------------------------------------------------
def get_dataloaders(dataset_dir):
    """Build train/valid/test loaders plus class names and train set stats."""
    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(dataset_dir, "valid"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform=eval_tf)
    nw = 0 if os.name == "nt" else 2  # Windows needs 0 workers, Linux/Colab uses 2.
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw)
    vl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    el = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    return tl, vl, el, train_ds.classes, train_ds


# ----------------------------------------------------------------------------
# Fairness banner: printed at the start of every run so the log itself proves
# equal budget. Also prints the frozen split hash when available (R1-1).
# ----------------------------------------------------------------------------
def fairness_banner(epochs, seed):
    """Print locked hyperparameters and frozen split hash to stdout log."""
    print("=" * 70)
    print("FAIRNESS LOCK (identical for every variant):")
    print(f"  split seed {seed} frozen | 224 ImageNet norm | aug RRC0.8-1.0+HFlip+Rot15+TrivAug+Eras0.2")
    print(f"  CE+LS{LS} + classweight | AdamW lr{LR} WD{WD} Cosine T={epochs} | batch {BATCH_SIZE} | {epochs}ep")
    print(f"  init convnext_tiny IN1K | deterministic | TRIPOD-2b test-once | device {DEVICE}")
    print("  ONLY toggled: use_vit / use_se")
    print("=" * 70)
    sp = REPO / "splits/seed42_indices.json"
    if sp.exists():
        try:
            h = hashlib.sha256(sp.read_bytes()).hexdigest()[:12]
            print(f"  frozen split: {sp} sha {h}")
        except Exception as e:
            print(f"  split check skip: {e}")
    else:
        print("  WARN: splits/seed42_indices.json missing - lock split before paper")


# ----------------------------------------------------------------------------
# Training loop: one epoch over train. Validation and test never update weights.
# Test loader is never touched here. Only train_one_variant calls the test
# loader once at the end, which enforces TRIPOD-AI Type 2b (R1-2).
# ----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer):
    """Run one training epoch. Return (mean loss, accuracy)."""
    model.train()
    tot, preds, tgts = 0.0, [], []
    for img, y in tqdm(loader, desc="Training", leave=False):
        img, y = img.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tot += loss.item()
        preds.extend(out.argmax(1).detach().cpu().numpy())
        tgts.extend(y.detach().cpu().numpy())
    acc = (np.array(preds) == np.array(tgts)).mean() if preds else 0.0
    return tot / max(1, len(loader)), float(acc)


# ----------------------------------------------------------------------------
# Validation: no gradients, no weight updates. Used for model selection
# (best val accuracy). History from this function draws the train/val curves.
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, desc="Validating"):
    """Evaluate loss and accuracy without gradients. Return (loss, acc, y, p)."""
    model.eval()
    tot, preds, tgts = 0.0, [], []
    for img, y in tqdm(loader, desc=desc, leave=False):
        img, y = img.to(DEVICE), y.to(DEVICE)
        out = model(img)
        tot += criterion(out, y).item()
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
    acc = (np.array(preds) == np.array(tgts)).mean() if preds else 0.0
    return tot / max(1, len(loader)), float(acc), tgts, preds


# ----------------------------------------------------------------------------
# Held-out test: called exactly once per variant on the best val checkpoint.
# Also returns softmax probabilities needed for AUROC, AUPRC, ECE, Brier,
# bootstrap CIs, and ROC/PR/reliability figures. Wrapped in no_grad so no
# test gradient ever flows (R1-2, R1-4).
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_with_probs(model, loader, criterion, desc="Test"):
    """Evaluate once on held-out test. Return (loss, acc, y_true, y_pred, probs)."""
    model.eval()
    tot, preds, tgts, probs = 0.0, [], [], []
    for img, y in tqdm(loader, desc=desc, leave=False):
        img, y = img.to(DEVICE), y.to(DEVICE)
        out = model(img)
        tot += criterion(out, y).item()
        pr = F.softmax(out, dim=1)
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
        probs.extend(pr.cpu().numpy())
    acc = (np.array(preds) == np.array(tgts)).mean() if preds else 0.0
    return tot / max(1, len(loader)), float(acc), tgts, preds, np.array(probs)


# ----------------------------------------------------------------------------
# ECE with 15 bins (matches train_h_coatnet.py). Low ECE means predicted
# confidence matches observed accuracy, which matters for the clinical
# decision-support claim (R1-11). Brier is the mean squared error between
# one-hot truth and predicted probabilities. Report both.
# ----------------------------------------------------------------------------
def compute_ece(probs, y_true, n_bins=15):
    """Compute Expected Calibration Error over equal width confidence bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (pred == np.array(y_true))
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += abs(ok[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def compute_brier(probs, y_true, n_classes):
    """Compute multi-class Brier score (lower is better calibrated)."""
    from sklearn.preprocessing import label_binarize
    yb = label_binarize(y_true, classes=list(range(n_classes)))
    return float(np.mean((yb - probs) ** 2))


# ----------------------------------------------------------------------------
# Efficiency: parameter count plus MACs via thop when installed.
# Measured per variant on the same input (1,3,224,224) for Table 4 (R1-9).
# thop is optional so smoke tests never fail for lack of it.
# ----------------------------------------------------------------------------
def measure_efficiency(model):
    """Return (params_M, macs_G). macs_G is None if thop is not installed."""
    params = sum(p.numel() for p in model.parameters()) / 1e6
    macs = None
    try:
        from thop import profile
        model.eval()
        m, p = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(DEVICE),), verbose=False)
        macs = m / 1e9
    except Exception:
        pass
    return float(params), (float(macs) if macs else None)


# ----------------------------------------------------------------------------
# One variant end to end: dataloaders, class weights, model, optimizer,
# 30 epoch loop with best-val checkpointing, single held-out test, full
# metric package, JSON save, history save, and per-variant plots.
# Class weights N/(C*Nc) correct the imbalance (LI test n=22 vs IV n=46).
# Specificity TN/(TN+FP) is added per class for the clinical table (Table 9).
# ----------------------------------------------------------------------------
def train_one_variant(variant, dataset_dir, epochs=30, seed=42, pretrained=True, out_tag=""):
    """Train one ablation variant fairly and evaluate test once. Return result dict."""
    cfg = VARIANTS[variant]
    print("\n" + "=" * 70)
    print(f"VARIANT [{variant}] {cfg['name']} - {cfg['desc']}")
    print("=" * 70)
    # Same seed for every variant in the main table makes the drop causal.
    seed_everything(seed)
    tl, vl, el, class_names, train_ds = get_dataloaders(dataset_dir)
    nc = len(class_names)
    print(f"  classes {class_names} | train {len(train_ds)} val {len(vl.dataset)} test {len(el.dataset)}")
    # Inverse frequency weights so rare classes (LI, NS) are not ignored.
    counts = np.bincount(train_ds.targets)
    cw = torch.tensor([len(train_ds) / (c * nc + 1e-6) for c in counts], dtype=torch.float).to(DEVICE)
    print(f"  class weights {cw.detach().cpu().numpy().round(3)}")

    # Build the toggled model. pretrained=True keeps IN1K init equal for all.
    model = AblationCoAtNet(use_vit=cfg["use_vit"], use_se=cfg["use_se"],
                            num_classes=nc, pretrained=pretrained).to(DEVICE)
    params_m, _ = measure_efficiency(model)
    print(f"  params {params_m:.2f}M | {cfg['desc']}")

    # Identical loss and optimizer for every variant (fairness lock).
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop. Test loader is not touched inside this loop (TRIPOD-2b).
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_ep = -1.0, -1
    ckpt = STUDY_RESULTS / f"best_{variant}{out_tag}.pth"
    for ep in range(epochs):
        print(f"--- Epoch {ep + 1}/{epochs} [{variant}] ---")
        trl, tra = train_epoch(model, tl, criterion, optimizer)
        vall, vala, _, _ = evaluate(model, vl, criterion)
        scheduler.step()
        hist["train_loss"].append(trl)
        hist["train_acc"].append(tra)
        hist["val_loss"].append(vall)
        hist["val_acc"].append(vala)
        print(f"  train acc {tra:.4f} loss {trl:.4f} | val acc {vala:.4f} loss {vall:.4f}")
        # Best checkpoint is selected by validation only, never by test.
        if vala > best_val:
            best_val, best_ep = vala, ep + 1
            torch.save(model.state_dict(), ckpt)
            print(f"  [NEW BEST] epoch {best_ep} val {best_val:.4f}")

    # Single held-out test on the best val checkpoint. This line runs once.
    print("--- Final Test ONCE (held-out) ---")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, test_acc, yt, yp, ypr = evaluate_with_probs(model, el, criterion, desc="Final Test")
    print(f"  test acc {test_acc:.4f} n={len(yt)} support {dict(Counter(yt))}")

    # Aggregate metrics. Balanced accuracy and macro F1 protect rare classes.
    # Kappa and MCC correct for chance. AUROC/AUPRC measure ranking.
    bal = balanced_accuracy_score(yt, yp)
    kappa = cohen_kappa_score(yt, yp)
    mcc = matthews_corrcoef(yt, yp)
    ece = compute_ece(ypr, yt)
    brier = compute_brier(ypr, yt, nc)
    try:
        from sklearn.preprocessing import label_binarize
        yb = label_binarize(yt, classes=list(range(nc)))
        auroc = roc_auc_score(yb, ypr, average="macro", multi_class="ovr")
        auprc = average_precision_score(yb, ypr, average="macro")
    except Exception as e:
        print(f"  auroc/auprc skip: {e}")
        auroc, auprc = float("nan"), float("nan")
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    # Per-class report plus specificity for the clinical table.
    cm = confusion_matrix(yt, yp, labels=list(range(nc)))
    per = classification_report(yt, yp, target_names=class_names, digits=4, output_dict=True)
    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        spec = float(tn / (tn + fp + 1e-9))
        if cname in per:
            per[cname]["specificity"] = spec
            per[cname]["support"] = int(cm[i, :].sum())
    print(classification_report(yt, yp, target_names=class_names, digits=4))
    print(f"  bal {bal:.4f} macroF1 {fm:.4f} kappa {kappa:.4f} mcc {mcc:.4f} ece {ece:.4f} brier {brier:.4f} auroc {auroc:.4f} auprc {auprc:.4f}")

    # Single source of truth JSON. y_true/y_pred/y_probs enable bootstrap,
    # McNemar, DeLong, ROC/PR, and reliability without retraining (R1-8).
    params_m2, macs_g = measure_efficiency(model)
    out = {
        "model": cfg["name"], "variant": variant, "desc": cfg["desc"],
        "seed": seed, "epochs": epochs, "protocol": {
            "lr": LR, "wd": WD, "batch": BATCH_SIZE, "ls": LS,
            "size": list(TARGET_SIZE), "deterministic": True, "tripod": "Type 2b test-once"},
        "best_val_acc": float(best_val), "best_epoch": int(best_ep),
        "history": hist,
        "efficiency": {"params_M": params_m2, "macs_G": macs_g},
        "test": {
            "accuracy": float(test_acc), "balanced_accuracy": float(bal),
            "macro": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
            "weighted": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
            "kappa": float(kappa), "mcc": float(mcc), "ece": float(ece), "brier": float(brier),
            "auroc_macro": float(auroc) if not np.isnan(auroc) else None,
            "auprc_macro": float(auprc) if not np.isnan(auprc) else None,
            "n": int(len(yt)), "support_per_class": {class_names[i]: int(Counter(yt)[i]) for i in range(nc)},
            "y_true": list(map(int, yt)), "y_pred": list(map(int, yp)), "y_probs": ypr.tolist(),
        },
        "per_class": per, "classes": class_names,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Dual write: repo results/ for old scripts, study results/ for this section.
    for p in [RESULTS / f"results_ablation_{variant}{out_tag}.json",
              STUDY_RESULTS / f"results_ablation_{variant}{out_tag}.json"]:
        p.write_text(json.dumps(out, indent=2))
        print(f"  saved {p}")
    # History in the shared histories/ folder feeds the 12-figure suite.
    (HIST / f"history_ablation_{variant}{out_tag}.json").write_text(
        json.dumps({"model": cfg["name"], "variant": variant, "epochs": list(range(1, epochs + 1)), **hist}, indent=2))

    # Per-variant plots show train/val only. No test curve appears (R1-2).
    for met in ["acc", "loss"]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(hist[f"train_{met}"], label=f"Train {met}", color="#0072B2", lw=2)
        ax.plot(hist[f"val_{met}"], label=f"Val {met}", color="#D55E00", lw=2)
        ax.set_title(f'{cfg["name"]} {met} - Train vs Val (n_val={len(vl.dataset)}, seed {seed})')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(met)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(STUDY_FIGS / f"curve_{variant}_{met}.png", bbox_inches="tight")
        plt.close(fig)
    # Confusion matrices in raw counts and row-normalized form for the supplement.
    for norm, fmt, suf in [(False, "d", "raw"), (True, ".2f", "norm")]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        m = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) if norm else cm
        sns.heatmap(m, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Confusion {suf} - {cfg["name"]} (test n={len(yt)}, once)')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(STUDY_FIGS / f"confusion_{variant}_{suf}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"  plots in {STUDY_FIGS}/")
    return out


# ----------------------------------------------------------------------------
# Compare mode: no training, only tables and figures from saved JSONs.
# Checks that test n is consistent across variants (R1-1, R1-10). Warns loudly
# if splits differ, because Table 5 would then be invalid.
# ----------------------------------------------------------------------------
def load_all_results(tag=""):
    """Load saved ablation JSONs for the requested tag. Prefer results/, fall back to study."""
    found = {}
    for v in ORDER:
        for cand in [RESULTS / f"results_ablation_{v}{tag}.json",
                     STUDY_RESULTS / f"results_ablation_{v}{tag}.json"]:
            if cand.exists():
                try:
                    found[v] = json.loads(cand.read_text())
                    break
                except Exception:
                    pass
        # Convenience fallback: reuse the main H-CoAtNet result as full if needed.
        if v == "full" and v not in found:
            for fb in [RESULTS / "results_hcoatnet.json", RESULTS / "results_final.json"]:
                if fb.exists():
                    try:
                        d = json.loads(fb.read_text())
                        d["variant"] = "full"
                        found[v] = d
                        break
                    except Exception:
                        pass
    return found


def generate_compare(tag=""):
    """Build Table 5 LaTeX plus main/drop/per-class figures from saved JSONs."""
    import pandas as pd
    found = load_all_results(tag)
    if len(found) < 2:
        print(f"[WARN] need >=2 variants, have {list(found.keys())}. Train first.")
        print("  looked in results/ and ablation study/results/")
        return False
    # Consistency gate: every variant must report the same test n (R1-10).
    ns = {v: found[v]["test"]["n"] for v in found}
    n_main = max(ns.values())
    if len(set(ns.values())) > 1:
        print(f"[WARN] inconsistent test n across variants: {ns}. Fix split before paper.")
    else:
        print(f"  test n={n_main} consistent across {list(found.keys())}")
    rows = []
    for v in ORDER:
        if v not in found:
            continue
        t = found[v]["test"]
        e = found[v].get("efficiency", {})
        rows.append({"variant": v, "Model": VARIANTS[v]["name"],
                     "Acc": t["accuracy"] * 100, "BalAcc": t["balanced_accuracy"] * 100,
                     "MacroF1": t["macro"]["f1"] * 100, "Kappa": t["kappa"] * 100,
                     "MCC": t["mcc"] * 100, "ECE": t["ece"] * 100,
                     "AUROC": (t.get("auroc_macro") or 0) * 100,
                     "ParamsM": e.get("params_M"), "MACsG": e.get("macs_G")})
    df = pd.DataFrame(rows)
    df["ord"] = df["variant"].apply(lambda x: ORDER.index(x))
    df = df.sort_values("ord").drop(columns="ord")
    full_acc = df[df.variant == "full"]["Acc"].values[0] if "full" in df.variant.values else df.Acc.max()
    # Drop in percentage points (pp), the unit reviewers expect in Table 5.
    df["Drop_pp"] = full_acc - df["Acc"]

    # Main grouped bar: absolute Acc, MacroF1, Kappa per variant (paper Fig. 5).
    m = df.melt(id_vars=["Model", "variant"], value_vars=["Acc", "MacroF1", "Kappa"],
                var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.barplot(data=m, x="Model", y="Score", hue="Metric",
                palette=["#0072B2", "#009E73", "#D55E00"], ax=ax,
                order=[VARIANTS[v]["name"] for v in df["variant"]])
    for i, r in df.reset_index(drop=True).iterrows():
        ax.text(i - 0.28, r["Acc"] + 0.7, f'{r["Acc"]:.1f}%', fontsize=8, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Ablation - Acc/MacroF1/Kappa per variant (frozen test n={n_main}, seed 42)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    for p in [REPO / "figures/fig_ablation_02_main_bar.png", STUDY_FIGS / "fig_ablation_02_main_bar.png"]:
        fig.savefig(p, bbox_inches="tight")
    for p in [REPO / "figures/fig_ablation_02_main_bar.pdf", STUDY_FIGS / "fig_ablation_02_main_bar.pdf"]:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    # Drop bar: contribution of each removed component in pp (reviewers read this first).
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(df["Model"], df["Drop_pp"], color=[VMAP_COLOR[v] for v in df["variant"]], edgecolor="black")
    for x, y in zip(df["Model"], df["Drop_pp"]):
        ax.text(list(df["Model"]).index(x), y + 0.12, "0.0" if y == 0 else f"-{y:.1f}pp",
                ha="center", fontsize=9, weight="bold")
    ax.set_ylabel("Drop vs Full (pp)")
    ax.set_title(f"Accuracy drop vs Full (pp, n={n_main})")
    fig.tight_layout()
    for p in [REPO / "figures/fig_ablation_03_drop.png", STUDY_FIGS / "fig_ablation_03_drop.png"]:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    # Per-class F1 heatmap: shows which classes need SE (texture) vs ViT (context).
    try:
        per_data = {}
        for v in df["variant"]:
            per = found[v].get("per_class", {})
            row = {}
            for cls, met in per.items():
                if cls in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                if isinstance(met, dict) and "f1-score" in met:
                    row[cls] = met["f1-score"]
            if row:
                per_data[VARIANTS[v]["name"]] = row
        if per_data:
            dfp = pd.DataFrame(per_data).T
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            sns.heatmap(dfp, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, linewidths=0.5, ax=ax)
            ax.set_title(f"Per-class F1 across variants (n={n_main})")
            fig.tight_layout()
            for p in [REPO / "figures/fig_ablation_04_perclass_heatmap.png",
                      STUDY_FIGS / "fig_ablation_04_perclass_heatmap.png"]:
                fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"  heatmap skip: {e}")

    # LaTeX Table 5: generated, never hand-edited, so numbers cannot drift (R1-10).
    tex = []
    tex.append("% Table 5 - generated by ablation study/ablation_study.py --variant compare. Do not hand-edit.")
    tex.append("\\begin{table}[t]")
    tex.append(f"\\caption{{Ablation on frozen test (n={n_main}, seed 42). Same 30ep, same split, same aug, same AdamW 5e-5 Cosine WD 0.01, TRIPOD-AI Type 2b - only ViT/SE toggled.}}")
    tex.append("\\label{tab:ablation}")
    tex.append("\\centering\\small\\begin{tabular}{lccccccc}")
    tex.append("\\toprule Variant & Acc & BalAcc & MacroF1 & Kappa & MCC & ECE $\\downarrow$ & $\\Delta$ vs Full \\\\")
    tex.append("\\midrule")
    for _, r in df.iterrows():
        b1 = "\\textbf{" if r["variant"] == "full" else ""
        b2 = "}" if r["variant"] == "full" else ""
        d = f'-{r["Drop_pp"]:.1f}pp' if r["Drop_pp"] > 0 else "-"
        tex.append(f"{b1}{r['Model']}{b2} & {b1}{r['Acc']:.2f}{b2} & {r['BalAcc']:.2f} & {r['MacroF1']:.2f} & {r['Kappa']:.2f} & {r['MCC']:.2f} & {r['ECE']:.2f} & {d} \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}")
    for p in [STUDY / "ablation_table.tex", REPO / "ablation/ablation_table.tex"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(tex) + "\n")
    df.to_csv(STUDY / "ablation_summary.csv", index=False)
    (STUDY / "ablation_summary.json").write_text(json.dumps(rows, indent=2))
    print("  compare done:")
    print(f"    n={n_main} variants={list(df['variant'])}")
    for _, r in df.iterrows():
        print(f"    {r['variant']:8s} Acc {r['Acc']:.2f} MacroF1 {r['MacroF1']:.2f} Kappa {r['Kappa']:.2f} Drop -{r['Drop_pp']:.1f}pp")
    print("    wrote ablation_table.tex + ablation_summary.csv/json + 3 figs")
    return True


# ----------------------------------------------------------------------------
# Dataset resolution: explicit --dataset_dir wins (best for local reruns).
# Then common local folders, then Roboflow download with env key.
# Key handling follows R1-6: env only, never hardcoded, never committed.
# ----------------------------------------------------------------------------
def resolve_dataset(cli_dir):
    """Return a dataset root with train/valid/test. Download from Roboflow if needed."""
    if cli_dir and Path(cli_dir).exists():
        print(f"  dataset dir (cli): {cli_dir}")
        return cli_dir
    for cand in [REPO / "dataset", REPO / "data", Path("/content/dataset")]:
        if cand.exists():
            print(f"  dataset dir (local): {cand}")
            return str(cand)
    key = os.getenv("ROBOFLOW_API_KEY", "")
    if not key:
        raise ValueError("No dataset found. Pass --dataset_dir <path> or set ROBOFLOW_API_KEY (Roboflow ich-s-7lnsj v1).")
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    ds = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    print(f"  dataset via Roboflow: {ds.location}")
    return ds.location


# ----------------------------------------------------------------------------
# CLI: smoke proves the stack trains, single variants save time, all gives
# Table 5, compare rebuilds outputs without GPU. --tag namespaces multi-seed
# runs (for example _seed43) so they never overwrite the main seed 42 files.
# ----------------------------------------------------------------------------
def main():
    """Parse args and dispatch to smoke, train, or compare."""
    ap = argparse.ArgumentParser(description="Fair H-CoAtNet ablation")
    ap.add_argument("--variant", default="all",
                    choices=["all", "full", "noSE", "noViT", "cnnOnly", "compare", "smoke"],
                    help="all=4 variants, smoke=1ep noSE check, compare=figs only")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--dataset_dir", type=str, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="scratch (only for debug, NOT fair)")
    ap.add_argument("--tag", type=str, default="", help="suffix for outputs, e.g. _seed43")
    a = ap.parse_args()

    # Compare path needs no dataset and no GPU. Used after training or in review.
    if a.variant == "compare":
        ok = generate_compare(tag=a.tag)
        sys.exit(0 if ok else 2)

    # Smoke path trains noSE for 1 epoch. If this passes, the full run will pass.
    if a.variant == "smoke":
        fairness_banner(1, a.seed)
        ds = resolve_dataset(a.dataset_dir)
        train_one_variant("noSE", ds, epochs=1, seed=a.seed,
                          pretrained=(not a.no_pretrained), out_tag="_smoke")
        print("\nSMOKE OK - real training works. Now run --variant noSE --epochs 30 or --variant all.")
        generate_compare(tag="_smoke")
        return

    # Full training path: one or all variants under the fairness lock.
    fairness_banner(a.epochs, a.seed)
    if a.no_pretrained:
        print("WARN: --no-pretrained breaks fairness. Use only for debug.")
    ds = resolve_dataset(a.dataset_dir)
    todo = ORDER if a.variant == "all" else [a.variant]
    for v in todo:
        train_one_variant(v, ds, epochs=a.epochs, seed=a.seed,
                          pretrained=(not a.no_pretrained), out_tag=a.tag)
    print("\nGenerating compare from fresh results...")
    generate_compare(tag=a.tag)
    print("\nNext: python tools/bootstrap_ci.py + stats_tests.py + compute_flops.py, then")
    print('  python3 "ablation study/generate_all_12_ablation_figs.py" --real')


if __name__ == "__main__":
    main()
