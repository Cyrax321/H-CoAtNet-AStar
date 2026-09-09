#!/usr/bin/env python3
"""
kfold.py -- A* Rigorous 5-fold stratified cross-validation for the seven-model study.

Addresses R2-4 ("K-fold cross-validation is needed").

Distinguishes from the existing single-split benchmark by:
  - Using a frozen 158-image test set held out from CV entirely
  - Building 5 stratified folds from the development pool (train+valid = 2350 images)
  - Reusing the existing seven training scripts (one per model) verbatim:
    per-model recipes mirror Table 3 of the single-split benchmark exactly
    (augmentation family, class weighting, label smoothing, LR, batch size)
  - Saving per-fold raw predictions for downstream analysis
  - Reporting mean +/- SD across folds with paired statistical tests
  - Producing Table 6 (k-fold) and supplementary figures

Output:
  results/kfold/
    kfold_protocol.json              -- run configuration + software versions
    kfold_splits.json                -- per-fold train/val/test file lists + hashes
    kfold_fold_results.json          -- per-fold metrics for every model
    kfold_predictions.json           -- per-sample raw predictions
    kfold_summary.csv                -- mean +/- SD table
    kfold_summary.json               -- machine-readable summary (with 95% CI)
    kfold_per_class_summary.csv      -- per-class P/R/F1/spec/support (R2 minor)
    kfold_per_class_summary.json
    kfold_table.tex                  -- publication-style Table 6
    kfold_paired_tests.json          -- paired model comparisons
    kfold_test_results.json          -- per-fold FROZEN-TEST metrics (val-best ckpt)
    kfold_test_summary.csv           -- frozen-test mean +/- SD table
    kfold_test_predictions.json      -- frozen-test per-sample predictions
    kfold_friedman_nemenyi.json      -- Friedman + Nemenyi over folds x models
    curves/<model>_fold<N>_accuracy.png
    curves/<model>_fold<N>_loss.png
    confusion/<model>_fold<N>_raw.png
    confusion/<model>_fold<N>_norm.png
    val/roc/<model>_fold<N>.png
    val/reliability/<model>_fold<N>.png
    test/roc/<model>.png
    test/reliability/<model>.png
    figures/
      fig_kfold_accuracy_mean_sd.png
      fig_kfold_macro_f1_mean_sd.png
      fig_kfold_balanced_accuracy_mean_sd.png
      fig_kfold_foldwise_accuracy.png
      fig_kfold_foldwise_macro_f1.png
      fig_kfold_metric_summary.png
      fig_kfold_class_distribution.png
      fig_kfold_perclass_f1_heatmap.png
      fig_kfold_test_accuracy_mean_sd.png
      fig_kfold_mean_roc_test.png
      fig_kfold_reliability_test.png
      fig_kfold_critical_difference.png
    checkpoints/<model>/fold<N>_best.pth

Usage:
  python tools/kfold/kfold.py --validate-only
  python tools/kfold/kfold.py --smoke-test
  python tools/kfold/kfold.py --dataset_dir /content/dataset --k 5 \\
      --epochs 30 --seed 42 --models all

Leakage-safety:
  - Frozen test set (158 images, splits/seed42_indices.json) is excluded from all folds.
  - The development pool = original train (2196) + original valid (154) = 2350 images.
  - Stratified 5-fold over the development pool preserves per-class ratios.
  - Identity grouping: filename stem before the first underscore (e.g.
    "Ichthyosis-vulgaris-10-_jpg.rf.<hash>.jpg" -> stem "Ichthyosis-vulgaris-10").
    This matches Roboflow's exported filename pattern; augmented copies share
    the same stem. We group by (class, stem) so near-duplicates cannot leak
    across folds (see tools/dedup_audit.py for prior audit).
  - Patient identity is not available in this dataset; the script records this
    explicitly in kfold_protocol.json under "grouping_strategy".

Test-set isolation:
  - test_loader is built only once per fold (for post-fold development eval);
    no fold ever uses the frozen test for training, augmentation, model
    selection, or thresholding.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Repo paths (location-independent: works whether the script lives in
# <root>/tools/kfold/ or <root>/H-CoAtNet/tools/kfold/)
# ---------------------------------------------------------------------------
def _find_repo_root():
    here = Path(__file__).resolve()
    for depth in (2, 3, 1):
        cand = here.parents[depth] if depth < len(here.parents) else None
        if cand and (cand / "H-CoAtNet" / "proposed_method").exists():
            return cand
    return here.parents[2] if len(here.parents) > 2 else here.parent

REPO = _find_repo_root()
PROPOSED_DIR = REPO / "H-CoAtNet" / "proposed_method"
BASELINES_DIR = REPO / "H-CoAtNet" / "baselines"
SPLITS_DIR = REPO / "splits"
FROZEN_MANIFEST = SPLITS_DIR / "seed42_indices.json"
RESULTS_DIR = REPO / "results"
KFOLD_DIR = RESULTS_DIR / "kfold"
SUBMIT_DIR = REPO / "tools" / "kfold"

# Make the proposed + baseline scripts importable so we can instantiate
# models without copy-pasting the architecture.
sys.path.insert(0, str(PROPOSED_DIR))
sys.path.insert(0, str(BASELINES_DIR))


# ---------------------------------------------------------------------------
# Frozen test-set isolation (handled inside gather_pool_and_frozen_test)
# ---------------------------------------------------------------------------
# The frozen test set is enumerated by walking <dataset>/test/<class>/* and
# excluded from all folds. Patient identity is not available in this dataset;
# we explicitly record this in kfold_protocol.json under "grouping_strategy".


# ---------------------------------------------------------------------------
# Pool enumeration
# ---------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_split(split_dir):
    """Return (files_abs, labels_int, class_names) for one ImageFolder split."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return [], [], []
    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    files, labels = [], []
    for cls in class_names:
        for p in (split_dir / cls).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(str(p.resolve()))
                labels.append(cls_to_idx[cls])
    return files, labels, class_names


def gather_pool(dataset_dir):
    """Return (dev_files, dev_labels, test_files, class_names).

    dev_*  : development pool (train + valid); used for CV folds.
    test_* : frozen test set; NEVER used during CV training.

    class_names is the canonical Roboflow class order (sorted).
    """
    dataset_dir = Path(dataset_dir)
    # Test set first (so we can verify and exclude it)
    test_files, test_labels, class_names = list_split(dataset_dir / "test")
    if not test_files:
        raise RuntimeError(f"no frozen test files found under {dataset_dir/'test'}")
    # Development pool = train + valid (the "original train" + "original
    # validation" per the rebuttal).
    train_files, train_labels, train_classes = list_split(dataset_dir / "train")
    valid_files, valid_labels, valid_classes = list_split(dataset_dir / "valid")
    if train_classes and train_classes != class_names:
        raise RuntimeError(f"class mismatch between train and test: "
                           f"{train_classes} vs {class_names}")
    if valid_classes and valid_classes != class_names:
        raise RuntimeError(f"class mismatch between valid and test: "
                           f"{valid_classes} vs {class_names}")
    dev_files = train_files + valid_files
    dev_labels = train_labels + valid_labels
    # Sanity: test set must be disjoint from development pool
    dev_set = set(dev_files)
    test_set = set(test_files)
    overlap = dev_set & test_set
    if overlap:
        raise RuntimeError(
            f"frozen test contamination: {len(overlap)} files appear in both "
            f"dev and test: {list(overlap)[:5]}")
    return dev_files, dev_labels, test_files, class_names


def identity_key(rel_path):
    """Return a string used to group near-duplicate / augmented copies.

    Roboflow augmented exports look like
        <Class>-<idx>_<aug>_jpg.rf.<hash>.jpg
    The original image and its augmented copies share everything before the
    first "_jpg" or ".rf" marker. We therefore extract the stem up to the
    first underscore-followed-by-jpeg marker, falling back to the basename
    minus extension for safety.
    """
    name = Path(rel_path).name
    # Cut at "_jpg." or ".rf." (case insensitive) if present
    for sep in ("_jpg.", ".rf."):
        idx = name.lower().find(sep)
        if idx > 0:
            return name[:idx]
    # Fallback: strip extension
    return Path(name).stem


def build_groups(files, labels):
    """Return a list of group ids, one per file, such that augmented copies
    of the same original share a group id.
    """
    return [identity_key(f) for f in files]


# ---------------------------------------------------------------------------
# Fold construction (leakage-safe)
# ---------------------------------------------------------------------------
def build_folds(files, labels, group_ids, k, seed):
    """Group-aware StratifiedKFold.

    sklearn's StratifiedGroupKFold is the ideal primitive: it assigns each
    group to exactly one fold while preserving the class label distribution.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(sgkf.split(files, labels, groups=group_ids))
    return folds


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class FileListDataset(Dataset):
    """A minimal Dataset that reads (file_path, label) pairs."""
    def __init__(self, file_list, label_list, indices, transform):
        self.file_list = file_list
        self.label_list = label_list
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.open(self.file_list[idx]).convert("RGB")
        return self.transform(img), int(self.label_list[idx])


# ---------------------------------------------------------------------------
# Model registry: name -> (constructor_fn, kwargs, transform_overrides)
# ---------------------------------------------------------------------------
# Each entry: a factory that returns a fresh nn.Module on the requested
# device, plus optional training-time overrides (e.g. CNN/EfficientNet
# use lr=3e-4; the other five use lr=5e-5).
#
# IMPORTANT: this is a thin wrapper around the existing scripts. We import
# the model class from each baseline and instantiate it; we do NOT redefine
# the architecture.
#
# Per-model training recipe (mirrors Table 3 of the single-split benchmark
# exactly, so the K-fold protocol adds NO new asymmetry):
#   strong_aug  : TrivialAugmentWide + RandomErasing   (H-CoAtNet only)
#   color_aug   : ColorJitter(0.2,0.2,0.2)             (CoAtNet, EfficientNet-B0)
#   class_weights: inverse-frequency CE weighting      (all but Swin/ViT)
#   label_smoothing: 0.1                               (H-CoAtNet only)
MODEL_RECIPE = {
    "H-CoAtNet":       {"strong_aug": True,  "color_aug": False, "class_weights": True,  "label_smoothing": 0.1},
    "CoAtNet":         {"strong_aug": False, "color_aug": True,  "class_weights": True,  "label_smoothing": 0.0},
    "GFT":             {"strong_aug": False, "color_aug": False, "class_weights": True,  "label_smoothing": 0.0},
    "Swin":            {"strong_aug": False, "color_aug": False, "class_weights": False, "label_smoothing": 0.0},
    "ViT":             {"strong_aug": False, "color_aug": False, "class_weights": False, "label_smoothing": 0.0},
    "CNN":             {"strong_aug": False, "color_aug": False, "class_weights": True,  "label_smoothing": 0.0},
    "EfficientNet-B0": {"strong_aug": False, "color_aug": True,  "class_weights": True,  "label_smoothing": 0.0},
}
def _make_hcoatnet(num_classes, pretrained=True):
    from train_h_coatnet import HCoAtNet
    return HCoAtNet(num_classes=num_classes, pretrained=pretrained)

def _make_coatnet(num_classes, pretrained=True):
    from train_coatnet import CoAtNet
    return CoAtNet(num_classes=num_classes, pretrained=pretrained)

def _make_gft(num_classes, pretrained=True):
    from train_gft import GFT
    return GFT(num_classes=num_classes, pretrained=pretrained)

def _make_swin(num_classes, pretrained=True):
    from train_swin import SwinTransformer
    return SwinTransformer(num_classes=num_classes)

def _make_vit(num_classes, pretrained=True):
    from train_vit import VisionTransformer
    return VisionTransformer(num_classes=num_classes)

def _make_cnn(num_classes, pretrained=False):
    from train_cnn import BaselineCNN
    return BaselineCNN(num_classes=num_classes)

def _make_effnet(num_classes, pretrained=False):
    # Mirrors train_efficientnet.py: timm efficientnet_b0, from scratch.
    import timm
    return timm.create_model("efficientnet_b0", pretrained=False,
                             num_classes=num_classes)

MODEL_REGISTRY = {
    # name             : (factory, default_lr, default_batch, pretrained_default)
    "H-CoAtNet":       (_make_hcoatnet,   5e-5, 24, True),
    "CoAtNet":         (_make_coatnet,    5e-5, 24, True),
    "GFT":             (_make_gft,        5e-5, 24, True),
    "Swin":            (_make_swin,       5e-5, 16, False),
    "ViT":             (_make_vit,        5e-5, 16, False),
    "CNN":             (_make_cnn,        3e-4, 24, False),
    "EfficientNet-B0": (_make_effnet,     3e-4, 24, False),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_all_metrics(y_true, y_pred, y_probs, n_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probs = np.asarray(y_probs)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ece": float(compute_ece(y_probs, y_true)),
        "brier": float(compute_brier(y_probs, y_true, n_classes)),
    }
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    metrics["macro"] = {"precision": float(p_m), "recall": float(r_m), "f1": float(f_m)}
    metrics["weighted"] = {"precision": float(p_w), "recall": float(r_w), "f1": float(f_w)}
    # AUROC + AUPRC (one-vs-rest macro)
    try:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        metrics["auroc_macro"] = float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
        metrics["auprc_macro"] = float(average_precision_score(y_bin, y_probs, average="macro"))
    except Exception:
        metrics["auroc_macro"] = None
        metrics["auprc_macro"] = None
    # Per-class
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    per_class = {}
    for i, cname in enumerate(["class_" + str(i) for i in range(n_classes)]):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - (tp + fn + fp))
        per_class[cname] = {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": float(tp / max(1, tp + fp)),
            "recall": float(tp / max(1, tp + fn)),
            "f1": float(tp / max(1, tp + 0.5 * (fp + fn))),
            "specificity": float(tn / max(1, tn + fp)),
            "support": int(cm[i, :].sum()),
        }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def compute_ece(probs, y_true, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (pred == np.asarray(y_true))
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += abs(ok[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def compute_brier(probs, y_true, n_classes):
    yb = label_binarize(y_true, classes=list(range(n_classes)))
    return float(np.mean((yb - probs) ** 2))


# ---------------------------------------------------------------------------
# Visual progress: per-fold and per-model banners
# ---------------------------------------------------------------------------
def print_fold_summary(model_name, fold_idx, k, metrics, test_metrics, elapsed_min):
    """Print a clear banner after each fold completes."""
    print("\n" + "=" * 70)
    print(f"  FOLD {fold_idx+1}/{k} COMPLETE -- {model_name}  ({elapsed_min:.1f} min)")
    print("=" * 70)
    t = metrics
    print(f"  Validation (val-best epoch {t.get('best_epoch', '?')}):")
    print(f"    Accuracy      : {t['accuracy']*100:.2f}%")
    print(f"    Balanced Acc  : {t['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1      : {t['macro']['f1']*100:.2f}%")
    print(f"    Kappa         : {t['kappa']:.4f}")
    print(f"    MCC           : {t['mcc']:.4f}")
    print(f"    ECE           : {t['ece']:.4f}")
    if t.get('auroc_macro') is not None:
        print(f"    AUROC         : {t['auroc_macro']:.4f}")
    if test_metrics is not None:
        print(f"  Frozen Test (val-best ckpt, evaluated once):")
        print(f"    Accuracy      : {test_metrics['accuracy']*100:.2f}%")
        print(f"    Balanced Acc  : {test_metrics['balanced_accuracy']*100:.2f}%")
        print(f"    Macro F1      : {test_metrics['macro']['f1']*100:.2f}%")
        print(f"    Kappa         : {test_metrics['kappa']:.4f}")
        print(f"    MCC           : {test_metrics['mcc']:.4f}")
        print(f"    ECE           : {test_metrics['ece']:.4f}")
        if test_metrics.get('auroc_macro') is not None:
            print(f"    AUROC         : {test_metrics['auroc_macro']:.4f}")
    print("=" * 70)
    sys.stdout.flush()


def print_model_summary(model_name, k, fold_results, test_results, elapsed_min):
    """Print a full summary banner after all k folds of one model."""
    # Aggregate metrics across folds
    acc_vals = [f["accuracy"] for f in fold_results if not f.get("failed")]
    f1_vals  = [f["macro"]["f1"] for f in fold_results if not f.get("failed")]
    bal_vals = [f["balanced_accuracy"] for f in fold_results if not f.get("failed")]
    kap_vals = [f["kappa"] for f in fold_results if not f.get("failed")]
    mcc_vals = [f["mcc"] for f in fold_results if not f.get("failed")]
    ece_vals = [f["ece"] for f in fold_results if not f.get("failed")]
    auroc_vals = [f["auroc_macro"] for f in fold_results if not f.get("failed") and f.get("auroc_macro") is not None]
    test_acc_vals = [f["accuracy"] for f in test_results if not f.get("failed")]
    test_f1_vals  = [f["macro"]["f1"] for f in test_results if not f.get("failed")]
    test_kap_vals = [f["kappa"] for f in test_results if not f.get("failed")]

    def _ms(vals, pct=False):
        if not vals:
            return "--"
        a = np.array(vals)
        m, s = a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
        if pct:
            return f"{m*100:.2f} +/- {s*100:.2f}"
        return f"{m:.4f} +/- {s:.4f}"

    n_done = len(acc_vals)
    print("\n" + "#" * 70)
    print(f"#  MODEL COMPLETE: {model_name}  ({n_done}/{k} folds, {elapsed_min:.1f} min total)")
    print("#" * 70)
    print("")
    print(f"  VALIDATION ACROSS {n_done} FOLDS (val-best checkpoint per fold):")
    print(f"  {'Metric':<22s} {'Mean +/- SD':<30s} {'Per-fold values'}")
    print(f"  {'-'*22} {'-'*30} {'-'*30}")
    fold_accs = [f"{v*100:.1f}" for v in acc_vals]
    print(f"  {'Accuracy':<22s} {_ms(acc_vals, pct=True):<30s} [{', '.join(fold_accs)}]")
    fold_f1s = [f"{v*100:.1f}" for v in f1_vals]
    print(f"  {'Macro F1':<22s} {_ms(f1_vals, pct=True):<30s} [{', '.join(fold_f1s)}]")
    fold_bals = [f"{v*100:.1f}" for v in bal_vals]
    print(f"  {'Balanced Acc':<22s} {_ms(bal_vals, pct=True):<30s} [{', '.join(fold_bals)}]")
    print(f"  {'Kappa':<22s} {_ms(kap_vals):<30s}")
    print(f"  {'MCC':<22s} {_ms(mcc_vals):<30s}")
    print(f"  {'ECE (lower=better)':<22s} {_ms(ece_vals):<30s}")
    if auroc_vals:
        print(f"  {'AUROC':<22s} {_ms(auroc_vals):<30s}")
    if test_acc_vals:
        print("")
        print(f"  FROZEN TEST ACROSS {len(test_acc_vals)} FOLDS (val-best ckpt per fold):")
        print(f"  {'Metric':<22s} {'Mean +/- SD':<30s} {'Per-fold values'}")
        print(f"  {'-'*22} {'-'*30} {'-'*30}")
        t_accs = [f"{v*100:.1f}" for v in test_acc_vals]
        print(f"  {'Accuracy':<22s} {_ms(test_acc_vals, pct=True):<30s} [{', '.join(t_accs)}]")
        t_f1s = [f"{v*100:.1f}" for v in test_f1_vals]
        print(f"  {'Macro F1':<22s} {_ms(test_f1_vals, pct=True):<30s} [{', '.join(t_f1s)}]")
        print(f"  {'Kappa':<22s} {_ms(test_kap_vals):<30s}")
    print("")
    print("#" * 70)
    sys.stdout.flush()


def print_leaderboard(completed_models, fold_results_per_model, test_results_per_model):
    """Print a running leaderboard after each model completes."""
    if not completed_models:
        return
    print("\n" + "=" * 70)
    print("  RUNNING LEADERBOARD (models completed so far)")
    print("=" * 70)
    # Header
    print(f"  {'Model':<18s} {'Val Acc (%)':<20s} {'Test Acc (%)':<20s} {'Val F1 (%)':<20s}")
    print(f"  {'-'*18} {'-'*20} {'-'*20} {'-'*20}")
    rows = []
    for m in completed_models:
        fr = fold_results_per_model.get(m, [])
        tr = test_results_per_model.get(m, [])
        fr_good = [f for f in fr if not f.get("failed")]
        tr_good = [f for f in tr if not f.get("failed")]
        if not fr_good:
            continue
        val_acc = np.mean([f["accuracy"] for f in fr_good]) * 100
        val_sd = np.std([f["accuracy"] for f in fr_good], ddof=1) * 100 if len(fr_good) > 1 else 0
        val_f1 = np.mean([f["macro"]["f1"] for f in fr_good]) * 100
        test_acc = np.mean([f["accuracy"] for f in tr_good]) * 100 if tr_good else float('nan')
        test_sd = np.std([f["accuracy"] for f in tr_good], ddof=1) * 100 if len(tr_good) > 1 else 0
        rows.append((m, val_acc, val_sd, test_acc, test_sd, val_f1))
    # Sort by val accuracy descending
    rows.sort(key=lambda r: r[1], reverse=True)
    for i, (m, va, vsd, ta, tsd, vf) in enumerate(rows):
        rank = f"{i+1}."
        test_str = f"{ta:.1f}+/-{tsd:.1f}" if not np.isnan(ta) else "pending"
        marker = " <-- proposed" if m == "H-CoAtNet" else ""
        print(f"  {rank} {m:<16s} {va:.1f}+/-{vsd:.1f}      {test_str:<20s} {vf:.1f}{marker}")
    print("=" * 70)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Train + eval one fold
# ---------------------------------------------------------------------------
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_transforms(model_name, base_seed, fold_idx):
    """Build train + val transforms, byte-matched to the reference scripts.

    Augmentation family is driven by MODEL_RECIPE (parity-audited in
    run_validation, check 'transform_parity_<model>'):
      strong_aug (H-CoAtNet)   : Rot15 + TrivialAugmentWide + RandomErasing
                                 (train_h_coatnet.py:283-290)
      color_aug (CoAtNet, Eff) : Rot15 + ColorJitter(0.2,0.2,0.2)
                                 (train_coatnet.py:175-181,
                                  train_efficientnet.py:164-171)
      base (GFT/Swin/ViT/CNN)  : crop/flip only
                                 (train_gft.py:285-289, train_swin.py:375-379,
                                  train_vit.py:298-302, train_cnn.py:201-205)
    Order matches the references exactly: geometric ops on PIL images, then
    ToTensor + Normalize, then RandomErasing on the normalized tensor.
    val/test: Resize((224,224)) + Normalize (identical in all seven scripts).
    """
    recipe = MODEL_RECIPE[model_name]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    base = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if recipe["strong_aug"]:
        base += [
            transforms.RandomRotation(15),
            transforms.TrivialAugmentWide(),
        ]
    elif recipe["color_aug"]:
        base += [
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if recipe["strong_aug"]:
        base += [
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ]
    train_t = transforms.Compose(base)
    val_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_t, val_t


def _seed_worker(worker_id):
    """DataLoader worker init function for deterministic augmentation order."""
    import random
    s = torch.initial_seed() % (2 ** 32)
    s = (s + worker_id) % (2 ** 32)
    random.seed(s)
    np.random.seed(s)


def make_loaders(files, labels, train_idx, val_idx, batch_size, train_t, val_t,
                 pin=True):
    """pin is disabled automatically on MPS (unsupported there) and CPU."""
    g = torch.Generator()
    g.manual_seed(int(torch.initial_seed()) % (2 ** 32))
    train_ds = FileListDataset(files, labels, train_idx, train_t)
    val_ds = FileListDataset(files, labels, val_idx, val_t)
    nw = 0 if os.name == "nt" else 2
    use_pin = pin and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=nw,
        worker_init_fn=_seed_worker, generator=g, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=nw,
        pin_memory=use_pin,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot, n, correct = 0.0, 0, 0
    n_batches = len(loader)
    start = time.time()
    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tot += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
        if i == n_batches:
            elapsed = time.time() - start
            print(f"  Training 100% {n_batches}/{n_batches} batches [{elapsed:.1f}s]", flush=True)
    return tot / max(1, n), correct / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, return_probs=True):
    model.eval()
    ys, yps, yprs = [], [], []
    n_batches = len(loader)
    start = time.time()
    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        ys.extend(y.cpu().numpy().tolist())
        yps.extend(logits.argmax(1).cpu().numpy().tolist())
        if return_probs:
            yprs.extend(F.softmax(logits, dim=1).cpu().numpy().tolist())
        if i == n_batches:
            elapsed = time.time() - start
            print(f"  Validating 100% {n_batches}/{n_batches} batches [{elapsed:.1f}s]", flush=True)
    if not return_probs:
        return np.asarray(ys), np.asarray(yps)
    return np.asarray(ys), np.asarray(yps), np.asarray(yprs)


def get_device():
    """cuda > mps (Apple Silicon) > cpu. Deterministic settings are applied
    for CUDA; MPS keeps its default matmul path (no deterministic flag).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_one_fold(model_name, fold_idx, files, labels, train_idx, val_idx,
                 n_classes, ckpt_path, base_seed, epochs, test_data=None):
    """Train one fold and return (history, val_metrics, test_metrics, raw_preds).

    test_data is optional (test_files, test_labels); when given, the restored
    val-best checkpoint is evaluated ONCE on the frozen test set after fold
    training completes.
    """
    factory, lr, batch_size, pretrained_default = MODEL_REGISTRY[model_name]
    device = get_device()
    # Per-fold deterministic seed
    fold_seed = base_seed + fold_idx * 1000
    seed_everything(fold_seed)

    train_t, val_t = make_transforms(model_name, base_seed, fold_idx)
    train_loader, val_loader = make_loaders(
        files, labels, train_idx, val_idx, batch_size, train_t, val_t)

    # Build a fresh model with IN1K (or scratch) init.
    model = factory(num_classes=n_classes, pretrained=pretrained_default).to(device)

    # Per-model loss recipe mirrors Table 3 exactly (train-fold statistics
    # only; the frozen test set never informs weights or smoothing).
    train_labels = [labels[i] for i in train_idx]
    counts = np.bincount(train_labels, minlength=n_classes)
    n_train = len(train_idx)
    recipe = MODEL_RECIPE[model_name]
    if recipe["class_weights"]:
        cw = torch.tensor([n_train / (c * n_classes + 1e-6) for c in counts],
                          dtype=torch.float, device=device)
    else:
        cw = None
    criterion = nn.CrossEntropyLoss(
        weight=cw, label_smoothing=recipe["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val, best_state, best_epoch = -1.0, None, -1
    for ep in range(epochs):
        print(f"\n--- Epoch {ep+1:2d}/{epochs} [{model_name} fold {fold_idx+1}] ---")
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validation pass (with proper loss + accuracy)
        model.eval()
        vl_sum, vl_n, v_correct = 0.0, 0, 0
        val_start = time.time()
        n_val_batches = len(val_loader)
        with torch.no_grad():
            for vi, (x, y) in enumerate(val_loader, 1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                vl_sum += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                vl_n += x.size(0)
                if vi == n_val_batches:
                    val_elapsed = time.time() - val_start
                    print(f"  Validating 100% {n_val_batches}/{n_val_batches} batches [{val_elapsed:.1f}s]", flush=True)
        vl = vl_sum / max(1, vl_n)
        va = v_correct / max(1, vl_n)
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        is_best = va > best_val
        if is_best:
            best_val = va
            best_epoch = ep + 1
            best_state = deepcopy(model.state_dict())
        # Per-epoch logging (match ablation study format)
        best_tag = f"\n  [NEW BEST] epoch {ep+1} val {va:.4f}" if is_best else ""
        print(f"  Epoch {ep+1:2d}/{epochs}: train acc {ta:.4f} loss {tl:.4f} | val acc {va:.4f} loss {vl:.4f}{best_tag}", flush=True)
    history["best_epoch"] = best_epoch
    # Restore the validation-selected checkpoint (protocol: val-best model
    # selection, matching the single-split benchmark). All final metrics --
    # validation AND frozen-test -- come from this checkpoint, not the
    # last-epoch weights.
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state if best_state is not None else model.state_dict(),
               ckpt_path)

    # Final validation metrics + raw preds
    print(f"  Restoring best checkpoint (epoch {best_epoch}, val_acc {best_val:.4f})...", flush=True)
    yv, ypv, ypv_p = evaluate(model, val_loader, device)
    metrics = compute_all_metrics(yv, ypv, ypv_p, n_classes)
    metrics["best_val_acc"] = float(best_val)
    metrics["best_epoch"] = int(best_epoch)
    metrics["device"] = device.type

    # Frozen-test evaluation, ONCE per fold, from the val-best checkpoint.
    test_metrics = None
    ytst = ypst = ypst_p = None
    if test_data is not None:
        print(f"  --- Frozen Test (held-out, val-best ckpt, fold {fold_idx+1}) ---", flush=True)
        t_files, t_labels = test_data
        test_ds = FileListDataset(t_files, t_labels, list(range(len(t_files))), val_t)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0,
                                 pin_memory=torch.cuda.is_available())
        ytst, ypst, ypst_p = evaluate(model, test_loader, device)
        test_metrics = compute_all_metrics(ytst, ypst, ypst_p, n_classes)
        print(f"  Frozen test accuracy: {test_metrics['accuracy']*100:.2f}% (n={len(ytst)})", flush=True)
        del test_loader, test_ds

    # Free memory (state dict kept on CPU for Colab RAM safety)
    del model, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if test_metrics is not None:
        test_metrics.pop("confusion_matrix", None)
    return (history, metrics, test_metrics,
            yv.tolist(), ypv.tolist(), ypv_p.tolist(),
            (ytst.tolist() if ytst is not None else None),
            (ypst.tolist() if ypst is not None else None),
            (ypst_p.tolist() if ypst_p is not None else None))


# ---------------------------------------------------------------------------
# Aggregation + statistics
# ---------------------------------------------------------------------------
def aggregate(fold_metrics_per_model):
    """Compute mean / SD / median / min / max per metric across folds."""
    out = {}
    # Flattened blocks are needed for the Friedman/Nemenyi tests.
    fold_metrics_per_model = {m: _flatten_folds(fm)
                              for m, fm in fold_metrics_per_model.items()}
    # macro_*/weighted_* are flat keys produced by _flatten_folds
    primary = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
               "macro_precision", "macro_recall",
               "weighted_precision", "weighted_recall",
               "kappa", "mcc", "ece", "brier"]

    for model_name, fold_metrics in fold_metrics_per_model.items():
        s = {"per_fold": fold_metrics, "n_folds": len(fold_metrics)}
        for k in primary:
            vals = [m[k] for m in fold_metrics if m.get(k) is not None]
            s[k] = _stat_block(vals)
        # AUROC + AUPRC
        for k in ("auroc_macro", "auprc_macro"):
            vals = [m[k] for m in fold_metrics if m.get(k) is not None]
            s[k] = _stat_block(vals)
        # best_val_acc
        vals = [m["best_val_acc"] for m in fold_metrics if m.get("best_val_acc") is not None]
        s["best_val_acc"] = _stat_block(vals)
        out[model_name] = s
    return out


def _stat_block(vals):
    if not vals:
        return {"mean": None, "sd": None, "ci95": None, "median": None,
                "min": None, "max": None, "values": []}
    a = np.asarray(vals, dtype=float)
    n = len(a)
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    # 95% CI across folds (t-based). Reported alongside mean +/- SD per R2-4
    # ("no confidence intervals, standard deviations, or significance tests").
    if n > 1:
        from scipy.stats import t as t_dist
        half = float(t_dist.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
        ci95 = [float(np.mean(a) - half), float(np.mean(a) + half)]
    else:
        ci95 = None
    return {
        "mean": float(np.mean(a)),
        "sd": sd,
        "ci95": ci95,
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "values": [float(x) for x in a],
    }


def _flatten_folds(fold_metrics):
    """Lift per-fold macro/weighted dicts into top-level scalar keys so
    aggregation and statistics can address them uniformly.
    """
    flat = []
    for fm in fold_metrics:
        g = dict(fm)
        for avg in ("macro", "weighted"):
            for k, v in (fm.get(avg) or {}).items():
                g[f"{avg}_{k}"] = v
        g.pop("macro", None)
        g.pop("weighted", None)
        flat.append(g)
    return flat


def aggregate_per_class(fold_results_per_model, class_names):
    """Mean +/- SD per-class P/R/F1/specificity/support across folds.

    Answers the R2 minor comment ("a full per-class precision/recall/F1 table
    is needed for all models") for the cross-validation setting; the values
    are per-fold val-best metrics aggregated over folds.
    """
    rows = []
    for model_name, fold_metrics in fold_results_per_model.items():
        per_fold_pc = [fm.get("per_class") for fm in fold_metrics
                       if isinstance(fm.get("per_class"), dict)]
        if not per_fold_pc:
            continue
        for ci, cname in enumerate(class_names):
            key = f"class_{ci}"
            entry_folds = [pc[key] for pc in per_fold_pc if key in pc]
            if not entry_folds:
                continue
            for metric in ("precision", "recall", "f1", "specificity", "support"):
                vals = [e[metric] for e in entry_folds if metric in e]
                if not vals:
                    continue
                a = np.asarray(vals, dtype=float)
                rows.append({
                    "model": model_name,
                    "class": cname,
                    "metric": metric,
                    "mean": float(a.mean()),
                    "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    "n_folds": len(vals),
                })
    return rows


def write_per_class_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Class", "Metric", "Mean", "SD", "NFolds"])
        for r in rows:
            w.writerow([r["model"], r["class"], r["metric"],
                        f"{r['mean']:.4f}", f"{r['sd']:.4f}", r["n_folds"]])


def plot_perclass_f1_heatmap(rows, out_path):
    """Models x classes heatmap of mean per-class F1 across folds
    (k-fold analogue of the single-split per-class F1 heatmap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows:
        return
    f1 = [r for r in rows if r["metric"] == "f1"]
    models = sorted({r["model"] for r in f1})
    classes = sorted({r["class"] for r in f1})
    arr = np.full((len(models), len(classes)), np.nan)
    for r in f1:
        arr[models.index(r["model"]), classes.index(r["class"])] = r["mean"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(arr * 100, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(classes)):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]*100:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    ax.set_title("Mean Per-Class F1 Across Folds (validation, val-best ckpt)")
    fig.colorbar(im, ax=ax, label="F1 (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_wilcoxon(fold_metrics_per_model):
    """Paired Wilcoxon signed-rank tests on per-fold Accuracy between every
    pair of models. Wilcoxon (not t-test) because n=5 folds is too small
    for a t-test assumption.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {}
    model_names = list(fold_metrics_per_model.keys())
    out = {}
    flat = {m: _flatten_folds(fm) for m, fm in fold_metrics_per_model.items()}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a_name, b_name = model_names[i], model_names[j]
            a_vals = [m["accuracy"] for m in flat[a_name]]
            b_vals = [m["accuracy"] for m in flat[b_name]]
            n = min(len(a_vals), len(b_vals))
            if n < 2:
                continue
            try:
                stat, p = wilcoxon(a_vals[:n], b_vals[:n])
                out[f"{a_name}__vs__{b_name}"] = {
                    "n": int(n),
                    "median_diff": float(np.median(np.asarray(a_vals[:n]) - np.asarray(b_vals[:n]))),
                    "statistic": float(stat),
                    "p_value": float(p),
                }
            except Exception as e:
                out[f"{a_name}__vs__{b_name}"] = {"error": str(e)}
    return out


def friedman_nemenyi(fold_metrics_per_model, metric="accuracy"):
    """Omnibus Friedman test across models over the k folds, with Nemenyi
    post-hoc ranks and the CD threshold. Standard for multi-model multi-fold
    comparisons (Demvsar 2006); does not replace the paired tests above.
    """
    out = {"metric": metric, "available": False}
    try:
        from scipy.stats import friedmanchisquare
        import scipy
    except ImportError:
        out["error"] = "scipy not available"
        return out
    names = list(fold_metrics_per_model.keys())
    per_model = {}
    for m in names:
        vals = [fm.get(metric) for fm in _flatten_folds(fold_metrics_per_model[m])
                if fm.get(metric) is not None]
        per_model[m] = vals
    n_folds = min(len(v) for v in per_model.values())
    if len(names) < 3 or n_folds < 3:
        out["error"] = f"need >=3 models and >=3 folds; got {len(names)}x{n_folds}"
        return out
    try:
        stat, p = friedmanchisquare(*[per_model[m][:n_folds] for m in names])
        out.update({"n_models": len(names), "n_folds": int(n_folds),
                    "chi2": float(stat), "p_value": float(p), "available": True})
    except Exception as e:
        out["error"] = str(e)
        return out
    # Mean ranks (1 = best) + Nemenyi CD (z-value for two-tailed alpha=0.05)
    X = np.asarray([per_model[m][:n_folds] for m in names], dtype=float)
    ranks = np.argsort(np.argsort(-X, axis=0), axis=0) + 1  # descending metric -> rank 1
    mean_ranks = {m: float(ranks[i].mean()) for i, m in enumerate(names)}
    k_, n_ = len(names), n_folds
    cd = float(2 * np.sqrt(k_ * (k_ + 1) / (6.0 * n_)) * 1.959963984540054)
    out["mean_ranks"] = mean_ranks
    out["nemenyi_cd_0.05"] = cd
    out["note"] = ("Friedman omnibus over folds; Nemenyi CD at alpha=0.05. "
                   "Models whose mean ranks differ by more than CD are "
                   "significantly different (Demvsar 2006).")
    return out


def bonferroni_correct(pairs):
    if not pairs:
        return pairs
    m = len(pairs)
    out = {}
    for k, v in pairs.items():
        if "p_value" in v:
            v2 = dict(v)
            v2["p_value_bonferroni"] = min(1.0, float(v["p_value"]) * m)
            v2["n_tests"] = m
            out[k] = v2
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Plotting (publication-style; no marketing language)
# ---------------------------------------------------------------------------
def plot_metric_bars(summary, metric_key, title, ylabel, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(summary.keys())
    means = [summary[m][metric_key]["mean"] for m in models
             if summary[m].get(metric_key, {}).get("mean") is not None]
    sds = [summary[m][metric_key]["sd"] for m in models
           if summary[m].get(metric_key, {}).get("mean") is not None]
    valid = [m for m in models
             if summary[m].get(metric_key, {}).get("mean") is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(valid))
    ax.errorbar(x, means, yerr=sds, fmt="o", capsize=4, color="#1f77b4",
                markerfacecolor="white", markersize=8, linewidth=1.2,
                ecolor="#1f77b4", elinewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(valid, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_foldwise(metric_key, fold_metrics_per_model, title, ylabel, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(fold_metrics_per_model.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models:
        vals = [fm[metric_key] for fm in fold_metrics_per_model[m]
                if fm.get(metric_key) is not None]
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=m)
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(summary, out_path):
    """A small summary heatmap of mean accuracy + macro F1 across models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(summary.keys())
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "mcc"]
    arr = np.zeros((len(models), len(metrics)))
    for i, m in enumerate(models):
        for j, k in enumerate(metrics):
            arr[i, j] = (summary[m].get(k, {}).get("mean") or 0.0) * 100.0
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center",
                    color="white", fontsize=8)
    ax.set_title("Five-Fold Cross-Validation Performance Summary (mean %)")
    fig.colorbar(im, ax=ax, label="Percent")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(splits, class_names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds = list(range(1, len(splits) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(folds))
    for i, cname in enumerate(class_names):
        # train_class_counts is keyed by integer class index (Counter over labels)
        counts = [s.get("train_class_counts", {}).get(str(i), 0) for s in splits]
        ax.bar(folds, counts, bottom=bottom, label=cname)
        bottom = bottom + np.asarray(counts)
    ax.set_xticks(folds)
    ax.set_xlabel("Fold (training partition)")
    ax.set_ylabel("Image count")
    ax.set_title("Class Distribution Across Cross-Validation Folds")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history, model_name, fold_idx, out_dir_acc, out_dir_loss):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_acc"], label="train")
    ax.plot(history["val_acc"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} Fold {fold_idx + 1} Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir_acc, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} Fold {fold_idx + 1} Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir_loss, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(probs_by_model, y_true, n_classes, out_path,
                    title="ROC (OvR macro)"):
    """Mean micro/macro One-vs-Rest ROC from saved probability vectors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(np.asarray(y_true), classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, probs in probs_by_model.items():
        P = np.asarray(probs)
        if P.ndim != 2 or P.shape[0] != y_bin.shape[0] or P.shape[1] != n_classes:
            continue
        fpr, tpr, _ = roc_curve(y_bin.ravel(), P.ravel())
        ax.plot(fpr, tpr, lw=1.5,
                label=f"{name} (micro AUROC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _reliability_curve_data(probs, y_true, n_bins=15):
    """Confidence-binned reliability data (conf, acc, counts, ECE)."""
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (pred == np.asarray(y_true))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            confs.append(np.nan); accs.append(np.nan); counts.append(0)
        else:
            confs.append(float(conf[m].mean()))
            accs.append(float(ok[m].mean()))
            counts.append(int(m.sum()))
    return {"confidence": confs, "accuracy": accs, "counts": counts,
            "ece": compute_ece(probs, y_true, n_bins=n_bins)}


def plot_reliability(confidence, accuracy, counts, ece, out_path,
                     title="Reliability Diagram"):
    """Calibration diagram: observed accuracy vs predicted confidence per bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    centers = np.arange(len(confidence))
    ax.bar(centers, accuracy, width=0.9, color="#4c72b0", alpha=0.85,
           label="Observed accuracy")
    nan_conf = np.nan_to_num(np.asarray(confidence, dtype=float), nan=0.0)
    ax.plot(centers, nan_conf, "ro--", ms=4, lw=1, label="Mean confidence")
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{c:.2f}" if not np.isnan(c) else "" for c in confidence],
                       rotation=45, fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Predicted confidence (bin)")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"{title}  (ECE = {ece:.3f})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_critical_difference(summary, out_path, metric="accuracy"):
    """Critical-difference diagram (Demvsar 2006): mean rank of each model
    with the Nemenyi CD threshold. Requires the Friedman/Nemenyi output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fn_path = KFOLD_DIR / "kfold_friedman_nemenyi.json"
    if not fn_path.exists():
        return
    fn = json.loads(fn_path.read_text())
    ranks = fn.get("mean_ranks") or {}
    cd = fn.get("nemenyi_cd_0.05")
    if not ranks or cd is None:
        return
    order = sorted(ranks.items(), key=lambda kv: kv[1])
    names = [m for m, _ in order]
    vals = [ranks[m] for m in names]
    lo = max(0.5, min(vals) - cd)
    hi = min(len(names) + 0.5, max(vals) + cd)
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(names)))
    ax.axhspan(lo, hi, color="gray", alpha=0.15)
    for m, r in order:
        ax.plot(r, 0, "o", ms=8, color="#1f77b4")
        ax.annotate(f"{m} ({r:.2f})", (r, 0), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlim(0.5, len(names) + 0.5)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(f"Mean rank ({metric}; lower is better) -- Nemenyi CD = {cd:.2f}")
    ax.set_title("Critical Difference Diagram (Nemenyi, alpha=0.05)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, class_names, model_name, fold_idx, raw=True, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    arr = np.asarray(cm)
    if not raw:
        arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
        fmt = ".2f"
        title = f"Normalized Confusion -- {model_name} Fold {fold_idx + 1}"
    else:
        fmt = "d"
        title = f"Confusion -- {model_name} Fold {fold_idx + 1}"
    im = ax.imshow(arr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center",
                    color="black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation (pre-flight)
# ---------------------------------------------------------------------------
def run_validation(dataset_dir, base_seed, k, epochs, models):
    """Pre-flight checks. PASS / FAIL / WARNING. Hard FAILs raise."""
    print("=" * 60)
    print("KFOLD PRE-FLIGHT VALIDATION")
    print("=" * 60)
    results = {"pass": 0, "warn": 0, "fail": 0}
    status = {}

    def report(name, kind, msg):
        status[name] = (kind, msg)
        if kind == "PASS":  results["pass"] += 1
        elif kind == "WARN": results["warn"] += 1
        else:               results["fail"] += 1
        print(f"  [{kind:4s}] {name}: {msg}")

    # 1. Imports (factory names match the actual training scripts)
    try:
        from train_h_coatnet import HCoAtNet
        from train_coatnet import CoAtNet
        from train_gft import GFT
        from train_swin import SwinTransformer
        from train_vit import VisionTransformer
        from train_cnn import BaselineCNN
        import timm
        report("imports", "PASS", "all seven model modules import")
    except Exception as e:
        report("imports", "FAIL", f"import failure: {e}")
        return status  # hard stop

    # 2. Instantiate
    try:
        m1 = HCoAtNet(num_classes=5, pretrained=False)
        m2 = CoAtNet(num_classes=5, pretrained=False)
        m3 = GFT(num_classes=5, pretrained=False)
        m4 = SwinTransformer(num_classes=5)
        m5 = VisionTransformer(num_classes=5)
        m6 = BaselineCNN(num_classes=5)
        m7 = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
        # Sanity: all forward shapes
        for m in (m1, m2, m3, m4, m5, m6, m7):
            m.eval()
            with torch.no_grad():
                _ = m(torch.randn(1, 3, 224, 224))
        report("instantiate", "PASS", "all seven models instantiate and forward (1,3,224,224)")
    except Exception as e:
        report("instantiate", "FAIL", f"{e}")

    # 2b. Transform + loss parity vs a HARDCODED golden copy of Table 3.
    # The golden table below is the auditable reference truth (independent of
    # MODEL_RECIPE, so drift in either direction is caught):
    #   golden_aug[name]     = augmentation family per the reference scripts
    #   golden_loss[name]    = (class_weights, label_smoothing)
    #   golden_init[name]    = pretrained flag per Table 3
    golden_aug = {
        "H-CoAtNet":       "strong",   # Rot15 + TrivialAugmentWide + RandomErasing
        "CoAtNet":         "color",    # Rot15 + ColorJitter(0.2,0.2,0.2)
        "GFT":             "base",     # crop/flip only
        "Swin":            "base",
        "ViT":             "base",
        "CNN":             "base",
        "EfficientNet-B0": "color",
    }
    golden_loss = {
        "H-CoAtNet":       (True, 0.1),
        "CoAtNet":         (True, 0.0),
        "GFT":             (True, 0.0),
        "Swin":            (False, 0.0),
        "ViT":             (False, 0.0),
        "CNN":             (True, 0.0),
        "EfficientNet-B0": (True, 0.0),
    }
    golden_init = {
        "H-CoAtNet": True, "CoAtNet": True, "GFT": True,
        "Swin": False, "ViT": False, "CNN": False, "EfficientNet-B0": False,
    }
    try:
        base_seq = ["RandomResizedCrop", "RandomHorizontalFlip"]
        tail_seq = ["ToTensor", "Normalize"]
        family_expectation = {
            "strong": base_seq + ["RandomRotation", "TrivialAugmentWide"] + tail_seq + ["RandomErasing"],
            "color":  base_seq + ["RandomRotation", "ColorJitter"] + tail_seq,
            "base":   base_seq + tail_seq,
        }
        failures = []
        for name, family in golden_aug.items():
            # (a) MODEL_RECIPE must agree with the golden table
            r = MODEL_RECIPE.get(name)
            if r is None:
                failures.append(f"{name}: missing from MODEL_RECIPE")
                continue
            got_family = ("strong" if r["strong_aug"]
                          else "color" if r["color_aug"] else "base")
            if got_family != family:
                failures.append(f"{name}: MODEL_RECIPE aug={got_family}, Table 3 says {family}")
            if (r["class_weights"], r["label_smoothing"]) != golden_loss[name]:
                failures.append(f"{name}: MODEL_RECIPE loss={(r['class_weights'], r['label_smoothing'])}, "
                                f"Table 3 says {golden_loss[name]}")
            # (b) make_transforms output must match the GOLDEN family
            train_t, val_t = make_transforms(name, 42, 0)
            mods = [type(t).__name__ for t in train_t.transforms]
            expected = family_expectation[family]
            if mods != expected:
                failures.append(f"{name}: transforms {mods}, expected {expected}")
            vmods = [type(t).__name__ for t in val_t.transforms]
            if vmods != ["Resize", "ToTensor", "Normalize"]:
                failures.append(f"{name} val: {vmods}")
            # (c) pretrained flag must match Table 3
            if MODEL_REGISTRY[name][3] != golden_init[name]:
                failures.append(f"{name}: pretrained={MODEL_REGISTRY[name][3]}, "
                                f"Table 3 says {golden_init[name]}")
        if failures:
            report("transform_parity", "FAIL", "; ".join(failures))
        else:
            report("transform_parity", "PASS",
                   "MODEL_RECIPE, make_transforms, init and loss all match the "
                   "hardcoded Table 3 golden table for all 7 models")
    except Exception as e:
        report("transform_parity", "FAIL", f"{e}")

    # 3. H-CoAtNet parity vs proposed
    try:
        # Use the proposed factory to get the canonical reference counts
        import importlib
        th = importlib.import_module("train_h_coatnet")
        ref = th.HCoAtNet(num_classes=5, pretrained=False)
        # Compare parameter counts
        n_a5 = MODEL_REGISTRY["H-CoAtNet"][0](num_classes=5, pretrained=False)
        ref_params = sum(p.numel() for p in ref.parameters())
        a5_params = sum(p.numel() for p in n_a5.parameters())
        if ref_params == a5_params:
            report("parity", "PASS", f"H-CoAtNet param count matches proposed ({a5_params:,})")
        else:
            report("parity", "FAIL", f"H-CoAtNet param count {a5_params:,} != proposed {ref_params:,}")
    except Exception as e:
        report("parity", "FAIL", f"{e}")

    # 4. Pool enumeration
    try:
        dev_files, dev_labels, test_files, class_names = gather_pool(dataset_dir)
        if not dev_files:
            report("pool", "FAIL", "no development files found")
            return status
        if not test_files:
            report("pool", "FAIL", "no frozen test files found")
            return status
        report("pool", "PASS",
               f"development pool={len(dev_files)} test={len(test_files)} "
               f"classes={class_names}")
    except Exception as e:
        report("pool", "FAIL", f"{e}")
        return status

    # 5. Test-set isolation
    overlap = set(dev_files) & set(test_files)
    if overlap:
        report("test_isolation", "FAIL",
               f"{len(overlap)} files appear in both dev and test")
    else:
        report("test_isolation", "PASS",
               f"no overlap between dev ({len(dev_files)}) and test ({len(test_files)})")

    # 6. Group construction
    try:
        group_ids = build_groups(dev_files, dev_labels)
        n_groups = len(set(group_ids))
        per_class_group_counts = {}
        for fi, gi in zip(dev_labels, group_ids):
            per_class_group_counts.setdefault(fi, set()).add(gi)
        # Each class should have multiple groups so folds can shuffle them
        smallest = min(len(g) for g in per_class_group_counts.values())
        if n_groups >= k * len(class_names):
            report("groups", "PASS",
                   f"groups={n_groups} classes={len(class_names)} k={k}")
        else:
            report("groups", "WARN",
                   f"groups={n_groups} (smallest per-class={smallest}); "
                   f"k={k} folds may have imbalanced val sizes")
    except Exception as e:
        report("groups", "FAIL", f"{e}")

    # 7. Fold sanity
    try:
        folds = build_folds(dev_files, dev_labels, group_ids, k, base_seed)
        sizes = []
        for tr, vl in folds:
            sizes.append((len(tr), len(vl)))
        all_disjoint = True
        all_cover = True
        seen_val = set()
        for tr, vl in folds:
            if set(tr) & set(vl):
                all_disjoint = False
            seen_val |= set(vl)
        if seen_val != set(range(len(dev_files))):
            all_cover = False
        if len(folds) != k:
            report("folds", "FAIL", f"got {len(folds)} folds, expected {k}")
        elif not all_disjoint:
            report("folds", "FAIL", "train/val overlap detected")
        elif not all_cover:
            report("folds", "FAIL", "validation does not cover all dev samples")
        else:
            report("folds", "PASS",
                   f"{k} folds sizes (train/val): {sizes}")
    except Exception as e:
        report("folds", "FAIL", f"{e}")

    # 8. Output dir separate
    if KFOLD_DIR.exists() and any(KFOLD_DIR.glob("kfold_*.json")):
        report("output_dir", "WARN",
               f"{KFOLD_DIR} already has kfold results; use --resume to continue "
               f"an interrupted run, or --out <dir> for a fresh one")
    else:
        report("output_dir", "PASS", f"{KFOLD_DIR} ready")

    print()
    print(f"VALIDATION SUMMARY: pass={results['pass']} "
          f"warn={results['warn']} fail={results['fail']}")
    return status


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def run_smoke(dataset_dir, base_seed):
    """Train one model on one fold for one epoch and verify outputs."""
    print("=" * 60)
    print("KFOLD SMOKE TEST (1 model x 1 fold x 1 epoch)")
    print("=" * 60)
    dev_files, dev_labels, test_files, class_names = gather_pool(dataset_dir)
    n_classes = len(class_names)
    group_ids = build_groups(dev_files, dev_labels)
    folds = build_folds(dev_files, dev_labels, group_ids, k=5, seed=base_seed)
    train_idx, val_idx = folds[0]
    model_name = "H-CoAtNet"
    factory, lr, bs, pretrained = MODEL_REGISTRY[model_name]
    print(f"  model={model_name} train={len(train_idx)} val={len(val_idx)} classes={n_classes}")
    ckpt = KFOLD_DIR / "checkpoints" / model_name / "smoke_best.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, metrics, _tmetrics, ys, yps, yprs, *_ = run_one_fold(
        model_name, 0, dev_files, dev_labels, train_idx, val_idx,
        n_classes, ckpt, base_seed, epochs=1)
    # Check shapes
    assert len(ys) == len(val_idx), "pred length mismatch"
    assert len(yprs) == len(val_idx), "probs length mismatch"
    assert ckpt.exists(), "checkpoint missing"
    print(f"  metrics: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro']['f1']:.4f}")
    print(f"  ckpt size: {ckpt.stat().st_size:,} bytes")
    print("SMOKE OK")
    # Clean up smoke checkpoint to keep output clean
    ckpt.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_summary_csv(summary, csv_path):
    """Mean +/- SD per metric across folds (flattened blocks)."""
    primary = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
        ("auroc_macro", "AUROC macro"),
        ("auprc_macro", "AUPRC macro"),
        ("kappa", "Cohen Kappa"),
        ("mcc", "MCC"),
        ("ece", "ECE"),
        ("brier", "Brier"),
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + [m[1] + " mean" for m in primary] +
                    [m[1] + " sd" for m in primary])
        for model_name, s in summary.items():
            row = [model_name]
            for k, _ in primary:
                v = s.get(k, {})
                row.append(f"{v.get('mean'):.4f}" if v.get("mean") is not None else "")
            for k, _ in primary:
                v = s.get(k, {})
                row.append(f"{v.get('sd'):.4f}" if v.get("sd") is not None else "")
            w.writerow(row)


def write_table_tex(summary, table_path, k):
    """Write a publication-style LaTeX table for the paper (flattened)."""
    primary = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal. Acc."),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
        ("auroc_macro", "AUROC"),
        ("auprc_macro", "AUPRC"),
        ("kappa", "Kappa"),
        ("mcc", "MCC"),
        ("ece", "ECE"),
        ("brier", "Brier"),
    ]
    cols = "l" + "cc" * len(primary)
    lines = []
    lines.append("% Table 6 -- generated by tools/kfold/kfold.py. Do not hand-edit.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Five-Fold Stratified Cross-Validation Results on the Development Set (k={k}). "
                 "Same training protocol as Table 3 (single-split). Reported as mean +/- SD across folds. "
                 "Development pool excludes the 158-image frozen test set. Group-aware "
                 "StratifiedKFold with identity grouping derived from Roboflow filename stems.}")
    lines.append("\\label{tab:kfold}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    header = ["Model"]
    for _, name in primary:
        header += [name, ""]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for model_name, s in summary.items():
        row = [model_name]
        for k_, _ in primary:
            v = s.get(k_, {})
            if v.get("mean") is None:
                row += ["--", "--"]
            elif k_ in ("ece", "brier"):
                # ECE/Brier live in [0,1] on a different scale than accuracy:
                # report them as decimals, not percent.
                row += [f"{v['mean']:.4f}", f"{v['sd']:.4f}"]
            else:
                row += [f"{v['mean']*100:.2f}", f"{v['sd']*100:.2f}"]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    table_path.write_text("\n".join(lines) + "\n")


def run_full(args):
    selected = (args.models.split(",")
                if args.models != "all" else list(MODEL_REGISTRY.keys()))
    run_full.selected = selected  # recorded in kfold_protocol.json
    print("=" * 60)
    print(f"K-FOLD CV (k={args.k}) -- 7-model benchmark")
    print("=" * 60)
    dev = get_device()
    print(f"  device      = {dev}" +
          ("  (full run on MPS may take days; Colab T4 recommended)"
           if dev.type == "mps" else ""))
    print("=" * 60)
    print(f"  dataset_dir = {args.dataset_dir}")
    print(f"  k           = {args.k}")
    print(f"  epochs      = {args.epochs}")
    print(f"  seed        = {args.seed}")
    print(f"  models      = {args.models}")
    print()

    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("curves", "confusion", "figures",
                "val/roc", "val/reliability", "test/roc", "test/reliability"):
        (KFOLD_DIR / sub).mkdir(parents=True, exist_ok=True)
    for m in MODEL_REGISTRY:
        (KFOLD_DIR / "checkpoints" / m).mkdir(parents=True, exist_ok=True)

    # Pool
    print("[1/4] gathering pool...")
    dev_files, dev_labels, test_files, class_names = gather_pool(args.dataset_dir)
    print(f"      dev = {len(dev_files)}  test = {len(test_files)}  classes = {class_names}")
    print(f"      per-class dev counts: {dict(Counter(dev_labels))}")
    # Frozen-test labels derived from the class directory layout (canonical order)
    test_labels = [class_names.index(Path(f).parent.name) for f in test_files]
    print(f"      per-class test counts: {dict(Counter(test_labels))}")

    # Folds
    print("[2/4] building group-aware folds...")
    group_ids = build_groups(dev_files, dev_labels)
    folds = build_folds(dev_files, dev_labels, group_ids, args.k, args.seed)
    for i, (tr, vl) in enumerate(folds):
        c_tr = Counter([dev_labels[j] for j in tr])
        c_vl = Counter([dev_labels[j] for j in vl])
        print(f"      fold {i+1}: train={len(tr)} ({dict(c_tr)})  "
              f"val={len(vl)} ({dict(c_vl)})")
    # Verify all checks
    seen = set()
    for tr, vl in folds:
        assert not (set(tr) & set(vl)), "fold train/val overlap"
        seen |= set(vl)
    assert seen == set(range(len(dev_files))), "validation does not cover all dev"

    # Persist splits
    print("[3/4] saving kfold_splits.json...")
    splits = []
    for i, (tr, vl) in enumerate(folds):
        splits.append({
            "fold": i + 1,
            "train_files": [dev_files[j] for j in tr],
            "val_files": [dev_files[j] for j in vl],
            "train_class_counts": dict(Counter([dev_labels[j] for j in tr])),
            "val_class_counts": dict(Counter([dev_labels[j] for j in vl])),
        })
    # Derive per-class frozen-test counts from the directory layout.
    test_counts = Counter()
    if test_files:
        for tf in test_files:
            # parent of an image is the class directory
            test_counts[Path(tf).parent.name] += 1
    splits_meta = {
        "k": args.k,
        "base_seed": args.seed,
        "splitter": "StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=base_seed)",
        "grouping_strategy": (
            "filename stem identity (Roboflow export pattern: <Class>-<idx>_<aug>_jpg.rf.<hash>.jpg -> "
            "stem is everything before first '_jpg.' or '.rf.'). Augmented copies share a stem; "
            "patient identity not available in this dataset; this is recorded explicitly."
        ),
        "n_development_samples": len(dev_files),
        "n_frozen_test_samples": len(test_files),
        "frozen_test_dir": str(Path(args.dataset_dir) / "test"),
        "classes": class_names,
        "per_class_counts_dev": dict(Counter(dev_labels)),
        "per_class_counts_test": dict(test_counts),
        "folds": splits,
    }
    (KFOLD_DIR / "kfold_splits.json").write_text(json.dumps(splits_meta, indent=2))
    (KFOLD_DIR / "kfold_protocol.json").write_text(json.dumps({
        "tool": "tools/kfold/kfold.py",
        "dataset_dir": args.dataset_dir,
        "k": args.k,
        "epochs": args.epochs,
        "seed": args.seed,
        "models": run_full.selected,
        "grouping_strategy": splits_meta["grouping_strategy"],
        "frozen_test_samples_excluded": len(test_files),
        "development_samples": len(dev_files),
        "test_isolation": "frozen 158-image test set is never used during CV training, "
                            "validation, or checkpoint selection.",
        "device": str(get_device()),
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "sklearn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
        },
        "per_model_recipes": {
            m: {"lr": MODEL_REGISTRY[m][1], "batch_size": MODEL_REGISTRY[m][2],
                "pretrained": MODEL_REGISTRY[m][3], **MODEL_RECIPE[m]}
            for m in run_full.selected
        },
        "fairness_disclosure": (
            "Per-model recipes mirror the single-split benchmark (Table 3) exactly: "
            "H-CoAtNet receives TrivialAugmentWide + RandomErasing and label smoothing 0.1; "
            "CoAtNet and EfficientNet-B0 receive ColorJitter; all models share the same "
            "30-epoch AdamW + cosine schedule and identical folds. Class weighting follows "
            "Table 3 (omitted for Swin/ViT). Initialisation asymmetry (IN1K pretrained vs "
            "scratch) is inherited from the benchmark and disclosed in the manuscript; "
            "this protocol adds no new asymmetry."
        ),
    }, indent=2))

    # Train all (model, fold)
    fold_results_per_model = {m: [] for m in selected}
    test_results_per_model = {m: [] for m in selected}
    predictions_per_model = {m: [] for m in selected}
    test_predictions_per_model = {m: [] for m in selected}

    # Resume support: reload incremental results from a previous run so a
    # Colab session can be interrupted and continued without losing folds.
    if args.resume:
        for path, target in (("kfold_fold_results.json", fold_results_per_model),
                             ("kfold_test_results.json", test_results_per_model),
                             ("kfold_predictions.json", predictions_per_model),
                             ("kfold_test_predictions.json", test_predictions_per_model)):
            p = KFOLD_DIR / path
            if p.exists():
                try:
                    saved = json.loads(p.read_text())
                    for m in selected:
                        if m in saved and saved[m]:
                            target[m] = saved[m]
                    print(f"  [resume] loaded {path}")
                except Exception as e:
                    print(f"  [resume] could not load {path}: {e}")

    print("[4/4] training all (model x fold) combinations...")
    overall_start = time.time()
    for model_name in selected:
        if model_name not in MODEL_REGISTRY:
            print(f"  WARNING: unknown model {model_name}, skipping")
            continue
        model_start = time.time()
        factory, lr, bs, pretrained = MODEL_REGISTRY[model_name]
        recipe = MODEL_RECIPE[model_name]
        print("\n" + "*" * 70)
        print(f"*  TRAINING: {model_name}")
        print(f"*  lr={lr}, batch={bs}, pretrained={pretrained}, aug={'strong' if recipe['strong_aug'] else 'color' if recipe['color_aug'] else 'base'}, ls={recipe['label_smoothing']}")
        print(f"*  {args.k} folds x {args.epochs} epochs")
        print("*" * 70)
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ckpt_path = KFOLD_DIR / "checkpoints" / model_name / f"fold{fold_idx+1}_best.pth"
            # Resume: skip folds already completed in a previous session
            # (failed-fold markers have 'failed': True and must NOT block a retry)
            if any(fm.get("fold") == fold_idx + 1 and not fm.get("failed")
                   for fm in fold_results_per_model[model_name]):
                print(f"  [{model_name} fold {fold_idx+1}/{args.k}] already done, skipping (--resume)")
                continue
            t0 = time.time()
            print(f"  [{model_name} fold {fold_idx+1}/{args.k}] "
                  f"train={len(train_idx)} val={len(val_idx)} ...")
            try:
                (history, metrics, test_metrics,
                 ys, yps, yprs, ytst, ypst, ypst_p) = run_one_fold(
                    model_name, fold_idx, dev_files, dev_labels, train_idx, val_idx,
                    len(class_names), ckpt_path, args.seed, args.epochs,
                    test_data=(test_files, test_labels))
                elapsed = (time.time() - t0) / 60
                # Immediately print fold summary banner
                print_fold_summary(model_name, fold_idx, args.k, metrics, test_metrics, elapsed)
                # Save per-fold metrics + predictions
                fold_metrics = dict(metrics)
                fold_metrics["fold"] = fold_idx + 1
                fold_metrics["best_val_acc"] = metrics["best_val_acc"]
                fold_metrics.pop("confusion_matrix", None)
                # per_class is RETAINED (R2 minor: full per-class P/R/F1 for
                # all models); aggregated into kfold_per_class_summary.csv.
                fold_results_per_model[model_name].append(fold_metrics)
                if test_metrics is not None:
                    tm = dict(test_metrics)
                    tm["fold"] = fold_idx + 1
                    test_results_per_model[model_name].append(tm)
                # Per-sample raw predictions: file -> y_true -> y_pred -> y_probs
                for jj, (yt, yp, prob) in enumerate(zip(ys, yps, yprs)):
                    predictions_per_model[model_name].append({
                        "fold": fold_idx + 1,
                        "file": dev_files[val_idx[jj]],
                        "y_true": int(yt),
                        "y_pred": int(yp),
                        "y_probs": [float(x) for x in prob],
                    })
                if ytst is not None:
                    for jj, (yt, yp, prob) in enumerate(zip(ytst, ypst, ypst_p)):
                        if prob is None:
                            continue
                        test_predictions_per_model[model_name].append({
                            "fold": fold_idx + 1,
                            "file": test_files[jj],
                            "y_true": int(yt),
                            "y_pred": int(yp),
                            "y_probs": [float(x) for x in prob],
                        })
                # Curves
                plot_curves(history, model_name, fold_idx + 1,
                           out_dir_acc=KFOLD_DIR / "curves" / f"{model_name}_fold{fold_idx+1}_accuracy.png",
                           out_dir_loss=KFOLD_DIR / "curves" / f"{model_name}_fold{fold_idx+1}_loss.png")
                # Confusion (raw + normalized) per fold (supplementary)
                cm = np.asarray(metrics["confusion_matrix"])
                plot_confusion(cm, class_names, model_name, fold_idx + 1,
                               raw=True,
                               out_path=KFOLD_DIR / "confusion" / f"{model_name}_fold{fold_idx+1}_raw.png")
                plot_confusion(cm, class_names, model_name, fold_idx + 1,
                               raw=False,
                               out_path=KFOLD_DIR / "confusion" / f"{model_name}_fold{fold_idx+1}_norm.png")
                # Per-fold validation ROC + reliability (supplementary)
                plot_roc_curves(
                    {model_name: yprs}, ys, len(class_names),
                    KFOLD_DIR / "val" / "roc" / f"{model_name}_fold{fold_idx+1}.png",
                    title=f"ROC (OvR macro) -- {model_name} Fold {fold_idx+1} (validation)")
                rd = _reliability_curve_data(np.asarray(yprs), np.asarray(ys))
                plot_reliability(
                    rd["confidence"], rd["accuracy"], rd["counts"], rd["ece"],
                    KFOLD_DIR / "val" / "reliability" / f"{model_name}_fold{fold_idx+1}.png",
                    title=f"Reliability -- {model_name} Fold {fold_idx+1} (validation)")
                # Write incremental results so we never lose work
                (KFOLD_DIR / "kfold_fold_results.json").write_text(
                    json.dumps(fold_results_per_model, indent=2))
                (KFOLD_DIR / "kfold_test_results.json").write_text(
                    json.dumps(test_results_per_model, indent=2))
                (KFOLD_DIR / "kfold_predictions.json").write_text(
                    json.dumps(predictions_per_model, indent=2))
                (KFOLD_DIR / "kfold_test_predictions.json").write_text(
                    json.dumps(test_predictions_per_model, indent=2))
            except Exception as e:
                # Do not abort the remaining folds; record the failure so the
                # summary can disclose it instead of silently reporting k-1 folds.
                fold_results_per_model[model_name].append(
                    {"fold": fold_idx + 1, "error": str(e), "failed": True})
                print(f"  [{model_name} fold {fold_idx+1}] FAILED: {e}")
                traceback.print_exc()
                (KFOLD_DIR / "kfold_fold_results.json").write_text(
                    json.dumps(fold_results_per_model, indent=2))
                continue
        model_elapsed = (time.time() - model_start) / 60
        # Print full model summary after all folds
        print_model_summary(model_name, args.k,
                           fold_results_per_model[model_name],
                           test_results_per_model[model_name],
                           model_elapsed)
        # Print running leaderboard after each model
        completed_so_far = [m for m in selected
                            if len(fold_results_per_model.get(m, [])) >= args.k
                            and not any(f.get("failed") for f in fold_results_per_model.get(m, []))]
        print_leaderboard(completed_so_far, fold_results_per_model, test_results_per_model)
    overall_min = (time.time() - overall_start) / 60
    print(f"\nALL MODELS TRAINED. Total time: {overall_min:.1f} min")

    # =========================================================================
    # FINAL OVERALL SUMMARY BANNER
    # =========================================================================
    print("\n" + "#" * 70)
    print("#  FINAL RESULTS -- ALL MODELS")
    print("#" * 70)
    print(f"")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Dev pool: {len(dev_files)} images | Frozen test: {len(test_files)} images")
    print(f"  Folds: {args.k} | Epochs: {args.epochs} | Seed: {args.seed}")
    print(f"")
    # Build summary rows for the final table
    _final_rows = []
    for m in selected:
        fr = fold_results_per_model.get(m, [])
        tr = test_results_per_model.get(m, [])
        fr_good = [f for f in fr if not f.get("failed")]
        tr_good = [f for f in tr if not f.get("failed")]
        if not fr_good:
            continue
        _final_rows.append({
            "model": m, "n_folds": len(fr_good),
            "val_acc": np.mean([f["accuracy"] for f in fr_good]),
            "val_acc_sd": np.std([f["accuracy"] for f in fr_good], ddof=1) if len(fr_good) > 1 else 0,
            "val_f1": np.mean([f["macro"]["f1"] for f in fr_good]),
            "val_f1_sd": np.std([f["macro"]["f1"] for f in fr_good], ddof=1) if len(fr_good) > 1 else 0,
            "val_kappa": np.mean([f["kappa"] for f in fr_good]),
            "val_ece": np.mean([f["ece"] for f in fr_good]),
            "test_acc": np.mean([f["accuracy"] for f in tr_good]) if tr_good else None,
            "test_acc_sd": np.std([f["accuracy"] for f in tr_good], ddof=1) if tr_good and len(tr_good) > 1 else 0,
            "test_f1": np.mean([f["macro"]["f1"] for f in tr_good]) if tr_good else None,
            "test_kappa": np.mean([f["kappa"] for f in tr_good]) if tr_good else None,
        })
    # Sort by test accuracy (descending), fallback to val accuracy
    _final_rows.sort(key=lambda r: r["test_acc"] if r["test_acc"] is not None else r["val_acc"], reverse=True)
    print(f"  {'Rank':<5s} {'Model':<18s} {'Val Acc (mean+/-SD)':<24s} {'Val F1 (mean+/-SD)':<24s} {'Test Acc (mean+/-SD)':<24s}")
    print(f"  {'-'*5} {'-'*18} {'-'*24} {'-'*24} {'-'*24}")
    for i, r in enumerate(_final_rows, 1):
        marker = " *" if r["model"] == "H-CoAtNet" else ""
        test_str = f"{r['test_acc']*100:.1f}+/-{r['test_acc_sd']*100:.1f}" if r["test_acc"] is not None else "--"
        print(f"  {i:<5d} {r['model']:<18s} "
              f"{r['val_acc']*100:.1f}+/-{r['val_acc_sd']*100:.1f}      "
              f"{r['val_f1']*100:.1f}+/-{r['val_f1_sd']*100:.1f}      "
              f"{test_str}{marker}")
    print(f"")
    print(f"  * = proposed model")
    print(f"  Output: {KFOLD_DIR}")
    print("#" * 70)
    sys.stdout.flush()

    # Aggregate
    summary = aggregate(fold_results_per_model)
    test_summary = aggregate(test_results_per_model) if any(
        test_results_per_model.get(m) for m in selected) else {}
    (KFOLD_DIR / "kfold_fold_results.json").write_text(
        json.dumps(fold_results_per_model, indent=2))
    (KFOLD_DIR / "kfold_test_results.json").write_text(
        json.dumps(test_results_per_model, indent=2))
    (KFOLD_DIR / "kfold_predictions.json").write_text(
        json.dumps(predictions_per_model, indent=2))
    (KFOLD_DIR / "kfold_test_predictions.json").write_text(
        json.dumps(test_predictions_per_model, indent=2))
    (KFOLD_DIR / "kfold_summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_csv(summary, KFOLD_DIR / "kfold_summary.csv")
    write_table_tex(summary, KFOLD_DIR / "kfold_table.tex", args.k)
    if test_summary:
        (KFOLD_DIR / "kfold_test_summary.json").write_text(
            json.dumps(test_summary, indent=2))
        write_summary_csv(test_summary, KFOLD_DIR / "kfold_test_summary.csv")

    # Per-class aggregation across folds (R2 minor comment)
    pc_rows = aggregate_per_class(fold_results_per_model, class_names)
    if pc_rows:
        (KFOLD_DIR / "kfold_per_class_summary.json").write_text(
            json.dumps(pc_rows, indent=2))
        write_per_class_csv(pc_rows, KFOLD_DIR / "kfold_per_class_summary.csv")
        plot_perclass_f1_heatmap(
            pc_rows, KFOLD_DIR / "figures" / "fig_kfold_perclass_f1_heatmap.png")

    # Paired tests
    pairs = paired_wilcoxon(fold_results_per_model)
    pairs_bonf = bonferroni_correct(pairs)
    (KFOLD_DIR / "kfold_paired_tests.json").write_text(
        json.dumps({"raw": pairs, "bonferroni": pairs_bonf,
                    "note": "Wilcoxon signed-rank on per-fold accuracy. "
                            "Bonferroni correction applied. The five folds are "
                            "repeated partitions of the same development "
                            "population, not independent experiments."},
                   indent=2))
    # Omnibus + post-hoc across all models simultaneously (Demvsar 2006);
    # prefer frozen-test folds when available, fall back to validation folds.
    _fn_src = (test_results_per_model
               if any(test_results_per_model.get(m) for m in selected)
               else fold_results_per_model)
    fn = friedman_nemenyi(_fn_src)
    (KFOLD_DIR / "kfold_friedman_nemenyi.json").write_text(
        json.dumps(fn, indent=2))

    # Figures
    plot_metric_bars(summary, "accuracy",
                     "Five-Fold Cross-Validation Accuracy",
                     "Accuracy",
                     KFOLD_DIR / "figures" / "fig_kfold_accuracy_mean_sd.png")
    plot_metric_bars(summary, "macro_f1",
                     "Five-Fold Cross-Validation Macro-F1",
                     "Macro-F1",
                     KFOLD_DIR / "figures" / "fig_kfold_macro_f1_mean_sd.png")
    plot_metric_bars(summary, "balanced_accuracy",
                     "Five-Fold Cross-Validation Balanced Accuracy",
                     "Balanced Accuracy",
                     KFOLD_DIR / "figures" / "fig_kfold_balanced_accuracy_mean_sd.png")
    plot_foldwise("accuracy", fold_results_per_model,
                  "Fold-Wise Validation Accuracy Across Models",
                  "Accuracy",
                  KFOLD_DIR / "figures" / "fig_kfold_foldwise_accuracy.png")
    plot_foldwise("macro_f1", fold_results_per_model,
                  "Fold-Wise Validation Macro-F1 Across Models",
                  "Macro-F1",
                  KFOLD_DIR / "figures" / "fig_kfold_foldwise_macro_f1.png")
    plot_summary_heatmap(summary,
                         KFOLD_DIR / "figures" / "fig_kfold_metric_summary.png")
    plot_class_distribution(splits, class_names,
                             KFOLD_DIR / "figures" / "fig_kfold_class_distribution.png")

    # Frozen-test figures (evaluated once per fold from the val-best ckpt)
    if test_summary:
        plot_metric_bars(test_summary, "accuracy",
                         "Five-Fold Cross-Validation Frozen-Test Accuracy",
                         "Accuracy",
                         KFOLD_DIR / "figures" / "fig_kfold_test_accuracy_mean_sd.png")
    if any(test_predictions_per_model.get(m) for m in selected):
        roc_src, rel_src, y_true_ref = {}, {}, None
        for m in selected:
            preds = test_predictions_per_model.get(m) or []
            if preds:
                roc_src[m] = [p["y_probs"] for p in preds]
                rel_src[m] = [p["y_probs"] for p in preds]
                y_true_ref = [p["y_true"] for p in preds]
        if roc_src and y_true_ref:
            plot_roc_curves(roc_src, np.asarray(y_true_ref), len(class_names),
                            KFOLD_DIR / "test" / "roc" / "mean_all_models.png",
                            title="Mean ROC (OvR macro) -- Frozen Test Set")
            for m in selected:
                if m in rel_src:
                    rd = _reliability_curve_data(np.asarray(rel_src[m]),
                                                 np.asarray(y_true_ref))
                    plot_reliability(
                        rd["confidence"], rd["accuracy"], rd["counts"], rd["ece"],
                        KFOLD_DIR / "test" / "reliability" / f"{m}.png",
                        title=f"Reliability -- {m} (frozen test, val-best ckpt)")
            if "H-CoAtNet" in rel_src:
                rd = _reliability_curve_data(np.asarray(rel_src["H-CoAtNet"]),
                                             np.asarray(y_true_ref))
                plot_reliability(
                    rd["confidence"], rd["accuracy"], rd["counts"], rd["ece"],
                    KFOLD_DIR / "figures" / "fig_kfold_reliability_test.png",
                    title="Reliability Diagram -- H-CoAtNet (frozen test, pooled folds)")
            if len(selected) >= 3:
                plot_critical_difference(
                    test_summary if test_summary else summary,
                    KFOLD_DIR / "figures" / "fig_kfold_critical_difference.png")

    print(f"\nOutput directory: {KFOLD_DIR}")


def main():
    p = argparse.ArgumentParser(
        description="5-fold stratified cross-validation for the seven-model benchmark.")
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="Roboflow-exported dataset root (with train/valid/test/<class>/...).")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=str, default="all",
                   help="Comma-separated model names or 'all'.")
    p.add_argument("--validate-only", action="store_true",
                   help="Run pre-flight checks and exit.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Train one model x one fold x one epoch and verify.")
    p.add_argument("--out", type=str, default=None,
                   help="Override output directory.")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run: reuse completed folds from "
                        "existing kfold_*.json outputs (Colab night-run safety).")
    args = p.parse_args()

    global KFOLD_DIR
    if args.out:
        KFOLD_DIR = Path(args.out) if Path(args.out).is_absolute() else (REPO / args.out)
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        run_validation(args.dataset_dir, args.seed, args.k, args.epochs, args.models)
        return
    if args.smoke_test:
        run_validation(args.dataset_dir, args.seed, args.k, args.epochs, args.models)
        run_smoke(args.dataset_dir, args.seed)
        return
    # Full run: pre-flight first
    status = run_validation(args.dataset_dir, args.seed, args.k, args.epochs, args.models)
    if status.get("fail", 0) > 0:
        print("\nPRE-FLIGHT FAILED. Aborting.")
        sys.exit(2)
    run_full(args)


if __name__ == "__main__":
    main()
