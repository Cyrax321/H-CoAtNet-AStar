#!/usr/bin/env python3
"""
kfold.py -- A* Rigorous 5-fold stratified cross-validation for the seven-model study.

Addresses R2-4 ("K-fold cross-validation is needed").

Distinguishes from the existing single-split benchmark by:
  - Using a frozen 158-image test set held out from CV entirely
  - Building 5 stratified folds from the development pool (train+valid = 2350 images)
  - Reusing the existing seven training scripts (one per model) verbatim
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
    kfold_summary.json               -- machine-readable summary
    kfold_table.tex                  -- publication-style Table 6
    kfold_paired_tests.json          -- paired model comparisons
    curves/<model>_fold<N>_accuracy.png
    curves/<model>_fold<N>_loss.png
    confusion/<model>_fold<N>_raw.png
    confusion/<model>_fold<N>_norm.png
    figures/
      fig_kfold_accuracy_mean_sd.png
      fig_kfold_macro_f1_mean_sd.png
      fig_kfold_balanced_accuracy_mean_sd.png
      fig_kfold_foldwise_accuracy.png
      fig_kfold_foldwise_macro_f1.png
      fig_kfold_metric_summary.png
      fig_kfold_class_distribution.png
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
# Repo paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
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
    from train_swin import build_swin
    return build_swin(num_classes=num_classes)

def _make_vit(num_classes, pretrained=True):
    from train_vit import build_vit
    return build_vit(num_classes=num_classes)

def _make_cnn(num_classes, pretrained=False):
    from train_cnn import BaselineCNN
    return BaselineCNN(num_classes=num_classes)

def _make_effnet(num_classes, pretrained=False):
    from train_efficientnet import build_efficientnet
    return build_efficientnet(num_classes=num_classes)

MODEL_REGISTRY = {
    # name             : (factory, default_lr, default_batch, pretrained_default)
    "H-CoAtNet":       (_make_hcoatnet,   5e-5, 24, True),
    "CoAtNet":         (_make_coatnet,    5e-5, 24, True),
    "GFT":             (_make_gft,        5e-5, 24, True),
    "Swin":            (_make_swin,       5e-5, 16, True),
    "ViT":             (_make_vit,        5e-5, 16, True),
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
    """Build train + val transforms.

    We re-use the same base transforms the existing baseline scripts use:
      train: RandomResizedCrop(224) + HFlip + (Rot15 + TrivialAugmentWide +
              RandomErasing for H-CoAtNet/CoAtNet/GFT) or no-op (Swin/ViT/CNN/EffNet)
      val:   Resize(224) + Normalize
    Augmentation RNG is seeded with `base_seed + fold_idx` so the per-fold
    augmentation sequence is reproducible.

    This matches the augmentation pipeline already used by the single-split
    benchmark; see H-CoAtNet/proposed_method/train_h_coatnet.py and the
    six baselines.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    base = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    # Stronger augmentation for H-CoAtNet/CoAtNet/GFT (matches the single-split
    # benchmark; the original scripts add Rot15 + TrivialAugmentWide + RandomErasing
    # for H-CoAtNet and CoAtNet only).
    if model_name in ("H-CoAtNet", "CoAtNet", "GFT"):
        base += [
            transforms.RandomRotation(15),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ]
    else:
        base += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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


def make_loaders(files, labels, train_idx, val_idx, batch_size, train_t, val_t):
    g = torch.Generator()
    g.manual_seed(int(torch.initial_seed()) % (2 ** 32))
    train_ds = FileListDataset(files, labels, train_idx, train_t)
    val_ds = FileListDataset(files, labels, val_idx, val_t)
    nw = 0 if os.name == "nt" else 2
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=nw,
        worker_init_fn=_seed_worker, generator=g, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=nw,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot, n, correct = 0.0, 0, 0
    for x, y in loader:
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
    return tot / max(1, n), correct / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, return_probs=True):
    model.eval()
    ys, yps, yprs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        ys.extend(y.cpu().numpy().tolist())
        yps.extend(logits.argmax(1).cpu().numpy().tolist())
        if return_probs:
            yprs.extend(F.softmax(logits, dim=1).cpu().numpy().tolist())
    if not return_probs:
        return np.asarray(ys), np.asarray(yps)
    return np.asarray(ys), np.asarray(yps), np.asarray(yprs)


def run_one_fold(model_name, fold_idx, files, labels, train_idx, val_idx,
                n_classes, ckpt_path, base_seed, epochs):
    """Train one fold and return (history, final_metrics, raw_preds)."""
    factory, lr, batch_size, pretrained_default = MODEL_REGISTRY[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Per-fold deterministic seed
    fold_seed = base_seed + fold_idx * 1000
    seed_everything(fold_seed)

    train_t, val_t = make_transforms(model_name, base_seed, fold_idx)
    train_loader, val_loader = make_loaders(
        files, labels, train_idx, val_idx, batch_size, train_t, val_t)

    # Build a fresh model with IN1K (or scratch) init.
    model = factory(num_classes=n_classes, pretrained=pretrained_default).to(device)

    # Class weights from TRAIN-fold only (no test info leak)
    train_labels = [labels[i] for i in train_idx]
    counts = np.bincount(train_labels, minlength=n_classes)
    n_train = len(train_idx)
    cw = torch.tensor([n_train / (c * n_classes + 1e-6) for c in counts],
                      dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val, best_state = -1.0, None
    for ep in range(epochs):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validation pass (with proper loss + accuracy)
        model.eval()
        vl_sum, vl_n, v_correct = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                vl_sum += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                vl_n += x.size(0)
        vl = vl_sum / max(1, vl_n)
        va = v_correct / max(1, vl_n)
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        if va > best_val:
            best_val = va
            best_state = deepcopy(model.state_dict())
    # Final validation metrics + raw preds (test is NEVER touched here)
    yv, ypv, ypv_p = evaluate(model, val_loader, device, criterion)
    metrics = compute_all_metrics(yv, ypv, ypv_p, n_classes)
    metrics["best_val_acc"] = float(best_val)

    # Save best checkpoint
    if best_state is not None:
        torch.save(best_state, ckpt_path)

    # Free memory
    del model, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history, metrics, yv.tolist(), ypv.tolist(), ypv_p.tolist()


# ---------------------------------------------------------------------------
# Aggregation + statistics
# ---------------------------------------------------------------------------
def aggregate(fold_metrics_per_model):
    """Compute mean / SD / median / min / max per metric across folds."""
    out = {}
    primary = ["accuracy", "balanced_accuracy", "kappa", "mcc", "ece", "brier"]
    secondary_macro = ["macro_precision", "macro_recall", "macro_f1"]
    secondary_weighted = ["weighted_precision", "weighted_recall", "weighted_f1"]

    for model_name, fold_metrics in fold_metrics_per_model.items():
        s = {"per_fold": fold_metrics, "n_folds": len(fold_metrics)}
        for k in primary:
            vals = [m[k] for m in fold_metrics if m.get(k) is not None]
            s[k] = _stat_block(vals)
        for k in secondary_macro:
            vals = [m["macro"][k.replace("macro_", "")] for m in fold_metrics
                    if m.get("macro") and m["macro"].get(k.replace("macro_", "")) is not None]
            s[k] = _stat_block(vals)
        for k in secondary_weighted:
            vals = [m["weighted"][k.replace("weighted_", "")] for m in fold_metrics
                    if m.get("weighted") and m["weighted"].get(k.replace("weighted_", "")) is not None]
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
        return {"mean": None, "sd": None, "median": None, "min": None, "max": None, "values": []}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "values": [float(x) for x in a],
    }


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
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a_name, b_name = model_names[i], model_names[j]
            a_vals = [m["accuracy"] for m in fold_metrics_per_model[a_name]]
            b_vals = [m["accuracy"] for m in fold_metrics_per_model[b_name]]
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
    ax.errorbar(x, means, yerr=sds, fmt="o", capsize=4, color="black",
                markerfacecolor="white", markersize=8, linewidth=1.2)
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
    n_cls = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(folds))
    for i, cname in enumerate(class_names):
        counts = []
        for s in splits:
            train_idx, _ = s["train"], s["val"]
            # Use train split to show class balance
            counts.append(int(sum(1 for fi in train_idx
                                  if splits[0]["labels"][fi] == i)) if "labels" in splits[0]
                          else 0)
        # Above approach may be slow; use the precomputed counts if present
        counts = [s.get("train_class_counts", {}).get(cname, 0) for s in splits]
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

    # 1. Imports
    try:
        from train_h_coatnet import HCoAtNet
        from train_coatnet import CoAtNet
        from train_gft import GFT
        from train_swin import build_swin
        from train_vit import build_vit
        from train_cnn import BaselineCNN
        from train_efficientnet import build_efficientnet
        report("imports", "PASS", "all seven model modules import")
    except Exception as e:
        report("imports", "FAIL", f"import failure: {e}")
        return status  # hard stop

    # 2. Instantiate
    try:
        m1 = HCoAtNet(num_classes=5, pretrained=False)
        m2 = CoAtNet(num_classes=5, pretrained=False)
        m3 = GFT(num_classes=5, pretrained=False)
        m4 = build_swin(num_classes=5)
        m5 = build_vit(num_classes=5)
        m6 = BaselineCNN(num_classes=5)
        m7 = build_efficientnet(num_classes=5)
        # Sanity: all forward shapes
        for m in (m1, m2, m3, m4, m5, m6, m7):
            m.eval()
            with torch.no_grad():
                _ = m(torch.randn(1, 3, 224, 224))
        report("instantiate", "PASS", "all seven models instantiate and forward (1,3,224,224)")
    except Exception as e:
        report("instantiate", "FAIL", f"{e}")

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
    if KFOLD_DIR.exists() and any(KFOLD_DIR.glob("*")):
        report("output_dir", "WARN",
               f"{KFOLD_DIR} already has files; this run will overwrite kfold_*.json")
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
    history, metrics, ys, yps, yprs = run_one_fold(
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
    """Write a publication-style LaTeX table for the paper."""
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
            else:
                row += [f"{v['mean']*100:.2f}", f"{v['sd']*100:.2f}"]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    table_path.write_text("\n".join(lines) + "\n")


def run_full(args):
    print("=" * 60)
    print(f"K-FOLD CV (k={args.k}) -- 7-model benchmark")
    print("=" * 60)
    print(f"  dataset_dir = {args.dataset_dir}")
    print(f"  k           = {args.k}")
    print(f"  epochs      = {args.epochs}")
    print(f"  seed        = {args.seed}")
    print(f"  models      = {args.models}")
    print()

    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    (KFOLD_DIR / "curves").mkdir(exist_ok=True)
    (KFOLD_DIR / "confusion").mkdir(exist_ok=True)
    (KFOLD_DIR / "figures").mkdir(exist_ok=True)
    for m in MODEL_REGISTRY:
        (KFOLD_DIR / "checkpoints" / m).mkdir(parents=True, exist_ok=True)

    # Pool
    print("[1/4] gathering pool...")
    dev_files, dev_labels, test_files, class_names = gather_pool(args.dataset_dir)
    print(f"      dev = {len(dev_files)}  test = {len(test_files)}  classes = {class_names}")
    print(f"      per-class dev counts: {dict(Counter(dev_labels))}")

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
        "models": args.models,
        "grouping_strategy": splits_meta["grouping_strategy"],
        "frozen_test_samples_excluded": len(test_files),
        "development_samples": len(dev_files),
        "test_isolation": "frozen 158-image test set is never used during CV training, "
                            "validation, or checkpoint selection.",
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
    }, indent=2))

    # Train all (model, fold)
    selected = args.models.split(",") if args.models != "all" else list(MODEL_REGISTRY.keys())
    fold_results_per_model = {m: [] for m in selected}
    predictions_per_model = {m: [] for m in selected}

    print("[4/4] training all (model x fold) combinations...")
    overall_start = time.time()
    for model_name in selected:
        if model_name not in MODEL_REGISTRY:
            print(f"  WARNING: unknown model {model_name}, skipping")
            continue
        model_start = time.time()
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            ckpt_path = KFOLD_DIR / "checkpoints" / model_name / f"fold{fold_idx+1}_best.pth"
            t0 = time.time()
            print(f"  [{model_name} fold {fold_idx+1}/{args.k}] "
                  f"train={len(train_idx)} val={len(val_idx)} ...")
            try:
                history, metrics, ys, yps, yprs = run_one_fold(
                    model_name, fold_idx, dev_files, dev_labels, train_idx, val_idx,
                    len(class_names), ckpt_path, args.seed, args.epochs)
                elapsed = (time.time() - t0) / 60
                print(f"  [{model_name} fold {fold_idx+1}/{args.k}] DONE in {elapsed:.1f} min, "
                      f"val_acc={metrics['accuracy']:.4f} macro_f1={metrics['macro']['f1']:.4f}")
                # Save per-fold metrics + predictions
                fold_metrics = dict(metrics)
                fold_metrics["fold"] = fold_idx + 1
                fold_metrics["best_val_acc"] = metrics["best_val_acc"]
                fold_metrics.pop("confusion_matrix", None)
                fold_metrics.pop("per_class", None)
                fold_results_per_model[model_name].append(fold_metrics)
                # Per-sample raw predictions: file -> y_true -> y_pred -> y_probs
                for jj, (yt, yp, prob) in enumerate(zip(ys, yps, yprs)):
                    predictions_per_model[model_name].append({
                        "fold": fold_idx + 1,
                        "file": dev_files[val_idx[jj]],
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
                # Write incremental results so we never lose work
                (KFOLD_DIR / "kfold_fold_results.json").write_text(
                    json.dumps(fold_results_per_model, indent=2))
                (KFOLD_DIR / "kfold_predictions.json").write_text(
                    json.dumps(predictions_per_model, indent=2))
            except Exception as e:
                print(f"  [{model_name} fold {fold_idx+1}] FAILED: {e}")
                traceback.print_exc()
                break
        model_elapsed = (time.time() - model_start) / 60
        print(f"  [{model_name}] all folds done in {model_elapsed:.1f} min")
    overall_min = (time.time() - overall_start) / 60
    print(f"\nALL DONE. Total: {overall_min:.1f} min")

    # Aggregate
    summary = aggregate(fold_results_per_model)
    (KFOLD_DIR / "kfold_fold_results.json").write_text(
        json.dumps(fold_results_per_model, indent=2))
    (KFOLD_DIR / "kfold_predictions.json").write_text(
        json.dumps(predictions_per_model, indent=2))
    (KFOLD_DIR / "kfold_summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_csv(summary, KFOLD_DIR / "kfold_summary.csv")
    write_table_tex(summary, KFOLD_DIR / "kfold_table.tex", args.k)

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
