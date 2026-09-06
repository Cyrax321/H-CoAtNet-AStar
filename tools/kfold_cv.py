#!/usr/bin/env python3
"""
kfold_cv.py -- A* Reproducibility: K-fold cross-validation for H-CoAtNet ablation.

Addresses R2-4: "K-fold cross-validation is needed" (Reviewer 2, comment 4).

Usage:
  python tools/kfold_cv.py --dataset_dir /path/to/ich-s-7lnsj --k 5 --variants all --epochs 30 --seed 42

For each variant in {cnnOnly, vitOnly, seNoPrune, randomPrune, directL2,
hierarchical, hierarchicalRaw, noViT}, runs k-fold stratified CV. For each fold
the model is trained from scratch on k-1 folds and evaluated on the held-out
fold (test set per-fold). Outputs:

  results/kfold_results.json      -- per-fold metrics for each variant
  results/kfold_summary.csv        -- mean +/- std across folds
  results/kfold_summary.md         -- human-readable Table 6

Fairness lock (identical for every fold + variant):
  - Same IN1K pretrained ConvNeXt-Tiny
  - Same AdamW lr=5e-5, weight_decay=0.01, CosineAnnealing T_max=epochs
  - Same label smoothing 0.1 + class-weighted CE
  - Same augmentation pipeline
  - Same seed=42 (with k-fold-shuffled splits; per-fold seed = base_seed + fold_idx)

Test-isolation per fold:
  - test set = the held-out fold (one fold at a time)
  - training uses k-1 folds
  - validation = a small fixed slice of the training folds (10% of train)
  - test never enters training or selection
"""

import os
import sys
import json
import argparse
import random
import gc
import time
import shutil
import traceback
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "H-CoAtNet" / "proposed_method"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablation study"))

from train_h_coatnet import HierarchicalSE
from ablation_study import (
    AblationCoAtNet, VARIANTS, seed_everything,
    build_transforms, _verify_split_manifest,
    compute_ece, compute_brier,
)
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
K_RESULTS_DIR = RESULTS_DIR  # output into the same results/ folder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gather_pool(dataset_dir):
    """Walk dataset_dir and gather ALL image files with their class labels.

    Supports both:
      (a) Roboflow structure: <dataset>/{train,valid,test}/<class>/<files>
      (b) Flat structure:    <dataset>/<class>/<files>   (k-fold builds train/val/test splits)
    """
    dataset_dir = Path(dataset_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # Detect Roboflow structure
    has_roboflow = (dataset_dir / "train").exists() and (dataset_dir / "valid").exists()
    files, labels = [], []
    class_names = None
    if has_roboflow:
        for split in ["train", "valid", "test"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            if class_names is None:
                class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            cls_to_idx = {c: i for i, c in enumerate(class_names)}
            for cls in class_names:
                for p in (split_dir / cls).rglob("*"):
                    if p.is_file() and p.suffix.lower() in image_exts:
                        files.append(str(p))
                        labels.append(cls_to_idx[cls])
    else:
        # Flat: dataset_dir/<class>/<files>
        class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        cls_to_idx = {c: i for i, c in enumerate(class_names)}
        for cls in class_names:
            for p in (dataset_dir / cls).rglob("*"):
                if p.is_file() and p.suffix.lower() in image_exts:
                    files.append(str(p))
                    labels.append(cls_to_idx[cls])
    return files, labels, class_names


def make_folds(labels, k, seed):
    """StratifiedKFold yields k splits. Returns list of (train_idx, test_idx)."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


def make_loaders_for_fold(files, labels, train_idx, test_idx, batch_size,
                          train_tf, eval_tf, num_workers):
    """Build a train_loader and test_loader for one fold.

    Validation = fixed 10% slice of the training indices (deterministic).
    """
    # Build ImageFolder-like dataset with custom file list
    # We use a Subset approach: build a base ImageFolder on the parent dir
    # (each sub-folder of <root>/<class>/<files>). For simplicity we
    # construct a tiny wrapper.
    class FileListDataset(torch.utils.data.Dataset):
        def __init__(self, file_list, label_list, indices, transform):
            self.file_list = file_list
            self.label_list = label_list
            self.indices = list(indices)
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            idx = self.indices[i]
            from PIL import Image
            img = Image.open(self.file_list[idx]).convert("RGB")
            return self.transform(img), self.label_list[idx]

    # Split off 10% of train for validation (deterministic per fold)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(round(0.1 * len(train_idx))))
    val_idx = perm[:n_val]
    train_only_idx = perm[n_val:]

    train_ds = FileListDataset(files, labels, train_only_idx, train_tf)
    val_ds = FileListDataset(files, labels, val_idx, eval_tf)
    test_ds = FileListDataset(files, labels, test_idx, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    tot, preds, tgts = 0.0, [], []
    for img, y in loader:
        img, y = img.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tot += loss.item() * img.size(0)
        preds.extend(out.argmax(1).detach().cpu().numpy())
        tgts.extend(y.detach().cpu().numpy())
    avg = tot / max(1, len(loader.dataset))
    acc = float(np.mean(np.array(preds) == np.array(tgts))) if preds else 0.0
    return avg, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot, preds, tgts, probs = 0.0, [], [], []
    for img, y in loader:
        img, y = img.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = model(img)
        tot += criterion(out, y).item() * img.size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
        probs.extend(F.softmax(out, dim=1).cpu().numpy())
    avg = tot / max(1, len(loader.dataset))
    acc = float(np.mean(np.array(preds) == np.array(tgts))) if preds else 0.0
    return avg, acc, np.array(tgts), np.array(preds), np.array(probs)


def run_one_fold(variant_name, fold_idx, files, labels, train_idx, test_idx,
                 class_names, epochs, seed, lr, wd, ls, batch_size,
                 pretrained, results_dir):
    """Train one fold of one variant. Return metrics dict."""
    train_tf, eval_tf = build_transforms()
    num_workers = 0 if os.name == "nt" else 2
    train_loader, val_loader, test_loader = make_loaders_for_fold(
        files, labels, train_idx, test_idx, batch_size, train_tf, eval_tf, num_workers)

    # Seed torch + numpy for THIS fold (deterministic but per-fold varied)
    seed_everything(seed + fold_idx)

    cfg = VARIANTS[variant_name]
    model = AblationCoAtNet(
        use_vit=cfg["use_vit"], use_se=cfg["use_se"], exec_mode=cfg["exec_mode"],
        num_classes=len(class_names), pretrained=pretrained,
    ).to(DEVICE)

    # Class weights from the TRAIN-fold only (not test fold)
    train_labels = [labels[i] for i in train_idx]
    counts = np.bincount(train_labels, minlength=len(class_names))
    n_train = len(train_idx)
    nc = len(class_names)
    cw = torch.tensor([n_train / (c * nc + 1e-6) for c in counts],
                      dtype=torch.float, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=ls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train + val-select best checkpoint, test once
    best_val, best_state = -1.0, None
    fold_tag = f"{variant_name}_fold{fold_idx}"
    ckpt_path = results_dir / f"best_{fold_tag}.pth"
    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)
    # Test on held-out fold (single evaluation)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_loss, test_acc, yt, yp, ypr = evaluate(model, test_loader, criterion)

    # Aggregate metrics
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
    except Exception:
        auroc, auprc = float("nan"), float("nan")
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro",
                                                    zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted",
                                                    zero_division=0)

    out = {
        "variant": variant_name, "fold": fold_idx, "seed": seed + fold_idx,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "best_val_acc": float(best_val),
        "test": {
            "accuracy": float(test_acc),
            "balanced_accuracy": float(bal),
            "macro": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
            "weighted": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
            "kappa": float(kappa), "mcc": float(mcc),
            "ece": float(ece), "brier": float(brier),
            "auroc_macro": float(auroc) if not np.isnan(auroc) else None,
            "auprc_macro": float(auprc) if not np.isnan(auprc) else None,
        },
    }
    # Free GPU memory before next fold
    del model, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def aggregate_kfold(fold_results, k):
    """Aggregate per-fold results: mean +/- std across folds for each metric."""
    if not fold_results:
        return None
    metric_keys = ["accuracy", "balanced_accuracy", "kappa", "mcc", "ece", "brier"]
    macro_keys = ["precision", "recall", "f1"]
    agg = {"n_folds": len(fold_results), "per_fold": fold_results}
    summary = {}
    for mk in metric_keys:
        vals = [r["test"][mk] for r in fold_results if r["test"][mk] is not None]
        summary[mk] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    for mk in macro_keys:
        vals = [r["test"]["macro"][mk] for r in fold_results
                if r["test"]["macro"][mk] is not None]
        summary[f"macro_{mk}"] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    vals = [r["best_val_acc"] for r in fold_results]
    summary["best_val_acc"] = {
        "mean": float(np.mean(vals)) if vals else None,
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "values": vals,
    }
    agg["summary"] = summary
    return agg


def main():
    parser = argparse.ArgumentParser(description="K-fold CV for H-CoAtNet ablation")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset with train/valid/test/<class>/* (Roboflow) "
                             "or flat <class>/* structure.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variant names or 'all' (default)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per fold")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--ls", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", type=str, default="results/kfold",
                        help="Output directory (relative to repo root)")
    parser.add_argument("--variants_quick", type=str, default=None,
                        help="Optional comma-separated subset of variants for quick runs")
    args = parser.parse_args()

    out_dir = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"K-FOLD CV (k={args.k}) -- A* R2-4 reproducibility")
    print("=" * 60)
    print(f"  dataset_dir = {args.dataset_dir}")
    print(f"  variants    = {args.variants}")
    print(f"  epochs      = {args.epochs}")
    print(f"  seed        = {args.seed}")
    print(f"  pretrained  = {not args.no_pretrained}")
    print(f"  out_dir     = {out_dir}")
    print()

    print("[1/4] gathering image pool...")
    files, labels, class_names = gather_pool(args.dataset_dir)
    print(f"      pool size = {len(files)} images, {len(class_names)} classes")
    print(f"      class counts: {Counter(labels)}")
    print()

    print("[2/4] building stratified folds...")
    folds = make_folds(labels, args.k, args.seed)
    for i, (tr, te) in enumerate(folds):
        print(f"      fold {i}: train={len(tr)} test={len(te)}")

    if args.variants == "all":
        variants = list(VARIANTS.keys())
    elif args.variants_quick:
        variants = args.variants_quick.split(",")
    else:
        variants = args.variants.split(",")

    print()
    print(f"[3/4] running {len(variants)} variant(s) x {args.k} folds = "
          f"{len(variants)*args.k} trainings...")

    all_results = {}
    overall_start = time.time()
    for v in variants:
        if v not in VARIANTS:
            print(f"  WARNING: unknown variant {v}, skipping")
            continue
        v_results = []
        v_start = time.time()
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            t0 = time.time()
            print(f"  [{v} fold {fold_idx+1}/{args.k}] training "
                  f"(train={len(train_idx)} test={len(test_idx)})...")
            try:
                fr = run_one_fold(v, fold_idx, files, labels, train_idx, test_idx,
                                  class_names, args.epochs, args.seed,
                                  args.lr, args.wd, args.ls, args.batch_size,
                                  not args.no_pretrained, out_dir)
                v_results.append(fr)
                elapsed = (time.time() - t0) / 60
                acc = fr["test"]["accuracy"]
                print(f"  [{v} fold {fold_idx+1}/{args.k}] DONE in {elapsed:.1f} min, "
                      f"test acc = {acc:.4f}")
            except Exception as e:
                print(f"  [{v} fold {fold_idx+1}/{args.k}] FAILED: {e}")
                traceback.print_exc()
                break
        if v_results:
            agg = aggregate_kfold(v_results, args.k)
            all_results[v] = agg
            mean_acc = agg["summary"]["accuracy"]["mean"]
            std_acc = agg["summary"]["accuracy"]["std"]
            v_elapsed = (time.time() - v_start) / 60
            print(f"  [{v}] all folds done in {v_elapsed:.1f} min, "
                  f"mean acc = {mean_acc:.4f} +/- {std_acc:.4f}")
        # Save intermediate after each variant (in case of crash)
        with open(out_dir / "kfold_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    overall_min = (time.time() - overall_start) / 60
    print()
    print(f"[4/4] done. Total time: {overall_min:.1f} min")
    print(f"      output: {out_dir / 'kfold_results.json'}")

    # Write a summary CSV
    csv_path = out_dir / "kfold_summary.csv"
    with open(csv_path, "w") as f:
        f.write("variant,n_folds,acc_mean,acc_std,balacc_mean,balacc_std,"
                "macro_f1_mean,macro_f1_std,kappa_mean,kappa_std,mcc_mean,mcc_std\n")
        for v, agg in all_results.items():
            s = agg["summary"]
            row = [
                v, agg["n_folds"],
                f"{s['accuracy']['mean']:.4f}", f"{s['accuracy']['std']:.4f}",
                f"{s['balanced_accuracy']['mean']:.4f}", f"{s['balanced_accuracy']['std']:.4f}",
                f"{s['macro_f1']['mean']:.4f}", f"{s['macro_f1']['std']:.4f}",
                f"{s['kappa']['mean']:.4f}", f"{s['kappa']['std']:.4f}",
                f"{s['mcc']['mean']:.4f}", f"{s['mcc']['std']:.4f}",
            ]
            f.write(",".join(row) + "\n")
    print(f"      summary CSV: {csv_path}")

    # Write a human-readable markdown Table 6
    md_path = out_dir / "kfold_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Table 6. K-Fold ({args.k}) Cross-Validation Results\n\n")
        f.write(f"Frozen base seed: {args.seed}. Per-fold seed = base + fold_idx.\n")
        f.write(f"Pool: {len(files)} images, {len(class_names)} classes.\n")
        f.write(f"Identical training protocol as Table 5 (IN1K ConvNeXt-Tiny, "
                f"AdamW 5e-5, Cosine T=30, CE+LS0.1+classweight).\n\n")
        f.write("| Variant | Acc% (mean +/- std) | BalAcc% | MacroF1% | Kappa | MCC |\n")
        f.write("|---|---|---|---|---|---|\n")
        for v, agg in all_results.items():
            s = agg["summary"]
            f.write(f"| {VARIANTS[v]['name']} "
                    f"| {100*s['accuracy']['mean']:.2f} +/- {100*s['accuracy']['std']:.2f} "
                    f"| {100*s['balanced_accuracy']['mean']:.2f} +/- {100*s['balanced_accuracy']['std']:.2f} "
                    f"| {100*s['macro_f1']['mean']:.2f} +/- {100*s['macro_f1']['std']:.2f} "
                    f"| {s['kappa']['mean']:.4f} +/- {s['kappa']['std']:.4f} "
                    f"| {s['mcc']['mean']:.4f} +/- {s['mcc']['std']:.4f} |\n")
    print(f"      summary MD:  {md_path}")


if __name__ == "__main__":
    main()
