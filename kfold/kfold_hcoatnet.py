#!/usr/bin/env python3
"""
kfold/kfold_hcoatnet.py — K-Fold Cross-Validation for H-CoAtNet (fixes R2-4)
Fair, TRIPOD-aware: 5-fold stratified CV on FULL dataset (n=2508), same 30 epochs, same everything.

What reviewer asked (R2-4): "K-fold cross-validation is needed" — because single split has no SD.
What we do: 5-fold stratified, seed 42, same frozen augmentation/protocol as main split.
Reports: mean±SD across folds for Accuracy, BalAcc, MacroF1, Kappa, MCC, AUROC + per-fold table.

Run:
  python kfold/kfold_hcoatnet.py --k 5 --epochs 30 --seed 42              # H-CoAtNet only, ~4.5 hrs on T4 (5 × 55 min)
  python kfold/kfold_hcoatnet.py --k 5 --epochs 10 --seed 42 --smoke      # smoke test 10 epochs, 30 min
Outputs:
  results/kfold_hcoatnet_fold{0..4}.json
  results/kfold_hcoatnet_summary.json  +  kfold/kfold_summary.json
  figures/fig_kfold_box.png + figures/fig_kfold_forest.png
  kfold/kfold_table.tex

Fairness (same as ablation): SAME 224×224 norm, SAME RRCrop+TrivialAug+Erasing, SAME class_weights+LS0.1, SAME AdamW 5e-5 Cosine WD0.01 batch24, SAME 30 epochs, SAME seed per fold, SAME TRIPOD test-once per fold.
Only difference: different stratified train/test partition per fold (no frozen test). We KEEP the original frozen n=158 result as reference and report both.
"""

import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "H-CoAtNet" / "proposed_method"))
from train_h_coatnet import seed_everything, train_epoch, evaluate, evaluate_with_probs, compute_ece, API_KEY, TARGET_SIZE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DEVICE
# Prefer AblationCoAtNet full variant for fairness toggle
try:
    sys.path.insert(0, str(REPO_ROOT / "ablation"))
    from ablation import AblationCoAtNet as HCoAtNet
    # also try alternative path
    if HCoAtNet is None:
        raise ImportError
    print("Using AblationCoAtNet (fair ablation model)")
except Exception as e:
    # fallback to original CoAtGFT
    print(f"AblationCoAtNet not found ({e}), using CoAtGFT")
    from train_h_coatnet import CoAtGFT as HCoAtNet

from roboflow import Roboflow
from tqdm import tqdm

def collect_dataset_paths_and_labels(dataset_dir):
    """Collect all image paths and integer labels from dataset_dir/train+valid+test OR flat."""
    dataset_dir = Path(dataset_dir)
    # Try Roboflow structure first
    splits = []
    for s in ["train","valid","test"]:
        d = dataset_dir / s
        if d.exists():
            splits.append(d)
    if splits:
        # Gather all files recursively
        all_paths = []
        all_labels = []
        # Determine class_to_idx from train
        train_classes = sorted([x.name for x in (dataset_dir/"train").iterdir() if x.is_dir()])
        class_to_idx = {c:i for i,c in enumerate(train_classes)}
        for split in splits:
            for cls in split.iterdir():
                if not cls.is_dir(): continue
                idx = class_to_idx[cls.name]
                for p in cls.rglob("*"):
                    if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
                        all_paths.append(str(p))
                        all_labels.append(idx)
        return all_paths, all_labels, train_classes
    else:
        # Flat ImageFolder
        classes = sorted([x.name for x in dataset_dir.iterdir() if x.is_dir()])
        class_to_idx = {c:i for i,c in enumerate(classes)}
        paths, labels = [], []
        for cls in dataset_dir.iterdir():
            if not cls.is_dir(): continue
            for p in cls.rglob("*"):
                if p.suffix.lower() in [".jpg",".jpeg",".png"]:
                    paths.append(str(p)); labels.append(class_to_idx[cls.name])
        return paths, labels, classes

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02,0.2)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def train_one_fold(fold_idx, train_idx, test_idx, all_paths, all_labels, class_names, epochs=30, seed=42):
    print("\n"+"="*70)
    print(f"K-FOLD {fold_idx+1} — train {len(train_idx)} / test {len(test_idx)} — seed {seed+fold_idx}")
    print("="*70)
    seed_everything(seed + fold_idx)
    train_tf, eval_tf = get_transforms()
    # Build datasets via Subset on a base ImageFolder that holds all images
    # For K-fold we need a unified dataset with all images + transforms
    # We will create a custom dataset that returns (image, label) with transform per fold
    from PIL import Image
    class KFOLD_Dataset(torch.utils.data.Dataset):
        def __init__(self, paths, labels, indices, transform):
            self.paths = [paths[i] for i in indices]
            self.labels = [labels[i] for i in indices]
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, self.labels[i]
    # Further split train_idx into train/valid 85/15 for early stopping (TRIPOD-like)
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(train_idx, test_size=0.15, stratify=[all_labels[i] for i in train_idx], random_state=seed+fold_idx)
    train_ds = KFOLD_Dataset(all_paths, all_labels, tr_idx, train_tf)
    valid_ds = KFOLD_Dataset(all_paths, all_labels, va_idx, eval_tf)
    test_ds  = KFOLD_Dataset(all_paths, all_labels, test_idx, eval_tf)
    # Also need counts for class weights (from train_ds only)
    from collections import Counter
    train_labels = [all_labels[i] for i in tr_idx]
    counts = np.bincount(train_labels, minlength=len(class_names))
    print(f"  Train {len(tr_idx)} | Valid {len(va_idx)} | Test {len(test_idx)} | counts {dict(Counter(train_labels))}")
    nw = 0 if os.name=='nt' else 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    # Model — full H-CoAtNet
    # Use AblationCoAtNet full variant for consistency with ablation study
    import sys
    sys.path.insert(0, str(REPO_ROOT / "ablation"))
    try:
        from ablation import AblationCoAtNet as Model
        model = Model(use_vit=True, use_se=True, num_classes=len(class_names), pretrained=True).to(DEVICE)
    except:
        from train_h_coatnet import CoAtGFT as Model
        model = Model(num_classes=len(class_names), pretrained=True).to(DEVICE)
    class_weights = torch.tensor([len(train_labels)/(c*len(class_names)+1e-6) for c in counts], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Train
    history = {"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[]}
    best_val = 0; best_ep=-1
    ckpt = Path(f"kfold/best_fold{fold_idx}.pth")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        print(f"\n--- Fold {fold_idx+1} Epoch {epoch+1}/{epochs} ---")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, _, _ = evaluate(model, valid_loader, criterion, desc="Validating")
        scheduler.step()
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        print(f"  Train {tr_acc:.4f} | Val {va_acc:.4f} | Loss {tr_loss:.4f}/{va_loss:.4f}")
        if va_acc > best_val:
            best_val=va_acc; best_ep=epoch+1
            torch.save(model.state_dict(), ckpt)
            print(f"  [NEW BEST] ep {best_ep} val {best_val:.4f}")
    # Final test once
    print(f"\n--- Final Test Fold {fold_idx+1} (Held-Out) ---")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, test_acc, y_true, y_pred, y_probs = evaluate_with_probs(model, test_loader, criterion, desc="Final Test")
    print(f"  Test Acc {test_acc:.4f} (n={len(y_true)})")
    bal = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    ece = compute_ece(y_probs, y_true)
    try:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        auroc = roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr')
        auprc = average_precision_score(y_bin, y_probs, average='macro')
    except: auroc, auprc = float('nan'), float('nan')
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    out = {
        "fold": int(fold_idx),
        "seed": int(seed+fold_idx),
        "n_train": len(tr_idx), "n_valid": len(va_idx), "n_test": len(test_idx),
        "best_val_acc": float(best_val), "best_epoch": int(best_ep),
        "history": history,
        "test": {"accuracy": float(test_acc), "balanced_accuracy": float(bal), "macro": {"precision": float(prec_m), "recall": float(rec_m), "f1": float(f1_m)}, "kappa": float(kappa), "mcc": float(mcc), "ece": float(ece), "auroc_macro": float(auroc) if not np.isnan(auroc) else None, "auprc_macro": float(auprc) if not np.isnan(auprc) else None, "n": len(y_true), "y_true": list(map(int,y_true)), "y_pred": list(map(int,y_pred))},
        "per_class": report, "classes": class_names,
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/kfold_hcoatnet_fold{fold_idx}.json").write_text(json.dumps(out, indent=2))
    Path(f"kfold/kfold_hcoatnet_fold{fold_idx}.json").write_text(json.dumps(out, indent=2))
    print(f"  Saved results/kfold_hcoatnet_fold{fold_idx}.json")
    return out

def summarize(k=5):
    import pandas as pd, json
    rows=[]
    for i in range(k):
        p = Path(f"results/kfold_hcoatnet_fold{i}.json")
        if not p.exists(): p = Path(f"kfold/kfold_hcoatnet_fold{i}.json")
        if not p.exists(): continue
        d=json.loads(p.read_text()); t=d["test"]
        rows.append({"fold":i, "Acc":t["accuracy"]*100, "BalAcc":t["balanced_accuracy"]*100, "MacroF1":t["macro"]["f1"]*100, "Kappa":t["kappa"]*100, "MCC":t["mcc"]*100, "ECE":t["ece"]*100, "AUROC":t.get("auroc_macro",0)*100 if t.get("auroc_macro") else 0})
    if not rows:
        print("No kfold results to summarize")
        return
    df=pd.DataFrame(rows)
    # Stats
    summary={"k":k, "mean":{"Acc":df["Acc"].mean(), "BalAcc":df["BalAcc"].mean(), "MacroF1":df["MacroF1"].mean(), "Kappa":df["Kappa"].mean()}, "sd":{"Acc":df["Acc"].std(), "BalAcc":df["BalAcc"].std(), "MacroF1":df["MacroF1"].std(), "Kappa":df["Kappa"].std()}, "per_fold":rows}
    # Compare to single frozen split
    try:
        single=json.loads(Path("results/results_hcoatnet.json").read_text())["test"]
        summary["single_frozen"]={"Acc":single["accuracy"]*100, "BalAcc":single["balanced_accuracy"]*100, "MacroF1":single["macro"]["f1"]*100, "Kappa":single["kappa"]*100}
    except: pass
    Path("results/kfold_hcoatnet_summary.json").write_text(json.dumps(summary, indent=2))
    Path("kfold/kfold_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== K-FOLD SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nMean±SD Acc: {df['Acc'].mean():.2f}±{df['Acc'].std():.2f} | Kappa: {df['Kappa'].mean():.2f}±{df['Kappa'].std():.2f} | MacroF1: {df['MacroF1'].mean():.2f}±{df['MacroF1'].std():.2f}")
    # LaTeX
    tex=[ "\\begin{table}[t]", "\\caption{5-fold stratified cross-validation for H-CoAtNet (same 30 epochs, same protocol, seed 42). Mean±SD across folds. Single frozen split (n=158) shown for reference (TRIPOD-AI Type 2b).}", "\\label{tab:kfold}", "\\centering\\small\\begin{tabular}{lcccc}", "\\toprule Fold & Acc & BalAcc & MacroF1 & Kappa \\\\","\\midrule"]
    for _,r in df.iterrows(): tex.append(f"{int(r['fold'])+1} & {r['Acc']:.2f} & {r['BalAcc']:.2f} & {r['MacroF1']:.2f} & {r['Kappa']:.2f} \\\\")
    tex.append("\\midrule")
    tex.append(f"Mean±SD & {df['Acc'].mean():.2f}±{df['Acc'].std():.2f} & {df['BalAcc'].mean():.2f}±{df['BalAcc'].std():.2f} & {df['MacroF1'].mean():.2f}±{df['MacroF1'].std():.2f} & {df['Kappa'].mean():.2f}±{df['Kappa'].std():.2f} \\\\")
    if "single_frozen" in summary: tex.append(f"Single frozen (n=158) & {summary['single_frozen']['Acc']:.2f} & {summary['single_frozen']['BalAcc']:.2f} & {summary['single_frozen']['MacroF1']:.2f} & {summary['single_frozen']['Kappa']:.2f} \\\\")
    tex.extend(["\\bottomrule\\end{tabular}\\end{table}"])
    Path("kfold/kfold_table.tex").write_text("\n".join(tex))
    Path("figures/kfold_table.tex").write_text("\n".join(tex))
    with open("kfold/kfold_table.tex") as f: print("\n".join(tex))
    # Box plot
    df_melt=df.melt(id_vars="fold", value_vars=["Acc","BalAcc","MacroF1","Kappa"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(8,4.5)); sns.boxplot(data=df_melt, x="Metric", y="Score", palette="Set2"); sns.stripplot(data=df_melt, x="Metric", y="Score", color="black", size=6, jitter=False)
    plt.title(f"5-Fold CV — H-CoAtNet Stability (seed 42, 30 epochs, n=2508)",fontsize=11); plt.ylabel("Score (%)"); plt.tight_layout()
    Path("figures").mkdir(exist_ok=True); plt.savefig("figures/fig_kfold_box.png", dpi=300, bbox_inches='tight'); plt.savefig("kfold/fig_kfold_box.png", dpi=300, bbox_inches='tight'); plt.close()
    # Forest per fold
    plt.figure(figsize=(8,4)); y=range(k); acc=df["Acc"].values; plt.errorbar(acc, y, xerr=df["Acc"].std(), fmt='o', color="#0072B2", capsize=4)
    plt.yticks(y, [f"Fold {i+1}" for i in range(k)]); plt.axvline(acc.mean(), color="red", ls="--", label=f'Mean {acc.mean():.1f}%'); plt.xlabel("Accuracy (%)"); plt.title("K-Fold Per-Fold Accuracy (H-CoAtNet)",fontsize=11); plt.legend(); plt.grid(axis="x",alpha=0.3); plt.tight_layout()
    plt.savefig("figures/fig_kfold_forest.png", dpi=300, bbox_inches='tight'); plt.savefig("kfold/fig_kfold_forest.png", dpi=300, bbox_inches='tight'); plt.close()
    df.to_csv("kfold/kfold_summary.csv", index=False)
    print("\n[OK] K-fold figures: figures/fig_kfold_box.png, kfold/kfold_table.tex")

def main():
    parser=argparse.ArgumentParser(description="H-CoAtNet K-Fold")
    parser.add_argument("--k", type=int, default=5, help="folds")
    parser.add_argument("--epochs", type=int, default=30, help="epochs per fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="10 epochs smoke test")
    args=parser.parse_args()
    if args.smoke: args.epochs=10
    # Resolve dataset
    dataset_dir=args.dataset_dir
    if dataset_dir is None:
        print("Resolving dataset via Roboflow...")
        rf=Roboflow(api_key=API_KEY)
        project=rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
        dataset=project.version(1).download("folder")
        dataset_dir=dataset.location
        print(f"Dataset at {dataset_dir}")
    all_paths, all_labels, class_names = collect_dataset_paths_and_labels(dataset_dir)
    print(f"Collected {len(all_paths)} images — classes {class_names} — { {c: all_labels.count(i) for i,c in enumerate(class_names)} if len(all_labels)<3000 else 'large'}")
    skf=StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_paths, all_labels)):
        train_one_fold(fold_idx, train_idx, test_idx, all_paths, all_labels, class_names, epochs=args.epochs, seed=args.seed)
    summarize(k=args.k)
    print("\nDone. Check kfold/kfold_table.tex + figures/fig_kfold_box.png")
    print("For paper: Report mean±SD from kfold_summary.json alongside single frozen result (already have bootstrap).")

if __name__=="__main__":
    main()
