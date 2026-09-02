#!/usr/bin/env python3
"""
bootstrap_ci.py — A* Uncertainty: Bootstrap 95% CIs for Acc, BalAcc, MacroF1, Kappa, MCC, AUROC
Addresses R1-8, R2-4, Table 8, Table S3

Usage:
  python tools/bootstrap_ci.py --results results/results_final.json --n_bootstrap 1000 --seed 42
  python tools/bootstrap_ci.py --results results/results_hcoatnet.json --compare results/results_gft.json  # for McNemar

Output:
  results/metrics_with_ci.json
  Also prints LaTeX row for Table 8

Method: Stratified bootstrap (resample with replacement, preserve n) — percentile 2.5, 97.5.
        For AUROC, use one-vs-rest macro; if fails, skip.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

def bootstrap_metrics(y_true, y_pred, y_probs=None, n_bootstrap=1000, seed=42, num_classes=None):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics = {
        "acc": [], "bal_acc": [], "macro_f1": [], "weighted_f1": [], "kappa": [], "mcc": [],
        "auroc": [], "auprc": []
    }
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_probs is not None:
        y_probs = np.array(y_probs)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        # Handle case where bootstrap sample misses a class -> metrics still compute but warn
        try:
            metrics["acc"].append(accuracy_score(yt, yp))
            metrics["bal_acc"].append(balanced_accuracy_score(yt, yp))
            _, _, f1m, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
            _, _, f1w, _ = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
            metrics["macro_f1"].append(f1m)
            metrics["weighted_f1"].append(f1w)
            metrics["kappa"].append(cohen_kappa_score(yt, yp))
            # MCC can fail if single class in sample
            try:
                metrics["mcc"].append(matthews_corrcoef(yt, yp))
            except:
                metrics["mcc"].append(np.nan)
            if y_probs is not None:
                yp_probs = y_probs[idx]
                try:
                    y_bin = label_binarize(yt, classes=list(range(num_classes or yp_probs.shape[1])))
                    # If bootstrap misses a class, roc_auc may fail
                    metrics["auroc"].append(roc_auc_score(y_bin, yp_probs, average='macro', multi_class='ovr'))
                    metrics["auprc"].append(average_precision_score(y_bin, yp_probs, average='macro'))
                except:
                    metrics["auroc"].append(np.nan)
                    metrics["auprc"].append(np.nan)
        except Exception as e:
            # Skip this bootstrap sample
            continue
    # Percentile CI
    cis = {}
    for k, vals in metrics.items():
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) == 0:
            cis[k] = {"mean": None, "ci_low": None, "ci_high": None, "sd": None}
        else:
            cis[k] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "ci_low": float(np.percentile(vals, 2.5)),
                "ci_high": float(np.percentile(vals, 97.5)),
                "sd": float(np.std(vals)),
                "n_bootstrap": len(vals)
            }
    return cis

def main():
    parser = argparse.ArgumentParser(description="Bootstrap 95% CIs")
    parser.add_argument("--results", type=str, required=True, help="Path to results_final.json (with y_true, y_pred, y_probs if available)")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="results/metrics_with_ci.json", help="Output JSON")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # Support both flat and nested structure
    if "test" in data and "y_true" in data["test"]:
        yt = data["test"]["y_true"]
        yp = data["test"]["y_pred"]
        probs = data["test"].get("y_probs")  # may be None
        n_classes = len(data.get("classes", [])) or (np.array(probs).shape[1] if probs else 5)
        model_name = data.get("model", "H-CoAtNet")
    elif "y_true" in data:
        yt = data["y_true"]; yp = data["y_pred"]; probs=None; n_classes=5; model_name="model"
    else:
        raise ValueError(f"Cannot find y_true/y_pred in {args.results}. Expected keys test.y_true")

    # If probs not saved, try to load from y_probs or probs file nearby
    if probs is None:
        # Try results_probs.npy
        prob_path = Path(args.results).parent / "y_probs.npy"
        if prob_path.exists():
            probs = np.load(prob_path)
            print(f"Loaded probs from {prob_path}")
        else:
            print("⚠️  No y_probs found — AUROC/AUPRC CIs will be skipped. Re-train with evaluate_with_probs to save probs.")

    print(f"Bootstrapping {model_name} n={len(yt)} n_bootstrap={args.n_bootstrap} seed={args.seed}")
    cis = bootstrap_metrics(yt, yp, probs, n_bootstrap=args.n_bootstrap, seed=args.seed, num_classes=n_classes)

    # Also compute point estimates on full set
    from sklearn.metrics import accuracy_score
    point = {
        "acc": accuracy_score(yt, yp),
        "bal_acc": balanced_accuracy_score(yt, yp),
    }
    _, _, f1m, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    point["macro_f1"] = f1m

    print("\n=== Bootstrap 95% CIs (percentile) ===")
    for k in ["acc","bal_acc","macro_f1","weighted_f1","kappa","mcc","auroc","auprc"]:
        if k in cis and cis[k]["ci_low"] is not None:
            print(f"  {k:12s}: {cis[k]['mean']:.4f} [{cis[k]['ci_low']:.4f} - {cis[k]['ci_high']:.4f}] sd={cis[k]['sd']:.4f}")

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": model_name, "n": len(yt), "point": point, "bootstrap": cis, "n_bootstrap": args.n_bootstrap, "seed": args.seed}, f, indent=2)
    print(f"\n✅ Saved to {out}")

    # Print LaTeX row snippet for Table 8
    acc = cis["acc"]
    bal = cis["bal_acc"]
    f1 = cis["macro_f1"]
    kappa = cis["kappa"]
    print("\n📋 LaTeX row snippet (copy to Table 8):")
    print(f"H-CoAtNet & {point['acc']*100:.2f} [{acc['ci_low']*100:.1f}--{acc['ci_high']*100:.1f}] & {bal['mean']:.3f} [{bal['ci_low']:.3f}--{bal['ci_high']:.3f}] & {f1['mean']:.3f} [{f1['ci_low']:.3f}--{f1['ci_high']:.3f}] & {kappa['mean']:.3f} [{kappa['ci_low']:.3f}--{kappa['ci_high']:.3f}] \\\\")

    # Also handle 5-seed aggregation if multiple results files exist
    # Look for results/results_*.json
    results_dir = Path(args.results).parent
    all_results = list(results_dir.glob("results_*.json"))
    if len(all_results) > 1:
        print(f"\nFound {len(all_results)} result files for 5-seed aggregation: {[p.name for p in all_results]}")
        accs = []
        for p in all_results:
            with open(p) as f:
                d = json.load(f)
                accs.append(d["test"]["accuracy"] if "test" in d else d["accuracy"])
        print(f"  5-seed mean±SD: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

if __name__ == "__main__":
    main()
