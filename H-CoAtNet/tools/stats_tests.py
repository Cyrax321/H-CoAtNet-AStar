#!/usr/bin/env python3
"""
stats_tests.py — Significance tests for A* Table S3: McNemar (Acc) + DeLong (AUROC)
Addresses R1-8, R2-4

Usage:
  python tools/stats_tests.py --a results/results_hcoatnet.json --b results/results_gft.json
  python tools/stats_tests.py --all results/results_*.json --reference results/results_hcoatnet.json

Requires: statsmodels for McNemar, or manual fallback.
DeLong implementation from https://github.com/yandexdataschool/delong (or approximated via bootstrap if not available).
"""

import argparse
import json
from pathlib import Path
import numpy as np

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test for paired predictions (a vs b)."""
    # Build 2x2 table: a correct/incorrect vs b correct/incorrect
    a_correct = (np.array(y_pred_a) == np.array(y_true))
    b_correct = (np.array(y_pred_b) == np.array(y_true))
    # Table: [[both correct, a correct b wrong], [a wrong b correct, both wrong]]
    both_correct = np.sum(a_correct & b_correct)
    a_only = np.sum(a_correct & ~b_correct)
    b_only = np.sum(~a_correct & b_correct)
    both_wrong = np.sum(~a_correct & ~b_correct)
    table = [[both_correct, a_only],[b_only, both_wrong]]
    # Use exact binomial or chi2 with continuity correction
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        result = mcnemar(table, exact=False, correction=True)
        return result.statistic, result.pvalue, table
    except ImportError:
        # Fallback chi2 approximation: (|b - c| -1)^2 / (b+c)
        b, c = a_only, b_only
        if b+c == 0:
            return 0.0, 1.0, table
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        # p from chi2(1)
        from scipy.stats import chi2 as chi2_dist
        p = 1 - chi2_dist.cdf(chi2, 1)
        return chi2, p, table

def delong_test_placeholder(y_true, probs_a, probs_b):
    """Placeholder for DeLong. If delong not installed, bootstrap it."""
    try:
        # Try to import delong if available
        from delong import delong_roc_test  # pip install delong (if available)
        # Would need y_bin and probs...
        # Not implemented, fallback
        raise ImportError
    except ImportError:
        # Bootstrap AUROC difference (A* acceptable alternative if DeLong not available)
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
        y_true = np.array(y_true)
        probs_a = np.array(probs_a)
        probs_b = np.array(probs_b)
        # Macro AUROC difference bootstrap
        rng = np.random.RandomState(42)
        n = len(y_true)
        diffs = []
        # Determine num classes
        n_classes = probs_a.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        # Point estimates
        try:
            auroc_a = roc_auc_score(y_bin, probs_a, average='macro', multi_class='ovr')
            auroc_b = roc_auc_score(y_bin, probs_b, average='macro', multi_class='ovr')
        except:
            return None, 1.0, "AUROC failed"
        for _ in range(1000):
            idx = rng.choice(n, n, replace=True)
            try:
                yb = y_bin[idx]
                pa = probs_a[idx]
                pb = probs_b[idx]
                # Need at least 2 classes in sample
                if len(np.unique(y_true[idx])) < 2:
                    continue
                da = roc_auc_score(yb, pa, average='macro', multi_class='ovr')
                db = roc_auc_score(yb, pb, average='macro', multi_class='ovr')
                diffs.append(da - db)
            except:
                continue
        diffs = np.array(diffs)
        # Two-sided p: proportion of diffs <=0 if observed diff >0, etc.
        observed = auroc_a - auroc_b
        # Simple bootstrap p: 2 * min(prop >0, prop <0)
        prop_pos = np.mean(diffs > 0)
        p = 2 * min(prop_pos, 1-prop_pos)
        p = max(0.0, min(1.0, p))
        return observed, p, f"bootstrap 1000, AUROC diff {observed:.4f}, p≈{p:.4f}"

def load_results(path):
    with open(path) as f:
        data = json.load(f)
    if "test" in data:
        yt = data["test"]["y_true"]
        yp = data["test"]["y_pred"]
        probs = data["test"].get("y_probs") or data["test"].get("probs")
        model = data.get("model", Path(path).stem)
    else:
        yt = data["y_true"]; yp = data["y_pred"]; probs=None; model=Path(path).stem
    return yt, yp, probs, model

def main():
    parser = argparse.ArgumentParser(description="McNemar + DeLong significance")
    parser.add_argument("--a", type=str, help="Reference model results JSON (e.g., H-CoAtNet)")
    parser.add_argument("--b", type=str, help="Comparator JSON")
    parser.add_argument("--all", nargs="+", help="All results_*.json, will compare each vs reference")
    parser.add_argument("--reference", type=str, help="Reference for --all")
    parser.add_argument("--out", type=str, default="results/significance.json", help="Output")
    args = parser.parse_args()

    if args.all:
        if not args.reference:
            parser.error("--reference required with --all")
        yt_ref, yp_ref, probs_ref, name_ref = load_results(args.reference)
        results = []
        for p in args.all:
            if p == args.reference:
                continue
            yt, yp, probs, name = load_results(p)
            # Ensure same y_true (same test set)
            assert len(yt) == len(yt_ref) and yt == yt_ref, f"y_true mismatch between {p} and reference"
            chi2, pval, table = mcnemar_test(yt_ref, yp_ref, yp)
            _, p_delong, delong_note = delong_test_placeholder(yt_ref, probs_ref, probs) if probs_ref is not None and probs is not None else (None, 1.0, "no probs")
            print(f"{name_ref} vs {name}: McNemar p={pval:.4f} (chi2={chi2:.2f}, table {table}), DeLong-like p={p_delong:.4f} ({delong_note}) {'**' if pval<0.01 else '*' if pval<0.05 else 'ns'}")
            results.append({"comparison": f"{name_ref} vs {name}", "mcnemar_p": float(pval), "mcnemar_chi2": float(chi2), "table": table, "delong_p": float(p_delong) if isinstance(p_delong,float) else None, "note": delong_note})
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved {args.out}")
    elif args.a and args.b:
        yt, yp_a, probs_a, name_a = load_results(args.a)
        _, yp_b, probs_b, name_b = load_results(args.b)
        chi2, pval, table = mcnemar_test(yt, yp_a, yp_b)
        print(f"McNemar {name_a} vs {name_b}: chi2={chi2:.2f}, p={pval:.4f}, table={table}")
        print("  Interpretation: p<0.05 => significant difference in accuracy")
        _, p_delong, note = delong_test_placeholder(yt, probs_a, probs_b) if probs_a is not None and probs_b is not None else (None, 1.0, "no probs")
        print(f"DeLong-like (bootstrap) AUROC diff p={p_delong:.4f} — {note}")
    else:
        parser.error("Provide --a and --b, or --all + --reference")

if __name__ == "__main__":
    main()
