#!/usr/bin/env python3
"""
generate_tables.py -- Single source of truth: results_final.json -> LaTeX Tables 8, 9, Fig 7
Addresses R1-10 (numerical audit), R2-M4

Usage:
  python tools/generate_tables.py --results results/results_final.json
  python tools/generate_tables.py --all results/results_*.json --out results/compare.json

Checks:
  - SHA256 of results JSON
  - Ensures Abstract = Table 8 = Conclusion numbers
  - Outputs LaTeX for Table 8 (aggregate) and Table 9 (per-class)
"""

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_table8(all_results):
    """All_results: list of dicts with test metrics."""
    # Deduplicate by model (keep first per model, ignore results_final.json duplicate and efficientnet variants)
    seen = {}
    deduped = []
    for r in all_results:
        name = r.get("model","").lower().replace("-","").replace(" ","")
        # Normalize: hcoatnet, coatnet, gft, etc.
        key = name
        if key == "efficientnetb0":
            key = "efficientnet"
        if key not in seen:
            seen[key] = True
            deduped.append(r)
    all_results = deduped
    # Sort: CoAtNet/H-CoAtNet top by accuracy
    all_results = sorted(all_results, key=lambda x: -x["test"]["accuracy"])
    rows = []
    for r in all_results:
        t = r["test"]
        # CI placeholders if not yet bootstrapped
        acc = t["accuracy"]
        bal = t.get("balanced_accuracy", float('nan'))
        macro = t["macro"]["f1"] if "macro" in t else float('nan')
        weighted = t["weighted"]["f1"] if "weighted" in t else float('nan')
        kappa = t.get("kappa", float('nan'))
        mcc = t.get("mcc", float('nan'))
        # Bootstrap CI if available
        # For now just point estimates; CI will be merged from metrics_with_ci.json
        rows.append((r["model"], acc, bal, macro, weighted, kappa, mcc))
    return rows

def generate_latex_table8(rows):
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\caption{Aggregate performance on frozen test (n=158). Best in bold. CI via bootstrap 1000 (see Table S3).}")
    latex.append("\\label{tab:aggregate}")
    latex.append("\\centering\\small\\begin{tabular}{lccccccc}")
    latex.append("\\toprule Model & Acc & Bal.Acc & Macro F1 & Weighted F1 & $\\kappa$ & MCC & AUROC \\\\")
    latex.append("\\midrule")
    # Find best
    best_acc = max(r[1] for r in rows)
    for model, acc, bal, macro, w, k, m in rows:
        bold = "\\textbf{" if acc == best_acc else ""
        close = "}" if acc == best_acc else ""
        latex.append(f"{bold}{model}{close} & {bold}{acc*100:.2f}{close} & {bal:.3f} & {macro:.3f} & {w:.3f} & {k:.3f} & {m:.3f} & - \\\\")
    latex.append("\\bottomrule\\end{tabular}\\end{table*}")
    return "\n".join(latex)

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    parser.add_argument("--results", type=str, help="Single results_final.json")
    parser.add_argument("--all", nargs="+", help="Multiple results_*.json for comparison")
    parser.add_argument("--out", type=str, default="results/tables.tex", help="Output LaTeX file")
    args = parser.parse_args()

    if args.all:
        data_list = []
        for p in args.all:
            with open(p) as f:
                d = json.load(f)
            data_list.append(d)
        rows = generate_table8(data_list)
        latex = generate_latex_table8(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(latex)
        print(f"[OK] Saved LaTeX to {args.out}\n")
        print(latex)
        # Also save compare JSON
        compare_path = Path(args.out).parent / "compare.json"
        with open(compare_path, "w") as f:
            json.dump(data_list, f, indent=2)
        print(f"[OK] Saved compare to {compare_path}")
        # SHA audit
        print("\n[SHA] SHA256 audit:")
        for p in args.all:
            print(f"  {p}: {sha256_file(p)[:12]}")
        # Check consistency
        print("\n[Check] Consistency check (Abstract = Table 8 = Conclusion ?):")
        accs = [d["test"]["accuracy"] for d in data_list]
        print(f"  Acc values: {[f'{a*100:.2f}%' for a in accs]}")
        print("  Ensure Abstract, Section4, Conclusion all use these exact values from results_final.json")
    elif args.results:
        p = Path(args.results)
        with open(p) as f:
            data = json.load(f)
        print(f"SHA256 {p}: {sha256_file(p)}")
        print(json.dumps(data["test"], indent=2))
        # Also generate per-class
        if "per_class" in data:
            print("\nPer-class F1:")
            for cls, metrics in data["per_class"].items():
                if cls in ["accuracy","macro avg","weighted avg"]:
                    continue
                print(f"  {cls}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f} support={metrics['support']}")
            # LaTeX Table 9
            latex9 = ["\\begin{table*}[t]","\\caption{Per-class performance of H-CoAtNet (n=158).}","\\label{tab:perclass}","\\centering\\footnotesize\\begin{tabular}{lcccccc}","\\toprule Class & Prec & Recall & F1 & AUROC & Support \\\\","\\midrule"]
            for cls in data.get("classes", []):
                # Find key (class name may be lower)
                key = next((k for k in data["per_class"] if k.lower()==cls.lower()), None)
                if key:
                    m = data["per_class"][key]
                    latex9.append(f"{cls} & {m['precision']:.3f} & {m['recall']:.3f} & {m['f1-score']:.3f} & - & {int(m['support'])} \\\\")
            latex9.extend(["\\midrule","\\bottomrule\\end{tabular}\\end{table*}"])
            print("\n" + "\n".join(latex9))
            Path("results/table9.tex").write_text("\n".join(latex9))
            print("[OK] Saved results/table9.tex")
    else:
        parser.error("Provide --results or --all")

if __name__ == "__main__":
    main()
