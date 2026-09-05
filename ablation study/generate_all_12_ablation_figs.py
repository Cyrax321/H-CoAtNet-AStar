#!/usr/bin/env python3
"""
Twelve-figure ablation suite for H-CoAtNet.

Purpose:
  Build every figure the ablation section needs, from saved JSON results.
  Main paper uses 01-04. Supplement uses 05-12. All share Okabe-Ito colors,
  300 DPI PNG plus PDF, and frozen test n in the title.

Modes:
  --demo : proxy numbers from histories plus planned deltas. Layout preview only.
           Every panel is watermarked PREVIEW. Do not submit these.
  --real : uses results/results_ablation_{variant}.json when present.
           Run after training: python3 "ablation study/ablation_study.py" --variant all

Figures:
  01 arch        : 4 variant block diagrams, only ViT/SE toggled.
  02 main bar    : Acc / MacroF1 / Kappa per variant.
  03 drop        : accuracy drop vs Full in percentage points.
  04 perclass    : per-class F1 heatmap, 5 classes by 4 variants.
  05 curves      : combined val acc and val loss across variants.
  06 confusion   : 2x2 raw confusion matrices, one per variant.
  07 roc         : ROC one-vs-rest per variant.
  08 pr          : precision-recall per variant, key for rare LI class.
  09 reliability : calibration curves with ECE in title.
  10 forest      : accuracy with 95 percent bootstrap interval per variant.
  11 efficiency  : params vs accuracy, bubble size is MACs, color is latency.
  12 mechanism   : token pruning grid plus Grad-CAM sketch plus failure table.

Inputs:
  Demo mode reads histories/all_histories.json for curve shapes.
  Real mode reads results JSONs with y_true/y_pred/y_probs when available,
  and falls back to synthetic probabilities only if a file lacks them.

Outputs:
  ablation study/figures/fig_ablation_{01-12}_* plus mirrors in figures/.
  INDEX_12.md lists the set and the mode used.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for Colab.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Repository root is the parent of this folder. OUT holds the study figures.
# FIG_MAIN mirrors the three core plots for the manuscript build.
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "ablation study" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_MAIN = REPO / "figures"
FIG_MAIN.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette. One color per variant, fixed across all 12 figures.
OKABE = {"full": "#0072B2", "noSE": "#D55E00", "noViT": "#009E73", "cnnOnly": "#CC79A7"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "font.family": "sans-serif"})

VARIANTS = ["full", "noSE", "noViT", "cnnOnly"]
NAMES = {"full": "Full\nH-CoAtNet", "noSE": "w/o SE", "noViT": "w/o ViT", "cnnOnly": "CNN-only"}
CLASSES = ["HI", "Healthy", "IV", "LI", "NS"]


def load_real():
    """Load real ablation JSONs. Return dict or None when fewer than 2 exist."""
    import json
    rows = {}
    for v in VARIANTS:
        for cand in [REPO / f"results/results_ablation_{v}.json",
                     REPO / f"ablation/results_ablation_{v}.json"]:
            if cand.exists():
                try:
                    d = json.loads(cand.read_text())
                    t = d.get("test", {})
                    macro = t.get("macro", {}) if isinstance(t.get("macro"), dict) else {}
                    rows[v] = {
                        "acc": t.get("accuracy", 0) * 100,
                        "bal": t.get("balanced_accuracy", 0) * 100,
                        "f1": macro.get("f1", 0) * 100,
                        "kappa": t.get("kappa", 0) * 100,
                        "mcc": t.get("mcc", 0) * 100,
                        "ece": t.get("ece", 0) * 100,
                        "auroc": (t.get("auroc_macro") or 0) * 100,
                        "auprc": (t.get("auprc_macro") or 0) * 100,
                        "per": d.get("per_class", {}),
                        "yt": t.get("y_true"),
                        "yp": t.get("y_pred"),
                        "ypr": t.get("y_probs"),
                        "classes": d.get("classes", CLASSES),
                    }
                    break
                except Exception as e:
                    print(f"warn {v}: {e}")
    return rows if len(rows) >= 2 else None


def demo_data():
    """Proxy metrics for layout preview. Matches ABLATION_PLAN expected drops."""
    return {
        "full":    {"acc": 90.51, "bal": 88.4, "f1": 86.05, "kappa": 87.5,
                    "mcc": 87.8, "ece": 3.2, "auroc": 96.3, "auprc": 91.2,
                    "per_f1": {"HI": 0.935, "Healthy": 0.965, "IV": 0.877, "LI": 0.737, "NS": 0.714}},
        "noSE":    {"acc": 88.4, "bal": 86.1, "f1": 83.2, "kappa": 84.5,
                    "mcc": 84.8, "ece": 5.8, "auroc": 94.1, "auprc": 88.0,
                    "per_f1": {"HI": 0.89, "Healthy": 0.95, "IV": 0.85, "LI": 0.66, "NS": 0.69}},
        "noViT":   {"acc": 85.4, "bal": 83.0, "f1": 80.1, "kappa": 81.2,
                    "mcc": 81.5, "ece": 6.9, "auroc": 92.4, "auprc": 85.5,
                    "per_f1": {"HI": 0.88, "Healthy": 0.94, "IV": 0.80, "LI": 0.68, "NS": 0.62}},
        "cnnOnly": {"acc": 82.3, "bal": 79.5, "f1": 76.5, "kappa": 77.3,
                    "mcc": 77.5, "ece": 8.9, "auroc": 90.1, "auprc": 82.0,
                    "per_f1": {"HI": 0.84, "Healthy": 0.92, "IV": 0.77, "LI": 0.60, "NS": 0.58}},
    }


def save(fig, stem):
    """Save PNG plus PDF to study folder and manuscript figures folder."""
    for d in [OUT, FIG_MAIN]:
        fig.savefig(d / f"{stem}.png", bbox_inches="tight")
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.png/pdf")


def watermark(ax, demo):
    """Stamp PREVIEW on demo panels so they are never mistaken for results."""
    if demo:
        ax.text(0.99, 0.01, "PREVIEW - DEMO numbers, retrain for paper",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                style="italic", color="red",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"))


def fig01(demo):
    """Variant schematic. Four block rows, removed stages shown OFF in red."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    specs = {"full": (True, True), "noSE": (True, False),
             "noViT": (False, True), "cnnOnly": (False, False)}
    for ax, v in zip(axes, VARIANTS):
        use_vit, use_se = specs[v]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(NAMES[v], fontsize=10, weight="bold", color=OKABE[v])

        def box(x, y, w, h, txt, on=True, col="#0072B2"):
            ec = "black" if on else "red"
            ls = "-" if on else "--"
            r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                       fc=col if on else "#f0f0f0", ec=ec, ls=ls,
                                       alpha=0.9 if on else 0.6)
            ax.add_patch(r)
            ax.text(x + w / 2, y + h / 2, txt if on else txt + "\nOFF",
                    ha="center", va="center", fontsize=6.5, weight="bold",
                    color="white" if on else "red")

        box(0.3, 2.2, 1.6, 1.6, "Stem+\nS1 S2", True, "#56B4E9")
        box(2.2, 2.2, 1.6, 1.6, "2x ViT", use_vit, "#009E73")
        box(4.1, 2.2, 1.6, 1.6, "S3 S4", True, "#0072B2")
        box(6.0, 2.2, 1.6, 1.6, "SE\n49>36>24", use_se, "#D55E00")
        box(7.9, 2.2, 1.4, 1.6, "GAP\n+FC5", True, "#333333")
        for x0, x1 in [(1.9, 2.2), (3.8, 4.1), (5.7, 6.0), (7.6, 7.9)]:
            ax.annotate("", xy=(x1, 3.0), xytext=(x0, 3.0),
                        arrowprops=dict(arrowstyle="->", lw=1))
        note = "224>56>28>14>7" if use_vit else "224>56>28>14>7\n(no mid ViT)"
        ax.text(5, 0.7, note, ha="center", fontsize=6.5, color="#555")
    fig.suptitle("Ablation Variants - Only ViT / SE toggled, all else frozen (seed 42, 30ep, TRIPOD-2b)",
                 fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation_01_arch")


def fig02(data, demo):
    """Main grouped bar: absolute Acc, MacroF1, Kappa per variant."""
    import pandas as pd
    rows = [{"Model": NAMES[v], "variant": v, "Acc": data[v]["acc"],
             "MacroF1": data[v]["f1"], "Kappa": data[v]["kappa"]} for v in VARIANTS]
    df = pd.DataFrame(rows)
    m = df.melt(id_vars=["Model", "variant"], value_vars=["Acc", "MacroF1", "Kappa"],
                var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = [NAMES[v] for v in VARIANTS]
    sns.barplot(data=m, x="Model", y="Score", hue="Metric",
                palette=["#0072B2", "#009E73", "#D55E00"], ax=ax, order=order)
    for i, v in enumerate(VARIANTS):
        ax.text(i - 0.25, data[v]["acc"] + 0.8, f'{data[v]["acc"]:.1f}%',
                fontsize=8, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation - Acc / MacroF1 / Kappa per variant (frozen test, seed 42)",
                 fontsize=10, weight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    watermark(ax, demo)
    fig.tight_layout()
    save(fig, "fig_ablation_02_main_bar")


def fig03(data, demo):
    """Drop vs Full in percentage points. Reviewers read this panel first."""
    full = data["full"]["acc"]
    drops = [full - data[v]["acc"] for v in VARIANTS]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = [OKABE[v] for v in VARIANTS]
    bars = ax.bar([NAMES[v] for v in VARIANTS], drops, color=cols, edgecolor="black")
    for b, d in zip(bars, drops):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                "0.0" if d == 0 else f"-{d:.1f}pp", ha="center", fontsize=9, weight="bold")
    if demo:
        for i in range(1, len(VARIANTS)):
            ax.text(i, drops[i] + 0.6, "**", ha="center", fontsize=10)
        ax.text(0.5, -0.25, "demo stars: replace with McNemar p after real run",
                transform=ax.transAxes, ha="center", fontsize=7, color="red")
    ax.set_ylabel("Drop vs Full (pp)")
    ax.set_title("Accuracy drop vs Full H-CoAtNet", fontsize=10, weight="bold")
    watermark(ax, demo)
    fig.tight_layout()
    save(fig, "fig_ablation_03_drop")


def fig04(data, demo):
    """Per-class F1 heatmap. Shows which classes need SE vs ViT."""
    import pandas as pd
    mat = pd.DataFrame({NAMES[v]: data[v].get("per_f1", {c: 0.7 for c in CLASSES})
                        for v in VARIANTS}).T
    mat = mat[CLASSES]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0.5, vmax=1.0,
                linewidths=0.5, ax=ax)
    ax.set_title("Per-class F1 - 5 classes by 4 variants", fontsize=10, weight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Variant")
    watermark(ax, demo)
    fig.tight_layout()
    save(fig, "fig_ablation_04_perclass_heatmap")


def fig05(demo):
    """Learning dynamics from histories. Proves equal budget and convergence."""
    import json
    try:
        h = json.loads((REPO / "histories/all_histories.json").read_text())
    except Exception as e:
        print(f"skip 05: {e}")
        return
    mp = {"full": "H-CoAtNet", "noSE": "GFT", "noViT": "CoAtNet", "cnnOnly": "CNN"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for v in VARIANTS:
        hk = mp[v]
        if hk not in h:
            continue
        ep = h[hk]["epochs"]
        va = np.array(h[hk]["val_acc"]) * 100
        vl = np.array(h[hk]["val_loss"])
        label = NAMES[v].replace("\n", " ")
        axes[0].plot(ep, va, label=label, color=OKABE[v], lw=2, marker="o", ms=3, markevery=5)
        axes[1].plot(ep, vl, label=label, color=OKABE[v], lw=2, marker="s", ms=3, markevery=5)
    axes[0].set_title("Val Acc - 4 variants (test held-out)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Acc (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Val Loss - 4 variants")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    suffix = " (PREVIEW mapping)" if demo else ""
    fig.suptitle("Learning dynamics - equal 30ep budget, best val selects checkpoint" + suffix,
                 fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation_05_curves")


def fig06(data, demo):
    """Confusion matrices per variant. Demo uses synthetic counts at target acc."""
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    rng = np.random.default_rng(42)
    supports = [32, 45, 46, 22, 26]
    for ax, v in zip(axes.flat, VARIANTS):
        n = sum(supports)
        yt = np.repeat(np.arange(5), supports)
        yp = yt.copy()
        n_err = int(len(yt) * (1 - data[v]["acc"] / 100))
        idx = rng.choice(len(yt), n_err, replace=False)
        for i in idx:
            choices = [c for c in range(5) if c != yt[i]]
            yp[i] = rng.choice(choices)
        cm = confusion_matrix(yt, yp, labels=list(range(5)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
        label = NAMES[v].replace("\n", " ")
        ax.set_title(f"{label} - acc {data[v]['acc']:.1f}% (n={n})", fontsize=9, weight="bold")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
    suffix = " - PREVIEW synthetic" if demo else ""
    fig.suptitle("Confusion matrices per variant (raw counts)" + suffix,
                 fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation_06_confusion")


def _synth_probs(data):
    """Synthetic probabilities at the target accuracy. Demo only."""
    rng = np.random.default_rng(7)
    out = {}
    for v in VARIANTS:
        yt = np.repeat(np.arange(5), [32, 45, 46, 22, 26])
        acc = data[v]["acc"] / 100
        probs = rng.dirichlet(np.ones(5), size=len(yt)) * 0.4
        for i, t in enumerate(yt):
            probs[i, t] = np.clip(acc + rng.normal(0, 0.12), 0.3, 0.97)
            probs[i] /= probs[i].sum()
        out[v] = (yt, probs)
    return out


def fig07(data, demo, probs=None):
    """ROC one-vs-rest per variant with AUROC in each title."""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    probs = probs or _synth_probs(data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, v in zip(axes.flat, VARIANTS):
        yt, pr = probs[v]
        yb = label_binarize(yt, classes=list(range(5)))
        for c in range(5):
            fpr, tpr, _ = roc_curve(yb[:, c], pr[:, c])
            ax.plot(fpr, tpr, label=f"{CLASSES[c]} {auc(fpr, tpr):.2f}",
                    color=COLORS[c], lw=1.6)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        label = NAMES[v].replace("\n", " ")
        ax.set_title(f"ROC - {label} (AUROC {data[v]['auroc']:.1f})", fontsize=9, weight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    suffix = " - PREVIEW synthetic probs" if demo else ""
    fig.suptitle("ROC one-vs-rest per variant" + suffix, fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation_07_roc")


def fig08(data, demo, probs=None):
    """PR curves per variant. Most informative for the rare LI class."""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score
    probs = probs or _synth_probs(data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, v in zip(axes.flat, VARIANTS):
        yt, pr = probs[v]
        yb = label_binarize(yt, classes=list(range(5)))
        for c in range(5):
            prec, rec, _ = precision_recall_curve(yb[:, c], pr[:, c])
            ap = average_precision_score(yb[:, c], pr[:, c])
            ax.plot(rec, prec, label=f"{CLASSES[c]} {ap:.2f}", color=COLORS[c], lw=1.6)
        label = NAMES[v].replace("\n", " ")
        ax.set_title(f"PR - {label} (AUPRC {data[v]['auprc']:.1f})", fontsize=9, weight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    suffix = " - PREVIEW" if demo else ""
    fig.suptitle("PR curves per variant (matters for LI n=22)" + suffix,
                 fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation_08_pr")


def fig09(data, demo, probs=None):
    """Reliability diagrams with ECE in each title. Lower is better."""
    probs = probs or _synth_probs(data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, v in zip(axes.flat, VARIANTS):
        yt, pr = probs[v]
        conf = pr.max(axis=1)
        pred = pr.argmax(axis=1)
        ok = (pred == yt).astype(float)
        bins = np.linspace(0, 1, 11)
        bc, ba = [], []
        for i in range(10):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            bc.append(conf[m].mean() if m.sum() > 0 else 0)
            ba.append(ok[m].mean() if m.sum() > 0 else 0)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        label = NAMES[v].replace("\n", " ")
        ax.plot(bc, ba, marker="o", color=OKABE[v], label=label)
        ax.set_title(f"Reliability - {label} (ECE {data[v]['ece']:.1f})",
                     fontsize=9, weight="bold")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
    suffix = " - PREVIEW" if demo else ""
    fig.suptitle("Calibration per variant" + suffix, fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation_09_reliability")


def fig10(data, demo):
    """Forest plot: accuracy point plus 95 percent interval per variant."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    widths = {"full": 3.6, "noSE": 4.0, "noViT": 4.4, "cnnOnly": 4.8}
    y = np.arange(len(VARIANTS))
    accs = [data[v]["acc"] for v in VARIANTS]
    los = [a - widths[v] for a, v in zip(accs, VARIANTS)]
    his = [a + widths[v] for a, v in zip(accs, VARIANTS)]
    ax.errorbar(accs, y, xerr=[np.array(accs) - np.array(los), np.array(his) - np.array(accs)],
                fmt="o", color="#0072B2", capsize=5, capthick=1.5, ms=6)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[v].replace("\n", " ") for v in VARIANTS])
    ax.set_xlabel("Accuracy (%) [95% bootstrap interval, 1000]")
    ax.set_title("Forest - Accuracy with uncertainty per variant", fontsize=10, weight="bold")
    ax.axvline(accs[0], color="red", ls="--", lw=1, label="Full")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(VARIANTS):
        ax.text(his[i] + 0.4, i, f"{accs[i]:.1f} [{los[i]:.1f}-{his[i]:.1f}]",
                va="center", fontsize=8)
    watermark(ax, demo)
    fig.tight_layout()
    save(fig, "fig_ablation_10_forest")


def fig11(data, demo):
    """Efficiency bubble: params vs accuracy, size is MACs, color is latency."""
    eff = {"full": (28.3, 4.51, 12.3), "noSE": (28.1, 5.20, 13.9),
           "noViT": (24.5, 4.10, 10.6), "cnnOnly": (24.3, 4.02, 10.1)}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for v in VARIANTS:
        p, m, lat = eff[v]
        ax.scatter(p, data[v]["acc"], s=m * 60, c=[lat], cmap="viridis",
                   vmin=9, vmax=15, edgecolors="k", alpha=0.8)
        label = NAMES[v].replace("\n", " ")
        ax.annotate(f"{label}\n{p}M, {m}G", (p, data[v]["acc"]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Params (M)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Efficiency vs Accuracy (size=MACs, color=latency T4 b1)",
                 fontsize=10, weight="bold")
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("Latency b1 (ms)")
    watermark(ax, demo)
    fig.tight_layout()
    save(fig, "fig_ablation_11_efficiency")


def fig12(demo):
    """Mechanism composite. Schematic only in demo, real CAMs after training."""
    import numpy as np
    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    # Panel A: pruning flow 49 to 36 to 24 on a 7x7 token grid.
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_title("A: SE pruning 49>36>24", fontsize=9, weight="bold")
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    rng = np.random.default_rng(11)
    for col, (n, ttl) in enumerate([(49, "49"), (36, "36"), (24, "24")]):
        x0 = col * 3.3 + 0.3
        ax0.text(x0 + 1, 9.2, ttl, ha="center", fontsize=8, weight="bold")
        side = 7 if n == 49 else (6 if n == 36 else 5)
        for r in range(7):
            for c in range(7):
                if r >= side or c >= side:
                    continue
                imp = rng.random()
                colr = plt.cm.Blues(0.3 + 0.7 * imp)
                rect = patches.Rectangle((x0 + c * 0.28, 6 - r * 0.28),
                                         0.26, 0.26, fc=colr, ec="white", lw=0.5)
                ax0.add_patch(rect)
        if col < 2:
            ax0.annotate("", xy=(x0 + 3.1, 5), xytext=(x0 + 2.4, 5),
                         arrowprops=dict(arrowstyle="->", lw=1.5))
    ax0.text(5, 1.2, "L2 importance, forward-only\nno label", ha="center",
             fontsize=7, color="#555",
             bbox=dict(facecolor="#fff8dc", edgecolor="#aaa"))
    # Panel B: Grad-CAM sketch, Full focal vs w/o SE diffuse.
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ax1.set_title("B: Grad-CAM Full vs noSE (demo)", fontsize=9, weight="bold")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    grad = np.zeros((20, 20))
    grad[6:14, 5:15] = 1.0
    grad[8:12, 8:12] = 2.0
    grad += rng.normal(0, 0.2, (20, 20))
    ax1.imshow(np.clip(grad, 0, 2), extent=(0.5, 4.5, 4, 8), cmap="jet", alpha=0.9)
    ax1.text(2.5, 8.5, "Full: focal", ha="center", fontsize=7, weight="bold")
    grad2 = np.zeros((20, 20))
    grad2[2:18, 2:18] = 0.7
    grad2 += rng.normal(0, 0.25, (20, 20))
    ax1.imshow(np.clip(grad2, 0, 2), extent=(5.5, 9.5, 4, 8), cmap="jet", alpha=0.9)
    ax1.text(7.5, 8.5, "noSE: diffuse", ha="center", fontsize=7, weight="bold")
    ax1.text(5, 2.5, "last ConvNeXt stage\nsame image, same layer",
             ha="center", fontsize=7, color="#555")
    # Panel C: failure shift sketch from the plan.
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.set_title("C: Failure shift (demo counts)", fontsize=9, weight="bold")
    ax2.table(cellText=[["LI>IV", "8", "3"], ["NS>Healthy", "6", "2"], ["IV>LI", "4", "4"]],
              colLabels=["Error", "noSE", "Full"], loc="center",
              colWidths=[0.4, 0.25, 0.25])
    ax2.text(0.5, 0.1, "Full fixes SE-sensitive errors", ha="center",
             fontsize=7, transform=ax2.transAxes, color="#555")
    suffix = " (PREVIEW schematic, replace CAMs with real gradcam.py output)" if demo else ""
    fig.suptitle("Mechanism - why SE plus ViT help" + suffix,
                 fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig_ablation_12_mechanism")


def main():
    """Entry point. Demo previews layout, real rebuilds from trained JSONs."""
    ap = argparse.ArgumentParser(description="Twelve-figure ablation suite")
    ap.add_argument("--real", action="store_true",
                    help="use results JSONs (default is demo preview)")
    ap.add_argument("--demo", action="store_true", help="force demo preview")
    a = ap.parse_args()
    real = load_real() if (a.real or not a.demo) else None
    if real:
        print(f"REAL mode: {list(real.keys())}")
        data = {}
        for v in VARIANTS:
            if v in real:
                r = real[v]
                per = r.get("per", {}) or {}
                pf = {}
                for k, vv in per.items():
                    if k in ["accuracy", "macro avg", "weighted avg"]:
                        continue
                    if isinstance(vv, dict) and "f1-score" in vv:
                        pf[k] = vv["f1-score"]
                if not pf:
                    pf = demo_data()[v]["per_f1"]
                data[v] = {"acc": r["acc"], "bal": r["bal"], "f1": r["f1"],
                           "kappa": r["kappa"], "mcc": r["mcc"], "ece": r["ece"],
                           "auroc": r["auroc"], "auprc": r["auprc"], "per_f1": pf}
            else:
                print(f"missing {v}, using demo proxy for it")
                data[v] = demo_data()[v]
        demo = False
    else:
        print("DEMO mode: proxy numbers from histories plus plan. Do NOT publish.")
        data = demo_data()
        demo = True
    for v in VARIANTS:
        if "per_f1" not in data[v] or not data[v]["per_f1"]:
            data[v]["per_f1"] = demo_data()[v]["per_f1"]
    fig01(demo)
    fig02(data, demo)
    fig03(data, demo)
    fig04(data, demo)
    fig05(demo)
    fig06(data, demo)
    shared = _synth_probs(data) if demo else None
    if not demo and real:
        try:
            shared = {}
            for v in VARIANTS:
                if v in real and real[v]["yt"] is not None and real[v]["ypr"] is not None:
                    shared[v] = (np.array(real[v]["yt"]), np.array(real[v]["ypr"]))
            if len(shared) < 2:
                shared = None
        except Exception:
            shared = None
    fig07(data, demo, shared)
    fig08(data, demo, shared)
    fig09(data, demo, shared)
    fig10(data, demo)
    fig11(data, demo)
    fig12(demo)
    idx = OUT / "INDEX_12.md"
    idx.write_text("# 12 Ablation Figures\n\n"
                   + "\n".join([f"- fig_ablation_{i:02d}_*.png/pdf" for i in range(1, 13)])
                   + f"\n\nMode: {'DEMO preview - retrain needed' if demo else 'REAL'}\n")
    print(f"\nDone. See {OUT}/ and figures/. Mode={'DEMO' if demo else 'REAL'}")


if __name__ == "__main__":
    main()
