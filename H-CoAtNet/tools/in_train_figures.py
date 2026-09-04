#!/usr/bin/env python3
"""Called DURING training final evaluation. Safe: all in try/except, never breaks training."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def save_in_train_figures(y_true, y_probs, class_names, model_tag, results_dir="results", fig_dir="figures"):
    """Saves per-model ROC, PR, reliability + y_probs.npy. Returns dict of paths."""
    y_true = np.array(y_true); y_probs = np.array(y_probs)
    Path(results_dir).mkdir(exist_ok=True); Path(fig_dir).mkdir(exist_ok=True)
    out = {}
    try:
        np.save(Path(results_dir) / f"y_probs_{model_tag}.npy", y_probs)
        out["npy"] = str(Path(results_dir) / f"y_probs_{model_tag}.npy")
    except Exception as e:
        print(f"  [Fig] npy save skip: {e}")
    # ROC + PR need one-vs-rest
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        n_cls = y_probs.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        COLORS = ["#0072B2","#D55E00","#009E73","#CC79A7","#56B4E9"]
        plt.figure(figsize=(6,5))
        for i in range(n_cls):
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_probs[:,i])
            lbl = class_names[i] if i < len(class_names) else f"C{i}"
            plt.plot(fpr, tpr, label=f"{lbl} {auc(fpr,tpr):.2f}", color=COLORS[i%len(COLORS)])
        plt.plot([0,1],[0,1],"k--", label="Chance")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {model_tag} (n={len(y_true)})", fontsize=10)
        plt.legend(fontsize=7); plt.tight_layout()
        p = Path(results_dir) / f"roc_{model_tag}.png"; plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
        q = Path(fig_dir) / f"fig7_roc_{model_tag}.png"; 
        try:
            import shutil; shutil.copy(p, q)
        except: pass
        out["roc"] = str(p)
        plt.figure(figsize=(6,5))
        for i in range(n_cls):
            prec, rec, _ = precision_recall_curve(y_bin[:,i], y_probs[:,i])
            ap = average_precision_score(y_bin[:,i], y_probs[:,i])
            lbl = class_names[i] if i < len(class_names) else f"C{i}"
            plt.plot(rec, prec, label=f"{lbl} {ap:.2f}", color=COLORS[i%len(COLORS)])
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {model_tag}", fontsize=10)
        plt.legend(fontsize=7); plt.tight_layout()
        p2 = Path(results_dir) / f"pr_{model_tag}.png"; plt.savefig(p2, dpi=300, bbox_inches="tight"); plt.close()
        try:
            import shutil; shutil.copy(p2, Path(fig_dir) / f"fig7_pr_{model_tag}.png")
        except: pass
        out["pr"] = str(p2)
        print(f"  [Fig] ROC/PR saved for {model_tag}")
    except Exception as e:
        print(f"  [Fig] ROC/PR skip {model_tag}: {e}")
    try:
        conf = np.max(y_probs, axis=1); pred = np.argmax(y_probs, axis=1)
        acc = (pred == y_true).astype(float)
        bins = np.linspace(0,1,16); bc, ba = [], []
        for i in range(15):
            m = (conf > bins[i]) & (conf <= bins[i+1])
            bc.append(float(conf[m].mean()) if m.sum()>0 else 0); ba.append(float(acc[m].mean()) if m.sum()>0 else 0)
        plt.figure(figsize=(5,5))
        plt.plot([0,1],[0,1],"k--", label="Perfect")
        plt.plot(bc, ba, marker="o", label=model_tag)
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"Reliability — {model_tag}", fontsize=10)
        plt.legend(); plt.tight_layout()
        p3 = Path(results_dir) / f"reliability_{model_tag}.png"; plt.savefig(p3, dpi=300, bbox_inches="tight"); plt.close()
        try:
            import shutil; shutil.copy(p3, Path(fig_dir) / f"fig8_reliability_{model_tag}.png")
        except: pass
        out["rel"] = str(p3)
        print(f"  [Fig] Reliability saved for {model_tag}")
    except Exception as e:
        print(f"  [Fig] Reliability skip {model_tag}: {e}")
    return out
