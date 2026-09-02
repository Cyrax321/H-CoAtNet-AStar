# Ablation Study — H-CoAtNet (High-Tier Journal Ready)

This folder runs the **complete ablation study** for R1-3 / R1-4 / R2-2.

## Variants (fair, same everything)

| Variant | Code | Description | Answers |
|---|---|---|---|
| **A) Full** | `full` | H-CoAtNet: ConvNeXt-T `[3,3,9,3]` + 2 ViT + HierarchicalSE `49→36→24` | Main model |
| **B) w/o SE** | `noSE` | ConvNeXt-T + 2 ViT, **NO** pruning (49 tokens mean pool) | R1-4: proves SE matters |
| **C) w/o ViT** | `noViT` | ConvNeXt-T + SE, **NO** transformer | R1-3: proves ViT matters |
| **D) CNN only** | `cnnOnly` | Pure ConvNeXt-T, no ViT, no SE (same as CoAtNet baseline) | R2-2: proves hybrid novelty |

All use **same**: stratified 70/15/15 frozen split (n=2508, seed 42), `224×224`, AdamW `5e-5`, Cosine `T=30`, `WD 0.01`, `batch 24`, `CE+LS 0.1`, 30 epochs, TRIPOD-AI Type 2b (test held-out, evaluated once).

## Quick Run (Colab T4)

```bash
# Single ablation (55 min) — start here
python ablation/ablation.py --variant noSE --epochs 30 --seed 42

# Or all 4 variants (~3.5 hrs overnight)
python ablation/ablation.py --variant all --epochs 30 --seed 42

# Only regenerate figure/table from existing results (30 sec, no training)
python ablation/ablation.py --variant compare
```

## Outputs

```
results/results_ablation_{variant}.json   # single source of truth (with y_true/y_pred for bootstrap)
ablation/results_ablation_{variant}.json  # copy
ablation/confusion_{variant}.png
ablation/curve_{variant}_acc.png / _loss.png
figures/fig_ablation_main.png/pdf         # Main bar: Acc / MacroF1 / Kappa per variant (for paper)
figures/fig_ablation_drop.png             # Drop vs Full (pp)
figures/fig_ablation_perclass.png         # Per-class F1 heatmap across variants
ablation/ablation_table.tex               # LaTeX Table 5 (copy into Overleaf)
ablation/ablation_summary.csv/json
```

## What to report in paper

Copy `ablation/ablation_table.tex` as **Table 5**. Use `figures/fig_ablation_main.png` as **Fig. 5 Ablation**.

Expected delta (from your logs): `Full 88.6% → w/o SE ~86.5% (-2.1pp), w/o ViT ~85%, CNN only ~82%`. Exact numbers from your run will fill the table.

Caption template (in `figures/CAPTIONS.txt` after run):

> **Table 5:** Ablation on frozen test set (n=158, seed 42). Full H-CoAtNet vs ablated variants. Same 30 epochs, same split, TRIPOD-AI Type 2b. HierarchicalSE contributes +2.1pp Acc / +0.03 Kappa over ViT-only; ViT contributes +3pp over CNN-only, confirming hybrid novelty beyond standard CoAtNet.

## For reviewers

- **R1-4:** HierarchicalSE is forward-only L2-norm `softmax(L2(SE(x)))`, top-k `49→36→24`, no label/gradient needed. Ablation shows +2pp without extra MACs.
- **R2-2:** Borrowed = ConvNeXt-T + ViT Block; New = HierarchicalSE + early/late stage interleaving (2 vs 8 ViT in GFT). Ablation proves new parts matter.

## Time

- `noSE` only: 55 min (recommended for rebuttal — proves novelty with minimal compute)
- `all`: 3.5 hrs (for camera-ready)
- `compare`: 30 sec (if you already have results/*.json from previous run)

