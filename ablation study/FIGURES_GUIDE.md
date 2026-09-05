# Ablation Figures Guide - What We Need and Why

This maps what your repo already generates + what reviewers expect for an ablation.
Code ref: `ablation/ablation.py::generate_ablation_figure_and_table()` + `tools/generate_figures.py`

## Core set (main paper, must have)

### F-A1: Architecture variants schematic (new, draw once)
What: One row, 4 blocks. Full vs noSE vs noViT vs cnnOnly.
Show ConvNeXt stem, stages 1-2, 2x ViT box (crossed out when removed), stages 3-4, SE 49->36->24 box (crossed out when removed), GAP, logits.
Why: Reviewers R1-3 / R2-2 complain about naming confusion. A picture ends it.
Style: Input 224x224x3 dims on each arrow, token counts, B x C x H x W.
File: `figures/fig_ablation_arch.png/pdf` (make in PowerPoint / draw.io, not code)

### F-A2: Main grouped bar (already coded)
What: Grouped bar per variant: Acc, MacroF1, Kappa. Annotate Acc on top.
Why: This is Fig.5 in paper. Shows absolute performance.
Code: generates `figures/fig_ablation_main.png/pdf`
Must include: test n, seed 42 in title, Okabe-Ito colors #0072B2 / #009E73 / #D55E00, 300 DPI, values to 1 decimal.

### F-A3: Drop vs Full in pp (already coded)
What: Bar of Acc drop: Full 0.0, noSE -2.1pp, noViT -3.x pp, cnnOnly -6.x pp.
Why: Reviewers prefer pp drops over raw bars. Makes contribution obvious.
Code: `figures/fig_ablation_drop.png`
Add p-values as stars (* p<0.05, ** p<0.01 from McNemar) on top of each bar.

### F-A4: Per-class F1 heatmap 5 x 4 (already coded)
What: Rows = Full / noSE / noViT / cnnOnly, cols = HI, Healthy, IV, LI, NS. Cell = F1.
Why: Shows where each part helps. Expect LI and HI to suffer most without SE. NS and IV to suffer most without ViT.
Code: `figures/fig_ablation_perclass.png`, Blues cmap, annot .2f, vmin 0 vmax 1.

## Training proof (fairness, must have for supplement)

### F-B1: Train/val curves per variant
What: 4 variants x 2 plots = 8 PNGs: `ablation/curve_{variant}_acc.png` + `_loss.png`. Train vs val only, no test.
Why: Proves equal 30 epoch budget, convergence by epoch 25, no cherry picked early stop. Addresses R1-2 leakage + R2-5 30 epoch concern.
Check: all use same axes, same grid, legend outside.

### F-B2: Combined val comparison
What: One axes, 4 lines: val acc across epochs for all variants. Second plot for val loss. Same as Fig3c/d but for ablations only.
Why: Lets reviewer see Full separates early and stays on top.
Style: markers every 5 epochs, best epoch dot with black edge.

### F-B3: Confusion matrices per variant
What: 4 x 2 = 8 plots: raw counts + row-normalized. `ablation/confusion_{variant}.png`
Why: Shows error shift. Example: without SE, LI confused as IV more. Without ViT, NS confused as Healthy more. Put Full raw in main, rest in supplement.
Must include: n per plot, class names short, fmt d for raw and .2f for norm.

## Performance depth (detailed analysis)

### F-C1: ROC one-vs-rest per variant + PR per variant
What: 5 lines per plot (one per class) + chance line. Report macro AUROC / AUPRC in legend.
Why: Accuracy hides rare class ranking. LI n=22 needs AUPRC. If Full AUROC 0.963 vs noSE 0.93, that supports SE even if Acc gap looks small.
Needs: y_probs saved in JSON (your code already does `evaluate_with_probs`). Code ref `tools/generate_figures.py::fig7_roc_pr`

### F-C2: Reliability diagram + ECE
What: Confidence vs accuracy, 10 bins, diagonal perfect line. Title includes ECE=0.032 etc.
Why: Clinical claim needs calibration. R1-11. If Full ECE 0.032 vs noSE 0.06, SE also helps calibration, strong point.
Code: `fig8_reliability_{variant}.png`

### F-C3: Forest plot with 95 percent CI
What: Y = variants, X = accuracy percent, point + horizontal CI bar from bootstrap 1000. Add vertical line at Full.
Why: This is what makes the analysis complete. Shows uncertainty, not just points. LI wide CI is honest.
Code: `figures/fig10_forest_ci.png`, data in `fig10_data.csv`. Extend current script which only does main models to also do 4 ablation variants.

### F-C4: Efficiency bubble
What: X = Params M, Y = Acc percent, bubble size = MACs G, color = latency ms T4 b1. 4 bubbles.
Why: Answers R1-9. SE should show same or lower MACs with higher Acc (pruning saves 13 percent vs no pruning). ViT adds small cost for clear gain.
Code: `figures/fig9_efficiency_bubble.png`, data `results/efficiency.json` via `tools/compute_flops.py --all`

## Mechanism proof (this sells novelty)

### F-D1: Token pruning visualization
What: For 6 example images (2 HI, 2 LI, 2 NS): original + L2 importance heat + kept 36 mask + kept 24 mask.
Why: R1-4 asks how SE prunes 49->36->24 without label. Show it focuses on scales/fissures, drops background/healthy skin.
How: Hook `HierarchicalSE` importance output, overlay on 7x7 grid upsampled to 224. Not yet coded, needs small script.

### F-D2: Grad-CAM Full vs noSE side by side
What: 2x3 grid: col1 original, col2 Full CAM, col3 noSE CAM. Same image, same layer (last ConvNeXt stage).
Why: Visual proof SE changes focus. Dermatologist rating 4/6 plausible adds clinical weight for Suppl Fig S3.
Code base: `tools/gradcam.py`

### F-D3: Failure shift grid
What: 6 misclassified cases where Full is right but ablated is wrong (and vice versa). Show pred vs true + confidence.
Why: Honest error analysis. Reviewers love this. Shows limits for LI n=22.

## Tables that go with figures

T5: `ablation/ablation_table.tex` - Variant | Acc [CI] | BalAcc | MacroF1 | Kappa | MCC | ECE | Params | MACs | delta vs Full
T-S3: 5-seed + McNemar p + DeLong p for ablations
CSV: `ablation/ablation_summary.csv/json` + `figures/fig_ablation_*.csv` for reproducibility

## Minimal vs full submission

Minimal rebuttal (55 min + 30 sec compare):
F-A2 + F-A3 + T5. Enough to answer R1-4.

Full run (overnight 3.5 hr):
All of F-A + F-B + F-C + F-D1/D2. About 25 files. Put F-A2/A3 + T5 in main, rest in supplement.

## Style checklist for every figure

- 300 DPI PNG + PDF, tight bbox, 10pt min font, colorblind safe Okabe-Ito
- Title includes n and seed: Test n=237 (lock number), seed 42
- Axes labeled with units, legend outside, grid alpha 0.3
- No test curve in train plots (TRIPOD Type 2b caption)
- Numbers in figure = numbers in JSON = numbers in table (run audit)
- Save data CSV alongside each figure for reproducibility

Open: confirm test n, then regenerate F-A2/A3/A4 via `--compare` once JSONs exist.
