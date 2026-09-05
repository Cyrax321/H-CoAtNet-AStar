# Ablation Study Plan - H-CoAtNet

**Goal:** Prove that each new piece in H-CoAtNet actually matters, under fully fair conditions. This is what reviewers will check for R1-3, R1-4, R2-2.

**Main model:** ConvNeXt-Tiny `[3,3,9,3]` + 2 ViT blocks + HierarchicalSE `49 → 36 → 24`
**Dataset:** ich-s-7lnsj v1, frozen stratified 70/15/15, seed 42, 224x224, ImageNet norm
**Protocol:** TRIPOD-AI Type 2b (test evaluated once, after val selection)

Existing code for this lives in `ablation/ablation.py`. This folder (`ablation study/`) is for planning and discussion, not duplicate code.

---

## 1. Research Questions

We need to answer three reviewer questions with numbers, not claims:

1. **RQ1 (R1-4): Does HierarchicalSE help?**
 Compare Full vs w/o SE. If Full wins by ~2pp with same or lower MACs, SE is justified.

2. **RQ2 (R1-3): Does the ViT part help?**
 Compare Full vs w/o ViT. If Full wins by ~3pp, the transformer adds global context CNN alone misses.

3. **RQ3 (R2-2): Is the hybrid actually novel beyond standard CoAtNet?**
 Compare Full vs CNN-only (pure ConvNeXt-T, which equals your CoAtNet baseline). If Full wins by ~6-8pp, novelty holds.

Secondary (only if time permits for camera-ready):
- RQ4: Is `49 → 36 → 24` better than `49 → 24` direct or no pruning?
- RQ5: Are 2 ViT blocks enough vs 1 or 4? (cost vs gain)
- RQ6: Which classes benefit most? (per-class F1 heatmap)

For rebuttal, RQ1-RQ3 are enough. RQ4-RQ6 are bonus.

---

## 2. Variants (only 4, keep it clean)

| ID | Variant flag | Architecture | What it proves |
|----|--------------|--------------|----------------|
| A | `full` | ConvNeXt-T + 2 ViT + HierarchicalSE 49→36→24 | Main result, reference |
| B | `noSE` | ConvNeXt-T + 2 ViT, no pruning (mean pool 49) | SE contribution |
| C | `noViT` | ConvNeXt-T + SE, no transformer | ViT contribution |
| D | `cnnOnly` | ConvNeXt-T only | Hybrid novelty |

No other changes between variants. That is the whole point.

Implementation: `ablation/ablation.py` uses `AblationCoAtNet(use_vit, use_se)` factory. It imports training loops directly from `train_h_coatnet.py` so there is no drift.

---

## 3. Fairness Controls (reviewers check this first)

All variants share:

- Same frozen split: `splits/seed42_indices.json`, seed 42, same Roboflow v1 files (train 1106 / val 237 / test 237 in README, 2196/154/158 in ablation doc - confirm exact n before paper and freeze one number)
- Same preprocessing: 224x224, ImageNet mean/std
- Same augmentation: RRCrop 0.8-1.0, HFlip, Rot15, TrivialAugWide, Erasing 0.2
- Same loss: CE + label smoothing 0.1 + class weights `N/(C*Nc)`
- Same optimizer: AdamW lr 5e-5, WD 0.01, Cosine T=30, no extra warmup tricks for one variant only
- Same batch 24, same 30 epochs, same seed 42 deterministic
- Same hardware for timing: Colab T4, batch 1 latency
- Same selection rule: best val acc → test once

Unfair (do not do):
- Tuning LR separately per variant
- Changing epochs for one variant
- Using different augmentation or pretrained weights per variant
- Touching test set during tuning

---

## 4. Metrics to Report

Per variant, on frozen test once:

**Aggregate:**
- Accuracy, Balanced Accuracy
- Macro P/R/F1, Weighted P/R/F1
- Cohen Kappa, MCC
- AUROC macro OVR, AUPRC macro
- ECE + Brier (calibration matters for clinical claim)

**Per class (HI, IV, LI, NS, Healthy):**
- Precision, Recall, F1, AUROC, support

**Efficiency:**
- Params (M), MACs (G), latency ms (b1, T4), throughput, peak mem
- Use `tools/compute_flops.py` for all variants on same machine

**Stats:**
- 95% bootstrap CI, 1000 resamples, percentile (`tools/bootstrap_ci.py`)
- McNemar for accuracy Full vs each ablated
- Bootstrap DeLong-style for AUROC
- 5-seed mean±SD only if time (42-46) for Table S3, else single seed 42 + CI is acceptable for rebuttal

---

## 5. Execution Workflow

### Step 0: Smoke test (2 min, must do first)
```bash
python ablation/ablation.py --variant noSE --epochs 1 --seed 42
```
Check `results/results_ablation_noSE.json` has real accuracy. If this passes, full run will pass.

### Step 1: Minimum viable rebuttal (55 min)
Run one ablated variant only:
```bash
python ablation/ablation.py --variant noSE --epochs 30 --seed 42
```
This alone proves SE novelty (+2pp claim). Cheapest path if deadline is tight.

### Step 2: Full ablation (3.5 hrs overnight on T4)
```bash
python ablation/ablation.py --variant all --epochs 30 --seed 42
```
Produces 4 JSONs: `results_ablation_full/noSE/noViT/cnnOnly.json`

### Step 3: Compare only (30 sec, no training)
```bash
python ablation/ablation.py --variant compare
# or
python ablation/ablation.py --compare
```
Generates:
- `figures/fig_ablation_main.png/pdf` (bar: Acc / MacroF1 / Kappa per variant)
- `figures/fig_ablation_drop.png` (drop vs Full in pp)
- `figures/fig_ablation_perclass.png` (per-class F1 heatmap)
- `ablation/ablation_table.tex` (LaTeX Table 5)
- `ablation/ablation_summary.csv/json`

### Step 4: Stats + efficiency
```bash
python tools/bootstrap_ci.py --results results/results_ablation_full.json
python tools/stats_tests.py --all results/results_ablation_*.json --reference results/results_ablation_full.json
python tools/compute_flops.py --all
```

---

## 6. Expected Pattern and How to Interpret

Based on your logs (to be replaced with exact run):

- Full ~88-90% Acc, MacroF1 ~0.86, Kappa ~0.87
- w/o SE ~86.5% (-2.1pp) → SE works, and should also show slightly higher MACs without pruning or same MACs with lower acc
- w/o ViT ~85% (-3 to -4pp) → ViT adds global fissure reasoning
- CNN-only ~82% or 74% depending on baseline alignment (-6 to -8pp) → hybrid justified

Interpretation rules:
- If noSE drop <1pp and p>0.05: SE is weak, need Grad-CAM + per-class analysis to salvage (check LI and HI which need token focus)
- If noViT drop is large on NS and IV only: argue ViT helps diffuse patterns, CNN keeps local texture
- If cnnOnly ≈ Full: problem, means hybrid adds nothing. Check training (did ViT actually train or frozen by mistake?)

Always report drop in percentage points (pp), not relative percent. Reviewers prefer pp.

---

## 7. Paper Integration

**Table 5 (from `ablation_table.tex`):**
Columns: Variant | Acc [95% CI] | MacroF1 | Kappa | AUROC | ECE | Params | MACs | Δ vs Full

**Fig 5:**
- (a) grouped bar Acc / MacroF1 / Kappa per variant
- (b) drop vs Full
- (c) per-class F1 heatmap

**Caption draft:**
> Table 5: Ablation on frozen test set (n=237, seed 42). Full H-CoAtNet vs ablated variants under identical 30-epoch protocol, TRIPOD-AI Type 2b. HierarchicalSE contributes +2.1pp Acc / +0.03 Kappa over ViT-only; ViT contributes +3pp over CNN-only.

**Text patch for Methods §3:**
> All ablations share frozen split, preprocessing, optimizer, and 30-epoch budget; only use_vit and use_se are toggled. SE importance is forward-only L2-norm softmax(L2(SE(x))), top-k 49→36→24, requiring no label.

**Reviewer mapping:**
- R1-4 → cite B vs A + Alg.1 + forward-only proof
- R1-3 → cite C vs A + per-class gains
- R2-2 → cite D vs A + Borrowed vs New box (ConvNeXt+ViT borrowed, SE + interleaving new)

---

## 8. Checklist Before Calling It Done

- [ ] Frozen split SHA recorded, same n reported everywhere (fix 237 vs 158 mismatch now)
- [ ] 4 JSONs exist with y_true/y_pred/y_probs for bootstrap
- [ ] Table 5 numbers = JSON numbers = text numbers (no manual copy errors)
- [ ] CIs computed, McNemar p reported
- [ ] Efficiency measured on same HW
- [ ] Curves show train/val only (no test leakage in plots)
- [ ] Confusion matrices saved per variant
- [ ] LaTeX compiles in Overleaf

---

## 9. Open Points to Discuss

1. Do we run minimal (noSE only) or full 4-variant now? Compute budget?
2. Confirm test n: 237 or 158? README says 237, ablation README says 158. Must lock one.
3. Do we add RQ4 (pruning ratio sweep) for camera-ready or skip?
4. Colab or local? T4 required for fair latency numbers.
5. Do we need 5-seed for ablation or is single-seed + bootstrap enough for rebuttal?

Pick these together, then launch smoke test.
