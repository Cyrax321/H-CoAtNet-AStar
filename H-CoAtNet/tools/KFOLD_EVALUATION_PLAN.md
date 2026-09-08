# K-Fold Cross-Validation Evaluation Plan for H-CoAtNet
## A* Standard for Top-Tier Medical Imaging Publication (MICCAI / MedIA / TMI / IEEE-TMI)

---

## 1. OBJECTIVE

Produce a rigorous **5-fold stratified group-aware cross-validation** benchmark for **7 models** (H-CoAtNet + 6 baselines) on the ichthyosis classification dataset that:
- **Excludes the frozen test set** from all CV folds (TRIPOD-AI Type 2b)
- Uses **identical training protocol** as the single-split benchmark (Table 3)
- Reports **mean ± SD** across folds with **paired statistical tests** (Wilcoxon + Bonferroni)
- Generates **publication-ready Table 6**, supplementary figures, and raw data for reproducibility
- Matches the fairness standards of the ablation study (same seed, same protocol, only architecture differs)

---

## 2. EXPERIMENTAL DESIGN

### 2.1 Data Splits (Leakage-Safe)

| Split | Source | N | Purpose |
|-------|--------|---|---------|
| **Frozen Test** | `dataset/test/` | 158 | Held-out, NEVER used during CV. Final paper Table 4 evaluation. |
| **Development Pool** | `dataset/train/` + `dataset/valid/` | 2,350 | 5-fold CV only. Excludes frozen test entirely. |

**Group-aware stratification:**
- Roboflow exports: `<Class>-<idx>_<aug>_jpg.rf.<hash>.jpg`
- Group key = filename stem **before** first `_jpg.` or `.rf.`
- All augmented copies of the same original image share a group
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` ensures no group leaks across train/val

**Per-class counts (dev pool):**
| Class | Development Pool | Frozen Test |
|-------|------------------|-------------|
| Harlequin ichthyosis | ~384 | 32 |
| Healthy skin | ~540 | 45 |
| Ichthyosis vulgaris | ~552 | 46 |
| Lamellar ichthyosis | ~264 | 22 |
| Netherton syndrome | ~156 | 13 |
| **Total** | **2,350** | **158** |

### 2.2 Models (7 Total)

| Model | Factory | LR | Batch | Pretrained | Notes |
|-------|---------|----|-------|------------|-------|
| **H-CoAtNet** | `HCoAtNet` | 5e-5 | 24 | IN1K | ConvNeXt-T + 2 ViT + HierarchicalSE 49→36→24 |
| **CoAtNet** | `CoAtNet` | 5e-5 | 24 | IN1K | Standard CoAtNet-D (Dai et al. NeurIPS'21) |
| **GFT** | `GFT` | 5e-5 | 24 | IN1K | 8 ViT blocks + 3 GALA stages 75%→50%→25% |
| **Swin-T** | `build_swin` | 5e-5 | 16 | IN1K | Swin Transformer Tiny |
| **ViT-B/16** | `build_vit` | 5e-5 | 16 | IN1K | Vision Transformer Base |
| **CNN** | `BaselineCNN` | 3e-4 | 24 | Scratch | ConvNeXt-Tiny from scratch (ablation A0) |
| **EfficientNet-B0** | `build_efficientnet` | 3e-4 | 24 | Scratch | EfficientNet-B0 from scratch |

### 2.3 Training Protocol (IDENTICAL to Single-Split)

| Parameter | Value | Source |
|-----------|-------|--------|
| Epochs | 30 | Table 3 / ablation fairness lock |
| Optimizer | AdamW | All models |
| LR | 5e-5 (ConvNeXt-based), 3e-4 (CNN/EffNet) | Match ablation / single-split |
| Weight Decay | 0.01 | All models |
| LR Schedule | CosineAnnealing T=epochs | No warmup (equal budget) |
| Loss | CE + Label Smoothing 0.1 | Ablation / single-split |
| Class Weights | N/(C·Nc) per fold (train-fold only) | No test leakage |
| Input | 224×224 | All models |
| Normalization | ImageNet mean/std | Standard |
| Augmentation (H-CoAtNet/CoAtNet/GFT) | RRC(0.8-1.0) + HFlip + Rot15 + TrivialAugmentWide + RandomErasing(0.2) | Match ablation |
| Augmentation (Swin/ViT/CNN/EffNet) | RRC(0.8-1.0) + HFlip only | Baseline scripts |
| Batch Size | 24 (16 for Swin/ViT) | Match single-split |
| Workers | 2 (Linux), 0 (Windows) | Standard |
| Seed | 42 + fold×1000 | Per-fold deterministic |
| DataLoader RNG | Explicit `Generator` + `worker_init_fn` | Reproducible minibatch order |

### 2.4 Per-Fold Checkpointing
- Best model by **validation accuracy** (val loss not used)
- Saved to `results/kfold/checkpoints/<model>/fold<N>_best.pth`
- Frozen test is **NEVER** used for model selection

---

## 3. METRICS (Per Fold + Aggregated)

### 3.1 Primary Metrics (per fold)
| Metric | Function | Notes |
|--------|----------|-------|
| Accuracy | `accuracy_score` | Standard |
| Balanced Accuracy | `balanced_accuracy_score` | Protects rare classes (LI n=22) |
| Cohen's Kappa | `cohen_kappa_score` | Chance-corrected |
| Matthews Correlation Coeff | `matthews_corrcoef` | Balanced binary extension |
| Expected Calibration Error | Custom (15 bins) | Match ablation `compute_ece` |
| Brier Score | Custom | Multi-class probabilistic |
| Macro F1 | `precision_recall_fscore_support(average='macro')` | Per-class mean |
| Weighted F1 | `precision_recall_fscore_support(average='weighted')` | Support-weighted |

### 3.2 Ranking Metrics
| Metric | Function | Notes |
|--------|----------|-------|
| AUROC (macro, OVR) | `roc_auc_score(multi_class='ovr')` | One-vs-rest |
| AUPRC (macro) | `average_precision_score` | Critical for rare LI class |

### 3.3 Per-Class Metrics (per fold)
| Metric | Computation |
|--------|-------------|
| Precision / Recall / F1 / Specificity / Support | From confusion matrix |
| Class-wise confusion matrix | Raw + row-normalized |

### 3.4 Aggregation Across 5 Folds
For each metric: **mean, SD (ddof=1), median, min, max, raw values**
- Reported as `mean ± SD` in Table 6
- Raw per-fold values saved in JSON for transparency

### 3.5 Paired Statistical Tests
- **Wilcoxon signed-rank test** on per-fold accuracy between all 21 model pairs
- **Bonferroni correction** (m=21 tests)
- Report: `statistic`, `p_value`, `p_value_bonferroni`, `median_diff`
- Justification: n=5 folds too small for t-test assumptions; Wilcoxon is non-parametric

---

## 4. OUTPUTS (Complete Artifact List)

### 4.1 Core JSON Outputs (`results/kfold/`)

| File | Purpose |
|------|---------|
| `kfold_protocol.json` | Full run configuration + software versions |
| `kfold_splits.json` | Per-fold file lists + class counts + grouping strategy |
| `kfold_fold_results.json` | Per-fold metrics for all 7 models |
| `kfold_predictions.json` | Per-sample predictions (file, fold, y_true, y_pred, y_probs) |
| `kfold_summary.json` | Aggregated mean/SD/median/min/max per metric |
| `kfold_paired_tests.json` | Wilcoxon + Bonferroni results |

### 4.2 Publication Tables
| File | Purpose |
|------|---------|
| `kfold_summary.csv` | Machine-readable summary table |
| `kfold_table.tex` | **Table 6** LaTeX (mean ± SD, 10 metrics × 7 models) |

### 4.3 Figures (PNG + PDF, 300 DPI)

| Figure | Description |
|--------|-------------|
| `fig_kfold_accuracy_mean_sd.png` | Bar chart with error bars |
| `fig_kfold_macro_f1_mean_sd.png` | Bar chart with error bars |
| `fig_kfold_balanced_accuracy_mean_sd.png` | Bar chart with error bars |
| `fig_kfold_foldwise_accuracy.png` | Lines per model across 5 folds |
| `fig_kfold_foldwise_macro_f1.png` | Lines per model across 5 folds |
| `fig_kfold_metric_summary.png` | Heatmap (7 models × 5 metrics) |
| `fig_kfold_class_distribution.png` | Stacked bar per fold |
| `curves/<model>_fold<N>_accuracy.png` | Train/val curves per fold (suppl) |
| `curves/<model>_fold<N>_loss.png` | Train/val loss curves per fold |
| `confusion/<model>_fold<N>_raw.png` | Raw confusion matrix per fold |
| `confusion/<model>_fold<N>_norm.png` | Normalized confusion matrix per fold |

### 4.4 Checkpoints
| Path | Content |
|------|---------|
| `checkpoints/<model>/fold<N>_best.pth` | Best val checkpoint per fold |

---

## 5. REPRODUCIBILITY REQUIREMENTS

### 5.1 Determinism Stack
- `seed_everything(seed)`: Python, NumPy, Torch, cuDNN
- Per-fold seed = `base_seed + fold_idx * 1000`
- DataLoader: explicit `Generator` + `worker_init_fn` seeded from `torch.initial_seed()`
- cuDNN: `deterministic=True`, `benchmark=False`

### 5.2 Software Versions (Recorded in `kfold_protocol.json`)
- Python, NumPy, PyTorch, torchvision, timm, scikit-learn, scipy
- Exact versions logged at runtime

### 5.3 Split Manifest
- `splits/seed42_indices.json` (frozen test set file list) is the source of truth
- `kfold_splits.json` records per-fold train/val file lists + SHA hashes
- Any discrepancy → hard FAIL in validation

### 5.4 Identity Grouping Documentation
- Explicit in `kfold_protocol.json`: "patient identity not available; grouping by Roboflow filename stem"
- Grouping function `identity_key()` tested in validation

---

## 6. VALIDATION / PRE-FLIGHT CHECKS (Must All PASS)

| Check | Type | Failure = |
|-------|------|-----------|
| All 7 model modules import | PASS/FAIL | Abort |
| All 7 models instantiate + forward (1,3,224,224) | PASS/FAIL | Abort |
| H-CoAtNet param count matches `train_h_coatnet.py` | PASS/FAIL | Abort |
| Dev pool > 0, test pool = 158 | PASS/FAIL | Abort |
| Zero dev/test overlap | PASS/FAIL | Abort |
| Groups ≥ k × classes | PASS/WARN | Warn only |
| 5 folds cover all dev samples, train∩val=∅ | PASS/FAIL | Abort |
| Output dir ready (or warn if non-empty) | PASS/WARN | Warn only |

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1: Restore `tools/kfold/kfold.py` (from git history ✓)
- Copy the exact implementation from commit `07a8405`
- Verify all 7 model factories work with current baseline scripts
- Update any import paths if baseline scripts changed

### Phase 2: Pre-flight Validation
```bash
python tools/kfold/kfold.py --dataset_dir /content/dataset --validate-only
```
Must show: `pass=7+ warn≤1 fail=0`

### Phase 3: Smoke Test (1 model × 1 fold × 1 epoch)
```bash
python tools/kfold/kfold.py --dataset_dir /content/dataset --smoke-test
```
Verifies: checkpoint saves, metrics compute, curves generate, no OOM

### Phase 4: Full 7×5 = 35 Training Runs
```bash
python tools/kfold/kfold.py --dataset_dir /content/dataset --k 5 --epochs 30 --seed 42 --models all
```
Estimated: **~3.5 hours** on T4 (35 runs × ~6 min avg)

### Phase 5: Post-Processing
- Verify all 35 `kfold_fold_results.json` entries exist
- Run `paired_wilcoxon` + Bonferroni
- Generate all figures + Table 6 LaTeX
- Cross-check with single-split results (Table 3/4)

### Phase 6: Freeze & Backup
- Zip `results/kfold/` → Drive
- Record SHA of `kfold_splits.json` + `kfold_protocol.json`
- Tag git commit for exact code version

---

## 8. FAIRNESS CHECKLIST (vs Ablation / Single-Split)

| Aspect | Ablation / Single-Split | K-Fold | Match? |
|--------|-------------------------|--------|--------|
| Frozen test excluded from CV | N/A (single split) | ✅ Hard exclusion | ✅ |
| Dev pool = train + valid | N/A | ✅ 2,350 images | ✅ |
| Group-aware (no aug leakage) | Not applicable | ✅ StratifiedGroupKFold | ✅ |
| Same 30 epochs | ✅ | ✅ | ✅ |
| Same LR / WD / scheduler | ✅ | ✅ | ✅ |
| Same class weights (fold-local) | ✅ | ✅ train-fold only | ✅ |
| Same augmentation per model | ✅ | ✅ | ✅ |
| Same IN1K pretrained init | ✅ | ✅ | ✅ |
| Same batch size | ✅ | ✅ | ✅ |
| Best val selects checkpoint | ✅ | ✅ | ✅ |
| Test evaluated once (after CV) | ✅ | ✅ (Table 4 separate) | ✅ |
| Deterministic seeds | ✅ | ✅ per-fold deterministic | ✅ |
| Only architecture differs | ✅ | ✅ | ✅ |

---

## 9. STATISTICAL REPORTING STANDARDS

### Table 6 (Main Paper)
| Row | Columns |
|-----|---------|
| H-CoAtNet | Acc±SD, BalAcc±SD, MacroF1±SD, W-F1±SD, AUROC±SD, AUPRC±SD, Kappa±SD, MCC±SD, ECE±SD, Brier±SD |
| CoAtNet | ... |
| GFT | ... |
| Swin | ... |
| ViT | ... |
| CNN | ... |
| EfficientNet-B0 | ... |

### Supplementary
- Fold-wise line plots (accuracy, macro F1)
- Per-fold confusion matrices (raw + norm)
- Class distribution across folds
- Per-class metrics per fold (JSON only, not in paper)
- Paired test table with Bonferroni

---

## 10. COLAB EXECUTION TEMPLATE

```python
# 1. Environment
!git clone https://github.com/Cyrax321/H-CoAtNet-AStar.git
%cd H-CoAtNet-AStar
!pip install -q -r requirements-colab.txt

# 2. Dataset (once)
!python tools/freeze_split.py --seed 42  # if not already done
# Download to /content/dataset via Roboflow or copy from Drive

# 3. Validate
!python tools/kfold/kfold.py --dataset_dir /content/dataset --validate-only

# 4. Smoke
!python tools/kfold/kfold.py --dataset_dir /content/dataset --smoke-test

# 5. Full run (3.5 hours)
!python tools/kfold/kfold.py --dataset_dir /content/dataset --k 5 --epochs 30 --seed 42 --models all

# 6. Verify outputs
!ls -la results/kfold/
!cat results/kfold/kfold_summary.csv
!cat results/kfold/kfold_table.tex

# 7. Backup
!zip -r /content/drive/MyDrive/HCoAtNet_kfold/kfold_outputs.zip results/kfold/
```

---

## 11. RISKS & MITIGATIONS

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| T4 OOM (batch 24) | Low | Batch 24 works in ablation; Swin/ViT use 16 |
| Colab disconnect | Medium | Incremental JSON writes after each fold; resume by skipping completed (model,fold) pairs |
| Baseline script import changes | Low | Factories import from baseline modules; validate imports first |
| Group collision (rare class) | Low | 5 classes × ~30-50 groups/class > 5 folds; validated in pre-flight |
| Wilcoxon power (n=5) | Known limitation | Report raw values + median diff; Bonferroni is conservative; don't overinterpret p>0.05 |

---

## 12. SIGN-OFF CRITERIA (Before Paper Submission)

- [ ] `validate-only` → all PASS
- [ ] `smoke-test` → completes without error
- [ ] Full 7×5 runs complete (35 fold results in JSON)
- [ ] `kfold_summary.csv` has 7 rows × 20 columns (10 metrics × mean+SD)
- [ ] `kfold_table.tex` compiles in paper LaTeX
- [ ] All 11 figures generated (PNG + PDF)
- [ ] Paired tests JSON has 21 entries + Bonferroni
- [ ] Frozen test set evaluated separately on per-fold best checkpoints (optional Table 4b)
- [ ] Drive backup complete + SHA recorded
- [ ] Git commit tagged `kfold-v1`

---

## 13. RELATION TO EXISTING EXPERIMENTS

| Experiment | Output | Relation to K-Fold |
|------------|--------|-------------------|
| Single-split (Table 3/4) | `results/results_*.json` | Main paper; k-fold is supplementary robustness |
| Ablation (Table 5) | `results/results_ablation_*.json` | Component analysis; k-fold validates full model stability |
| Bootstrap CI | `tools/bootstrap_ci.py` | Applied to frozen test predictions |
| McNemar / DeLong | `tools/stats_tests.py` | Pairwise on frozen test; k-fold uses Wilcoxon on dev |

**K-Fold does NOT replace single-split.** It supplements it per reviewer R2-4.

---

## APPENDIX: EXACT METRIC DEFINITIONS (for reviewer transparency)

### ECE (15 bins, equal width)
```python
bins = np.linspace(0, 1, 16)
conf = probs.max(axis=1)
pred = probs.argmax(axis=1)
ok = (pred == y_true)
ece = sum(abs(ok[m].mean() - conf[m].mean()) * m.mean() for m in masks if m.sum() > 0)
```

### Brier Score (multi-class)
```python
yb = label_binarize(y_true, classes=range(n_classes))
brier = np.mean((yb - probs) ** 2)
```

### Per-Class Specificity
```python
tn = cm.sum() - (tp + fn + fp)
spec = tn / (tn + fp + 1e-9)
```

### Wilcoxon Signed-Rank
- Null: median difference = 0
- Alternative: two-sided
- Bonferroni: `p_adj = min(1, p_raw * 21)`

---

**End of Plan. Ready for implementation.**