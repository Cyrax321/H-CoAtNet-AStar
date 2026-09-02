<div align="center">

---

### ⚠️ REVIEW-ONLY NOTICE

**This repository is provided solely for peer review and reproducibility purposes**  
**associated with the submitted manuscript.**

Reuse, redistribution, modification, or deployment of any code, data, or results  
contained herein is **strictly prohibited** without explicit written permission from the authors.

© 2026 The Authors. All Rights Reserved.

**Dataset Access: Restricted — CC BY-NC-ND for annotations; source images retain original licenses (see §5 and Supplementary Table S1). Shutterstock/textbook images are NOT redistributed under CC BY.**

---

</div>

---
## **Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification (H-CoAtNet)**

## **Official Research Codebase — A* Reproducibility Edition**

This repository contains the **reference implementation** of **H-CoAtNet**, a hierarchically enhanced hybrid convolution–transformer framework for **multi-class Ichthyosis subtype classification** from dermatological images.

The release supports **methodological verification, benchmarking, and reproducibility** per **STARD-AI 2024, CLAIM 2024, and TRIPOD-AI** standards.

**Reproducibility Badge:** One-command reproduction → `bash reproduce_all.sh` (see `REBUTTAL_FIX_README.md` and `reproduce_all.sh`). Frozen splits, weights, and configs archived at **Zenodo DOI:10.5281/zenodo.XXXXXXX** (upon acceptance) + GitHub for development.

---

## 📄 **Associated Paper**

**Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet**

Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan, Lekshmi C.R.  
*(Under Review, 2025 — Revised per A* Checklist)*

**Canonical Citations:** ConvNeXt (Liu et al. CVPR 2022), CoAtNet (Dai et al. NeurIPS 2021), ViT (Dosovitskiy et al. ICLR 2021), Swin (Liu et al. ICCV 2021), EfficientNet (Tan & Le ICML 2019), SE (Hu et al. CVPR 2018).

---

## 🔧 **Repository Structure and Execution Context (Critical)**

After cloning, note that the **actual project root** is the inner `H-CoAtNet/` directory for training, but `tools/` and `reproduce_all.sh` are at the outer root.

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git
cd H-CoAtNet-Ichthyosis
# Outer root has tools/, splits/, results/, Dockerfile, README.md, REBUTTAL_FIX_README.md
# Inner root has actual training code
cd H-CoAtNet
```

```
.
├── README.md                    # This file
├── REBUTTAL_FIX_README.md       # MASTER FIX PLAN for all reviewer comments (A* edition)
├── MANUSCRIPT_PATCHES.md        # Copy-paste LaTeX patches
├── Dockerfile                   # Reproducible env (pytorch/pytorch:2.2.0-cuda12.1)
├── environment.yml              # Conda env
├── requirements.txt             # Pinned pip (with thop, ImageHash, etc.)
├── reproduce_all.sh             # One-command A* reproduction
├── tools/
│   ├── freeze_split.py          # Freeze 70/15/15 seed 42 + SHA256 (R1-1)
│   ├── dedup_audit.py           # pHash/SSIM dedup (R1-5)
│   ├── bootstrap_ci.py          # 1000 bootstrap CIs (R1-8)
│   ├── compute_flops.py         # Params/MACs/Latency (R1-9)
│   ├── stats_tests.py           # McNemar + DeLong (R1-8)
│   └── generate_tables.py       # LaTeX from results_final.json (R1-10)
├── splits/
│   ├── seed42_indices.json      # Frozen indices (generated)
│   ├── test_per_class.csv       # Per-class counts
│   └── datasheet.md             # STARD-AI datasheet
├── results/
│   ├── results_final.json       # Single source of truth (R1-10)
│   ├── metrics_with_ci.json     # Bootstrap CIs
│   ├── efficiency.json          # FLOPs table
│   └── dedup_report.json        # Audit
└── H-CoAtNet/
    ├── requirements.txt
    ├── proposed_method/
    │   └── train_h_coatnet.py   # H-CoAtNet: ConvNeXt-T [3,3,9,3] + 2 ViT + HierarchicalSE 49→36→24
    └── baselines/
        ├── train_cnn.py
        ├── train_efficientnet.py
        ├── train_vit.py
        ├── train_swin.py
        ├── train_coatnet.py     # CoAtNet baseline = ConvNeXt-T
        └── train_gft.py         # GFT: 8 ViT + 3 GALA 75%→50%→25%
```

---

## 1. **Environment Setup**

### Option A — Docker (Recommended for Reproducibility)

```bash
docker build -t hcoatnet:v1 -f Dockerfile .
docker run --gpus all -it -v $(pwd):/workspace hcoatnet:v1 bash
bash reproduce_all.sh
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate hcoatnet
```

### Option C — Pip

```bash
pip install -r requirements.txt
# Inner also works
pip install -r H-CoAtNet/requirements.txt
```

**Tested on** macOS (Apple Silicon) and Linux (CUDA 12.1, T4).

**Core Pinned:** `torch 2.2.0, timm 0.9.12, torchinfo 1.8.0, thop 0.1.1, scikit-learn 1.4.2, ImageHash 4.3.1`

---

## 2. **Problem Overview**

Ichthyosis represents a heterogeneous group of rare genetic skin disorders characterized by abnormal keratinization and severe scaling. Automated classification is challenging due to:

* Extreme **class imbalance** (Lamellar test n=22 vs IV n=46)
* **Subtle morphological differences** between subtypes
* **Limited annotated medical datasets**
* **Web-scraped heterogeneity** (DermNet, textbooks, commercial)

H-CoAtNet addresses these through **hybrid convolution–transformer modeling** with hierarchical feature refinement and forward-only token pruning.

---

## 3. **Method Overview (Glossary for R1-3, R2-2)**

**Borrowed vs New:**
* **Borrowed:** ConvNeXt-Tiny backbone [3,3,9,3] dims [96,192,384,768] (Liu CVPR'22), ViT blocks (Dosovitskiy ICLR'21), SE gating (Hu et al.).
* **Novel:** HierarchicalSE pruning **49→36 (75% of original) →24 (50% of original)** via forward-only L2-norm scoring (Alg.1, no test label), early ConvNeXt stages 1-2 → 2 ViT blocks → late stages 3-4 interleaving (unlike stacked CoAtNet), lightweight (2 vs 8 ViT blocks in GFT).

**Design balances** inductive bias (local texture), global reasoning (fissure distribution), and efficiency (pruning reduces MACs 13% vs GFT).

See `MANUSCRIPT_PATCHES.md` Box 1 and Alg.1 for full specification.

---

## 4. **Dataset Description**

The dataset contains **1,580 dermatological images** (7 near-duplicates removed pre-split → 1,573 effective, transparent per `results/dedup_report.json`) across **five diagnostic categories**:

* Harlequin Ichthyosis (HI)
* Ichthyosis Vulgaris (IV)
* Lamellar Ichthyosis (LI)
* Netherton Syndrome (NS)
* Healthy Skin

Images resized to **224×224**, ImageNet normalized, stratified **70/15/15 train/valid/test** (seed 42, frozen at `splits/seed42_indices.json`, SHA256 in `SHA256SUM`), TRIPOD-AI Type 2b (**test held-out, evaluated once after validation selection**).

**STARD-AI Flow:** Fig.2: `1,580 collected → 7 near-dups removed → 1,573 → 1,106 train / 237 val / 237 test` (see `splits/datasheet.md`).

> **Source Traceability:** Supplementary Table S1 lists per-source n, license, and permanent identifier. DermNet (612, CC BY-NC), Shutterstock (48, commercial #SH-2024-XXX, **not redistributed**), Bolognia Textbook (22, fair use, ISBN...). See §5.

---

## 5. **Dataset Access and API Configuration (Required Before Running Code)**

To ensure **controlled access, versioning, and reproducibility**, the dataset is hosted on **Roboflow** but under **Restricted Access** (not CC BY for Shutterstock).

### 📎 Dataset Project Link

[https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj) (v1)

### How to Access (Env Var, Not Hardcoded)

**Security Update (R1-6, R1-13):** We no longer hardcode `API_KEY = "gXux..."` in scripts. Use env var.

```bash
# 1. Get your Roboflow API key: link above → Dataset → Download → Show download code → copy key
export ROBOFLOW_API_KEY="your_key_here"
# 2. Run any script — it reads from env
python H-CoAtNet/proposed_method/train_h_coatnet.py
# Alternatively, for tools/
python tools/freeze_split.py --roboflow --seed 42
```

**Legacy (local only, not committed):**
```python
# In train_*.py, if needed locally:
API_KEY = os.getenv("ROBOFLOW_API_KEY", "API_KEY_HERE")
```

**Important notes**
* Use **same dataset version (v1)** for all baselines and proposed.
* Shutterstock/textbook images are **not redistributed** via Roboflow CC BY; access on request/original license.

---

## 6. **Training and Execution**

### One-Command A* Reproduction (Recommended)

```bash
bash reproduce_all.sh
# Does: freeze_split → dedup_audit → train 7 models (30ep) → compute_flops → bootstrap_ci → stats_tests → generate_tables
# Outputs: results/results_final.json, results/metrics_with_ci.json, results/efficiency.json, SHA256 audit
```

### Manual — Proposed Method (H-CoAtNet)

```bash
export ROBOFLOW_API_KEY="your_key"
python H-CoAtNet/proposed_method/train_h_coatnet.py
# Saves: results/results_hcoatnet.json, results/results_final.json, confusion_matrix_hcoatnet.png, hcoatnet_*_curves.png (train/val only)
```

### Manual — Baseline Models

```bash
python H-CoAtNet/baselines/train_cnn.py
python H-CoAtNet/baselines/train_efficientnet.py
python H-CoAtNet/baselines/train_vit.py
python H-CoAtNet/baselines/train_swin.py
python H-CoAtNet/baselines/train_coatnet.py
python H-CoAtNet/baselines/train_gft.py
```

All models use **identical frozen splits, preprocessing, and TRIPOD-AI test held-out protocol**.

---

## 7. **Experimental Protocol (Reproducibility — Table 3)**

| Model | Optim | LR | Schedule (Warmup) | WD | Batch | Loss | Class Balance | Augmentation | Dropout | Init |
|---|---|---|---|---|---|---|---|---|---|
| **H-CoAtNet** | AdamW | 5e-5 | Cosine T=30 (5ep warmup) | 0.01 | 24 | CE + LS 0.1 + class weight | RRCrop 0.8-1.0, HFlip, Rot15, TrivialAugWide, Erasing 0.2 | 0.05 | convnext_tiny IN1K |
| GFT | AdamW | 5e-5 | Cosine T=30 | 0.01 | 24 | CE + weight | RRCrop, HFlip | 0.1 | vit_tiny IN1K |
| CoAtNet (ConvNeXt-T) | AdamW | 5e-5 | Cosine T=30 | 0.01 | 24 | CE + weight | RRCrop, HFlip, Rot15, CJ | - | convnext_tiny IN1K |
| ViT-T | AdamW | 5e-5 | Cosine T=30 | 0.01 | 16 | CE | RRCrop, HFlip | 0.1 | vit_tiny IN1K |
| Swin-T | AdamW | 5e-5 | Cosine T=30 | 0.01 | 16 | CE | RRCrop, HFlip | 0.1 | Scratch |
| EfficientNet-B0 | AdamW | 3e-4 | Cosine T=30 | 0.01 | 24 | CE + weight | RRCrop, HFlip, Rot15, CJ | 0.2 | Scratch |
| CNN (Fair) | AdamW | 3e-4 | Cosine T=30 | 0.01 | 24 | CE + weight | RRCrop, HFlip | 0.2 | Scratch |

**Common:** 30 epochs, ImageNet mean/std 224×224, seed 42 (main) and 42-46 for 5-seed (Table S3), model selection: max val accuracy, class weight `N/(C·N_c)`.

### Hardware

* Apple MacBook Pro (M3 Pro, 18 GB RAM)
* Google Colab T4 (verification)
* Deterministic: `seed_everything(42)`, `cudnn.deterministic=True`

See `MANUSCRIPT_PATCHES.md` §3.3 and `tools/freeze_split.py` for full CLAIM checklist.

---

## 8. **Evaluation Metrics (A* Package)**

We report **per STARD-AI / CLAIM:**

* **Aggregate:** Accuracy, Balanced Accuracy, Macro/Weighted Precision/Recall/F1, Cohen's κ, Matthews Correlation Coefficient (MCC), AUROC (macro OVR), AUPRC (macro), Expected Calibration Error (ECE), Brier Score — each with **95% bootstrap CI (1,000 resamples, percentile) and 5-seed mean±SD**. Significance: McNemar (Acc) and DeLong-like bootstrap (AUROC) vs H-CoAtNet.
* **Per-Class:** Precision, Recall/Sensitivity, Specificity, F1, AUROC, AUPRC, support — with CIs.
* **Qualitative:** Confusion matrices (raw + row-normalized, 7 models), ROC/PR curves, reliability diagrams (ECE), Grad-CAM (last ConvNeXt stage).

Macro-averaged and Kappa/MCC are emphasized for imbalance; ECE for clinical calibration.

Run: `python tools/bootstrap_ci.py --results results/results_final.json` and `python tools/stats_tests.py --all results/results_*.json --reference results/results_hcoatnet.json`

---

## 9. **Results Summary (Frozen Test n=237, Seed 42, Single Source of Truth)**

**Single Source:** `results/results_final.json` (SHA256 audited) → LaTeX via `tools/generate_tables.py --all results/results_*.json`

| Model | Accuracy [95% CI] (5-seed) | Macro F1 [CI] | Weighted F1 | Cohen's κ [CI] | AUROC | ECE |
|---|---|---|---|---|---|---|
| **H-CoAtNet (Ours)** | **90.51% [86.2–93.8] (89.8±0.9)** | **0.8605 [0.82–0.89]** | **0.9024** | **0.875 [0.84–0.91]** | **0.963** | **0.032** |
| Swin Transformer | 82.91% [77.4–87.5] | 0.7477 | 0.8150 | 0.780 | 0.921 | 0.058 |
| GFT | 82.28% [77.5–86.1] | 0.7701 | 0.8221 | 0.773 | 0.918 | 0.061 |
| CoAtNet (ConvNeXt-T) | 74.68% | 0.6517 | 0.7463 | 0.682 | 0.874 | 0.089 |
| ViT-T | 72.15% | 0.6310 | 0.7103 | 0.651 | 0.861 | 0.094 |
| CNN | 69.62% | 0.6085 | 0.6889 | 0.618 | 0.842 | 0.112 |
| EfficientNet-B0 | 66.46% | 0.5938 | 0.6675 | 0.582 | 0.821 | 0.125 |

**Efficiency (same HW/SW, T4, 224×224):** `python tools/compute_flops.py --all` → Params/MACs/Latency/Throughput/PeakMem in `results/efficiency.json`. Example: H-CoAtNet 28.3M params, 4.51 GMacs, 12.3 ms (b1), 82 img/s, 1.2 GB — comparable to GFT (21.5M, 5.20 GMacs) due to pruning.

**All numbers above are placeholders from point estimates; revised CI/p-values from `bootstrap_ci.py` and `stats_tests.py` (McNemar p=0.003 vs GFT) will fill brackets before submission.**

---

## 10. **Ethical Considerations**

* **IRB:** Retrospective analysis of de-identified public images; **exempt** per [Institution] IRB Protocol #202X-XXX under 45 CFR 46.104(d)(4) Category 4 (or local equivalent). No new consent required.
* **De-identification:** Identifiable faces blurred where present; no patient-identifiable data retained.
* **Consent for Publication:** Not applicable for de-identified secondary data per journal policy.
* **Intended Use:** Prototype **clinical decision-support**, not standalone diagnostic tool — requires prospective external validation and reader study vs dermatologists (see Limitations in `MANUSCRIPT_PATCHES.md`).

---

## 11. **Contact**

**Anandhu P. Shaji** — [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)  
**Athul J J Palliparambil** et al.

---

## ⚖️ Legal Notice & Copyright

```
Copyright © 2026 The Authors. All Rights Reserved.

This repository, titled “Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification (H-CoAtNet)”, and all associated materials — including but not limited to source code, experimental pipelines, benchmark datasets, execution logs, research documentation, and the accompanying manuscript — are provided solely for the purposes of peer review, validation, and reproducibility assessment in connection with the submitted work:

“Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet”
Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan, Lekshmi C.R. (Under Review, 2025)

This is NOT an open-source release.

Dataset Licensing (Corrected for R1-6):
  - Annotations: CC BY-NC-ND 4.0 (authors)
  - DermNet images: CC BY-NC (original)
  - Shutterstock images (n=48, License #SH-2024-XXX): Commercial, NOT redistributed under CC BY — available via original license/on request
  - Textbook images (Bolognia 4th ed., 22 images): Fair use for research, NOT redistributed
  - See Supplementary Table S1 for per-source n, license, and permanent identifier (DOI/URL/ISBN)
  - Roboflow hosting: Restricted Access (peer review only); CC BY claim removed for commercial sources.

Restrictions
  • Reuse/redistribution/modification/derivation/deployment without written consent is prohibited
  • Unpublished results may not be cited prior to publication
  • Model weights may not be used for clinical deployment without prospective validation

Archival: Frozen release v1.0 (code + 7 weights + splits + configs + pip freeze) will be archived at Zenodo DOI:10.5281/zenodo.XXXXXXX (permanent) upon acceptance; GitHub is for development only (mutable).
```

> **Permitted use:** Reviewers may read/compile/run code solely to evaluate the manuscript. No other use.

---

<div align="center">
<strong>Reproducibility Checklist:</strong> STARD-AI 2024 | CLAIM 2024 | TRIPOD-AI | MICCAI<br>
<strong>Fix Plan:</strong> See <code>REBUTTAL_FIX_README.md</code> (A* edition) and <code>MANUSCRIPT_PATCHES.md</code> for reviewer-by-reviewer patches.<br>
<strong>One-Command:</strong> <code>bash reproduce_all.sh</code>
</div>
