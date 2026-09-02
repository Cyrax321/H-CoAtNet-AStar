#!/usr/bin/env python3
"""
COLAB_COMPLETE_RUN.py — One-paste complete training for H-CoAtNet revision
Paste this ENTIRE file into Colab (GPU T4) and run overnight.
Does: clone -> install -> download (hardcoded key) -> freeze 70/15/15 seed 42 -> dedup -> train 7 models x30ep -> bootstrap CI -> flops -> tables -> graphs
All graphs have exact model names (no A*), fair comparison same split/metrics.
Runtime: ~3.5 hrs for 7 models on T4.
"""

# ========== CELL 1 — Clone (if not already) ==========
# In Colab, run this cell first:
"""
!git clone https://github.com/Cyrax321/H-CoAtNet-AStar.git
%cd H-CoAtNet-AStar
!ls -lh
"""

# ========== CELL 2 — Install ==========
"""
!pip install -r requirements.txt -q
!pip install roboflow -q
"""

# ========== CELL 3 — Download dataset (hardcoded, 2 min) ==========
"""
from roboflow import Roboflow
rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
dataset = project.version(1).download("folder")
print("DATASET:", dataset.location)
# Save location for next cells
import pathlib
pathlib.Path("/tmp/dataset_path.txt").write_text(dataset.location)
"""

# ========== CELL 4 — Freeze + Dedup (before training) ==========
"""
import pathlib
DATASET = pathlib.Path("/tmp/dataset_path.txt").read_text().strip()
!python tools/freeze_split.py --dataset_dir "{DATASET}" --seed 42 --out splits/seed42_indices.json
!cat splits/test_per_class.csv
!python tools/dedup_audit.py --dataset_dir "{DATASET}" --out results/dedup_report.json

"""

# ========== CELL 5 — Train ALL 7 models sequentially (3.5 hrs) ==========
"""
# Each saves: results/results_<model>.json, confusion_matrix, curves with EXACT names
# Fair: same frozen split, same 30ep, same Cosine, same ImageNet norm, val selects, test held-out once

!python H-CoAtNet/proposed_method/train_h_coatnet.py
!python H-CoAtNet/baselines/train_gft.py
!python H-CoAtNet/baselines/train_coatnet.py
!python H-CoAtNet/baselines/train_swin.py
!python H-CoAtNet/baselines/train_vit.py
!python H-CoAtNet/baselines/train_cnn.py
!python H-CoAtNet/baselines/train_efficientnet.py
"""

# ========== CELL 6 — After training: Bootstrap CI + FLOPs + Tables + Graphs (2 min, no GPU) ==========
"""
!python tools/bootstrap_ci.py --results results/results_hcoatnet.json --n_bootstrap 1000 --out results/metrics_hcoatnet_ci.json
!python tools/compute_flops.py --all --out results/efficiency.json
!cat results/efficiency.json
!python tools/generate_tables.py --all results/results_*.json --out results/tables.tex
!cat results/tables.tex
!python tools/stats_tests.py --all results/results_*.json --reference results/results_hcoatnet.json --out results/significance.json
!cat results/significance.json
"""

# ========== CELL 7 — Check all graphs generated with exact names (fair) ==========
"""
!ls -lh results/*.png results/*.json
!ls -lh H-CoAtNet/proposed_method/*.png H-CoAtNet/baselines/*.png 2>/dev/null | head -n 50
# Exact names (no A*):
# results/confusion_matrix_hcoatnet.png, confusion_matrix_hcoatnet_norm.png
# results/hcoatnet_acc_curves.png, hcoatnet_loss_curves.png  (train vs val only, no test leakage)
# results/confusion_matrix_gft.png, confusion_matrix_swin.png, etc. (one per model, same style)
# results/efficiency.json (all models same HW)
# results/tables.tex (Table 8 + 9 from single results_final.json)
"""

# ========== CELL 8 — Save to Drive (optional, for tomorrow) ==========
"""
from google.colab import drive
drive.mount('/content/drive')
!cp -r results /content/drive/MyDrive/HCoAtNet_results_$(date +%Y%m%d)
!cp -r splits /content/drive/MyDrive/
print("Saved to Drive")
"""

# ========== NOTES FOR FAIRNESS ==========
"""
Fair comparison ensured:
- Same frozen split: splits/seed42_indices.json (seed 42, stratified 70/15/15, SHA256)
- Same preprocessing: 224x224, ImageNet mean/std
- Same protocol: AdamW, Cosine T=30, 30ep, batch 24/16, val selects best, test evaluated ONCE (TRIPOD-AI)
- Same metrics for all: Accuracy [95% CI], Balanced Acc, Macro F1, Kappa, MCC, AUROC, ECE, confusion matrix (same cmap, same dpi 300)
- Same hardware: Colab T4, same latency measurement (100 runs)
- Graphs: All use 10pt font, 300dpi, Blues cmap, titles with exact model names (H-CoAtNet, GFT, CoAtNet, ViT, Swin, CNN, EfficientNet-B0) — no A* in filenames/titles
"""
