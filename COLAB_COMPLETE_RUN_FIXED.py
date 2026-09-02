#!/usr/bin/env python3
"""
COLAB_COMPLETE_RUN_FIXED.py — Fixed for Colab Python 3.13 + double-clone bug
- Fixes: Fatal Python error (torch 2.2.0 pin broke Colab 3.13 site)
- Fixes: SecretNotFoundError (use hardcoded key, not userdata.get)
- Fixes: Double clone H-CoAtNet-AStar/H-CoAtNet-AStar
Paste into FRESH runtime (Runtime -> Restart session) then run.
"""

# ========== CELL 1 — Clean clone (handles double-nested bug) ==========
"""
# Run in FRESH runtime (Restart first!)
%cd /content
!rm -rf H-CoAtNet-AStar  # remove old double-nested if exists
!git clone https://github.com/Cyrax321/H-CoAtNet-AStar.git
%cd H-CoAtNet-AStar
!pwd
!ls -lh
# Should be /content/H-CoAtNet-AStar, NOT /content/H-CoAtNet-AStar/H-CoAtNet-AStar
"""

# ========== CELL 2 — Install WITHOUT breaking Colab Python 3.13 ==========
"""
# DO NOT use requirements.txt with torch==2.2.0 on Colab 3.13 (breaks site)
# Use Colab's preinstalled torch + install only needed extras

!pip install -q timm torchinfo thop scikit-learn matplotlib seaborn pillow tqdm roboflow ImageHash scikit-image grad-cam statsmodels scipy -q
!pip show torch | grep Version
!python -c "import torch; print(torch.__version__)"
"""

# ========== CELL 3 — Download dataset (HARDCODED, no secrets) ==========
"""
from roboflow import Roboflow
rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")  # hardcoded, no userdata.get
project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
dataset = project.version(1).download("folder")
print("DATASET:", dataset.location)
import pathlib
pathlib.Path("/tmp/dataset_path.txt").write_text(dataset.location)
"""

# ========== CELL 4 — Freeze + Dedup ==========
"""
import pathlib as _p
DATASET = _p.Path("/tmp/dataset_path.txt").read_text().strip()
print(DATASET)
!python tools/freeze_split.py --dataset_dir "{DATASET}" --seed 42 --out splits/seed42_indices.json
!cat splits/test_per_class.csv
!python tools/dedup_audit.py --dataset_dir "{DATASET}" --out results/dedup_report.json
"""

# ========== CELL 5 — Train (same, fair) ==========
"""
!python H-CoAtNet/proposed_method/train_h_coatnet.py
!python H-CoAtNet/baselines/train_gft.py
!python H-CoAtNet/baselines/train_coatnet.py
!python H-CoAtNet/baselines/train_swin.py
!python H-CoAtNet/baselines/train_vit.py
!python H-CoAtNet/baselines/train_cnn.py
!python H-CoAtNet/baselines/train_efficientnet.py
"""

# ========== CELL 6 — CI + FLOPs + Tables ==========
"""
!python tools/bootstrap_ci.py --results results/results_hcoatnet.json --n_bootstrap 1000 --out results/metrics_hcoatnet_ci.json
!python tools/compute_flops.py --all --out results/efficiency.json; cat results/efficiency.json
!python tools/generate_tables.py --all results/results_*.json --out results/tables.tex; cat results/tables.tex
!python tools/stats_tests.py --all results/results_*.json --reference results/results_hcoatnet.json --out results/significance.json; cat results/significance.json
!ls -lh results/*.png results/*.json
"""
