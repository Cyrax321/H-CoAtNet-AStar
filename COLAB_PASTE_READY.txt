# H-CoAtNet — FULL COLAB RUN (public, hardcoded key, exact graph names no A*, fair, saves ALL to Drive)
# Paste ENTIRE block as ONE cell after Restart (GPU T4, ~3.5 hrs overnight)
# Runtime -> Change runtime type -> T4 GPU -> Restart session -> Paste -> Run

# 1. CLEAN CLONE (fixes double H-CoAtNet-AStar/H-CoAtNet-AStar)
%cd /content
!rm -rf H-CoAtNet-AStar
!git clone https://github.com/Cyrax321/H-CoAtNet-AStar.git
%cd H-CoAtNet-AStar
!pwd
!ls -lh

# 2. INSTALL WITHOUT BREAKING COLAB PYTHON 3.13 (fixes Fatal site error)
# Do NOT use requirements.txt with torch==2.2.0
!pip install -q -r requirements-colab.txt
!pip install -q roboflow
!python -c "import torch; print('torch', torch.__version__)"
!python -c "import timm; print('timm', timm.__version__)"

# 3. DOWNLOAD DATASET (hardcoded, no userdata.get / Secret error)
from roboflow import Roboflow
rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
dataset = project.version(1).download("folder")
print("DATASET:", dataset.location)
import pathlib
pathlib.Path("/tmp/dataset_path.txt").write_text(dataset.location)

# 4. FREEZE 70/15/15 seed 42 + DEDUP (before training, TRIPOD-AI)
import pathlib as _p
DATASET = _p.Path("/tmp/dataset_path.txt").read_text().strip()
print("Freeze split from:", DATASET)
!python tools/freeze_split.py --dataset_dir "{DATASET}" --seed 42 --out splits/seed42_indices.json
!cat splits/test_per_class.csv
!python tools/dedup_audit.py --dataset_dir "{DATASET}" --out results/dedup_report.json
!cat results/dedup_report.json | head -n 60

# 5. TRAIN ALL 7 MODELS (fair: same frozen split, same 30ep, same Cosine, val selects, test held-out once)
# Each ~20 min, total ~3.5 hrs — exact graph names (H-CoAtNet / GFT / CoAtNet / Swin / ViT / CNN / EfficientNet-B0)
!python H-CoAtNet/proposed_method/train_h_coatnet.py
!python H-CoAtNet/baselines/train_gft.py
!python H-CoAtNet/baselines/train_coatnet.py
!python H-CoAtNet/baselines/train_swin.py
!python H-CoAtNet/baselines/train_vit.py
!python H-CoAtNet/baselines/train_cnn.py
!python H-CoAtNet/baselines/train_efficientnet.py

# 6. AFTER TRAINING: CI + FLOPS + TABLES (2 min, no GPU)
!python tools/bootstrap_ci.py --results results/results_hcoatnet.json --n_bootstrap 1000 --out results/metrics_hcoatnet_ci.json
!python tools/compute_flops.py --all --out results/efficiency.json
!cat results/efficiency.json
!python tools/generate_tables.py --all results/results_*.json --out results/tables.tex
!cat results/tables.tex
!python tools/stats_tests.py --all results/results_*.json --reference results/results_hcoatnet.json --out results/significance.json
!cat results/significance.json

# 7. LIST ALL GRAPHS WITH EXACT NAMES (fair, no A*, 300dpi Blues)
!echo "=== RESULTS JSONS ==="
!ls -1 results/*.json
!echo "=== GRAPHS (exact names) ==="
!ls -1 results/*.png 2>/dev/null
!ls -1 H-CoAtNet/**/*.png 2>/dev/null | head -n 20
!echo "=== CURVES (train vs val only, no test leakage) ==="
!ls -1 results/*curves.png 2>/dev/null

# 8. SAVE EVERYTHING TO DRIVE — ALL GRAPHS AS SEPARATE PER-MODEL FOLDERS (fair, exact names)
from google.colab import drive
drive.mount('/content/drive')
import datetime, pathlib, shutil
DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")
DRIVE_OUT = pathlib.Path(f"/content/drive/MyDrive/HCoAtNet_AStar_{DATE}")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print(f"Saving to {DRIVE_OUT}")
# Base
!mkdir -p "{DRIVE_OUT}/results" "{DRIVE_OUT}/splits" "{DRIVE_OUT}/weights"
!cp -r results/* "{DRIVE_OUT}/results/" 2>/dev/null
!cp -r splits/* "{DRIVE_OUT}/splits/" 2>/dev/null
!cp results/tables.tex "{DRIVE_OUT}/results/" 2>/dev/null
# Per-model separate folders for diagrams (as you asked)
import pathlib as _pp
for model in ["H-CoAtNet", "GFT", "CoAtNet", "Swin", "ViT", "CNN", "EfficientNet-B0"]:
    (_pp.Path(DRIVE_OUT) / "diagrams" / model).mkdir(parents=True, exist_ok=True)
# Copy each model's PNGs to its own folder (exact names, no A*)
!cp -n results/confusion_matrix_hcoatnet*.png "{DRIVE_OUT}/diagrams/H-CoAtNet/" 2>/dev/null; cp -n results/hcoatnet*.png "{DRIVE_OUT}/diagrams/H-CoAtNet/" 2>/dev/null
!cp -n results/confusion_matrix_gft*.png "{DRIVE_OUT}/diagrams/GFT/" 2>/dev/null; cp -n results/gft*.png "{DRIVE_OUT}/diagrams/GFT/" 2>/dev/null
!cp -n results/confusion_matrix*coatnet*.png "{DRIVE_OUT}/diagrams/CoAtNet/" 2>/dev/null; cp -n results/coatnet*.png "{DRIVE_OUT}/diagrams/CoAtNet/" 2>/dev/null
!cp -n results/confusion_matrix*swin*.png "{DRIVE_OUT}/diagrams/Swin/" 2>/dev/null; cp -n results/swin*.png "{DRIVE_OUT}/diagrams/Swin/" 2>/dev/null
!cp -n results/confusion_matrix*vit*.png "{DRIVE_OUT}/diagrams/ViT/" 2>/dev/null; cp -n results/vit*.png "{DRIVE_OUT}/diagrams/ViT/" 2>/dev/null
!cp -n results/confusion_matrix*cnn*.png "{DRIVE_OUT}/diagrams/CNN/" 2>/dev/null; cp -n results/cnn*.png "{DRIVE_OUT}/diagrams/CNN/" 2>/dev/null
!cp -n results/confusion_matrix*efficient*.png "{DRIVE_OUT}/diagrams/EfficientNet-B0/" 2>/dev/null; cp -n results/efficient*.png "{DRIVE_OUT}/diagrams/EfficientNet-B0/" 2>/dev/null
# Also catch any legacy H-CoAtNet/**/*.png (fair, exact names)
!find H-CoAtNet -maxdepth 3 -name "*.png" -exec cp -n {} "{DRIVE_OUT}/diagrams/H-CoAtNet/" \; 2>/dev/null
# Weights per model
!find H-CoAtNet -name "*.pth" -exec cp -n {} "{DRIVE_OUT}/weights/" \; 2>/dev/null
!find . -maxdepth 2 -name "*.pth" -exec cp -n {} "{DRIVE_OUT}/weights/" \; 2>/dev/null
print("\n=== SAVED TO DRIVE — SEPARATE FOLDERS PER MODEL ===")
!ls -R "{DRIVE_OUT}/diagrams" 2>/dev/null
!echo "--- All results ---"; ls -lh "{DRIVE_OUT}/results" | head -n 60
!echo "--- Weights ---"; ls -lh "{DRIVE_OUT}/weights" | head -n 20
print(f"\nDone: {DRIVE_OUT}")
print("Check Drive: MyDrive -> HCoAtNet_AStar_... -> diagrams/H-CoAtNet/, diagrams/GFT/, ... -> each has its confusion + curves (exact names, fair)")
