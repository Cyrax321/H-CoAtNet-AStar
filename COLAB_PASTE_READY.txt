# H-CoAtNet — FULL COLAB RUN (public, hardcoded key, exact graph names no A*, fair, saves ALL to Drive per-model)
# Paste ENTIRE block as ONE cell after Restart (GPU T4, ~3.5 hrs overnight)
# Runtime -> Change runtime type -> T4 GPU -> Restart session -> Paste -> Run

# 1. CLEAN CLONE — FIXED for Colab (uses zip, no git Username prompt, works for PUBLIC)
%cd /content
!rm -rf H-CoAtNet-AStar H-CoAtNet-AStar-main /tmp/repo.zip
!wget -q https://github.com/Cyrax321/H-CoAtNet-AStar/archive/refs/heads/main.zip -O /tmp/repo.zip
!unzip -q /tmp/repo.zip -d /tmp
!mv /tmp/H-CoAtNet-AStar-main /content/H-CoAtNet-AStar
%cd H-CoAtNet-AStar
!pwd
!ls -lh

# 2. INSTALL WITHOUT BREAKING COLAB PYTHON 3.13
!pip install -q -r requirements-colab.txt
!pip install -q roboflow
!python -c "import torch; print('torch', torch.__version__)"

# 3. DOWNLOAD DATASET (hardcoded)
from roboflow import Roboflow
rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
dataset = project.version(1).download("folder")
print("DATASET:", dataset.location)
import pathlib
pathlib.Path("/tmp/dataset_path.txt").write_text(dataset.location)

# 4. FREEZE 70/15/15 seed 42 + DEDUP
import pathlib as _p
DATASET = _p.Path("/tmp/dataset_path.txt").read_text().strip()
!python tools/freeze_split.py --dataset_dir "{DATASET}" --seed 42 --out splits/seed42_indices.json
!cat splits/test_per_class.csv
!python tools/dedup_audit.py --dataset_dir "{DATASET}" --out results/dedup_report.json

# 5. TRAIN ALL 7 MODELS (fair, exact names, ~20 min each)
!python H-CoAtNet/proposed_method/train_h_coatnet.py
!python H-CoAtNet/baselines/train_gft.py
!python H-CoAtNet/baselines/train_coatnet.py
!python H-CoAtNet/baselines/train_swin.py
!python H-CoAtNet/baselines/train_vit.py
!python H-CoAtNet/baselines/train_cnn.py
!python H-CoAtNet/baselines/train_efficientnet.py

# 6. CI + FLOPS + TABLES
!python tools/bootstrap_ci.py --results results/results_hcoatnet.json --n_bootstrap 1000 --out results/metrics_hcoatnet_ci.json
!python tools/compute_flops.py --all --out results/efficiency.json; cat results/efficiency.json
!python tools/generate_tables.py --all results/results_*.json --out results/tables.tex; cat results/tables.tex
!python tools/stats_tests.py --all results/results_*.json --reference results/results_hcoatnet.json --out results/significance.json; cat results/significance.json

# 7. SAVE EVERYTHING TO DRIVE — SEPARATE PER-MODEL FOLDERS
from google.colab import drive
drive.mount('/content/drive')
import datetime, pathlib
DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")
DRIVE_OUT = pathlib.Path(f"/content/drive/MyDrive/HCoAtNet_AStar_{DATE}")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print(f"Saving to {DRIVE_OUT}")
!mkdir -p "{DRIVE_OUT}/results" "{DRIVE_OUT}/splits" "{DRIVE_OUT}/weights"
!cp -r results/* "{DRIVE_OUT}/results/" 2>/dev/null
!cp -r splits/* "{DRIVE_OUT}/splits/" 2>/dev/null
for model in ["H-CoAtNet","GFT","CoAtNet","Swin","ViT","CNN","EfficientNet-B0"]:
    (pathlib.Path(DRIVE_OUT)/"diagrams"/model).mkdir(parents=True, exist_ok=True)
!cp -n results/confusion_matrix_hcoatnet*.png "{DRIVE_OUT}/diagrams/H-CoAtNet/" 2>/dev/null; cp -n results/hcoatnet*.png "{DRIVE_OUT}/diagrams/H-CoAtNet/" 2>/dev/null
!cp -n results/confusion_matrix_gft*.png "{DRIVE_OUT}/diagrams/GFT/" 2>/dev/null; cp -n results/gft*.png "{DRIVE_OUT}/diagrams/GFT/" 2>/dev/null
!cp -n results/confusion_matrix*coatnet*.png "{DRIVE_OUT}/diagrams/CoAtNet/" 2>/dev/null; cp -n results/coatnet*.png "{DRIVE_OUT}/diagrams/CoAtNet/" 2>/dev/null
!cp -n results/confusion_matrix*swin*.png "{DRIVE_OUT}/diagrams/Swin/" 2>/dev/null; cp -n results/swin*.png "{DRIVE_OUT}/diagrams/Swin/" 2>/dev/null
!cp -n results/confusion_matrix*vit*.png "{DRIVE_OUT}/diagrams/ViT/" 2>/dev/null; cp -n results/vit*.png "{DRIVE_OUT}/diagrams/ViT/" 2>/dev/null
!cp -n results/confusion_matrix*cnn*.png "{DRIVE_OUT}/diagrams/CNN/" 2>/dev/null; cp -n results/cnn*.png "{DRIVE_OUT}/diagrams/CNN/" 2>/dev/null
!cp -n results/confusion_matrix*efficient*.png "{DRIVE_OUT}/diagrams/EfficientNet-B0/" 2>/dev/null; cp -n results/efficient*.png "{DRIVE_OUT}/diagrams/EfficientNet-B0/" 2>/dev/null
!find H-CoAtNet -name "*.png" -exec cp -n {} "{DRIVE_OUT}/diagrams/H-CoAtNet/" \; 2>/dev/null
!find H-CoAtNet -name "*.pth" -exec cp -n {} "{DRIVE_OUT}/weights/" \; 2>/dev/null
!find . -maxdepth 2 -name "*.pth" -exec cp -n {} "{DRIVE_OUT}/weights/" \; 2>/dev/null
print("\n=== SAVED TO DRIVE — SEPARATE FOLDERS ===")
!ls -R "{DRIVE_OUT}/diagrams" 2>/dev/null
!ls -lh "{DRIVE_OUT}/results" | head -n 30
print(f"\nDone: {DRIVE_OUT} — sleep now, check Drive tomorrow: diagrams/H-CoAtNet/, GFT/, ...")
