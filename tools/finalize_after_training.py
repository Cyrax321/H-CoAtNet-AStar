#!/usr/bin/env python3
"""Run ONCE after all 7 trainings. Builds cross-model figs from real results_*.json."""
import subprocess, sys
cmds = [
  [sys.executable, "tools/generate_tables.py", "--all", "results/results_hcoatnet.json", "results/results_gft.json", "results/results_coatnet.json", "results/results_swin.json", "results/results_vit.json", "results/results_cnn.json", "results/results_efficientnetb0.json"],
  [sys.executable, "tools/bootstrap_ci.py", "--results", "results/results_hcoatnet.json"],
  [sys.executable, "tools/stats_tests.py", "--all", "results/results_hcoatnet.json", "results/results_gft.json", "results/results_coatnet.json", "results/results_swin.json", "results/results_vit.json", "results/results_cnn.json", "results/results_efficientnetb0.json", "--reference", "results/results_hcoatnet.json"],
  [sys.executable, "H-CoAtNet/tools/compute_flops.py", "--all"],
  [sys.executable, "tools/generate_figures.py"],
]
# Grad-CAM auto (best-effort, never fails finalize): needs weights + test dir + results json
import os
_grad_weights = "results/best_hcoatnet.pth"
_grad_results = "results/results_hcoatnet.json"
# Try common Roboflow test locations; skip gracefully if absent
_grad_test_candidates = [os.environ.get("TEST_DIR", ""), "ich-s-1/test", "ich-s-7lnsj/test", "/content/H-CoAtNet-AStar/ich-s-1/test", "/content/H-CoAtNet-Rebuttal-Clean/ich-s-1/test"]
import glob as _glob
for _g in _glob.glob("*/test") + _glob.glob("/content/*/ich-s-*/test") + _glob.glob("ich-s-*/test"):
    if _g not in _grad_test_candidates: _grad_test_candidates.append(_g)
_grad_test = next((p for p in _grad_test_candidates if p and os.path.isdir(p)), "")
if os.path.exists(_grad_weights) and os.path.exists(_grad_results) and _grad_test:
    cmds.append([sys.executable, "H-CoAtNet/tools/gradcam.py", "--weights", _grad_weights, "--dataset_dir", _grad_test, "--results", _grad_results, "--out", "figures/gradcam", "--n", "6"])
else:
    print(f"[SKIP] Grad-CAM auto (need {_grad_weights}, {_grad_results}, test dir). Run manually: python H-CoAtNet/tools/gradcam.py --weights {_grad_weights} --dataset_dir <test/> --results {_grad_results}")
for c in cmds:
    print(">>", " ".join(c))
    r = subprocess.run(c)
    if r.returncode != 0:
        print(f"WARN {c[1]} exited {r.returncode}, continuing")
# Colab: inline show + Drive backup (best-effort)
try:
    sys.path.insert(0, "tools")
    from colab_show_and_save import show_images, backup_to_drive
    show_images()
    backup_to_drive()
except Exception as e:
    print(f"[SKIP] show/backup: {e}")
print("[OK] finalize done")
