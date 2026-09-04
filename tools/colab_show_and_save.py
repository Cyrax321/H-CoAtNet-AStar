#!/usr/bin/env python3
"""Colab inline display + Drive backup. Best-effort, never fails finalize."""
from pathlib import Path

def show_images(dirs=("results", "figures/gradcam", "figures"), patterns=("confusion*.png", "roc_*.png", "pr_*.png", "reliability_*.png", "gradcam_*.png", "fig*.png"), max_n=24):
    try:
        from IPython.display import Image, display
    except Exception:
        print("[SKIP] display: not in IPython"); return
    shown = 0
    for d in dirs:
        for pat in patterns:
            for p in sorted(Path(d).glob(pat)):
                if shown >= max_n: return
                try:
                    print(f"--- {p} ---")
                    display(Image(str(p)))
                    shown += 1
                except Exception as e:
                    print(f"[SKIP] show {p}: {e}")

def backup_to_drive(src_dirs=("results", "figures", "splits"), drive_root="/content/drive/MyDrive/HCoAtNet_runs"):
    import shutil, time, os
    if not os.path.isdir("/content/drive"):
        print("[SKIP] Drive not mounted. Run: from google.colab import drive; drive.mount('/content/drive')")
        return ""
    dst = Path(drive_root) / time.strftime("%Y%m%d_%H%M")
    dst.mkdir(parents=True, exist_ok=True)
    for s in src_dirs:
        sp = Path(s)
        if sp.exists():
            try:
                shutil.copytree(sp, dst / sp.name, dirs_exist_ok=True)
                print(f"[OK] {sp} -> {dst/sp.name}")
            except Exception as e:
                print(f"[WARN] backup {sp}: {e}")
    print(f"[OK] Drive backup: {dst}")
    return str(dst)
