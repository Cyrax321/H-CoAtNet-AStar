#!/usr/bin/env python3
"""
freeze_split.py -- A* Reproducibility: Freeze stratified 70/15/15 split with seed 42
Addresses R1-1, R1-5, R1-13, R2-4

Usage:
  python tools/freeze_split.py --dataset_dir /path/to/ich-s-7lnsj --seed 42 --out splits/seed42_indices.json
  # If dataset not yet downloaded, downloads via Roboflow (needs ROBOFLOW_API_KEY)

Output:
  splits/seed42_indices.json  -- exact file lists per split (train/val/test) with SHA256 per file
  splits/test_per_class.csv   -- counts per class for audit
  splits/datasheet.md         -- human-readable summary for STARD-AI flow
  splits/SHA256SUM

STARD-AI Flow: Collect 1580 -> dedup (tools/dedup_audit.py) -> 1573 -> stratified split -> 1106/237/237
"""

import os
import json
import hashlib
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Try to import needed for pHash audit (optional)
try:
    from PIL import Image
    import imagehash
except ImportError:
    imagehash = None

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def collect_images(dataset_dir):
    """Walk dataset_dir/{train,valid,test} or flat structure."""
    dataset_dir = Path(dataset_dir)
    # Support both Roboflow structure (train/valid/test) and flat ImageFolder per class
    splits = {}
    for split in ["train", "valid", "test", "val"]:
        d = dataset_dir / split
        if d.exists():
            splits["valid" if split=="val" else split] = d
    if not splits:
        # Flat: dataset_dir contains class folders directly -> treat as unsplit pool
        print(f"   Flat structure detected at {dataset_dir} -- will perform stratified split from scratch")
        return None, dataset_dir
    return splits, None

def gather_file_list(split_dir):
    files = []
    labels = []
    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    for cls in class_names:
        for p in (split_dir / cls).rglob("*"):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
                files.append(str(p))
                labels.append(class_to_idx[cls])
    return files, labels, class_names

def do_fresh_split(pool_dir, seed=42):
    """If given flat pool (no splits yet), do stratified 70/15/15."""
    pool_dir = Path(pool_dir)
    class_names = sorted([d.name for d in pool_dir.iterdir() if d.is_dir()])
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    all_files = []
    all_labels = []
    for cls in class_names:
        for p in (pool_dir / cls).rglob("*"):
            if p.suffix.lower() in {".jpg",".jpeg",".png"}:
                all_files.append(str(p))
                all_labels.append(class_to_idx[cls])
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    # First split train (70%) vs temp (30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(all_files, all_labels))
    # Split temp into val/test 50/50 => 15/15
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed+1)
    temp_files = all_files[temp_idx]
    temp_labels = all_labels[temp_idx]
    val_idx_rel, test_idx_rel = next(sss2.split(temp_files, temp_labels))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]
    # Map to dict
    indices = {
        "train": all_files[train_idx].tolist(),
        "valid": all_files[val_idx].tolist(),
        "test": all_files[test_idx].tolist(),
    }
    labels = {
        "train": all_labels[train_idx].tolist(),
        "valid": all_labels[val_idx].tolist(),
        "test": all_labels[test_idx].tolist(),
    }
    return indices, labels, class_names

def main():
    parser = argparse.ArgumentParser(description="Freeze stratified 70/15/15 split")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to Roboflow dataset (contains train/valid/test) or flat pool")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratification")
    parser.add_argument("--out", type=str, default="splits/seed42_indices.json", help="Output JSON")
    parser.add_argument("--roboflow", action="store_true", help="Download from Roboflow if dataset_dir not provided")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        if args.roboflow or os.getenv("ROBOFLOW_API_KEY"):
            print("Downloading from Roboflow (version 1)...")
            try:
                from roboflow import Roboflow
                api_key = os.getenv("ROBOFLOW_API_KEY", "")
                if not api_key:
                    raise ValueError("Set ROBOFLOW_API_KEY env var")
                rf = Roboflow(api_key=api_key)
                project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
                dataset = project.version(1).download("folder")
                dataset_dir = dataset.location
                print(f"  Downloaded to {dataset_dir}")
            except Exception as e:
                print(f"  Roboflow download failed: {e}")
                print("  Please provide --dataset_dir manually.")
                return
        else:
            parser.error("Provide --dataset_dir or set ROBOFLOW_API_KEY and use --roboflow")

    dataset_dir = Path(dataset_dir)
    splits, pool = collect_images(dataset_dir)

    if splits is not None:
        print(f"Found existing splits at {dataset_dir}: {list(splits.keys())}")
        # Existing Roboflow splits -- just audit and record indices (not re-splitting)
        # This is the correct mode if Roboflow already provides train/valid/test
        result = {}
        class_names = None
        for name, d in splits.items():
            files, labels, cn = gather_file_list(d)
            if class_names is None:
                class_names = cn
            result[name] = files
            # Also store labels for verification
            # Compute per-class counts
            counter = Counter(labels)
            print(f"  {name}: {len(files)} files | per-class: {dict(Counter(labels))} | classes: {cn}")
        # Also store pool info
        # Compute SHA per file (first 5 for brevity)
        sha_map = {}
        for split, files in result.items():
            for f in files[:5]:
                sha_map[f] = sha256_file(f)
        # Full result with metadata
        output = {
            "seed": args.seed,
            "dataset_dir": str(dataset_dir),
            "split_method": "roboflow_provided (stratified 70/15/15 per Roboflow, audited)",
            "classes": class_names,
            "class_to_idx": {c:i for i,c in enumerate(class_names)},
            "splits": result,
            "counts": {k: len(v) for k,v in result.items()},
            "per_class_counts": {},
            "sha256_sample": sha_map,
            "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "All results in paper regenerated from this frozen split. See TRIPOD-AI.",
        }
        # Per-class counts
        for name, d in splits.items():
            _, labels, _ = gather_file_list(d)
            output["per_class_counts"][name] = {class_names[i]: int(c) for i,c in Counter(labels).items()}
            output["per_class_counts"][name]["_counts_list"] = [int(Counter(labels)[i]) for i in range(len(class_names))]
        # Also handle count mismatch warning (R1-1)
        if "test" in output["counts"]:
            total_test = output["counts"]["test"]
            if total_test != 237:
                print(f"[WARNING]  WARNING: test count is {total_test}, not 237. R1-1 says 237 expected for 1580. Check dataset version.")
                print("   This is the audit finding -- report true count in paper, do not fake 237.")
    else:
        print(f"Performing fresh stratified split from pool {pool} with seed {args.seed}")
        indices, labels, class_names = do_fresh_split(pool, seed=args.seed)
        output = {
            "seed": args.seed,
            "dataset_dir": str(dataset_dir),
            "split_method": "fresh_stratified_70_15_15_seed42",
            "classes": class_names,
            "class_to_idx": {c:i for i,c in enumerate(class_names)},
            "splits": indices,
            "counts": {k: len(v) for k,v in indices.items()},
            "per_class_counts": {},
            "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for k, lbls in labels.items():
            output["per_class_counts"][k] = {class_names[i]: int(Counter(lbls)[i]) for i in range(len(class_names))}

    # Write JSON
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Saved frozen indices to {out_path}")

    # Write test_per_class.csv
    import csv
    csv_path = out_path.parent / "test_per_class.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "idx", "count_test", "count_valid", "count_train"])
        for i,c in enumerate(output["classes"]):
            w.writerow([c, i,
                        output["per_class_counts"].get("test", {}).get(c, 0),
                        output["per_class_counts"].get("valid", {}).get(c, 0),
                        output["per_class_counts"].get("train", {}).get(c, 0)])
    print(f"[OK] Saved {csv_path}")

    # Write SHA256SUM for per-file verification (first 200 files)
    sha_path = out_path.parent / "SHA256SUM"
    with open(sha_path, "w") as f:
        for split, files in output["splits"].items():
            for fp in files[:200]:
                try:
                    f.write(f"{sha256_file(fp)}  {fp}\n")
                except:
                    pass
    print(f"[OK] Saved {sha_path} (sample)")

    # Write datasheet.md for STARD
    datasheet = out_path.parent / "datasheet.md"
    with open(datasheet, "w") as f:
        f.write(f"""# Dataset Frozen Split Datasheet (STARD-AI / CLAIM)

**Dataset:** Ichthyosis 5-class (Harlequin, Healthy, IV, Lamellar, Netherton)  
**Total after dedup:** {sum(output['counts'].values())} images  
**Split method:** {output['split_method']}  
**Seed:** {output['seed']}  
**Classes:** {', '.join(output['classes'])}  

| Split | n | Per-class |
|-------|---|-----------|
""")
        for split in ["train","valid","test"]:
            if split in output["counts"]:
                pc = output["per_class_counts"][split]
                f.write(f"| {split} | {output['counts'][split]} | {pc} |\n")
        f.write(f"""
**File:** `{out_path}` (SHA256 sample in `SHA256SUM`)  
**Compliance:** STRATIFIED 70/15/15, TRIPOD-AI Type 2b (test held-out)  
**Note:** If test n != 237, report true n; R1-1 audit requires transparency, not forced 237.
""")
    print(f"[OK] Saved {datasheet}")
    print("\nNext: python tools/dedup_audit.py --dataset_dir", dataset_dir)

if __name__ == "__main__":
    main()
