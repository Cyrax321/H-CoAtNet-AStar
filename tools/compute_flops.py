#!/usr/bin/env python3
"""
compute_flops.py -- A* Efficiency Table: Params, MACs/FLOPs, Latency, Throughput, Peak Mem
Addresses R1-9

Usage:
  python tools/compute_flops.py --model hcoatnet --batch 1
  python tools/compute_flops.py --all  # all 7 models

Measures on same HW/SW (report your GPU/CPU). Uses thop for MACs, torch for params, time for latency.
Requires: pip install thop torchinfo timm
"""

import argparse
import time
import json
from pathlib import Path
import torch
import torch.nn as nn

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_size=(1,3,224,224), device="cpu", runs=100, warmup=20):
    model.eval()
    x = torch.randn(*input_size, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device != "cpu":
                torch.cuda.synchronize()
    # Measure
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)
            if device != "cpu":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / runs * 1000
    throughput = runs / elapsed
    peak_mem = 0
    if device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    return avg_ms, throughput, peak_mem

def get_model(name, num_classes=5):
    name = name.lower()
    if name in ["hcoatnet", "h-coatnet", "h_coatnet"]:
        import sys
        sys.path.insert(0, "H-CoAtNet/proposed_method")
        from train_h_coatnet import HCoAtNet
        return HCoAtNet(num_classes=num_classes, pretrained=False)
    elif name == "gft":
        import sys
        sys.path.insert(0, "H-CoAtNet/baselines")
        from train_gft import GFT
        return GFT(num_classes=num_classes, pretrained=False)
    elif name == "coatnet":
        from timm import create_model
        return create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
    elif name == "vit":
        from timm import create_model
        return create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    elif name == "swin":
        import sys
        sys.path.insert(0, "H-CoAtNet/baselines")
        from train_swin import SwinTransformer
        return SwinTransformer(num_classes=num_classes)
    elif name == "efficientnet":
        from timm import create_model
        return create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    elif name == "cnn":
        import sys
        sys.path.insert(0, "H-CoAtNet/baselines")
        from train_cnn import BaselineCNN
        return BaselineCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {name}")

def compute_for_model(name, device, batch_size=1):
    print(f"\n{'='*60}\nModel: {name} | Input {batch_size}x3x224x224 | Device {device}")
    model = get_model(name).to(device)
    model.eval()
    params = count_params(model)
    print(f"  Params: {params/1e6:.2f} M ({params})")
    # FLOPs via thop
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(torch.randn(batch_size, 3, 224, 224, device=device),), verbose=False)
        print(f"  MACs: {macs/1e9:.2f} G | FLOPs (2*MACs): {macs*2/1e9:.2f} G")
        flops_g = macs*2/1e9
        macs_g = macs/1e9
    except ImportError:
        print("  thop not installed: pip install thop")
        macs_g = None; flops_g = None
    except Exception as e:
        print(f"  thop failed: {e}")
        macs_g = None; flops_g = None
    # torchinfo
    try:
        from torchinfo import summary
        s = summary(model, input_size=(batch_size, 3, 224, 224), verbose=0, device=device)
        print(f"  torchinfo params: {s.total_params/1e6:.2f} M")
    except:
        pass
    # Latency
    try:
        avg_ms, thr, peak = measure_latency(model, (batch_size,3,224,224), device, runs=100)
        print(f"  Latency: {avg_ms:.2f} ms (avg over 100 runs)")
        print(f"  Throughput: {thr:.1f} img/s")
        print(f"  Peak mem: {peak:.2f} GB")
    except Exception as e:
        print(f"  Latency failed: {e}")
        avg_ms, thr, peak = None, None, None
    return {
        "model": name,
        "params_M": round(params/1e6, 2),
        "params": int(params),
        "macs_G": round(macs_g, 2) if macs_g else None,
        "flops_G": round(flops_g, 2) if flops_g else None,
        "latency_ms_b1" if batch_size==1 else f"latency_ms_b{batch_size}": round(avg_ms,2) if avg_ms else None,
        "throughput_img_s": round(thr,1) if thr else None,
        "peak_mem_GB": round(peak,2) if peak else None,
        "input": f"{batch_size}x3x224x224",
        "device": str(device),
    }

def main():
    parser = argparse.ArgumentParser(description="Compute FLOPs/Params/Latency")
    parser.add_argument("--model", type=str, default="hcoatnet", choices=["hcoatnet","gft","coatnet","vit","swin","efficientnet","cnn"])
    parser.add_argument("--all", action="store_true", help="All 7 models")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for latency (1 and 32 recommended)")
    parser.add_argument("--out", type=str, default="results/efficiency.json", help="Output JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | PyTorch {torch.__version__}")

    results = []
    if args.all:
        for m in ["hcoatnet","gft","coatnet","vit","swin","efficientnet","cnn"]:
            try:
                results.append(compute_for_model(m, str(device), batch_size=1))
                # Also batch 32 for throughput
                results.append(compute_for_model(m, str(device), batch_size=32))
            except Exception as e:
                print(f"Failed for {m}: {e}")
                import traceback; traceback.print_exc()
    else:
        results.append(compute_for_model(args.model, str(device), batch_size=args.batch))

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"device": str(device), "pytorch": torch.__version__, "models": results, "input": "224x224", "note": "FLOPs via thop, latency mean 100 runs, same HW/SW for all models. Report both b1 and b32."}, f, indent=2)
    print(f"\n[OK] Saved to {out}")
    # Print LaTeX table
    print("\n[LaTeX] LaTeX Table 4 snippet:")
    print("\\begin{tabular}{lccccl}")
    print("\\toprule Model & Params (M) & MACs (G) & Latency b1 (ms) & Throughput & Acc \\\\")
    for r in results:
        if r.get("input","").startswith("1x"):
            print(f"{r['model']} & {r['params_M']} & {r['macs_G']} & {r.get('latency_ms_b1','-')} & {r.get('throughput_img_s','-')} & TBD \\\\")
    print("\\bottomrule")

if __name__ == "__main__":
    main()
