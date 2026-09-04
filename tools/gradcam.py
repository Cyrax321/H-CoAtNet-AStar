#!/usr/bin/env python3
"""gradcam.py -- Representative Grad-CAM for R1-11 interpretability.
Method: Grad-CAM on HCoAtNet cnn_stage4 (7x7), no test label used for map (pred class).
Usage:
  python H-CoAtNet/tools/gradcam.py --weights results/best_hcoatnet.pth --dataset_dir /path/to/ich-s-1/test --results results/results_hcoatnet.json --out figures/gradcam --n 6
Output: gradcam_<idx>_true<T>_pred<P>.png + method.txt (layer, norm, threshold, no expert rating unless added).
Safe: CPU-only, try/except per image, never breaks training.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def load_model(weights, num_classes=5):
    import sys
    sys.path.insert(0, "H-CoAtNet/proposed_method")
    from train_h_coatnet import HCoAtNet
    m = HCoAtNet(num_classes=num_classes, pretrained=False)
    sd = torch.load(weights, map_location="cpu")
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def gradcam_map(model, img_tensor, target_layer="cnn_stage4"):
    feats, grads = {}, {}
    layer = dict(model.named_children())[target_layer] if target_layer in dict(model.named_children()) else model.cnn_stage4
    def fwd_hook(_, __, out): feats["v"] = out.detach()
    def bwd_hook(_, gi, go): grads["v"] = go[0].detach()
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook) if hasattr(layer, "register_full_backward_hook") else layer.register_backward_hook(bwd_hook)
    out = model(img_tensor)
    pred = int(out.argmax(1).item())
    model.zero_grad()
    out[0, pred].backward()
    h1.remove(); h2.remove()
    F = feats["v"][0]  # C,7,7
    G = grads["v"][0]  # C,7,7
    w = G.mean(dim=(1,2))  # C
    cam = torch.relu((w[:, None, None] * F).sum(0)).numpy()  # 7,7
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam, pred, torch.softmax(out, dim=1).detach().numpy()[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset_dir", required=True, help="test/ folder with class subfolders")
    ap.add_argument("--results", required=True, help="results_hcoatnet.json with y_true/y_pred")
    ap.add_argument("--out", default="figures/gradcam")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "method.txt").write_text("Grad-CAM, layer=cnn_stage4 (7x7), weights=channel-mean grad of pred logit, norm min-max, colormap jet alpha 0.45, no test label for map. Expert rating: pending blinded dermatologist (see manuscript placeholder).\n")
    data = json.loads(Path(a.results).read_text())
    yt = data["test"]["y_true"]; yp = data["test"]["y_pred"]
    classes = data.get("classes", [])
    # collect test files in order matching ImageFolder (sorted) — best effort, document if mismatch
    from torchvision.datasets import ImageFolder
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    try:
        ds = ImageFolder(a.dataset_dir, transform=tf)
    except Exception as e:
        print(f"[FAIL] dataset {e}"); return
    # pick n failures + n correct
    idx_fail = [i for i,(t,p) in enumerate(zip(yt,yp)) if t!=p][:a.n//2]
    idx_ok = [i for i,(t,p) in enumerate(zip(yt,yp)) if t==p][:a.n//2]
    model = load_model(a.weights, num_classes=len(classes) if classes else 5)
    inv_norm = transforms.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225], std=[1/0.229,1/0.224,1/0.225])
    done = 0
    for i in idx_fail + idx_ok:
        try:
            img_t, _ = ds[i]
            cam, pred, probs = gradcam_map(model, img_t.unsqueeze(0))
            base = inv_norm(img_t).clamp(0,1).permute(1,2,0).numpy()
            cam_up = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((224,224))) / 255.0
            plt.figure(figsize=(4,4))
            plt.imshow(base); plt.imshow(cam_up, cmap="jet", alpha=0.45)
            plt.title(f"idx{i} true{yt[i]} pred{yp[i]} p={probs[pred]:.2f}", fontsize=8)
            plt.axis("off"); plt.tight_layout()
            plt.savefig(out / f"gradcam_{i}_true{yt[i]}_pred{yp[i]}.png", dpi=200, bbox_inches="tight"); plt.close()
            done += 1
        except Exception as e:
            print(f"[SKIP] {i}: {e}")
    print(f"[OK] {done} maps -> {out}/ + method.txt")

if __name__ == "__main__":
    main()
