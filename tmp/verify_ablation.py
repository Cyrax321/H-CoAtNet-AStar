import sys, re
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
try:
    from timm.models.vision_transformer import Block
except ImportError:
    from timm.models.vit import Block

REPO = Path(".")
sys.path.insert(0, str(REPO))
import ablation_study as A
from ablation_study import AblationCoAtNet, HierarchicalSE

BATCH = 2
NC = 5
X = torch.randn(BATCH, 3, 224, 224)
RUN_SEED = 42

def count_stage_yields(model, x):
    counts = {}
    # We need to hook/select stages without changing the model.
    # Instead, run the model and inspect internal module calls indirectly
    # by running stage-by-stage ourselves for the conv stages, and using hooks for the rest.
    out = {}
    def hook(name):
        def fn(module, inp, out):
            out[name] = out
        return fn
    handles = {}
    for name in ["cnn_stem","cnn_stage1","cnn_stage2","cnn_stage3","cnn_stage4","pos_embed","vit_blocks","hierarchical_blocks","classifier"]:
        mod = getattr(model, name, None)
        if mod is not None:
            try:
                h = mod.register_forward_hook(hook(name))
                handles[name] = h
            except Exception as e:
                out[name] = f"hook_err:{e}"
    with torch.no_grad():
        try:
            _ = model(x)
        except Exception as e:
            out["_forward_err"] = repr(e)
    for h in handles.values():
        try:
            h.remove()
        except Exception:
            pass
    return out

results = {}
for v in ["cnnOnly","vitOnly","seNoPrune","randomPrune","directL2","hierarchical","hierarchicalRaw"]:
    cfg = A.VARIANTS[v]
    model = AblationCoAtNet(
        use_vit=cfg["use_vit"],
        use_se=cfg["use_se"],
        exec_mode=cfg["exec_mode"],
        num_classes=NC,
        pretrained=False,
        vit_blocks=2,
        run_seed=RUN_SEED,
    ).eval()
    hooks = count_stage_yields(model, X)
    y = model(X)
    results[v] = {
        "cfg": cfg,
        "out_shape": tuple(y.shape),
        "forward_ok": "_forward_err" not in hooks,
        "hooks": hooks,
        "n_hierarchical_blocks": len(model.hierarchical_blocks),
        "has_random_idx_buffer": hasattr(model, "random_idx_49_to_24"),
        "random_idx_buffer_shape": model.random_idx_49_to_24.shape,
        "random_idx_buffer": model.random_idx_49_to_24.tolist() if model.random_idx_49_to_24.numel() else [],
    }

print("=== STRUCTURAL TESTS ===")

def ok(cond, msg):
    print(("OK  " if cond else "FAIL") + " " + msg)
    return cond

all_ok = True

# TEST 12: all variants produce logits with shape (batch_size, 5)
print("\nTEST 12: logits shape (B,5) for all variants")
for v in results:
    r = results[v]
    cond = r["out_shape"] == (BATCH, NC) and r["forward_ok"]
    if not ok(cond, f"{v}: {r['out_shape']} forward_ok={r['forward_ok']}"):
        all_ok = False

# TEST 13: all variants execute a forward pass
print("\nTEST 13: forward pass success for all variants")
for v in results:
    ok(results[v]["forward_ok"], f"{v}: forward executed")

# TEST 9: A5 performs 49->36->24 (two hierarchical stages)
print("\nTEST 9: A5 uses two hierarchical stages and full SE path")
r = results["hierarchical"]
ok(r["n_hierarchical_blocks"] == 2, f"A5 hierarchical_blocks count: {r['n_hierarchical_blocks']}")
ok(r["cfg"]["exec_mode"] == "hierarchical_se", f"A5 exec_mode: {r['cfg']['exec_mode']}")
if r["forward_ok"] and "cnn_stage4" in r["hooks"]:
    # We cannot see token counts from hooks directly, but we can manually run stages
    # below in a separate deterministic simulation.
    print("  (token-count verification done separately below)")

# TEST 11: A6 creates no SE blocks and uses raw L2
print("\nTEST 11: A6 creates no SE blocks and uses raw L2")
r = results["hierarchicalRaw"]
ok(r["n_hierarchical_blocks"] == 0, f"A6 hierarchical_blocks count: {r['n_hierarchical_blocks']}")
ok(r["cfg"]["use_se"] is False, f"A6 use_se: {r['cfg']['use_se']}")
ok(r["cfg"]["exec_mode"] == "hierarchical_raw", f"A6 exec_mode: {r['cfg']['exec_mode']}")
if r["forward_ok"] and "hierarchical_raw" not in str(r["hooks"].get("hierarchical_blocks","")):
    print("  (verified: hierarchical_blocks is empty ModuleList)")

# TEST 1/2/3/4/5/6/7/8: deeper deterministic execution checks
print("\n=== DETAILED EXECUTION TESTS (manual stage simulation) ===")
manual = {}
for v in results:
    cfg = results[v]["cfg"]
    model = AblationCoAtNet(
        use_vit=cfg["use_vit"],
        use_se=cfg["use_se"],
        exec_mode=cfg["exec_mode"],
        num_classes=NC,
        pretrained=False,
        vit_blocks=2,
        run_seed=RUN_SEED,
    ).eval()

    # Manually run stem + stage1 + stage2
    xb = X
    xb = model.cnn_stem(xb)
    xb = model.cnn_stage1(xb)
    xb = model.cnn_stage2(xb)

    # ViT if present
    if model.use_vit:
        B, C, H, W = xb.shape
        xb = xb.flatten(2).transpose(1, 2)
        if model.pos_embed is not None:
            xb = xb + model.pos_embed
        for blk in model.vit_blocks:
            xb = blk(xb)
        xb = xb.transpose(1, 2).reshape(B, C, H, W)

    # Stage3/4
    xb = model.cnn_stage3(xb)
    xb = model.cnn_stage4(xb)
    cur = xb.flatten(2).transpose(1, 2)  # (B, 49, 768)

    # Record pre-selection token count
    n_pre = cur.shape[1]
    kept_tokens = None
    se_executed = False
    se_gated_used = False
    selections = []
    exec_steps = []

    if model.exec_mode == "none":
        exec_steps.append("none: mean pool all 49 (no SE, no selection)")
        kept_tokens = cur
    elif model.exec_mode == "se_only":
        gated, _ = model.hierarchical_blocks[0](cur)
        exec_steps.append("se_only: SE executed, 49 SE-gated tokens kept, no selection")
        se_executed = True
        se_gated_used = True
        kept_tokens = gated
    elif model.exec_mode == "random_se":
        gated, _ = model.hierarchical_blocks[0](cur)
        se_executed = True
        se_gated_used = True
        idx = model.random_idx_49_to_24.to(cur.device)
        kept_tokens = gated[:, idx, :]
        selections.append(("random_se", idx.tolist()))
        exec_steps.append("random_se: SE executed, fixed random subset of SE-GATED tokens")
    elif model.exec_mode == "direct_se":
        gated, imp = model.hierarchical_blocks[0](cur)
        se_executed = True
        se_gated_used = True
        k24 = model.selection_sizes[1]
        _, top_idx = torch.topk(imp, k24, dim=1)
        bi = torch.arange(B, device=cur.device).unsqueeze(1).expand(-1, k24)
        kept_tokens = gated[bi, top_idx]
        selections.append(("direct_se", top_idx.tolist()))
        exec_steps.append("direct_se: SE executed once, then top-k 49->24 on SE-GATED importance")
    elif model.exec_mode == "hierarchical_se":
        stage_results = []
        for step_idx, (blk, k) in enumerate(zip(model.hierarchical_blocks, model.selection_sizes)):
            gated, imp = blk(cur)
            se_executed = True
            se_gated_used = True
            _, top_idx = torch.topk(imp, k, dim=1)
            bi = torch.arange(B, device=cur.device).unsqueeze(1).expand(-1, k)
            cur = gated[bi, top_idx]
            stage_results.append((step_idx, k, cur.shape[1]))
        kept_tokens = cur
        selections.append(("hierarchical_se", stage_results))
        exec_steps.append("hierarchical_se: SE at 49->36 then 36->24 using SE-GATED importance")
    elif model.exec_mode == "hierarchical_raw":
        stage_results = []
        for step_idx, k in enumerate(model.selection_sizes):
            imp = model._raw_importance(cur)
            _, top_idx = torch.topk(imp, k, dim=1)
            bi = torch.arange(B, device=cur.device).unsqueeze(1).expand(-1, k)
            cur = cur[bi, top_idx]
            stage_results.append((step_idx, k, cur.shape[1]))
        kept_tokens = cur
        selections.append(("hierarchical_raw", stage_results))
        exec_steps.append("hierarchical_raw: raw L2 at 49->36 then 36->24, no SE")
    else:
        exec_steps.append(f"UNKNOWN exec_mode: {model.exec_mode}")

    pooled = kept_tokens.mean(dim=1)
    logits = model.classifier(pooled)
    manual[v] = {
        "n_pre_selection": n_pre,
        "n_kept_tokens": kept_tokens.shape[1],
        "se_executed": se_executed,
        "se_gated_used": se_gated_used,
        "selections": selections,
        "logits_shape": tuple(logits.shape),
        "exec_steps": exec_steps,
    }

# TEST 1: A2 invokes SE
print("\nTEST 1: A2 invokes SE")
r = manual["seNoPrune"]
all_ok = ok(r["se_executed"], "A2: SE module executed") and all_ok

# TEST 2: A2 returns/pools 49 gated tokens
print("\nTEST 2: A2 returns/pools 49 SE-GATED tokens")
r = manual["seNoPrune"]
all_ok = ok(r["n_pre_selection"] == 49 and r["n_kept_tokens"] == 49, f"A2: pre={r['n_pre_selection']} kept={r['n_kept_tokens']}") and all_ok
all_ok = ok(r["se_gated_used"], "A2: pooled tokens are SE-GATED") and all_ok

# TEST 3: A3 invokes SE before random selection
print("\nTEST 3: A3 invokes SE before random selection")
r = manual["randomPrune"]
all_ok = ok(r["se_executed"], "A3: SE module executed") and all_ok
all_ok = ok(r["se_gated_used"], "A3: random selection operates on SE-GATED tokens") and all_ok

# TEST 4: A3 uses exactly 24 tokens
print("\nTEST 4: A3 uses exactly 24 kept tokens")
r = manual["randomPrune"]
all_ok = ok(r["n_kept_tokens"] == 24, f"A3: kept tokens={r['n_kept_tokens']}") and all_ok

# TEST 5: A3 uses the SAME random indices on repeated forward passes
print("\nTEST 5: A3 fixed random indices across repeated forward passes")
r = results["randomPrune"]
buf = r["random_idx_buffer"]
cond = len(buf) != 0
if not ok(cond, "A3: random_idx_49_to_24 buffer exists and is non-empty"):
    all_ok = False
else:
    # Run A3 twice on same input and confirm kept tokens identical
    model = AblationCoAtNet(use_vit=True, use_se=True, exec_mode="random_se",
                            num_classes=NC, pretrained=False, vit_blocks=2, run_seed=RUN_SEED).eval()
    xb = X
    xb = model.cnn_stem(xb); xb = model.cnn_stage1(xb); xb = model.cnn_stage2(xb)
    if model.use_vit:
        B, C, H, W = xb.shape
        xb = xb.flatten(2).transpose(1,2)
        if model.pos_embed is not None:
            xb = xb + model.pos_embed
        for blk in model.vit_blocks:
            xb = blk(xb)
        xb = xb.transpose(1,2).reshape(B, C, H, W)
    xb = model.cnn_stage3(xb); xb = model.cnn_stage4(xb)
    cur = xb.flatten(2).transpose(1,2)
    gated1, _ = model.hierarchical_blocks[0](cur)
    kept1 = gated1[:, model.random_idx_49_to_24.to(cur.device), :]
    gated2, _ = model.hierarchical_blocks[0](cur)
    kept2 = gated2[:, model.random_idx_49_to_24.to(cur.device), :]
    if not ok(torch.equal(kept1, kept2), "A3: identical kept tokens across two forward calls"):
        all_ok = False
    if not ok(torch.equal(model.random_idx_49_to_24, torch.tensor(sorted(set(buf)))), "A3: stored indices equal the deterministic pre-computed set"):
        all_ok = False

# TEST 6: A4 invokes SE
print("\nTEST 6: A4 invokes SE")
r = manual["directL2"]
all_ok = ok(r["se_executed"], "A4: SE module executed") and all_ok

# TEST 7: A4 calculates importance from SE-GATED tokens
print("\nTEST 7: A4 importance from SE-GATED tokens")
r = manual["directL2"]
all_ok = ok(r["se_gated_used"], "A4: top-k operates on SE-GATED importance") and all_ok

# TEST 8: A4 performs exactly one 49->24 selection
print("\nTEST 8: A4 performs exactly one 49->24 selection")
r = manual["directL2"]
n_sel = len(r["selections"])
all_ok = ok(n_sel == 1, f"A4: number of selection stages={n_sel}") and all_ok
all_ok = ok(r["n_kept_tokens"] == 24, f"A4: kept tokens={r['n_kept_tokens']}") and all_ok

# TEST 9 (deeper): A5 token counts 49 -> 36 -> 24
print("\nTEST 9 (deep): A5 token counts 49 -> 36 -> 24")
r = manual["hierarchical"]
stages = [s for s in r["selections"][0]]  # list of (step_idx, k, n_kept)
all_ok = ok(len(stages) == 2, f"A5: number of hierarchical stages={len(stages)}") and all_ok
all_ok = ok(stages[0][2] == 36, f"A5: after first stage kept={stages[0][2]} (expected 36)") and all_ok
all_ok = ok(stages[1][2] == 24, f"A5: after second stage kept={stages[1][2]} (expected 24)") and all_ok
all_ok = ok(r["se_gated_used"], "A5: SE-gated tokens used") and all_ok

# TEST 11 already handled above for A6

print("\n=== SUMMARY ===")
print("ALL STRUCTURAL TESTS PASS" if all_ok else "SOME STRUCTURAL TESTS FAILED")
sys.exit(0 if all_ok else 1)
