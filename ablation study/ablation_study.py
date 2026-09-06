#!/usr/bin/env python3
"""
Ablation study for H-CoAtNet: fair component removal under identical protocol.

Paper:
  Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification (H-CoAtNet).
  Authors: Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan, Lekshmi C.R.
  Status: Under review (revision). This script supplies Table 5 and Fig. 5.

Purpose:
  Quantify the contribution of each component under a controlled ablation
  protocol. We toggle the architectural degrees of freedom one at a time and
  measure the change on the same frozen test set. This is the standard
  component ablation used in medical imaging venues such as MICCAI, MedIA,
  and TMI. We do NOT claim necessity of any single component; we report the
  measured drop in accuracy, calibration, and ranking metrics.

Reviewer mapping (see REBUTTAL_FIX_README.md):
  - R1-4 (token pruning without test label): Full vs w/o SE.
  - R1-3 (architecture consistency, ViT role): Full vs w/o ViT.
  - R2-2 and E-1 (novelty beyond standard CoAtNet): Full vs CNN-only.
  - R1-7 and R2-5 (reproducibility, 30 epoch concern): identical protocol block below.
  - R1-8 and R2-4 (uncertainty): JSON outputs carry y_true/y_pred/y_probs for
    bootstrap_ci.py and stats_tests.py. Run those after training.
  - R1-9 (efficiency): efficiency block per variant for compute_flops.py merge.
  - R1-2 (leakage): TRIPOD-AI Type 2b. Validation selects the checkpoint.
    Test is evaluated once at the end. No test curve appears during training.

Variants (only these two flags toggle, nothing else):
  - full:    ConvNeXt-Tiny [3,3,9,3] + 2 ViT blocks + HierarchicalSE 49->36->24.
             Reference model. Matches H-CoAtNet/proposed_method/train_h_coatnet.py.
  - noSE:    Same backbone + 2 ViT, no pruning. Mean pool over 49 tokens.
             Isolates the HierarchicalSE contribution.
  - noViT:   Same backbone + SE, no transformer. Straight ConvNeXt path 56->28->14->7.
             Isolates the ViT contribution.
  - cnnOnly: Pure ConvNeXt-Tiny. No ViT, no SE. Equals the CoAtNet baseline shape.
             Isolates the full hybrid novelty.

Fairness lock (identical for every variant, see ABLATION_PLAN.md Section 3):
  - Frozen stratified split, seed 42, Roboflow ich-s-7lnsj v1.
  - Image size 224x224, ImageNet mean/std.
  - Augmentation: RandomResizedCrop 0.8-1.0, HFlip, Rot15, TrivialAugmentWide,
    RandomErasing 0.2. Same object for all variants.
  - Loss: cross entropy + label smoothing 0.1 + class weights N/(C*Nc).
  - Optimizer: AdamW lr 5e-5, weight decay 0.01, CosineAnnealing T=epochs.
  - Batch 24, 30 epochs, convnext_tiny ImageNet-1K pretrained, deterministic.
  Changing any of these per variant would break fairness and invalidate Table 5.

Usage (Colab T4, quote the path because it contains a space):
  python3 "ablation study/ablation_study.py" --variant smoke --epochs 1 --seed 42
  python3 "ablation study/ablation_study.py" --variant noSE --epochs 30 --seed 42
  python3 "ablation study/ablation_study.py" --variant all --epochs 30 --seed 42
  python3 "ablation study/ablation_study.py" --variant compare

Outputs (dual write keeps old paths working):
  - results/results_ablation_{v}.json and ablation study/results/... (single source).
  - histories/history_ablation_{v}.json (curves for the 12-figure suite).
  - ablation study/figures/curve_{v}_{acc,loss}.png (train/val only).
  - ablation study/figures/confusion_{v}_{raw,norm}.png.
  - figures/fig_ablation_02_main_bar, 03_drop, 04_perclass_heatmap (PNG+PDF).
  - ablation study/ablation_table.tex (Table 5), ablation_summary.csv/json.

Reproducibility:
  - Deterministic seed 42 via seed_everything(). Same seed for all variants in
    the main table. Multi-seed 42-46 goes to the supplement with --tag.
  - Roboflow key is read from env ROBOFLOW_API_KEY only. Never hardcoded (R1-6).
  - Self-contained model definition. No import from train_h_coatnet.py, so no
    drift if that file changes. Parity was checked line by line on 2026-09-05.

References (canonical, verify before submission):
  - ConvNeXt: Liu et al., A ConvNet for the 2020s, CVPR 2022.
  - CoAtNet: Dai et al., CoAtNet, NeurIPS 2021.
  - ViT: Dosovitskiy et al., ICLR 2021.
  - SE: Hu et al., Squeeze-and-Excitation, CVPR 2018.
  - TRIPOD-AI, STARD-AI 2024, CLAIM 2024 checklists.
"""

# ----------------------------------------------------------------------------
# Imports: stdlib first, then scientific stack. All are in requirements-colab.txt
# except torch/torchvision which Colab pre-installs. No custom dependencies.
# ----------------------------------------------------------------------------
import os
import sys
import json
import argparse
import random
import gc
import hashlib
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Headless backend. Required on Colab, safe locally.
import matplotlib.pyplot as plt
import seaborn as sns
from timm import create_model
try:
    # Pinned path for timm 0.9.12 (see H-CoAtNet/requirements.txt).
    from timm.models.vision_transformer import Block
except ImportError:
    try:
        # Fallback for Colab latest where timm reorganized modules.
        from timm.models.vit import Block
    except ImportError:
        # Last resort, same Block class re-exported under timm.layers.
        from timm.layers import Block
from sklearn.metrics import (classification_report, confusion_matrix, cohen_kappa_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score)

# ----------------------------------------------------------------------------
# Paths: REPO is the outer checkout. STUDY is this folder (contains a space,
# so every shell call must quote it). We write to both results/ and
# ablation study/results/ so old notebooks keep working.
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "ablation study"
RESULTS = REPO / "results"
STUDY_RESULTS = STUDY / "results"
STUDY_FIGS = STUDY / "figures"
HIST = REPO / "histories"
for d in [RESULTS, STUDY_RESULTS, STUDY_FIGS, HIST, REPO / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Locked protocol. Do not change per variant. Values match
# H-CoAtNet/proposed_method/train_h_coatnet.py lines 31-45 and 283-321.
# Table 3 in the manuscript must list these exact numbers.
# ----------------------------------------------------------------------------
TARGET_SIZE = (224, 224)  # Input resolution for all variants.
BATCH_SIZE = 24  # Matches H-CoAtNet. ViT/Swin baselines use 16, not used here.
EPOCHS_DEFAULT = 30  # Equal budget. Addresses R2-5 (convergence by epoch 25).
LR = 5e-5  # AdamW LR for ConvNeXt-based models. EfficientNet/CNN use 3e-4 elsewhere.
WD = 0.01  # Weight decay for all ablation variants.
LS = 0.1  # Label smoothing in CE. Helps calibration (ECE) on small classes.
SEED_DEFAULT = 42  # Main seed. Supplement uses 42-46 with --tag.
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Standard ImageNet norm, all splits.
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Variant registry.
# This is an A* causal ablation ladder. The goal is not only "does SE help?"
# but "WHY does the H-CoAtNet mechanism help?" We build up the model piece by
# piece and also swap the token-selection mechanism so each claim has a direct
# controlled comparison.
#
# Causal ladder (reader-facing order; ORDER fixes every figure and Table 5):
#   A0 cnnOnly         Pure ConvNeXt-T. No ViT, no SE, no token selection, mean pool 49.
#   A1 vitOnly         ConvNeXt-T + 2 ViT, no SE, no token selection, mean pool 49.
#                      "+ViT noSE": isolates the ViT contribution beyond the CNN backbone.
#   A2 seNoPrune       ConvNeXt-T + 2 ViT + SE gating, NO token selection, mean pool 49.
#   A3 randomPrune     ConvNeXt-T + 2 ViT + SE gating + random 49->24 (no importance signal).
#   A4 directL2        ConvNeXt-T + 2 ViT + SE gating + one-shot L2 top-k 49->24.
#   A5 hierarchical    ConvNeXt-T + 2 ViT + SE gating + hierarchical 49->36->24 (FULL).
#   A6 hierarchicalRaw ConvNeXt-T + 2 ViT + raw L2 importance + hierarchical 49->36->24.
#
# Controlled comparisons the ladder is designed to answer:
#   A1 vs A0 : does the intermediate 2-block ViT add anything beyond the CNN stem?
#   A2 vs A1 : does SE channel gating help when we do NOT reduce tokens?
#   A3 vs A2 : does token reduction itself help, even if the kept tokens are random?
#   A4 vs A3 : does importance-aware selection beat arbitrary token reduction?
#   A5 vs A4 : does hierarchical 49->36->24 beat a single 49->24 top-k step?
#   A6 vs A5 : does SE gating actually improve the token-importance representation?
#
# Fairness lock: only use_vit / use_se / exec_mode differ across rows.
# Everything else is frozen: split, transforms, loss, optimizer, scheduler,
# batch, epochs, seed, IN1K pretrained init, class weights, deterministic mode.
# ----------------------------------------------------------------------------
VARIANTS = {
    "cnnOnly": {"use_vit": False, "use_se": False, "exec_mode": "none",
                "name": "A0 CNN only",
                "desc": "Pure ConvNeXt-T"},
    "vitOnly": {"use_vit": True,  "use_se": False, "exec_mode": "none",
                "name": "A1 +ViT, no SE",
                "desc": "ConvNeXt-T + 2 ViT, mean pool 49 (no SE gating)"},
    "seNoPrune": {"use_vit": True, "use_se": True, "exec_mode": "se_only",
                "name": "A2 +SE, no pruning",
                "desc": "ConvNeXt-T + 2 ViT + SE gating + mean pool 49 SE-gated tokens"},
    "randomPrune": {"use_vit": True, "use_se": True, "exec_mode": "random_se",
                "name": "A3 Fixed-random 49->24",
                "desc": "ConvNeXt-T + 2 ViT + SE gating + fixed random 49->24 from SE-gated tokens"},
    "directL2": {"use_vit": True, "use_se": True, "exec_mode": "direct_se",
                "name": "A4 One-shot SE 49->24",
                "desc": "ConvNeXt-T + 2 ViT + SE gating + one-shot L2 49->24 from SE-gated tokens"},
    "hierarchical": {"use_vit": True, "use_se": True, "exec_mode": "hierarchical_se",
                "name": "A5 Full H-CoAtNet",
                "desc": "ConvNeXt-T + 2 ViT + SE gating + hierarchical 49->36->24 (matches proposed)"},
    "hierarchicalRaw": {"use_vit": True, "use_se": False, "exec_mode": "hierarchical_raw",
                "name": "A6 Hierar. raw-token scoring",
                "desc": "ConvNeXt-T + 2 ViT + NO SE + raw L2 + hierarchical 49->36->24"},
    "noViT": {"use_vit": False, "use_se": True, "exec_mode": "hierarchical_se",
                "name": "A7 Full H-CoAtNet w/o ViT",
                "desc": "ConvNeXt-T + 2 HierarchicalSE + hierarchical 49->36->24, NO ViT blocks"},
}
ORDER = ["cnnOnly", "vitOnly", "seNoPrune", "randomPrune", "directL2",
         "hierarchical", "hierarchicalRaw", "noViT"]  # public variant labels; exec_mode drives the true forward path
VMAP_COLOR = {
    "cnnOnly": "#000000",
    "vitOnly": "#E69F00",
    "seNoPrune": "#D55E00",
    "randomPrune": "#CC79A7",
    "directL2": "#009E73",
    "hierarchical": "#0072B2",
    "hierarchicalRaw": "#56B4E9",
    "noViT": "#8C564B",
}


# ----------------------------------------------------------------------------
# Determinism: same seed gives same init, same shuffle, same aug sequence.
# Required so the drop in Table 5 is causal (component effect, not luck).
# Supplement adds multi-seed mean+-SD to show the result is not seed lucky.
# ----------------------------------------------------------------------------
def seed_everything(seed=42):
    """Fix Python, NumPy, Torch, and cuDNN randomness for this process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# HierarchicalSE: channel gating plus forward-only token scoring (Alg. 1).
# Input x has shape (B, N, C) with N=49 and C=768 from ConvNeXt stage 4 (7x7).
# Importance uses the L2 norm of gated tokens, standardized then softmaxed.
# No ground truth label, no loss gradient, and no backward pass are used,
# so it is valid at test time. This directly answers R1-4.
# Pruning keeps 36 tokens (75 percent of 49) then 24 (50 percent of 49).
# ----------------------------------------------------------------------------
class HierarchicalSE(nn.Module):
    """Channel-wise SE gating with forward-only L2 token importance."""

    def __init__(self, dim, reduction=16, dropout=0.0):
        """Build the two-layer SE bottleneck. reduction=16 matches the paper.
    
    IMPORTANT FOR FAIRNESS: the proposed H-CoAtNet defines HierarchicalSE with
    default dropout=0.0, then instantiates it with dropout=0.05 in both hierarchical
    stages. This ablation mirrors that exact interface and uses dropout=0.05 here too,
    so the full/hierarchical variant is architecturally identical to the proposed model,
    not a "lower-dropout" re-run.
    
    For the A6 raw-L2 control we do NOT create HierarchicalSE at all and we leave
    use_se=False, so no SE module runs and _raw_importance is used for both stages.
    That keeps A6 a genuine SE-free ablation instead of "SE present with Dropout=0".
    """
        super().__init__()
        mid = max(1, dim // reduction)
        self.se = nn.Sequential(nn.Linear(dim, mid), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(mid, dim), nn.Sigmoid())

    def forward(self, x):
        """Gate channels, score tokens by L2 norm, return both for top-k."""
        # Global average over tokens gives one descriptor per channel.
        s = x.mean(dim=1)
        # Sigmoid gates in [0,1] reweight each channel.
        gates = self.se(s).unsqueeze(1)
        # Apply gating, keep shape (B, N, C).
        out = x * gates
        # Forward-only score: L2 norm per token, no label needed.
        scores = out.norm(dim=-1)
        # Standardize across tokens for stable softmax temperature.
        scores = scores - scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True) + 1e-6
        importance = F.softmax(scores / std, dim=-1)
        return out, importance


# ----------------------------------------------------------------------------
# AblationCoAtNet: ConvNeXt-Tiny backbone with switchable ViT and SE stages.
# Borrowed (cite): ConvNeXt-Tiny stages [3,3,9,3], dims [96,192,384,768]
#   (Liu CVPR 22), ViT Block 192-d 6 heads (Dosovitskiy ICLR 21).
# New (ours): early stages 1-2, then 2 ViT blocks, then late stages 3-4
#   interleaving plus HierarchicalSE 49->36->24. Unlike stacked CoAtNet and
#   unlike GFT which uses 8 ViT plus 3 GALA stages. Addresses R1-3 and R2-2.
# Self-contained on purpose: no import from train_h_coatnet.py, so reviewer
# can read this file alone and trust there is no hidden difference.
# ----------------------------------------------------------------------------
class AblationCoAtNet(nn.Module):
    """ConvNeXt-Tiny backbone with switchable mid ViT, SE gating, and token selection.

    This is the single shared backbone used for the whole A0-A6 causal ladder below.
    The only degrees of freedom across variants are use_vit, use_se, and exec_mode;
    everything else is frozen (split, transforms, loss, optimizer, scheduler, batch,
    epochs, seed, IN1K init, class weights) so the measured drop is causal.

    Fairness-critical note on dropout and weight parity (R1-7 / R2-5):
    Variants that include an SE path insert nn.Dropout(0.05) inside HierarchicalSE.
    When we re-create AblationCoAtNet per variant under the same seed and optimizer,
    the ConvNeXt backbone weights start identical, but the SE path sees a different
    dropout mask at every forward. That is by design: dropout is part of the SE
    component being ablated, and freezing it across variants would instead hide the
    SE effect. The fairness lock still holds because (1) the backbone init, LR, WD,
    schedule, batch, augmentations, and class weights are byte-identical across variants,
    (2) no variant gets extra warmup or per-variant hyperparameter tuning, and (3) the
    only architectural degrees of freedom are use_vit, use_se, and exec_mode.

    Note on A2 vs A3 vs A4 (actual executable paths):
    - A2 (seNoPrune, use_se=True, exec_mode="se_only"): the forward pass calls the SE
      module on the 49 stage-4 tokens, returns the gated tokens, and mean-pools all 49.
      No token selection is executed. This isolates SE channel gating while keeping the
      token count fixed at 49.
    - A3 (randomPrune, use_se=True, exec_mode="random_se"): the forward pass calls the SE
      module first, then selects a fixed-image-independent random subset of the SE-gated
      tokens down to 24. This is the "fixed-random control" -- the random indices are
      generated once in __init__ from the run_seed and reused for every forward.
    - A4 (directL2, use_se=True, exec_mode="direct_se"): the forward pass calls the SE
      module, then computes L2 importance from the SE-gated tokens and applies exactly
      one top-k selection 49->24. No second selection stage is executed.

    Note on A5 vs A6 (SE-gated vs raw importance): both share the same hierarchical
    49->36->24 sizes. The difference is whether the importance scores are computed on
    SE-gated tokens or on raw tokens. This isolates whether SE improves the importance
    representation specifically, rather than whether importance selection helps at all.
    For A6 we do not pass use_se=True with a no-op SE; instead we leave use_se=False so
    no HierarchicalSE modules are created and _raw_importance is used for both stages.

    Note on A7 vs A5 (full H-CoAtNet with vs without ViT): both use the same two
    HierarchicalSE modules and the same 49->36->24 hierarchical schedule. A7 differs
    from A5 only by removing the two intermediate ViT blocks. This isolates the
    contribution of the ViT component when the rest of the full H-CoAtNet
    architecture (hierarchical SE-gated selection, classifier head) is preserved.
    """

    def __init__(self, use_vit=True, use_se=True, exec_mode="none", run_seed=42,
                 num_classes=5, pretrained=True, vit_blocks=2):
        """Create backbone and optional stages. pretrained=True uses IN1K.

        exec_mode controls the EXECUTABLE pathway on the 49 tokens from stage 4:
          "none"            -> mean pool all 49 tokens; no SE run, no selection.
          "se_only"         -> run SE, keep SE-GATED tokens, mean pool all 49.
          "random_se"       -> run SE, then fixed random subset of SE-GATED tokens 49->24.
          "direct_se"       -> run SE once, one-shot top-k 49->24 on SE-GATED importance.
          "hierarchical_se" -> run SE at 49->36 then 36->24 (matches proposed model exactly).
                              Used by both A5 (with ViT) and A7 (without ViT).
          "hierarchical_raw" -> no SE run; raw-token L2 importance at 49->36 then 36->24.

        SE module allocation rule (parameter-accounting fairness):
          - A5/A7 (exec_mode="hierarchical_se"): TWO HierarchicalSE modules created
            because two genuine stages 49->36 and 36->24 are executed.
          - A2/A3/A4 (exec_mode != "hierarchical_se" / "hierarchical_raw"): ONE
            HierarchicalSE module created because a single SE pass is performed.
          - A0/A1/A6 (use_se=False): ZERO HierarchicalSE modules created.
        """

        super().__init__()
        self.use_vit = use_vit
        self.use_se = use_se
        self.exec_mode = exec_mode
        # ConvNeXt-Tiny without its classifier head. Stages hold [3,3,9,3] blocks.
        backbone = create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
        self.cnn_stem = backbone.stem      # 224->56, 96 channels.
        self.cnn_stage1 = backbone.stages[0]  # 56x56, 96ch, 3 blocks.
        self.cnn_stage2 = backbone.stages[1]  # 28x28, 192ch, 3 blocks.
        self.cnn_stage3 = backbone.stages[2]  # 14x14, 384ch, 9 blocks.
        self.cnn_stage4 = backbone.stages[3]  # 7x7, 768ch, 3 blocks.
        # Mid transformer operates at 28x28 with 192 channels (784 tokens).
        if use_vit:
            self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, 192))
            self.vit_blocks = nn.ModuleList([Block(dim=192, num_heads=6) for _ in range(vit_blocks)])
        else:
            self.pos_embed = None
            self.vit_blocks = nn.ModuleList([])
        # Late selection operates on 49 tokens of 768-d from stage 4 (7x7).
        self.selection_sizes = [int(49 * 0.75), int(49 * 0.5)]  # Resolves to [36, 24].
        # IMPORTANT (parameter-accounting fairness, reviewer R-fix-1):
        # Only A5 and A7 (exec_mode="hierarchical_se") genuinely need TWO SE modules
        # because they run both hierarchical stages 49->36 and 36->24. A2/A3/A4
        # execute exactly ONE SE pass on the 49 stage-4 tokens, so creating a
        # second unused module would inflate their parameter count and make the
        # Table 5 efficiency column misleading. The single_se module below is
        # reused when exec_mode is se_only / random_se / direct_se. A0/A1/A6
        # (use_se=False) create no SE modules at all.
        self.single_se = None
        self.hierarchical_blocks = nn.ModuleList([])
        if use_se:
            if exec_mode == "hierarchical_se":
                # A5 and A7 both need two genuine hierarchical SE stages.
                self.hierarchical_blocks = nn.ModuleList(
                    [HierarchicalSE(dim=768, reduction=16, dropout=0.05) for _ in self.selection_sizes])
            else:
                # A2/A3/A4: one SE pass only.
                self.single_se = HierarchicalSE(dim=768, reduction=16, dropout=0.05)
        # Classifier head on the mean pooled kept tokens.
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, num_classes))
        self.run_seed = int(run_seed)
        # A3 reproducibility: fixed random subset created once, reused every forward.
        if self.use_se and self.exec_mode == "random_se":
            rng = torch.Generator().manual_seed(self.run_seed)
            idx49 = torch.randperm(49, generator=rng)
            self.register_buffer("random_idx_49_to_24", idx49[:24].clone())
        else:
            self.register_buffer("random_idx_49_to_24", torch.tensor([], dtype=torch.long), persistent=False)

    def select_patches(self, tokens, importance, k):
        """Keep top-k tokens by importance. Batched gather, no label used.

        IMPORTANT: this operates on whomever called it. For SE-gated variants the caller
        must pass the SE-GATED tokens; for raw-L2 variants the caller must pass RAW tokens.
        """
        B, N, C = tokens.size()
        k = min(k, N)
        _, idx = torch.topk(importance, k, dim=1)
        bi = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
        return tokens[bi, idx]

    def _raw_importance(self, tokens):
        """Forward-only L2 importance from raw tokens (no SE gating).

        Matches Alg.1 but without the SE bottleneck, so A6 isolates whether SE
        improves the importance representation rather than whether importance helps.
        """
        scores = tokens.norm(dim=-1)
        scores = scores - scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True) + 1e-6
        return F.softmax(scores / std, dim=-1)

    def _select(self, cur, step_idx):
        """Return the token set after one selection stage, driven by exec_mode.

        Semantics MUST match this table exactly:
          exec_mode="random_se"     -> cur is SE-GATED tokens (via single_se), fixed random subset kept.
          exec_mode="direct_se"     -> cur is SE-GATED tokens (via single_se), SE-GATED importance used, one selection.
          exec_mode="hierarchical_se" -> cur is SE-GATED tokens, SE-GATED importance used at this stage.
          exec_mode="hierarchical_raw" -> cur is RAW tokens, raw L2 importance used at this stage.
        """
        k = self.selection_sizes[step_idx]
        if self.exec_mode == "random_se":
            # A3: SE gate first (via dedicated single_se module) so we are consistent
            # with the documented "random 49->24 from SE-gated tokens" claim, then keep
            # a fixed random subset of the gated tokens. Uses single_se (NOT
            # hierarchical_blocks[1]) so only one SE module exists for A3.
            gated, _ = self.single_se(cur)
            idx = self.random_idx_49_to_24.to(gated.device)[:k]
            bi = torch.arange(gated.size(0), device=gated.device).unsqueeze(1).expand(-1, k)
            return gated[bi, idx]
        if self.exec_mode == "direct_se":
            # A4: single SE pass + one-shot top-k 49->24. Uses single_se (NOT
            # hierarchical_blocks[1]) so only one SE module exists for A4.
            gated, imp = self.single_se(cur)
            return self.select_patches(gated, imp, k)
        if self.exec_mode == "hierarchical_se":
            # A5 / A7: two genuine stages 49->36 and 36->24.
            gated, imp = self.hierarchical_blocks[step_idx](cur)
            return self.select_patches(gated, imp, k)
        if self.exec_mode == "hierarchical_raw":
            imp = self._raw_importance(cur)
            return self.select_patches(cur, imp, k)
        # Fallback: mean pool remaining tokens (should not be reached for a
        # configured variant, but keeps the forward safe).
        return cur

    def forward(self, x):
        """Early CNN, optional ViT, late CNN, optional SE execution + selection, mean pool, linear.

        The EXECUTABLE pathway is driven by exec_mode, not by overloaded use_se flags.
        That guarantees the following controlled behavior:
          A0/A1: exec_mode="none"       -> mean pool all 49 tokens (no SE, no selection).
          A2: exec_mode="se_only"       -> SE called, then mean pool all 49 SE-GATED tokens.
          A3: exec_mode="random_se"     -> SE called, then fixed random subset of SE-GATED tokens.
          A4: exec_mode="direct_se"     -> SE called once, then one-shot top-k 49->24 on SE-GATED importance.
          A5/A7: exec_mode="hierarchical_se" -> SE called at both stages 49->36->24.
          A6: exec_mode="hierarchical_raw" -> no SE; raw-token L2 importance used at both stages.
        """
        # Early local texture: stem plus stages 1-2. Output is (B,192,28,28).
        x = self.cnn_stem(x)
        x = self.cnn_stage1(x)
        x = self.cnn_stage2(x)
        # Mid global context: 2 ViT blocks on 784 tokens. Skipped when no ViT.
        if self.use_vit:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2) + self.pos_embed
            for blk in self.vit_blocks:
                x = blk(x)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        # Late refinement: stages 3-4. Output is (B,768,7,7).
        x = self.cnn_stage3(x)
        x = self.cnn_stage4(x)
        # Flatten to 49 tokens of 768-d for selection.
        x = x.flatten(2).transpose(1, 2)
        cur = x
        if self.exec_mode == "none":
            # Mean pool all 49 tokens: cnnOnly, vitOnly, and ALSO seNoPrune AFTER SE.
            # seNoPrune executes SE first (see below), then falls through to this mean pool.
            pass
        elif self.exec_mode == "se_only":
            # A2: single SE pass on the 49 stage-4 tokens, then mean-pool all 49.
            # Uses single_se (NOT hierarchical_blocks[0]) so only one SE module exists.
            cur = self.single_se(cur)[0]
        elif self.exec_mode == "direct_se":
            # A4: single SE pass + one-shot top-k 49->24 on SE-GATED importance.
            # Calls _select with step_idx=1 to get k=24 from selection_sizes.
            cur = self._select(cur, 1)
        elif self.exec_mode in ("hierarchical_se", "hierarchical_raw"):
            # A5 / A6 / A7: two-stage hierarchical selection 49->36->24.
            for i, k in enumerate(self.selection_sizes):
                cur = self._select(cur, i)
        elif self.exec_mode == "random_se":
            # A3: single SE pass + fixed random subset of SE-gated tokens (k=24).
            cur = self._select(cur, 1)
        # Mean over kept tokens then linear to 5 ichthyosis classes.
        return self.classifier(cur.mean(dim=1))


# ----------------------------------------------------------------------------
# Transforms: byte-identical to train_h_coatnet.py train/val transforms.
# Train uses stochastic aug. Val and test use deterministic Resize only.
# Same object is used for all 4 variants, which is what makes the comparison
# fair. Fallback keeps Colab running on old torchvision without changing
# fairness (fallback still applies equally to every variant).
# ----------------------------------------------------------------------------
def build_transforms():
    """Return (train_transform, eval_transform) shared by all variants."""
    try:
        trivial = transforms.TrivialAugmentWide()
    except AttributeError:
        # Very old torchvision lacks TrivialAugmentWide. Fall back but warn.
        # Fairness holds because the fallback is still shared across variants.
        print("  WARN: TrivialAugmentWide missing, using AutoAugment fallback (still identical across variants)")
        try:
            trivial = transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        except AttributeError:
            # No-op placeholder keeps the Compose length stable for logging.
            trivial = transforms.RandomHorizontalFlip(p=0.0)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        trivial,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# Dataloaders: ImageFolder with train/valid/test subfolders from Roboflow v1.
# Note the folder is named valid (not val) in the Roboflow export. Shuffling
# is on for train only. Batch 24 and 2 workers match the main training.
# ----------------------------------------------------------------------------
def get_dataloaders(dataset_dir):
    """Build train/valid/test loaders plus class names and train set stats.

    Split enforcement (reviewer R-fix-3): if splits/seed42_indices.json
    exists, we cross-check the actual train/valid/test folder membership
    against the manifest. If they disagree we RAISE RuntimeError -- failing
    loudly is required so reviewers cannot accidentally train on a split
    that does not match the published manifest. A missing manifest is
    also a hard error (paper-safety); see _verify_split_manifest for the
    HCOATNET_SKIP_SPLIT_CHECK escape hatch (debug only).
    """
    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(dataset_dir, "valid"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform=eval_tf)
    _verify_split_manifest(dataset_dir, train_ds, valid_ds, test_ds)
    nw = 0 if os.name == "nt" else 2  # Windows needs 0 workers, Linux/Colab uses 2.
    g = torch.Generator()
    g.manual_seed(SEED_DEFAULT)
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw,
                    worker_init_fn=_seed_worker, generator=g)
    vl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw,
                    worker_init_fn=_seed_worker, generator=g)
    el = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw,
                    worker_init_fn=_seed_worker, generator=g)
    return tl, vl, el, train_ds.classes, train_ds


def _seed_worker(worker_id):
    """Deterministic per-worker seed (DataLoader fairness hardening).

    PyTorch's DataLoader spawns N worker processes whose RNG state is otherwise
    inherited from the parent at fork time. To make augmentation order
    reproducible across variants and across runs, each worker is given an
    explicit seed derived from SEED_DEFAULT and worker_id. Combined with
    seed_everything() at the top of each variant this makes the training
    shuffle sequence byte-identical across variants A0-A7.
    """
    seed = SEED_DEFAULT + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _verify_split_manifest(dataset_dir, train_ds, valid_ds, test_ds):
    """Cross-check the actual train/valid/test membership against splits/seed42_indices.json.

    Paper-safety: a missing manifest is now a HARD ERROR (was previously a warning).
    The frozen split must be verified before any variant trains, otherwise the
    ablation could silently use a different train/valid/test split and Table 5
    would be invalid. To skip verification (debug only) set the environment
    variable HCOATNET_SKIP_SPLIT_CHECK=1.

    The manifest may take one of two shapes (auto-detected):
      (a) Filename-list: {"train": [...], "valid": [...], "test": [...]}
          where each entry is a relative file path under dataset_dir.
      (b) Count-only: {"counts": {"train": N, "valid": N, "test": N},
                        "per_class_counts": {"train": {...}, ...}}
          which is what freeze_split.py currently emits.

    Both shapes are verified. A mismatch raises RuntimeError so the run cannot
    silently use a different split from the published manifest.
    """
    if os.environ.get("HCOATNET_SKIP_SPLIT_CHECK", "") == "1":
        print("  [SPLIT] verification skipped via HCOATNET_SKIP_SPLIT_CHECK=1 (debug only)")
        return
    manifest = REPO / "splits/seed42_indices.json"
    if not manifest.exists():
        raise RuntimeError(
            f"FROZEN-SPLIT MANIFEST MISSING.\n"
            f"  Expected: {manifest}\n"
            f"  Action: run tools/freeze_split.py --seed 42 to record the frozen split\n"
            f"  before any paper run. To skip (debug only): export HCOATNET_SKIP_SPLIT_CHECK=1"
        )
    try:
        import json as _json
        data = _json.loads(manifest.read_text())

        # Build RELATIVE path sets for shape (a) comparison
        rel = lambda p: str(Path(p).relative_to(dataset_dir)) if Path(p).is_absolute() and str(p).startswith(str(dataset_dir)) else os.path.basename(str(p))
        got_train_rel = {rel(s) for s, _ in train_ds.samples}
        got_valid_rel = {rel(s) for s, _ in valid_ds.samples}
        got_test_rel = {rel(s) for s, _ in test_ds.samples}

        want_train = data.get("train")
        want_valid = data.get("valid")
        want_test = data.get("test")

        if want_train is not None and want_valid is not None and want_test is not None:
            # Shape (a): filename-list comparison. Use FULL relative paths to
            # avoid false matches caused by duplicate basenames in different dirs.
            def to_rel(x):
                s = str(x)
                # If looks like absolute under dataset_dir, take relative.
                if os.path.isabs(s) and str(dataset_dir) in s:
                    return os.path.relpath(s, dataset_dir)
                return s
            wt = {to_rel(x) for x in want_train}
            wv = {to_rel(x) for x in want_valid}
            wte = {to_rel(x) for x in want_test}
            miss_train = wt - got_train_rel
            miss_valid = wv - got_valid_rel
            miss_test = wte - got_test_rel
            extra_train = got_train_rel - wt
            extra_valid = got_valid_rel - wv
            extra_test = got_test_rel - wte
            if miss_train or miss_valid or miss_test or extra_train or extra_valid or extra_test:
                raise RuntimeError(
                    f"FROZEN-SPLIT MISMATCH (filename-list shape).\n"
                    f"  missing from disk: train={len(miss_train)} valid={len(miss_valid)} test={len(miss_test)}\n"
                    f"  extra on disk:     train={len(extra_train)} valid={len(extra_valid)} test={len(extra_test)}\n"
                    f"  Dataset: {dataset_dir}\n"
                    f"  Manifest: {manifest}\n"
                    f"  Action: re-run tools/freeze_split.py --seed 42 or pass --dataset_dir to the correct folder."
                )
            print(f"  [SPLIT OK filename-list] train {len(got_train_rel)} valid {len(got_valid_rel)} test {len(got_test_rel)} all match manifest")
            return

        # Shape (b): count-only comparison
        counts = data.get("counts")
        if counts is not None:
            want_n_train = int(counts.get("train", -1))
            want_n_valid = int(counts.get("valid", -1))
            want_n_test = int(counts.get("test", -1))
            got_n_train = len(train_ds.samples)
            got_n_valid = len(valid_ds.samples)
            got_n_test = len(test_ds.samples)
            if want_n_train >= 0 and got_n_train != want_n_train:
                raise RuntimeError(
                    f"FROZEN-SPLIT MISMATCH (count shape): train count {got_n_train} != manifest {want_n_train}"
                )
            if want_n_valid >= 0 and got_n_valid != want_n_valid:
                raise RuntimeError(
                    f"FROZEN-SPLIT MISMATCH (count shape): valid count {got_n_valid} != manifest {want_n_valid}"
                )
            if want_n_test >= 0 and got_n_test != want_n_test:
                raise RuntimeError(
                    f"FROZEN-SPLIT MISMATCH (count shape): test count {got_n_test} != manifest {want_n_test}"
                )
            # Also cross-check per-class counts when present
            pc = data.get("per_class_counts")
            if pc is not None:
                cls_order = data.get("classes", train_ds.classes)
                for split, want_dict in pc.items():
                    if split not in {"train", "valid", "test"}:
                        continue
                    if "_counts_list" in want_dict:
                        want_list = want_dict["_counts_list"]
                        ds = {"train": train_ds, "valid": valid_ds, "test": test_ds}[split]
                        # Count labels in ds.samples
                        got_counter = Counter([ds.samples[i][1] for i in range(len(ds.samples))])
                        got_list = [got_counter.get(i, 0) for i in range(len(cls_order))]
                        if list(want_list) != got_list:
                            raise RuntimeError(
                                f"FROZEN-SPLIT MISMATCH (per-class shape): {split} per-class {got_list} != manifest {list(want_list)}"
                            )
            print(f"  [SPLIT OK count] train {got_n_train} valid {got_n_valid} test {got_n_test} all match manifest")
            return

        # Neither shape recognized
        print(f"  [SPLIT WARN] manifest has no 'train/valid/test' keys and no 'counts' key; cannot verify (keys: {list(data.keys())})")
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  [SPLIT WARN] could not verify manifest: {e}")


# ----------------------------------------------------------------------------
# Fairness banner: printed at the start of every run so the log itself proves
# equal budget. Also prints the frozen split hash when available (R1-1).
# ----------------------------------------------------------------------------
def fairness_banner(epochs, seed):
    """Print locked hyperparameters and frozen split hash to stdout log."""
    print("=" * 70)
    print("FAIRNESS LOCK (identical for every variant):")
    print(f"  split seed {seed} frozen | 224 ImageNet norm | aug RRC0.8-1.0+HFlip+Rot15+TrivAug+Eras0.2")
    print(f"  CE+LS{LS} + classweight | AdamW lr{LR} WD{WD} Cosine T={epochs} | batch {BATCH_SIZE} | {epochs}ep")
    print(f"  init convnext_tiny IN1K | deterministic | TRIPOD-2b test-once | device {DEVICE}")
    print("  ONLY toggled: use_vit / use_se / exec_mode (see VARIANTS for each variant's config)")
    print("=" * 70)
    sp = REPO / "splits/seed42_indices.json"
    if sp.exists():
        try:
            h = hashlib.sha256(sp.read_bytes()).hexdigest()[:12]
            print(f"  frozen split: {sp} sha {h}")
        except Exception as e:
            print(f"  split check skip: {e}")
    else:
        print("  WARN: splits/seed42_indices.json missing - freeze split (tools/freeze_split.py --seed 42) before paper. R1-1 and all test-n counts in Table 5 depend on one source of truth.")


# ----------------------------------------------------------------------------
# Training loop: one epoch over train. Validation and test never update weights.
# Test loader is never touched here. Only train_one_variant calls the test
# loader once at the end, which enforces TRIPOD-AI Type 2b (R1-2).
# ----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer):
    """Run one training epoch. Return (mean loss, accuracy)."""
    model.train()
    tot, preds, tgts = 0.0, [], []
    for img, y in tqdm(loader, desc="Training", leave=False):
        img, y = img.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tot += loss.item()
        preds.extend(out.argmax(1).detach().cpu().numpy())
        tgts.extend(y.detach().cpu().numpy())
    acc = (np.array(preds) == np.array(tgts)).mean() if preds else 0.0
    return tot / max(1, len(loader)), float(acc)


# ----------------------------------------------------------------------------
# Validation: no gradients, no weight updates. Used for model selection
# (best val accuracy). History from this function draws the train/val curves.
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, desc="Validating"):
    """Evaluate loss and accuracy without gradients. Return (loss, acc, y, p)."""
    model.eval()
    tot, preds, tgts = 0.0, [], []
    for img, y in tqdm(loader, desc=desc, leave=False):
        img, y = img.to(DEVICE), y.to(DEVICE)
        out = model(img)
        tot += criterion(out, y).item()
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
    acc = (np.array(preds) == np.array(tgts)).mean() if preds else 0.0
    return tot / max(1, len(loader)), float(acc), tgts, preds


# ----------------------------------------------------------------------------
# Held-out test: called exactly once per variant on the best val checkpoint.
# Also returns softmax probabilities needed for AUROC, AUPRC, ECE, Brier,
# bootstrap CIs, and ROC/PR/reliability figures. Wrapped in no_grad so no
# test gradient ever flows (R1-2, R1-4).
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_with_probs(model, loader, criterion, desc="Test"):
    """Evaluate once on held-out test. Return (loss, acc, y_true, y_pred, probs)."""
    model.eval()
    tot, preds, tgts, probs = 0.0, [], [], []
    for img, y in tqdm(loader, desc=desc, leave=False):
        img, y = img.to(DEVICE), y.to(DEVICE)
        out = model(img)
        tot += criterion(out, y).item()
        pr = F.softmax(out, dim=1)
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
        probs.extend(pr.cpu().numpy())
    acc = (np.array(preds) == np.array(tgts)).mean() if preds else 0.0
    return tot / max(1, len(loader)), float(acc), tgts, preds, np.array(probs)


# ----------------------------------------------------------------------------
# ECE with 15 bins (matches train_h_coatnet.py). Low ECE means predicted
# confidence matches observed accuracy, which matters for the clinical
# decision-support claim (R1-11). Brier is the mean squared error between
# one-hot truth and predicted probabilities. Report both.
# ----------------------------------------------------------------------------
def compute_ece(probs, y_true, n_bins=15):
    """Compute Expected Calibration Error over equal width confidence bins.

    Note on comparability: this binning scheme is identical across every ablation
    variant and matches train_h_coatnet.py, so ECE values in Table 5 are directly
    comparable. If a future revision wants binned ECE only on the held-out test
    (to avoid any train-signal bleed), the same function is reused on y_probs from
    the single held-out evaluation; no retrain is needed.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (pred == np.array(y_true))
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += abs(ok[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def compute_brier(probs, y_true, n_classes):
    """Compute multi-class Brier score (lower is better calibrated).

    This complements ECE: Brier summarizes the full probability vector error, while
    ECE summarizes confidence-accuracy mismatch per bin. Both are reported in the
    JSON so reviewers can choose the calibration view they prefer.
    """
    from sklearn.preprocessing import label_binarize
    yb = label_binarize(y_true, classes=list(range(n_classes)))
    return float(np.mean((yb - probs) ** 2))


# ----------------------------------------------------------------------------
# Efficiency: parameter count plus MACs via thop when installed.
# Measured per variant on the same input (1,3,224,224) for Table 4 (R1-9).
# thop is optional so smoke tests never fail for lack of it.
# ----------------------------------------------------------------------------
def measure_efficiency(model):
    """Return (params_M, macs_G). macs_G is None if thop is not installed.

    Measured per variant on the same (1,3,224,224) input so the efficiency column
    in Table 5 compares like with like (R1-9).
    """
    params = sum(p.numel() for p in model.parameters()) / 1e6
    macs = None
    try:
        from thop import profile
        model.eval()
        m, p = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(DEVICE),), verbose=False)
        macs = m / 1e9
    except Exception:
        pass
    return float(params), (float(macs) if macs else None)


# ----------------------------------------------------------------------------
# One variant end to end: dataloaders, class weights, model, optimizer,
# 30 epoch loop with best-val checkpointing, single held-out test, full
# metric package, JSON save, history save, and per-variant plots.
# Class weights N/(C*Nc) correct the imbalance (LI test n=22 vs IV n=46).
# Specificity TN/(TN+FP) is added per class for the clinical table (Table 9).
# ----------------------------------------------------------------------------
def train_one_variant(variant, dataset_dir, epochs=30, seed=42, pretrained=True, out_tag=""):
    """Train one ablation variant fairly and evaluate test once. Return result dict."""
    cfg = VARIANTS[variant]
    print("\n" + "=" * 70)
    print(f"VARIANT [{variant}] {cfg['name']} - {cfg['desc']}")
    print("=" * 70)
    # Same seed for every variant in the main table makes the drop causal.
    seed_everything(seed)
    tl, vl, el, class_names, train_ds = get_dataloaders(dataset_dir)
    nc = len(class_names)
    print(f"  classes {class_names} | train {len(train_ds)} val {len(vl.dataset)} test {len(el.dataset)}")
    # Inverse frequency weights so rare classes (LI, NS) are not ignored.
    counts = np.bincount(train_ds.targets)
    cw = torch.tensor([len(train_ds) / (c * nc + 1e-6) for c in counts], dtype=torch.float).to(DEVICE)
    print(f"  class weights {cw.detach().cpu().numpy().round(3)}")

    # Build the toggled model. pretrained=True keeps IN1K init equal for all.
    model = AblationCoAtNet(
        use_vit=cfg["use_vit"], use_se=cfg["use_se"], exec_mode=cfg["exec_mode"],
        num_classes=nc, pretrained=pretrained).to(DEVICE)
    params_m, _ = measure_efficiency(model)
    print(f"  params {params_m:.2f}M | {cfg['desc']}")
    print(f"  actual path: use_vit={cfg['use_vit']} use_se={cfg['use_se']} exec_mode={cfg['exec_mode']!r}")
    em = cfg['exec_mode']
    if em == "se_only":
        print("  note: A2 path = SE gated, then mean pool all 49 tokens (no selection)")
    elif em == "none" and not cfg['use_se'] and cfg['use_vit']:
        print("  note: A1 path = +ViT, then mean pool all 49 tokens (no SE, no selection)")
    elif em == "random_se":
        print("  note: A3 path = SE gated, then random 49->24 from SE-gated tokens")
    elif em == "direct_se":
        print("  note: A4 path = SE gated, then one-shot L2 top-k 49->24 from SE-gated tokens")
    elif em == "hierarchical_se":
        print("  note: A5 path = SE gated, then hierarchical 49->36->24 (FULL)")
    elif em == "hierarchical_raw":
        print("  note: A6 path = NO SE, then raw-L2 hierarchical 49->36->24")
    elif em == "none" and not cfg['use_vit'] and not cfg['use_se']:
        print("  note: A0 path = ConvNeXt-T only, then mean pool all 49 tokens")

    # Identical loss and optimizer for every variant (fairness lock).
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop. Test loader is not touched inside this loop (TRIPOD-2b).
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_ep = -1.0, -1
    ckpt = STUDY_RESULTS / f"best_{variant}{out_tag}.pth"
    for ep in range(epochs):
        print(f"--- Epoch {ep + 1}/{epochs} [{variant}] ---")
        trl, tra = train_epoch(model, tl, criterion, optimizer)
        vall, vala, _, _ = evaluate(model, vl, criterion)
        scheduler.step()
        hist["train_loss"].append(trl)
        hist["train_acc"].append(tra)
        hist["val_loss"].append(vall)
        hist["val_acc"].append(vala)
        print(f"  train acc {tra:.4f} loss {trl:.4f} | val acc {vala:.4f} loss {vall:.4f}")
        # Best checkpoint is selected by validation only, never by test.
        if vala > best_val:
            best_val, best_ep = vala, ep + 1
            torch.save(model.state_dict(), ckpt)
            print(f"  [NEW BEST] epoch {best_ep} val {best_val:.4f}")

    # Single held-out test on the best val checkpoint. This line runs once.
    print("--- Final Test ONCE (held-out) ---")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, test_acc, yt, yp, ypr = evaluate_with_probs(model, el, criterion, desc="Final Test")
    print(f"  test acc {test_acc:.4f} n={len(yt)} support {dict(Counter(yt))}")

    # Aggregate metrics. Balanced accuracy and macro F1 protect rare classes.
    # Kappa and MCC correct for chance. AUROC/AUPRC measure ranking.
    bal = balanced_accuracy_score(yt, yp)
    kappa = cohen_kappa_score(yt, yp)
    mcc = matthews_corrcoef(yt, yp)
    ece = compute_ece(ypr, yt)
    brier = compute_brier(ypr, yt, nc)
    try:
        from sklearn.preprocessing import label_binarize
        yb = label_binarize(yt, classes=list(range(nc)))
        auroc = roc_auc_score(yb, ypr, average="macro", multi_class="ovr")
        auprc = average_precision_score(yb, ypr, average="macro")
    except Exception as e:
        print(f"  auroc/auprc skip: {e}")
        auroc, auprc = float("nan"), float("nan")
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    # Per-class report plus specificity for the clinical table.
    cm = confusion_matrix(yt, yp, labels=list(range(nc)))
    per = classification_report(yt, yp, target_names=class_names, digits=4, output_dict=True)
    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        spec = float(tn / (tn + fp + 1e-9))
        if cname in per:
            per[cname]["specificity"] = spec
            per[cname]["support"] = int(cm[i, :].sum())
    print(classification_report(yt, yp, target_names=class_names, digits=4))
    print(f"  bal {bal:.4f} macroF1 {fm:.4f} kappa {kappa:.4f} mcc {mcc:.4f} ece {ece:.4f} brier {brier:.4f} auroc {auroc:.4f} auprc {auprc:.4f}")

    # Validate y_probs before saving (catches edge cases early).
    if ypr.size == 0 or np.isnan(ypr).all():
        print(f"  WARN: y_probs empty or all-NaN for {variant}. Checkpoint may be corrupt.")
    # Single source of truth JSON. y_true/y_pred/y_probs enable bootstrap,
    # McNemar, DeLong, ROC/PR, and reliability without retraining (R1-8).
    # All four variant JSONs share one schema so compare diffs them without transcription (R1-10).
    params_m2, macs_g = measure_efficiency(model)
    out = {
        "model": cfg["name"], "variant": variant, "desc": cfg["desc"],
        "seed": seed, "epochs": epochs, "protocol": {
            "lr": LR, "wd": WD, "batch": BATCH_SIZE, "ls": LS,
            "size": list(TARGET_SIZE), "deterministic": True, "tripod": "Type 2b test-once"},
        "best_val_acc": float(best_val), "best_epoch": int(best_ep),
        "history": hist,
        "efficiency": {"params_M": params_m2, "macs_G": macs_g},
        "test": {
            "accuracy": float(test_acc), "balanced_accuracy": float(bal),
            "macro": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
            "weighted": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
            "kappa": float(kappa), "mcc": float(mcc), "ece": float(ece), "brier": float(brier),
            "auroc_macro": float(auroc) if not np.isnan(auroc) else None,
            "auprc_macro": float(auprc) if not np.isnan(auprc) else None,
            "n": int(len(yt)), "support_per_class": {class_names[i]: int(Counter(yt)[i]) for i in range(nc)},
            "y_true": list(map(int, yt)), "y_pred": list(map(int, yp)), "y_probs": ypr.tolist(),
        },
        "per_class": per, "classes": class_names,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Dual write: repo results/ for old scripts, study results/ for this section.
    for p in [RESULTS / f"results_ablation_{variant}{out_tag}.json",
              STUDY_RESULTS / f"results_ablation_{variant}{out_tag}.json"]:
        p.write_text(json.dumps(out, indent=2))
        print(f"  saved {p}")
    # History in the shared histories/ folder feeds the 12-figure suite.
    (HIST / f"history_ablation_{variant}{out_tag}.json").write_text(
        json.dumps({"model": cfg["name"], "variant": variant, "epochs": list(range(1, epochs + 1)), **hist}, indent=2))

    # Per-variant plots show train/val only. No test curve appears (R1-2).
    for met in ["acc", "loss"]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(hist[f"train_{met}"], label=f"Train {met}", color="#0072B2", lw=2)
        ax.plot(hist[f"val_{met}"], label=f"Val {met}", color="#D55E00", lw=2)
        ax.set_title(f'{cfg["name"]} {met} - Train vs Val (n_val={len(vl.dataset)}, seed {seed})')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(met)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(STUDY_FIGS / f"curve_{variant}_{met}.png", bbox_inches="tight")
        plt.close(fig)
    # Confusion matrices in raw counts and row-normalized form for the supplement.
    for norm, fmt, suf in [(False, "d", "raw"), (True, ".2f", "norm")]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        m = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) if norm else cm
        sns.heatmap(m, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Confusion {suf} - {cfg["name"]} (test n={len(yt)}, once)')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(STUDY_FIGS / f"confusion_{variant}_{suf}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"  plots in {STUDY_FIGS}/")
    return out


# ----------------------------------------------------------------------------
# Compare mode: no training, only tables and figures from saved JSONs.
# Checks that test n is consistent across variants (R1-1, R1-10). Warns loudly
# if splits differ, because Table 5 would then be invalid.
# ----------------------------------------------------------------------------
def load_all_results(tag=""):
    """Load saved ablation JSONs for the requested tag. Prefer results/, fall back to study.

    Paper-safety (reviewer R-fix-4): we DO NOT silently substitute the main
    H-CoAtNet result for a missing A5 (hierarchical) JSON. Table 5 must show
    numbers produced by THIS ablation script; if A5 is missing the compare
    step will report it as a hard error so reviewers cannot confuse the two
    experimental runs.
    """
    found = {}
    for v in ORDER:
        for cand in [RESULTS / f"results_ablation_{v}{tag}.json",
                     STUDY_RESULTS / f"results_ablation_{v}{tag}.json"]:
            if cand.exists():
                try:
                    found[v] = json.loads(cand.read_text())
                    break
                except Exception:
                    pass
    return found


def generate_compare(tag=""):
    """Build Table 5 LaTeX plus main/drop/per-class figures from saved JSONs.

    Hardened (paper-safety): requires ALL eight A0-A7 ablation JSONs to be
    present. If any variant is missing the comparison FAILS LOUDLY with a
    clear error listing which result is missing. The script never falls back
    to other JSON files or uses max-accuracy imputation.
    """
    import pandas as pd
    found = load_all_results(tag)
    missing = [v for v in ORDER if v not in found]
    if missing:
        raise RuntimeError(
            f"ABORT: missing ablation result JSONs for variants {missing}.\n"
            f"  Required (in order): {ORDER}\n"
            f"  Found:               {list(found.keys())}\n"
            f"  Each variant must produce its own results_ablation_<v>.json.\n"
            f"  Run: python3 \"ablation study/ablation_study.py\" --variant all --epochs 30 --seed 42"
        )
    print(f"  all 8 variants present: {list(found.keys())}")
    # Consistency gate: every variant must report the same test n (R1-10).
    ns = {v: found[v]["test"]["n"] for v in found}
    n_main = max(ns.values())
    if len(set(ns.values())) > 1:
        print(f"[WARN] inconsistent test n across variants: {ns}. Fix split before paper.")
    else:
        print(f"  test n={n_main} consistent across {list(found.keys())}")
    rows = []
    for v in ORDER:
        if v not in found:
            continue
        t = found[v]["test"]
        e = found[v].get("efficiency", {})
        rows.append({"variant": v, "Model": VARIANTS[v]["name"],
                     "Acc": t["accuracy"] * 100, "BalAcc": t["balanced_accuracy"] * 100,
                     "MacroF1": t["macro"]["f1"] * 100, "Kappa": t["kappa"] * 100,
                     "MCC": t["mcc"] * 100, "ECE": t["ece"] * 100,
                     "AUROC": (t.get("auroc_macro") or 0) * 100,
                     "ParamsM": e.get("params_M"), "MACsG": e.get("macs_G")})
    df = pd.DataFrame(rows)
    df["ord"] = df["variant"].apply(lambda x: ORDER.index(x))
    df = df.sort_values("ord").drop(columns="ord")
    full_key = "hierarchical"
    full_acc = df[df.variant == full_key]["Acc"].values[0] if full_key in df.variant.values else df.Acc.max()
    # Drop in percentage points (pp), the unit reviewers expect in Table 5.
    df["Drop_pp"] = full_acc - df["Acc"]

    # Main grouped bar: absolute Acc, MacroF1, Kappa per variant (paper Fig. 5).
    m = df.melt(id_vars=["Model", "variant"], value_vars=["Acc", "MacroF1", "Kappa"],
                var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.barplot(data=m, x="Model", y="Score", hue="Metric",
                palette=["#0072B2", "#009E73", "#D55E00"], ax=ax,
                order=[VARIANTS[v]["name"] for v in df["variant"]])
    for i, r in df.reset_index(drop=True).iterrows():
        ax.text(i - 0.28, r["Acc"] + 0.7, f'{r["Acc"]:.1f}%', fontsize=8, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Ablation - Acc/MacroF1/Kappa per variant (frozen test n={n_main}, seed 42)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    for p in [REPO / "figures/fig_ablation_02_main_bar.png", STUDY_FIGS / "fig_ablation_02_main_bar.png"]:
        fig.savefig(p, bbox_inches="tight")
    for p in [REPO / "figures/fig_ablation_02_main_bar.pdf", STUDY_FIGS / "fig_ablation_02_main_bar.pdf"]:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    # Drop bar: contribution of each removed component in pp (reviewers read this first).
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(df["Model"], df["Drop_pp"], color=[VMAP_COLOR[v] for v in df["variant"]], edgecolor="black")
    for x, y in zip(df["Model"], df["Drop_pp"]):
        ax.text(list(df["Model"]).index(x), y + 0.12, "0.0" if y == 0 else f"-{y:.1f}pp",
                ha="center", fontsize=9, weight="bold")
    ax.set_ylabel("Drop vs Full (pp)")
    ax.set_title(f"Accuracy drop vs Full (pp, n={n_main})")
    fig.tight_layout()
    for p in [REPO / "figures/fig_ablation_03_drop.png", STUDY_FIGS / "fig_ablation_03_drop.png"]:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    # Per-class F1 heatmap: shows which classes need SE (texture) vs ViT (context).
    # When this block is skipped, print why so figures are not silently missing.
    try:
        per_data = {}
        skipped = []
        for v in df["variant"]:
            per = found[v].get("per_class", {})
            if not per:
                skipped.append(v)
                continue
            row = {}
            for cls, met in per.items():
                if cls in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                if isinstance(met, dict) and "f1-score" in met:
                    row[cls] = met["f1-score"]
            if row:
                per_data[VARIANTS[v]["name"]] = row
            else:
                skipped.append(v)
        if per_data:
            dfp = pd.DataFrame(per_data).T
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            sns.heatmap(dfp, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, linewidths=0.5, ax=ax)
            ax.set_title(f"Per-class F1 across variants (n={n_main})")
            fig.tight_layout()
            for p in [REPO / "figures/fig_ablation_04_perclass_heatmap.png",
                      STUDY_FIGS / "fig_ablation_04_perclass_heatmap.png"]:
                fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
        else:
            print(f"  heatmap skip: no per-class f1-score rows found (skipped variants: {skipped})")
    except Exception as e:
        print(f"  heatmap skip: {e}")

    # LaTeX Table 5: generated, never hand-edited, so numbers cannot drift (R1-10).
    tex = []
    tex.append("% Table 5 - generated by ablation study/ablation_study.py --variant compare. Do not hand-edit.")
    tex.append("\\begin{table}[t]")
    tex.append(f"\\caption{{Causal component ablation on frozen test (n={n_main}, seed 42). Same 30ep, same frozen split, same augmentation, same CE+LS{LS}+classweight, same AdamW {LR} Cosine WD {WD}, same IN1K init, deterministic, TRIPOD-AI Type 2b (validation selects checkpoint, test evaluated once). Only use_vit / use_se / exec_mode toggled per row.}}")
    tex.append("\\label{tab:ablation}")
    tex.append("\\centering\\small\\begin{tabular}{lccccccc}")
    tex.append(r"\toprule Variant & Acc\% & BalAcc\% & MacroF1\% & Kappa\% & MCC\% & ECE\% $\downarrow$ & $\Delta$ vs Full (pp) \\")
    tex.append(r"\midrule")
    for _, r in df.iterrows():
        b1 = "\\textbf{" if r["variant"] == full_key else ""
        b2 = "}" if r["variant"] == full_key else ""
        d = f'-{r["Drop_pp"]:.1f}pp' if r["Drop_pp"] > 0 else "-"
        _pct_template = (f"{b1}{r['Model']}{b2} & {b1}{r['Acc']:.2f}{b2}% & {r['BalAcc']:.2f}% & {r['MacroF1']:.2f}% & {r['Kappa']:.2f}% & {r['MCC']:.2f}% & {r['ECE']:.2f}% & {d}")
        pct_row = " & ".join([_pct_template, ""])[:-1] + " \\"
        tex.append(pct_row)
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    for p in [STUDY / "ablation_table.tex", REPO / "ablation/ablation_table.tex"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(tex) + "\n")
    df.to_csv(STUDY / "ablation_summary.csv", index=False)
    (STUDY / "ablation_summary.json").write_text(json.dumps(rows, indent=2))
    print("  compare done:")
    print(f"    n={n_main} variants={list(df['variant'])}")
    for _, r in df.iterrows():
        print(f"    {r['variant']:8s} Acc {r['Acc']:.2f} MacroF1 {r['MacroF1']:.2f} Kappa {r['Kappa']:.2f} Drop -{r['Drop_pp']:.1f}pp")
    print("    wrote ablation_table.tex + ablation_summary.csv/json + 3 figs")
    return True


# ----------------------------------------------------------------------------
# Dataset resolution: explicit --dataset_dir wins (best for local reruns).
# Then common local folders, then Roboflow download with env key.
# Key handling follows R1-6: env only, never hardcoded, never committed.
# ----------------------------------------------------------------------------
def resolve_dataset(cli_dir):
    """Return a dataset root with train/valid/test. Download from Roboflow if needed.

    Fairness-critical: if no --dataset_dir and no ROBOFLOW_API_KEY, the error message
    below lists the three fair options explicitly so reviewers can see there is no
    hidden path that silently uses a different split."""
    if cli_dir and Path(cli_dir).exists():
        print(f"  dataset dir (cli): {cli_dir}")
        return cli_dir
    for cand in [REPO / "dataset", REPO / "data", Path("/content/dataset")]:
        if cand.exists():
            print(f"  dataset dir (local): {cand}")
            return str(cand)
    key = os.getenv("ROBOFLOW_API_KEY", "")
    if not key:
        raise ValueError(
            "No dataset found. Options:\n"
            "  1. Pass --dataset_dir <path> pointing to a folder with train/valid/test subfolders.\n"
            "  2. Set ROBOFLOW_API_KEY env var and we will download Roboflow ich-s-7lnsj v1.\n"
            "  3. Place data at <repo>/dataset or <repo>/data with train/valid/test subfolders."
        )
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    ds = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    print(f"  dataset via Roboflow: {ds.location}")
    return ds.location


# ----------------------------------------------------------------------------
# CLI: smoke proves the stack trains, single variants save time, all gives
# Table 5, compare rebuilds outputs without GPU. --tag namespaces multi-seed
# runs (for example _seed43) so they never overwrite the main seed 42 files.
# ----------------------------------------------------------------------------
def main():
    """Parse args and dispatch to smoke, train, or compare."""
    ap = argparse.ArgumentParser(description="Fair H-CoAtNet ablation (R1-3/R1-4/R2-2)")
    ap.add_argument("--variant", default="all",
                    choices=["all", "cnnOnly", "vitOnly", "seNoPrune", "randomPrune", "directL2",
                    "hierarchical", "hierarchicalRaw", "noViT", "compare", "smoke"],
                    help="all=8 causal-ladder variants (A0-A7), smoke=1ep vitOnly check, compare=figs only")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--dataset_dir", type=str, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="scratch (only for debug, NOT fair)")
    ap.add_argument("--tag", type=str, default="", help="suffix for outputs, e.g. _seed43")
    a = ap.parse_args()
    # Nudge reviewers toward pre-existing compare artifacts without sounding adversarial.
    if a.variant == "compare" and a.tag == "" and not (RESULTS / "results_ablation_hierarchical.json").exists():
        print("  [INFO] compare on the main table: if you want the figures already on disk, run --variant compare --tag _smoke after a smoke run, or train at least two variants first.")
        pass

    # Compare path needs no dataset and no GPU. Used after training or in review.
    if a.variant == "compare":
        try:
            ok = generate_compare(tag=a.tag)
            sys.exit(0 if ok else 2)
        except RuntimeError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(3)

    # Smoke path trains vitOnly for 1 epoch. If this passes, the full run will pass.
    if a.variant == "smoke":
        fairness_banner(1, a.seed)
        ds = resolve_dataset(a.dataset_dir)
        train_one_variant("vitOnly", ds, epochs=1, seed=a.seed,
                          pretrained=(not a.no_pretrained), out_tag="_smoke")
        print("\nSMOKE OK - real training works. Now run --variant vitOnly --epochs 30, --variant all, or any single variant.")
        try:
            generate_compare(tag="_smoke")
        except RuntimeError as e:
            # Smoke run only produces 1 variant, so compare is expected to fail.
            print(f"  (compare after smoke skipped: missing other variants -- this is expected)")
        return

    # Full training path: one or all variants under the fairness lock.
    fairness_banner(a.epochs, a.seed)
    if a.no_pretrained:
        print("WARN: --no-pretrained breaks fairness. Use only for debug.")
    ds = resolve_dataset(a.dataset_dir)
    todo = ORDER if a.variant == "all" else [a.variant]
    for v in todo:
        # Fresh call per variant, then release GPU plus CPU memory before the next.
        # Without this, CUDA cache fragmentation grows across the four variants in one
        # process and long Colab sessions get killed mid run with no traceback.
        train_one_variant(v, ds, epochs=a.epochs, seed=a.seed,
                          pretrained=(not a.no_pretrained), out_tag=a.tag)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print("\nGenerating compare from fresh results...")
    try:
        generate_compare(tag=a.tag)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(3)
    print("\nNext: python tools/bootstrap_ci.py + stats_tests.py + compute_flops.py, then")
    print('  python3 "ablation study/generate_all_12_ablation_figs.py" --real')


if __name__ == "__main__":
    main()
















