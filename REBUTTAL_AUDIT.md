# Reviewer Rebuttal — Code Audit (2026-09-06)

Cross-check of every reviewer concern in `rebuttal_comments.md` against the
actual files in this repo (not against the Drive copies referenced inside the
rebuttal, which are not part of the released artefacts).

Legend: OK = present and correct in repo. GAP = claimed in prose, missing here.

## Reviewer 1

| # | Concern | Status | Evidence |
|---|---|---|---|
| R1-1 | Frozen split n=158 + per-class counts | OK | `splits/seed42_indices.json`, `splits/test_per_class.csv` |
| R1-2 | Test untouched during training | OK | `H-CoAtNet/proposed_method/train_h_coatnet.py:301,:365` |
| R1-3 | Architecture spec matches code | OK | `[3,3,9,3]`, 2 ViT, 49->36->24 verified |
| R1-4 | Forward-only token importance | OK | `HierarchicalSE.forward` (train_h_coatnet.py:53-81) |
| R1-5 | Dedup audit (MD5/pHash/SSIM/cross-split) | OK | `tools/dedup_audit.py` |
| R1-6 | Env-only Roboflow key | OK | `os.getenv` in all training scripts |
| R1-7 | Reproducible protocol table | OK | LR/WD/BS/LS constants locked in code |
| R1-8 | Bootstrap CI + McNemar/DeLong | OK | `tools/bootstrap_ci.py`, `tools/stats_tests.py` |
| R1-9 | Efficiency, claim withdrawn | OK | `tools/compute_flops.py`, claim deleted from prose |
| R1-10 | Single source of numbers | OK | JSON schema in all training scripts |
| R1-11 | Grad-CAM, clinical moderation | OK | `tools/gradcam.py` |
| R1-12 | Canonical cites, no SOTA in code | OK | Citations in `train_h_coatnet.py:87` |
| R1-13 | Splits/weights/configs released | PARTIAL | `splits/` present; weights in Drive only |
| R1-14 | Naming (H-CoAtNet) + tensor dims | OK | Verified across `*.py` |

## Reviewer 2

| # | Concern | Status | Evidence |
|---|---|---|---|
| R2-1 | Image sources / ethics / labels | GAP | Manuscript only; `splits/test_per_class.csv` covers counts |
| R2-2 | Novelty vs standard CoAtNet | OK (after fix) | Renamed `HCoAtNet` vs `CoAtNet`; ablation fixed today |
| R2-3 | Abstract/Conclusion numbers + name | OK | Uniform "H-CoAtNet"; numbers from compare.json |
| R2-4 | Bootstrap CIs, k-fold | PARTIAL | Bootstrap done; k-fold 5-fold not run |
| R2-5 | 30 epochs from scratch | OK | Acknowledged; matched-pretrain re-run still pending |
| R2-6 | Wrong citation [9] | OK | Zero Hauser/Chanda hits in *.py |
| R2-7 | Irrelevant refs (1998/2002/2010) | OK | Canonical cites only in *.py |
| Minor | Typos, Related split, CC-BY, etc. | GAP | Manuscript only |

## Gaps in this repo vs rebuttal prose

1. `results/results_*.json` (7 per-model) — claimed but absent locally. Only
   `results/model_training.txt` and `efficiency_T4_GFT_patch.txt` are here.
   Reviewers reading the repo (not the Drive backup) will see this.

2. `splits/SHA256SUM` and `splits/datasheet.md` — claimed, absent.

3. `figures/` — empty locally. All 119 figures live on Drive only.

4. `ablation/` — `ablation.py` exists but `results_ablation_*.json`,
   `ablation_table.tex`, `ablation_summary.csv/json`, `fig_ablation_*` not
   generated yet (require full 30-epoch run).

5. K-fold cross-validation (R2-4 second half) — not run.

6. Source list / expert annotations / IRB basis (R1-5, R1-6, R2-1) — manuscript
   only, no code path possible.

## Recommendation

To make the released artefacts match the rebuttal text, regenerate locally:

```
# After training finishes
python tools/freeze_split.py --seed 42          # writes splits/seed42_indices.json (already there) + SHA + datasheet
python tools/bootstrap_ci.py --results results/results_final.json
python tools/stats_tests.py
python tools/compute_flops.py --all
python tools/gradcam.py --model hcoatnet --n 6
python tools/generate_tables.py --all
python "ablation study/ablation_study.py" --variant all --epochs 30 --seed 42
python "ablation study/ablation_study.py" --variant compare
```

Then commit all generated JSON/PNG/PDF to the repo (currently only the code is
tracked; reviewers expecting artefacts will find an empty `figures/` and
mostly empty `results/`).
