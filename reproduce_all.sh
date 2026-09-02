#!/bin/bash
# reproduce_all.sh — One-command A* reproduction (TRIPOD-AI Type 2b)
# Addresses R1-13 (reproducibility)

set -e
echo "H-CoAtNet — Reproduce All (A* Edition)"
echo "======================================="
echo "Seed: 42 | Dataset: ich-s-7lnsj v1 | 30 epochs | 7 models"

# Check API key
if [ "$ROBOFLOW_API_KEY" == "" ]; then
  echo "⚠️  Set ROBOFLOW_API_KEY: export ROBOFLOW_API_KEY='your_key'"
  echo "   Get key: https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj"
  exit 1
fi

# 1. Environment
pip install -r requirements.txt || pip install -r H-CoAtNet/requirements.txt
echo "✅ Env ready"

# 2. Freeze split + dedup audit (before training)
python tools/freeze_split.py --roboflow --seed 42 --out splits/seed42_indices.json
python tools/dedup_audit.py --dataset_dir $(python -c "import json; print(json.load(open('splits/seed42_indices.json'))['dataset_dir'])") --out results/dedup_report.json
echo "✅ Split frozen + dedup audited"

# 3. Train all 7 models (30 epochs each, ~3h total on T4)
# Note: Each train script saves best to results/ and evaluates test ONCE
for model in hcoatnet gft coatnet vit swin efficientnet cnn; do
  echo "Training $model ..."
  if [ "$model" == "hcoatnet" ]; then
    python H-CoAtNet/proposed_method/train_h_coatnet.py
  else
    python H-CoAtNet/baselines/train_${model}.py
  fi
done
echo "✅ All 7 models trained"

# 4. Efficiency
python tools/compute_flops.py --all --out results/efficiency.json
echo "✅ Efficiency table done"

# 5. Bootstrap CI + significance
python tools/bootstrap_ci.py --results results/results_final.json --n_bootstrap 1000
python tools/stats_tests.py --all results/results_*.json --reference results/results_hcoatnet.json --out results/significance.json
echo "✅ CIs + significance done"

# 6. Generate LaTeX tables
python tools/generate_tables.py --all results/results_*.json --out results/tables.tex
echo "✅ Tables generated"

# 7. Audit hash
echo "SHA256 audit:"
sha256sum results/*.json splits/*.json || shasum -a 256 results/*.json
echo ""
echo "Done. Compare results/tables.tex with manuscript Tables 8/9."
echo "If Abstract = Table 8 = Conclusion, you pass R1-10."
echo "Upload results/, splits/, and weights to Zenodo for DOI."
