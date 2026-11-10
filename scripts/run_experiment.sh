#!/usr/bin/env bash
# Train MAPLE (3 seeds x 2 teacher context budgets) and run the benchmark suite.
# Results land in benchmarks/results/<run>/ ; aggregate with
#   python benchmarks/aggregate_results.py
# Usage: bash scripts/run_experiment.sh [SEEDS...]   (default: 42 43 44)
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=("${@:-42 43 44}")
[ "$#" -eq 0 ] && SEEDS=(42 43 44)
PY="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY=python
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
R=benchmarks/results
mkdir -p logs models "$R"

stamp() { date +%H:%M:%S; }
collect() {  # move the given result files into $R/$1/
  local dest="$R/$1"; shift; mkdir -p "$dest"
  for f in "$@"; do for g in $R/$f; do if [ -e "$g" ]; then mv "$g" "$dest/"; fi; done; done
}
oracle_dir() { [ "$1" = 4096 ] && echo data/oracle/t4096 || echo data/oracle; }

# ---- 1. Train + held-out retrieval quality (every seed, both budgets) ----
for B in 1536 4096; do
  for S in "${SEEDS[@]}"; do
    RUN="b${B}_s${S}"; M="models/maple_${RUN}.pth"
    echo "[$(stamp)] train $RUN"
    "$PY" scripts/train_generalist.py --seed "$S" --oracle-dir "$(oracle_dir $B)" --output "$M" > "logs/train_${RUN}.log" 2>&1
    echo "[$(stamp)]   $(grep 'Best val' "logs/train_${RUN}.log")"
    echo "[$(stamp)] 01 retrieval quality $RUN"
    "$PY" benchmarks/01_recall_enhanced.py --model "$M" --oracle-dir "$(oracle_dir $B)" > "logs/bench01_${RUN}.log" 2>&1
    collect "$RUN" retrieval_quality.json retrieval_quality_per_item.json recall_grouped_bar.png
  done
done

# Default checkpoint used by examples / benchmarks: v0.2.0-comparable budget, first seed
cp "models/maple_b1536_s${SEEDS[0]}.pth" models/maple_generalist.pth

# ---- 2. Long-context, needle, chunk ablation (first seed of each budget) ----
for B in 1536 4096; do
  RUN="b${B}_s${SEEDS[0]}"; M="models/maple_${RUN}.pth"
  echo "[$(stamp)] 07 long context $RUN"
  "$PY" benchmarks/07_long_context.py --model "$M" --max-questions 300 > "logs/bench07_${RUN}.log" 2>&1
  collect "$RUN" long_context.json
  echo "[$(stamp)] 03 needle $RUN"
  "$PY" benchmarks/03_needle_robust.py --model "$M" --trials 10 > "logs/bench03_${RUN}.log" 2>&1
  collect "$RUN" needle_robustness.json "needle_heatmap_*.png"
  echo "[$(stamp)] 05 chunk ablation $RUN"
  "$PY" benchmarks/05_ablation_chunking.py --model "$M" --max-questions 300 > "logs/bench05_${RUN}.log" 2>&1
  collect "$RUN" ablation_chunking.json ablation_tradeoff.png
done

# ---- 3. Latency scaling (model-agnostic beyond architecture) ----
echo "[$(stamp)] 02 latency scaling"
"$PY" benchmarks/02_latency_scaling.py > logs/bench02.log 2>&1 || echo "[$(stamp)]   02 FAILED (see logs/bench02.log)"
collect latency latency_results.json "latency_*.png"

echo "[$(stamp)] EXPERIMENT DONE"
