#!/usr/bin/env bash
# Train a grid of MAPLE v2 variants (3 seeds x 2 teacher budgets) and evaluate each on the
# held-out oracle sets with benchmark 01. Resumable: finished (variant, budget, seed) runs are skipped.
#
# Usage: bash scripts/run_gate.sh GATE [SEEDS...]       e.g.  bash scripts/run_gate.sh gate_a
# Output: models/<gate>/<variant>_b<budget>_s<seed>.pt
#         benchmarks/results/<gate>/<variant>_b<budget>_s<seed>/{retrieval_quality*.json,recall_grouped_bar.png}
#         logs/<gate>/*.log   (and a runs/ registry entry per training / benchmark call)
# Then:   python benchmarks/aggregate_results.py --root benchmarks/results/<gate>
set -euo pipefail
cd "$(dirname "$0")/.."

GATE="${1:?usage: run_gate.sh GATE [SEEDS...]}"; shift || true
SEEDS=("$@"); [ "${#SEEDS[@]}" -eq 0 ] && SEEDS=(42 43 44)
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8

# variant grid: "id model_kind target_source temperature datasets(both|hotpot)"
variants() {
  case "$1" in
    gate_a)
      echo "A1 residual all_layers 1.0 both"
      echo "A2 residual all_layers 0.5 both"
      echo "A3 residual last 1.0 both"
      echo "A4 mlp all_layers 1.0 both"
      echo "A5 residual gold 1.0 hotpot" ;;
    *) echo "unknown gate '$1' (define it in scripts/run_gate.sh)" >&2; return 1 ;;
  esac
}
oracle_dir() { if [ "$1" = 4096 ]; then echo data/oracle/t4096; else echo data/oracle; fi; }
stamp() { date +%H:%M:%S; }

R="benchmarks/results/$GATE"; M="models/$GATE"; L="logs/$GATE"
mkdir -p "$R" "$M" "$L"

echo "[$(stamp)] caches"
"$PY" scripts/embedding_cache.py data/oracle/*_train.json data/oracle/*_test.json data/oracle/*_validation.json \
  data/oracle/t4096/*_train.json data/oracle/t4096/*_test.json data/oracle/t4096/*_validation.json > "$L/caches.log" 2>&1

VARIANTS=$(variants "$GATE")
for B in 1536 4096; do
  DIR=$(oracle_dir "$B")
  while read -r V KIND SRC TEMP DS; do
    for S in "${SEEDS[@]}"; do
      RUN="${V}_b${B}_s${S}"; OUT="$R/$RUN"
      if [ -f "$OUT/retrieval_quality.json" ]; then echo "[$(stamp)] skip $RUN (done)"; continue; fi
      if [ "$DS" = hotpot ]; then ORACLES="$DIR/hotpotqa_train.json"; else ORACLES="$DIR/hotpotqa_train.json $DIR/narrativeqa_train.json"; fi
      echo "[$(stamp)] train $RUN ($KIND, $SRC, T=$TEMP, $DS)"
      "$PY" scripts/train_v2.py --train-oracles $ORACLES --output "$M/$RUN.pt" --model-kind "$KIND" \
        --target-source "$SRC" --temperature "$TEMP" --seed "$S" > "$L/train_$RUN.log" 2>&1
      echo "[$(stamp)]   $(grep -h 'best ' "$L/train_$RUN.log" | tail -n 1)"
      "$PY" benchmarks/01_recall_enhanced.py --model "$M/$RUN.pt" --oracle-dir "$DIR" > "$L/bench01_$RUN.log" 2>&1
      mkdir -p "$OUT"
      for f in retrieval_quality.json retrieval_quality_per_item.json recall_grouped_bar.png; do
        if [ -e "benchmarks/results/$f" ]; then mv "benchmarks/results/$f" "$OUT/"; fi
      done
    done
  done <<< "$VARIANTS"
done

echo "[$(stamp)] aggregate"
"$PY" benchmarks/aggregate_results.py --root "$R" > "$L/aggregate.log" 2>&1
echo "[$(stamp)] GATE DONE: $R/SUMMARY.md"
