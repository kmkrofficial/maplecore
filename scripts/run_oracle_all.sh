#!/usr/bin/env bash
# Generate every oracle file used by the experiment, sequentially on one GPU.
# Usage: bash scripts/run_oracle_all.sh [TRAIN_N] [HELDOUT_N]
#   Budget 1536 (v0.2.0 reproduction) -> data/oracle/*.json
#   Budget 4096 (longer teacher context) -> data/oracle/t4096/*.json
# Re-running resumes each file where it stopped. Uses the fast attention path
# (SDPA + last-row capture), validated against eager: scripts/compare_oracles.py.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_N="${1:-2000}"
HELDOUT_N="${2:-500}"
PY="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY=python

export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
mkdir -p logs data/oracle/t4096

run() {  # name, command...
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@" --resume > "logs/oracle_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name ($(tail -n 3 "logs/oracle_${name}.log" | head -n 1))"
}

for T in 1536 4096; do
  DIR=data/oracle; BS=8
  [ "$T" = 4096 ] && DIR=data/oracle/t4096 && BS=4   # throughput saturates at bs>=2-4
  run "hotpot_val_$T"   "$PY" scripts/generate_oracle_hotpot.py --split validation --max-samples "$HELDOUT_N" --max-tokens "$T" --batch-size "$BS" --output "$DIR/hotpotqa_validation.json"
  run "nqa_test_$T"     "$PY" scripts/generate_oracle.py        --split test       --max-samples "$HELDOUT_N" --max-tokens "$T" --batch-size "$BS" --output "$DIR/narrativeqa_test.json"
  run "hotpot_train_$T" "$PY" scripts/generate_oracle_hotpot.py --split train      --max-samples "$TRAIN_N"   --max-tokens "$T" --batch-size "$BS" --output "$DIR/hotpotqa_train.json"
  run "nqa_train_$T"    "$PY" scripts/generate_oracle.py        --split train      --max-samples "$TRAIN_N"   --max-tokens "$T" --batch-size "$BS" --output "$DIR/narrativeqa_train.json"
done
echo "[$(date +%H:%M:%S)] ALL ORACLE FILES DONE"
