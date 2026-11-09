#!/usr/bin/env bash
# MAPLE-Embed v2: as run_embed.sh plus in-batch cross-document negatives. 3 seeds, teacher budget 1536.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
for s in 42 43 44; do
  echo "[embed-v2] seed $s start $(date +%T)"
  if "$PY" scripts/train_biencoder.py --train-oracles data/oracle/hotpotqa_train.json data/oracle/narrativeqa_train.json \
       --output models/maple_embed_xneg_b1536_s$s --seed $s --epochs 2 --cross-negatives --max-gpu-mem-frac 0.90; then
    echo "[embed-v2] seed $s ok $(date +%T)"
  else
    echo "[embed-v2] seed $s FAILED $(date +%T)"
  fi
done
echo "EMBED V2 DONE"
