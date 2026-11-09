#!/usr/bin/env bash
# Train MAPLE-Embed (attention-distilled bi-encoder), 3 seeds, teacher budget 1536.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
for s in 42 43 44; do
  echo "[embed] seed $s start $(date +%T)"
  if "$PY" scripts/train_biencoder.py --train-oracles data/oracle/hotpotqa_train.json data/oracle/narrativeqa_train.json \
       --output models/maple_embed_b1536_s$s --seed $s --epochs 2; then
    echo "[embed] seed $s ok $(date +%T)"
  else
    echo "[embed] seed $s FAILED $(date +%T)"
  fi
done
echo "EMBED DONE"
