#!/usr/bin/env bash
# MAPLE-Embed v2: as run_embed.sh plus in-batch cross-document negatives. 3 seeds, teacher budget 1536.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
# Repo-local Hugging Face caches (on D:), offline: the user cache on C: is nearly full
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
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
