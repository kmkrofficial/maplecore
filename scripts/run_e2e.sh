#!/usr/bin/env bash
# Reader-fixed end-to-end QA (bench 10): smoke run (8 questions per task), then the full run if it succeeds.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
# Repo-local Hugging Face caches (on D:), offline: the user cache on C: is nearly full
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
R=benchmarks/results
ARGS="--rankers bm25 dense hybrid cross_encoder maple:a5 maple:a6 dense:models/maple_embed_r_b1536_s42 --model a5=models/gate_a/A5_b1536_s42.pt --model a6=models/gate_a/A6_b1536_s42.pt --max-gpu-mem-frac 0.80"
echo "[e2e] smoke start $(date +%T)"
if "$PY" benchmarks/10_e2e_longbench.py $ARGS --limit 8; then
  cp $R/e2e_reader_fixed.json $R/e2e_reader_fixed_smoke.json; echo "[e2e] smoke ok $(date +%T)"
  [ -f logs/e2e_SMOKE_ONLY ] && { echo "E2E DONE (smoke only) $(date +%T)"; exit 0; }
  echo "[e2e] full start $(date +%T)"
  "$PY" benchmarks/10_e2e_longbench.py $ARGS && cp $R/e2e_reader_fixed.json $R/e2e_reader_fixed_full.json \
    && echo "[e2e] full ok $(date +%T)"
else
  echo "[e2e] smoke FAILED $(date +%T)"
fi
echo "E2E DONE $(date +%T)"
