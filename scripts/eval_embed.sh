#!/usr/bin/env bash
# Held-out retrieval quality (bench 01) with the base encoder and each MAPLE-Embed seed (cosine/dense rows).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
run() { # name encoder
  out=benchmarks/results/embed/$1; mkdir -p "$out"
  echo "[eval] $1 start $(date +%T)"
  if "$PY" benchmarks/01_recall_enhanced.py --embedding-model "$2"; then
    cp benchmarks/results/retrieval_quality.json benchmarks/results/retrieval_quality_per_item.json "$out"/
    echo "[eval] $1 ok $(date +%T)"
  else echo "[eval] $1 FAILED"; fi
}
run base BAAI/bge-small-en-v1.5
for s in 42 43 44; do run s$s models/maple_embed_b1536_s$s; done
echo "EVAL DONE"
