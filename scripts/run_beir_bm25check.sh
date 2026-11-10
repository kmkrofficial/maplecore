#!/usr/bin/env bash
# Check: corpus-fitted BM25 re-ranking BM25's own top-100 must reproduce the first-stage nDCG@10.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
# Repo-local Hugging Face caches (on D:), offline: the user cache on C: is nearly full
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
"$PY" benchmarks/08_beir_rerank.py --rankers bm25 hybrid --device cpu \
  && cp benchmarks/results/beir_rerank.json benchmarks/results/beir_rerank_bm25fix.json && echo "BM25 CHECK DONE" || echo "BM25 CHECK FAILED"
