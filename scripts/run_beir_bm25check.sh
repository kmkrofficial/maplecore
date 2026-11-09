#!/usr/bin/env bash
# Check: corpus-fitted BM25 re-ranking BM25's own top-100 must reproduce the first-stage nDCG@10.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
"$PY" benchmarks/08_beir_rerank.py --rankers bm25 hybrid --device cpu \
  && cp benchmarks/results/beir_rerank.json benchmarks/results/beir_rerank_bm25fix.json && echo "BM25 CHECK DONE" || echo "BM25 CHECK FAILED"
