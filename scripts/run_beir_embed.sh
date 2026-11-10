#!/usr/bin/env bash
# BEIR top-100 re-ranking (bench 08): BM25 + base dense + MAPLE-Embed encoders (3 seeds).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
# Repo-local Hugging Face caches (on D:), offline: the user cache on C: is nearly full
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
"$PY" benchmarks/08_beir_rerank.py --rankers bm25 dense dense:models/maple_embed_b1536_s42 dense:models/maple_embed_b1536_s43 dense:models/maple_embed_b1536_s44 \
  && cp benchmarks/results/beir_rerank.json benchmarks/results/beir_rerank_embed.json && echo "BEIR EMBED DONE" || echo "BEIR EMBED FAILED"
