#!/usr/bin/env bash
# Multi-hop re-ranking (bench 09): BM25 + base dense + MAPLE-Embed encoders (3 seeds).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
# Repo-local Hugging Face caches (on D:), offline: the user cache on C: is nearly full
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
cp benchmarks/results/multihop_rerank.json benchmarks/results/multihop_rerank_prev.json 2>/dev/null
"$PY" benchmarks/09_multihop_rerank.py --rankers bm25 dense dense:models/maple_embed_b1536_s42 dense:models/maple_embed_b1536_s43 dense:models/maple_embed_b1536_s44 \
  && cp benchmarks/results/multihop_rerank.json benchmarks/results/multihop_rerank_embed.json && echo "MULTIHOP EMBED DONE" || echo "MULTIHOP EMBED FAILED"
