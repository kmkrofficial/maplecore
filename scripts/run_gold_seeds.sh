#!/usr/bin/env bash
# Gold-label control, seeds 43 and 44 (seed 42 ran 2026-11-09), after the reader run; held-out (01) + multi-hop (09).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
R=benchmarks/results
until grep -q "E2E DONE" logs/e2e.log 2>/dev/null; do sleep 30; done
for s in 43 44; do
  echo "[gold] train s$s start $(date +%T)"
  "$PY" scripts/train_biencoder.py --train-oracles data/oracle/hotpotqa_train.json data/oracle/narrativeqa_train.json \
    --output models/maple_embed_gold_b1536_s$s --seed $s --epochs 2 --target-source gold --max-gpu-mem-frac 0.80 \
    && echo "[gold] train s$s ok $(date +%T)"
  mkdir -p $R/embed_gold/s$s
  "$PY" benchmarks/01_recall_enhanced.py --embedding-model models/maple_embed_gold_b1536_s$s \
    && cp $R/retrieval_quality.json $R/retrieval_quality_per_item.json $R/embed_gold/s$s/ && echo "[gold] 01 s$s ok $(date +%T)"
done
"$PY" benchmarks/09_multihop_rerank.py --rankers dense dense:models/maple_embed_gold_b1536_s43 dense:models/maple_embed_gold_b1536_s44 \
  && cp $R/multihop_rerank.json $R/multihop_rerank_embed_gold_s43_s44.json && echo "[gold] 09 ok $(date +%T)"
echo "GOLD DONE $(date +%T)"
