#!/usr/bin/env bash
# ME-R (pre-registered in docs/private/decisions/maple_embed.md, 2026-11-10): top-2 layers + base KL 1.0, 3 seeds,
# then held-out (01), multi-hop (09) and BEIR (08). Strictly sequential on the GPU.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
R=benchmarks/results
for s in 42 43 44; do
  echo "[me-r] train s$s start $(date +%T)"
  "$PY" scripts/train_biencoder.py --train-oracles data/oracle/hotpotqa_train.json data/oracle/narrativeqa_train.json \
    --output models/maple_embed_r_b1536_s$s --seed $s --epochs 2 --train-last-layers 2 --base-kl 1.0 \
    --max-gpu-mem-frac 0.80 && echo "[me-r] train s$s ok $(date +%T)"
done
for s in 42 43 44; do
  mkdir -p $R/embed_r/s$s
  "$PY" benchmarks/01_recall_enhanced.py --embedding-model models/maple_embed_r_b1536_s$s \
    && cp $R/retrieval_quality.json $R/retrieval_quality_per_item.json $R/embed_r/s$s/ && echo "[me-r] 01 s$s ok $(date +%T)"
done
"$PY" benchmarks/09_multihop_rerank.py --rankers dense dense:models/maple_embed_r_b1536_s42 dense:models/maple_embed_r_b1536_s43 dense:models/maple_embed_r_b1536_s44 \
  && cp $R/multihop_rerank.json $R/multihop_rerank_embed_r.json && echo "[me-r] 09 ok $(date +%T)"
"$PY" benchmarks/08_beir_rerank.py --rankers dense dense:models/maple_embed_r_b1536_s42 dense:models/maple_embed_r_b1536_s43 dense:models/maple_embed_r_b1536_s44 \
  && cp $R/beir_rerank.json $R/beir_rerank_embed_r.json && echo "[me-r] 08 ok $(date +%T)"
echo "ME-R DONE $(date +%T)"
