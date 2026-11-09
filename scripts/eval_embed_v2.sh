#!/usr/bin/env bash
# MAPLE-Embed v2 (cross-document negatives): held-out (01), multi-hop (09), BEIR (08). One GPU job at a time.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
R=benchmarks/results
for s in 42 43 44; do
  out=$R/embed_xneg/s$s; mkdir -p "$out"
  echo "[v2-eval] 01 s$s start $(date +%T)"
  "$PY" benchmarks/01_recall_enhanced.py --embedding-model models/maple_embed_xneg_b1536_s$s \
    && cp $R/retrieval_quality.json $R/retrieval_quality_per_item.json "$out"/ && echo "[v2-eval] 01 s$s ok $(date +%T)"
done
echo "[v2-eval] 09 start $(date +%T)"
"$PY" benchmarks/09_multihop_rerank.py --rankers bm25 dense dense:models/maple_embed_xneg_b1536_s42 dense:models/maple_embed_xneg_b1536_s43 dense:models/maple_embed_xneg_b1536_s44 \
  && cp $R/multihop_rerank.json $R/multihop_rerank_embed_xneg.json && echo "[v2-eval] 09 ok $(date +%T)"
while pgrep -f run_beir_bm25check >/dev/null 2>&1 || ps -ef | grep -v grep | grep -q run_beir_bm25check; do sleep 30; done  # bench 08 shares beir_rerank.json
echo "[v2-eval] 08 start $(date +%T)"
"$PY" benchmarks/08_beir_rerank.py --rankers bm25 dense dense:models/maple_embed_xneg_b1536_s42 dense:models/maple_embed_xneg_b1536_s43 dense:models/maple_embed_xneg_b1536_s44 \
  && cp $R/beir_rerank.json $R/beir_rerank_embed_xneg.json && echo "[v2-eval] 08 ok $(date +%T)"
echo "V2 EVAL DONE"
