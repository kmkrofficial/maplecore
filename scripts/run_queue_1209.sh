#!/usr/bin/env bash
# GPU queue after the MAPLE-Embed v2 BEIR run (strictly sequential, all logged):
#  1) control: MAPLE-Embed trained on human gold-evidence targets (seed 42) -> bench 01 + 09
#  2) efficiency of every method (bench 11)    3) latency scaling of the search strategies (bench 02)
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
R=benchmarks/results
until grep -q "V2 EVAL DONE" logs/embed_v2_eval.log; do sleep 30; done
echo "[queue] gold-control train start $(date +%T)"
"$PY" scripts/train_biencoder.py --train-oracles data/oracle/hotpotqa_train.json data/oracle/narrativeqa_train.json \
  --output models/maple_embed_gold_b1536_s42 --seed 42 --epochs 2 --target-source gold --max-gpu-mem-frac 0.90 \
  && echo "[queue] gold-control train ok $(date +%T)"
mkdir -p $R/embed_gold/s42
"$PY" benchmarks/01_recall_enhanced.py --embedding-model models/maple_embed_gold_b1536_s42 \
  && cp $R/retrieval_quality.json $R/retrieval_quality_per_item.json $R/embed_gold/s42/ && echo "[queue] gold 01 ok $(date +%T)"
"$PY" benchmarks/09_multihop_rerank.py --rankers dense dense:models/maple_embed_gold_b1536_s42 \
  && cp $R/multihop_rerank.json $R/multihop_rerank_embed_gold.json && echo "[queue] gold 09 ok $(date +%T)"
echo "[queue] efficiency start $(date +%T)"
"$PY" benchmarks/11_efficiency.py --rankers bm25 dense hybrid cross_encoder maple:a5 maple:a6 \
  --model a5=models/gate_a/A5_b1536_s42.pt --model a6=models/gate_a/A6_b1536_s42.pt \
  && echo "[queue] efficiency ok $(date +%T)"
echo "[queue] latency start $(date +%T)"
"$PY" benchmarks/02_latency_scaling.py --devices cuda cpu --model models/gate_a/A5_b1536_s42.pt \
  && echo "[queue] latency ok $(date +%T)"
echo "QUEUE DONE $(date +%T)"
