#!/usr/bin/env bash
# Gate B GPU queue (after ME-R): MAPLE heads A5/A6 on multi-hop (09) and BEIR (08); then the reader-fixed
# end-to-end benchmark (10): an 8-question smoke run, and the full run only if the smoke run succeeds.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
# Repo-local Hugging Face caches (on D:), offline: the user cache on C: is nearly full
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
R=benchmarks/results
HEADS="--model a5=models/gate_a/A5_b1536_s42.pt --model a6=models/gate_a/A6_b1536_s42.pt"
until grep -q "ME-R DONE" logs/embed_r.log; do sleep 30; done
echo "[gateb] 09 heads start $(date +%T)"
"$PY" benchmarks/09_multihop_rerank.py --rankers bm25 dense hybrid maple:a5 maple:a6 $HEADS \
  && cp $R/multihop_rerank.json $R/multihop_rerank_heads.json && echo "[gateb] 09 heads ok $(date +%T)"
echo "[gateb] 08 heads start $(date +%T)"
"$PY" benchmarks/08_beir_rerank.py --rankers bm25 dense hybrid maple:a5 maple:a6 $HEADS \
  && cp $R/beir_rerank.json $R/beir_rerank_heads.json && echo "[gateb] 08 heads ok $(date +%T)"
E2E_RANKERS="bm25 dense hybrid cross_encoder maple:a5 maple:a6 dense:models/maple_embed_r_b1536_s42"
echo "[gateb] 10 smoke start $(date +%T)"
if "$PY" benchmarks/10_e2e_longbench.py --rankers $E2E_RANKERS $HEADS --limit 8; then
  cp $R/e2e_reader_fixed.json $R/e2e_reader_fixed_smoke.json; echo "[gateb] 10 smoke ok $(date +%T)"
  echo "[gateb] 10 full start $(date +%T)"
  "$PY" benchmarks/10_e2e_longbench.py --rankers $E2E_RANKERS $HEADS \
    && cp $R/e2e_reader_fixed.json $R/e2e_reader_fixed_full.json && echo "[gateb] 10 full ok $(date +%T)"
else
  echo "[gateb] 10 smoke FAILED $(date +%T)"
fi
echo "GATEB QUEUE DONE $(date +%T)"
