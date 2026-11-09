#!/usr/bin/env bash
# Reduced efficiency + latency runs that fit before a fixed shutdown (see ledger ruling 2026-11-09 13:33).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
echo "[queue] efficiency start $(date +%T)"
"$PY" benchmarks/11_efficiency.py --rankers bm25 dense hybrid cross_encoder maple:a5 maple:a6 \
  --model a5=models/gate_a/A5_b1536_s42.pt --model a6=models/gate_a/A6_b1536_s42.pt \
  --n-queries 20 --max-seconds-per-query 2 \
  && echo "[queue] efficiency ok $(date +%T)"
echo "[queue] latency start $(date +%T)"
"$PY" benchmarks/02_latency_scaling.py --devices cuda cpu --model models/gate_a/A5_b1536_s42.pt \
  && echo "[queue] latency ok $(date +%T)"
echo "QUEUE DONE $(date +%T)"
