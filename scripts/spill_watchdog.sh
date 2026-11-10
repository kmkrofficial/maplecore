#!/usr/bin/env bash
# Stops the reader benchmark if Windows starts paging a python process's VRAM into shared memory.
# Reads the sampler CSV (scripts/gpu_mem_sampler.ps1). Usage: bash scripts/spill_watchdog.sh <csv> <done-log> [limit_mb]
CSV=$1; DONE_LOG=$2; LIMIT=${3:-1000}
cd "$(dirname "$0")/.."
while ! grep -q "E2E DONE" "$DONE_LOG" 2>/dev/null; do
  last=$(grep ',python,' "$CSV" | tail -1)
  shared=$(echo "$last" | awk -F, '{print $5}')
  if [ -n "$shared" ] && [ "$shared" -gt "$LIMIT" ]; then
    echo "[watchdog] $(date +%T) shared GPU memory $shared MB > $LIMIT MB ($last): stopping bench 10" | tee -a "$DONE_LOG"
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*10_e2e_longbench*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    exit 1
  fi
  sleep 15
done
