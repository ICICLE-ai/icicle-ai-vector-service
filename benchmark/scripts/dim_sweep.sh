#!/usr/bin/env bash
#
# Sweeps vector dimension at a fixed fetch_k to find where reranking starts
# degrading unrelated requests.
#
# Run benchmark/k6/dim_seed.js first to create the per-dimension collections.
#
#   BASE_URL=... TAPIS_TOKEN=... ./benchmark/scripts/dim_sweep.sh
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/results/dimsweep-$(date +%Y%m%d-%H%M%S)"
DIMS="${SWEEP_DIMS:-768,1024,1536,2048,3072,4096}"
mkdir -p "$OUT"

: "${BASE_URL:?set BASE_URL}"
: "${TAPIS_TOKEN:?set TAPIS_TOKEN}"

# Control first: the probe load with no rerank traffic at all. Every later run
# is compared against this.
echo "▶ control (no rerank)"
SWEEP_DIM=0 k6 run --summary-export "$OUT/control.json" \
  "$HERE/k6/dim_sweep.js" >"$OUT/control.log" 2>&1

for dim in ${DIMS//,/ }; do
  echo "▶ dim=$dim"
  SWEEP_DIM="$dim" k6 run --summary-export "$OUT/dim-$dim.json" \
    "$HERE/k6/dim_sweep.js" >"$OUT/dim-$dim.log" 2>&1
done

echo
python3 "$HERE/scripts/dim_report.py" "$OUT"   # prints the table and writes dim_sweep.png
echo "Raw results: $OUT"
