#!/usr/bin/env bash
#
# Orchestrates a full benchmark run: smoke -> seed -> scenarios -> plots.
#
# Each scenario writes newline-delimited JSON to results/<run-id>/, which
# plot.py turns into figures. The run id is a timestamp, so runs never overwrite
# each other and can be compared afterwards.
#
#   export BASE_URL=https://icicleaivecserver.pods.icicleai.tapis.io
#   export TAPIS_TOKEN=...                 # or TAPIS_TOKENS=t1,t2,t3
#   ./benchmark/scripts/run.sh             # everything
#   ./benchmark/scripts/run.sh mixed       # one scenario
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K6_DIR="$HERE/k6"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="$HERE/results/$RUN_ID"

: "${BASE_URL:?set BASE_URL, e.g. https://icicleaivecserver.pods.icicleai.tapis.io}"
if [[ -z "${TAPIS_TOKEN:-}${TAPIS_TOKENS:-}" ]]; then
  echo "error: set TAPIS_TOKEN (or TAPIS_TOKENS for several users)" >&2
  exit 1
fi

command -v k6 >/dev/null 2>&1 || {
  echo "error: k6 not found. Install it: https://grafana.com/docs/k6/latest/set-up/install-k6/" >&2
  exit 1
}

mkdir -p "$OUT"
echo "Run id      : $RUN_ID"
echo "Target      : $BASE_URL"
echo "Results     : $OUT"
echo

# Record the conditions the numbers were produced under. A latency figure
# without its corpus shape and load settings is not reproducible.
cat > "$OUT/run-config.json" <<JSON
{
  "run_id": "$RUN_ID",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "base_url": "$BASE_URL",
  "rate_rps": ${RATE:-50},
  "vus": ${VUS:-50},
  "duration": "${DURATION:-2m}",
  "vector_dim": ${VECTOR_DIM:-768},
  "points_per_user": ${POINTS_PER_USER:-500},
  "top_k": ${TOP_K:-10},
  "fetch_k": ${FETCH_K:-50},
  "cross_rate": ${CROSS_RATE:-2},
  "rerank_model": "${RERANK_MODEL:-<service default>}",
  "client_host": "$(hostname)"
}
JSON

run_scenario() {
  local name="$1"
  echo "──────────────────────────────────────────────────────────"
  echo "▶ $name"
  echo "──────────────────────────────────────────────────────────"
  # k6 exits 99 when a threshold is crossed. That is the finding we are here to
  # collect, not a reason to abandon the remaining scenarios, so absorb it and
  # note it in the summary instead of letting `set -e` kill the run.
  set +e
  k6 run \
    --out "json=$OUT/$name.json" \
    --summary-export "$OUT/$name.summary.json" \
    "$K6_DIR/$name.js" 2>&1 | tee "$OUT/$name.log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -eq 99 ]]; then
    echo "⚠ $name: thresholds crossed (see $OUT/$name.log) — continuing"
    echo "$name" >> "$OUT/thresholds-crossed.txt"
  elif [[ $status -ne 0 ]]; then
    echo "✗ $name: k6 exited $status"
    echo "$name (exit $status)" >> "$OUT/scenario-errors.txt"
  fi
  echo
}

WHICH="${1:-all}"

# Always smoke-test first: a bad URL or stale token should fail in seconds.
echo "▶ smoke"
k6 run "$K6_DIR/smoke.js" 2>&1 | tee "$OUT/smoke.log"
echo

case "$WHICH" in
  all)
    echo "▶ seed (writes the corpus; not a measured scenario)"
    k6 run "$K6_DIR/seed.js" 2>&1 | tee "$OUT/seed.log"
    echo
    run_scenario mixed
    run_scenario retrieve
    if grep -q '"cross_encoder"' <<<"$(curl -fsS "$BASE_URL/healthz" || true)" \
       || [[ "$(curl -fsS "$BASE_URL/healthz" | grep -o '"cross_encoder":true' || true)" ]]; then
      run_scenario rerank_cross
    else
      echo "⚠ cross_encoder unavailable on this deployment — skipping that scenario"
    fi
    ;;
  seed)    k6 run "$K6_DIR/seed.js" | tee "$OUT/seed.log" ;;
  cleanup) k6 run "$K6_DIR/cleanup.js" | tee "$OUT/cleanup.log" ;;
  *)       run_scenario "$WHICH" ;;
esac

echo "──────────────────────────────────────────────────────────"
echo "Raw results in $OUT"
if command -v python3 >/dev/null 2>&1; then
  echo "Plotting..."
  python3 "$HERE/scripts/plot.py" "$OUT" || \
    echo "⚠ plotting failed (need: pip install pandas matplotlib)"
fi
echo "Done."
