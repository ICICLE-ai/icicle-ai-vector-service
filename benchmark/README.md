# Benchmark

Load, scaling and retrieval-quality measurement for the ICICLE AI Vector Service.
Everything runs against a deployed instance over the network, so results include TLS,
the internet hop, FastAPI and Qdrant — what a client actually experiences.

Results from the current deployment: [REPORT.md](REPORT.md).

---

## Setup

```bash
brew install k6                       # https://grafana.com/docs/k6/latest/set-up/install-k6/
uv pip install -e ".[rerank,dev]" --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install pandas matplotlib requests

export BASE_URL=https://icicleaivecserver.pods.icicleai.tapis.io
export TAPIS_TOKEN=eyJhbGciOi...      # no surrounding whitespace
```

Check the token before a long run — a malformed one returns
`401 The access token could not be validated`, which looks like a service fault:

```bash
curl -s -o /dev/null -w "%{http_code}\n" "$BASE_URL/v1/collections" \
  -H "X-Tapis-Token: $TAPIS_TOKEN"     # expect 200
```

---

## Why 50 RPS

**50 requests/second = 3,000 per minute = 180,000 per hour.**

That target is roughly **2× the expected load** for this tenant — about 50–100
users, most of them idle at any moment. Sizing the test above real demand leaves
headroom and makes saturation visible before production hits it.

**It is a starting point, not a fixed requirement.** A deployment with more users,
bursty traffic, or a heavier query shape should raise `RATE` and re-run. Every
number in the report is specific to the load, corpus and query shape it was measured
under.

The scenarios use k6's arrival-rate executors, which hold the offered rate regardless
of how slowly the service responds. `dropped_iterations > 0` means the service could
not keep up — read that before any percentile.

---

## The suites

### 1. Load — throughput and latency

```bash
./benchmark/scripts/run.sh            # smoke → seed → mixed → ramps
./benchmark/scripts/run.sh mixed      # just 50 RPS mixed          (~3 min)
./benchmark/scripts/run.sh cleanup    # purge this token's data
```

Writes raw k6 JSON, `percentiles.csv` and four figures to
`benchmark/results/<run-id>/`.

### 2. Per-endpoint latency

```bash
python benchmark/scripts/endpoint_latency.py --collection scifact
```

p50/p95/p99 for all 16 endpoints, one at a time against an idle service.

### 3. Dimension sweep — does reranking affect unrelated requests?

```bash
k6 run benchmark/k6/dim_seed.js       # one collection per dimension  (~2 min)
./benchmark/scripts/dim_sweep.sh      # control + 6 dimensions        (~10 min)
```

### 4. Collection-count scaling

```bash
python benchmark/scaling/collection_count.py --max 200 --step 25   # ~8 min
python benchmark/scaling/collection_count.py --cleanup             # always
```

### 5. Rerank implementation comparison

```bash
python benchmark/scripts/rerank_implementations.py
```

Times the scalar and vectorised MMR implementations on identical inputs. Local CPU
only — no service, no network.

### 6. Quality — nDCG on BEIR SciFact

```bash
python benchmark/quality/prepare.py                      # download + embed (~6 min CPU)
python benchmark/quality/upload.py --collection scifact  # ~1 min
python benchmark/quality/evaluate.py --collection scifact --top-k 10 --fetch-k 50
python benchmark/quality/significance.py benchmark/results/<run-id>
python benchmark/quality/report.py benchmark/results/<run-id>
```

`prepare.py` caches embeddings locally, so re-uploading after a Qdrant reset needs
only step 2. Runtime is dominated by the cross-encoders — use `--limit` to bound it.

---

## Configuration

Environment-driven, so a run is reproducible from the `run-config.json` that
`run.sh` writes beside the results.

| Variable | Default | Meaning |
| --- | --- | --- |
| `BASE_URL` | `http://localhost:8000` | Service under test |
| `TAPIS_TOKEN` / `TAPIS_TOKENS` | — | One token, or a comma-separated list |
| `RATE` | `50` | Offered requests/sec |
| `VUS` | `50` | Generator worker pool |
| `DURATION` | `2m` | Hold time at the target rate |
| `VECTOR_DIM` | `768` | **Must match your embedding model** |
| `POINTS_PER_USER` | `500` | Corpus seeded per user |
| `TOP_K` / `FETCH_K` | `10` / `50` | Query shape |
| `CROSS_RATE` | `2` | Cross-encoder ramp start |
| `RERANK_MODEL` | service default | e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `MIX_RETRIEVE` / `MIX_RERANK_VECTOR` / `MIX_STORE` | `70` / `20` / `10` | Traffic mix |

**Report the corpus shape with any result.** "p95 = 40 ms" means nothing without it:
500 points per user and 500,000 are different services. `VECTOR_DIM` must match the
model you deploy with.

---

## Before you measure

**Pin thread counts on the pod.** In a container `nproc` reports the *host's* CPU
count, not the cgroup quota, so torch, OpenBLAS and tokenizers oversubscribe and the
container throttles. Set `RERANK_THREADS`, `OMP_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` to the quota, or you are measuring throttling rather than the
service.

**One token = one user.** `user_id` comes from the token, so VU count does not create
distinct users. A single token measures concurrency against one user's collections —
valid, and the pessimistic case for contention. Pass `TAPIS_TOKENS=t1,t2,...` for
genuine multi-user load.

**Clean up between runs.** The load test and `collection_count.py` both leave data
behind and both start from whatever already exists.

---

## Reading the output

In order:

1. **`auth_failures`** — non-zero means the token expired mid-run; the rest is suspect.
2. **`dropped_iterations`** — non-zero means saturation, and the reported latency is a floor.
3. **p50 vs p99 per workload** — a wide gap is queueing. `latency_over_time.png` shows when it began.
4. **Achieved vs offered throughput** — the gap is headroom you do not have.

Figures: `latency_cdf.png` (distribution shape, log x), `latency_over_time.png` (the
knee, where p99 lifts from p50), `throughput.png` (achieved vs offered),
`percentiles.png` (per workload, log y — the cross-encoder is ~1000× a search).

Regenerate figures without re-running the load:

```bash
python benchmark/scripts/plot.py benchmark/results/<run-id>
```

---

## Scope

Latency alone cannot justify reranking — that is what suite 5 is for. *"MiniLM costs
+2.3 s and buys −0.04 nDCG"* is a decision; *"MiniLM costs +2.3 s"* is not.

Raw results go to `benchmark/results/<run-id>/`, and the SciFact corpus plus its
cached vectors to `benchmark/quality/data/`; both are gitignored and regenerated
by the commands above. The figures and result JSON cited by the report are
committed under `benchmark/figures/`.
