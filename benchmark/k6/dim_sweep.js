// Finds the vector dimension at which reranking starts hurting *other* requests.
//
// fetch_k is fixed at 50 (the service default), so dimension is the only thing
// varied. Each run seeds and queries one collection per dimension.
//
// The signal is the BACKGROUND retrieve latency, not the rerank latency. Rerank
// gets slower with dimension by definition — that is just more bytes and more
// arithmetic. What matters is whether it steals the event loop from unrelated
// requests. So a steady stream of cheap retrieves runs against a small, fixed
// 768-dim collection throughout; if their p99 lifts as SWEEP_DIM grows, the
// rerank is blocking the loop and an offload is justified.
//
//   SWEEP_DIM=0    control: background retrieve only, no rerank traffic
//   SWEEP_DIM=4096 background retrieve + rerank over 4096-dim vectors

import http from 'k6/http';
import { Trend } from 'k6/metrics';

const BASE = (__ENV.BASE_URL || '').replace(/\/$/, '');
const TOKEN = __ENV.TAPIS_TOKEN;
const DIM = Number(__ENV.SWEEP_DIM || 0);
const PROBE_DIM = Number(__ENV.PROBE_DIM || 768);
const PROBE_COLLECTION = __ENV.PROBE_COLLECTION || 'sweep_probe';
const RETRIEVE_RATE = Number(__ENV.RETRIEVE_RATE || 20);
const RERANK_RATE = Number(__ENV.RERANK_RATE || 5);
const DURATION = __ENV.SWEEP_DURATION || '45s';
const FETCH_K = Number(__ENV.FETCH_K || 50);

const H = { 'X-Tapis-Token': TOKEN, 'Content-Type': 'application/json' };

// The victim metric: cheap searches that should be unaffected by reranking.
export const lat_probe = new Trend('lat_probe', true);
// Reported for context; expected to grow with dimension.
export const lat_rerank = new Trend('lat_rerank', true);

function vec(dim, seed) {
  const out = new Array(dim);
  for (let i = 0; i < dim; i++) out[i] = Math.sin((i + seed) * 0.37);
  return out;
}
const PROBE_VEC = vec(PROBE_DIM, 1);
const SWEEP_VEC = DIM ? vec(DIM, 1) : null;

const scenarios = {
  probe: {
    executor: 'constant-arrival-rate',
    rate: RETRIEVE_RATE, timeUnit: '1s', duration: DURATION,
    preAllocatedVUs: 40, maxVUs: 400, exec: 'probe',
  },
};
if (DIM > 0) {
  scenarios.rerank = {
    executor: 'constant-arrival-rate',
    rate: RERANK_RATE, timeUnit: '1s', duration: DURATION,
    preAllocatedVUs: 40, maxVUs: 400, exec: 'rerank',
  };
}

export const options = { scenarios, summaryTrendStats: ['p(50)', 'p(95)', 'p(99)', 'max'] };

export function probe() {
  const r = http.post(
    `${BASE}/v1/retrieve`,
    JSON.stringify({ query_embedding: PROBE_VEC, collection: PROBE_COLLECTION, top_k: 10 }),
    { headers: H, tags: { op: 'probe' } }
  );
  lat_probe.add(r.timings.duration);
}

export function rerank() {
  const r = http.post(
    `${BASE}/v1/rerank`,
    JSON.stringify({
      query_embedding: SWEEP_VEC,
      collection: `sweep_d${DIM}`,
      method: 'mmr',
      top_k: 10,
      fetch_k: FETCH_K,
    }),
    { headers: H, tags: { op: 'rerank' }, timeout: '120s' }
  );
  lat_rerank.add(r.timings.duration);
}
