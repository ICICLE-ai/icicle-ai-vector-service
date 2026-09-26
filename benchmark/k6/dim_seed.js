// Seeds one collection per dimension for the sweep, plus the fixed probe
// collection. Only needs enough points to satisfy fetch_k.
import exec from 'k6/execution';
import http from 'k6/http';
import { fail } from 'k6';

const BASE = (__ENV.BASE_URL || '').replace(/\/$/, '');
const TOKEN = __ENV.TAPIS_TOKEN;
const DIMS = (__ENV.SWEEP_DIMS || '768,1024,1536,2048,3072,4096').split(',').map(Number);
const POINTS = Number(__ENV.SWEEP_POINTS || 120);
const PROBE_DIM = Number(__ENV.PROBE_DIM || 768);
const PROBE_COLLECTION = __ENV.PROBE_COLLECTION || 'sweep_probe';
const H = { 'X-Tapis-Token': TOKEN, 'Content-Type': 'application/json' };

// (dim, collection) pairs, flattened so VUs can share the work.
const TASKS = [];
for (let i = 0; i < POINTS; i++) TASKS.push([PROBE_DIM, PROBE_COLLECTION, i]);
for (const d of DIMS) for (let i = 0; i < POINTS; i++) TASKS.push([d, `sweep_d${d}`, i]);

export const options = {
  scenarios: {
    seed: { executor: 'shared-iterations', vus: 8, iterations: TASKS.length, maxDuration: '30m' },
  },
};

function vec(dim, seed) {
  const out = new Array(dim);
  for (let i = 0; i < dim; i++) out[i] = Math.sin((i + seed) * 0.11);
  return out;
}

export function setup() {
  console.log(`Seeding ${POINTS} points into ${DIMS.length} collections (${DIMS.join(', ')}d) + probe`);
}

export default function () {
  // __ITER is per-VU, so with several VUs every one of them would walk the
  // start of the list and the later collections would never be created.
  // iterationInTest is the global counter across the whole scenario.
  const [dim, collection, i] = TASKS[exec.scenario.iterationInTest % TASKS.length];
  const r = http.post(`${BASE}/v1/embeddings`, JSON.stringify({
    embedding: vec(dim, i), collection, chunks: [`sweep point ${i}`],
    embedding_model: 'sweep-synthetic',
  }), { headers: H, timeout: '120s' });
  if (r.status !== 201) fail(`seed failed (${collection}): ${r.status} ${r.body}`);
}
