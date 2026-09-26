// Populate the corpus each measured scenario queries against.
//
// Run once before the read scenarios. Latency is meaningless without a stated
// corpus, and an empty collection makes every search trivially fast.
//
// Each VU seeds as its own token, so with N tokens you get N users' corpora
// built in parallel — which is also what creates the per-user collections the
// read scenarios exercise.

import http from 'k6/http';
import { fail } from 'k6';
import { BASE_URL, POINTS_PER_USER, VECTOR_DIM, COLLECTION } from './lib/config.js';
import { headers, TOKENS } from './lib/auth.js';
import { storeBody } from './lib/data.js';

export const options = {
  scenarios: {
    seed: {
      executor: 'shared-iterations',
      vus: TOKENS.length,
      iterations: POINTS_PER_USER * TOKENS.length,
      maxDuration: '30m',
    },
  },
  // Seeding is setup, not measurement — a slow write here is not a finding.
  thresholds: { http_req_failed: ['rate<0.01'] },
};

export function setup() {
  console.log(
    `Seeding ${POINTS_PER_USER} points x ${TOKENS.length} user(s) ` +
      `at ${VECTOR_DIM} dims into '${COLLECTION}'`
  );
}

export default function () {
  const seed = (__ITER % POINTS_PER_USER) + 1;
  const response = http.post(`${BASE_URL}/v1/embeddings`, storeBody(seed), {
    headers: headers(__VU),
    tags: { op: 'seed' },
  });
  if (response.status !== 201) {
    fail(`seed write failed: ${response.status} ${response.body}`);
  }
}

export function teardown() {
  const response = http.get(`${BASE_URL}/v1/collections`, { headers: headers(1) });
  console.log(`Seeded. VU1 now sees: ${response.body}`);
}
