// Search in isolation, ramped to find the knee.
//
// The mixed run answers "does 50 RPS work?". This one answers "where does it
// stop working?" — it ramps the offered rate and lets you read the point at
// which p99 departs from p50 and dropped_iterations goes non-zero.

import http from 'k6/http';
import { check } from 'k6';
import { BASE_URL, RATE, VUS, COLLECTION, TOP_K, POINTS_PER_USER } from './lib/config.js';
import { headers, checkTokenLifetime } from './lib/auth.js';
import { queryVector } from './lib/data.js';
import { record } from './lib/metrics.js';

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-arrival-rate',
      startRate: Math.max(1, Math.floor(RATE / 5)),
      timeUnit: '1s',
      preAllocatedVUs: VUS,
      maxVUs: VUS * 8,
      stages: [
        { target: RATE, duration: '30s' },        // reach the design point
        { target: RATE, duration: '60s' },        // hold: this is the reported figure
        { target: RATE * 2, duration: '30s' },    // push past it
        { target: RATE * 2, duration: '30s' },
        { target: RATE * 4, duration: '30s' },    // find the knee
        { target: RATE * 4, duration: '30s' },
      ],
    },
  },
  thresholds: { 'lat_retrieve': ['p(95)<2000'] },
};

export function setup() {
  checkTokenLifetime(300);
}

export default function () {
  const response = http.post(
    `${BASE_URL}/v1/retrieve`,
    JSON.stringify({
      query_embedding: queryVector(__ITER, POINTS_PER_USER),
      collection: COLLECTION,
      top_k: TOP_K,
    }),
    { headers: headers(__VU), tags: { op: 'retrieve' } }
  );
  record('retrieve', response);
  check(response, { 'retrieve ok': (r) => r.status === 200 });
}
