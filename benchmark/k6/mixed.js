// The headline run: 50 requests/second across 50 simultaneous users.
//
// WHY AN ARRIVAL-RATE EXECUTOR, NOT FIXED VUs
// With `constant-vus`, each VU waits for a response before sending again — so
// when the service slows, the generator automatically offers *less* load and
// the measured latency flatters the server. That is coordinated omission.
// `constant-arrival-rate` holds 50 RPS regardless of how slow responses get, so
// a saturated service shows up as rising latency and non-zero
// `dropped_iterations` rather than as a comfortable-looking plateau.
//
// preAllocatedVUs is the "50 simultaneous users" part. If latency rises, k6
// needs more concurrent VUs to sustain 50 RPS; maxVUs allows that headroom, and
// the run reports if it was reached.
//
// Traffic is weighted (default 70/20/10 read/rerank/write) because a per-endpoint
// number does not tell you how the service behaves when real traffic interleaves
// reads and writes against the same collections.

import http from 'k6/http';
import { check } from 'k6';
import {
  BASE_URL, RATE, VUS, DURATION, COLLECTION, TOP_K, FETCH_K, POINTS_PER_USER, MIX,
  summariseConfig,
} from './lib/config.js';
import { headers, checkTokenLifetime, TOKENS } from './lib/auth.js';
import { queryVector, storeBody, topicFor } from './lib/data.js';
import { record } from './lib/metrics.js';

const TOTAL = MIX.retrieve + MIX.rerankVector + MIX.store;

export const options = {
  scenarios: {
    mixed: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: VUS,
      // Headroom so a latency rise does not silently throttle offered load.
      // If k6 needs more than this, it reports dropped_iterations instead.
      maxVUs: VUS * 4,
    },
  },
  thresholds: {
    // Deliberately generous: these are guard rails that catch a broken run, not
    // an SLO. Set real targets once you have a baseline from your own hardware.
    'lat_retrieve': ['p(50)<100', 'p(95)<500', 'p(99)<1000'],
    'lat_rerank_vector': ['p(95)<1000'],
    'lat_store': ['p(95)<1000'],
    'ok_rate': ['rate>0.99'],
    // Saturation signal: k6 could not keep up with the offered rate.
    'dropped_iterations': ['count<1'],
  },
};

export function setup() {
  const runSeconds = Math.ceil(parseDuration(DURATION) + 60);
  checkTokenLifetime(runSeconds);
  console.log(`Config: ${JSON.stringify(summariseConfig())}`);
  console.log(
    `Users: ${TOKENS.length} token(s). ` +
      (TOKENS.length === 1
        ? 'All VUs act as ONE user — this measures concurrency, not multi-user isolation.'
        : `${TOKENS.length} distinct users, each with their own collections.`)
  );
  return { startedAt: Date.now() };
}

function parseDuration(text) {
  const match = /^(\d+)([smh])$/.exec(text.trim());
  if (!match) return 120;
  const value = Number(match[1]);
  return match[2] === 'h' ? value * 3600 : match[2] === 'm' ? value * 60 : value;
}

export default function () {
  const h = headers(__VU);
  const roll = Math.random() * TOTAL;

  if (roll < MIX.retrieve) {
    const response = http.post(
      `${BASE_URL}/v1/retrieve`,
      JSON.stringify({
        query_embedding: queryVector(__ITER, POINTS_PER_USER),
        collection: COLLECTION,
        top_k: TOP_K,
      }),
      { headers: h, tags: { op: 'retrieve' } }
    );
    record('retrieve', response);
    check(response, { 'retrieve ok': (r) => r.status === 200 });
  } else if (roll < MIX.retrieve + MIX.rerankVector) {
    const response = http.post(
      `${BASE_URL}/v1/rerank`,
      JSON.stringify({
        query_embedding: queryVector(__ITER, POINTS_PER_USER),
        collection: COLLECTION,
        method: 'mmr',
        top_k: TOP_K,
        fetch_k: FETCH_K,
      }),
      { headers: h, tags: { op: 'rerank_vector' } }
    );
    record('rerank_vector', response);
    check(response, { 'rerank ok': (r) => r.status === 200 });
  } else {
    // Writes use a seed beyond the seeded range so they add points rather than
    // overwriting the corpus the reads depend on.
    const seed = POINTS_PER_USER + __ITER + __VU * 100000;
    const response = http.post(`${BASE_URL}/v1/embeddings`, storeBody(seed), {
      headers: h,
      tags: { op: 'store' },
    });
    record('store', response);
    check(response, { 'store ok': (r) => r.status === 201 });
  }
}
