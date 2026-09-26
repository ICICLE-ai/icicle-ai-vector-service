// Cross-encoder reranking, benchmarked on its own terms.
//
// WHY THIS IS A SEPARATE RUN
// A cross-encode is 0.2-1.2s of CPU depending on model and fetch_k, against
// single-digit milliseconds for a plain retrieve. Folding it into the 50 RPS
// figure would produce a p99 that describes neither workload. It also cannot
// physically sustain 50 RPS on one pod: at ~1.2s of CPU per request, 50 RPS
// would need ~60 cores.
//
// WHAT TO READ FROM IT
// Concurrency is the variable that matters. Inference runs in a worker thread
// and torch uses several cores per call, so beyond a small number of concurrent
// requests they queue. This ramp finds that point — which is the number you
// need to size the pod, set a concurrency limit, or justify moving the reranker
// to its own service.

import http from 'k6/http';
import { check } from 'k6';
import {
  BASE_URL, CROSS_RATE, COLLECTION, TOP_K, FETCH_K, POINTS_PER_USER, RERANK_MODEL,
} from './lib/config.js';
import { headers, checkTokenLifetime } from './lib/auth.js';
import { queryVector } from './lib/data.js';
import { record } from './lib/metrics.js';

const QUERIES = [
  'How do plants convert sunlight into chemical energy?',
  'What happens during cell division?',
  'How are elements organised in the periodic table?',
  'What makes a covalent bond strong?',
  'How do enzymes speed up reactions?',
];

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-arrival-rate',
      startRate: 1,
      timeUnit: '1s',
      preAllocatedVUs: 10,
      maxVUs: 100,
      stages: [
        { target: CROSS_RATE, duration: '30s' },
        { target: CROSS_RATE, duration: '60s' },
        { target: CROSS_RATE * 2, duration: '30s' },
        { target: CROSS_RATE * 2, duration: '60s' },
        { target: CROSS_RATE * 4, duration: '30s' },
        { target: CROSS_RATE * 4, duration: '60s' },
      ],
    },
  },
  thresholds: {
    // No hard threshold: the purpose is to find where it breaks, not to assert
    // that it does not. dropped_iterations is the signal to read.
    'lat_rerank_cross': ['p(95)<30000'],
  },
};

export function setup() {
  checkTokenLifetime(400);

  // The first request pays model download and load — tens of seconds. Warming
  // up here keeps that cost out of the measured percentiles; measure it
  // deliberately with `--cold-start` in run.sh if you care about it.
  const body = JSON.stringify({
    query_embedding: queryVector(1, POINTS_PER_USER),
    query_text: QUERIES[0],
    collection: COLLECTION,
    method: 'cross_encoder',
    top_k: TOP_K,
    fetch_k: FETCH_K,
    ...(RERANK_MODEL ? { rerank_model: RERANK_MODEL } : {}),
  });

  console.log('Warming the reranker (first call loads the model — may take 10-30s)...');
  const started = Date.now();
  const response = http.post(`${BASE_URL}/v1/rerank`, body, {
    headers: headers(1),
    timeout: '180s',
  });
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);

  if (response.status === 503) {
    throw new Error(
      'Cross-encoder unavailable: the service was built without the [rerank] extra.'
    );
  }
  if (response.status !== 200) {
    throw new Error(`Warmup failed: ${response.status} ${response.body}`);
  }
  console.log(`Model ready in ${elapsed}s (model: ${response.json('model')})`);
  return { coldStartSeconds: Number(elapsed) };
}

export default function () {
  const response = http.post(
    `${BASE_URL}/v1/rerank`,
    JSON.stringify({
      query_embedding: queryVector(__ITER, POINTS_PER_USER),
      query_text: QUERIES[__ITER % QUERIES.length],
      collection: COLLECTION,
      method: 'cross_encoder',
      top_k: TOP_K,
      fetch_k: FETCH_K,
      ...(RERANK_MODEL ? { rerank_model: RERANK_MODEL } : {}),
    }),
    { headers: headers(__VU), tags: { op: 'rerank_cross' }, timeout: '120s' }
  );
  record('rerank_cross', response);
  check(response, { 'cross-encoder ok': (r) => r.status === 200 });
}
