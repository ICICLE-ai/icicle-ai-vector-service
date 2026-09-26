// Sanity check before spending time on a real run.
//
// Verifies the service is up, the token works, and every endpoint under test
// responds. One VU, a handful of requests. Run this first — it turns a
// misconfigured URL or a stale token into a clear failure in two seconds
// instead of a confusing wall of errors two minutes in.

import http from 'k6/http';
import { check } from 'k6';
import { BASE_URL, COLLECTION, TOP_K } from './lib/config.js';
import { headers, checkTokenLifetime, TOKENS } from './lib/auth.js';
import { storeBody, queryVector } from './lib/data.js';

export const options = { vus: 1, iterations: 1 };

export default function () {
  console.log(`Base URL   : ${BASE_URL}`);
  console.log(`Tokens     : ${TOKENS.length} (${TOKENS.length === 1 ? 'single user' : 'multi-user'})`);
  checkTokenLifetime(60);

  const health = http.get(`${BASE_URL}/healthz`);
  check(health, {
    'healthz 200': (r) => r.status === 200,
    'qdrant reachable': (r) => r.json('qdrant') === 'ok',
  });
  console.log(`Health     : ${health.body}`);

  const h = headers(1);

  const store = http.post(`${BASE_URL}/v1/embeddings`, storeBody(1), { headers: h });
  check(store, { 'store 201': (r) => r.status === 201 });
  if (store.status !== 201) console.error(`store failed: ${store.status} ${store.body}`);

  const retrieve = http.post(
    `${BASE_URL}/v1/retrieve`,
    JSON.stringify({ query_embedding: queryVector(1, 1), collection: COLLECTION, top_k: TOP_K }),
    { headers: h }
  );
  check(retrieve, { 'retrieve 200': (r) => r.status === 200 });
  if (retrieve.status !== 200) console.error(`retrieve failed: ${retrieve.status} ${retrieve.body}`);

  const methods = http.get(`${BASE_URL}/v1/rerank/methods`, { headers: h });
  check(methods, { 'rerank methods 200': (r) => r.status === 200 });
  if (methods.status === 200) {
    const available = methods.json('methods').filter((m) => m.available).map((m) => m.name);
    console.log(`Rerank     : ${available.join(', ')}`);
    console.log(`Models     : ${methods.json('allowed_models').join(', ')}`);
    if (!available.includes('cross_encoder')) {
      console.warn('cross_encoder unavailable — the cross-encoder scenario will fail');
    }
  }

  const collections = http.get(`${BASE_URL}/v1/collections`, { headers: h });
  check(collections, { 'collections 200': (r) => r.status === 200 });
  console.log(`Collections: ${collections.body}`);
}
