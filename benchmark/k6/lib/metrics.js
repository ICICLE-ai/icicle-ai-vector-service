// Custom metrics.
//
// k6's built-in http_req_duration mixes every endpoint together. Since this
// service spans three orders of magnitude of latency, each workload gets its
// own Trend so percentiles stay meaningful.

import { Trend, Counter, Rate } from 'k6/metrics';

export const latency = {
  retrieve: new Trend('lat_retrieve', true),
  rerank_vector: new Trend('lat_rerank_vector', true),
  rerank_cross: new Trend('lat_rerank_cross', true),
  store: new Trend('lat_store', true),
  collections: new Trend('lat_collections', true),
};

export const failures = new Counter('failed_requests');
export const authFailures = new Counter('auth_failures');   // 401: token expired
export const okRate = new Rate('ok_rate');

// Record one response against the right workload, classifying failures so an
// expired token is never mistaken for a server problem.
export function record(kind, response) {
  latency[kind].add(response.timings.duration);
  const ok = response.status >= 200 && response.status < 300;
  okRate.add(ok);
  if (!ok) {
    failures.add(1);
    if (response.status === 401) authFailures.add(1);
  }
  return ok;
}
