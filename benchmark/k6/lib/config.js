// Shared configuration, all overridable by environment variable.
//
// Every knob that changes what the numbers *mean* lives here rather than being
// buried in a scenario, so a result can be reproduced from the env alone.

export const BASE_URL = (__ENV.BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

// Target load. 50 RPS is the design point; VUS is how many simultaneous users
// that load is spread across.
export const RATE = Number(__ENV.RATE || 50);         // requests per second
export const VUS = Number(__ENV.VUS || 50);           // simultaneous users
export const DURATION = __ENV.DURATION || '2m';
export const WARMUP = __ENV.WARMUP || '20s';

// Corpus shape. These decide what the latency figure actually describes, so
// they are reported in the summary alongside the percentiles.
export const VECTOR_DIM = Number(__ENV.VECTOR_DIM || 768);
export const POINTS_PER_USER = Number(__ENV.POINTS_PER_USER || 500);
export const COLLECTION = __ENV.COLLECTION || 'bench';
export const TOPICS = (__ENV.TOPICS || 'alpha,beta,gamma').split(',');

// Query shape.
export const TOP_K = Number(__ENV.TOP_K || 10);
export const FETCH_K = Number(__ENV.FETCH_K || 50);
export const RERANK_MODEL = __ENV.RERANK_MODEL || '';   // '' = service default

// Mixed-workload weighting, as percentages. Reads dominate, which is what a
// RAG-style service actually does.
export const MIX = {
  retrieve: Number(__ENV.MIX_RETRIEVE || 70),
  rerankVector: Number(__ENV.MIX_RERANK_VECTOR || 20),
  store: Number(__ENV.MIX_STORE || 10),
};

// Cross-encoder runs at its own, much lower rate: a single request takes
// 0.2-1.2s of CPU, so 50 RPS of it would need far more cores than any one pod
// has. Benchmarked separately, never mixed into the 50 RPS figure.
export const CROSS_RATE = Number(__ENV.CROSS_RATE || 2);

export function summariseConfig() {
  return {
    base_url: BASE_URL,
    rate_rps: RATE,
    vus: VUS,
    duration: DURATION,
    vector_dim: VECTOR_DIM,
    points_per_user: POINTS_PER_USER,
    top_k: TOP_K,
    fetch_k: FETCH_K,
  };
}
