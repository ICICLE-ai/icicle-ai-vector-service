// Deterministic test data.
//
// Vectors are generated, not embedded: this benchmark measures the *service*,
// so the numbers must not depend on an external embedding API being fast or
// even reachable.

import { VECTOR_DIM, COLLECTION, TOPICS } from './config.js';

// A cheap deterministic PRNG. Deterministic matters: the same seed produces the
// same corpus and the same queries, so two runs are comparable.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Unit-normalised so cosine scores span a realistic range rather than clustering
// near 1.0, which would make every query look like a trivial match.
export function vector(seed, dim = VECTOR_DIM) {
  const random = mulberry32(seed);
  const values = new Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    const v = random() * 2 - 1;
    values[i] = v;
    norm += v * v;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) values[i] = values[i] / norm;
  return values;
}

export function topicFor(seed) {
  return TOPICS[seed % TOPICS.length];
}

export function collectionFor() {
  return COLLECTION;
}

const SENTENCES = [
  'Photosynthesis converts light energy into chemical energy in green plants.',
  'Mitosis produces two genetically identical daughter cells.',
  'The periodic table organises elements by increasing atomic number.',
  'Covalent bonds form when atoms share electron pairs.',
  'Enzymes lower the activation energy of biochemical reactions.',
];

export function chunkFor(seed) {
  return `${SENTENCES[seed % SENTENCES.length]} (record ${seed})`;
}

// Queries reuse corpus seeds so they land near real points rather than in empty
// space — an unrealistically distant query makes HNSW exit early and look fast.
export function queryVector(iteration, pointsPerUser) {
  return vector((iteration % pointsPerUser) + 1);
}

export function storeBody(seed) {
  return JSON.stringify({
    embedding: vector(seed),
    collection: collectionFor(),
    topic: topicFor(seed),
    chunks: [chunkFor(seed)],
    metadata: { source: `doc_${seed % 20}.pdf`, page: seed % 50 },
    embedding_model: 'benchmark-synthetic',
  });
}
