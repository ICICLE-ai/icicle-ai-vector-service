// Remove everything the benchmark created.
//
// Purges each token's data. With per-user collections this drops the caller's
// own collections outright and cannot touch anyone else's — so it is safe to
// run against a shared deployment, provided the benchmark tokens belong to
// accounts whose data you are willing to lose.

import http from 'k6/http';
import { check } from 'k6';
import { BASE_URL } from './lib/config.js';
import { headers, TOKENS } from './lib/auth.js';

export const options = { vus: 1, iterations: 1 };

export default function () {
  console.warn(
    `About to purge ALL data for ${TOKENS.length} benchmark token(s). ` +
      'This deletes every collection those users own, not just the benchmark one.'
  );
  for (let i = 1; i <= TOKENS.length; i++) {
    const before = http.get(`${BASE_URL}/v1/collections`, { headers: headers(i) });
    const response = http.del(`${BASE_URL}/v1/collections?confirm=true`, null, {
      headers: headers(i),
    });
    check(response, { 'purge ok': (r) => r.status === 200 });
    console.log(
      `token ${i}: had ${before.json('count')} collection(s) -> ${response.body}`
    );
  }
}
