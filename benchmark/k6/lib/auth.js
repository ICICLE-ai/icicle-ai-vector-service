import encoding from 'k6/encoding';

// Token handling.
//
// TAPIS_TOKENS may hold several comma-separated tokens. Each VU is pinned to
// one, so N tokens means N distinct users hitting the service concurrently.
// With a single token every VU authenticates as the same user — still a valid
// concurrency test, but it measures one user's collections, not isolation under
// load. The distinction is printed at startup and recorded in the results.

const RAW = __ENV.TAPIS_TOKENS || __ENV.TAPIS_TOKEN || '';
export const TOKENS = RAW.split(',').map((t) => t.trim()).filter(Boolean);

// Deliberately not thrown at module scope: that would break `k6 inspect`,
// `k6 archive` and any tooling that merely loads the script without running it.
// The check fires on first use instead.
export function requireTokens() {
  if (TOKENS.length === 0) {
    throw new Error(
      'No token supplied. Set TAPIS_TOKEN=<token>, or TAPIS_TOKENS=<t1,t2,...> ' +
        'to drive the service as several distinct users.'
    );
  }
  return TOKENS;
}

// Pin each VU to a token so a given user's requests always hit the same
// collections — round-robin would smear one user's corpus across all of them.
export function tokenForVU(vu) {
  requireTokens();
  return TOKENS[(vu - 1) % TOKENS.length];
}

export function headers(vu) {
  return {
    'X-Tapis-Token': tokenForVU(vu),
    'Content-Type': 'application/json',
  };
}

// Decode the exp claim without verifying — this is a liveness check on our own
// test credentials, not an auth decision. A run that outlives its token reports
// a wall of 401s that look like server errors but are not.
function expiryOf(token) {
  try {
    const payload = token.split('.')[1];
    const json = JSON.parse(
      encoding.b64decode(payload.replace(/-/g, '+').replace(/_/g, '/'), 'rawstd', 's')
    );
    return json.exp ? Number(json.exp) : null;
  } catch (e) {
    return null;
  }
}

export function checkTokenLifetime(requiredSeconds) {
  requireTokens();
  const now = Math.floor(Date.now() / 1000);
  const remaining = TOKENS.map(expiryOf).map((exp) => (exp ? exp - now : null));

  remaining.forEach((left, i) => {
    if (left === null) {
      console.warn(`token ${i + 1}: could not read exp claim; cannot check lifetime`);
    } else if (left <= 0) {
      throw new Error(`token ${i + 1} has already expired — mint a fresh one`);
    } else if (left < requiredSeconds) {
      console.warn(
        `token ${i + 1} expires in ${Math.floor(left / 60)} min, which is less than ` +
          `this run needs (${Math.ceil(requiredSeconds / 60)} min). Expect 401s near the end.`
      );
    } else {
      console.log(`token ${i + 1}: ${Math.floor(left / 60)} min remaining`);
    }
  });
  return remaining;
}
