#!/usr/bin/env python3
"""Paired bootstrap significance test over per-query nDCG.

A difference in mean nDCG is not evidence on its own — with 100 queries, a swing
of 0.02 can easily be noise. This resamples the *same* queries with replacement
(paired, so each resample scores every method on the identical query set) and
reports how often each method beats the baseline.

p is the fraction of resamples in which the method did NOT beat the baseline;
small p means the observed difference is unlikely to be chance. The 95% interval
is the 2.5th-97.5th percentile of the per-resample difference.

Requires per-query scores, which evaluate.py records as `per_query_ndcg`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def bootstrap(baseline: list[float], other: list[float], rounds: int, seed: int):
    n = min(len(baseline), len(other))
    baseline, other = baseline[:n], other[:n]
    observed = sum(other) / n - sum(baseline) / n

    rng = random.Random(seed)
    diffs = []
    wins = 0
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(baseline[i] for i in idx) / n
        o = sum(other[i] for i in idx) / n
        diffs.append(o - b)
        if o > b:
            wins += 1
    diffs.sort()
    lo = diffs[int(0.025 * rounds)]
    hi = diffs[int(0.975 * rounds)]
    # Two-sided: how often the resampled difference lands on the other side of 0.
    p = 2 * min(wins, rounds - wins) / rounds
    return observed, lo, hi, min(p, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    out = Path(args.results_dir)
    data = json.loads((out / "results.json").read_text())
    conditions = data["conditions"]

    if "per_query_ndcg" not in conditions[0]:
        print("This run predates per-query score recording. Re-run evaluate.py.")
        return 1

    baseline = conditions[0]["per_query_ndcg"]
    top_k = data["top_k"]
    print(f"Paired bootstrap, {args.rounds:,} resamples, n={len(baseline)} queries")
    print(f"Baseline: {conditions[0]['condition']}\n")
    print(f"{'method':<36}{'Δ nDCG':>9}{'95% CI':>20}{'p':>8}   verdict")
    print(f"{'-'*36}{'-'*9}{'-'*20}{'-'*8}   {'-'*22}")

    rows = []
    for cond in conditions[1:]:
        observed, lo, hi, p = bootstrap(baseline, cond["per_query_ndcg"],
                                        args.rounds, args.seed)
        if lo == hi == 0.0:
            # Identical on every query, so every resample differs by exactly 0.
            # p is meaningless here; the methods are the same ranking.
            verdict, p = "identical to baseline", float("nan")
        elif lo <= 0 <= hi:
            verdict = "not distinguishable"
        elif observed > 0:
            verdict = "better"
        else:
            verdict = "WORSE"
        p_text = "  n/a" if p != p else f"{p:>5.3f}"
        print(f"{cond['condition']:<36}{observed:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>20}{p_text:>8}   {verdict}")
        rows.append({"condition": cond["condition"], "delta_ndcg": round(observed, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "p": None if p != p else round(p, 4), "verdict": verdict})

    (out / "significance.json").write_text(json.dumps({
        "baseline": conditions[0]["condition"], "metric": f"ndcg@{top_k}",
        "queries": len(baseline), "resamples": args.rounds, "seed": args.seed,
        "results": rows,
    }, indent=2))
    print(f"\nWrote {out}/significance.json")
    print("\nA CI spanning 0 means the data cannot distinguish that method from the")
    print("baseline at this sample size — neither better nor worse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
