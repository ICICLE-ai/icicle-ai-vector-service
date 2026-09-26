#!/usr/bin/env python3
"""Summarise a dimension sweep: does reranking degrade unrelated requests?

Prints a table and writes dim_sweep.png, which plots the probe (the unrelated
search that must not suffer) against the rerank itself as dimension grows.
"""
import json
import sys
from pathlib import Path


def stats(path: Path, metric: str) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return (data.get("metrics") or {}).get(metric)


def main() -> int:
    out = Path(sys.argv[1])
    control = stats(out / "control.json", "lat_probe")
    if not control:
        sys.exit("no control run found")
    base99 = control["p(99)"]

    print(f"{'dim':>6} {'elements':>9} | {'probe p50':>10} {'probe p95':>10} {'probe p99':>10} "
          f"{'vs control':>11} | {'rerank p50':>11} {'rerank p99':>11}")
    print(f"{'-'*6} {'-'*9} | {'-'*10} {'-'*10} {'-'*10} {'-'*11} | {'-'*11} {'-'*11}")
    print(f"{'none':>6} {'-':>9} | {control['p(50)']:>9.0f}ms {control['p(95)']:>9.0f}ms "
          f"{base99:>9.0f}ms {'baseline':>11} | {'-':>11} {'-':>11}")

    for path in sorted(out.glob("dim-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        dim = int(path.stem.split("-")[1])
        probe = stats(path, "lat_probe")
        rerank = stats(path, "lat_rerank")
        if not probe:
            continue
        delta = (probe["p(99)"] / base99 - 1) * 100 if base99 else 0
        flag = "  <-- degraded" if delta > 50 else ""
        rr50 = f"{rerank['p(50)']:>10.0f}ms" if rerank else f"{'-':>11}"
        rr99 = f"{rerank['p(99)']:>10.0f}ms" if rerank else f"{'-':>11}"
        print(f"{dim:>6} {dim*50:>9,} | {probe['p(50)']:>9.0f}ms {probe['p(95)']:>9.0f}ms "
              f"{probe['p(99)']:>9.0f}ms {delta:>+10.0f}% | {rr50} {rr99}{flag}")

    print("\nprobe = cheap 768-dim search running throughout, the thing that must not suffer.")
    print("rerank latency rising with dimension is expected; probe latency rising is not.")
    plot(out, control)
    return 0


def plot(out: Path, control: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib for the figure)")
        return

    dims, p50, p95, p99, r50, r95 = [], [], [], [], [], []
    for path in sorted(out.glob("dim-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        probe = stats(path, "lat_probe")
        rerank = stats(path, "lat_rerank")
        if not probe:
            continue
        dims.append(int(path.stem.split("-")[1]))
        p50.append(probe["p(50)"]); p95.append(probe["p(95)"]); p99.append(probe["p(99)"])
        r50.append(rerank["p(50)"] if rerank else 0)
        r95.append(rerank["p(95)"] if rerank else 0)
    if not dims:
        return

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: the victim. A flat line here means reranking is not stealing the loop.
    left.plot(dims, p50, "o-", color="#2563eb", label="probe p50", linewidth=2)
    left.plot(dims, p95, "s-", color="#7c3aed", label="probe p95", linewidth=2)
    left.plot(dims, p99, "^--", color="#dc2626", label="probe p99 (noisy)", alpha=0.6)
    for level, style, name in ((control["p(50)"], ":", "p50"), (control["p(95)"], "--", "p95")):
        left.axhline(level, color="#999", linestyle=style, linewidth=1)
        left.annotate(f"control {name}", (dims[0], level), fontsize=8, color="#666",
                      va="bottom")
    left.set_xlabel("rerank vector dimension")
    left.set_ylabel("latency (ms)")
    left.set_title("Unrelated search, while reranking runs\n(the line that must stay flat)")
    left.grid(alpha=0.3); left.legend(fontsize=9)

    # Right: the rerank itself. Rising here is expected — more bytes, more maths.
    right.plot(dims, r50, "o-", color="#059669", label="rerank p50", linewidth=2)
    right.plot(dims, r95, "s-", color="#d97706", label="rerank p95", linewidth=2)
    right.set_xlabel("rerank vector dimension")
    right.set_ylabel("latency (ms)")
    right.set_title("The rerank itself\n(rising is expected: payload scales with dim)")
    right.grid(alpha=0.3); right.legend(fontsize=9)

    fig.suptitle("Rerank cost vs vector dimension (fetch_k=50)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "dim_sweep.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {out / 'dim_sweep.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
