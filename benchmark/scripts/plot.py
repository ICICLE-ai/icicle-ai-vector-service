#!/usr/bin/env python3
"""Turn k6 JSON output into figures.

    python benchmark/scripts/plot.py benchmark/results/<run-id>

Produces four figures, chosen because each answers a question the others cannot:

1. **latency_cdf.png** — the full distribution per workload. A bar chart of
   p50/p95/p99 shows three points; a CDF shows the shape of the tail, which is
   where the interesting behaviour is.
2. **latency_over_time.png** — latency against wall-clock, with the ramp stages
   visible. This is where a knee appears: the moment p99 lifts away from p50.
3. **throughput.png** — achieved requests/sec against offered. The gap is the
   saturation point.
4. **percentiles.png** — p50/p95/p99 per workload on a LOG axis. Log is not
   optional here: a cross-encode is ~1000x a retrieve, so on a linear axis every
   other bar is flat against zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("This script needs: pip install pandas matplotlib")

# One colour per workload, kept consistent across every figure.
COLOURS = {
    "retrieve": "#2563eb",
    "rerank_vector": "#7c3aed",
    "rerank_cross": "#dc2626",
    "store": "#059669",
    "collections": "#d97706",
}
LABELS = {
    "retrieve": "retrieve",
    "rerank_vector": "rerank (mmr)",
    "rerank_cross": "rerank (cross-encoder)",
    "store": "store",
    "collections": "list collections",
}


def load_points(path: Path) -> pd.DataFrame:
    """Read k6's newline-delimited JSON into a frame of latency samples."""
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "Point":
                continue
            metric = record.get("metric", "")
            data = record.get("data", {})
            if metric.startswith("lat_"):
                rows.append(
                    {
                        "workload": metric[4:],
                        "ms": data.get("value"),
                        "time": data.get("time"),
                    }
                )
            elif metric == "http_reqs":
                rows.append({"workload": "__reqs__", "ms": None, "time": data.get("time")})
    if not rows:
        return pd.DataFrame(columns=["workload", "ms", "time"])
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"], format="mixed", utc=True)
    return frame


def collect(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name.endswith(".summary.json") or path.name == "run-config.json":
            continue
        frame = load_points(path)
        if not frame.empty:
            frame["scenario"] = path.stem
            frames.append(frame)
    if not frames:
        sys.exit(f"No k6 Point data found in {run_dir}")
    return pd.concat(frames, ignore_index=True)


def latency_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data[(data.workload != "__reqs__") & data.ms.notna()]


def plot_cdf(data: pd.DataFrame, out: Path) -> None:
    latency = latency_frame(data)
    if latency.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for workload, group in latency.groupby("workload"):
        values = np.sort(group.ms.values)
        if len(values) < 2:
            continue
        percentile = np.arange(1, len(values) + 1) / len(values) * 100
        ax.plot(
            values,
            percentile,
            label=f"{LABELS.get(workload, workload)}  (n={len(values):,})",
            color=COLOURS.get(workload, "#666"),
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("latency (ms, log scale)")
    ax.set_ylabel("percentile")
    ax.set_title("Latency distribution by workload")
    ax.grid(True, alpha=0.3, which="both")
    # The percentiles everyone quotes, marked so they can be read off directly.
    for level in (50, 95, 99):
        ax.axhline(level, color="#999", linestyle=":", linewidth=1)
        ax.annotate(f"p{level}", (ax.get_xlim()[0], level), fontsize=8, color="#666")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "latency_cdf.png", dpi=150)
    plt.close(fig)


def plot_over_time(data: pd.DataFrame, out: Path) -> None:
    latency = latency_frame(data)
    if latency.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for workload, group in latency.groupby("workload"):
        group = group.set_index("time").sort_index()
        # 5-second buckets: fine enough to see a ramp stage, coarse enough that
        # per-request noise does not swamp the trend.
        buckets = group.ms.resample("5s")
        ax.plot(
            buckets.median().index,
            buckets.median().values,
            color=COLOURS.get(workload, "#666"),
            label=f"{LABELS.get(workload, workload)} p50",
            linewidth=2,
        )
        ax.plot(
            buckets.quantile(0.99).index,
            buckets.quantile(0.99).values,
            color=COLOURS.get(workload, "#666"),
            label=f"{LABELS.get(workload, workload)} p99",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    ax.set_yscale("log")
    ax.set_xlabel("wall clock")
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title("Latency over time — p99 lifting away from p50 marks the knee")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "latency_over_time.png", dpi=150)
    plt.close(fig)


def plot_throughput(data: pd.DataFrame, out: Path) -> None:
    requests = data[data.workload == "__reqs__"]
    if requests.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for scenario, group in requests.groupby("scenario"):
        per_second = group.set_index("time").resample("1s").size()
        ax.plot(per_second.index, per_second.values, label=scenario, linewidth=1.5)
    ax.set_xlabel("wall clock")
    ax.set_ylabel("achieved requests/sec")
    ax.set_title("Achieved throughput — a plateau below the offered rate means saturation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)


def plot_percentiles(data: pd.DataFrame, out: Path) -> pd.DataFrame:
    latency = latency_frame(data)
    if latency.empty:
        return pd.DataFrame()
    stats = (
        latency.groupby("workload")
        .ms.agg(
            n="count",
            p50=lambda s: s.quantile(0.50),
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
            max="max",
        )
        .sort_values("p50")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stats))
    width = 0.26
    for offset, level, alpha in ((-width, "p50", 1.0), (0, "p95", 0.75), (width, "p99", 0.5)):
        ax.bar(
            x + offset,
            stats[level],
            width,
            label=level,
            color=[COLOURS.get(w, "#666") for w in stats.index],
            alpha=alpha,
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(w, w) for w in stats.index], fontsize=9)
    ax.set_yscale("log")  # mandatory: the range spans ~1000x
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title("Latency percentiles by workload")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "percentiles.png", dpi=150)
    plt.close(fig)
    return stats


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: plot.py <results-dir>")
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")

    data = collect(run_dir)
    plot_cdf(data, run_dir)
    plot_over_time(data, run_dir)
    plot_throughput(data, run_dir)
    stats = plot_percentiles(data, run_dir)

    if not stats.empty:
        stats.round(1).to_csv(run_dir / "percentiles.csv")
        print("\nLatency (ms)")
        print(stats.round(1).to_string())
        print(f"\nFigures and percentiles.csv written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
