#!/usr/bin/env python3
"""Render the quality results as a figure plus a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHORT = {
    "retrieve (bi-encoder baseline)": "retrieve\n(baseline)",
    "mmr (lambda=0.7)": "mmr\nλ=0.7",
    "cosine_rescore": "cosine\nrescore",
    "cross_encoder: MiniLM-L-6": "cross-enc\nMiniLM",
    "cross_encoder: bge-reranker-base": "cross-enc\nbge-base",
}
COLOUR = ["#64748b", "#7c3aed", "#2563eb", "#059669", "#dc2626"]


def figure(data: dict, out: Path) -> None:
    rows = data["conditions"]
    top_k = data["top_k"]
    labels = [SHORT.get(r["condition"], r["condition"]) for r in rows]
    ndcg = [r[f"ndcg@{top_k}"] for r in rows]
    p50 = [r["latency_p50_ms"] for r in rows]
    baseline = ndcg[0]
    ceiling = data["recall_at_fetch_k"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(rows))

    bars = left.bar(x, ndcg, color=COLOUR[: len(rows)], edgecolor="white")
    left.axhline(baseline, color="#64748b", linestyle="--", linewidth=1.2)
    left.annotate("bi-encoder baseline", (len(rows) - 0.4, baseline), fontsize=8,
                  color="#64748b", va="bottom", ha="right")
    for bar, value in zip(bars, ndcg):
        delta = value - baseline
        tag = f"{value:.3f}" + (f"\n({delta:+.3f})" if abs(delta) > 1e-9 else "\n(±0)")
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.012, tag,
                  ha="center", fontsize=8)
    left.set_xticks(x); left.set_xticklabels(labels, fontsize=8)
    left.set_ylabel(f"nDCG@{top_k}")
    left.set_ylim(0, max(ndcg) * 1.25)
    left.set_title(f"Retrieval quality — BEIR {data['dataset']}\n"
                   f"(Recall@{data['fetch_k']} ceiling = {ceiling:.3f})")
    left.grid(alpha=0.3, axis="y")

    right.bar(x, p50, color=COLOUR[: len(rows)], edgecolor="white")
    for i, value in enumerate(p50):
        right.text(i, value * 1.08, f"{value:,.0f}ms", ha="center", fontsize=8)
    right.set_xticks(x); right.set_xticklabels(labels, fontsize=8)
    right.set_ylabel("latency p50 (ms, log scale)")
    right.set_yscale("log")
    right.set_title("Cost per query\n(single user, client-side, warm models)")
    right.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle(
        f"Rerank method comparison — {data['documents']:,} docs, "
        f"{data['queries_evaluated']} queries, {data['embedding_model']} "
        f"({data['dimension']}d), fetch_k={data['fetch_k']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "quality.png", dpi=150)
    plt.close(fig)


def table(data: dict) -> str:
    top_k = data["top_k"]
    baseline = data["conditions"][0][f"ndcg@{top_k}"]
    lines = [
        f"| Method | nDCG@{top_k} | vs baseline | Recall@{top_k} | p50 | p95 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in data["conditions"]:
        delta = r[f"ndcg@{top_k}"] - baseline
        lines.append(
            f"| {r['condition']} | {r[f'ndcg@{top_k}']:.4f} | {delta:+.4f} | "
            f"{r[f'recall@{top_k}']:.4f} | {r['latency_p50_ms']:,.0f}ms | "
            f"{r['latency_p95_ms']:,.0f}ms |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    args = parser.parse_args()
    out = Path(args.results_dir)
    data = json.loads((out / "results.json").read_text())
    figure(data, out)
    (out / "table.md").write_text(table(data) + "\n")
    print(table(data))
    print(f"\nFigure: {out / 'quality.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
