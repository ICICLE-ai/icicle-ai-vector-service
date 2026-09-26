#!/usr/bin/env python3
"""Plot collection-count scaling: which endpoints care how many collections exist."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
rows = json.loads((out / "results.json").read_text())["rows"]
n = [r["collections"] for r in rows]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(n, [r["list_ms"] for r in rows], "o-", color="#dc2626", linewidth=2.5,
        label="GET /v1/collections  (4 Qdrant calls per collection)")
ax.plot(n, [r["search_ms"] for r in rows], "s-", color="#2563eb", linewidth=2,
        label="POST /v1/retrieve  (one collection)")
ax.plot(n, [r["write_ms"] for r in rows], "^-", color="#059669", linewidth=2,
        label="POST /v1/embeddings  (one collection)")

span = n[-1] - n[0]
per = (rows[-1]["list_ms"] - rows[0]["list_ms"]) / span
ax.annotate(f"{per:.0f} ms per collection, linear\n→ ~{rows[0]['list_ms'] + per*(1000-n[0]):,.0f} ms at 1,000",
            xy=(n[-1], rows[-1]["list_ms"]), xytext=(n[-1] * 0.45, rows[-1]["list_ms"] * 0.92),
            fontsize=9, color="#dc2626",
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))
ax.annotate("search and write are flat:\ncollection count is not a global cost",
            xy=(n[len(n)//2], 115), xytext=(n[0] + span*0.25, 900),
            fontsize=9, color="#2563eb",
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1))

ax.set_xlabel("collections owned by the user")
ax.set_ylabel("latency (ms, log scale)")
ax.set_yscale("log")
ax.set_title("Cost of collection count — per-user-collection design\n"
             "(single token, 128-dim probe vectors)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="center right")
fig.tight_layout()
fig.savefig(out / "collection_scaling.png", dpi=150)
print(f"Wrote {out / 'collection_scaling.png'}")
