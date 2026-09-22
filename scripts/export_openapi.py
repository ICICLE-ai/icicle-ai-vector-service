#!/usr/bin/env python3
"""Generate ``openapi.json`` from the live FastAPI app.

FastAPI derives most of the spec automatically, but two things it cannot infer
are added here: the ``X-Tapis-Token`` header declared as a proper apiKey security
scheme (so Swagger's Authorize button and generated clients work), and the
server/tag metadata.

    python scripts/export_openapi.py           # write openapi.json
    python scripts/export_openapi.py --check   # fail if it is out of date (CI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC_PATH = ROOT / "openapi.json"

SECURITY_SCHEME = {
    "TapisToken": {
        "type": "apiKey",
        "in": "header",
        "name": "X-Tapis-Token",
        "description": (
            "ICICLE AI tenant Tapis access token. The username is read from the "
            "tapis/username claim and used to isolate your data."
        ),
    }
}

SERVERS = [
    {
        "url": "https://icicleaivecserver.pods.icicleai.tapis.io",
        "description": "Production (ICICLE AI Tapis pod)",
    },
    {"url": "http://localhost:8000", "description": "Local development"},
]

TAGS = [
    {"name": "health", "description": "Liveness and readiness. No token required."},
    {
        "name": "embeddings",
        "description": "Create, read, update and delete embeddings, individually or in bulk.",
    },
    {
        "name": "collections",
        "description": "Discover your collections and delete them, one at a time or all at once.",
    },
    {"name": "search", "description": "Vector similarity search and reranking."},
]

# Every endpoint except the health probe requires a token.
PUBLIC_PATHS = {"/healthz"}


def build() -> dict:
    from src.app.main import app

    spec = app.openapi()
    spec.setdefault("components", {})["securitySchemes"] = SECURITY_SCHEME
    for path, operations in spec["paths"].items():
        if path in PUBLIC_PATHS:
            continue
        for operation in operations.values():
            if isinstance(operation, dict):
                operation["security"] = [{"TapisToken": []}]
    spec["servers"] = SERVERS
    spec["tags"] = TAGS
    return spec


def main() -> int:
    spec = build()
    rendered = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"

    if "--check" in sys.argv:
        if not SPEC_PATH.exists():
            print("openapi.json is missing. Run: python scripts/export_openapi.py")
            return 1
        if SPEC_PATH.read_text() != rendered:
            print(
                "openapi.json is out of date with the app. "
                "Run: python scripts/export_openapi.py"
            )
            return 1
        print("openapi.json is up to date")
        return 0

    SPEC_PATH.write_text(rendered)
    print(
        f"Wrote {SPEC_PATH.relative_to(ROOT)} "
        f"(v{spec['info']['version']}, {len(spec['paths'])} paths, "
        f"{len(spec['components']['schemas'])} schemas)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
