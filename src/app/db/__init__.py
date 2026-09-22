"""Qdrant access layer.

Split three ways:

* ``client``     — the shared ``AsyncQdrantClient`` and its FastAPI dependency.
* ``filters``    — builds every Qdrant filter. This is the tenancy boundary.
* ``repository`` — every read and write, each one scoped through ``filters``.

Route handlers never import ``qdrant_client`` directly; they call the
repository, so there is exactly one place where user scoping can be got wrong.
"""

from .client import close_client, get_qdrant_client, get_shared_client

__all__ = ["close_client", "get_qdrant_client", "get_shared_client"]
