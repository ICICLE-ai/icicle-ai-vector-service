"""Physical Qdrant collection naming.

Every user gets their *own* Qdrant collections. A user-facing collection name
like ``biology`` becomes a physical collection named after the owner:

    alice + "biology"  ->  alice_9f2a1c4e7b30__biology
    bob   + "biology"  ->  bob_3d81f5a0c2e9__biology

This is the primary isolation mechanism, and it is structural rather than
enforced: two users' data live in different Qdrant collections, so there is no
shared object for an operation to race on or leak through. Concretely it means:

* **Dropping an emptied collection is safe.** It is the caller's own collection;
  no other user can be writing into it, so the "is it empty?" / "drop it" pair
  no longer has a window in which another user's first write is destroyed.
* **Vector dimensions are per user.** One user's 768-dimensional ``biology``
  does not stop another from creating a 1024-dimensional ``biology``, which
  matters when users bring their own domain-specific embedding models.
* **Listing is a prefix scan**, not a filtered count across every collection in
  the cluster.

The payload ``user_id`` filter is *kept* on every query anyway. It is no longer
load-bearing, but it costs almost nothing and means a hypothetical bug in this
module still could not surface another user's data.

Naming rules, and why:

* The owner prefix is ``<readable>_<hash>``. The readable part makes collections
  identifiable when browsing Qdrant directly; the hash is what actually
  guarantees uniqueness.
* The hash covers ``tenant_id`` *and* ``username``. Slugifying alone is unsafe —
  ``a.b`` and ``a-b`` both slugify to ``a_b``, which would silently merge two
  users. Distinct identities always produce distinct hashes.
* ``__`` separates the owner prefix from the collection name. Slugs collapse
  every run of non-alphanumeric characters to a *single* underscore and the hash
  is hex, so neither component can contain ``__`` and the split is unambiguous.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Separates the owner prefix from the collection name.
SEPARATOR = "__"
# Hex characters of the identity digest kept in the prefix. 12 hex chars = 48
# bits; the chance of a collision across even a few thousand users is negligible.
_HASH_LENGTH = 12
# Keeps the human-readable part of a prefix bounded for long usernames.
_MAX_READABLE = 40


def slugify(value: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to '_', strip the edges."""
    return _SLUG_RE.sub("_", value.lower()).strip("_")


@dataclass(frozen=True)
class Owner:
    """Who a collection belongs to.

    Frozen, and built from the verified token, so the identity used to compute a
    physical collection name cannot be altered partway through a request.
    """

    username: str
    tenant_id: str

    def __post_init__(self) -> None:
        if not self.username:
            raise ValueError("username must not be empty")
        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")

    @classmethod
    def from_context(cls, context) -> "Owner":
        """Build from a :class:`app.core.security.UserContext`."""
        return cls(username=context.username, tenant_id=context.tenant_id)

    @property
    def namespace(self) -> str:
        """The physical collection-name prefix for this owner.

        Includes a digest of ``(tenant_id, username)`` so two distinct users can
        never share a namespace, even if their usernames slugify identically.
        """
        digest = hashlib.sha256(
            f"{self.tenant_id}\0{self.username}".encode("utf-8")
        ).hexdigest()[:_HASH_LENGTH]
        readable = slugify(self.username)[:_MAX_READABLE] or "user"
        return f"{readable}_{digest}"

    @property
    def prefix(self) -> str:
        """The namespace plus separator — what every collection of theirs starts with."""
        return f"{self.namespace}{SEPARATOR}"


def physical_name(owner: Owner, collection: str) -> str | None:
    """Map a user-facing collection name to its physical Qdrant name.

    Returns None when ``collection`` contains no usable characters, which the
    caller turns into a 422 rather than creating a nameless collection.
    """
    slug = slugify(collection)
    if not slug:
        return None
    return f"{owner.prefix}{slug}"


def owns(owner: Owner, physical: str) -> bool:
    """Whether a physical collection belongs to ``owner``."""
    return physical.startswith(owner.prefix)


def display_name(owner: Owner, physical: str) -> str | None:
    """Recover the user-facing name from a physical one, or None if not theirs.

    Returning None for a collection the owner does not hold is what keeps
    listings from mentioning anyone else's collections.
    """
    if not owns(owner, physical):
        return None
    return physical[len(owner.prefix) :] or None
