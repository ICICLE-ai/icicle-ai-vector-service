"""Cross-cutting concerns: configuration and authentication.

Nothing here knows about Qdrant or about HTTP routes — it is the layer both of
those depend on.
"""

from .security import UserContext, get_current_user
from .settings import settings

__all__ = ["UserContext", "get_current_user", "settings"]
