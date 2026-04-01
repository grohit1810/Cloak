"""Base protocol for replacement strategies."""

from typing import Any, Protocol


class ReplacementStrategy(Protocol):
    """Interface that all replacement strategies must satisfy."""

    def can_handle(self, label: str) -> bool:
        """Return True if this strategy can handle the given entity label."""
        ...

    def get_replacement(self, entity: dict[str, Any]) -> str | None:
        """Generate a replacement for the entity, or None if unable."""
        ...
