"""Shared constants for the Cloak package."""

DEFAULT_LABELS: list[str] = ["person", "date", "location", "organization"]

DEFAULT_CHUNK_SIZE: int = 600
DEFAULT_MAX_PASSES: int = 2
DEFAULT_MIN_CONFIDENCE: float = 0.3
DEFAULT_CACHE_SIZE: int = 128
DEFAULT_MAX_WORKERS: int = 4

FIRST_PASS_THRESHOLD: float = 0.5
SUBSEQUENT_PASS_THRESHOLD: float = 0.30

MAX_ENTITY_LENGTH: int = 200
