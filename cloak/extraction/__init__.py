"""
Core extraction modules for Cloak NER pipeline.

Provides:
- EntityExtractor: Multi-pass extraction with intelligent thresholds
- ParallelEntityProcessor: Word-based parallel processing for large texts
- Text chunking utilities with semantic boundary preservation
"""

from .chunker import chunk_text, estimate_chunk_count, get_chunk_info, validate_chunks
from .extractor import EntityExtractor
from .parallel_processor import ParallelEntityProcessor

__all__ = [
    "EntityExtractor",
    "chunk_text",
    "estimate_chunk_count",
    "validate_chunks",
    "get_chunk_info",
    "ParallelEntityProcessor",
]
