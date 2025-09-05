"""
Core extraction modules for Cloak NER pipeline.

Provides:
- EntityExtractor: Multi-pass extraction with intelligent thresholds
- ParallelEntityProcessor: Word-based parallel processing for large texts  
- Text chunking utilities with semantic boundary preservation
"""

from .extractor import EntityExtractor
from .chunker import chunk_text, estimate_chunk_count, validate_chunks, get_chunk_info
from .parallel_processor import ParallelEntityProcessor, extract_entities_in_parallel

__all__ = [
    'EntityExtractor',
    'chunk_text',
    'estimate_chunk_count', 
    'validate_chunks',
    'get_chunk_info',
    'ParallelEntityProcessor',
    'extract_entities_in_parallel'
]
