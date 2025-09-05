"""
Utility modules for Cloak NER extraction pipeline.

Provides:
- EntityMerger: Merges adjacent entities with same labels
- CacheManager & CachedEntityExtractor: Advanced caching with analytics
- EntityValidator: Comprehensive validation and overlap resolution
"""

from .merger import EntityMerger
from .cache_manager import CacheManager, CachedEntityExtractor
from .entity_validator import EntityValidator

__all__ = [
    'EntityMerger', 
    'CacheManager', 
    'CachedEntityExtractor', 
    'EntityValidator'
]
