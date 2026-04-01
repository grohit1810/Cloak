"""
Utility modules for Cloak NER extraction pipeline.

Provides:
- EntityMerger: Merges adjacent entities with same labels
- CacheManager & CachedEntityExtractor: Advanced caching with analytics
- EntityValidator: Comprehensive validation and overlap resolution
"""

from .cache_manager import CachedEntityExtractor, CacheManager
from .entity_validator import EntityValidator
from .merger import EntityMerger

__all__ = ["EntityMerger", "CacheManager", "CachedEntityExtractor", "EntityValidator"]
