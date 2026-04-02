"""
Cache Manager - Enhanced

Implements advanced caching strategy using functools.lru_cache with detailed analytics.
Enhanced with better statistics tracking and error handling.

Author: G Rohit (Enhanced from original)
Version: 1.0.0
"""

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Enhanced cache manager for entity extraction results.

    Implements caching strategy using @lru_cache decorator with comprehensive
    statistics tracking and performance monitoring.
    """

    def __init__(self, maxsize: int = 128):
        """
        Initialize the cache manager.

        Args:
            maxsize: Maximum number of cached results (default: 128)
        """
        self.maxsize = maxsize

        logger.info("CacheManager initialized with maxsize: %d", maxsize)

    def create_cached_extractor(self, extractor_func):
        """Create a cached version of an extractor function."""

        @functools.lru_cache(maxsize=self.maxsize)
        def cached_extractor(text: str, labels_tuple: tuple[str, ...]):
            labels_list = list(labels_tuple) if labels_tuple else None
            return extractor_func(text, labels_list)

        return cached_extractor

    def get_cache_stats(self, cached_func=None) -> dict[str, Any]:
        """Get cache performance statistics derived from lru_cache info."""
        if cached_func and hasattr(cached_func, "cache_info"):
            info = cached_func.cache_info()
            total = info.hits + info.misses
            hit_rate = (info.hits / total * 100) if total > 0 else 0
            return {
                "cache_hits": info.hits,
                "cache_misses": info.misses,
                "total_requests": total,
                "hit_rate_percentage": round(hit_rate, 2),
                "maxsize": self.maxsize,
            }
        return {"maxsize": self.maxsize, "total_requests": 0}


class CachedEntityExtractor:
    """
    Enhanced wrapper that adds caching to any entity extractor.

    Implements the caching pattern with comprehensive analytics and error handling.
    """

    def __init__(self, extractor, maxsize: int = 128):
        """
        Initialize cached extractor wrapper.

        Args:
            extractor: EntityExtractor instance to wrap with caching
            maxsize: Maximum cache size
        """
        self.extractor = extractor
        self.cache_manager = CacheManager(maxsize)

        # Create cached version of the predict method
        self._cached_predict = self.cache_manager.create_cached_extractor(self._uncached_predict)

        logger.info("CachedEntityExtractor initialized with cache size %d", maxsize)

    def _uncached_predict(self, text: str, labels: list[str] | None) -> list[dict[str, Any]]:
        """Internal method that performs actual prediction without caching."""
        try:
            return self.extractor.predict(text, labels)
        except Exception as e:
            logger.error("Prediction failed in cached extractor: %s", str(e))
            return []

    def predict(
        self, text: str, labels: list[str] | None = None, use_cache: bool = True
    ) -> list[dict[str, Any]]:
        """
        Predict entities with optional caching.

        Args:
            text: Input text
            labels: Entity labels to detect
            use_cache: Whether to use caching (default: True)

        Returns:
            List of detected entities
        """
        if not use_cache:
            return self._uncached_predict(text, labels)

        # Labels are sorted to normalize the cache key — GLiNER output is label-order-independent
        labels_tuple = tuple(sorted(labels)) if labels else ()

        try:
            result = self._cached_predict(text, labels_tuple)
            return list(result)  # Defensive copy — don't expose cached list
        except Exception as e:
            logger.error("Cached prediction failed: %s", e)
            # Fallback to uncached prediction
            return self._uncached_predict(text, labels)

    def get_cache_info(self) -> dict[str, Any]:
        """Get detailed cache information from lru_cache stats."""
        try:
            info = self._cached_predict.cache_info()
            total = info.hits + info.misses
            return {
                "lru_cache_hits": info.hits,
                "lru_cache_misses": info.misses,
                "lru_cache_current_size": info.currsize,
                "lru_cache_max_size": info.maxsize,
                "lru_cache_hit_ratio": info.hits / total if total > 0 else 0,
            }
        except Exception as e:
            logger.error("Failed to get cache info: %s", e)
            return {"error": str(e)}

    def clear_cache(self):
        """Clear the cache."""
        try:
            self._cached_predict.cache_clear()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error("Failed to clear cache: %s", e)

    def get_model_info(self) -> dict[str, str]:
        """Get information about the underlying model."""
        try:
            return self.extractor.get_model_info()
        except Exception as e:
            logger.error("Failed to get model info: %s", str(e))
            return {"error": str(e)}
