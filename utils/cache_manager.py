"""
Cache Manager - Enhanced

Implements advanced caching strategy using functools.lru_cache with detailed analytics.
Enhanced with better statistics tracking and error handling.

Author: G Rohit (Enhanced from original)
Version: 1.0.0
"""

import functools
import logging
from typing import List, Dict, Any, Tuple, Optional

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
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0

        logger.info(f"CacheManager initialized with maxsize: {maxsize}")

    @staticmethod
    def create_cache_key(text: str, labels: List[str]) -> Tuple[str, Tuple[str, ...]]:
        """
        Create a cache key from text and labels.
        Labels are converted to tuple for immutability (required for caching).

        Args:
            text: Input text
            labels: List of entity labels

        Returns:
            Tuple of (text, labels_tuple) suitable for use as cache key
        """
        labels_tuple = tuple(sorted(labels)) if labels else tuple()
        return (text, labels_tuple)

    def create_cached_extractor(self, extractor_func):
        """
        Create a cached version of an extractor function.
        This implements @lru_cache strategy.

        Args:
            extractor_func: Function that takes (text, labels_tuple) and returns entities

        Returns:
            Cached version of the function
        """
        @functools.lru_cache(maxsize=self.maxsize)
        def cached_extractor(text: str, labels_tuple: Tuple[str, ...]):
            """
            Cached extraction function.
            Labels must be passed as a tuple for caching to work.
            """
            # Convert tuple back to list for the actual extractor
            labels_list = list(labels_tuple) if labels_tuple else None
            result = extractor_func(text, labels_list)
            return result

        # Wrap to handle cache hit tracking
        original_cached = cached_extractor

        def tracking_cached_extractor(text: str, labels_tuple: Tuple[str, ...]):
            # Track total requests
            self.total_requests += 1

            # Check if this would be a cache hit
            cache_info_before = original_cached.cache_info()
            result = original_cached(text, labels_tuple)
            cache_info_after = original_cached.cache_info()

            # Update our tracking
            if cache_info_after.hits > cache_info_before.hits:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            return result

        # Attach cache_info method and other lru_cache attributes
        tracking_cached_extractor.cache_info = original_cached.cache_info
        tracking_cached_extractor.cache_clear = original_cached.cache_clear

        return tracking_cached_extractor

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        miss_rate = (self.cache_misses / self.total_requests * 100) if self.total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": self.total_requests,
            "hit_rate_percentage": round(hit_rate, 2),
            "miss_rate_percentage": round(miss_rate, 2),
            "maxsize": self.maxsize,
            "efficiency": "High" if hit_rate > 70 else "Medium" if hit_rate > 40 else "Low"
        }

    def clear_stats(self):
        """Reset cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0 
        self.total_requests = 0
        logger.info("Cache statistics reset")

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
        self._cached_predict = self.cache_manager.create_cached_extractor(
            self._uncached_predict
        )

        logger.info(f"CachedEntityExtractor initialized with cache size {maxsize}")

    def _uncached_predict(self, text: str, labels: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Internal method that performs actual prediction without caching."""
        try:
            return self.extractor.predict(text, labels)
        except Exception as e:
            logger.error(f"Prediction failed in cached extractor: {str(e)}")
            return []

    def predict(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
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

        # Convert labels to tuple for caching
        labels_tuple = tuple(sorted(labels)) if labels else tuple()

        try:
            return self._cached_predict(text, labels_tuple)
        except Exception as e:
            logger.error(f"Cached prediction failed: {str(e)}")
            # Fallback to uncached prediction
            return self._uncached_predict(text, labels)

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information including built-in lru_cache stats.

        Returns:
            Dictionary with cache information
        """
        try:
            cache_info = self._cached_predict.cache_info()
            manager_stats = self.cache_manager.get_cache_stats()

            return {
                "lru_cache_hits": cache_info.hits,
                "lru_cache_misses": cache_info.misses,
                "lru_cache_current_size": cache_info.currsize,
                "lru_cache_max_size": cache_info.maxsize,
                "lru_cache_hit_ratio": cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
                "manager_stats": manager_stats
            }
        except Exception as e:
            logger.error(f"Failed to get cache info: {str(e)}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clear the cache and reset statistics."""
        try:
            self._cached_predict.cache_clear()
            self.cache_manager.clear_stats()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the underlying model."""
        try:
            return self.extractor.get_model_info()
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}
