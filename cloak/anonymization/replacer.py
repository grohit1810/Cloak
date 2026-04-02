"""
Entity Replacer - Synthetic Data Replacement System

Provides enterprise-grade entity replacement with:
- Faker integration for realistic synthetic data
- Consistent replacement for identical entities
- Custom replacement strategies per entity type
- Fallback mechanisms for unknown entity types

Author: G Rohit
Version: 1.0.0
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

try:
    from faker import Faker

    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    logging.warning("Faker not available. Install with: pip install faker")

from .strategies.base import ReplacementStrategy
from .strategies.country_strategy import CountryReplacementStrategy
from .strategies.date_strategy import DateReplacementStrategy
from .strategies.default_strategy import DefaultReplacementStrategy
from .strategies.faker_strategy import FakerReplacementStrategy

logger = logging.getLogger(__name__)


@dataclass
class ReplacementDetail:
    """Details about a single entity replacement."""

    label: str
    original: str
    replacement: str
    start: int
    end: int
    score: float
    strategy_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "original": self.original,
            "replacement": self.replacement,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "strategy_used": self.strategy_used,
        }


class EntityReplacer:
    """
    Advanced entity replacer with multiple strategies and consistency tracking.

    Features:
    - Faker integration for realistic synthetic data
    - Consistent replacement for identical entities
    - Pluggable replacement strategies
    - Custom user data support
    - Fallback mechanisms
    """

    def __init__(
        self, locale: str = "en_US", ensure_consistency: bool = True, seed: int | None = None
    ):
        """
        Initialize the entity replacer.

        Args:
            locale: Faker locale for data generation (default: 'en_US')
            ensure_consistency: Whether to ensure consistent replacements (default: True)
            seed: Optional seed for Faker's random generator for reproducible output
        """
        self.locale = locale
        self.ensure_consistency = ensure_consistency
        self.seed = seed
        self.replacement_cache = {}  # Cache for consistent replacements

        # Initialize Faker if available
        if FAKER_AVAILABLE:
            self.faker = Faker(locale)
            if seed is not None:
                self.faker.seed_instance(seed)
        else:
            self.faker = None

        # Initialize replacement strategies
        self._init_strategies()

        logger.info("EntityReplacer initialized with locale: %s", locale)
        logger.info("Faker available: %s", FAKER_AVAILABLE)
        logger.info("Consistency enabled: %s", ensure_consistency)

    def _init_strategies(self):
        """Initialize replacement strategies."""
        self.strategies: dict[str, ReplacementStrategy | None] = {
            "faker": FakerReplacementStrategy(self.faker) if self.faker else None,
            "country": CountryReplacementStrategy(),
            "date": DateReplacementStrategy(self.faker),
            "default": DefaultReplacementStrategy(),
        }

        # Entity type to strategy mapping
        self.strategy_mapping = {
            "person": ["faker", "default"],
            "location": ["country", "faker", "default"],
            "date": ["date", "faker", "default"],
            "organization": ["faker", "default"],
            "company": ["faker", "default"],
            "email": ["faker", "default"],
            "phone": ["faker", "default"],
            "address": ["faker", "default"],
            "age": ["faker", "default"],
            "nationality": ["country", "default"],
            "country": ["country", "default"],
        }

    def replace(
        self,
        text: str,
        entities: list[dict[str, Any]],
        ensure_consistency: bool | None = None,
        custom_strategies: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Replace entities with synthetic alternatives.

        Args:
            text: Original text to process
            entities: List of entity dictionaries
            ensure_consistency: Override default consistency behavior
            custom_strategies: Custom strategy mapping for entity types

        Returns:
            Dictionary with replacement results and metadata
        """
        if not entities:
            return {
                "anonymized_text": text,
                "replacements": [],
                "replacement_info": {"entities_processed": 0, "replacements_applied": 0},
            }

        consistency = (
            ensure_consistency if ensure_consistency is not None else self.ensure_consistency
        )
        strategies = custom_strategies or {}

        logger.info("Starting replacement of %d entities", len(entities))
        logger.info("Consistency enabled: %s", consistency)

        # Sort entities by start position ascending for forward-pass assembly
        sorted_entities = sorted(entities, key=lambda x: x["start"])

        replacement_details = []
        replacement_map = {}

        # First pass: compute replacements for each entity
        replacements_to_apply: list[tuple[int, int, str]] = []
        for entity in sorted_entities:
            try:
                label = entity["label"].lower()
                original_text = entity["text"]
                start_pos = entity["start"]
                end_pos = entity["end"]
                score = entity.get("score", 1.0)

                # Get replacement
                replacement, strategy_used = self._get_replacement(
                    entity=entity, consistency=consistency, custom_strategy=strategies.get(label)
                )

                # Track replacement
                if replacement and replacement != original_text:
                    replacements_to_apply.append((start_pos, end_pos, replacement))

                    # Create replacement detail
                    detail = ReplacementDetail(
                        label=label,
                        original=original_text,
                        replacement=replacement,
                        start=start_pos,
                        end=end_pos,
                        score=score,
                        strategy_used=strategy_used,
                    )
                    replacement_details.append(detail)
                    replacement_map[original_text] = replacement

                    logger.debug(
                        "Replaced '%s' -> '%s' using %s", original_text, replacement, strategy_used
                    )
                else:
                    logger.debug("No replacement for '%s' (label: %s)", original_text, label)

            except Exception as e:
                logger.error("Error replacing entity %s: %s", entity, str(e))
                continue

        # Build output text in a single forward pass (O(N+L) instead of O(N*L))
        segments: list[str] = []
        prev_end = 0
        for start_pos, end_pos, replacement in replacements_to_apply:
            segments.append(text[prev_end:start_pos])
            segments.append(replacement)
            prev_end = end_pos
        segments.append(text[prev_end:])
        replaced_text = "".join(segments)

        # Calculate statistics
        strategies_used = defaultdict(int)
        for detail in replacement_details:
            strategies_used[detail.strategy_used] += 1

        result = {
            "anonymized_text": replaced_text,
            "replacements": replacement_details,
            "replacement_info": {
                "entities_processed": len(entities),
                "replacements_applied": len(replacement_details),
                "consistency_enabled": consistency,
                "strategies_used": dict(strategies_used),
                "unique_replacements": len(set(d.replacement for d in replacement_details)),
            },
            "replacement_map": replacement_map,
        }

        logger.info("Replacement complete: %d replacements applied", len(replacement_details))
        return result

    def replace_with_user_data(
        self,
        text: str,
        entities: list[dict[str, Any]],
        user_replacements: dict[str, str | list[str]],
        ensure_consistency: bool | None = None,
    ) -> dict[str, Any]:
        """
        Replace entities with user-provided data.

        Args:
            text: Original text to process
            entities: List of entity dictionaries
            user_replacements: Dictionary mapping labels to replacement values
            ensure_consistency: Override default consistency behavior

        Returns:
            Dictionary with replacement results and metadata
        """
        if not entities or not user_replacements:
            return {
                "anonymized_text": text,
                "replacements": [],
                "replacement_info": {"entities_processed": 0, "replacements_applied": 0},
            }

        consistency = (
            ensure_consistency if ensure_consistency is not None else self.ensure_consistency
        )

        logger.info("Starting user data replacement of %d entities", len(entities))
        logger.info("User replacements for labels: %s", list(user_replacements.keys()))

        # Sort entities by start position ascending for forward-pass assembly
        sorted_entities = sorted(entities, key=lambda x: x["start"])

        replacement_details = []
        replacement_map = {}

        # Build consistency cache for user data
        consistency_cache: dict[tuple[str, str], str] = {}

        # First pass: compute replacements for each entity
        replacements_to_apply: list[tuple[int, int, str]] = []
        for entity in sorted_entities:
            try:
                label = entity["label"].lower()
                original_text = entity["text"]
                start_pos = entity["start"]
                end_pos = entity["end"]
                score = entity.get("score", 1.0)

                # Get user replacement for this label
                if label in user_replacements:
                    user_data = user_replacements[label]

                    if consistency:
                        # Use cached replacement if available
                        cache_key = (label, original_text)
                        if cache_key in consistency_cache:
                            replacement = consistency_cache[cache_key]
                        else:
                            replacement = self._select_user_replacement(user_data)
                            consistency_cache[cache_key] = replacement
                    else:
                        replacement = self._select_user_replacement(user_data)

                    # Track replacement
                    if replacement:
                        replacements_to_apply.append((start_pos, end_pos, replacement))

                        detail = ReplacementDetail(
                            label=label,
                            original=original_text,
                            replacement=replacement,
                            start=start_pos,
                            end=end_pos,
                            score=score,
                            strategy_used="user_data",
                        )
                        replacement_details.append(detail)
                        replacement_map[original_text] = replacement

                        logger.debug("User replaced '%s' -> '%s'", original_text, replacement)

            except Exception as e:
                logger.error("Error in user replacement for entity %s: %s", entity, str(e))
                continue

        # Build output text in a single forward pass (O(N+L) instead of O(N*L))
        segments: list[str] = []
        prev_end = 0
        for start_pos, end_pos, replacement in replacements_to_apply:
            segments.append(text[prev_end:start_pos])
            segments.append(replacement)
            prev_end = end_pos
        segments.append(text[prev_end:])
        replaced_text = "".join(segments)

        result = {
            "anonymized_text": replaced_text,
            "replacements": replacement_details,
            "replacement_info": {
                "entities_processed": len(entities),
                "replacements_applied": len(replacement_details),
                "consistency_enabled": consistency,
                "user_data_labels": list(user_replacements.keys()),
                "strategy_used": "user_data",
            },
            "replacement_map": replacement_map,
        }

        logger.info(
            "User data replacement complete: %d replacements applied", len(replacement_details)
        )
        return result

    def _get_replacement(
        self, entity: dict[str, Any], consistency: bool, custom_strategy: str | None = None
    ) -> tuple[str, str]:
        """Get replacement for an entity using appropriate strategy."""
        label = entity["label"].lower()
        original_text = entity["text"]
        cache_key = (label, original_text)

        # Check consistency cache first
        if consistency:
            if cache_key in self.replacement_cache:
                cached_replacement, cached_strategy = self.replacement_cache[cache_key]
                return cached_replacement, f"{cached_strategy}_cached"

        # Determine strategies to try
        if custom_strategy:
            strategies_to_try = [custom_strategy]
        else:
            strategies_to_try = self.strategy_mapping.get(label, ["faker", "default"])

        # Try strategies in order
        for strategy_name in strategies_to_try:
            strategy = self.strategies.get(strategy_name)
            if strategy and strategy.can_handle(label):
                try:
                    replacement = strategy.get_replacement(entity)
                    if replacement and replacement != original_text:
                        # Cache the result if consistency is enabled
                        if consistency:
                            self.replacement_cache[cache_key] = (replacement, strategy_name)
                        return replacement, strategy_name
                except Exception as e:
                    logger.debug("Strategy %s failed for %s: %s", strategy_name, label, str(e))
                    continue

        # Fallback to default strategy
        try:
            replacement = self.strategies["default"].get_replacement(entity)
            if consistency:
                self.replacement_cache[cache_key] = (replacement, "default")
            return replacement, "default"
        except Exception as e:
            logger.error("Default strategy failed for %s: %s", entity, str(e))
            return original_text, "none"

    def _select_user_replacement(self, user_data: str | list[str]) -> str:
        """Select a replacement from user data."""
        if isinstance(user_data, str):
            return user_data
        elif isinstance(user_data, list) and user_data:
            return random.choice(user_data)
        else:
            return ""

    def clear_cache(self):
        """Clear the replacement consistency cache."""
        self.replacement_cache.clear()
        logger.info("Replacement cache cleared")

    def get_replacement_stats(self) -> dict[str, Any]:
        """Get statistics about replacement operations."""
        cache_size = len(self.replacement_cache)

        # Analyze cached strategies
        strategy_distribution = defaultdict(int)
        for _, (_, strategy) in self.replacement_cache.items():
            strategy_distribution[strategy] += 1

        return {
            "cache_size": cache_size,
            "consistency_enabled": self.ensure_consistency,
            "faker_available": FAKER_AVAILABLE,
            "locale": self.locale,
            "cached_strategies": dict(strategy_distribution),
            "available_strategies": list(self.strategies.keys()),
        }
