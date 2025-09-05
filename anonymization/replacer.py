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

import random
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    logging.warning("Faker not available. Install with: pip install faker")

from .strategies.faker_strategy import FakerReplacementStrategy
from .strategies.country_strategy import CountryReplacementStrategy  
from .strategies.date_strategy import DateReplacementStrategy
from .strategies.default_strategy import DefaultReplacementStrategy

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

    def __init__(self, locale: str = 'en_US', ensure_consistency: bool = True):
        """
        Initialize the entity replacer.

        Args:
            locale: Faker locale for data generation (default: 'en_US')
            ensure_consistency: Whether to ensure consistent replacements (default: True)
        """
        self.locale = locale
        self.ensure_consistency = ensure_consistency
        self.replacement_cache = {}  # Cache for consistent replacements

        # Initialize Faker if available
        if FAKER_AVAILABLE:
            self.faker = Faker(locale)
        else:
            self.faker = None

        # Initialize replacement strategies
        self._init_strategies()

        logger.info(f"EntityReplacer initialized with locale: {locale}")
        logger.info(f"Faker available: {FAKER_AVAILABLE}")
        logger.info(f"Consistency enabled: {ensure_consistency}")

    def _init_strategies(self):
        """Initialize replacement strategies."""
        self.strategies = {
            'faker': FakerReplacementStrategy(self.faker) if self.faker else None,
            'country': CountryReplacementStrategy(),
            'date': DateReplacementStrategy(self.faker),
            'default': DefaultReplacementStrategy()
        }

        # Entity type to strategy mapping
        self.strategy_mapping = {
            'person': ['faker', 'default'],
            'location': ['country', 'faker', 'default'], 
            'date': ['date', 'faker', 'default'],
            'organization': ['faker', 'default'],
            'company': ['faker', 'default'],
            'email': ['faker', 'default'],
            'phone': ['faker', 'default'],
            'address': ['faker', 'default'],
            'age': ['faker', 'default'],
            'nationality': ['country', 'default'],
            'country': ['country', 'default']
        }

    def replace(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        ensure_consistency: Optional[bool] = None,
        custom_strategies: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
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
                'anonymized_text': text,
                'replacements': [],
                'replacement_info': {'entities_processed': 0, 'replacements_applied': 0}
            }

        consistency = ensure_consistency if ensure_consistency is not None else self.ensure_consistency
        strategies = custom_strategies or {}

        logger.info(f"Starting replacement of {len(entities)} entities")
        logger.info(f"Consistency enabled: {consistency}")

        # Sort entities by start position (reverse order for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)

        replaced_text = text
        replacement_details = []
        replacement_map = {}

        # Process entities
        for entity in sorted_entities:
            try:
                label = entity['label'].lower()
                original_text = entity['text']
                start_pos = entity['start']
                end_pos = entity['end']
                score = entity.get('score', 1.0)

                # Get replacement
                replacement, strategy_used = self._get_replacement(
                    entity=entity,
                    consistency=consistency,
                    custom_strategy=strategies.get(label)
                )

                # Apply replacement  
                if replacement and replacement != original_text:
                    replaced_text = replaced_text[:start_pos] + replacement + replaced_text[end_pos:]

                    # Create replacement detail
                    detail = ReplacementDetail(
                        label=label,
                        original=original_text,
                        replacement=replacement,
                        start=start_pos,
                        end=end_pos,
                        score=score,
                        strategy_used=strategy_used
                    )
                    replacement_details.append(detail)
                    replacement_map[original_text] = replacement

                    logger.debug(f"Replaced '{original_text}' -> '{replacement}' using {strategy_used}")
                else:
                    logger.debug(f"No replacement for '{original_text}' (label: {label})")

            except Exception as e:
                logger.error(f"Error replacing entity {entity}: {str(e)}")
                continue

        # Sort replacement details by original position
        replacement_details.sort(key=lambda x: x.start)

        # Calculate statistics
        strategies_used = defaultdict(int)
        for detail in replacement_details:
            strategies_used[detail.strategy_used] += 1

        result = {
            'anonymized_text': replaced_text,
            'replacements': replacement_details,
            'replacement_info': {
                'entities_processed': len(entities),
                'replacements_applied': len(replacement_details),
                'consistency_enabled': consistency,
                'strategies_used': dict(strategies_used),
                'unique_replacements': len(set(d.replacement for d in replacement_details))
            },
            'replacement_map': replacement_map
        }

        logger.info(f"Replacement complete: {len(replacement_details)} replacements applied")
        return result

    def replace_with_user_data(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        user_replacements: Dict[str, Union[str, List[str]]],
        ensure_consistency: Optional[bool] = None
    ) -> Dict[str, Any]:
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
                'anonymized_text': text,
                'replacements': [],
                'replacement_info': {'entities_processed': 0, 'replacements_applied': 0}
            }

        consistency = ensure_consistency if ensure_consistency is not None else self.ensure_consistency

        logger.info(f"Starting user data replacement of {len(entities)} entities")
        logger.info(f"User replacements for labels: {list(user_replacements.keys())}")

        # Sort entities by start position (reverse order)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)

        replaced_text = text
        replacement_details = []
        replacement_map = {}

        # Build consistency cache for user data
        if consistency:
            consistency_cache = {}

        for entity in sorted_entities:
            try:
                label = entity['label'].lower()
                original_text = entity['text']
                start_pos = entity['start']
                end_pos = entity['end']
                score = entity.get('score', 1.0)

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

                    # Apply replacement
                    if replacement:
                        replaced_text = replaced_text[:start_pos] + replacement + replaced_text[end_pos:]

                        detail = ReplacementDetail(
                            label=label,
                            original=original_text,
                            replacement=replacement,
                            start=start_pos,
                            end=end_pos,
                            score=score,
                            strategy_used='user_data'
                        )
                        replacement_details.append(detail)
                        replacement_map[original_text] = replacement

                        logger.debug(f"User replaced '{original_text}' -> '{replacement}'")

            except Exception as e:
                logger.error(f"Error in user replacement for entity {entity}: {str(e)}")
                continue

        # Sort by original position
        replacement_details.sort(key=lambda x: x.start)

        result = {
            'anonymized_text': replaced_text,
            'replacements': replacement_details,
            'replacement_info': {
                'entities_processed': len(entities),
                'replacements_applied': len(replacement_details),
                'consistency_enabled': consistency,
                'user_data_labels': list(user_replacements.keys()),
                'strategy_used': 'user_data'
            },
            'replacement_map': replacement_map
        }

        logger.info(f"User data replacement complete: {len(replacement_details)} replacements applied")
        return result

    def _get_replacement(
        self,
        entity: Dict[str, Any],
        consistency: bool,
        custom_strategy: Optional[str] = None
    ) -> tuple[str, str]:
        """Get replacement for an entity using appropriate strategy."""
        label = entity['label'].lower()
        original_text = entity['text']

        # Check consistency cache first
        if consistency:
            cache_key = (label, original_text)
            if cache_key in self.replacement_cache:
                cached_replacement, cached_strategy = self.replacement_cache[cache_key]
                return cached_replacement, f"{cached_strategy}_cached"

        # Determine strategies to try
        if custom_strategy:
            strategies_to_try = [custom_strategy]
        else:
            strategies_to_try = self.strategy_mapping.get(label, ['faker', 'default'])

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
                    logger.debug(f"Strategy {strategy_name} failed for {label}: {str(e)}")
                    continue

        # Fallback to default strategy
        try:
            replacement = self.strategies['default'].get_replacement(entity)
            if consistency:
                self.replacement_cache[cache_key] = (replacement, 'default')
            return replacement, 'default'
        except Exception as e:
            logger.error(f"Default strategy failed for {entity}: {str(e)}")
            return original_text, 'none'

    def _select_user_replacement(self, user_data: Union[str, List[str]]) -> str:
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

    def get_replacement_stats(self) -> Dict[str, Any]:
        """Get statistics about replacement operations."""
        cache_size = len(self.replacement_cache)

        # Analyze cached strategies
        strategy_distribution = defaultdict(int)
        for _, (_, strategy) in self.replacement_cache.items():
            strategy_distribution[strategy] += 1

        return {
            'cache_size': cache_size,
            'consistency_enabled': self.ensure_consistency,
            'faker_available': FAKER_AVAILABLE,
            'locale': self.locale,
            'cached_strategies': dict(strategy_distribution),
            'available_strategies': list(self.strategies.keys())
        }
