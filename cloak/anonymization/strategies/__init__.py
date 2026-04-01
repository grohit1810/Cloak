"""
Replacement strategies for entity anonymization.

Available strategies:
- FakerReplacementStrategy: Realistic synthetic data using Faker
- CountryReplacementStrategy: Geographical data with context preservation
- DateReplacementStrategy: Date handling with format preservation
- DefaultReplacementStrategy: Fallback for any entity type
"""

from .base import ReplacementStrategy
from .country_strategy import CountryReplacementStrategy
from .date_strategy import DateReplacementStrategy
from .default_strategy import DefaultReplacementStrategy
from .faker_strategy import FakerReplacementStrategy

__all__ = [
    "ReplacementStrategy",
    "FakerReplacementStrategy",
    "CountryReplacementStrategy",
    "DateReplacementStrategy",
    "DefaultReplacementStrategy",
]
