"""
Replacement strategies for entity anonymization.

Available strategies:
- FakerReplacementStrategy: Realistic synthetic data using Faker
- CountryReplacementStrategy: Geographical data with context preservation  
- DateReplacementStrategy: Date handling with format preservation
- DefaultReplacementStrategy: Fallback for any entity type
"""

from .faker_strategy import FakerReplacementStrategy
from .country_strategy import CountryReplacementStrategy
from .date_strategy import DateReplacementStrategy
from .default_strategy import DefaultReplacementStrategy

__all__ = [
    'FakerReplacementStrategy',
    'CountryReplacementStrategy', 
    'DateReplacementStrategy',
    'DefaultReplacementStrategy'
]
