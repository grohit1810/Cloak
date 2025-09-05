"""
Anonymization modules for Cloak.

Provides:
- EntityRedactor: Numbered redaction with consistent placeholders
- EntityReplacer: Synthetic data replacement using multiple strategies
- Replacement strategies: Faker, Country, Date, Default
"""

from .redactor import EntityRedactor, RedactionDetail
from .replacer import EntityReplacer, ReplacementDetail

__all__ = [
    'EntityRedactor', 'RedactionDetail',
    'EntityReplacer', 'ReplacementDetail'
]
