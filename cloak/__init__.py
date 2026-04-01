"""
Cloak - Enterprise NER Extraction with Advanced Anonymization
"""

from .api import anonymize, extract, mask, redact, replace, replace_with_data
from .extraction_pipeline import CloakExtraction

__version__ = "1.0.0"
__all__ = [
    "extract",
    "redact",
    "replace",
    "replace_with_data",
    "anonymize",
    "mask",
    "CloakExtraction",
]
