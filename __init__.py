"""
Cloak - Enterprise NER Extraction with Advanced Anonymization

A comprehensive solution for:
- Advanced Named Entity Recognition with validation
- Numbered redaction for consistent privacy protection  
- Synthetic data replacement using Faker
- Custom replacement strategies
- Enterprise-grade caching and parallel processing

Author: G Rohit
Version: 1.0.0
"""

from . import cloak
from .CloakExtraction import CloakExtraction

# Import main API functions for convenient access
from .cloak import extract, redact, replace, replace_with_data, anonymize, mask

__version__ = "1.0.0"
__author__ = "G Rohit"

__all__ = [
    'extract', 'redact', 'replace', 'replace_with_data', 
    'anonymize', 'mask', 'CloakExtraction'
]
