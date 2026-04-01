"""
Cloak - Enterprise NER Extraction with Advanced Anonymization

Main API entry point providing simple, intuitive functions for:
- Entity extraction with advanced validation
- Numbered redaction for privacy protection
- Synthetic data replacement using Faker
- Custom data replacement

Author: G Rohit
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Union

from .anonymization.redactor import EntityRedactor
from .anonymization.replacer import EntityReplacer
from .constants import DEFAULT_LABELS
from .extraction_pipeline import CloakExtraction

# Global instances for caching across calls
_global_cloak_instance: CloakExtraction | None = None
_global_cloak_model_path: str | None = None
_global_redactor: EntityRedactor | None = None
_global_replacer: EntityReplacer | None = None


def _reset_global_instances() -> None:
    """Reset all global instances. Used for testing."""
    global _global_cloak_instance, _global_cloak_model_path
    global _global_redactor, _global_replacer
    _global_cloak_instance = None
    _global_cloak_model_path = None
    _global_redactor = None
    _global_replacer = None


def _get_cloak_instance(model_path: str | None = None, **kwargs) -> CloakExtraction:
    """Get or create global Cloak instance, recreating if model_path changes."""
    global _global_cloak_instance, _global_cloak_model_path
    if _global_cloak_instance is not None and model_path == _global_cloak_model_path:
        return _global_cloak_instance
    _global_cloak_instance = CloakExtraction(model_path=model_path, **kwargs)
    _global_cloak_model_path = model_path
    return _global_cloak_instance


def _get_global_redactor_instance() -> EntityRedactor:
    """Get or create global redactor instance."""
    global _global_redactor
    if _global_redactor is None:
        _global_redactor = EntityRedactor()
    return _global_redactor


def _get_global_replacer_instance() -> EntityReplacer:
    """Get or create global replacer instance."""
    global _global_replacer
    if _global_replacer is None:
        _global_replacer = EntityReplacer()
    return _global_replacer


def extract(
    text: str, labels: Optional[List[str]] = None, model_path: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Extract entities from text with advanced validation and processing.

    Args:
        text: Input text to analyze
        labels: Entity labels to detect (e.g., ['person', 'location', 'date'])
        model_path: Path to GLINER ONNX model (required for first call)
        **kwargs: Additional extraction parameters

    Returns:
        Dictionary with extracted entities and processing metadata

    Example:
        >>> result = cloak.extract("John works at Google", labels=['person', 'company'])
        >>> print(result['entities'])
    """
    cloak_instance = _get_cloak_instance(model_path=model_path, **kwargs)
    return cloak_instance.extract_entities(text, labels or DEFAULT_LABELS)


def redact(
    text: str,
    labels: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    numbered: bool = True,
    placeholder_format: str = "#{id}_{label}_REDACTED",
    **kwargs,
) -> Dict[str, Any]:
    """
    Redact sensitive entities with numbered placeholders for consistency.

    Args:
        text: Input text to redact
        labels: Entity labels to redact (e.g., ['person', 'location'])
        model_path: Path to GLINER ONNX model (required for first call)
        numbered: Whether to use numbered placeholders (default: True)
        placeholder_format: Format string for placeholders (default: "#{id}_{label}_REDACTED")
        **kwargs: Additional extraction parameters

    Returns:
        Dictionary with:
        - 'anonymized_text': Text with entities redacted
        - 'entities': List of detected entities
        - 'replacements': Mapping of original -> placeholder
        - 'processing_info': Metadata about the process

    Example:
        >>> result = cloak.redact("John works at Google Inc.", labels=['person', 'company'])
        >>> print(result['anonymized_text'])
        # "#1_PERSON_REDACTED works at #1_COMPANY_REDACTED."
    """
    # First extract entities
    cloak_instance = _get_cloak_instance(model_path=model_path, **kwargs)
    extraction_result = cloak_instance.extract_entities(text, labels or DEFAULT_LABELS)

    # Then redact them
    redactor = _get_global_redactor_instance()
    redaction_result = redactor.redact(
        text=text,
        entities=extraction_result["entities"],
        numbered=numbered,
        placeholder_format=placeholder_format,
    )

    # Combine results
    return {
        "anonymized_text": redaction_result["anonymized_text"],
        "entities": extraction_result["entities"],
        "replacements": redaction_result["replacements"],
        "processing_info": extraction_result["processing_info"],
        "redaction_info": redaction_result["redaction_info"],
    }


def replace(
    text: str,
    labels: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    ensure_consistency: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Replace sensitive entities with realistic synthetic alternatives.

    Args:
        text: Input text to anonymize
        labels: Entity labels to replace (e.g., ['person', 'location'])
        model_path: Path to GLINER ONNX model (required for first call)
        ensure_consistency: Whether to use same replacement for identical entities
        **kwargs: Additional extraction parameters

    Returns:
        Dictionary with:
        - 'anonymized_text': Text with entities replaced
        - 'entities': List of detected entities
        - 'replacements': Mapping of original -> replacement
        - 'processing_info': Metadata about the process

    Example:
        >>> result = cloak.replace("John Smith lives in Paris", labels=['person', 'location'])
        >>> print(result['anonymized_text'])
        # "Alice Johnson lives in Berlin"
    """
    # First extract entities
    cloak_instance = _get_cloak_instance(model_path=model_path, **kwargs)
    extraction_result = cloak_instance.extract_entities(text, labels or DEFAULT_LABELS)

    # Then replace them
    replacer = _get_global_replacer_instance()
    replacement_result = replacer.replace(
        text=text, entities=extraction_result["entities"], ensure_consistency=ensure_consistency
    )

    # Combine results
    return {
        "anonymized_text": replacement_result["anonymized_text"],
        "entities": extraction_result["entities"],
        "replacements": replacement_result["replacements"],
        "processing_info": extraction_result["processing_info"],
        "replacement_info": replacement_result["replacement_info"],
    }


def replace_with_data(
    text: str,
    labels: Optional[List[str]] = None,
    user_replacements: Optional[Dict[str, Union[str, List[str]]]] = None,
    model_path: Optional[str] = None,
    ensure_consistency: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Replace entities with user-provided replacement data.

    Args:
        text: Input text to anonymize
        labels: Entity labels to replace
        user_replacements: Dictionary mapping labels to replacement values
        model_path: Path to GLINER ONNX model (required for first call)
        ensure_consistency: Whether to use same replacement for identical entities
        **kwargs: Additional extraction parameters

    Returns:
        Dictionary with anonymized text and replacement metadata

    Example:
        >>> replacements = {'person': 'Anonymous', 'company': ['TechCorp', 'DataCorp']}
        >>> result = cloak.replace_with_data(
        ...     "John works at Google",
        ...     labels=['person', 'company'],
        ...     user_replacements=replacements
        ... )
    """
    if not user_replacements:
        raise ValueError("user_replacements dictionary must be provided")

    # First extract entities
    cloak_instance = _get_cloak_instance(model_path=model_path, **kwargs)
    extraction_result = cloak_instance.extract_entities(text, labels or DEFAULT_LABELS)

    # Then replace with user data
    replacer = _get_global_replacer_instance()
    replacement_result = replacer.replace_with_user_data(
        text=text,
        entities=extraction_result["entities"],
        user_replacements=user_replacements,
        ensure_consistency=ensure_consistency,
    )

    # Combine results
    return {
        "anonymized_text": replacement_result["anonymized_text"],
        "entities": extraction_result["entities"],
        "replacements": replacement_result["replacements"],
        "processing_info": extraction_result["processing_info"],
        "replacement_info": replacement_result["replacement_info"],
    }


# Convenience aliases
anonymize = redact  # Alias for redact
mask = redact  # Alias for redact

__all__ = ["extract", "redact", "replace", "replace_with_data", "anonymize", "mask"]
