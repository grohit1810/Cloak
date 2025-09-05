"""
Basic tests for Cloak NER extraction and anonymization.

Author: G Rohit
Version: 1.0.0
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path so we can import cloak
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import cloak
    from CloakExtraction import CloakExtraction
    from anonymization.redactor import EntityRedactor, RedactionDetail
    from anonymization.replacer import EntityReplacer, ReplacementDetail
except ImportError as e:
    pytest.skip(f"Could not import modules: {e}", allow_module_level=True)

class TestBasicFunctionality:
    """Test basic Cloak functionality."""

    def test_redaction_detail_creation(self):
        """Test RedactionDetail dataclass creation."""
        detail = RedactionDetail(
            label="person",
            original="John",
            placeholder="#1_PERSON_REDACTED", 
            start=0,
            end=4,
            score=0.9,
            redaction_id="1"
        )

        assert detail.label == "person"
        assert detail.original == "John"
        assert detail.placeholder == "#1_PERSON_REDACTED"
        assert detail.start == 0
        assert detail.end == 4
        assert detail.score == 0.9
        assert detail.redaction_id == "1"

    def test_replacement_detail_creation(self):
        """Test ReplacementDetail dataclass creation."""
        detail = ReplacementDetail(
            label="person",
            original="John", 
            replacement="Alice",
            start=0,
            end=4,
            score=0.9,
            strategy_used="faker"
        )

        assert detail.label == "person"
        assert detail.original == "John"
        assert detail.replacement == "Alice"
        assert detail.start == 0
        assert detail.end == 4
        assert detail.score == 0.9
        assert detail.strategy_used == "faker"

    def test_entity_redactor_initialization(self):
        """Test EntityRedactor initialization.""" 
        redactor = EntityRedactor()

        assert redactor.default_format == "#{id}_{label}_REDACTED"
        assert redactor.entity_id_map == {}
        assert len(redactor.used_ids_per_label) == 0
        assert redactor.redaction_history == []

    def test_entity_redactor_with_custom_format(self):
        """Test EntityRedactor with custom format."""
        custom_format = "#{id}_{label}_HIDDEN"
        redactor = EntityRedactor(default_format=custom_format)

        assert redactor.default_format == custom_format

    def test_entity_replacer_initialization(self):
        """Test EntityReplacer initialization."""
        replacer = EntityReplacer()

        assert replacer.locale == 'en_US'
        assert replacer.ensure_consistency == True
        assert replacer.replacement_cache == {}
        assert replacer.strategies is not None

    def test_redactor_empty_entities(self):
        """Test redactor with empty entities list."""
        redactor = EntityRedactor()

        result = redactor.redact(
            text="This is a test.",
            entities=[]
        )

        assert result['anonymized_text'] == "This is a test."
        assert result['replacements'] == []
        assert result['redaction_info']['entities_processed'] == 0
        assert result['redaction_info']['redactions_applied'] == 0

    def test_replacer_empty_entities(self):
        """Test replacer with empty entities list."""
        replacer = EntityReplacer()

        result = replacer.replace(
            text="This is a test.",
            entities=[]
        )

        assert result['anonymized_text'] == "This is a test."
        assert result['replacements'] == []
        assert result['replacement_info']['entities_processed'] == 0
        assert result['replacement_info']['replacements_applied'] == 0

class TestUtilityFunctions:
    """Test utility functions and classes."""

    def test_mock_extraction_pipeline(self):
        """Test basic pipeline without actual model."""
        # This test would require mocking the GLiNER model
        # For now, just test that we can import the classes
        try:
            from utils.merger import EntityMerger
            from utils.cache_manager import CacheManager
            from utils.entity_validator import EntityValidator

            merger = EntityMerger()
            cache_manager = CacheManager()
            validator = EntityValidator()

            assert merger is not None
            assert cache_manager is not None  
            assert validator is not None

        except ImportError:
            pytest.skip("Could not import utility classes")

if __name__ == "__main__":
    pytest.main([__file__])
