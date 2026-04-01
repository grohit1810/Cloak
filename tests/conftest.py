"""Shared test fixtures for Cloak tests."""

import pytest

from cloak.anonymization.redactor import EntityRedactor
from cloak.anonymization.replacer import EntityReplacer
from cloak.utils.entity_validator import EntityValidator
from cloak.utils.merger import EntityMerger


@pytest.fixture
def sample_entities():
    return [
        {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9},
        {"label": "location", "text": "Paris", "start": 14, "end": 19, "score": 0.85},
    ]


@pytest.fixture
def sample_text():
    return "John lives in Paris and works at Google."


@pytest.fixture
def redactor():
    return EntityRedactor()


@pytest.fixture
def replacer():
    return EntityReplacer()


@pytest.fixture
def validator():
    return EntityValidator()


@pytest.fixture
def merger():
    return EntityMerger()
