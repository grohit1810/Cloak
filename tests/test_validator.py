"""Tests for EntityValidator."""

from cloak.utils.entity_validator import EntityValidator


class TestEntityValidator:
    def test_filters_low_confidence(self):
        validator = EntityValidator(min_confidence=0.5)
        entities = [
            {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.3},
            {"label": "person", "text": "Alice", "start": 10, "end": 15, "score": 0.8},
        ]
        result = validator.validate_entities(entities, "John xxx  Alice xxx")
        assert len(result) == 1
        assert result[0]["text"] == "Alice"

    def test_filters_invalid_position(self):
        validator = EntityValidator()
        entities = [
            {"label": "person", "text": "John", "start": -1, "end": 4, "score": 0.9},
            {"label": "person", "text": "Alice", "start": 100, "end": 105, "score": 0.9},
        ]
        result = validator.validate_entities(entities, "short text")
        assert len(result) == 0

    def test_does_not_inject_metadata(self):
        validator = EntityValidator()
        entities = [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}]
        result = validator.validate_entities(entities, "John works here")
        assert "validated" not in result[0]
        assert "validator_version" not in result[0]

    def test_configurable_max_entity_length(self):
        validator = EntityValidator(max_entity_length=500)
        long_text = "A" * 300
        entities = [{"label": "org", "text": long_text, "start": 0, "end": 300, "score": 0.9}]
        result = validator.validate_entities(entities, long_text)
        assert len(result) == 1

    def test_resolve_overlaps_highest_confidence(self):
        validator = EntityValidator()
        entities = [
            {"label": "person", "text": "John Smith", "start": 0, "end": 10, "score": 0.9},
            {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.7},
        ]
        result = validator.resolve_overlaps(entities, "highest_confidence")
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_resolve_overlaps_longest(self):
        validator = EntityValidator()
        entities = [
            {"label": "person", "text": "John Smith", "start": 0, "end": 10, "score": 0.7},
            {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9},
        ]
        result = validator.resolve_overlaps(entities, "longest")
        assert len(result) == 1
        assert result[0]["text"] == "John Smith"

    def test_deep_copy_does_not_mutate_input(self):
        validator = EntityValidator()
        original = {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}
        result = validator.validate_entities([original], "John works here")
        # Mutate the output — should not affect input
        result[0]["label"] = "MUTATED"
        assert original["label"] == "person"
