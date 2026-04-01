"""Tests for batch_redact cross-document consistency."""

import pytest

from cloak.anonymization.redactor import EntityRedactor


class TestBatchRedact:
    def test_same_entity_gets_same_id_across_texts(self):
        redactor = EntityRedactor()
        texts = ["John works here", "John lives there"]
        all_entities = [
            [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}],
            [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.85}],
        ]
        results = redactor.batch_redact(texts, all_entities)
        placeholder_0 = results[0]["replacements"][0].placeholder
        placeholder_1 = results[1]["replacements"][0].placeholder
        assert placeholder_0 == placeholder_1

    def test_different_entities_get_different_ids(self):
        redactor = EntityRedactor()
        texts = ["John works here", "Alice lives there"]
        all_entities = [
            [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}],
            [{"label": "person", "text": "Alice", "start": 0, "end": 5, "score": 0.85}],
        ]
        results = redactor.batch_redact(texts, all_entities)
        placeholder_0 = results[0]["replacements"][0].placeholder
        placeholder_1 = results[1]["replacements"][0].placeholder
        assert placeholder_0 != placeholder_1

    def test_batch_does_not_leak_ids(self):
        redactor = EntityRedactor()
        texts = ["John works here"]
        all_entities = [
            [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}],
        ]
        redactor.batch_redact(texts, all_entities)
        # After batch, a fresh redact should get fresh IDs
        redactor.clear_history()
        result = redactor.redact(
            "Alice works here",
            [{"label": "person", "text": "Alice", "start": 0, "end": 5, "score": 0.9}],
        )
        # Alice should get ID 1
        assert result["replacements"][0].redaction_id == "1"

    def test_batch_mismatched_lengths_raises(self):
        redactor = EntityRedactor()
        with pytest.raises(ValueError):
            redactor.batch_redact(["text1", "text2"], [[]])
