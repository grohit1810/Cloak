"""Tests for EntityRedactor."""

import json

from cloak.anonymization.redactor import EntityRedactor, RedactionDetail


class TestEntityRedactor:
    def test_redact_single_entity(self):
        redactor = EntityRedactor()
        entities = [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}]
        result = redactor.redact("John works here", entities)
        assert "John" not in result["anonymized_text"]
        assert "PERSON" in result["anonymized_text"]
        assert result["redaction_info"]["redactions_applied"] == 1

    def test_redact_consistent_ids(self):
        redactor = EntityRedactor()
        entities = [
            {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9},
            {"label": "person", "text": "John", "start": 18, "end": 22, "score": 0.85},
        ]
        result = redactor.redact("John works here.  John is nice.", entities)
        placeholders = [d.placeholder for d in result["replacements"]]
        assert placeholders[0] == placeholders[1]

    def test_redact_empty_entities(self):
        redactor = EntityRedactor()
        result = redactor.redact("Hello world", [])
        assert result["anonymized_text"] == "Hello world"

    def test_redact_without_re_id_map(self):
        redactor = EntityRedactor()
        entities = [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}]
        result = redactor.redact("John works here", entities, include_re_id_map=False)
        assert "re_identification_map" not in result

    def test_redact_with_re_id_map(self):
        redactor = EntityRedactor()
        entities = [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}]
        result = redactor.redact("John works here", entities, include_re_id_map=True)
        assert "re_identification_map" in result


class TestRedactionDetail:
    def test_to_dict_is_json_serializable(self):
        detail = RedactionDetail(
            label="person",
            original="Alice",
            placeholder="#1_PERSON_REDACTED",
            start=0,
            end=5,
            score=0.95,
            redaction_id="1",
        )
        d = detail.to_dict()
        result = json.dumps(d)
        assert '"person"' in result


class TestReplacerSeed:
    def test_seed_produces_reproducible_output(self):
        from cloak.anonymization.replacer import EntityReplacer

        entities = [{"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.9}]
        r1 = EntityReplacer(seed=42).replace("John works here", entities)
        r2 = EntityReplacer(seed=42).replace("John works here", entities)
        assert r1["anonymized_text"] == r2["anonymized_text"]


class TestReplacementDetail:
    def test_to_dict_is_json_serializable(self):
        from cloak.anonymization.replacer import ReplacementDetail

        detail = ReplacementDetail(
            label="person",
            original="John",
            replacement="Alice",
            start=0,
            end=4,
            score=0.9,
            strategy_used="faker",
        )
        d = detail.to_dict()
        json.dumps(d)  # Should not raise
