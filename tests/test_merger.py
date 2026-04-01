"""Tests for EntityMerger."""

from cloak.utils.merger import EntityMerger


class TestEntityMerger:
    def test_merge_adjacent_same_label(self):
        merger = EntityMerger()
        text = "New York City is great"
        entities = [
            {"label": "location", "text": "New", "start": 0, "end": 3, "score": 0.8},
            {"label": "location", "text": "York", "start": 4, "end": 8, "score": 0.7},
        ]
        result = merger.merge(entities, text)
        assert len(result) == 1
        assert result[0]["text"] == "New York"

    def test_no_merge_different_labels(self):
        merger = EntityMerger()
        text = "John Google"
        entities = [
            {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.8},
            {"label": "company", "text": "Google", "start": 5, "end": 11, "score": 0.7},
        ]
        result = merger.merge(entities, text)
        assert len(result) == 2

    def test_merge_stats_are_correct(self):
        merger = EntityMerger()
        text = "New York City is great"
        entities = [
            {"label": "location", "text": "New", "start": 0, "end": 3, "score": 0.8},
            {"label": "location", "text": "York", "start": 4, "end": 8, "score": 0.7},
        ]
        merger.merge(entities, text)
        assert merger.merge_stats["merges_by_label"]["location"] == 1

    def test_does_not_mutate_input(self):
        merger = EntityMerger()
        text = "John works"
        original = {"label": "person", "text": "John", "start": 0, "end": 4, "score": 0.8}
        entities = [original]
        merger.merge(entities, text)
        assert "count" not in original

    def test_empty_entities(self):
        merger = EntityMerger()
        result = merger.merge([], "some text")
        assert result == []
