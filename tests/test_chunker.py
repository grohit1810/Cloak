"""Tests for the text chunker."""

from cloak.extraction.chunker import chunk_text, estimate_chunk_count, validate_chunks


class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, chunk_size=50)
        assert len(chunks) == 2

    def test_chunk_offsets_are_correct(self):
        text = "Hello world this is a test"
        chunks = chunk_text(text, chunk_size=3)
        for chunk_str, offset in chunks:
            assert text[offset : offset + len(chunk_str)] == chunk_str

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_invalid_chunk_size(self):
        chunks = chunk_text("hello world", chunk_size=0)
        assert len(chunks) > 0  # Should use default

    def test_single_chunk_for_small_text(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0][0] == text
        assert chunks[0][1] == 0

    def test_preserves_whitespace_in_chunks(self):
        text = "Hello   world   foo   bar"
        chunks = chunk_text(text, chunk_size=2)
        for chunk_str, offset in chunks:
            assert text[offset : offset + len(chunk_str)] == chunk_str


class TestEstimateChunkCount:
    def test_estimate(self):
        text = " ".join(["word"] * 100)
        assert estimate_chunk_count(text, 50) == 2

    def test_empty_text(self):
        assert estimate_chunk_count("", 50) == 0

    def test_single_chunk(self):
        assert estimate_chunk_count("hello world", 100) == 1


class TestValidateChunks:
    def test_valid_chunks(self):
        text = "Hello world this is a test"
        chunks = chunk_text(text, chunk_size=3)
        assert validate_chunks(chunks, text) is True

    def test_empty_chunks(self):
        assert validate_chunks([], "some text") is False

    def test_invalid_chunks(self):
        assert validate_chunks([("wrong", 0)], "different") is False
