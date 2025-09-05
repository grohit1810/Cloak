"""
Text Chunker - Enhanced

Implements advanced text chunking strategy for parallel processing of large texts.
Enhanced with better error handling and validation.

Author: G Rohit (Enhanced from original)
Version: 1.0.0
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 600) -> List[Tuple[str, int]]:
    """
    Splits text into slices of up to `chunk_size` words, preserving original spacing.

    Args:
        text: Input text to be chunked.
        chunk_size: Maximum number of words per chunk.

    Returns:
        List of (chunk_str, offset) tuples where:
        - chunk_str: The exact substring from the original text, including whitespace.
        - offset: The starting character index in the original text.
    """
    if not text or not text.strip():
        logger.warning("Empty or invalid text provided for chunking")
        return []

    if chunk_size <= 0:
        logger.warning(f"Invalid chunk_size: {chunk_size}. Using default 600.")
        chunk_size = 600

    try:
        # Find all word spans (start/end indices) using regex
        word_spans = [m.span() for m in re.finditer(r"\S+", text)]

        if not word_spans:
            logger.warning("No words found in text")
            return []

        chunks: List[Tuple[str, int]] = []

        for i in range(0, len(word_spans), chunk_size):
            # Determine spans for this chunk
            chunk_spans = word_spans[i : i + chunk_size]
            if not chunk_spans:
                break

            # Chunk start is the start of the first word
            start_char = chunk_spans[0][0]
            # Chunk end is the end of the last word
            end_char = chunk_spans[-1][1]

            # Extract exact substring including whitespace
            chunk_str = text[start_char : end_char]
            chunks.append((chunk_str, start_char))

        logger.info(f"Text chunked into {len(chunks)} chunks (chunk_size={chunk_size} words)")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        return []

def estimate_chunk_count(text: str, chunk_size: int = 600) -> int:
    """
    Estimate how many chunks will be created based on word count.

    Args:
        text: Input text
        chunk_size: Words per chunk

    Returns:
        Estimated number of chunks
    """
    try:
        if not text or chunk_size <= 0:
            return 0

        total_words = len(re.findall(r"\S+", text))
        return (total_words + chunk_size - 1) // chunk_size

    except Exception as e:
        logger.error(f"Error estimating chunk count: {str(e)}")
        return 0

def validate_chunks(chunks: List[Tuple[str, int]], original_text: str) -> bool:
    """
    Validate that the concatenation of chunk_strs matches the corresponding
    sections of original_text in sequence.

    Args:
        chunks: List of (chunk_str, offset) tuples
        original_text: Original text that was chunked

    Returns:
        True if chunks are valid, False otherwise
    """
    try:
        if not chunks or not original_text:
            return False

        for chunk_str, offset in chunks:
            # Check that the substring at offset matches chunk_str
            if offset < 0 or offset >= len(original_text):
                logger.error(f"Invalid chunk offset: {offset}")
                return False

            expected_end = offset + len(chunk_str)
            if expected_end > len(original_text):
                logger.error(f"Chunk extends beyond text length: {expected_end} > {len(original_text)}")
                return False

            if original_text[offset : expected_end] != chunk_str:
                logger.error(f"Mismatch at offset {offset}")
                return False

        logger.info("Chunk validation passed")
        return True

    except Exception as e:
        logger.error(f"Error validating chunks: {str(e)}")
        return False

def get_chunk_info(chunks: List[Tuple[str, int]]) -> dict:
    """
    Get statistics about the chunks.

    Args:
        chunks: List of (chunk_str, offset) tuples

    Returns:
        Dictionary with chunk statistics
    """
    try:
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "average_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_words": 0,
                "average_words_per_chunk": 0
            }

        sizes = [len(chunk) for chunk, _ in chunks]
        words = [len(re.findall(r"\S+", chunk)) for chunk, _ in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "average_chunk_size": sum(sizes) / len(chunks),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_words": sum(words),
            "average_words_per_chunk": sum(words) / len(chunks)
        }

    except Exception as e:
        logger.error(f"Error getting chunk info: {str(e)}")
        return {"error": str(e)}
