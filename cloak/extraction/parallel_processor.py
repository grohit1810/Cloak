"""Parallel Entity Processor.

Implements ThreadPoolExecutor strategy for parallel entity extraction on large texts.
"""

import concurrent.futures
import logging
from typing import Any

from .chunker import chunk_text
from .extractor import EntityExtractor

logger = logging.getLogger(__name__)


def extract_entities_for_chunk(
    chunk_str: str,
    offset: int,
    extractor: EntityExtractor,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract entities from a single chunk, adjusting offsets to the original text.

    Args:
        chunk_str: The text content of this chunk.
        offset: Starting position of this chunk in the original text.
        extractor: EntityExtractor instance to use for prediction.
        categories: List of entity categories to detect.

    Returns:
        List of entity dicts with adjusted start/end positions.
    """
    try:
        raw_entities = extractor.predict(chunk_str, labels=categories)

        adjusted_entities = []
        for ent in raw_entities:
            try:
                adjusted_ent = ent.copy()
                adjusted_ent["start"] += offset
                adjusted_ent["end"] += offset
                adjusted_entities.append(adjusted_ent)
            except (KeyError, TypeError) as e:
                logger.warning("Error adjusting entity offset: %s", e)
                continue

        return adjusted_entities

    except Exception as e:
        logger.error("Error extracting entities for chunk: %s", e)
        return []


def extract_entities_in_parallel(
    full_text: str,
    extractor: EntityExtractor,
    chunk_size: int = 250,
    max_workers: int = 4,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Split text into chunks and extract entities from each in parallel.

    Args:
        full_text: The complete text to process.
        extractor: EntityExtractor instance to use for predictions.
        chunk_size: Size of each text chunk (default: 250).
        max_workers: Maximum number of worker threads (default: 4).
        categories: List of entity categories to detect.

    Returns:
        List of all entities found across all chunks, with corrected offsets.
    """
    if not full_text or not full_text.strip():
        logger.warning("Empty text provided for parallel extraction")
        return []

    logger.info(
        "Starting parallel extraction with %d workers, chunk_size=%d",
        max_workers,
        chunk_size,
    )

    try:
        chunks = chunk_text(full_text, chunk_size)
        logger.info("Split text into %d chunks", len(chunks))

        if not chunks:
            logger.warning("No chunks created from text")
            return []

        all_entities: list[dict[str, Any]] = []
        successful_chunks = 0
        failed_chunks = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chunk_str, offset in chunks:
                future = executor.submit(
                    extract_entities_for_chunk,
                    chunk_str=chunk_str,
                    offset=offset,
                    extractor=extractor,
                    categories=categories,
                )
                futures.append(future)

            logger.info("Submitted %d chunk processing tasks to thread pool", len(futures))

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    partial_entities = future.result()
                    all_entities.extend(partial_entities)
                    successful_chunks += 1
                except Exception as e:
                    failed_chunks += 1
                    logger.error("Error processing chunk %d: %s", i + 1, e)

        logger.info(
            "Parallel extraction complete. Processed %d/%d chunks successfully",
            successful_chunks,
            len(chunks),
        )
        if failed_chunks > 0:
            logger.warning("Failed to process %d chunks", failed_chunks)

        all_entities.sort(key=lambda x: x["start"])
        logger.info("Found %d total entities", len(all_entities))
        return all_entities

    except Exception as e:
        logger.error("Error in parallel extraction: %s", e)
        return []


class ParallelEntityProcessor:
    """High-level wrapper for parallel entity processing."""

    def __init__(self, model: object):
        """Accept a GLiNERModel instance.

        Args:
            model: A GLiNERModel instance used to create the internal EntityExtractor.
        """
        try:
            self.model_path = model.model_path  # type: ignore[attr-defined]
            self.extractor = EntityExtractor(model)  # type: ignore[arg-type]
            logger.info("ParallelEntityProcessor initialized")
        except Exception as e:
            logger.error("Failed to initialize ParallelEntityProcessor: %s", e)
            raise

    def process_text(
        self,
        text: str,
        labels: list[str] | None = None,
        chunk_size: int = 250,
        max_workers: int = 4,
        use_parallel: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Process text with automatic parallel detection.

        Args:
            text: Text to process.
            labels: Entity labels to detect.
            chunk_size: Size of text chunks for parallel processing.
            max_workers: Maximum number of worker threads.
            use_parallel: Force parallel processing (None=auto-detect based on text length).

        Returns:
            List of detected entities.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for processing")
            return []

        try:
            if use_parallel is None:
                word_count = len(text.split())
                use_parallel = word_count > chunk_size

            logger.info(
                "Processing text (%d chars, ~%d words) - Parallel: %s",
                len(text),
                len(text.split()),
                use_parallel,
            )

            if use_parallel:
                return extract_entities_in_parallel(
                    text,
                    self.extractor,
                    chunk_size=chunk_size,
                    max_workers=max_workers,
                    categories=labels,
                )
            else:
                return self.extractor.predict(text, labels)

        except Exception as e:
            logger.error("Error processing text: %s", e)
            return []

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        try:
            return self.extractor.get_model_info()
        except Exception as e:
            logger.error("Error getting model info: %s", e)
            return {"error": str(e)}
