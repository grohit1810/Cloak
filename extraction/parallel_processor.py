"""
Parallel Entity Processor - Enhanced

Implements ThreadPoolExecutor strategy for parallel entity extraction on large texts.
Enhanced with better error handling, progress tracking, and auto-parallel threshold.

Author: G Rohit (Enhanced from original)
Version: 1.0.0
"""

import concurrent.futures
import logging
from typing import List, Dict, Any, Optional

from .chunker import chunk_text
from .extractor import EntityExtractor

logger = logging.getLogger(__name__)

def extract_entities_for_chunk(
    chunk_text: str,
    offset: int,
    extractor: EntityExtractor,
    categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts entities from a single chunk using the provided extractor.
    Adjusts entity 'start'/'end' by 'offset' so they're correct
    relative to the original full text.

    Args:
        chunk_text: The text content of this chunk
        offset: Starting position of this chunk in the original text
        extractor: EntityExtractor instance to use for prediction
        categories: List of entity categories to detect

    Returns:
        List of entity dictionaries with adjusted start/end positions
    """
    try:
        # Run NER on this chunk (optionally respecting 'categories')
        raw_entities = extractor.predict(chunk_text, labels=categories)

        # Fix offsets to match the original text
        adjusted_entities = []
        for ent in raw_entities:
            try:
                adjusted_ent = ent.copy()
                adjusted_ent["start"] += offset
                adjusted_ent["end"] += offset
                adjusted_entities.append(adjusted_ent)
            except (KeyError, TypeError) as e:
                logger.warning(f"Error adjusting entity offset: {str(e)}")
                continue

        return adjusted_entities

    except Exception as e:
        logger.error(f"Error extracting entities for chunk: {str(e)}")
        return []

def extract_entities_in_parallel(
    full_text: str,
    extractor: EntityExtractor,
    chunk_size: int = 250,
    max_workers: int = 4,
    categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Splits the text into chunks, and extracts entities from each chunk in parallel
    using ThreadPoolExecutor. Enhanced with better error handling.

    Args:
        full_text: The complete text to process
        extractor: EntityExtractor instance to use for predictions
        chunk_size: Size of each text chunk (default: 250)
        max_workers: Maximum number of worker threads (default: 4)
        categories: List of entity categories to detect

    Returns:
        List of all entities found across all chunks, with corrected offsets
    """
    if not full_text or not full_text.strip():
        logger.warning("Empty text provided for parallel extraction")
        return []

    logger.info(f"Starting parallel extraction with {max_workers} workers, chunk_size={chunk_size}")

    try:
        # Step 1: Split text into chunks
        chunks = chunk_text(full_text, chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")

        if not chunks:
            logger.warning("No chunks created from text")
            return []

        # Step 2: Process chunks in parallel using ThreadPoolExecutor
        all_entities = []
        successful_chunks = 0
        failed_chunks = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            futures = []
            for chunk_str, offset in chunks:
                future = executor.submit(
                    extract_entities_for_chunk,
                    chunk_text=chunk_str,
                    offset=offset,
                    extractor=extractor,
                    categories=categories
                )
                futures.append(future)

            logger.info(f"Submitted {len(futures)} chunk processing tasks to thread pool")

            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    partial_entities = future.result()
                    all_entities.extend(partial_entities)
                    successful_chunks += 1
                    logger.debug(f"Completed chunk {successful_chunks}/{len(futures)}: found {len(partial_entities)} entities")
                except Exception as e:
                    failed_chunks += 1
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")

        logger.info(f"Parallel extraction complete. Processed {successful_chunks}/{len(chunks)} chunks successfully")
        if failed_chunks > 0:
            logger.warning(f"Failed to process {failed_chunks} chunks")

        # Step 3: Sort by start position for consistent output
        try:
            all_entities.sort(key=lambda x: x['start'])
        except (KeyError, TypeError) as e:
            logger.error(f"Error sorting entities: {str(e)}")

        logger.info(f"Found {len(all_entities)} total entities")
        return all_entities

    except Exception as e:
        logger.error(f"Error in parallel extraction: {str(e)}")
        return []

class ParallelEntityProcessor:
    """
    Enhanced high-level wrapper for parallel entity processing.

    Manages the extractor lifecycle with better error handling and auto-detection.
    """

    def __init__(self, model_path: str, onnx_model_file: str = "model.onnx"):
        """
        Initialize the parallel processor with a model.

        Args:
            model_path: Path to the GLINER ONNX model
            onnx_model_file: Name of the ONNX model file
        """
        try:
            self.model_path = model_path
            self.onnx_model_file = onnx_model_file
            self.extractor = EntityExtractor(model_path, onnx_model_file)
            logger.info(f"ParallelEntityProcessor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ParallelEntityProcessor: {str(e)}")
            raise

    def process_text(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        chunk_size: int = 250,
        max_workers: int = 4,
        use_parallel: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Process text with automatic parallel detection and enhanced error handling.

        Args:
            text: Text to process
            labels: Entity labels to detect
            chunk_size: Size of text chunks for parallel processing
            max_workers: Maximum number of worker threads
            use_parallel: Force parallel processing (None=auto-detect based on text length)

        Returns:
            List of detected entities
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for processing")
            return []

        try:
            # Auto-detect if parallel processing should be used
            if use_parallel is None:
                # Enhanced auto-parallel logic based on word count
                word_count = len(text.split())
                use_parallel = word_count > chunk_size

            logger.info(f"Processing text ({len(text)} chars, ~{len(text.split())} words) - Parallel: {use_parallel}")

            if use_parallel:
                return extract_entities_in_parallel(
                    text,
                    self.extractor,
                    chunk_size=chunk_size,
                    max_workers=max_workers,
                    categories=labels
                )
            else:
                # Single-pass processing for smaller texts
                return self.extractor.predict(text, labels)

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return []

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        try:
            return self.extractor.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
