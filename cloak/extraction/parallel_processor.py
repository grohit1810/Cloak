"""Parallel Entity Processor — Batched Inference.

Uses GLiNER's native batched inference for processing chunked text,
replacing the previous ThreadPoolExecutor approach which was serialized
by the model's thread lock.
"""

import logging
from typing import Any

from .chunker import chunk_text
from .extractor import EntityExtractor

logger = logging.getLogger(__name__)


class ParallelEntityProcessor:
    """Processes large texts by chunking and running batched model inference."""

    def __init__(self, model):
        """Accept a GLiNERModel instance.

        Args:
            model: A GLiNERModel instance used for batched inference.
        """
        from ..models.gliner_model import GLiNERModel

        self.model: GLiNERModel = model
        self.extractor = EntityExtractor(model)
        logger.info("ParallelEntityProcessor initialized")

    def process_text(
        self,
        text: str,
        labels: list[str] | None = None,
        chunk_size: int = 250,
        use_parallel: bool | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Process text, using batched inference for large texts.

        Args:
            text: Text to process.
            labels: Entity labels to detect.
            chunk_size: Size of text chunks (in words).
            use_parallel: Force chunked processing (None=auto-detect).

        Returns:
            List of detected entities with positions relative to original text.
        """
        if not text or not text.strip():
            return []

        word_count = len(text.split())
        if use_parallel is None:
            use_parallel = word_count > chunk_size

        logger.info(
            "Processing text (%d chars, ~%d words) - Batched: %s",
            len(text),
            word_count,
            use_parallel,
        )

        if not use_parallel:
            return self.extractor.predict(text, labels)

        # Chunk text and run batched inference
        chunks = chunk_text(text, chunk_size)
        if not chunks:
            return []

        if len(chunks) == 1:
            # Single chunk — no need for batching
            return self.extractor.predict(chunks[0][0], labels)

        chunk_texts = [c[0] for c in chunks]
        offsets = [c[1] for c in chunks]

        logger.info("Running batched inference on %d chunks", len(chunks))
        all_chunk_results = self.model.batch_inference(
            chunk_texts,
            labels or [],
            threshold=0.5,
        )

        # Adjust entity offsets to original text positions
        entities: list[dict[str, Any]] = []
        for chunk_entities, offset in zip(all_chunk_results, offsets):
            for ent in chunk_entities:
                adjusted = ent.copy()
                adjusted["start"] += offset
                adjusted["end"] += offset
                entities.append(adjusted)

        entities.sort(key=lambda x: x["start"])
        logger.info("Batched inference found %d total entities", len(entities))
        return entities

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return self.extractor.get_model_info()
