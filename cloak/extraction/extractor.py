"""Multi-pass entity extractor using GLiNER."""

import logging
from typing import Any

from ..constants import (
    DEFAULT_LABELS,
    DEFAULT_MAX_PASSES,
    FIRST_PASS_THRESHOLD,
    SUBSEQUENT_PASS_THRESHOLD,
)
from ..models.gliner_model import GLiNERModel

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Multi-pass entity extractor with masking between passes.

    Runs multiple passes over text. After each pass, found entities are
    masked with spaces so the model can discover entities it missed.
    Thresholds decrease across passes (0.5 -> 0.30).
    """

    def __init__(self, model: GLiNERModel):
        self.model = model
        logger.info("EntityExtractor initialized")

    def predict(
        self,
        text: str,
        labels: list[str] | None = None,
        max_passes: int = DEFAULT_MAX_PASSES,
    ) -> list[dict[str, Any]]:
        """Extract entities using multi-pass strategy with masking."""
        if not text or not text.strip():
            return []

        processed_labels = [label.lower() for label in (labels or DEFAULT_LABELS)]

        all_entities: list[dict[str, Any]] = []
        processed_spans: set[tuple[int, int]] = set()
        mutable_text_list = list(text)

        for pass_num in range(max_passes):
            current_text = "".join(mutable_text_list)
            threshold = FIRST_PASS_THRESHOLD if pass_num == 0 else SUBSEQUENT_PASS_THRESHOLD

            try:
                newly_found = self.model.predict_entities(
                    current_text,
                    processed_labels,
                    threshold=threshold,
                )
            except Exception as e:
                logger.error("Error in pass %d: %s", pass_num + 1, e)
                break

            if not newly_found:
                logger.info("Pass %d: no entities found, stopping", pass_num + 1)
                break

            unique_new = []
            for ent in newly_found:
                span = (ent["start"], ent["end"])
                if span not in processed_spans:
                    unique_new.append(ent)
                    processed_spans.add(span)

            if not unique_new:
                break

            logger.info(
                "Pass %d: %d new entities (threshold=%.2f)",
                pass_num + 1,
                len(unique_new),
                threshold,
            )
            all_entities.extend(unique_new)

            for entity in unique_new:
                for i in range(entity["start"], min(entity["end"], len(mutable_text_list))):
                    mutable_text_list[i] = " "

        all_entities.sort(key=lambda x: x["start"])
        return all_entities

    def get_model_info(self) -> dict[str, Any]:
        return self.model.get_model_info()
