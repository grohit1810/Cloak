"""
Entity Validator - Enhanced

Provides robust validation and filtering for detected entities.
Enhanced with improved error handling and additional validation features.

Author: G Rohit (Enhanced from original)
Version: 1.0.0
"""

import logging
import re
from typing import Any

from ..constants import MAX_ENTITY_LENGTH

logger = logging.getLogger(__name__)


class EntityValidator:
    """
    Enhanced entity validator for position consistency, text matching, and confidence thresholds.

    Implements advanced validation features with comprehensive error handling.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        strict_validation: bool = True,
        max_entity_length: int = MAX_ENTITY_LENGTH,
    ):
        """
        Initialize the entity validator.

        Args:
            min_confidence: Minimum confidence threshold for entities (default: 0.3)
            strict_validation: Whether to apply strict position/text validation
            max_entity_length: Maximum allowed entity length in characters (default: 200)
        """
        self.min_confidence = min_confidence
        self.strict_validation = strict_validation
        self.max_entity_length = max_entity_length
        self.validation_stats = {
            "total_entities": 0,
            "confidence_filtered": 0,
            "position_invalid": 0,
            "text_mismatch": 0,
            "valid_entities": 0,
            "validation_errors": 0,
        }

        logger.info(
            "EntityValidator initialized: min_confidence=%s, strict=%s",
            min_confidence,
            strict_validation,
        )

    def validate_entities(
        self, entities: list[dict[str, Any]], text: str, min_confidence: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Comprehensive entity validation pipeline.

        Args:
            entities: List of detected entities
            text: Original text
            min_confidence: Override default minimum confidence

        Returns:
            List of validated entities
        """
        if not entities:
            return []

        confidence_threshold = min_confidence if min_confidence is not None else self.min_confidence

        logger.info(
            "Validating %d entities (min_confidence=%s)", len(entities), confidence_threshold
        )

        # Reset stats
        self.validation_stats["total_entities"] = len(entities)
        self.validation_stats["confidence_filtered"] = 0
        self.validation_stats["position_invalid"] = 0
        self.validation_stats["text_mismatch"] = 0
        self.validation_stats["validation_errors"] = 0

        validated = []

        for entity in entities:
            try:
                # Step 1: Confidence filtering
                if not self._validate_confidence(entity, confidence_threshold):
                    self.validation_stats["confidence_filtered"] += 1
                    continue

                # Step 2: Position validation
                if self.strict_validation and not self._validate_position(entity, text):
                    self.validation_stats["position_invalid"] += 1
                    continue

                # Step 3: Text consistency validation
                if self.strict_validation and not self._validate_text_consistency(entity, text):
                    self.validation_stats["text_mismatch"] += 1
                    continue

                # Step 4: Additional cleanup and normalization
                cleaned_entity = self._clean_entity(entity, text)
                validated.append(cleaned_entity)

            except Exception as e:
                logger.error("Error validating entity %s: %s", entity, str(e))
                self.validation_stats["validation_errors"] += 1
                continue

        self.validation_stats["valid_entities"] = len(validated)

        logger.info("Validation complete: %d/%d entities passed", len(validated), len(entities))
        if self.validation_stats["confidence_filtered"] > 0:
            logger.info(
                " - Filtered by confidence: %d", self.validation_stats["confidence_filtered"]
            )
        if self.validation_stats["position_invalid"] > 0:
            logger.info(" - Invalid positions: %d", self.validation_stats["position_invalid"])
        if self.validation_stats["text_mismatch"] > 0:
            logger.info(" - Text mismatches: %d", self.validation_stats["text_mismatch"])
        if self.validation_stats["validation_errors"] > 0:
            logger.warning(" - Validation errors: %d", self.validation_stats["validation_errors"])

        return validated

    def _validate_confidence(self, entity: dict[str, Any], min_confidence: float) -> bool:
        """Validate entity confidence score."""
        try:
            score = entity.get("score", 0.0)
            return isinstance(score, (int, float)) and score >= min_confidence
        except Exception as e:
            logger.debug("Confidence validation error: %s", e)
            return False

    def _validate_position(self, entity: dict[str, Any], text: str) -> bool:
        """Validate entity position boundaries."""
        try:
            start = entity.get("start", -1)
            end = entity.get("end", -1)
            text_length = len(text)

            # Check basic position validity
            if not isinstance(start, int) or not isinstance(end, int):
                return False

            if start < 0 or end < 0:
                return False

            if start >= text_length or end > text_length:
                return False

            if start >= end:
                return False

            # Check for reasonable entity length (not too long)
            entity_length = end - start
            if entity_length > self.max_entity_length:
                return False

            return True

        except Exception as e:
            logger.debug("Position validation error: %s", e)
            return False

    def _validate_text_consistency(self, entity: dict[str, Any], text: str) -> bool:
        """Validate that entity text matches the text at specified positions."""
        try:
            start = entity.get("start")
            end = entity.get("end")
            entity_text = entity.get("text", "")

            if start is None or end is None:
                return False

            actual_text = text[start:end]

            # Normalize whitespace for comparison
            normalized_entity = re.sub(r"\s+", " ", entity_text.strip())
            normalized_actual = re.sub(r"\s+", " ", actual_text.strip())

            # Allow for minor whitespace differences
            if normalized_entity == normalized_actual:
                return True

            # Allow for case differences in some cases
            if normalized_entity.lower() == normalized_actual.lower():
                return True

            # Check if entity text is a clean subset (handles tokenization differences)
            if normalized_entity in normalized_actual or normalized_actual in normalized_entity:
                return True

            return False

        except (IndexError, TypeError) as e:
            logger.debug("Text consistency validation error: %s", e)
            return False

    def _clean_entity(self, entity: dict[str, Any], text: str) -> dict[str, Any]:
        """Clean and normalize entity data."""
        try:
            cleaned = entity.copy()

            # Normalize label
            if "label" in cleaned:
                cleaned["label"] = cleaned["label"].lower().strip()

            # Ensure score is a float
            if "score" in cleaned:
                cleaned["score"] = float(cleaned["score"])

            return cleaned

        except Exception as e:
            logger.error("Entity cleaning error: %s", e)
            return entity.copy()

    def resolve_overlaps(
        self, entities: list[dict[str, Any]], strategy: str = "highest_confidence"
    ) -> list[dict[str, Any]]:
        """
        Resolve overlapping entities using specified strategy.

        Args:
            entities: List of entities
            strategy: "highest_confidence", "longest", "first"

        Returns:
            List of entities with overlaps resolved
        """
        if not entities:
            return []

        try:
            overlaps = self._detect_overlaps(entities)
            if not overlaps:
                return [e.copy() for e in entities]

            logger.info(
                "Resolving %d overlapping entity pairs using '%s' strategy", len(overlaps), strategy
            )

            # Mark entities to remove
            to_remove = set()

            for idx1, idx2 in overlaps:
                if idx1 in to_remove or idx2 in to_remove:
                    continue

                entity1, entity2 = entities[idx1], entities[idx2]

                if strategy == "highest_confidence":
                    score1 = entity1.get("score", 0.0)
                    score2 = entity2.get("score", 0.0)
                    to_remove.add(idx1 if score2 > score1 else idx2)
                elif strategy == "longest":
                    len1 = entity1.get("end", 0) - entity1.get("start", 0)
                    len2 = entity2.get("end", 0) - entity2.get("start", 0)
                    to_remove.add(idx1 if len2 > len1 else idx2)
                elif strategy == "first":
                    to_remove.add(idx2)  # Keep the first one
                else:
                    logger.warning("Unknown overlap resolution strategy: %s", strategy)
                    to_remove.add(idx2)  # Default to first

            # Return entities not marked for removal
            resolved = [entity for i, entity in enumerate(entities) if i not in to_remove]

            logger.info("Overlap resolution: %d -> %d entities", len(entities), len(resolved))
            return resolved

        except Exception as e:
            logger.error("Error in overlap resolution: %s", str(e))
            return entities

    def _detect_overlaps(self, entities: list[dict[str, Any]]) -> list[tuple]:
        """Detect overlapping entities using sweep-line algorithm. O(n log n)."""
        if len(entities) <= 1:
            return []
        try:
            sorted_indices = sorted(range(len(entities)), key=lambda i: entities[i].get("start", 0))
            overlaps = []
            for k in range(len(sorted_indices) - 1):
                i = sorted_indices[k]
                end1 = entities[i].get("end", 0)
                for m in range(k + 1, len(sorted_indices)):
                    j = sorted_indices[m]
                    if entities[j].get("start", 0) >= end1:
                        break
                    overlaps.append((i, j))
            return overlaps
        except Exception as e:
            logger.error("Error detecting overlaps: %s", e)
            return []

    def _entities_overlap(self, entity1: dict[str, Any], entity2: dict[str, Any]) -> bool:
        """Check if two entities have overlapping positions."""
        try:
            start1, end1 = entity1.get("start", 0), entity1.get("end", 0)
            start2, end2 = entity2.get("start", 0), entity2.get("end", 0)
            return not (end1 <= start2 or end2 <= start1)
        except Exception as e:
            logger.debug("Error checking entity overlap: %s", e)
            return False

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        try:
            total = self.validation_stats["total_entities"]
            if total == 0:
                return self.validation_stats

            stats = self.validation_stats.copy()
            stats["confidence_filter_rate"] = stats["confidence_filtered"] / total
            stats["position_invalid_rate"] = stats["position_invalid"] / total
            stats["text_mismatch_rate"] = stats["text_mismatch"] / total
            stats["validation_success_rate"] = stats["valid_entities"] / total
            stats["error_rate"] = stats["validation_errors"] / total

            return stats

        except Exception as e:
            logger.error("Error computing validation stats: %s", str(e))
            return self.validation_stats
