"""
Entity Validator - Enhanced

Provides robust validation and filtering for detected entities.
Enhanced with improved error handling and additional validation features.

Author: G Rohit (Enhanced from original)  
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

class EntityValidator:
    """
    Enhanced entity validator for position consistency, text matching, and confidence thresholds.

    Implements advanced validation features with comprehensive error handling.
    """

    def __init__(self, min_confidence: float = 0.3, strict_validation: bool = True):
        """
        Initialize the entity validator.

        Args:
            min_confidence: Minimum confidence threshold for entities (default: 0.3)
            strict_validation: Whether to apply strict position/text validation
        """
        self.min_confidence = min_confidence
        self.strict_validation = strict_validation
        self.validation_stats = {
            "total_entities": 0,
            "confidence_filtered": 0,
            "position_invalid": 0,
            "text_mismatch": 0,
            "valid_entities": 0,
            "validation_errors": 0
        }

        logger.info(f"EntityValidator initialized: min_confidence={min_confidence}, strict={strict_validation}")

    def validate_entities(
        self,
        entities: List[Dict[str, Any]],
        text: str,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
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

        logger.info(f"Validating {len(entities)} entities (min_confidence={confidence_threshold})")

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
                logger.error(f"Error validating entity {entity}: {str(e)}")
                self.validation_stats["validation_errors"] += 1
                continue

        self.validation_stats["valid_entities"] = len(validated)

        logger.info(f"Validation complete: {len(validated)}/{len(entities)} entities passed")
        if self.validation_stats["confidence_filtered"] > 0:
            logger.info(f" - Filtered by confidence: {self.validation_stats['confidence_filtered']}")
        if self.validation_stats["position_invalid"] > 0:
            logger.info(f" - Invalid positions: {self.validation_stats['position_invalid']}")
        if self.validation_stats["text_mismatch"] > 0:
            logger.info(f" - Text mismatches: {self.validation_stats['text_mismatch']}")
        if self.validation_stats["validation_errors"] > 0:
            logger.warning(f" - Validation errors: {self.validation_stats['validation_errors']}")

        return validated

    def _validate_confidence(self, entity: Dict[str, Any], min_confidence: float) -> bool:
        """Validate entity confidence score."""
        try:
            score = entity.get('score', 0.0)
            return isinstance(score, (int, float)) and score >= min_confidence
        except Exception as e:
            logger.debug(f"Confidence validation error: {str(e)}")
            return False

    def _validate_position(self, entity: Dict[str, Any], text: str) -> bool:
        """Validate entity position boundaries."""
        try:
            start = entity.get('start', -1)
            end = entity.get('end', -1)
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
            if entity_length > 200:  # Entities longer than 200 chars are suspicious
                return False

            return True

        except Exception as e:
            logger.debug(f"Position validation error: {str(e)}")
            return False

    def _validate_text_consistency(self, entity: Dict[str, Any], text: str) -> bool:
        """Validate that entity text matches the text at specified positions."""
        try:
            start = entity.get('start')
            end = entity.get('end')
            entity_text = entity.get('text', '')

            if start is None or end is None:
                return False

            actual_text = text[start:end]

            # Normalize whitespace for comparison
            normalized_entity = re.sub(r'\s+', ' ', entity_text.strip())
            normalized_actual = re.sub(r'\s+', ' ', actual_text.strip())

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
            logger.debug(f"Text consistency validation error: {str(e)}")
            return False

    def _clean_entity(self, entity: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Clean and normalize entity data."""
        try:
            cleaned = entity.copy()

            # Ensure text matches actual position
            if self.strict_validation:
                start = cleaned.get('start')
                end = cleaned.get('end')
                if start is not None and end is not None:
                    actual_text = text[start:end]
                    cleaned['text'] = actual_text.strip()

            # Normalize label
            if 'label' in cleaned:
                cleaned['label'] = cleaned['label'].lower().strip()

            # Ensure score is a float
            if 'score' in cleaned:
                cleaned['score'] = float(cleaned['score'])

            # Add validation metadata
            cleaned['validated'] = True
            cleaned['validator_version'] = '1.0.0'

            return cleaned

        except Exception as e:
            logger.error(f"Entity cleaning error: {str(e)}")
            return entity

    def resolve_overlaps(
        self,
        entities: List[Dict[str, Any]],
        strategy: str = "highest_confidence"
    ) -> List[Dict[str, Any]]:
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
                return entities.copy()

            logger.info(f"Resolving {len(overlaps)} overlapping entity pairs using '{strategy}' strategy")

            # Mark entities to remove
            to_remove = set()

            for idx1, idx2 in overlaps:
                if idx1 in to_remove or idx2 in to_remove:
                    continue

                entity1, entity2 = entities[idx1], entities[idx2]

                if strategy == "highest_confidence":
                    score1 = entity1.get('score', 0.0)
                    score2 = entity2.get('score', 0.0)
                    to_remove.add(idx1 if score2 > score1 else idx2)
                elif strategy == "longest":
                    len1 = entity1.get('end', 0) - entity1.get('start', 0)
                    len2 = entity2.get('end', 0) - entity2.get('start', 0)
                    to_remove.add(idx1 if len2 > len1 else idx2)
                elif strategy == "first":
                    to_remove.add(idx2)  # Keep the first one
                else:
                    logger.warning(f"Unknown overlap resolution strategy: {strategy}")
                    to_remove.add(idx2)  # Default to first

            # Return entities not marked for removal
            resolved = [entity for i, entity in enumerate(entities) if i not in to_remove]

            logger.info(f"Overlap resolution: {len(entities)} -> {len(resolved)} entities")
            return resolved

        except Exception as e:
            logger.error(f"Error in overlap resolution: {str(e)}")
            return entities

    def _detect_overlaps(self, entities: List[Dict[str, Any]]) -> List[tuple]:
        """Detect overlapping entities."""
        overlaps = []

        try:
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    if self._entities_overlap(entity1, entity2):
                        overlaps.append((i, j))
        except Exception as e:
            logger.error(f"Error detecting overlaps: {str(e)}")

        return overlaps

    def _entities_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities have overlapping positions."""
        try:
            start1, end1 = entity1.get('start', 0), entity1.get('end', 0)
            start2, end2 = entity2.get('start', 0), entity2.get('end', 0)
            return not (end1 <= start2 or end2 <= start1)
        except Exception as e:
            logger.debug(f"Error checking entity overlap: {str(e)}")
            return False

    def get_validation_stats(self) -> Dict[str, Any]:
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
            logger.error(f"Error computing validation stats: {str(e)}")
            return self.validation_stats
