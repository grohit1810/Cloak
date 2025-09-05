"""
Entity Merger - Enhanced

Implements advanced merging strategy for combining adjacent entities.
Enhanced with better logging and statistics tracking.

Author: G Rohit (Enhanced from original)
Version: 1.0.0
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EntityMerger:
    """
    Enhanced entity merger with advanced merging logic and statistics.

    Merges entities based on their labels and positions in the text.
    Designed to handle entities that are close together or have the same label,
    merging them into a single entity when appropriate.
    """

    def __init__(self, max_gap: int = 1, enable_logging: bool = True):
        """
        Initialize the entity merger.

        Args:
            max_gap: Maximum character gap to allow for merging (default: 1)
            enable_logging: Whether to enable detailed logging (default: True)
        """
        self.max_gap = max_gap
        self.enable_logging = enable_logging
        self.merge_stats = {
            'total_processed': 0,
            'total_merged': 0,
            'merges_by_label': {}
        }

    def merge(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Merge adjacent entities with the same label.

        Args:
            entities: List of entity dictionaries to merge
            text: Original text (used to extract merged text content)

        Returns:
            List of merged entities
        """
        if not entities:
            return []

        original_count = len(entities)
        self.merge_stats['total_processed'] += original_count

        # Sort entities by start position to ensure proper merging
        entities = sorted(entities, key=lambda x: x['start'])

        merged = []
        current = entities[0].copy()
        current["count"] = 1

        for next_entity in entities[1:]:
            # Check if entities can be merged:
            # 1. Same label
            # 2. Adjacent (next starts where current ends) OR
            # 3. Small gap (next starts within max_gap after current ends)
            if (next_entity["label"] == current["label"] and
                (next_entity["start"] == current["end"] or
                 next_entity["start"] <= current["end"] + self.max_gap)):

                # Merge: text from current start to next entity's end
                merged_text = text[current["start"] : next_entity["end"]].strip()
                current["text"] = merged_text
                current["end"] = next_entity["end"]

                # Update score using weighted average
                current["score"] = (
                    current["score"] * current["count"] + next_entity["score"]
                ) / (current["count"] + 1)
                current["count"] += 1

                if self.enable_logging:
                    logger.debug(f"Merged entities: '{current['text']}' (count: {current['count']})")

            else:
                # Cannot merge, add current to results and start new current
                current.pop("count", None)  # Remove internal count field
                merged.append(current)
                current = next_entity.copy()
                current["count"] = 1

        # Add the last entity
        current.pop("count", None)  # Remove internal count field
        merged.append(current)

        # Update statistics
        merges_applied = original_count - len(merged)
        self.merge_stats['total_merged'] += merges_applied

        # Track merges by label
        for entity in entities:
            label = entity['label']
            if label not in self.merge_stats['merges_by_label']:
                self.merge_stats['merges_by_label'][label] = 0

        if self.enable_logging:
            logger.info(f"Entity merging complete: {original_count} -> {len(merged)} entities")

        return merged

    def can_merge(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """
        Check if two entities can be merged.

        Args:
            entity1: First entity
            entity2: Second entity (should come after entity1)

        Returns:
            True if entities can be merged, False otherwise
        """
        return (
            entity1["label"] == entity2["label"] and
            (entity2["start"] == entity1["end"] or
             entity2["start"] <= entity1["end"] + self.max_gap)
        )

    def get_merge_statistics(self, original_entities: List[Dict[str, Any]],
                           merged_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the merging process.

        Args:
            original_entities: Original entity list before merging
            merged_entities: Entity list after merging

        Returns:
            Dictionary with merge statistics
        """
        merge_reduction = len(original_entities) - len(merged_entities)
        reduction_percentage = (merge_reduction / len(original_entities) * 100) if original_entities else 0

        # Count entities by label in both lists
        original_by_label = {}
        merged_by_label = {}

        for entity in original_entities:
            label = entity["label"]
            original_by_label[label] = original_by_label.get(label, 0) + 1

        for entity in merged_entities:
            label = entity["label"]
            merged_by_label[label] = merged_by_label.get(label, 0) + 1

        return {
            "original_count": len(original_entities),
            "merged_count": len(merged_entities),
            "entities_merged": merge_reduction,
            "reduction_percentage": reduction_percentage,
            "original_by_label": original_by_label,
            "merged_by_label": merged_by_label,
            "global_stats": self.merge_stats
        }

    def reset_statistics(self):
        """Reset merge statistics."""
        self.merge_stats = {
            'total_processed': 0,
            'total_merged': 0,
            'merges_by_label': {}
        }
        if self.enable_logging:
            logger.info("Merge statistics reset")
