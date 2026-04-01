"""
Entity Redactor - Numbered Redaction System

Provides enterprise-grade entity redaction with intelligent numbering:
- Consistent numbering across identical entities
- Configurable placeholder formats
- Re-identification mapping for potential reversibility
- Support for various redaction strategies

Author: G Rohit
Version: 1.0.0
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RedactionDetail:
    """Details about a single entity redaction."""

    label: str
    original: str
    placeholder: str
    start: int
    end: int
    score: float
    redaction_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "original": self.original,
            "placeholder": self.placeholder,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "redaction_id": self.redaction_id,
        }


class EntityRedactor:
    """
    Advanced entity redactor with numbered placeholders and consistency tracking.

    Features:
    - Numbered redaction with consistent IDs (#1_PERSON_REDACTED)
    - Configurable placeholder formats
    - Entity consistency across document
    - Re-identification mapping
    """

    def __init__(self, default_format: str = "#{id}_{label}_REDACTED"):
        """
        Initialize the entity redactor.

        Args:
            default_format: Default placeholder format string
                          Available variables: {id}, {label}, {count}
        """
        self.default_format = default_format
        self.entity_id_map = {}  # Maps (label, text) -> unique_id
        self.used_ids_per_label = defaultdict(set)  # Tracks used IDs per label
        self.redaction_history = []  # Track all redactions

        logger.info("EntityRedactor initialized with format: %s", default_format)

    def redact(
        self,
        text: str,
        entities: list[dict[str, Any]],
        numbered: bool = True,
        placeholder_format: str | None = None,
        preserve_case: bool = False,
        consistent_ids: bool = True,
        include_re_id_map: bool = False,
    ) -> dict[str, Any]:
        """
        Redact entities in text with numbered placeholders.

        Args:
            text: Original text to redact
            entities: List of entity dictionaries with start, end, label, text
            numbered: Whether to use numbered placeholders (default: True)
            placeholder_format: Override default placeholder format
            preserve_case: Whether to preserve original text casing in placeholder
            consistent_ids: Whether to use consistent IDs for identical entities
            include_re_id_map: Whether to include the re-identification map in the
                result (default: False). Set to True only when reversibility is
                explicitly required, as the map contains the original sensitive data.

        Returns:
            Dictionary with:
            - 'anonymized_text': Text with entities redacted
            - 'replacements': List of RedactionDetail objects
            - 'redaction_info': Metadata about the redaction process
            - 're_identification_map': Mapping for potential reversibility
                (only present when include_re_id_map=True)
        """
        if not entities:
            result = {
                "anonymized_text": text,
                "replacements": [],
                "redaction_info": {"entities_processed": 0, "redactions_applied": 0},
            }
            if include_re_id_map:
                result["re_identification_map"] = {}
            return result

        format_str = placeholder_format or self.default_format
        logger.info("Starting redaction of %d entities", len(entities))
        logger.info("Using format: %s", format_str)

        # Sort entities by start position (reverse order for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

        redacted_text = text
        redaction_details = []
        re_identification_map = {}

        if numbered and consistent_ids:
            # Build entity consistency map first
            entity_to_id = self._build_entity_id_map(entities)

        # Process entities in reverse order to preserve indices
        for entity in sorted_entities:
            try:
                label = entity["label"].upper()
                original_text = entity["text"]
                start_pos = entity["start"]
                end_pos = entity["end"]
                score = entity.get("score", 1.0)

                # Generate placeholder
                if numbered:
                    if consistent_ids:
                        entity_key = (entity["label"], original_text)
                        redaction_id = entity_to_id[entity_key]
                    else:
                        redaction_id = self._generate_unique_id(label)

                    placeholder = format_str.format(
                        id=redaction_id, label=label, count=redaction_id
                    )
                else:
                    placeholder = f"{label}_REDACTED"
                    redaction_id = f"{label}_STATIC"

                # Apply redaction
                redacted_text = redacted_text[:start_pos] + placeholder + redacted_text[end_pos:]

                # Create redaction detail
                detail = RedactionDetail(
                    label=label,
                    original=original_text,
                    placeholder=placeholder,
                    start=start_pos,
                    end=end_pos,
                    score=score,
                    redaction_id=redaction_id,
                )
                redaction_details.append(detail)

                # Build re-identification map
                re_identification_map[placeholder] = original_text

                logger.debug("Redacted '%s' -> '%s'", original_text, placeholder)

            except Exception as e:
                logger.error("Error redacting entity %s: %s", entity, str(e))
                continue

        # Sort redaction details by original position for output
        redaction_details.sort(key=lambda x: x.start)

        result = {
            "anonymized_text": redacted_text,
            "replacements": redaction_details,
            "redaction_info": {
                "entities_processed": len(entities),
                "redactions_applied": len(redaction_details),
                "format_used": format_str,
                "numbered_redaction": numbered,
                "consistent_ids": consistent_ids,
                "unique_entities": len(set((d.label, d.original) for d in redaction_details)),
            },
        }
        if include_re_id_map:
            result["re_identification_map"] = re_identification_map

        logger.info("Redaction complete: %d redactions applied", len(redaction_details))
        return result

    def _build_entity_id_map(self, entities: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
        """Build consistent ID mapping for entities."""
        entity_to_id = {}

        for entity in entities:
            label = entity["label"]
            text = entity["text"]
            entity_key = (label, text)

            if entity_key not in entity_to_id:
                # Check for pre-assigned ID (e.g., from batch_redact)
                if entity_key in self.entity_id_map:
                    entity_to_id[entity_key] = self.entity_id_map[entity_key]
                else:
                    entity_to_id[entity_key] = self._generate_unique_id(label.upper())

        return entity_to_id

    def _generate_unique_id(self, label: str) -> str:
        """Generate a unique ID for the given label."""
        used_ids = self.used_ids_per_label[label]

        # Start from 1 and find the next available ID
        candidate_id = 1
        while str(candidate_id) in used_ids:
            candidate_id += 1

        used_ids.add(str(candidate_id))
        return str(candidate_id)

    def batch_redact(
        self, texts: list[str], all_entities: list[list[dict[str, Any]]], **kwargs
    ) -> list[dict[str, Any]]:
        """
        Redact multiple texts while maintaining ID consistency across all texts.

        Args:
            texts: List of texts to redact
            all_entities: List of entity lists corresponding to each text
            **kwargs: Additional arguments for redact method

        Returns:
            List of redaction results for each text
        """
        if len(texts) != len(all_entities):
            raise ValueError("Number of texts must match number of entity lists")

        logger.info("Starting batch redaction of %d texts", len(texts))

        # Build global entity ID map for consistency across all texts
        all_entity_keys = set()
        for entities in all_entities:
            for entity in entities:
                all_entity_keys.add((entity["label"], entity["text"]))

        # Snapshot state before pre-generation so entire batch is side-effect-free
        saved_map = self.entity_id_map.copy()
        saved_used_ids = {k: v.copy() for k, v in self.used_ids_per_label.items()}

        try:
            # Pre-generate IDs for all unique entities
            global_entity_map = {}
            for label, text in all_entity_keys:
                unique_id = self._generate_unique_id(label.upper())
                global_entity_map[(label, text)] = unique_id

            # Apply redaction to each text
            results = []
            for i, (text, entities) in enumerate(zip(texts, all_entities)):
                self.entity_id_map = global_entity_map
                result = self.redact(text, entities, **kwargs)
                results.append(result)
                logger.debug("Completed redaction for text %d/%d", i + 1, len(texts))
        finally:
            self.entity_id_map = saved_map
            self.used_ids_per_label = defaultdict(set, saved_used_ids)

            logger.debug("Completed redaction for text %d/%d", i + 1, len(texts))

        logger.info("Batch redaction complete")
        return results

    def clear_history(self):
        """Clear redaction history and ID mappings."""
        self.entity_id_map.clear()
        self.used_ids_per_label.clear()
        self.redaction_history.clear()
        logger.info("Redaction history cleared")

    def get_redaction_stats(self) -> dict[str, Any]:
        """Get statistics about redaction operations."""
        total_unique_entities = len(self.entity_id_map)
        labels_processed = len(self.used_ids_per_label)

        label_stats = {}
        for label, used_ids in self.used_ids_per_label.items():
            label_stats[label] = {
                "unique_entities": len(used_ids),
                "max_id_used": max(map(int, used_ids)) if used_ids else 0,
            }

        return {
            "total_unique_entities": total_unique_entities,
            "labels_processed": labels_processed,
            "label_statistics": label_stats,
            "default_format": self.default_format,
        }
