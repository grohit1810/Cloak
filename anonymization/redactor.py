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

import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

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

        logger.info(f"EntityRedactor initialized with format: {default_format}")

    def redact(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        numbered: bool = True,
        placeholder_format: Optional[str] = None,
        preserve_case: bool = False,
        consistent_ids: bool = True
    ) -> Dict[str, Any]:
        """
        Redact entities in text with numbered placeholders.

        Args:
            text: Original text to redact
            entities: List of entity dictionaries with start, end, label, text
            numbered: Whether to use numbered placeholders (default: True)
            placeholder_format: Override default placeholder format
            preserve_case: Whether to preserve original text casing in placeholder
            consistent_ids: Whether to use consistent IDs for identical entities

        Returns:
            Dictionary with:
            - 'anonymized_text': Text with entities redacted
            - 'replacements': List of RedactionDetail objects
            - 'redaction_info': Metadata about the redaction process
            - 're_identification_map': Mapping for potential reversibility
        """
        if not entities:
            return {
                'anonymized_text': text,
                'replacements': [],
                'redaction_info': {'entities_processed': 0, 'redactions_applied': 0},
                're_identification_map': {}
            }

        format_str = placeholder_format or self.default_format
        logger.info(f"Starting redaction of {len(entities)} entities")
        logger.info(f"Using format: {format_str}")

        # Sort entities by start position (reverse order for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)

        redacted_text = text
        redaction_details = []
        re_identification_map = {}

        if numbered and consistent_ids:
            # Build entity consistency map first
            entity_to_id = self._build_entity_id_map(entities)

        # Process entities in reverse order to preserve indices
        for entity in sorted_entities:
            try:
                label = entity['label'].upper()
                original_text = entity['text']
                start_pos = entity['start']
                end_pos = entity['end']
                score = entity.get('score', 1.0)

                # Generate placeholder
                if numbered:
                    if consistent_ids:
                        entity_key = (entity['label'], original_text)
                        redaction_id = entity_to_id[entity_key]
                    else:
                        redaction_id = self._generate_unique_id(label)

                    placeholder = format_str.format(
                        id=redaction_id,
                        label=label,
                        count=redaction_id
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
                    redaction_id=redaction_id
                )
                redaction_details.append(detail)

                # Build re-identification map
                re_identification_map[placeholder] = original_text

                logger.debug(f"Redacted '{original_text}' -> '{placeholder}'")

            except Exception as e:
                logger.error(f"Error redacting entity {entity}: {str(e)}")
                continue

        # Sort redaction details by original position for output
        redaction_details.sort(key=lambda x: x.start)

        result = {
            'anonymized_text': redacted_text,
            'replacements': redaction_details,
            'redaction_info': {
                'entities_processed': len(entities),
                'redactions_applied': len(redaction_details),
                'format_used': format_str,
                'numbered_redaction': numbered,
                'consistent_ids': consistent_ids,
                'unique_entities': len(set((d.label, d.original) for d in redaction_details))
            },
            're_identification_map': re_identification_map
        }

        logger.info(f"Redaction complete: {len(redaction_details)} redactions applied")
        return result

    def _build_entity_id_map(self, entities: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
        """Build consistent ID mapping for entities."""
        entity_to_id = {}

        for entity in entities:
            label = entity['label']
            text = entity['text']
            entity_key = (label, text)

            if entity_key not in entity_to_id:
                # Generate new unique ID for this entity
                unique_id = self._generate_unique_id(label.upper())
                entity_to_id[entity_key] = unique_id

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
        self,
        texts: List[str],
        all_entities: List[List[Dict[str, Any]]],
        **kwargs
    ) -> List[Dict[str, Any]]:
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

        logger.info(f"Starting batch redaction of {len(texts)} texts")

        # Build global entity ID map for consistency across all texts
        all_entity_keys = set()
        for entities in all_entities:
            for entity in entities:
                all_entity_keys.add((entity['label'], entity['text']))

        # Pre-generate IDs for all unique entities
        global_entity_map = {}
        for label, text in all_entity_keys:
            unique_id = self._generate_unique_id(label.upper())
            global_entity_map[(label, text)] = unique_id

        # Apply redaction to each text
        results = []
        for i, (text, entities) in enumerate(zip(texts, all_entities)):
            # Temporarily override entity map for consistency
            old_map = self.entity_id_map.copy()
            self.entity_id_map = global_entity_map

            result = self.redact(text, entities, **kwargs)
            results.append(result)

            # Restore original map
            self.entity_id_map = old_map

            logger.debug(f"Completed redaction for text {i+1}/{len(texts)}")

        logger.info("Batch redaction complete")
        return results

    def clear_history(self):
        """Clear redaction history and ID mappings."""
        self.entity_id_map.clear()
        self.used_ids_per_label.clear()
        self.redaction_history.clear()
        logger.info("Redaction history cleared")

    def get_redaction_stats(self) -> Dict[str, Any]:
        """Get statistics about redaction operations."""
        total_unique_entities = len(self.entity_id_map)
        labels_processed = len(self.used_ids_per_label)

        label_stats = {}
        for label, used_ids in self.used_ids_per_label.items():
            label_stats[label] = {
                'unique_entities': len(used_ids),
                'max_id_used': max(map(int, used_ids)) if used_ids else 0
            }

        return {
            'total_unique_entities': total_unique_entities,
            'labels_processed': labels_processed,
            'label_statistics': label_stats,
            'default_format': self.default_format
        }
