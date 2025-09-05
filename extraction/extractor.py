"""
Multi-pass Entity Extractor - Enhanced

Implements advanced multi-pass extraction strategy with masking and confidence thresholds.
Enhanced with better error handling and logging.

Author: G Rohit (Enhanced from original)
Version: 1.0.0
"""

import warnings
import logging
from typing import List, Dict, Any, Optional

# Placeholder for GLiNERONNXModel - would need to be implemented
# from models.gliner_onnx import GLiNERONNXModel

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Enhanced multi-pass entity extractor with intelligent threshold progression.

    Features:
    - Runs multiple passes over text (up to max_passes=2)
    - Masks found entities by replacing them with spaces after each pass
    - Uses decreasing confidence thresholds for broader coverage (0.5 -> 0.30)
    - Prevents infinite loops and duplicate entity detection
    - Comprehensive error handling and logging
    """

    def __init__(self, model_path: str, onnx_model_file: str = "model.onnx"):
        """
        Initialize the entity extractor with a local GLINER ONNX model.

        Args:
            model_path: Path to the GLINER ONNX model directory
            onnx_model_file: Name of the ONNX model file
        """
        self.model_path = model_path
        self.onnx_model_file = onnx_model_file

        try:
            # Initialize model - placeholder for actual implementation
            # self.model = GLiNERONNXModel(model_path, onnx_model_file)
            self.model = None  # Placeholder
            logger.info(f"EntityExtractor initialized with model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize EntityExtractor: {str(e)}")
            raise

    def predict(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        max_passes: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Iteratively finds entities by masking found entities and re-running the model.
        This is the core multi-pass strategy with enhanced error handling.

        Args:
            text: The input text to analyze
            labels: Entity labels to predict. If None, uses default labels
            max_passes: Maximum number of passes to prevent infinite loops (default: 2)

        Returns:
            List of all unique entities found across all passes, sorted by start position
        """
        if not text or not text.strip():
            logger.warning("Empty or invalid input text provided")
            return []

        # Use default labels if none provided
        if labels is None:
            labels = ["person", "date", "location", "organization"]

        # Convert labels to lowercase (works best for gliner models)
        processed_labels = [label.lower() for label in labels]

        logger.info(f"Starting multi-pass extraction with {max_passes} passes")
        logger.info(f"Labels: {processed_labels}")

        all_entities = []
        processed_spans = set()

        # Use a list of characters for easy replacement (strings are immutable)
        mutable_text_list = list(text)

        for pass_num in range(max_passes):
            try:
                current_text_to_process = "".join(mutable_text_list)

                # Determine threshold based on pass number - ENHANCED: decreasing thresholds
                if pass_num == 0:
                    # First pass: higher threshold for confident predictions
                    threshold = 0.5
                else:
                    # Subsequent passes: lower threshold for broader coverage
                    threshold = 0.30

                # Placeholder for actual model prediction
                # newly_found_entities = self.model.predict_entities(
                #     current_text_to_process,
                #     processed_labels,
                #     threshold=threshold
                # )
                newly_found_entities = []  # Placeholder

                logger.info(f"Pass {pass_num + 1}: Found {len(newly_found_entities)} entities with threshold {threshold}")

                # If the model finds nothing, we can stop
                if not newly_found_entities:
                    logger.info(f"No new entities found in pass {pass_num + 1}, stopping early")
                    break

                # Filter out any entities we've already processed to avoid loops
                unique_new_entities = []
                for ent in newly_found_entities:
                    span = (ent['start'], ent['end'])
                    if span not in processed_spans:
                        unique_new_entities.append(ent)
                        processed_spans.add(span)

                # If there were no *genuinely* new entities, stop
                if not unique_new_entities:
                    logger.info(f"No genuinely new entities found in pass {pass_num + 1}, stopping")
                    break

                logger.info(f"Pass {pass_num + 1}: {len(unique_new_entities)} new unique entities")

                # Add the unique new finds to our master list
                all_entities.extend(unique_new_entities)

                # "Mask" the found entities by replacing them with spaces
                # This preserves the indices for the next pass
                for entity in unique_new_entities:
                    try:
                        for i in range(entity['start'], entity['end']):
                            if i < len(mutable_text_list):
                                mutable_text_list[i] = ' '
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Error masking entity {entity}: {str(e)}")
                        continue

                logger.debug(f"Masked {len(unique_new_entities)} entities for next pass")

            except Exception as e:
                logger.error(f"Error in pass {pass_num + 1}: {str(e)}")
                break

        # Sort the final combined list by start position
        try:
            all_entities.sort(key=lambda x: x['start'])
        except (KeyError, TypeError) as e:
            logger.error(f"Error sorting entities: {str(e)}")

        logger.info(f"Total entities found across all passes: {len(all_entities)}")
        return all_entities

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        try:
            # Placeholder for actual model info
            return {
                "model_path": self.model_path,
                "model_file": self.onnx_model_file,
                "status": "initialized" if self.model else "not_initialized",
                "type": "GLiNER_ONNX",
                "version": "1.0.0"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
