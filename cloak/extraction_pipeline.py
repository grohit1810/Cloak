"""
Cloak Extraction - Enhanced NER Extraction Pipeline

Main orchestrator class that manages all NER extraction functionality with:
- Advanced entity validation and overlap resolution
- Sophisticated caching with detailed analytics
- Word-based parallel processing for large texts
- Multi-pass extraction with intelligent thresholds
- Enterprise-grade error handling and logging

Author: G Rohit
Version: 1.0.0
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from .constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LABELS,
    DEFAULT_MAX_PASSES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_CONFIDENCE,
)
from .extraction.extractor import EntityExtractor
from .extraction.parallel_processor import ParallelEntityProcessor
from .utils.cache_manager import CachedEntityExtractor
from .utils.entity_validator import EntityValidator
from .utils.merger import EntityMerger

logger = logging.getLogger(__name__)


class CloakExtraction:
    """
    Main orchestrator for the Cloak NER extraction pipeline.

    Provides enterprise-grade entity extraction with advanced validation,
    intelligent caching, and robust parallel processing capabilities.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        onnx_model_file: str = "model.onnx",
        use_onnx: bool = True,
        use_caching: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        strict_validation: bool = True,
        overlap_strategy: str = "highest_confidence",
    ):
        """
        Initialize the Cloak extraction pipeline.

        Args:
            model_path: Path to the local GLINER model directory or HuggingFace model ID
            onnx_model_file: Name of the ONNX model file
            use_onnx: Whether to use ONNX backend (default: True). If False, uses PyTorch.
            use_caching: Whether to enable caching (default: True)
            cache_size: Size of the cache (default: 128)
            min_confidence: Minimum confidence threshold for entities (default: 0.3)
            strict_validation: Enable strict position and text validation (default: True)
            overlap_strategy: Strategy for resolving overlaps
                ("highest_confidence", "longest", "first")
        """
        logger.info("Initializing Cloak extraction pipeline")

        self.model_path = model_path
        self.onnx_model_file = onnx_model_file
        self.use_onnx = use_onnx
        self.use_caching = use_caching
        self.cache_size = cache_size
        self.min_confidence = min_confidence
        self.strict_validation = strict_validation
        self.overlap_strategy = overlap_strategy

        # Initialize components only when model_path is provided
        if self.model_path:
            self._initialize_components()
        else:
            # Lazy initialization - components will be created when needed
            self.base_extractor = None
            self.extractor = None
            self.parallel_processor = None
            self.merger = None
            self.validator = None

        logger.info("Cloak extraction pipeline initialization complete")

    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            logger.info("Loading model from: %s", self.model_path)

            from .models.gliner_model import GLiNERModel

            gliner_model = GLiNERModel(
                model_path=str(self.model_path),
                use_onnx=self.use_onnx,
                onnx_model_file=self.onnx_model_file,
            )

            self.base_extractor = EntityExtractor(gliner_model)

            if self.use_caching:
                self.extractor = CachedEntityExtractor(self.base_extractor, self.cache_size)
                logger.info("Caching enabled with size %d", self.cache_size)
            else:
                self.extractor = self.base_extractor
                logger.info("Caching disabled")

            self.parallel_processor = ParallelEntityProcessor(gliner_model)
            self.merger = EntityMerger()
            self.validator = EntityValidator(
                min_confidence=self.min_confidence,
                strict_validation=self.strict_validation,
            )

            logger.info(
                "Entity validation: min_confidence=%s, strict=%s",
                self.min_confidence,
                self.strict_validation,
            )
            logger.info("Overlap resolution strategy: %s", self.overlap_strategy)

        except Exception as e:
            logger.error("Failed to initialize components: %s", e)
            raise

    def _ensure_initialized(self, model_path: Optional[str] = None):
        """Ensure components are initialized."""
        if self.base_extractor is not None:
            return  # Already fully initialized
        if model_path:
            self.model_path = model_path
        if self.model_path is None:
            raise ValueError("Model path must be provided either during init or method call")
        self._initialize_components()

    def extract_entities(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        max_passes: int = DEFAULT_MAX_PASSES,
        use_parallel: Optional[bool] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        merge_entities: bool = True,
        use_cache: bool = True,
        min_confidence: Optional[float] = None,
        enable_validation: bool = True,
        resolve_overlaps: bool = True,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract entities from text with advanced validation and processing.

        Args:
            text: Input text to analyze
            labels: List of entity labels to detect
            max_passes: Maximum number of passes for multi-pass extraction
            use_parallel: Force parallel processing (None=auto-detect)
            chunk_size: Size of word chunks for parallel processing
            max_workers: Maximum number of worker threads
            merge_entities: Whether to merge adjacent entities
            use_cache: Whether to use caching
            min_confidence: Override default minimum confidence threshold
            enable_validation: Whether to apply entity validation
            resolve_overlaps: Whether to resolve overlapping entities
            model_path: Path to model (for lazy initialization)

        Returns:
            Dictionary with extraction results and comprehensive metadata
        """
        # Ensure components are initialized
        self._ensure_initialized(model_path)

        # Input validation
        if not text or not text.strip():
            return self._empty_result()

        # Set default labels if none provided
        if labels is None:
            labels = DEFAULT_LABELS

        logger.info("Starting entity extraction:")
        logger.info(f"- Text length: {len(text)} characters")
        logger.info(f"- Labels: {labels}")
        logger.info(f"- Validation enabled: {enable_validation}")
        logger.info(f"- Overlap resolution: {resolve_overlaps}")

        start_time = time.time()

        try:
            # Determine processing method
            if use_parallel is None:
                word_count = len(re.findall(r"\S+", text))
                use_parallel = word_count > chunk_size

            method_used = "parallel" if use_parallel else "single-pass"
            logger.info(f"- Processing method: {method_used}")

            # Extract entities
            if use_parallel:
                entities = self.parallel_processor.process_text(
                    text=text,
                    labels=labels,
                    chunk_size=chunk_size,
                    max_workers=max_workers,
                    use_parallel=True,
                )
            else:
                if self.use_caching:
                    entities = self.extractor.predict(
                        text=text,
                        labels=labels,
                        use_cache=use_cache,
                    )
                else:
                    entities = self.base_extractor.predict(
                        text=text,
                        labels=labels,
                        max_passes=max_passes,
                    )
            passes_completed = max_passes

            logger.info(f"- Raw entities found: {len(entities)}")

            # Entity validation pipeline
            validation_stats = {}
            if enable_validation and entities:
                logger.info("--- Entity Validation Pipeline ---")

                # Step 1: Position and confidence validation
                confidence_threshold = (
                    min_confidence if min_confidence is not None else self.min_confidence
                )
                entities = self.validator.validate_entities(entities, text, confidence_threshold)

                # Step 2: Overlap resolution
                if resolve_overlaps:
                    entities = self.validator.resolve_overlaps(entities, self.overlap_strategy)

                validation_stats = self.validator.get_validation_stats()
                logger.info("--- Validation Complete ---")

            # Merge entities if requested (after validation)
            if merge_entities and entities:
                original_count = len(entities)
                entities = self.merger.merge(entities, text)
                logger.info(
                    f"- Entities after merging: {len(entities)} (reduced from {original_count})"
                )

            processing_time = time.time() - start_time

            # Prepare comprehensive result
            result = {
                "entities": entities,
                "processing_info": {
                    "text_length": len(text),
                    "processing_time": processing_time,
                    "method_used": method_used,
                    "passes_completed": passes_completed,
                    "entities_found": len(entities),
                    "merge_applied": merge_entities,
                    "cache_used": use_cache and self.use_caching,
                    "validation_applied": enable_validation,
                    "overlap_resolution_applied": resolve_overlaps,
                    "validation_stats": validation_stats,
                    "min_confidence_used": min_confidence
                    if min_confidence is not None
                    else self.min_confidence,
                    "labels_processed": labels,
                    "word_count": len(re.findall(r"\S+", text)),
                    "auto_parallel_triggered": use_parallel if use_parallel is not None else None,
                },
            }

            # Add cache statistics if caching is enabled
            if self.use_caching and hasattr(self.extractor, "get_cache_info"):
                result["processing_info"]["cache_stats"] = self.extractor.get_cache_info()

            logger.info(f"Extraction completed in {processing_time:.3f} seconds")
            logger.info(f"Found {len(entities)} entities using {method_used} processing")

            return result

        except Exception as e:
            logger.error(f"Error during entity extraction: {str(e)}")
            raise

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "entities": [],
            "processing_info": {
                "text_length": 0,
                "processing_time": 0,
                "method_used": "none",
                "passes_completed": 0,
                "entities_found": 0,
                "validation_applied": False,
                "overlap_resolution_applied": False,
                "error": "Empty or invalid input text",
            },
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the system configuration."""
        try:
            # Ensure components are initialized for info gathering
            if self.base_extractor is None:
                model_info = {"status": "Not initialized - model_path required"}
            else:
                model_info = self.base_extractor.get_model_info()

            info = {
                "system_info": {
                    "version": "1.0.0",
                    "model_path": str(self.model_path) if self.model_path else None,
                    "model_file": self.onnx_model_file,
                    "caching_enabled": self.use_caching,
                    "validation_enabled": True,
                    "min_confidence": self.min_confidence,
                    "strict_validation": self.strict_validation,
                    "overlap_strategy": self.overlap_strategy,
                },
                "model_info": model_info,
                "components": {
                    "base_extractor": type(self.base_extractor).__name__
                    if self.base_extractor
                    else "Not initialized",
                    "extractor": type(self.extractor).__name__
                    if self.extractor
                    else "Not initialized",
                    "parallel_processor": type(self.parallel_processor).__name__
                    if self.parallel_processor
                    else "Not initialized",
                    "merger": type(self.merger).__name__ if self.merger else "Not initialized",
                    "validator": type(self.validator).__name__
                    if self.validator
                    else "Not initialized",
                },
            }

            # Add cache info if available
            if self.use_caching and self.extractor and hasattr(self.extractor, "get_cache_info"):
                info["cache_info"] = self.extractor.get_cache_info()

            return info

        except Exception as e:
            logger.error(f"Error gathering system info: {str(e)}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clear the extraction cache if caching is enabled."""
        if self.use_caching and self.extractor and hasattr(self.extractor, "clear_cache"):
            self.extractor.clear_cache()
            logger.info("Cache cleared successfully")
        else:
            logger.warning("Cache not available or not enabled")
