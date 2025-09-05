#!/usr/bin/env python3
"""
Cloak - Enhanced NER Extraction and Anonymization CLI

Implements a comprehensive NER pipeline with advanced anonymization features:
- Multi-pass extraction with masking and confidence thresholds  
- ThreadPool executor for parallel processing of large texts
- Advanced caching with LRU strategy and detailed analytics
- Entity merging for adjacent entities with intelligent scoring
- Support for local GLINER ONNX models
- Entity position validation and confidence filtering
- Overlap detection and resolution with multiple strategies
- Numbered redaction for consistent privacy protection
- Synthetic data replacement using Faker and custom strategies
- Auto-parallel threshold detection

Author: Rohit G
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from CloakExtraction import CloakExtraction
import cloak

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Enhanced command-line interface for the Cloak Extraction Pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Cloak - Enterprise NER Extraction and Anonymization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python main.py --model /path/to/gliner --text "John works at Google" --labels person company

  # File processing with redaction
  python main.py --model ./model --text-file input.txt --redact --labels person location

  # Synthetic replacement with custom confidence
  python main.py --model ./model --text "Alice lives in Paris" --replace --min-confidence 0.4

  # Parallel processing with custom settings
  python main.py --model ./model --text-file large.txt --parallel --workers 8 --chunk-size 500

  # Validation and overlap resolution
  python main.py --model ./model --text "Text..." --overlap-strategy longest --verbose

  # Numbered redaction with custom format
  python main.py --model ./model --text "John and Mary work here" --redact --placeholder "#{id}_{label}_HIDDEN"
"""
    )

    # Model configuration
    parser.add_argument("--model", required=True,
                       help="Path to the GLINER ONNX model directory")
    parser.add_argument("--onnx-file", default="model.onnx",
                       help="Name of the ONNX model file (default: model.onnx)")

    # Input configuration
    parser.add_argument("--text",
                       help="Input text to analyze")
    parser.add_argument("--text-file",
                       help="Path to text file to analyze")
    parser.add_argument("--labels", nargs="+", default=["person", "date", "location", "organization"],
                       help="Entity labels to detect")

    # Processing mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--extract", action="store_true", default=True,
                           help="Extract entities only (default)")
    mode_group.add_argument("--redact", action="store_true",
                           help="Redact entities with numbered placeholders")  
    mode_group.add_argument("--replace", action="store_true",
                           help="Replace entities with synthetic data")

    # Redaction options
    parser.add_argument("--placeholder", default="#{id}_{label}_REDACTED",
                       help="Placeholder format for redaction (default: #{id}_{label}_REDACTED)")
    parser.add_argument("--numbered", action="store_true", default=True,
                       help="Use numbered redaction (default: True)")

    # Replacement options  
    parser.add_argument("--replacement-file",
                       help="JSON file with custom replacement data")
    parser.add_argument("--ensure-consistency", action="store_true", default=True,
                       help="Ensure consistent replacements for identical entities")

    # Processing configuration
    parser.add_argument("--max-passes", type=int, default=2,
                       help="Maximum number of passes for multi-pass extraction (default: 2)")
    parser.add_argument("--parallel", action="store_true",
                       help="Force parallel processing")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Force single-pass processing")
    parser.add_argument("--chunk-size", type=int, default=600,
                       help="Size of word chunks for parallel processing (default: 600)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Maximum number of worker threads (default: 4)")

    # Feature configuration
    parser.add_argument("--no-merge", action="store_true",
                       help="Disable entity merging")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--cache-size", type=int, default=128,
                       help="Cache size (default: 128)")

    # Validation configuration  
    parser.add_argument("--min-confidence", type=float, default=0.3,
                       help="Minimum confidence threshold for entities (default: 0.3)")
    parser.add_argument("--no-validation", action="store_true",
                       help="Disable entity position and text validation")
    parser.add_argument("--overlap-strategy", default="highest_confidence",
                       choices=["highest_confidence", "longest", "first"],
                       help="Strategy for resolving overlapping entities (default: highest_confidence)")

    # Output configuration
    parser.add_argument("--output",
                       help="Output file for results (JSON format)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--system-info", action="store_true",
                       help="Show system information and exit")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the Cloak system
    try:
        cloak_instance = CloakExtraction(
            model_path=args.model,
            onnx_model_file=args.onnx_file,
            use_caching=not args.no_cache,
            cache_size=args.cache_size,
            min_confidence=args.min_confidence,
            strict_validation=not args.no_validation,
            overlap_strategy=args.overlap_strategy
        )
        logger.info("Cloak system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return 1

    # Show system info if requested
    if args.system_info:
        system_info = cloak_instance.get_system_info()
        print(json.dumps(system_info, indent=2))
        return 0

    # Get input text
    if args.text:
        text = args.text
    elif args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return 1
    else:
        logger.error("Either --text or --text-file must be provided")
        return 1

    # Determine parallel processing setting
    use_parallel = None
    if args.parallel:
        use_parallel = True
    elif args.no_parallel:
        use_parallel = False

    # Load custom replacement data if provided
    user_replacements = None
    if args.replacement_file:
        try:
            with open(args.replacement_file, 'r', encoding='utf-8') as f:
                user_replacements = json.load(f)
        except Exception as e:
            logger.error(f"Error loading replacement file: {e}")
            return 1

    try:
        # Determine processing mode and execute
        if args.redact:
            logger.info("Running redaction mode")
            result = cloak.redact(
                text=text,
                labels=args.labels,
                model_path=args.model,
                numbered=args.numbered,
                placeholder_format=args.placeholder
            )

        elif args.replace:
            logger.info("Running replacement mode")
            if user_replacements:
                result = cloak.replace_with_data(
                    text=text,
                    labels=args.labels,
                    user_replacements=user_replacements,
                    model_path=args.model,
                    ensure_consistency=args.ensure_consistency
                )
            else:
                result = cloak.replace(
                    text=text,
                    labels=args.labels,
                    model_path=args.model,
                    ensure_consistency=args.ensure_consistency
                )
        else:
            logger.info("Running extraction mode")
            result = cloak.extract(
                text=text,
                labels=args.labels,
                model_path=args.model,
                max_passes=args.max_passes,
                use_parallel=use_parallel,
                chunk_size=args.chunk_size,
                max_workers=args.workers,
                merge_entities=not args.no_merge,
                use_cache=not args.no_cache,
                min_confidence=args.min_confidence,
                enable_validation=not args.no_validation,
                resolve_overlaps=True
            )

        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Results saved to {args.output}")
        else:
            # Print results to console
            print("\n" + "="*80)
            print("CLOAK PROCESSING RESULTS")
            print("="*80)

            if 'anonymized_text' in result:
                print(f"\nAnonymized Text:")
                print("-" * 40)
                print(result['anonymized_text'])
                print()

            entities = result.get("entities", [])
            if entities:
                print(f"\nFound {len(entities)} entities:")
                print("-" * 40)
                for i, entity in enumerate(entities, 1):
                    print(f"{i:2d}. {entity['label']:<15} | {entity['text']:<30} | Score: {entity['score']:.3f} | Pos: {entity['start']}-{entity['end']}")
            else:
                print("\nNo entities found.")

            # Show processing information
            if 'processing_info' in result:
                print(f"\nProcessing Information:")
                print("-" * 40)
                info = result["processing_info"]
                for key, value in info.items():
                    if key not in ["cache_stats", "validation_stats"]:
                        print(f"  {key}: {value}")

            # Show validation statistics
            if args.verbose and 'processing_info' in result and 'validation_stats' in result['processing_info']:
                v_stats = result['processing_info']['validation_stats']
                if v_stats:
                    print(f"\nValidation Statistics:")
                    print("-" * 40)
                    if "validation_success_rate" in v_stats:
                        print(f"  Success rate: {v_stats['validation_success_rate']:.1%}")
                    for key, value in v_stats.items():
                        if key.endswith("_filtered") or key.endswith("_invalid") or key.endswith("_mismatch"):
                            print(f"  {key}: {value}")

            # Show cache statistics if verbose
            if args.verbose and 'processing_info' in result and "cache_stats" in result['processing_info']:
                print(f"\nCache Statistics:")
                print("-" * 40)
                cache_stats = result['processing_info']['cache_stats']
                for key, value in cache_stats.items():
                    if key != 'manager_stats':
                        print(f"  {key}: {value}")

            # Show redaction/replacement info
            if 'redaction_info' in result:
                print(f"\nRedaction Information:")
                print("-" * 40)  
                for key, value in result['redaction_info'].items():
                    print(f"  {key}: {value}")

            if 'replacement_info' in result:
                print(f"\nReplacement Information:")
                print("-" * 40)
                for key, value in result['replacement_info'].items():
                    print(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
