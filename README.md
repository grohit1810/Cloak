# Cloak - NER Extraction and Anonymization Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Cloak** is an NER detection, redaction, and anonymization pipeline built on the [GLiNER](https://github.com/urchade/GLiNER) zero-shot model. It delivers multilingual entity extraction, runtime-configurable labels, numbered redaction placeholders with re-identification mapping, and Faker-powered PII replacements with document-wide consistency.

## Features

- **Zero-shot NER** — works with any entity type without retraining, powered by GLiNER
- **Dual backend** — ONNX-optimized (default) or PyTorch inference via a single `use_onnx` flag
- **Multi-pass extraction** — runs the model multiple times with decreasing confidence thresholds (0.5 -> 0.30), masking found entities between passes to discover more
- **Parallel processing** — automatic chunking and ThreadPoolExecutor for large texts
- **Numbered redaction** — consistent placeholders (`#1_PERSON_REDACTED`) with optional re-identification mapping
- **Synthetic replacement** — Faker-powered realistic alternatives with pluggable strategies per entity type
- **Thread-safe** — all shared state protected with locks; safe for concurrent use
- **Installable package** — `pip install`, CLI entry point, `pyproject.toml`

## Installation

```bash
# From source
git clone https://github.com/grohit1810/Cloak.git
cd Cloak
pip install -e ".[dev]"

# GPU support (optional)
pip install -e ".[gpu]"
```

### Requirements

- Python 3.10+
- Dependencies are managed via `pyproject.toml` (numpy, onnxruntime, transformers, gliner, faker, torch)

## Quick Start

### Python API

```python
import cloak

# Extract entities
result = cloak.extract(
    "John works at Google Inc.",
    labels=["person", "company"],
    model_path="urchade/gliner_large-v2.1"  # HuggingFace model ID or local path
)
print(result["entities"])
# [{"text": "John", "label": "person", "start": 0, "end": 4, "score": 0.95}, ...]

# Redact entities with numbered placeholders
result = cloak.redact(
    "Alice lives in Paris",
    labels=["person", "location"],
    model_path="urchade/gliner_large-v2.1"
)
print(result["anonymized_text"])
# "#1_PERSON_REDACTED lives in #1_LOCATION_REDACTED"

# Replace with synthetic data (Faker-powered)
result = cloak.replace(
    "Bob Smith works at Microsoft",
    labels=["person", "company"],
    model_path="urchade/gliner_large-v2.1"
)
print(result["anonymized_text"])
# "David Johnson works at TechCorp Inc"

# Replace with custom data
result = cloak.replace_with_data(
    "Jane works at Apple",
    labels=["person", "company"],
    user_replacements={"person": ["Anonymous User"], "company": "REDACTED_COMPANY"},
    model_path="urchade/gliner_large-v2.1"
)
```

### Command Line Interface

```bash
# Basic extraction
cloak --model urchade/gliner_large-v2.1 --text "John works at Google" --labels person company

# Redact with custom placeholder format
cloak --model ./local-model --text "Alice and Bob work here" --redact --placeholder "#{id}_{label}_HIDDEN"

# Replace with synthetic data (PyTorch backend)
cloak --model urchade/gliner_large-v2.1 --text-file input.txt --replace --no-onnx --labels person location date

# Parallel processing for large files
cloak --model ./model --text-file large.txt --parallel --workers 8 --chunk-size 500

# Validation and overlap resolution
cloak --model ./model --text "Text..." --overlap-strategy longest --verbose
```

### Class-Based API (Full Control)

```python
from cloak import CloakExtraction
from cloak.anonymization.redactor import EntityRedactor
from cloak.anonymization.replacer import EntityReplacer

# Create extraction pipeline with full configuration
pipeline = CloakExtraction(
    model_path="urchade/gliner_large-v2.1",
    use_onnx=True,           # ONNX backend (default) or PyTorch (False)
    use_caching=True,         # Cache extraction results
    min_confidence=0.3,       # Minimum entity confidence threshold
    overlap_strategy="highest_confidence",  # or "longest", "first"
)

# Extract
result = pipeline.extract_entities(
    "John Smith lives in Paris and works at Google.",
    labels=["person", "location", "organization"],
    max_passes=2,
    use_parallel=None,  # Auto-detect based on text length
)

# Then redact or replace independently
redactor = EntityRedactor()
redacted = redactor.redact(
    text="John Smith lives in Paris",
    entities=result["entities"],
    include_re_id_map=True,  # Opt-in: include reverse mapping
)
```

## Architecture

```
Input Text
    |
    v
+---------------------+
|   Text Chunking     |  <-- Word-based chunks (600 words default)
|   (if parallel)     |      Preserves character offsets
+---------------------+
    |
    v
+---------------------+
|  Multi-pass NER     |  <-- GLiNER model (ONNX or PyTorch)
|  Extraction         |      Pass 1: threshold 0.5
|                     |      Pass 2: threshold 0.3 (on masked text)
+---------------------+
    |
    v
+---------------------+
|  Entity Validation  |  <-- Confidence filtering
|  & Overlap          |      Position/text consistency checks
|  Resolution         |      Overlap resolution (3 strategies)
+---------------------+
    |
    v
+---------------------+
|  Entity Merging     |  <-- Adjacent entities with same label
|                     |      Weighted average scoring
+---------------------+
    |
    v
+---------------------+
|  Anonymization      |  <-- Redaction: numbered placeholders
|  (Optional)         |      Replacement: Faker / custom strategies
+---------------------+
    |
    v
Results + Analytics
```

### Module Structure

```
cloak/
  __init__.py              # Public API re-exports
  api.py                   # Module-level functions (extract, redact, replace)
  constants.py             # Shared configuration constants
  extraction_pipeline.py   # CloakExtraction orchestrator
  cli.py                   # Command-line interface

  models/
    gliner_model.py        # GLiNER wrapper (ONNX + PyTorch, thread-safe)

  extraction/
    extractor.py           # Multi-pass entity extraction with masking
    parallel_processor.py  # ThreadPoolExecutor parallel processing
    chunker.py             # Word-based text chunking

  anonymization/
    redactor.py            # Numbered redaction with re-identification mapping
    replacer.py            # Synthetic replacement with strategy chain
    strategies/
      base.py              # ReplacementStrategy Protocol
      faker_strategy.py    # Faker-powered realistic data
      country_strategy.py  # Geographic data preservation
      date_strategy.py     # Date format-preserving replacement
      default_strategy.py  # Fallback character-pattern replacement

  utils/
    cache_manager.py       # LRU cache with analytics
    entity_validator.py    # Confidence, position, overlap validation
    merger.py              # Adjacent entity merging (thread-safe)

  data/
    countries.json         # Country replacement data
    replacements.json      # Custom replacement pools
```

### Key Design Decisions

- **Dependency injection**: `GLiNERModel` is created once and shared across extractors. The parallel processor reuses the same model with thread-safe locking.
- **Strategy pattern**: Replacement strategies implement the `ReplacementStrategy` Protocol. New strategies can be added without modifying existing code.
- **Thread safety**: All shared state (model inference, singletons, merge stats) is protected with `threading.Lock`.
- **Immutability**: Entity dicts are deep-copied through the validation/merging pipeline to prevent mutation side effects.

## Configuration

### Extraction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | Required | HuggingFace model ID or local path |
| `use_onnx` | `True` | Use ONNX backend (faster) or PyTorch |
| `max_passes` | `2` | Multi-pass extraction rounds |
| `min_confidence` | `0.3` | Minimum entity confidence threshold |
| `chunk_size` | `600` | Words per chunk for parallel processing |
| `max_workers` | `4` | Thread pool size for parallel processing |
| `overlap_strategy` | `"highest_confidence"` | How to resolve overlapping entities (`"highest_confidence"`, `"longest"`, `"first"`) |

### Anonymization Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numbered` | `True` | Use numbered placeholders (`#1_PERSON_REDACTED`) |
| `placeholder_format` | `"#{id}_{label}_REDACTED"` | Customizable placeholder template |
| `ensure_consistency` | `True` | Same entity text gets same replacement |
| `include_re_id_map` | `False` | Include reverse mapping (opt-in for security) |
| `seed` | `None` | Faker seed for reproducible replacements |

## Development

```bash
# Setup
git clone https://github.com/grohit1810/Cloak.git
cd Cloak
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=cloak --cov-report=term-missing

# Lint and format
ruff check cloak/ tests/
ruff format cloak/ tests/
```

## Acknowledgments

- Built on the [GLiNER](https://github.com/urchade/GLiNER) architecture for zero-shot NER
- Uses [Faker](https://faker.readthedocs.io/) for realistic synthetic data generation
