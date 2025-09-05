# Cloak - Enterprise NER Extraction with Advanced Anonymization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Cloak** is an enterprise-grade Named Entity Recognition (NER) pipeline that combines superior extraction capabilities with advanced anonymization features. Built for production environments that require both high-accuracy entity detection and robust privacy protection.

## Key Features

### Advanced Entity Extraction
- **Multi-pass extraction** with intelligent threshold progression (0.5 → 0.30)
- **Word-based parallel processing** preserving semantic boundaries  
- **Comprehensive entity validation** with position, confidence, and text consistency checks
- **Sophisticated overlap resolution** with multiple strategies (highest confidence, longest, first)
- **Enterprise-grade caching** with detailed analytics and LRU strategy

### Privacy Protection & Anonymization
- **Numbered redaction** with consistent placeholders (`#1_PERSON_REDACTED`, `#2_PERSON_REDACTED`)
- **Synthetic data replacement** using Faker library for realistic alternatives
- **Custom replacement strategies** for different entity types (countries, dates, names)
- **User-defined replacement data** for domain-specific applications
- **Re-identification mapping** for potential reversibility

### Enterprise Features
- **Zero-shot capabilities** - works with any entity type without retraining
- **Configurable validation pipeline** with multiple quality control layers
- **Auto-parallel processing** with intelligent threshold detection
- **Comprehensive error handling** and logging throughout the pipeline
- **Rich analytics and statistics** for processing transparency

## Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/enterprise/cloak.git
cd cloak
pip install -e .

# Or install from PyPI (when available)
pip install cloak-ner
```

### Basic Usage

```python
import cloak

# Extract entities with advanced validation
result = cloak.extract("John works at Google Inc.", labels=['person', 'company'])
print(result['entities'])

# Redact entities with numbered placeholders
result = cloak.redact("Alice lives in Paris", labels=['person', 'location'])
print(result['anonymized_text'])
# Output: "#1_PERSON_REDACTED lives in #1_LOCATION_REDACTED"

# Replace with synthetic data
result = cloak.replace("Bob Smith works at Microsoft", labels=['person', 'company'])  
print(result['anonymized_text'])
# Output: "David Johnson works at TechCorp Inc"

# Custom replacement data
replacements = {
    'person': ['Anonymous User', 'John Doe'], 
    'company': 'REDACTED_COMPANY'
}
result = cloak.replace_with_data(
    "Jane works at Apple", 
    labels=['person', 'company'],
    user_replacements=replacements
)
```

### Command Line Interface

```bash
# Basic extraction
python main.py --model /path/to/gliner --text "John works at Google" --labels person company

# Redact with custom placeholder format  
python main.py --model ./model --text "Alice and Bob work here" --redact --placeholder "#{id}_{label}_HIDDEN"

# Replace with synthetic data
python main.py --model ./model --text-file input.txt --replace --labels person location date

# Parallel processing for large files
python main.py --model ./model --text-file large.txt --parallel --workers 8 --chunk-size 500
```

## Architecture Overview

```
Cloak Pipeline Architecture:

Input Text
    ↓
┌─────────────────────┐
│   Text Chunking     │ ← Word-based chunks (600 words default)
│   (if parallel)     │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Multi-pass NER     │ ← GLiNER ONNX model
│  Extraction         │ ← Decreasing thresholds (0.5 → 0.30)
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Entity Validation  │ ← Position, confidence, text consistency
│  & Overlap          │ ← Multiple resolution strategies
│  Resolution         │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Entity Merging     │ ← Adjacent entities with same label
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Anonymization      │ ← Redaction OR Replacement
│  (Optional)         │ ← Multiple strategies: Faker, Custom, etc.
└─────────────────────┘
    ↓
Results + Analytics
```

## Advanced Configuration

### Extraction Parameters

```python
# Comprehensive extraction configuration
result = cloak.extract(
    text="Your text here",
    labels=['person', 'location', 'organization', 'date'],
    model_path="/path/to/gliner",

    # Multi-pass settings
    max_passes=2,
    min_confidence=0.3,

    # Parallel processing
    use_parallel=None,  # Auto-detect
    chunk_size=600,     # Words per chunk
    max_workers=4,

    # Validation settings
    enable_validation=True,
    resolve_overlaps=True,
    overlap_strategy="highest_confidence",  # or "longest", "first"

    # Performance settings
    use_cache=True,
    cache_size=128,
    merge_entities=True
)
```

### Anonymization Options

```python
# Redaction with custom formatting
result = cloak.redact(
    text="Sensitive data here",
    labels=['person', 'ssn', 'email'],
    numbered=True,
    placeholder_format="#{id}_{label}_CLASSIFIED"
)

# Replacement with consistency
result = cloak.replace(
    text="John and John work together",
    labels=['person'],
    ensure_consistency=True  # Same "John" → Same replacement
)
```


### Development Setup

```bash
git clone https://github.com/grohit1810/cloak.git
cd cloak
pip install -e 
```

##  Acknowledgments

- Built upon the excellent GLiNER-X Large(Multilingual model) architecture for zero-shot NER
- Uses Faker library for realistic synthetic data generation
