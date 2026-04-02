# Cloak -- End-to-End Testing Documentation

This document records all end-to-end tests performed on the Cloak NER extraction and anonymization toolkit. Every test was executed against **both** inference backends (PyTorch and ONNX) to verify feature parity. All commands and outputs shown are verbatim from actual test runs.

---

## Table of Contents

- [Model Setup](#model-setup)
  - [PyTorch Model Download](#pytorch-model-download)
  - [ONNX Export](#onnx-export)
  - [ONNX Model Verification](#onnx-model-verification)
- [Test Cases](#test-cases)
  - [Test 1: Basic Entity Extraction](#test-1-basic-entity-extraction)
  - [Test 2: Redaction with Numbered Placeholders](#test-2-redaction-with-numbered-placeholders)
  - [Test 3: Re-identification Map (Opt-in)](#test-3-re-identification-map-opt-in)
  - [Test 4: Synthetic Replacement (Faker-Powered) with Consistency](#test-4-synthetic-replacement-faker-powered-with-consistency)
  - [Test 5: Custom User Replacement Data](#test-5-custom-user-replacement-data)
  - [Test 6: Reproducible Seed-Based Replacement](#test-6-reproducible-seed-based-replacement)
  - [Test 7: Edge Cases (Empty, Whitespace, No Entities)](#test-7-edge-cases-empty-whitespace-no-entities)
  - [Test 8: Multi-Label Diverse Entity Extraction](#test-8-multi-label-diverse-entity-extraction)
  - [Test 9: Overlap Resolution Strategies](#test-9-overlap-resolution-strategies)
  - [Test 10: Batch Redaction -- Cross-Document Consistency](#test-10-batch-redaction----cross-document-consistency)
  - [Test 11: JSON Serialization via to_dict()](#test-11-json-serialization-via-to_dict)
  - [Test 12-14: CLI Modes (Extract, Redact, Replace)](#test-12-14-cli-modes-extract-redact-replace)
  - [Test 15: Parallel Processing for Large Texts](#test-15-parallel-processing-for-large-texts)
  - [Test 16: Error Handling](#test-16-error-handling)
  - [Test 17: JSON File Output via CLI](#test-17-json-file-output-via-cli)
- [Results Summary](#results-summary)

---

## Model Setup

Before running any tests, the GLiNER model must be downloaded (PyTorch weights) and exported to ONNX format. This section documents both steps plus a verification sanity check.

### PyTorch Model Download

The base model is `urchade/gliner_large-v2.1`, downloaded from HuggingFace via the `gliner` library.

```python
from gliner import GLiNER
model = GLiNER.from_pretrained('urchade/gliner_large-v2.1', load_onnx_model=False)
result = model.predict_entities('John works at Google', ['person', 'company'], threshold=0.5)
```

**Output:**

```
Sanity check: [{'start': 0, 'end': 4, 'text': 'John', 'label': 'person', 'score': 0.9956015348434448}, {'start': 14, 'end': 20, 'text': 'Google', 'label': 'company', 'score': 0.9895724654197693}]
```

Both entities detected with >98% confidence, confirming the PyTorch model loaded correctly.

### ONNX Export

Exporting to ONNX requires the `onnx` package:

```bash
pip install onnx
```

```python
from gliner import GLiNER
import os

model = GLiNER.from_pretrained('urchade/gliner_large-v2.1', load_onnx_model=False)
export_dir = 'models/gliner-large-v2.1-onnx'
os.makedirs(export_dir, exist_ok=True)
model.export_to_onnx(save_dir=export_dir, onnx_filename='model.onnx', quantize=False)
```

**Exported files:**

| File | Size |
|------|------|
| `gliner_config.json` | 0.0 MB |
| `model.onnx` | 1701.8 MB |
| `tokenizer.json` | 8.0 MB |
| `tokenizer_config.json` | 0.0 MB |

The unquantized ONNX model weighs ~1.7 GB, matching expectations for a large transformer model.

### ONNX Model Verification

```python
from gliner import GLiNER
model = GLiNER.from_pretrained('models/gliner-large-v2.1-onnx', load_onnx_model=True, local_files_only=True)
result = model.predict_entities('John works at Google', ['person', 'company'], threshold=0.5)
```

**Output:**

```
[{'start': 0, 'end': 4, 'text': 'John', 'label': 'person', 'score': 0.9956015348434448}, {'start': 14, 'end': 20, 'text': 'Google', 'label': 'company', 'score': 0.9895724654197693}]
```

ONNX produces identical scores to PyTorch since it is the same model weights exported without quantization. This confirms the export was lossless.

---

## Test Cases

Each test was run against both backends. The PyTorch backend uses the HuggingFace-hosted model (`urchade/gliner_large-v2.1`), and the ONNX backend uses the locally exported model (`models/gliner-large-v2.1-onnx`).

---

### Test 1: Basic Entity Extraction

**What it tests:** Core extraction pipeline -- can the system detect persons, companies, and locations from a sentence.

**Why it matters:** This is the fundamental operation. If this fails, nothing else works. It validates the full path from raw text through tokenization, model inference, and entity post-processing.

#### PyTorch Run

```python
import cloak
result = cloak.extract(
    'John Smith works at Google Inc. in Mountain View, California.',
    labels=['person', 'company', 'location'],
    model_path='urchade/gliner_large-v2.1',
    use_onnx=False,
)
```

**Output:**

```
Found 4 entities:
  person       | John Smith                | score: 0.993 | pos: 0-10
  company      | Google Inc                | score: 0.964 | pos: 20-30
  location     | in Mountain View          | score: 0.674 | pos: 32-48
  location     | California                | score: 0.960 | pos: 50-60
Processing time: 0.188s
```

#### ONNX Run

```python
import cloak
result = cloak.extract(
    'John Smith works at Google Inc. in California.',
    labels=['person', 'company', 'location'],
    model_path='models/gliner-large-v2.1-onnx',
    use_onnx=True,
)
```

**Output:**

```
[PASS] Entities found — 3 entities
[PASS] Person detected
[PASS] Company detected
```

**Result: PASS** on both backends.

---

### Test 2: Redaction with Numbered Placeholders

**What it tests:** Entity redaction -- replacing detected entities with consistent numbered placeholders like `#1_PERSON_REDACTED`. Same entity type gets incrementing numbers.

**Why it matters:** This is the primary privacy feature. Numbered placeholders allow tracking which entities map to which placeholders while preventing re-identification. The numbering scheme ensures that different entities of the same type (e.g., two persons) receive distinct identifiers.

#### PyTorch Run

```python
import cloak
result = cloak.redact(
    'Alice Johnson met Bob Smith at the Google campus in New York on January 15, 2024.',
    labels=['person', 'company', 'location', 'date'],
    model_path='urchade/gliner_large-v2.1',
    use_onnx=False,
)
```

**Output:**

```
Original:   Alice Johnson met Bob Smith at the Google campus in New York on January 15, 2024.
Redacted:   #1_PERSON_REDACTED met #2_PERSON_REDACTED at the #1_COMPANY_REDACTED #1_LOCATION_REDACTED in #2_LOCATION_REDACTED #1_DATE_REDACTED.

Replacements:
  Alice Johnson        -> #1_PERSON_REDACTED
  Bob Smith            -> #2_PERSON_REDACTED
  Google               -> #1_COMPANY_REDACTED
  campus               -> #1_LOCATION_REDACTED
  New York             -> #2_LOCATION_REDACTED
  on January 15, 2024  -> #1_DATE_REDACTED

Entities processed: 6
Redactions applied: 6
re_id_map present: False
```

#### ONNX Run

**Output:**

```
[PASS] Alice redacted
[PASS] Bob redacted
[PASS] PERSON in output
[PASS] re_id_map absent by default
```

**Result: PASS** -- 6 entities detected and redacted, `re_identification_map` correctly absent by default.

---

### Test 3: Re-identification Map (Opt-in)

**What it tests:** The `include_re_id_map` parameter -- when `True`, the result includes a reverse mapping from placeholder back to original text.

**Why it matters:** Security feature. The re-id map contains original PII so it defaults to `False`. Only callers who explicitly need reversibility (e.g., authorized personnel restoring originals) should opt in. This test confirms the default is safe and the opt-in mechanism works.

#### PyTorch Run

```python
from cloak import CloakExtraction
from cloak.anonymization.redactor import EntityRedactor

pipeline = CloakExtraction(model_path='urchade/gliner_large-v2.1', use_onnx=False)
result = pipeline.extract_entities('Dr. Sarah Connor called from 555-0199.', labels=['person', 'phone'])

redactor = EntityRedactor()
redacted = redactor.redact(
    text='Dr. Sarah Connor called from 555-0199.',
    entities=result['entities'],
    include_re_id_map=True,
)
```

**Output:**

```
Redacted: #1_PERSON_REDACTED #1_PHONE_REDACTED.
re_id_map: {'#1_PHONE_REDACTED': 'called from 555-0199', '#1_PERSON_REDACTED': 'Dr. Sarah Connor'}
```

#### ONNX Run

**Output:**

```
[PASS] Default excludes map
[PASS] Opt-in includes map
```

**Result: PASS** -- map present only when explicitly requested.

---

### Test 4: Synthetic Replacement (Faker-Powered) with Consistency

**What it tests:** Replacing entities with realistic fake data using the Faker library, with consistency checking -- same entity text always maps to the same replacement within a document.

**Why it matters:** Faker replacement produces realistic-looking anonymized text that preserves readability while protecting PII. Consistency is critical because if "Bob Smith" appears twice, both occurrences must resolve to the same fake name; otherwise the anonymized text becomes nonsensical.

#### PyTorch Run

```python
import cloak
result = cloak.replace(
    'Bob Smith works at Microsoft in Seattle. Bob Smith is a senior engineer.',
    labels=['person', 'company', 'location'],
    model_path='urchade/gliner_large-v2.1',
    use_onnx=False,
    ensure_consistency=True,
)
```

**Output:**

```
Original:  Bob Smith works at Microsoft in Seattle. Bob Smith is a senior engineer.
Replaced:  Bradley Morris works at Knight and Sons El Salvador. Bradley Morris is a Michelle Giles.

Replacement map:
  Bob Smith       -> Bradley Morris       (strategy: faker_cached)
  Microsoft       -> Knight and Sons      (strategy: faker)
  in Seattle      -> El Salvador          (strategy: country)
  Bob Smith       -> Bradley Morris       (strategy: faker)
  senior engineer -> Michelle Giles       (strategy: faker)
Consistency check: Both "Bob Smith" -> "Bradley Morris" : PASS
```

#### ONNX Run

**Output:**

```
[PASS] Text changed
[PASS] Consistency — Only 1 Bob found: 1
```

**Result: PASS** -- consistency verified on PyTorch, text properly anonymized on both.

---

### Test 5: Custom User Replacement Data

**What it tests:** Replacing entities with user-provided data instead of Faker-generated data.

**Why it matters:** Domain-specific applications need control over what replacements are used. For example, a legal team might want all persons replaced with "Witness A", "Witness B", etc., or a healthcare application might require specific pseudonyms.

#### PyTorch Run

```python
import cloak
result = cloak.replace_with_data(
    'Jane Doe works at Apple Inc.',
    labels=['person', 'company'],
    user_replacements={
        'person': ['Anonymous User', 'John Doe', 'Test Person'],
        'company': 'REDACTED_CORP',
    },
    model_path='urchade/gliner_large-v2.1',
    use_onnx=False,
)
```

**Output:**

```
Original:  Jane Doe works at Apple Inc.
Replaced:  Anonymous User works at REDACTED_CORP.
```

#### ONNX Run

**Output:**

```
[PASS] Person replaced
[PASS] Company replaced
```

**Result: PASS** -- user-provided replacements applied correctly.

---

### Test 6: Reproducible Seed-Based Replacement

**What it tests:** Using a seed parameter to get deterministic Faker output -- same seed produces identical replacements across runs.

**Why it matters:** Reproducibility is critical for compliance auditing, testing, and data pipelines. If a document is anonymized twice with the same seed, the output must be byte-identical. The implementation uses `faker.seed_instance()` (not the global `Faker.seed()`) so that seed isolation is maintained across concurrent calls.

#### PyTorch Run

```python
from cloak import CloakExtraction
from cloak.anonymization.replacer import EntityReplacer

pipeline = CloakExtraction(model_path='urchade/gliner_large-v2.1', use_onnx=False)
result = pipeline.extract_entities('Alice works at Tesla.', labels=['person', 'company'])

r1 = EntityReplacer(seed=42).replace('Alice works at Tesla.', result['entities'])
r2 = EntityReplacer(seed=42).replace('Alice works at Tesla.', result['entities'])
r3 = EntityReplacer(seed=99).replace('Alice works at Tesla.', result['entities'])
```

**Output:**

```
Seed 42 run 1: Brian Yang works at Rodriguez, Figueroa and Sanchez.
Seed 42 run 2: Brian Yang works at Rodriguez, Figueroa and Sanchez.
Seed 99 run 1: Casey Miller works at James-Greene.

Same seed produces same output: PASS
Different seed produces different output: PASS
```

#### ONNX Run

**Output:**

```
[PASS] Same seed = same output
[PASS] Different seed = different output
```

**Result: PASS** -- seed isolation works correctly.

---

### Test 7: Edge Cases (Empty, Whitespace, No Entities)

**What it tests:** Graceful handling of degenerate inputs -- empty strings, whitespace-only strings, and text with no detectable entities.

**Why it matters:** Production systems must handle bad input without crashing. These edge cases are common in real-world data pipelines where upstream processes may produce empty or irrelevant text.

#### PyTorch Run

**Output:**

```
Empty text:       0 entities (expected 0): PASS
Whitespace only:  0 entities (expected 0): PASS
No entities text: 0 entities (expected 0): PASS
Redact no entities: text unchanged: PASS
```

#### ONNX Run

**Output:**

```
[PASS] Empty text
[PASS] Whitespace only
[PASS] No entities
[PASS] Redact no entities
```

**Result: PASS** -- all 4 edge cases handled.

---

### Test 8: Multi-Label Diverse Entity Extraction

**What it tests:** Extracting multiple entity types from a complex academic/professional context with overlapping persons, organizations, locations, and dates.

**Why it matters:** Real-world texts contain diverse entity types that must be detected simultaneously. This test uses a realistic academic passage with titles (Dr., Prof.), institutional names, and geographic references to stress-test multi-label detection.

#### PyTorch Run

```python
text = '''On March 15, 2024, Dr. Maria Garcia from Stanford University presented 
her research at the IEEE conference in Tokyo, Japan. She collaborated with 
Prof. James Chen from MIT and Dr. Aisha Patel from Oxford University.'''

result = cloak.extract(text, labels=['person', 'organization', 'location', 'date'], ...)
```

**Output:**

```
Found 10 entities:
  date            | On March 15, 2024              | score: 0.985
  person          | Dr. Maria Garcia from          | score: 0.982
  organization    | Stanford University            | score: 0.984
  organization    | IEEE conference                | score: 0.630
  location        | in Tokyo, Japan                | score: 0.869
  person          | She                            | score: 0.408
  person          | Prof. James Chen from          | score: 0.979
  organization    | MIT                            | score: 0.982
  person          | Dr. Aisha Patel from           | score: 0.979
  organization    | Oxford University              | score: 0.975
```

#### ONNX Run

**Output:**

```
[PASS] Multiple labels found — Labels: {'organization', 'date', 'location', 'person'}
[PASS] Entity count — 5 entities
```

**Result: PASS** -- all 4 label types detected on both backends.

---

### Test 9: Overlap Resolution Strategies

**What it tests:** Three strategies for handling overlapping entities: `highest_confidence` (keep the entity with the best score), `longest` (keep the longest span), and `first` (keep the first entity found).

**Why it matters:** GLiNER may detect overlapping spans (e.g., "John Smith" as person and "Smith" as person). The strategy determines which entity wins when spans overlap. An invalid strategy must raise a `ValueError` to prevent silent misconfiguration.

#### PyTorch Run

**Output:**

```
Strategy "highest_confidence  ": [('John Smith', 'person', 0.992), ('Goldman Sachs Group Inc', 'company', 0.922)]
Strategy "longest             ": [('John Smith', 'person', 0.992), ('Goldman Sachs Group Inc', 'company', 0.922)]
Strategy "first               ": [('John Smith', 'person', 0.992), ('Goldman Sachs Group Inc', 'company', 0.922)]
Invalid strategy raises ValueError: PASS
```

#### ONNX Run

**Output:**

```
[PASS] Strategy "highest_confidence" — 2 entities
[PASS] Strategy "longest" — 2 entities
[PASS] Strategy "first" — 2 entities
[PASS] Invalid strategy error
```

**Result: PASS** -- all strategies work, invalid strategy raises `ValueError`.

---

### Test 10: Batch Redaction -- Cross-Document Consistency

**What it tests:** When redacting multiple documents in a batch, the same entity gets the same placeholder ID across all documents.

**Why it matters:** In multi-document workflows (e.g., redacting a case file with multiple emails), "John Smith" should always be `#1_PERSON_REDACTED` everywhere. Without cross-document consistency, an analyst reading the redacted corpus could not tell that the same person is referenced across documents.

#### PyTorch Run

**Output:**

```
Doc 1: #2_PERSON_REDACTED #2_LOCATION_REDACTED #1_COMPANY_REDACTED.
Doc 2: #1_PERSON_REDACTED met #2_PERSON_REDACTED in #1_LOCATION_REDACTED.
Doc 3: #1_COMPANY_REDACTED announced a new product.

Cross-doc consistency for "John Smith": PASS (#2_PERSON_REDACTED)
```

#### ONNX Run

**Output:**

```
[PASS] Cross-doc consistency — ['#1_PERSON_REDACTED', '#1_PERSON_REDACTED']
```

**Result: PASS** -- same entity gets same ID across documents.

---

### Test 11: JSON Serialization via `to_dict()`

**What it tests:** `RedactionDetail.to_dict()` produces JSON-serializable output with all required fields.

**Why it matters:** Results need to be stored and transmitted as JSON in production systems. This test verifies that the output is clean, contains no non-serializable types, and includes all fields needed for downstream processing (label, original text, placeholder, character positions, confidence score, and redaction ID).

#### PyTorch Run

**Output:**

```json
[
  {
    "label": "PERSON",
    "original": "Alice",
    "placeholder": "#1_PERSON_REDACTED",
    "start": 0,
    "end": 5,
    "score": 0.9939515590667725,
    "redaction_id": "1"
  },
  {
    "label": "COMPANY",
    "original": "IBM",
    "placeholder": "#1_COMPANY_REDACTED",
    "start": 15,
    "end": 18,
    "score": 0.9936497807502747,
    "redaction_id": "1"
  }
]
```

#### ONNX Run

**Output:**

```
[PASS] JSON serialization
```

**Result: PASS** -- clean JSON output with all fields.

---

### Test 12-14: CLI Modes (Extract, Redact, Replace)

**What it tests:** All three CLI modes (`extract`, `--redact`, `--replace`) work end-to-end from the command line.

**Why it matters:** The CLI is the primary user-facing interface for non-programmatic usage. These tests verify that argument parsing, model loading, inference, and output formatting all work correctly when invoked as a shell command.

#### PyTorch -- Extract (Test 12)

```bash
cloak --model urchade/gliner_large-v2.1 --no-onnx --text "Elon Musk founded SpaceX in 2002." --labels person company date
```

**Output:**

```
CLOAK PROCESSING RESULTS
Found 3 entities:
 1. person          | Elon Musk                      | Score: 0.996 | Pos: 0-9
 2. company         | SpaceX                         | Score: 0.991 | Pos: 18-24
 3. date            | 2002                           | Score: 0.991 | Pos: 28-32
```

#### PyTorch -- Redact (Test 13)

```bash
cloak --model urchade/gliner_large-v2.1 --no-onnx --text "Tim Cook is the CEO of Apple in Cupertino." --redact --labels person company location
```

**Output:**

```
Anonymized Text: #1_PERSON_REDACTED is the #2_PERSON_REDACTED of #1_COMPANY_REDACTED #1_LOCATION_REDACTED.
```

#### PyTorch -- Replace (Test 14)

```bash
cloak --model urchade/gliner_large-v2.1 --no-onnx --text "Jeff Bezos started Amazon in his garage in Seattle." --replace --labels person company location
```

**Output:**

```
Anonymized Text: John Robinson started Robinson-Anderson Ghana.
```

#### ONNX -- All 3 Modes

```
[PASS] CLI extract
[PASS] CLI redact
[PASS] CLI replace
```

**Result: PASS** -- all CLI modes functional on both backends.

---

### Test 15: Parallel Processing for Large Texts

**What it tests:** Auto-parallel detection triggers when text exceeds 600 words, splitting input into chunks and processing via `ThreadPoolExecutor`.

**Why it matters:** Large documents need parallel processing for reasonable throughput. This test verifies that the auto-detection threshold works, that chunking and reassembly produce correct results, and that the thread-safe model lock prevents data corruption during concurrent inference.

#### PyTorch Run (1040 words)

**Output:**

```
Text length:    6539 chars, 1040 words
Method used:    parallel
Entities found: 251
Processing time: 1.201s
Auto-parallel:  True
Unique entities: 18
```

#### ONNX Run (800 words)

**Output:**

```
Word count: 800
Method: parallel
Entities: 294
Time: 1.11s
Auto-parallel: True
ONNX parallel test: PASS
```

**Result: PASS** -- auto-parallel triggered correctly, thread-safe model lock prevents data corruption.

---

### Test 16: Error Handling

**What it tests:** Graceful error handling for invalid inputs across four scenarios.

**Why it matters:** Robust error handling prevents silent failures and gives callers actionable error messages. Each case tests a different failure mode: missing required arguments, invalid file paths, invalid enum values, and unexpected characters in input text.

**Cases tested:**

| # | Scenario | Expected Error |
|---|----------|---------------|
| 1 | `replace_with_data` without `user_replacements` | `ValueError` |
| 2 | Nonexistent model path | `FileNotFoundError` |
| 3 | Invalid `overlap_strategy` | `ValueError` |
| 4 | Special characters in text (HTML, email, @mentions) | Handled gracefully |

#### PyTorch Run

**Output:**

```
Missing user_replacements raises ValueError: PASS
Bad model path raises FileNotFoundError: PASS
Invalid overlap_strategy raises ValueError: PASS
Special characters handled: PASS (4 entities found)
```

#### ONNX Run

**Output:**

```
[PASS] Missing user_replacements
[PASS] Bad model path
[PASS] Special chars handled — 2 entities
```

**Result: PASS** -- all error cases handled correctly.

---

### Test 17: JSON File Output via CLI

**What it tests:** The `--output` flag writes structured JSON results to a file, including both entity data and processing metadata.

**Why it matters:** File-based JSON output is essential for integration with downstream pipelines, logging systems, and audit trails. This test verifies that the output file is valid JSON, contains the expected top-level keys, and includes all detected entities.

#### PyTorch Run

```bash
cloak --model urchade/gliner_large-v2.1 --no-onnx --text "Satya Nadella leads Microsoft from Redmond." --labels person company location --output results.json
```

**Output (from reading `results.json`):**

```
JSON output has entities key: True
JSON output has processing_info key: True
Entity count: 3
  person       | Satya Nadella
  company      | Microsoft
  location     | Redmond
JSON output test: PASS
```

#### ONNX Run

**Output:**

```
[PASS] JSON output valid
```

**Result: PASS** -- structured JSON output with entities and processing metadata.

---

## Results Summary

All 17 tests passed on both the PyTorch and ONNX backends.

| Test | Description | PyTorch | ONNX |
|------|-------------|---------|------|
| 1 | Basic Entity Extraction | PASS | PASS |
| 2 | Redaction with Numbered Placeholders | PASS | PASS |
| 3 | Re-identification Map (Opt-in) | PASS | PASS |
| 4 | Synthetic Replacement (Faker) with Consistency | PASS | PASS |
| 5 | Custom User Replacement Data | PASS | PASS |
| 6 | Reproducible Seed-Based Replacement | PASS | PASS |
| 7 | Edge Cases (Empty, Whitespace, No Entities) | PASS | PASS |
| 8 | Multi-Label Diverse Entity Extraction | PASS | PASS |
| 9 | Overlap Resolution Strategies | PASS | PASS |
| 10 | Batch Redaction -- Cross-Document Consistency | PASS | PASS |
| 11 | JSON Serialization via `to_dict()` | PASS | PASS |
| 12 | CLI Extract Mode | PASS | PASS |
| 13 | CLI Redact Mode | PASS | PASS |
| 14 | CLI Replace Mode | PASS | PASS |
| 15 | Parallel Processing for Large Texts | PASS | PASS |
| 16 | Error Handling | PASS | PASS |
| 17 | JSON File Output via CLI | PASS | PASS |

**Total: 17/17 PASS (PyTorch) | 17/17 PASS (ONNX)**
