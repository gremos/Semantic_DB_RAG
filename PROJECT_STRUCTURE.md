# Project Structure

Complete folder and file structure for the GPT-5 Semantic Modeling & SQL Q&A System.

## Overview

```
gpt5-semantic-sql/
├── .env.example              # Environment configuration template
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── main.py                   # CLI entry point
├── data_upload/              # RDL files for discovery
├── cache/                    # Discovery & semantic caches (generated)
├── logs/                     # Log files (generated)
├── config/                   # Configuration module
├── src/                      # Source code
└── tests/                    # Unit tests
```

## Configuration Module (`config/`)

Manages environment settings and validation schemas.

```
config/
├── __init__.py               # Module initialization
├── settings.py               # Environment variable loader (Pydantic)
└── schemas.py                # JSON schemas for validation
```

### Key Files

- **settings.py**: Loads environment variables using pydantic-settings, validates required fields, provides helper properties
- **schemas.py**: JSON schemas for Discovery, Semantic Model, and Q&A validation (QuadRails)

## Source Code (`src/`)

All application logic organized by phase and responsibility.

```
src/
├── __init__.py
├── discovery/                # Phase 1: Database Discovery
│   ├── __init__.py
│   ├── introspector.py       # SQLAlchemy database introspection
│   ├── sampler.py            # Data sampling & statistics
│   ├── relationship_detector.py  # Implicit FK detection
│   ├── rdl_parser.py         # RDL file parsing
│   └── discovery_engine.py   # Main Phase 1 orchestrator
│
├── semantic/                 # Phase 2: Semantic Model Building
│   ├── __init__.py
│   ├── model_builder.py      # Main model builder
│   ├── compression.py        # Compression strategies (TODO)
│   ├── batch_processor.py    # LLM batching (TODO)
│   └── ranking.py            # Table ranking/dedup (TODO)
│
├── qa/                       # Phase 3: Question Answering
│   ├── __init__.py
│   ├── question_parser.py    # NL question parser
│   ├── sql_generator.py      # SQL generation (TODO)
│   ├── executor.py           # Query execution (TODO)
│   └── response_formatter.py # Answer formatting (TODO)
│
├── guardrails/               # QuadRails Validation
│   ├── __init__.py
│   ├── grounding.py          # Discovery grounding (TODO)
│   ├── validator.py          # JSON schema validation (TODO)
│   ├── sql_verifier.py       # SQLGlot verification (TODO)
│   └── confidence.py         # Confidence scoring (TODO)
│
├── llm/                      # LLM Integration
│   ├── __init__.py
│   ├── client.py             # Azure OpenAI client
│   └── prompts.py            # Prompt templates (TODO)
│
├── db/                       # Database Management
│   ├── __init__.py
│   ├── connection.py         # Connection manager (TODO)
│   └── readonly_guard.py     # Read-only enforcement (TODO)
│
├── utils/                    # Utilities
│   ├── __init__.py
│   ├── cache.py              # File-based caching
│   ├── logging_config.py     # Logging setup
│   ├── sql_utils.py          # SQLGlot utilities
│   └── json_utils.py         # JSON helpers (TODO)
│
└── models/                   # Pydantic Data Models
    ├── __init__.py
    ├── discovery_model.py    # Discovery data models (TODO)
    ├── semantic_model.py     # Semantic model models (TODO)
    └── qa_model.py           # Q&A models (TODO)
```

## Phase 1: Discovery (`src/discovery/`)

### Implemented Files

1. **introspector.py**
   - Uses SQLAlchemy to introspect database schema
   - Discovers tables, columns, keys, indexes
   - Applies exclusions (schemas, tables, patterns)
   - Enforces read-only connection

2. **sampler.py**
   - Samples up to 1000 rows for statistics
   - Stores up to 10 example rows per table
   - Collects: distinct_count, null_rate, min/max, sample_values
   - Detects currency/unit hints

3. **relationship_detector.py**
   - Detects implicit foreign keys
   - Uses value overlap (>80% threshold)
   - Determines cardinality (m:1 preferred)
   - Confidence scoring (high/medium/low)

4. **rdl_parser.py**
   - Parses RDL files (XML-based reports)
   - Extracts datasets, queries, parameters
   - Supports multiple RDL namespaces

5. **discovery_engine.py**
   - Main orchestrator for Phase 1
   - Coordinates introspection → sampling → relationships → RDL
   - Caches results with fingerprinting

## Phase 2: Semantic Model (`src/semantic/`)

### Status: Stub Implementation

Files are created but require LLM integration implementation:

- **model_builder.py**: Identifies entities, dimensions, facts
- **compression.py**: Compression strategies for LLM posts
- **batch_processor.py**: Batching for large schemas
- **ranking.py**: Table quality ranking (views > SPs > RDL > tables)

## Phase 3: Q&A (`src/qa/`)

### Status: Stub Implementation

Files are created but require implementation:

- **question_parser.py**: Parse NL questions
- **sql_generator.py**: Generate grounded SQL
- **executor.py**: Execute with timeouts/limits
- **response_formatter.py**: Format answers with evidence

## QuadRails (`src/guardrails/`)

### Status: Stub Implementation

Four-layer hallucination prevention:

1. **grounding.py**: Whitelist to Discovery JSON
2. **validator.py**: JSON schema validation (3 retries)
3. **sql_verifier.py**: SQLGlot parse + lint
4. **confidence.py**: Confidence scoring + escalation

## Utilities (`src/utils/`)

### Implemented Files

- **logging_config.py**: Colorized console + file logging
- **cache.py**: File-based cache with TTL
- **sql_utils.py**: SQLGlot parsing, normalization, validation

### TODO Files

- **json_utils.py**: JSON validation helpers

## LLM Integration (`src/llm/`)

- **client.py**: Azure OpenAI wrapper (stub)
- **prompts.py**: Prompt templates (TODO)

## Data Models (`src/models/`)

### Status: TODO

Pydantic models for type safety and validation:

- **discovery_model.py**: Discovery JSON structure
- **semantic_model.py**: Semantic model structure
- **qa_model.py**: Q&A request/response models

## Tests (`tests/`)

Unit tests for all modules (TODO).

```
tests/
├── __init__.py
├── test_discovery.py
├── test_semantic.py
├── test_qa.py
└── test_guardrails.py
```

## Generated Directories

These are created at runtime:

- **data_upload/**: Place RDL files here for discovery
- **cache/**: Stores discovery.json and semantic_model.json
- **logs/**: 
  - `discovery_semantic.log`: Debug logs
  - `qa_history.log.jsonl`: Q&A history for analysis

## CLI Commands

All commands are accessed through `main.py`:

```bash
# Phase 1: Discovery
python main.py discovery

# Phase 2: Build Semantic Model
python main.py model

# Phase 3: Ask Question
python main.py question "What were top 10 customers by revenue?"

# Cache Management
python main.py cache-clear

# Version Info
python main.py version
```

## Implementation Status

### ✅ Completed

- Project structure and scaffolding
- Configuration management (.env, settings, schemas)
- Phase 1 Discovery (full implementation)
  - Database introspection
  - Data sampling
  - Relationship detection
  - RDL parsing
  - Discovery engine
- Utilities (logging, caching, SQL utils)
- CLI framework

### 🔨 In Progress / TODO

- Phase 2 Semantic Model
  - LLM-based entity/dimension/fact identification
  - Compression strategies
  - Batching for large schemas
  - Table ranking algorithm

- Phase 3 Q&A
  - Question parsing with LLM
  - SQL generation with grounding
  - Query execution with safeguards
  - Result formatting
  - Q&A history logging

- QuadRails
  - Grounding verification
  - JSON schema validation with retries
  - SQL verification
  - Confidence scoring

- LLM Integration
  - Prompt engineering
  - JSON response handling
  - Retry logic with exponential backoff

- Data Models
  - Pydantic models for all structures

- Testing
  - Unit tests for all modules
  - Integration tests

## Next Steps for Development

1. **Implement Semantic Model Builder**
   - Create prompts for entity/dimension/fact identification
   - Implement compression strategies
   - Add batching for large schemas

2. **Implement Q&A Pipeline**
   - Question parsing with confidence scoring
   - SQL generation with grounding
   - Query execution with safeguards
   - Result formatting

3. **Implement QuadRails**
   - JSON schema validation
   - SQL verification with sqlglot
   - Grounding checks

4. **Add Pydantic Models**
   - Type-safe data structures
   - Automatic validation

5. **Write Tests**
   - Unit tests for each module
   - Integration tests for full pipeline

6. **Performance Optimization**
   - Async operations where possible
   - Connection pooling
   - Batch processing optimization

## Configuration Reference

See `.env.example` for all configuration options. Key variables:

- `AZURE_OPENAI_API_KEY`: Required for LLM operations
- `DATABASE_CONNECTION_STRING`: Required for discovery
- `DISCOVERY_CACHE_HOURS`: Cache TTL (default: 168 hours / 7 days)
- `*_BATCH_SIZE`: Control LLM batching
- `CONFIDENCE_*`: Confidence thresholds for Q&A

## Architecture Principles

- **KISS**: Keep it simple, avoid over-engineering
- **DRY**: Don't repeat yourself, use composition
- **YAGNI**: You aren't gonna need it, build only what's needed
- **Type Safety**: Use Pydantic for validation
- **Fail Fast**: Validate early, clear error messages
- **Evidence Chain**: Every answer must cite sources
- **Read-Only**: Never write to database
