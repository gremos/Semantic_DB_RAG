# GPT-5 Semantic Modeling & SQL Q&A System

A production-grade system that auto-discovers SQL databases, builds business-friendly semantic models, and answers natural language questions with provably grounded SQL queries.

## Features

- **Auto-Discovery**: Introspect any SQL database to discover schemas, tables, relationships
- **Semantic Modeling**: Build business-friendly models with entities, dimensions, and facts
- **NL to SQL**: Answer questions in natural language with verified SQL queries
- **QuadRails Prevention**: Four-layer hallucination prevention system
- **Read-Only**: Enforced read-only access at connection and query validator levels
- **Evidence Chain**: Every answer cites model objects, joins, and limits

## Core Principles

- **KISS / DRY / YAGNI**: Minimal, composable architecture
- **No Hallucination**: Every SQL token traces to Discovery JSON
- **Read-Only**: No write operations allowed
- **Determinism**: Stable sampling and schema-validated JSON
- **Evidence Chain**: Transparent reasoning for every answer

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gpt5-semantic-sql
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI and database credentials
```

## Quick Start

### Phase 1: Discovery
Discover and cache database schema, samples, and relationships:
```bash
python main.py discovery
```

### Phase 2: Build Semantic Model
Build business-friendly semantic model from discovery cache:
```bash
python main.py model
```

### Phase 3: Ask Questions
Query using natural language:
```bash
python main.py question "What were the top 10 customers by revenue last month?"
```

### Clear Cache
Clear discovery and semantic caches:
```bash
python main.py cache-clear
```

## Architecture

### Phase 1: Discovery
- Introspects schemas, tables, columns, keys, indexes
- Samples up to 1000 rows for statistics
- Stores up to 10 example rows per table
- Detects implicit foreign keys (>80% overlap)
- Normalizes views, stored procedures, and RDL files
- Applies exclusions and caches results

### Phase 2: Semantic Model
- Builds entities, dimensions, and facts
- Assigns semantic types and roles to columns
- Defines measures with units and formatting
- Ranks duplicate sources (views > SPs > RDL > tables)
- Compresses and batches for LLM consumption

### Phase 3: Q&A
- Parses natural language questions
- Generates SQL grounded in semantic model
- Validates with sqlglot (syntax + existence)
- Executes with read-only enforcement
- Returns results with evidence chain

## QuadRails (Hallucination Prevention)

1. **Grounding**: Whitelist to Discovery JSON objects only
2. **Constraint**: Strict JSON Schema validation (3 retries)
3. **Verification**: sqlglot parse + dry-run lint
4. **Escalation**: Ambiguity → structured refusal + clarifying questions

## Configuration

All configuration is managed through environment variables in `.env`:

### Azure OpenAI
- `DEPLOYMENT_NAME`: Model deployment name
- `API_VERSION`: Azure OpenAI API version
- `AZURE_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: API key

### Database
- `DATABASE_CONNECTION_STRING`: SQLAlchemy connection string

### Paths
- `RDL_PATH`: Path to RDL files for discovery
- `CACHE_DIR`: Cache storage directory
- `LOG_DIR`: Log files directory

### Discovery
- `DISCOVERY_TIMEOUT`: Timeout for discovery operations (seconds)
- `DISCOVERY_CACHE_HOURS`: Cache TTL for discovery results
- `SCHEMA_EXCLUSIONS`: Comma-separated schemas to exclude
- `TABLE_EXCLUSIONS`: Comma-separated table prefixes to exclude
- `TABLE_EXCLUSION_PATTERNS`: Regex patterns for table exclusions

### Semantic Model
- `SEMANTIC_CACHE_HOURS`: Cache TTL for semantic model
- `ENTITY_BATCH_SIZE`: Batch size for entity processing
- `DIMENSION_BATCH_SIZE`: Batch size for dimension processing
- `FACT_BATCH_SIZE`: Batch size for fact processing
- `COMPRESSION_STRATEGY`: Compression strategy (detailed, tldr, map_reduce, recap)

### Q&A
- `DEFAULT_ROW_LIMIT`: Default row limit for queries
- `MAX_ROW_LIMIT`: Maximum row limit allowed
- `QUERY_TIMEOUT_SEC`: Query execution timeout

### Confidence Scoring
- `CONFIDENCE_HIGH`: High confidence threshold (auto-execute)
- `CONFIDENCE_MEDIUM`: Medium confidence threshold (execute with disclaimer)
- `CONFIDENCE_LOW`: Low confidence threshold (refuse + clarify)

## Logging

### Discovery & Semantic Run Logs
- File: `${LOG_DIR}/discovery_semantic.log`
- Purpose: Track exceptions and debug LLM responses

### Q&A History Log
- File: `${LOG_DIR}/qa_history.log.jsonl`
- Purpose: Store questions, SQL, evidence, and top 10 rows for analysis
- Format: JSON Lines (one JSON object per line)

## Project Structure

```
gpt5-semantic-sql/
├── config/               # Configuration and schemas
├── src/
│   ├── discovery/       # Phase 1: Database discovery
│   ├── semantic/        # Phase 2: Semantic model building
│   ├── qa/              # Phase 3: Question answering
│   ├── guardrails/      # QuadRails validation
│   ├── llm/             # LLM client and prompts
│   ├── db/              # Database connection management
│   ├── utils/           # Utilities (caching, logging, SQL)
│   └── models/          # Pydantic data models
├── tests/               # Unit tests
├── data_upload/         # RDL files for discovery
├── cache/               # Discovery and semantic caches
└── logs/                # Log files
```

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Style
Follow PEP 8 guidelines and use type hints throughout.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
