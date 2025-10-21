# Semantic Engine

Auto-derive semantic models from SQL databases and answer questions with grounded, explainable SQL.

## Features

- **Phase 1: Discovery** - Introspect database schemas, views, stored procedures, and RDL files
- **Phase 2: Semantic Modeling** - LLM-driven business-friendly model creation
- **Phase 3: Q&A** - Natural language to SQL with full grounding and verification
- **QuadRails** - 4-layer hallucination prevention (Grounding, Constraint, Verification, Escalation)
- **Multi-dialect** - Supports MSSQL, PostgreSQL via sqlglot normalization
- **Caching** - TTL-based caching for discovery and models

## Installation
```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials and database connection
```

## Usage

### CLI
```bash
# Full pipeline
python main.py full "find opportunities to upsell to customers"

# Individual phases
python main.py discover
python main.py model --hints "sales, customers"
python main.py query "show revenue by region"
```

### API
```bash
# Start server
python api/routes.py

# Use API
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "find upsell opportunities"}'
```

## Architecture

- **DRY**: Single sources of truth for schemas, prompts, and validation
- **SOLID**: Modular design with clear interfaces
- **YAGNI**: Implements only required features, no over-engineering
- **QuadRails**: Multi-layer validation prevents hallucinations

## Testing
```bash
python -m unittest discover tests
```

## License

MIT