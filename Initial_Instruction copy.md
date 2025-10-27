# GPT-5 Semantic Modeling & SQL Q&A System

## 0. Purpose
**Purpose**
- Auto-discover any SQL DB → build a business-friendly semantic model → answer questions with provably grounded SQL (read-only), while logging runs and capturing Q&A result history for downstream LLM analysis.

**Non-Goals**
- No schema mutation or “auto-fixing” broken schemas
- No BI server, dashboards, or write operations

## 1. Core Principles
- KISS / DRY / YAGNI — minimal, composable, 
- semantic model as single source of truth
- No Hallucination — every SQL token must trace to Discovery JSON
- Read-Only — enforce at connection & query validator
- Determinism — stable sampling/order; schema-validated JSON (1 retry)
- Evidence Chain — every answer cites model objects + joins + limits

## 2. Environment & Runtime
env variables Load from `.env` (fail fast if missing):
```
# Azure OpenAI Configuration
DEPLOYMENT_NAME=gpt-5-mini
API_VERSION=2025-01-01-preview
AZURE_ENDPOINT=
AZURE_OPENAI_API_KEY=

# Database Configuration
DATABASE_CONNECTION_STRING=

RDL_PATH=./data_upload
CACHE_DIR=./cache
UTF8_ENCODING=true
SCHEMA_EXCLUSIONS=sys,information_schema
TABLE_EXCLUSIONS=temp_,test_,backup_,old_
# Regex patterns for table exclusions (comma-separated)
TABLE_EXCLUSION_PATTERNS=.*_\d{8}$,.*_\d{6}$,.*_backup.*,.*_archive.*,.*_copy.*,.*_old.*
# Encoding & Filtering
UTF8_ENCODING=true

DISCOVERY_CACHE_HOURS=168
SEMANTIC_CACHE_HOURS=168
# Connection timeout (seconds)
DISCOVERY_TIMEOUT=300

ENTITY_BATCH_SIZE=2
DIMENSION_BATCH_SIZE=2
FACT_BATCH_SIZE=1

# Increase max retries for assembly
ASSEMBLY_MAX_RETRIES=3
```
- LLM: langchain_openai.AzureChatOpenAI (gpt-5-mini doesnt support temperature )
- SQL: sqlalchemy (introspection), sqlglot (normalize/parse/lint)
- Connections: read-only credentials; guard against DML/DDL
---


## 3. CLI Commands
```
python main.py discovery           # Phase 1: discover & cache
python main.py model               # Phase 2: build semantic model from cache
python main.py question "..."      # Phase 3: NL → SQL (returns Answer JSON)
python main.py cache-clear         # Clear discovery + semantic caches
```

## 4. QuadRails (Hallucination Prevention)
- Grounding: whitelist to Discovery JSON objects
- Constraint: strict JSON Schema (1 retry)
- Verification: sqlglot parse + dry-run lint (tables/columns/joins exist)
- Escalation: ambiguity → structured refusal with clarifying Qs

## 5. Phase 1 — Discovery (with 10-row samples)
What we do
- Introspect schemas/tables/columns/PK/FK/indexes via sqlalchemy
- Normalize SQL (views/SPs/RDL) via sqlglot
- Sample ≤1000 rows for stats; persist ≤10 example rows per table (redactable)
- Detect implicit FKs (value overlap >80% + m:1 pattern + suffix match)
- Apply exclusions & fingerprint DB for cache

### Sources scanned
- Tables & Views (metadata + samples)
- Stored Procedures (normalized SQL → inferred measures/joins)
- RDL files under env RDL_PATH (datasets, queries, joins, parameters)

### Sampling policy
- Per column: store up to 5 sample_values (stable ORDER BY PK ASC or column ASC).
- Per table: optional sample_rows up to 10 (for presentation defaults).
- Basic stats: distinct_count, null_rate, min/max (when sensible), unit/currency hints if detectable.

Discovery JSON (delta)
```
{
  "database": {"vendor": "mssql", "version": "16.0"},
  "dialect": "mssql",
  "schemas": [
    {
      "name": "dbo",
      "tables": [
        {
          "name": "Orders",
          "type": "table",
          "columns": [
            {
              "name": "OrderID",
              "type": "int",
              "nullable": false,
              "stats": {
                "distinct_count": 123456,
                "null_rate": 0.0,
                "min": 1,
                "max": 999999,
                "sample_values": ["1","2","3","4","5"]
              }
            },
            {
              "name": "OrderDate",
              "type": "datetime",
              "nullable": false,
              "stats": {
                "null_rate": 0.0,
                "min": "2024-01-01",
                "max": "2025-10-20",
                "sample_values": ["2025-10-16","2025-10-17","2025-10-18","2025-10-19","2025-10-20"]
              }
            },
            {
              "name": "TotalAmount",
              "type": "numeric(18,2)",
              "nullable": false,
              "stats": {
                "null_rate": 0.0,
                "min": 0.0,
                "max": 14999.99,
                "sample_values": ["29.99","120.00","349.00","150.00","225.50"],
                "unit_hint": "currency",
                "currency_hint": "USD"
              }
            }
          ],
          "primary_key": ["OrderID"],
          "foreign_keys": [
            {"column": "CustomerID","ref_table": "dbo.Customer","ref_column": "CustomerID"}
          ],
          "rowcount_sample": 1000,
          "sample_rows": [
            {"OrderID":"101","OrderDate":"2025-10-16","TotalAmount":"349.00","CustomerID":"C-00042"},
            {"OrderID":"102","OrderDate":"2025-10-17","TotalAmount":"29.99","CustomerID":"C-00042"}
          ],
          "source_assets": []
        }
      ]
    }
  ],
  "named_assets": [
    {"kind": "view","name": "dbo.vRevenue","sql_normalized": "SELECT ..."},
    {"kind": "stored_procedure","name": "dbo.usp_GetRevenue","sql_normalized": "SELECT ..."},
    {"kind": "rdl","path": "/data_upload/Reports/Revenue.rdl","datasets": ["RevenueSummary"]}
  ],
  "inferred_relationships": [
    {
      "from": "dbo.Orders.CustomerID",
      "to": "dbo.Customer.CustomerID",
      "method": "value_overlap",
      "overlap_rate": 0.994,
      "cardinality": "many_to_one",
      "confidence": "high"
    }
  ]
}

```

Exclusions: sys, SCHEMA_EXCLUSIONS, TABLE_EXCLUSIONS, TABLE_EXCLUSION_PATTERNS
DISCOVERY TIMEOUT env DISCOVERY_TIMEOUT


## 6. Phase 2 — Semantic Model (Upgrades)

- Meaningful columns: role, semantic_type, aliases, description.
- Display hints: default label/search/sort/attribute order.
- Measures: with unit, currency, format_hint.
- Duplicate ranking: prefer curated assets (views/SPs/RDL) > raw tables.
- Compression + batching for posting discovery → LLM:
  - Chunk assets by type with ENTITY_BATCH_SIZE / DIMENSION_BATCH_SIZE / FACT_BATCH_SIZE.
  - COMPRESSION_STRATEGY
    - tldr: extract compact column lists, types, PK/FK, top 5 values
    - map_reduce: summarize per-table then combine
    - recap: dedupe synonyms/aliases across chunks
  - Abort if schema validation fails (1 retry max)


```
{
  "entities": [
    {
      "name": "Customer",
      "source": "dbo.Customer",
      "primary_key": ["CustomerID"],
      "display": {
        "display_name": "Customer",
        "default_label_column": "CustomerName",
        "default_search_columns": ["CustomerName","Email","CustomerID"],
        "default_sort": {"column": "CustomerName","direction": "asc"}
      },
      "columns": [
        {"name": "CustomerID","role": "primary_key","semantic_type": "id","description": "Internal ID"},
        {"name": "CustomerName","role": "label","semantic_type": "person_or_org_name","aliases": ["Name"]},
        {"name": "Email","role": "attribute","semantic_type": "email"}
      ]
    }
  ],
  "dimensions": [
    {
      "name": "Date",
      "source": "dbo.DimDate",
      "keys": ["DateKey"],
      "attributes": [
        {"name": "Date","semantic_type":"date"},
        {"name": "Year","semantic_type":"year"},
        {"name": "Month","semantic_type":"month_name"}
      ],
      "display": {"attribute_order": ["Year","Month","Date"]}
    }
  ],
  "facts": [
    {
      "name": "Sales",
      "source": "dbo.FactSales",
      "grain": ["OrderID","LineID"],
      "measures": [
        {"name": "Revenue","expression": "SUM(ExtendedAmount)","unit": "currency","currency":"USD","format_hint":"currency(2)"},
        {"name": "Units","expression":"SUM(Quantity)","unit":"count"}
      ],
      "foreign_keys": [
        {"column": "CustomerID","references":"Customer.CustomerID"},
        {"column": "DateKey","references":"Date.DateKey"}
      ],
      "display": {
        "default_breakdowns": ["Customer","Date"],
        "default_filters": [{"column":"Date.Year","op":">=","value":"2024"}]
      }
    }
  ],
  "relationships": [
    {"from":"Sales.CustomerID","to":"Customer.CustomerID","cardinality":"many_to_one","confidence":"high","verification":{"overlap_rate":0.997}}
  ],
  "table_rankings": [
    {"table":"dbo.vRevenue","duplicate_of":null,"rank":1,"reason":"curated view"},
    {"table":"dbo.FactSales","duplicate_of":"dbo.vRevenue","rank":2}
  ],
  "audit": {"dialect":"mssql"}
}

```

## 7. Q&A (Upgrades)

Rules
- Use only objects in Semantic Model JSON
- Prefer views > SPs > RDL datasets > tables
- Validate via sqlglot (syntax + existence + join keys)
- Enforce limits: TOP(10) default, timeout=60s.
- return results on terminal

Response
```
{
  "status": "ok|refuse",
  "sql": [{
    "dialect": "mssql",
    "statement": "SELECT TOP (10) ...",
    "explanation": "Grounding to measures/entities",
    "evidence": {"entities": ["..."], "measures": ["..."], "relationships": ["..."]},
    "limits": {"row_limit": 1000, "timeout_sec": 60}
  }],
  "result_preview": {
    "first_row": {"ColA": "v1", "ColB": 123},
    "first_row_meaning": "Plain-English interpretation of the first row post-order.",
    "rows_sampled": 10,
    "top_10_rows": [
      {"...": "..."}  // Stored into Q&A history log (see §6.2)
    ]
  },
  "suggested_questions": [
    "Follow-up #1",
    "Follow-up #2",
    "Follow-up #3"
  ],
  "next_steps": ["Optional tips"],
  "refusal": {"reason": "only when status=refuse"}
}

```
Always include:
- sql query
- first_row + first_row_meaning
- top_10_rows for history log (also persisted to disk, §6.2)
- 2–6 suggested_questions

## 8. Logs
- Discovery & Semantic Run Logs
  - File: ${LOG_DIR}/discovery_semantic.log
  - a row logging for exeptions and debug especially for return of llm post
- Q&A History Log (for later LLM mining)
  - File: ${LOG_DIR}/qa_history.log.jsonl
  - Purpose: keep question, generated SQL, evidence, and the top 10 rows.
  - This becomes uploadable context to help the LLM learn the org’s business patterns and suggest opportunities.

## 9. Guardrails (QuadRails)
- Grounding: Every SQL token must map to Discovery JSON; cross-check against Semantic Model.
- Constraint: Strict JSON schema validation (1 retry max).
- Verification: sqlglot dry-run (parse + lint + object existence + join keys).
- Escalation: Ambiguous intent (<70% confidence) → Refusal JSON + 2–3 clarifiers.

## 10. Source Priority (unchanged)
- Curated Views
- Stored Procedures
- RDL Datasets
- Raw Tables

## 11. Processing for LLM Posts (Compression + Batching)
- Batching
  - Split by asset type; cap at ENTITY_BATCH_SIZE, DIMENSION_BATCH_SIZE, FACT_BATCH_SIZE.
- Compression
  - tldr: column list, types, PK/FK, 5 sample values, null rates, measures.
  - map_reduce: summarize per object → combine with de-dup of aliases.
  - recap: unify synonyms/aliases across schemas (e.g., CustID, CustomerID).
- De-dup ranking
  - Prefer named assets (views/SPs/RDL datasets) over raw tables for the model when data overlaps.

## 12. Execution Defaults
- Row limit: default TOP(10); hard cap 1000.
- Timeout: 60s per query.
- Read-only: permission + lints block INSERT/UPDATE/DELETE/TRUNCATE/DROP/ALTER.
- Cache TTLs: DISCOVERY_CACHE_HOURS, SEMANTIC_CACHE_HOURS.
- Nightly: optional regeneration job + diff.

