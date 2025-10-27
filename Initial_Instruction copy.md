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
  - 






## 1. Environment & Config (Contract Only)
Load from `.env` (fail fast if missing):
```
DEPLOYMENT_NAME=gpt-5-mini  
MODEL_VERSION=2025-01-01-preview  
AZURE_ENDPOINT  
AZURE_OPENAI_API_KEY  
DATABASE_CONNECTION_STRING  
UTF8_ENCODING=true  
SCHEMA_EXCLUSIONS=sys,information_schema  
TABLE_EXCLUSIONS=temp_,test_,backup_,old_  
DISCOVERY_CACHE_HOURS=168  
SEMANTIC_CACHE_HOURS=168  
```
- Use `langchain_openai.AzureChatOpenAI` (endpoint + key).  
- Keep secrets in **Azure Key Vault** in production.  
- Read-only connections.
---
## 2. Roles & Flow
### Roles
- **Modeling:** derive neutral semantic model (JSON only).  
- **Q&A:** propose grounded SQL answers (JSON only).  
- Strict JSON Schema validation; 1 retry max.
### Flow
`Discovery → Semantic Model Creation → Q&A → Verification & Evidence`
---
## 3. Phase 1 — Discovery
**Inputs**
- Introspect DB metadata, views, SPs, RDLs (`/data_upload`).
- Normalize SQL via `sqlglot`.  
- Drop schemas/tables via exclusions.  
**Caching**
- TTL = DISCOVERY_CACHE_HOURS keyed by DB fingerprint.

**Output (Discovery JSON)**
```json
{
  "database": {"vendor": "mssql", "version": "string"},
  "dialect": "mssql",
  "schemas": [
    {
      "name": "string",
      "tables": [
        {
          "name": "string",
          "type": "table|view",
          "columns": [{"name": "string","type": "string","nullable": true}],
          "primary_key": ["col"],
          "foreign_keys": [{"column": "col","ref_table": "schema.table","ref_column": "col"}],
          "rowcount_sample": 0,
          "source_assets": [{"kind": "view|stored_procedure|rdl","path": "string"}]
        }
      ]
    }
  ],
  "named_assets": [
    {"kind": "view","name": "dbo.vSales","sql_normalized": "string"},
    {"kind": "stored_procedure","name": "dbo.usp_GetRevenue","sql_normalized": "string"},
    {"kind": "rdl","path": "/data_upload/Reports/Revenue.rdl","datasets": ["name1"]}
  ]
}
```
---
## 4. Phase 2 — Semantic Model Creation
**Inputs:** Discovery JSON (+ optional domain hints).  
**Rules:**  
- Star-schema bias (facts + dimensions).  
- Friendly names, typical analytical measures.  
- Each semantic object → concrete sources.  
- Cache TTL = SEMANTIC_CACHE_HOURS.
**Output (Semantic Model JSON)**
```json
{
  "entities": [{"name": "Customer","source": "dbo.Customer","primary_key": ["CustomerID"]}],
  "dimensions": [{"name": "Date","source": "dbo.DimDate","keys": ["DateKey"],"attributes": ["Year","Month"]}],
  "facts": [{
    "name": "Sales",
    "source": "dbo.FactSales",
    "grain": ["OrderID","LineID"],
    "measures": [{"name": "Revenue","expression": "SUM(ExtendedAmount)"}],
    "foreign_keys": [{"column": "CustomerID","references": "Customer.CustomerID"}]
  }],
  "relationships": [{"from": "Sales.CustomerID","to": "Customer.CustomerID","cardinality": "many_to_one"}],
  "metrics": [{
    "name": "Upsell Opportunities",
    "logic": "Customers with recent purchase but missing cross-sell product family"
  }],
  "audit": {"dialect": "mssql"}
}
```
---
## 5. Phase 3 — Question Answering
**Inputs:** Semantic Model JSON + natural language question.  
**Rules:**  
- Only use objects in the model.  
- Prefer curated views/SPs/RDLs.  
- Validate SQL via `sqlglot`.  
**Output (Answer JSON)**
```json
{
  "status": "ok|refuse",
  "sql": [{
    "dialect": "mssql",
    "statement": "SELECT ...",
    "explanation": "maps to Revenue measure",
    "evidence": {"entities": ["Customer","Sales"],"measures": ["Revenue"]},
    "limits": {"row_limit": 1000,"timeout_sec": 60}
  }],
  "next_steps": ["optional tips"],
  "refusal": {"reason": "only when status=refuse"}
}
```
---
## 6. QuadRails (Hallucination Prevention)
1. **Grounding:** only discovered objects; parse w/ sqlglot.  
2. **Constraint:** strict schema validation (1 retry).  
3. **Verification:** dry-run lint for existence + joins.  
4. **Escalation:** if ambiguous → Refusal JSON + clarifying Qs.  
---
## 7. Engineering Principles
- **DRY:** Discovery JSON = single truth.  
- **SOLID:** separate modules; open for dialects via sqlglot.  
- **YAGNI:** no unnecessary BI server logic.  
- **Readable:** small, composable, explicit fail paths.
---
## 8. SQL Source Priority
1. Curated views.  
2. Stored procedures.  
3. RDL datasets.  
4. Raw tables.  
All normalized via `sqlglot`.
---
## 9. Prompt Contracts
**SYSTEM — Modeling**
> Input: Discovery JSON  
> Output: Semantic Model JSON  
> Rules: No invention; abstain if uncertain; JSON-only.
**SYSTEM — Q&A**
> Input: Semantic Model JSON + NL question  
> Output: Answer JSON  
> Rules: Model-only objects; up to 3 clarifying Qs; JSON-only.
---
## 10. Execution Steps
1. Load env & init AzureChatOpenAI.  
2. **Discovery:** read catalogs, views, SPs, RDLs; normalize SQL.  
3. **Modeling:** validate JSON → cache.  
4. **Q&A:** generate SQL → lint → return evidence + explanation.  
5. Read-only enforcement (timeouts, limits).  
---
## 11. Example Metric — Upsell Opportunities
- Fact: Sales  
- Dims: Customer, Product, Date  
- Measure: Revenue = SUM(ExtendedAmount)  
- Metric: Customers purchased in last 90 days AND not ProductFamily = 'X'.  
- Result: CustomerID + LastPurchaseDate + Eligible Families.  
- Fallback: Refusal JSON if taxonomy missing.
---
## 12. Observability & QA
- Log prompts, cache hits/misses, lint failures, refusals.  
- Nightly snapshot = deterministic model regen.  
- Invalidate cache on RDL drift.
---
