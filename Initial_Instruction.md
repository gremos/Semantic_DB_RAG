# GPT-5 Semantic Modeling & SQL Q&A System (Contract Spec)
## 0. Objectives & Non-Goals
**Objectives**
- Auto-derive a neutral, business-friendly semantic model (entities, dimensions, measures, relationships, metrics) from any SQL DB.
- Use that model to answer questions with grounded, explainable SQL.
- Stay DB-agnostic via [`sqlglot`](https://sqlglot.com).
**Non-Goals**
- No schema mutation.
- No “auto-fix” of broken schemas.
---
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
## 13. References
- [`sqlglot`](https://sqlglot.com) — SQL parser/optimizer  
- [`LangChain AzureChatOpenAI`](https://api.python.langchain.com)  
- [Azure Key Vault Guidance](https://learn.microsoft.com/azure/key-vault/general/)  
- [SSRS RDL Docs](https://learn.microsoft.com/sql/reporting-services/reports/rdl-schema)  
- [Voiceflow on Guardrails](https://www.voiceflow.com/blog/guardrails-ai)