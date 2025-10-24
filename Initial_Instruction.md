0) Objectives & Non-Goals

Objectives

Auto-derive a neutral, business-friendly semantic model (entities, dimensions, measures, relationships, metrics) from any SQL database.

Use that model to answer questions (e.g., “find opportunities to upsell to customers”) with grounded, explainable SQL—no free-form hallucinations.

Stay database-agnostic; normalize metadata/SQL with sqlglot. 
sqlglot.com

Non-Goals

No schema mutation; read-only access to DB and RDL files.

Don’t attempt to “auto-fix” broken schemas; instead, abstain with a reason.

1) Environment & Configuration (no code—contract only)

Load the following from .env (fail fast if missing):

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

Notes:

Use LangChain langchain_openai.AzureChatOpenAI with the Azure endpoint + key via env vars. 
LangChain
+2
api.python.langchain.com
+2

Keep Azure OpenAI credentials in Key Vault in production; env vars only for dev. 
Microsoft Learn

2) Roles (“vibe coding”) & High-Level Flow
2.1 Roles

System (Modeling): “You are a semantic modeling assistant. Given database metadata and optional RDL/SQL assets, produce a neutral semantic model with entities, dimensions, measures, relationships, and business metrics. Return only JSON conforming to the provided schema. Do not invent fields or semantics; abstain if uncertain.”

System (Q&A): “You are a SQL answering agent. Use ONLY the provided semantic model and metadata to propose SQL (read-only). If required columns or joins are missing, return a Refusal JSON with reasons. Return only JSON conforming to the provided schema.”

Both roles must be strict-JSON-only responders; use JSON Schema validation. On validation failure, retry once with the model; otherwise fail closed.

2.2 Flow (phases)

Discovery → 2) Semantic Model Creation → 3) Q&A → 4) Verification & Evidence Return

3) Phase 1 — Discovery (schema & assets)

Inputs

Introspect DB catalogs/tables/columns/types/PKs/FKs/indexes.

Parse existing MSSQL views and stored procedures (read-only) as potential canonical logic.

Load RDL (.rdl) files from /data_upload to extract datasets/queries (they’re XML; treat as prior art/intent). 
Microsoft Learn
+2
Microsoft Learn
+2

Dialect handling

Normalize all discovered SQL and identifiers through sqlglot; record original dialect & normalized form. 
sqlglot.com

Exclusions

Drop schemas in SCHEMA_EXCLUSIONS and tables with prefixes in TABLE_EXCLUSIONS.

Caching

Persist a Discovery Cache (TTL = DISCOVERY_CACHE_HOURS) keyed by DB fingerprint (server, db name, rowcount hash of information_schema, etc.). Only re-scan if expired or explicit bypass.

Output Contract (Discovery JSON)

{
  "database": {"vendor": "mssql|postgres|...", "version": "string"},
  "dialect": "mssql|postgres|snowflake|...",
  "schemas": [
    {
      "name": "string",
      "tables": [
        {
          "name": "string",
          "type": "table|view",
          "columns": [{"name": "string","type": "string","nullable": true}],
          "primary_key": ["col", "..."],
          "foreign_keys": [{"column": "col","ref_table": "schema.table","ref_column": "col"}],
          "rowcount_sample": "integer|null",
          "source_assets": [{"kind": "view|stored_procedure|rdl","path": "string","note": "string"}]
        }
      ]
    }
  ],
  "named_assets": [
    {"kind": "view","name": "dbo.vSales","sql_normalized": "string"},
    {"kind": "stored_procedure","name": "dbo.usp_GetRevenue","sql_normalized": "string"},
    {"kind": "rdl","path": "/data_upload/Reports/Revenue.rdl","datasets": ["name1", "name2"]}
  ]
}

4) Phase 2 — Semantic Model Creation

Inputs

Discovery JSON (above).

Optional domain hints (e.g., “customers, subscriptions, invoices”).

Modeling Rules (LLM prompt content)

Star-schema bias: infer candidate facts (transactions, events, balances) and dimensions (customer/product/date).

Business names: rename technical objects to friendly names (“Customer”, “Order”, “Revenue”).

Measures: typical analytical logic (Revenue, ARPU, AOV, Churn Rate, Upsell Rate).

Relationships: infer PK/FK and grain; define many-to-one dimension joins.

Metrics: curated, business-facing definitions combining measures + filters/time (e.g., “Upsell Opportunities = customers with ≥1 purchase in last 90 days AND no purchase of product family X”).

Source mapping: each semantic object must reference concrete source tables/views (or stored procedure outputs if stable).

Caching

Persist a Semantic Model Cache (TTL = SEMANTIC_CACHE_HOURS) keyed by DB fingerprint + discovery hash.

Output Contract (Semantic Model JSON)

{
  "entities": [
    {"name": "Customer","source": "dbo.Customer","primary_key": ["CustomerID"],"description": "string"}
  ],
  "dimensions": [
    {"name": "Date","source": "dbo.DimDate","keys": ["DateKey"],"attributes": ["Year","Quarter","Month","Day"]}
  ],
  "facts": [
    {
      "name": "Sales",
      "source": "dbo.FactSales",
      "grain": ["OrderID","OrderLineID"],
      "measures": [
        {"name": "Revenue","expression": "SUM(ExtendedAmount)","datatype": "numeric","format": "currency","depends_on": ["ExtendedAmount"]}
      ],
      "foreign_keys": [{"column": "CustomerID","references": "Customer.CustomerID"}]
    }
  ],
  "relationships": [
    {"from": "Sales.CustomerID","to": "Customer.CustomerID","cardinality": "many_to_one","type": "referential"}
  ],
  "metrics": [
    {
      "name": "Upsell Opportunities",
      "purpose": "Identify customers likely to buy more",
      "logic": "Customers with recent purchase but missing cross-sell product family",
      "inputs": ["Sales","Customer","Product","Date"],
      "constraints": ["90_day_window","product_family_missing"],
      "explain": "string"
    }
  ],
  "audit": {
    "dialect": "mssql",
    "source_assets_used": [{"kind": "view|stored_procedure|rdl","name_or_path": "string"}],
    "assumptions": ["string"]
  }
}

5) Phase 3 — Question Answering (grounded SQL)

Inputs

Semantic Model JSON

Natural language question (e.g., “find opportunities to upsell to customers”).

Answering Rules

Map the question → metrics/measures/dimensions in the model.

Propose one or more candidate SQL queries (normalized with sqlglot) that compute the answer; include filters, joins, and grain explicitly.

If information is missing or ambiguous, return a Refusal JSON (see guardrails) with next best clarifying questions.

Output Contract (Answer JSON)

{
  "status": "ok|refuse",
  "sql": [
    {
      "dialect": "mssql|postgres|...",
      "statement": "string",
      "explanation": "how the SQL maps to the semantic model",
      "evidence": {
        "entities": ["Customer","Sales"],
        "measures": ["Revenue","Orders"],
        "metrics": ["Upsell Opportunities"]
      },
      "limits": {"row_limit": 1000, "timeout_sec": 60}
    }
  ],
  "next_steps": ["optional tips or filters"],
  "refusal": {"reason": "only when status=refuse"}
}


MSSQL-specific preferences (if available)

Favor views and stored procedures discovered earlier as stable, governed sources for aggregation logic.

Where appropriate, surface ready-to-run queries from stored procs/views as evidence alongside generated SQL.

6) QuadRails — Hallucination Prevention (comply-or-abstain)

Apply these four rails across both modeling and Q&A:

Grounding Rail

Only use columns/tables/entities present in Discovery JSON.

Enforce sqlglot parsing of any candidate SQL; reject if parse fails. 
sqlglot.com

Constraint Rail

Strict JSON Schema validation for every model/answer.

Max 1 retry on validation error; then return Refusal JSON.

Verification Rail

For each SQL answer, compute a dry-run lint: verify all referenced objects exist; check join keys match relationships; check for ambiguous grain.

If any check fails → Refusal JSON with specifics.

Prefer existing views/stored procs and RDL datasets when they cover the question (proven queries). 
Microsoft Learn
+1

Escalation Rail

When semantics are unclear (e.g., “Revenue” not defined), do not guess; respond with Refusal JSON suggesting 1-3 concrete clarifying questions (e.g., “gross vs net?”).

Never fabricate columns, tables, or business logic.

Cite which semantic objects are missing in refusal.reason.

Research consistently shows RAG/guardrails reduce hallucinations; we’re applying multiple layers here. 
Voiceflow
+1

7) DRY, SOLID, YAGNI, Readability & Maintainability (policy for implementers)

DRY: Single sources of truth: Discovery JSON, Semantic Model JSON, JSON Schemas, and Guardrail validators. No duplicated metadata.

SOLID

Single Responsibility: separate modules for Discovery, Modeling, Q&A, Validation, Caching.

Open/Closed: add new dialects via sqlglot configuration without touching core logic. 
sqlglot.com

Liskov Substitution: treat DB connectors uniformly (same discovery interface).

Interface Segregation: distinct interfaces for catalog read, SQL lint, RDL parse.

Dependency Inversion: orchestrator depends on abstractions (not concrete DB clients).

YAGNI: Implement only the features required by the contracts above; do not pre-build complex time-series engines or semantic BI servers.

Simple/Readable: Consistent JSON contracts, small composable prompts, short functions, explicit failure paths.

Maintainable: JSON Schema files versioned; cache keys include schema version.

8) SQL & Asset Prioritization (MSSQL-friendly)

When competing sources exist, prefer in this order:

Curated views (then SPs) discovered in Phase 1 (governed, reusable).

RDL datasets that materialize business logic already accepted by stakeholders. 
Microsoft Learn
+1

Raw tables from discovery.

Always normalize/validate queries with sqlglot to the target dialect; this keeps the system generic and portable. 
sqlglot.com

9) Prompt Contracts (ready to paste into your system prompts)

SYSTEM — Modeling

You are a semantic modeling assistant.

Input: Discovery JSON (database metadata + normalized SQL + optional RDL dataset info).

Task: Produce a neutral semantic model with: entities, dimensions, facts, measures, relationships, metrics, and audit, following the Semantic Model JSON schema above.

Use business-friendly names and typical analytical logic (Revenue, ARPU, Upsell).

Map every object to concrete sources (tables/views/SP outputs).

Do NOT invent fields, tables, or logic. If missing info prevents accurate modeling, return Refusal JSON with reasons.

Return only JSON matching the schema.

SYSTEM — Q&A

You are a SQL answering agent.

Input: Semantic Model JSON + a natural-language question.

Output: Answer JSON containing only SQL statements that can be derived from the model, with evidence mapping.

If information is insufficient or ambiguous, return Refusal JSON with up to 3 clarifying questions.

Never reference objects outside the model; never fabricate fields; never output anything but JSON.

10) Execution & Orchestration (no code—operational steps)

Load env and instantiate AzureChatOpenAI from langchain_openai (deployment + model version) using Azure endpoint and key. 
LangChain
+1

Discovery

Read DB catalogs; enumerate schemas/tables/columns/PK/FK.

Pull definitions of views and stored procs (text).

Parse /data_upload//*.rdl** (XML) to extract datasets and queries. 
Microsoft Learn

Normalize all SQL via sqlglot; persist Discovery JSON to cache. 
sqlglot.com

Modeling

Feed Discovery JSON to the Modeling role; validate output against Semantic Model JSON Schema; store to cache.

Q&A

Feed the question + Semantic Model JSON to the Q&A role.

Validate Answer JSON; run sqlglot parse on each statement; run lint checks against discovery metadata.

If any check fails, return Refusal JSON with reasons and suggestions.

Evidence-first UX

Return: (a) the validated SQL; (b) the evidence mapping to semantic objects; (c) explain how the SQL answers the question; (d) provenance (which view/SP/RDL dataset inspired the logic).

Read-only policy

Connections must be read-only; timeouts and row limits enforced by contract.

11) Example Policy for “Upsell Opportunities” (logic design, not code)

Candidate semantics:

Fact: Sales (grain: order line)

Dimensions: Customer, Product, Date

Measures: Revenue = SUM(ExtendedAmount), LastPurchaseDate = MAX(OrderDate)

Metric: Upsell Opportunities = Customers with purchase in last 90 days AND NOT purchased from ProductFamily = ‘X’

Result shape: Customer-level list with CustomerID, LastPurchaseDate, EligibleProductFamilies, SuggestedNextBestOffer (if product taxonomy exists).

Fallback: If product family taxonomy absent, Refusal JSON asking for taxonomy or alternative upsell rule.

12) Observability & QA

Log all prompts/outputs (PII-safe), cache hits/misses, SQL lint failures, and Refusal reasons.

Nightly snapshot test: regenerate the model from Discovery JSON; ensure deterministic equivalence (modulo timestamps).

Periodic RDL drift check: if RDL datasets change, invalidate semantic cache.

13) References

sqlglot (SQL parser/transpiler/optimizer; multi-dialect) 
sqlglot.com
+2
sqlglot.com
+2

LangChain langchain_openai.AzureChatOpenAI (Azure OpenAI integration & env setup) 
LangChain
+2
api.python.langchain.com
+2

Azure OpenAI env & security guidance (keys/endpoints; use Key Vault) 
Microsoft Learn
+2
Microsoft Learn
+2

SSRS RDL (definition & file format) 
Microsoft Learn
+2
Microsoft Learn
+2

Hallucination mitigation via grounding/guardrails 
Voiceflow
+1