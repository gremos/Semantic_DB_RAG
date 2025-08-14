# 🧠 Semantic Database RAG System — **v2.0 Production README**

> A practical, **production-ready** Text-to-SQL system for enterprise databases with **zero hallucinations** about schema. Built around three proven patterns, strict guardrails, and an optional 4-stage automated query pipeline you can toggle per environment.

---

## 🔭 What You Get (At a Glance)

* **A) Constrained + Execution-Guided (EG) Text-to-SQL core**
  Grammar/PICARD-style constrained decoding + execution-guided retry loop → **valid SQL** and higher end-to-end accuracy with minimal architectural change.
  PICARD (EMNLP 2021): [https://arxiv.org/abs/2109.05093](https://arxiv.org/abs/2109.05093) • ACL Anthology: [https://aclanthology.org/2021.emnlp-main.779/](https://aclanthology.org/2021.emnlp-main.779/) • EG Decoding (2018): [https://arxiv.org/abs/1807.03100](https://arxiv.org/abs/1807.03100) • Outlines (structured decoding): [https://github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) • LangChain + Outlines: [https://python.langchain.com/docs/integrations/providers/outlines/](https://python.langchain.com/docs/integrations/providers/outlines/)

* **B) Schema-first retrieval for big/complex DBs**
  A **catalog** of tables, views, columns, synonyms, embeddings, and value samples → retrieve **only** relevant objects before generation; explain why they were chosen.
  LangChain SQL-QA: [https://python.langchain.com/docs/tutorials/sql\_qa/](https://python.langchain.com/docs/tutorials/sql_qa/)

* **C) Enterprise guardrails**
  Read-only principal, **allow-list** of SQL forms, timeouts, cost/row caps, prompt isolation, input/output filters, AST validation via **SQLGlot**. Aligns with OWASP GenAI guidance.
  OWASP GenAI (LLM01): [https://genai.owasp.org/llmrisk/llm01-prompt-injection/](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) • SQLGlot: [https://github.com/tobymao/sqlglot](https://github.com/tobymao/sqlglot)

---

## 🧩 Non-Negotiable Principles

1. **Grounded in your actual schema**
   The system **must never invent** tables/columns. All analysis and SQL are limited to **discovered** objects (tables, views, columns, types, PK/FK) and **sampled values**. Any identifier not found in the discovery cache is rejected by grammar + AST checks.

2. **Views & Stored Procedures are first-class signals**
   We capture **full view definitions** and **stored procedure SELECT statements** (static SQL only) so the LLM learns **real join logic and business rules** directly from your code. We **do not execute** procedures; we analyze their text safely. (Dynamic SQL is flagged and skipped.)

3. **Deterministic, explainable retrieval**
   Before SQL generation, retrieve top-K tables/columns **and show why** (lexical/semantic matches, FK proximity, view-derived joins). Logged for observability; can be shown interactively.

4. **International/Unicode correctness**
   Full UTF-8 handling—Greek and other non-ASCII data are preserved in discovery caches and prompts.

---

## 🏗️ Architecture (Simple, Readable, Maintainable)

```
semantic-db-rag/
├── main.py                         # CLI with three options
├── shared/
│   ├── config.py                   # Env + feature flags
│   ├── models.py                   # Typed dataclasses
│   └── utils.py                    # I/O, logging, safety helpers
├── db/
│   └── discovery.py                # Schema discovery + samples + view/SP text
├── semantic/
│   └── analysis.py                 # Classification, relationships, templates
├── interactive/
│   └── query_interface.py          # 3-option CLI + 4-stage pipeline
└── data/
    ├── database_structure.json     # Canonical cache (schema + samples + views/SP text)
    ├── semantic_analysis.json      # Entity labels, relationship graph, templates
    └── query_patterns.json         # Learned successful query patterns (optional)
```

---

## ⚙️ Setup

**Environment (`.env`)**

```env
# Azure OpenAI (used by pipeline + analysis)
AZURE_OPENAI_API_KEY=...
AZURE_ENDPOINT=...
DEPLOYMENT_NAME=gpt-5-mini
MODEL_VERSION=2025-01-01-preview

# SQL Server (ODBC or SQLAlchemy DSN)
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=YourDB;Trusted_Connection=yes;

# Feature flags
ENABLE_4_STAGE_PIPELINE=true
ENABLE_VIEW_ANALYSIS=true
ENABLE_SPROC_ANALYSIS=true
ENABLE_RESULT_VALIDATION=true
ENABLE_QUERY_CACHING=true

# Performance & safety
DISCOVERY_CACHE_HOURS=24
USE_FAST_QUERIES=true
ROW_LIMIT_DEFAULT=100
QUERY_TIMEOUT_SECONDS=30
MAX_RETRY_ATTEMPTS=2
```

**Install**

```bash
pip install pyodbc python-dotenv tqdm langchain-openai sqlglot
# optional: vector store of your choice (faiss, chromadb, etc.)
```

**Run**

```bash
python main.py
```

---

## 🧱 The Three Stronger Patterns (Choose per Constraints)

### A) **Constrained + EG** Text-to-SQL Core

**What:** Keep your pipeline; add grammar/PICARD-style decoding and an execution-guided repair loop.
**Why:** Big jump in validity and accuracy with minimal change.
**How:**

* **Grammar-based decoding** (Outlines/PICARD): reject tokens that violate SQL grammar and **identifier allow-lists** (only objects discovered in `database_structure.json`).
  PICARD: [https://arxiv.org/abs/2109.05093](https://arxiv.org/abs/2109.05093) • [https://aclanthology.org/2021.emnlp-main.779/](https://aclanthology.org/2021.emnlp-main.779/)
  Outlines: [https://github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) • Docs: [https://dottxt-ai.github.io/outlines/](https://dottxt-ai.github.io/outlines/)
* **Execution-Guided repair loop**: on error/empty results, re-prompt with **DB error + schema excerpt** for 1–2 retries.
  EG: [https://arxiv.org/abs/1807.03100](https://arxiv.org/abs/1807.03100)

### B) **Schema-First Retrieval** for Large/Complex DBs

**What:** Build a **catalog** (columns/tables embeddings, synonyms, short value samples); vector/retrieval first, generate second.
**Why:** Beats keyword heuristics and gives **explainability** (why this table?).
**How:** Use LangChain SQL-QA retrieval → generation; persist embeddings near the DB.
Tutorial: [https://python.langchain.com/docs/tutorials/sql\_qa/](https://python.langchain.com/docs/tutorials/sql_qa/)

### C) **Enterprise Guardrails**

**What:** Production safety hardening: read-only principal, allow-list SQL (SELECT/WITH/EXPLAIN), timeouts, row/CPU caps, prompt isolation, input/output filters.
**How:**

* Enforce **minimal SQL EBNF**; **post-parse** with SQLGlot; block if AST contains DML/DDL or unknown identifiers.
  SQLGlot: [https://github.com/tobymao/sqlglot](https://github.com/tobymao/sqlglot)
* Align with **OWASP GenAI** controls (prompt injection & related risks).
  OWASP LLM01: [https://genai.owasp.org/llmrisk/llm01-prompt-injection/](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

---

## 🔍 Option 1 — **Advanced Database Discovery** (🔎)

**Purpose:** Produce the canonical `database_structure.json` used everywhere else.

### What we collect (only from the live DB)

* User schemas, tables, views (filter system/temporary).

* Columns (name, type, nullability), PKs, FKs.

* **Samples policy (updated):** **First 3 and Last 3 rows** per table/view:

  * Prefer `ORDER BY` **primary key** (or clustered index).
  * Example T-SQL (first 3):

    ```sql
    SELECT TOP (3) * FROM [schema].[table] ORDER BY [PrimaryKey] ASC;
    ```
  * Example T-SQL (last 3):

    ```sql
    SELECT * FROM (
      SELECT TOP (3) * FROM [schema].[table] ORDER BY [PrimaryKey] DESC
    ) t ORDER BY [PrimaryKey] ASC;
    ```
  * If no PK/index: fall back to a stable surrogate (e.g., a timestamp column). If none exists, sample **arbitrary** first/last 3 via `%%physloc%%`/`ROW_NUMBER()` with caution and **label** the order as non-deterministic.

* **View definition capture:**

  * Read definition text from `sys.views` + `sys.sql_modules`.
  * Persist the **exact** `CREATE VIEW` T-SQL (no redaction by default; consider PII/secret scanning).
  * Later parsed with SQLGlot to mine **real JOIN/ON patterns** and referenced objects.

* **Stored procedure SELECT capture:**

  * Read procedure text from `sys.procedures` + `sys.sql_modules`.
  * Extract **static** `SELECT ... FROM ... JOIN ...` statements.
  * **Do not execute** procedures. **Dynamic SQL** is detected; mark and skip (store the text but do not parse joins).

### Safety

No writes. No execution of procedures. Text is treated as **source** for parsing only.

### Output (shape)

```json
{
  "tables": {
    "dbo.Customers": {
      "columns": {"CustomerID":"INT","Name":"NVARCHAR(100)","City":"NVARCHAR(80)"},
      "primary_key": ["CustomerID"],
      "foreign_keys": [{"column":"RegionID","references":"dbo.Region(RegionID)"}],
      "samples": {
        "first_3": [
          {"CustomerID":1,"Name":"Μαρία","City":"Αθήνα"},
          {"CustomerID":2,"Name":"Γιάννης","City":"Θεσσαλονίκη"},
          {"CustomerID":3,"Name":"Ελένη","City":"Πάτρα"}
        ],
        "last_3": [
          {"CustomerID":1045,"Name":"Alice","City":"Heraklion"},
          {"CustomerID":1046,"Name":"Bob","City":"Volos"},
          {"CustomerID":1047,"Name":"Chris","City":"Larisa"}
        ],
        "ordering": {"column":"CustomerID","deterministic":true}
      }
    }
  },
  "views": {
    "dbo.vw_CustomerPayments": {
      "definition": "CREATE VIEW dbo.vw_CustomerPayments AS SELECT c.CustomerID, c.Name, p.PaymentID, p.Amount, p.PaymentDate FROM dbo.Customers c JOIN dbo.Payments p ON p.CustomerID = c.CustomerID WHERE p.Amount > 0;",
      "referenced_objects": ["dbo.Customers","dbo.Payments"],
      "parsed_joins": [
        {"left":"dbo.Customers.CustomerID","right":"dbo.Payments.CustomerID","type":"INNER"}
      ],
      "samples": {
        "first_3": [ { "CustomerID":1,"Name":"Μαρία","PaymentID":1001,"Amount":120.00,"PaymentDate":"2025-01-02" }, ... ],
        "last_3":  [ ... ]
      }
    }
  },
  "procedures": {
    "dbo.usp_CustomerRevenue": {
      "has_dynamic_sql": false,
      "select_statements": [
        "SELECT c.CustomerID, SUM(p.Amount) AS Revenue FROM dbo.Customers c JOIN dbo.Payments p ON p.CustomerID = c.CustomerID WHERE p.PaymentDate >= @StartDate AND p.PaymentDate < @EndDate GROUP BY c.CustomerID;"
      ],
      "parsed_joins": [
        {"left":"dbo.Customers.CustomerID","right":"dbo.Payments.CustomerID","type":"INNER"}
      ],
      "referenced_objects": ["dbo.Customers","dbo.Payments"]
    }
  }
}
```

> **Note:** In your environment, the **actual** `definition` (views) and `select_statements` (procedures) are written verbatim by discovery. The LLM consumes these exact texts—**no synthetic examples**.

---

## 🧠 Option 2 — **Multi-Stage Semantic Analysis**

**Purpose:** Enrich the schema cache with **business context** without guessing new identifiers.

**We do:**

* **Entity classification** (table purpose, “dimension/fact”, domain hints) using LLM prompts **restricted to discovered columns/samples**.
* **Relationship graph**:

  * Explicit FKs (from catalog)
  * **View-mined joins** (parse `CREATE VIEW` definitions to extract `JOIN ... ON ...`)
  * **Procedure-mined joins** (static `SELECT` statements only)
  * Conservative name/sample heuristics with confidence scores
* **Business templates**: Common patterns (e.g., “customer payments”), **derived solely** from real views/joins found—never invent identifiers.

Graph-augmented retrieval is used as an analogy to **GraphRAG** for structured data; we only use relationships mined from your DB’s views/FKs/SPs.
Microsoft GraphRAG overview: [https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

---

## 💬 Option 3 — **Interactive Queries** (4-Stage Automated Pipeline)

Turns a business question into validated SQL:

1. **Intent analysis** → understand metric, grouping, timeframe, entities.
2. **Relevant object retrieval** → select tables/columns via catalog & embeddings (**explain why**).
3. **Relationship resolution** → choose join paths from FK graph + **view/SP-mined** patterns.
4. **Validated SQL generation** → constrained decoding (Outlines/PICARD) + **AST check** (SQLGlot) + **EG retry** on error/0-rows.

**Strict “No-Hallucination” Rule**

* Candidate identifiers come **only** from `database_structure.json`.
* Decoder is limited to that allow-list; AST validation rejects anything else.

---

## 🛡️ Security, Safety & Compliance (Enterprise Guardrails)

* **Read-only DB principal**; rotate credentials.
* **Allow-list** SQL forms (SELECT/WITH/EXPLAIN); block DML/DDL/TRUNCATE/DROP.
* **Timeouts** (per query + total pipeline), **row caps**, and server-side governor hints.
* **Prompt isolation** (system vs user), **input/output filters**, and redaction.
* **AST validation** with SQLGlot (identifier existence, banned nodes).
  SQLGlot: [https://github.com/tobymao/sqlglot](https://github.com/tobymao/sqlglot)
* **OWASP GenAI** controls (LLM01 and related).
  OWASP LLM01: [https://genai.owasp.org/llmrisk/llm01-prompt-injection/](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

---

## 🔧 Concrete Upgrades (Ordered by ROI)

1. **Constrain SQL generation**: add grammar/PICARD or Outlines grammar for your SQL dialect; restrict identifiers to catalog.
   PICARD: [https://arxiv.org/abs/2109.05093](https://arxiv.org/abs/2109.05093) • [https://aclanthology.org/2021.emnlp-main.779/](https://aclanthology.org/2021.emnlp-main.779/) • Outlines: [https://github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
2. **Execution-Guided repair** on errors or empty results (1–2 retries with error + schema excerpt).
   EG: [https://arxiv.org/abs/1807.03100](https://arxiv.org/abs/1807.03100)
3. **Richer schema linking**: maintain a catalog with descriptions, synonyms, examples, embeddings; retrieve top-K.
   LangChain SQL-QA: [https://python.langchain.com/docs/tutorials/sql\_qa/](https://python.langchain.com/docs/tutorials/sql_qa/)
4. **Safer join discovery**: build a schema graph from PK/FK + **view/SP-mined joins**; prefer observed patterns.
   GraphRAG (analogy): [https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
5. **Deterministic result shaping**: date windows, `ORDER BY` on business keys, aggregates, pagination.
6. **Harden security**: read-only, timeouts, cost caps, allow-list, prompt isolation, **SQLGlot AST** validation.
7. **Observability & eval**: log retrieval choices, prompts, SQL, errors, latencies; evaluate on a **private Spider-style** set for your domain.
   Spider: [https://arxiv.org/abs/1809.08887](https://arxiv.org/abs/1809.08887) • [https://aclanthology.org/D18-1425/](https://aclanthology.org/D18-1425/) • [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)

---

## 🧪 Evaluation & Observability

* **Private Spider-style eval**: curate realistic NL↔SQL pairs from your domain to catch regressions.
* **Key logs per query**:

  * user question, intent parse
  * candidate objects + **why** (scores, matches, FK distance, view/SP pattern hits)
  * generated SQL, AST, errors, EG retries
  * rows returned, latency breakdowns
* **Dashboards**: accuracy, validity rate, retry rate, object-recall precision, token/latency costs.

---

## 🧵 Implementation Notes & Design Choices

### Discovery (`db/discovery.py`)

* Enumerate schemas/tables/views via `INFORMATION_SCHEMA` or `sys.*`; pull columns, types, PKs, FKs.
* **Samples = first\_3 + last\_3** using `ORDER BY` (PK preferred).
* Extract **view SQL** from `sys.sql_modules`; extract **procedure text** from `sys.procedures` + `sys.sql_modules`.
* Parse with **SQLGlot** to identify `JOIN`/`ON`, referenced objects, and a normalized join graph.
* **Do not execute** procedures; parse **static** `SELECT` statements only. Mark dynamic SQL.
* Normalize and persist into `data/database_structure.json` (UTF-8; preserve international text such as Greek).

### Semantic Analysis (`semantic/analysis.py`)

* **No new identifiers**: classification/relationships/templates are computed **only** over discovered objects and parsed joins.
* Build a relationship graph:

  * From FKs
  * From parsed view/procedure SELECT joins (via SQLGlot JOIN/ON extraction)
* Optional: embed (name, description, samples) for retrieval.
  LangChain SQL-QA: [https://python.langchain.com/docs/tutorials/sql\_qa/](https://python.langchain.com/docs/tutorials/sql_qa/)

### Interactive Pipeline (`interactive/query_interface.py`)

* **Stage 1**: intent (aggregate? window? group?).
* **Stage 2**: retrieve top-K tables/columns (lexical + embeddings + graph proximity).
* **Stage 3**: join selection from graph + view/SP patterns (prefer **observed** joins).
* **Stage 4**: constrained decoding (Outlines/PICARD) → SQLGlot AST check → EG retry on error/0-rows.

---

## 🔐 “Only Real Objects” Enforcement (How It Works)

* **Allow-listed vocabulary** of identifiers (schemas/tables/columns/functions) built from discovery cache.
* Grammar/PICARD decoding prunes tokens outside grammar **and** outside the identifier allow-list.
* Post-generation **SQLGlot AST** pass verifies:

  * identifiers exist in catalog
  * no DML/DDL/utility statements
  * joins/aggregations parse correctly
* Any violation → **block + EG retry** with error details.

---

## 🧰 Configuration (Common Flags)

```env
ENABLE_4_STAGE_PIPELINE=true        # Turn the full chain on/off
ENABLE_VIEW_ANALYSIS=true           # Parse and use view definitions
ENABLE_SPROC_ANALYSIS=true          # Parse stored procedure SELECTs (no execution)
ROW_LIMIT_DEFAULT=100               # Append TOP if user didn't
QUERY_TIMEOUT_SECONDS=30            # DB command timeout
MAX_RETRY_ATTEMPTS=2                # EG retries
TABLE_SELECTION_CONFIDENCE=0.7      # Retrieval threshold
SQL_SYNTAX_VALIDATION=true          # Enforce AST parse via SQLGlot
RELATIONSHIP_VALIDATION=true        # Validate join paths exist
```

---

## 🧭 Usage Tips (For “Vibe Coding” Sessions)

* Start with **Option 1** (Discovery). Re-run only when schema changes; the cache powers everything.
* Optionally run **Option 2** to enrich context (classes/relationships/templates)—still **zero new identifiers**.
* Use **Option 3** (Interactive) for NL → SQL with guardrails and explainable retrieval.
* For very large DBs, increase `DISCOVERY_CACHE_HOURS` and keep embeddings local to the DB host.

---

## 📚 References (Linked)

* PICARD (constrained decoding): [https://arxiv.org/abs/2109.05093](https://arxiv.org/abs/2109.05093) • [https://aclanthology.org/2021.emnlp-main.779/](https://aclanthology.org/2021.emnlp-main.779/)
* Execution-Guided Decoding: [https://arxiv.org/abs/1807.03100](https://arxiv.org/abs/1807.03100)
* Outlines (structured generation): [https://github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) • [https://dottxt-ai.github.io/outlines/](https://dottxt-ai.github.io/outlines/)
* LangChain SQL-QA tutorial: [https://python.langchain.com/docs/tutorials/sql\_qa/](https://python.langchain.com/docs/tutorials/sql_qa/)
* SQLGlot (AST/validation/transpiler): [https://github.com/tobymao/sqlglot](https://github.com/tobymao/sqlglot)
* OWASP GenAI LLM01: [https://genai.owasp.org/llmrisk/llm01-prompt-injection/](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
* Spider dataset (benchmark): [https://arxiv.org/abs/1809.08887](https://arxiv.org/abs/1809.08887) • [https://aclanthology.org/D18-1425/](https://aclanthology.org/D18-1425/) • [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)
* GraphRAG overview: [https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

---

### ✅ Ready for Implementation

This README is structured to be used **as-is** during pairing/vibe-coding: copy to your repo root, wire up `.env`, run **Option 1**, and iterate. It intentionally avoids synthetic table/column names; **all examples and SQL must be derived from your discovery cache only**—including **actual** view definitions and stored procedure `SELECT` statements captured from your database.


