# Refactoring Plan: Accurate SQL Generation Without Hallucination

## Problem Statement

The current Q&A system generates SQL from natural language but suffers from:
1. **Hallucination**: LLM invents table names, column names, or joins that don't exist
2. **Inaccuracy**: Generated SQL may be syntactically valid but semantically wrong
3. **Data Loss**: Rich data collected from audit/discovery/RDL is not effectively used

## Current Architecture

```
Audit        Discovery       RDL Parser
   │              │              │
   └──────────────┴──────────────┘
                  │
                  ▼
         Semantic Model Builder
          (LLM classification)
                  │
                  ▼
         Semantic Model (JSON)
          - entities, dimensions, facts
          - relationships
          - table_rankings
                  │
                  ▼
           SQL Generator
          (LLM generates SQL from scratch)
                  │
                  ▼
           Grounding Verifier
          (validates after generation)
```

**Key Issues:**
- SQL is generated from scratch by LLM with only schema context
- LLM has no examples of working SQL
- Rich production patterns from audit are underutilized
- RDL queries (94 production-verified SQL queries) are ignored
- View SQL (233 working queries) is not used as templates

---

## Proposed Solution: Template-Based SQL Generation

### Core Idea
Instead of asking the LLM to invent SQL, give it working examples and ask it to adapt them.

```
                      ┌─────────────────────────────────┐
                      │      SQL TEMPLATE LIBRARY       │
                      │  (from RDL, Views, Audit)       │
                      └─────────────────────────────────┘
                                    │
User Question ──────► Intent Match ─┴─► Select Best Template
                                              │
                                              ▼
                                   ┌─────────────────────────────────┐
                                   │   LLM: ADAPT (not invent)       │
                                   │   - Keep table names            │
                                   │   - Keep join patterns          │
                                   │   - Modify SELECT/WHERE/GROUP   │
                                   └─────────────────────────────────┘
                                              │
                                              ▼
                                      Validated SQL
```

---

## Data Sources Available

### 1. RDL Files (94 files in `data_upload/`)
- Production report queries with complex JOINs
- Verified working SQL
- Field mappings and parameters
- Example: `Report Παρασκευής - Contracts Campaigns.rdl` contains multi-table JOIN queries

### 2. Views (233 views from discovery)
- Database-embedded business logic
- Verified working SQL
- Example: `vwContractsPerCampaign`

### 3. Stored Procedures (302 SPs from discovery)
- Complex business logic
- Parameterized queries

### 4. Audit Data (`cache/audit_metrics.json`)
- Table access patterns (hot/warm/cold)
- Join frequency data
- Which queries are actually used in production

### 5. Discovery Data (`cache/discovery.json`)
- Schema metadata
- Foreign keys
- Column statistics
- Sample values

---

## Refactoring Steps

### Phase 1: SQL Template Library

**New File: `src/semantic/sql_templates.py`**

Extract and index SQL templates from all sources:

```python
SQLTemplate:
    id: str                    # unique identifier
    name: str                  # human-readable name
    source_type: str           # 'rdl' | 'view' | 'stored_procedure'
    source_file: str           # original file/object name

    # For matching user questions
    keywords: List[str]        # semantic keywords
    tables_used: List[str]     # tables referenced
    description: str           # what this query does

    # The actual SQL
    sql_template: str          # parameterized SQL
    sql_normalized: str        # normalized for comparison

    # Schema info (for validation)
    columns_available: List[str]
    joins_defined: List[JoinDef]

    # Usage stats (from audit)
    execution_count: int
    avg_runtime_ms: float
    last_used: datetime
```

**Extraction Process:**
1. Parse each RDL file, extract `<CommandText>` SQL from each dataset
2. Parse each view definition SQL
3. Parse stored procedure bodies for main SELECT statements
4. Normalize and deduplicate
5. Extract keywords using LLM (one-time batch process)
6. Index for fast retrieval

### Phase 2: Template Matching

**Update: `src/qa/sql_generator.py`**

New flow:
1. User asks question
2. Extract intent keywords
3. Search template library for matches (semantic similarity)
4. If match found (confidence > 0.7):
   - Use template-based generation
5. Else:
   - Fall back to schema-based generation (current approach)
   - But with lower confidence score

```python
def generate_sql(self, question: str) -> Dict:
    # Step 1: Find matching templates
    templates = self._find_matching_templates(question)

    if templates and templates[0].score > 0.7:
        # High confidence: adapt template
        return self._generate_from_template(question, templates[0])
    else:
        # Lower confidence: schema-based
        return self._generate_from_schema(question)
```

### Phase 3: Template Adaptation (LLM Task)

When adapting a template, the LLM prompt changes from:

**Current (Risky):**
```
Generate SQL for: "show contracts by campaign this month"
Available tables: Contract, Campaign, ...
Available columns: ...
```

**New (Constrained):**
```
Adapt this VERIFIED SQL template to answer the user's question.

TEMPLATE (from production report "Contracts per Campaign"):
---------------------------------------------------------
SELECT vw.[Καμπάνια], vw.[Κωδ. Πελάτη], vw.[Αξία]
FROM [dbo].[vwContractsPerCampaign] vw
WHERE vw.CampaignID = @CampaignID
AND [Ημ. Δημιουργίας] BETWEEN @DateFrom AND @DateTo
---------------------------------------------------------

Available columns: Καμπάνια, Κωδ. Πελάτη, Επωνυμία, Αξία, Πωλητής, ...
Available filters: CampaignID, Ημ. Υπογραφής, Ημ. Δημιουργίας

User question: "show contracts by campaign this month"

RULES:
1. DO NOT change table names - they are verified
2. DO NOT change JOIN conditions - they are verified
3. ONLY modify: SELECT columns, WHERE filters, GROUP BY, ORDER BY
4. Use ONLY columns from the available list

Generate adapted SQL:
```

### Phase 4: Enrich Semantic Model

**Update: `src/semantic/model_builder.py`**

Add to semantic model output:
```python
semantic_model = {
    # Existing
    "entities": [...],
    "dimensions": [...],
    "facts": [...],
    "relationships": [...],

    # NEW: SQL Template Library
    "sql_templates": [
        {
            "id": "contracts_per_campaign",
            "name": "Contracts per Campaign",
            "source": "rdl:Report Παρασκευής - Contracts Campaigns.rdl",
            "keywords": ["contract", "campaign", "sales", "συμβόλαιο", "καμπάνια"],
            "tables": ["vwContractsPerCampaign", "TargetGroupItemValue", "User"],
            "sql_template": "SELECT ... FROM ... WHERE ...",
            "columns": {...},
            "usage": {"execution_count": 5000, "avg_runtime_ms": 120}
        }
    ],

    # NEW: Production Join Patterns (from audit)
    "verified_joins": [
        {
            "pattern": "Contract.CampaignID = Campaign.ID",
            "frequency": 25000,
            "sources": ["rdl:Report1", "view:vwContracts", "audit"]
        }
    ],

    # NEW: Column Semantic Index
    "column_index": {
        "revenue": ["Contract.Amount", "vwContracts.Αξία"],
        "date": ["Contract.CreateDate", "Contract.SignDate"],
        "customer": ["Customer.Name", "Contract.CustomerID"]
    }
}
```

### Phase 5: Grounding Enhancement

**Update: `src/guardrails/grounding.py`**

Move grounding from post-validation to pre-generation:
- Before generating SQL, check if all referenced tables/columns exist
- If template-based: validate template's schema still matches database
- Reject invalid SQL before execution, not after

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/semantic/sql_templates.py` | **NEW** - Template extraction and indexing |
| `src/semantic/model_builder.py` | Add template library to semantic model |
| `src/qa/sql_generator.py` | Template matching + constrained adaptation |
| `src/guardrails/grounding.py` | Pre-generation validation |
| `src/discovery/rdl_parser.py` | Enhance SQL extraction with keyword generation |
| `config/settings.py` | Add template library settings |

---

## Implementation Order

1. **sql_templates.py** - Create template extraction
2. **model_builder.py** - Integrate templates into semantic model build
3. **sql_generator.py** - Add template matching and adaptation
4. **grounding.py** - Enhance pre-validation
5. **Test** - Run Q&A with template-based generation

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Hallucinated tables | ~20% | <2% |
| Hallucinated columns | ~30% | <5% |
| Wrong JOINs | ~40% | <5% |
| SQL execution errors | ~25% | <5% |
| User satisfaction | Low | High |

---

## Key Principles

1. **Trust Production Code**: RDL/View SQL works. Use it.
2. **Constrain the LLM**: Don't let it invent, let it adapt.
3. **Verify Early**: Check validity before generation, not after.
4. **Use Audit Data**: Production usage patterns are truth.
5. **Keep It Simple**: Template matching + constrained adaptation.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Template library too large | Index by keywords, limit search space |
| No matching template | Fall back to schema-based with lower confidence |
| Template outdated (schema changed) | Validate template schema on load, mark stale |
| Greek/English mixed keywords | Support both in keyword extraction |
| Performance overhead | Cache template index, lazy load SQL |

---

## Success Criteria

1. Q&A returns valid, executable SQL 95%+ of the time
2. SQL uses correct table/column names (grounded)
3. JOIN patterns match production usage
4. User questions about contracts/campaigns use the verified RDL patterns
5. Confidence scores accurately reflect reliability

---

## Token Analysis: Current State

### File Sizes and Token Estimates

| File | Size | ~Tokens | Purpose |
|------|------|---------|---------|
| `discovery.json` | 15 MB | 3,839,391 | Full schema + samples + views SQL |
| `audit_metrics.json` | 307 KB | 78,535 | Table access patterns, hot/cold |
| `semantic_model.json` | 20 KB | 5,060 | Entities/dimensions/facts (incomplete) |

### Current Semantic Model: Token Waste

Looking at a single entity (Address) in `semantic_model_incremental.json`:

```json
{
  "name": "ID",
  "role": "primary_key",           // ❌ NOT USED by sql_generator
  "semantic_type": "id",           // ❌ NOT USED
  "aliases": ["AddressID", "Address ID"],  // ❌ NOT USED
  "description": "Unique identifier..."    // ❌ NOT USED
}
```

**Per column: ~50 tokens generated, ~5 tokens actually used (just the name)**

For Address entity with 45 columns:
- Generated: ~2,250 tokens
- Actually used: ~225 tokens
- **Waste: 90%**

### What SQL Generator Actually Uses

From `sql_generator.py` lines 426-459:

```python
# USED:
obj.get('name')                    # Entity/dimension/fact name
obj.get('source')                  # Table source (dbo.TableName)
[c.get('name') for c in columns]   # Column names ONLY
m.get('name'), m.get('expression') # Measure name and expression

# NOT USED:
columns[].role
columns[].semantic_type
columns[].aliases
columns[].description
display.display_name
display.default_label_column
display.default_search_columns
display.default_sort
```

---

## Architecture Decision: Single File vs Multiple Files

### Option A: Everything in Semantic Model (Single File)

```
semantic_model.json
├── entities (minimal)
├── dimensions (minimal)
├── facts (minimal)
├── relationships
├── sql_templates (NEW)
├── verified_joins (NEW)
├── column_index (NEW)
├── audit_summary (hot/warm/cold lists)
└── table_rankings
```

**Pros:**
- Single file to load at Q&A time
- All context in one place
- Simpler code in sql_generator.py

**Cons:**
- File gets large with templates
- Must rebuild entire file when any source changes
- Duplicates data (audit data copied into semantic)

### Option B: Multiple Files at Q&A Time

```
Q&A loads:
├── semantic_model.json (minimal schema)
├── sql_templates.json (NEW - extracted templates)
├── audit_metrics.json (existing)
└── discovery.json (only if needed for fallback)
```

**Pros:**
- Each file serves one purpose
- Can update templates without rebuilding semantic
- Audit stays fresh (recollect without semantic rebuild)
- Smaller semantic model file

**Cons:**
- Multiple file loads
- Must coordinate data between files
- More complex sql_generator.py

### Option C: Hybrid (RECOMMENDED)

```
semantic_model.json (compact)
├── entities: [{name, source, columns: [name only]}]
├── dimensions: [{name, source, keys, columns: [name only]}]
├── facts: [{name, source, measures: [{name, expression}]}]
├── relationships: [{from, to, confidence}]
└── metadata: {discovery_hash, audit_hash}

sql_templates.json (NEW - extracted from RDL/Views)
├── templates: [{id, keywords, tables, sql, columns}]
└── template_index: {keyword -> template_ids}

audit_metrics.json (existing - unchanged)
├── table_metrics: [{full_name, access_pattern, access_score}]
├── join_frequency: {join_pattern -> count}
└── tables_to_prioritize, tables_to_skip
```

**At Q&A time:**
1. Load `semantic_model.json` (small, always needed)
2. Load `sql_templates.json` (search for matching template)
3. Load `audit_metrics.json` (for hot/cold prioritization)
4. **DO NOT** load `discovery.json` (too large, not needed)

---

## Recommended Data Architecture

### semantic_model.json (Target: <100KB)

**Simplify to only what's used:**

```json
{
  "entities": [
    {
      "name": "Address",
      "source": "dbo.Address",
      "primary_key": ["ID"],
      "columns": ["ID", "Line1", "Line2", "CountryID", ...]  // Just names!
    }
  ],
  "dimensions": [
    {
      "name": "Date",
      "source": "dbo.DimDate",
      "keys": ["DateKey"],
      "columns": ["DateKey", "Year", "Month", "Quarter"]
    }
  ],
  "facts": [
    {
      "name": "Sales",
      "source": "dbo.FactSales",
      "grain": ["DateKey", "CustomerKey"],
      "measures": [
        {"name": "Revenue", "expression": "SUM(Amount)"},
        {"name": "Units", "expression": "COUNT(*)"}
      ]
    }
  ],
  "relationships": [
    {"from": "Sales.CustomerKey", "to": "Customer.ID", "confidence": "high"}
  ]
}
```

**Token reduction: ~90%**

### sql_templates.json (NEW - Target: ~500KB)

```json
{
  "templates": [
    {
      "id": "contracts_by_campaign",
      "name": "Contracts per Campaign",
      "source_type": "rdl",
      "source_file": "Report Παρασκευής - Contracts Campaigns.rdl",
      "keywords": ["contract", "campaign", "συμβόλαιο", "καμπάνια", "sales"],
      "tables": ["dbo.vwContractsPerCampaign", "dbo.TargetGroupItemValue"],
      "sql": "SELECT vw.[Καμπάνια], vw.[Αξία] FROM dbo.vwContractsPerCampaign vw...",
      "columns_selectable": ["Καμπάνια", "Αξία", "Επωνυμία", "Πωλητής"],
      "columns_filterable": ["CampaignID", "Ημ. Δημιουργίας"],
      "joins": ["TargetGroupItemValue.ContractID = vw.Κωδ. Συμβολαίου"]
    }
  ],
  "index": {
    "contract": ["contracts_by_campaign", "contract_history"],
    "campaign": ["contracts_by_campaign", "campaign_summary"],
    "συμβόλαιο": ["contracts_by_campaign"]
  }
}
```

### audit_metrics.json (Keep as-is)

Already well-structured with:
- `table_metrics[]` - hot/warm/cold classification
- `join_frequency{}` - which joins are used most
- `tables_to_prioritize[]`, `tables_to_skip[]`

---

## Summary: What to Build

| Component | Action | Size Target |
|-----------|--------|-------------|
| `semantic_model.json` | Simplify (remove unused fields) | <100 KB |
| `sql_templates.json` | **NEW** - Extract from RDL/Views | ~500 KB |
| `audit_metrics.json` | Keep as-is | ~300 KB |
| `discovery.json` | Keep for rebuild, don't load at Q&A | 15 MB |

**Q&A Load Total: ~900 KB** (vs current ~15 MB if loading discovery)

---

## Token Flow at Q&A Time

```
User Question: "Show contracts by campaign this month"
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. TEMPLATE SEARCH (sql_templates.json ~500KB)              │
│    Keywords: ["contract", "campaign", "month"]              │
│    Match: "contracts_by_campaign" (score: 0.85)             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. AUDIT CHECK (audit_metrics.json ~300KB)                  │
│    vwContractsPerCampaign: HOT (access_score: 85)           │
│    TargetGroupItemValue: HOT (access_score: 55)             │
│    → Tables verified as actively used                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM ADAPTATION (semantic_model.json ~100KB for context)  │
│                                                             │
│    Prompt: "Adapt this template for 'this month'"           │
│    Template SQL: SELECT vw.[Καμπάνια]... WHERE ...          │
│    Available columns: [from template]                       │
│    → LLM only modifies WHERE clause                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    Grounded SQL
```

**LLM input tokens: ~2,000** (template + column list + question)
**vs Current: ~10,000+** (full schema context)

---

## Column Accuracy: The Core Problem

### Current State: Columns Lack Context

The semantic model generates useless metadata:

```json
// CURRENT - wasteful, not useful for matching
{
  "name": "CampaignID",
  "role": "foreign_key",
  "semantic_type": "id",
  "aliases": ["Campaign ID"],
  "description": "Foreign key to Campaign"
}
```

But discovery.json has **real data**:

```json
// DISCOVERY - actual column data
{
  "name": "CampaignID",
  "type": "INTEGER",
  "stats": {
    "distinct_count": 66,
    "min": "111",
    "max": "192",
    "sample_values": ["192", "138", "175"]
  }
}
```

### What We Need: Accurate Column Fingerprints

For Q&A to work without hallucination, each column needs:

1. **Data type** - INTEGER, VARCHAR, DATE, etc.
2. **Sample values** - Real examples from the database
3. **Value range** - Min/max for numerics, patterns for strings
4. **Cardinality** - How many distinct values (is it a lookup vs free text?)
5. **FK target** - If it's a foreign key, what table does it reference?
6. **Business meaning** - What does this column actually represent?

---

## Proposed: Column Fingerprint Schema

### New Column Structure

```json
{
  "name": "CampaignID",
  "type": "INTEGER",
  "is_pk": false,
  "is_fk": true,
  "fk_target": "dbo.Campaign.ID",
  "cardinality": 66,
  "sample_values": ["192", "138", "175", "172", "155"],
  "value_range": {"min": 111, "max": 192},
  "business_terms": ["campaign", "καμπάνια"],
  "used_in_filters": true,
  "used_in_groupby": true,
  "used_in_joins": true
}
```

### Where Data Comes From

| Field | Source |
|-------|--------|
| `name`, `type` | discovery.json (introspection) |
| `is_pk`, `is_fk`, `fk_target` | discovery.json (foreign_keys) |
| `cardinality`, `sample_values`, `value_range` | discovery.json (stats) |
| `business_terms` | Column name parsing + RDL field labels |
| `used_in_filters/groupby/joins` | RDL/View SQL analysis + audit |

---

## Accurate Matching Strategy

### Problem: User says "campaign", which column?

Multiple tables have campaign-related columns:

- `Contract.CampaignID` (FK to Campaign)
- `Campaign.ID` (PK)
- `Campaign.Name` (display name)
- `vwContractsPerCampaign.Καμπάνια` (Greek label)

### Solution: Multi-Signal Matching

```
User Question: "contracts by campaign this month"
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. KEYWORD EXTRACTION                                       │
│    Terms: ["contracts", "campaign", "month"]                │
│    Greek: ["συμβόλαιο", "καμπάνια", "μήνας"]               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. COLUMN MATCHING (from column fingerprints)               │
│                                                             │
│    "campaign" matches:                                      │
│    ├── Campaign.Name (business_term match) -> DISPLAY       │
│    ├── Contract.CampaignID (fk_target: Campaign) -> JOIN    │
│    └── vwContractsPerCampaign.Καμπάνια -> GREEK LABEL      │
│                                                             │
│    "contracts" matches:                                     │
│    ├── Contract table                                       │
│    └── vwContractsPerCampaign view                         │
│                                                             │
│    "month" matches:                                         │
│    ├── Contract.CreateDate (type: DATE)                     │
│    └── Contract.SignDate (type: DATE)                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TEMPLATE SELECTION (prefer RDL/View with these columns)  │
│                                                             │
│    Best match: "contracts_by_campaign" template             │
│    Source: vwContractsPerCampaign (already has JOINs)       │
│    Columns verified: Καμπάνια, Ημ. Δημιουργίας             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. SQL GENERATION (constrained to template + columns)       │
│                                                             │
│    SELECT vw.[Καμπάνια], COUNT(*) as ContractCount          │
│    FROM dbo.vwContractsPerCampaign vw                       │
│    WHERE vw.[Ημ. Δημιουργίας] >= DATEADD(MONTH,-1,GETDATE())│
│    GROUP BY vw.[Καμπάνια]                                   │
│                                                             │
│    Table verified (from template)                           │
│    Columns verified (from fingerprints)                     │
│    Date filter pattern (from template)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Files Architecture (Revised)

### 1. semantic_model.json (~200KB)

**Focus: Schema + Column Fingerprints**

```json
{
  "tables": [
    {
      "name": "Contract",
      "source": "dbo.Contract",
      "type": "entity",
      "primary_key": ["ID"],
      "access_pattern": "hot",
      "columns": [
        {
          "name": "ID",
          "type": "INTEGER",
          "is_pk": true,
          "cardinality": 500000,
          "business_terms": ["contract id", "κωδικός συμβολαίου"]
        },
        {
          "name": "CampaignID",
          "type": "INTEGER",
          "is_fk": true,
          "fk_target": "dbo.Campaign.ID",
          "cardinality": 200,
          "sample_values": [111, 138, 155, 172, 192],
          "business_terms": ["campaign", "καμπάνια"]
        },
        {
          "name": "Amount",
          "type": "DECIMAL(18,2)",
          "is_measure": true,
          "value_range": {"min": 0, "max": 50000},
          "business_terms": ["amount", "revenue", "αξία", "ποσό"]
        },
        {
          "name": "CreateDate",
          "type": "DATETIME",
          "is_date": true,
          "business_terms": ["created", "date", "ημερομηνία"]
        }
      ]
    }
  ],
  "relationships": [
    {
      "from": "Contract.CampaignID",
      "to": "Campaign.ID",
      "type": "many_to_one",
      "confidence": "very_high",
      "source": "fk"
    }
  ],
  "column_index": {
    "campaign": ["Campaign.ID", "Campaign.Name", "Contract.CampaignID"],
    "καμπάνια": ["Campaign.Name", "vwContractsPerCampaign.Καμπάνια"],
    "amount": ["Contract.Amount", "ContractItem.Amount"],
    "date": ["Contract.CreateDate", "Contract.SignDate"]
  }
}
```

### 2. sql_templates.json (~500KB)

**Focus: Verified SQL Patterns**

```json
{
  "templates": [
    {
      "id": "contracts_by_campaign",
      "name": "Contracts by Campaign",
      "source": "rdl:Report Παρασκευής - Contracts Campaigns.rdl",
      "keywords": ["contract", "campaign", "συμβόλαιο", "καμπάνια"],
      "intent_patterns": [
        "contracts by campaign",
        "contracts per campaign",
        "συμβόλαια ανά καμπάνια"
      ],
      "tables": ["dbo.vwContractsPerCampaign"],
      "columns": {
        "selectable": ["Καμπάνια", "Κωδ. Πελάτη", "Επωνυμία", "Αξία", "Πωλητής"],
        "filterable": ["CampaignID", "Ημ. Υπογραφής", "Ημ. Δημιουργίας"],
        "groupable": ["Καμπάνια", "Πωλητής"],
        "aggregatable": ["Αξία"]
      },
      "sql_template": "SELECT {columns} FROM dbo.vwContractsPerCampaign vw WHERE {filters}",
      "sql_example": "SELECT vw.[Καμπάνια], SUM(vw.[Αξία]) FROM dbo.vwContractsPerCampaign vw GROUP BY vw.[Καμπάνια]"
    }
  ]
}
```

### 3. audit_metrics.json (~300KB)

**Keep as-is** - Already has table_metrics, join_frequency, tables_to_prioritize

---

## Consistency: Same Question = Same Result

### Problem: LLM generates different SQL each time

```
Q: "show contracts by campaign"

Run 1: SELECT Campaign, COUNT(*) FROM Contract...     (wrong column)
Run 2: SELECT CampaignID, COUNT(*) FROM Contract...   (not readable)
Run 3: SELECT c.Name, COUNT(*) FROM Contract JOIN...  (different each time)
```

### Solution: Deterministic Template Selection

```python
def generate_sql(question: str) -> SQL:
    # Step 1: Extract keywords (deterministic - no LLM)
    keywords = extract_keywords(question)

    # Step 2: Find matching template (deterministic - best score wins)
    template = find_best_template(keywords)

    # Step 3: If template found, use it (constrained)
    if template.score > 0.8:
        return adapt_template(template, question)

    # Step 4: Fallback to schema-based (lower confidence)
    return generate_from_schema(question)
```

### Template Adaptation Rules (Constrained LLM)

```
RULES FOR LLM:
1. DO NOT change table names - use exactly: {template.tables}
2. DO NOT change column names - use only: {template.columns}
3. DO NOT invent JOINs - template already has correct joins
4. ONLY modify:
   - SELECT: choose from {template.columns.selectable}
   - WHERE: use {template.columns.filterable} with user criteria
   - GROUP BY: use {template.columns.groupable} if aggregating
   - ORDER BY: use any selected column
```

---

## Implementation Priority

### Phase 1: Column Fingerprints (HIGH IMPACT)

1. Extract `sample_values`, `cardinality`, `value_range` from discovery.json
2. Build `column_index` mapping business terms to columns
3. Detect FK relationships and store `fk_target`
4. Store in simplified semantic_model.json

### Phase 2: SQL Templates (HIGH IMPACT)

1. Extract SQL from RDL files (94 reports)
2. Extract SQL from Views (233 views)
3. Parse tables/columns used in each template
4. Index by keywords for matching

### Phase 3: Deterministic Matching (CONSISTENCY)

1. Keyword extraction (rule-based, not LLM)
2. Template scoring (TF-IDF or exact match)
3. Column validation (verify columns exist)
4. Constrained LLM adaptation

---


## Engineering Principles

- **KISS:** simplest working solution.
- **DRY:** reuse logic; avoid duplication.
- **YAGNI:** no unnecessary abstractions.
- **Minimal impact:** apply only changes explicitly required.


## Next Steps

1. ~~**Create column fingerprint extractor** - Pull real data from discovery.json~~
2. ~~**Build column_index** - Map business terms to actual columns~~
3. ~~**Create sql_templates.py** - Extract templates from RDL/Views~~
4. ~~**Update sql_generator.py** - Template matching + constrained adaptation~~
5. **Test** - Same question must return same SQL
6. **Update model_builder.py** - Simplify schema output (optional - lower priority)

---

## Implementation Status (December 2024)

### Completed

#### 1. Column Fingerprints Module (`src/semantic/column_fingerprints.py`)

New module that extracts real column data from discovery.json:

```python
# Usage
python main.py extract-fingerprints

# Output: cache/column_fingerprints.json
```

Features:
- Extracts `sample_values`, `cardinality`, `value_range` from discovery stats
- Builds `column_index` mapping business terms to `schema.table.column`
- Detects FK relationships from `inferred_relationships`
- Supports Greek/English term mapping

#### 2. SQL Templates Module (`src/semantic/sql_templates.py`)

New module that extracts verified SQL patterns:

```python
# Usage
python main.py extract-templates

# Output: cache/sql_templates.json
```

Features:
- Extracts templates from Views (complex, with JOINs)
- Extracts templates from RDL datasets
- Builds keyword index for template matching
- Filters out simple lookup views (Attr* views)

#### 3. Updated SQL Generator (`src/qa/sql_generator.py`)

Enhanced with template-based generation:

**New Flow:**
1. Extract keywords deterministically (NO LLM) - same Q = same keywords
2. Match against SQL templates by keywords
3. If match found (confidence >= 0.6): LLM adapts template
4. Else: Fall back to schema-based generation

**Key Additions:**
- `_extract_keywords_deterministic()` - Deterministic keyword extraction
- `_find_matching_template()` - Template matching without LLM
- `_generate_from_template()` - Constrained LLM adaptation
- Temporal keyword detection (this_month, last_year, etc.)
- Greek keyword support

**Response now includes:**
```json
{
  "generation_method": "template" | "schema",
  "evidence": {
    "template_used": "...",
    "template_source": "view:dbo.CampaignAnalysisView"
  }
}
```

### Usage Workflow

```bash
# 1. Run discovery (if not done)
python main.py discovery

# 2. Extract column fingerprints
python main.py extract-fingerprints

# 3. Extract SQL templates
python main.py extract-templates

# 4. Build semantic model
python main.py model

# 5. Ask questions (now uses templates!)
python main.py question "Show contracts by campaign"
```

### What's Changed

| Before | After |
|--------|-------|
| LLM invents SQL from scratch | LLM adapts verified templates |
| Random table/column hallucination | Tables/columns locked to template |
| Non-deterministic (same Q = different SQL) | Deterministic keyword matching |
| Low confidence (~60%) | High confidence (~90%) for template matches |
| No Greek keyword support | Greek/English keyword mapping |

### Remaining Work

1. **Testing** - Verify same question returns same SQL
2. **Model Builder Update** - Optional simplification of semantic model output
3. **Template Curation** - Manual review/enhancement of extracted templates
4. **Keyword Expansion** - Add more Greek business terms

---

## Simplification Decision: Remove column_fingerprints.py

### Analysis

After implementation review, `column_fingerprints.py` is **redundant**:

| What It Does | Actually Used? | Better Alternative |
|--------------|----------------|-------------------|
| Extracts sample_values, cardinality | ❌ Not used in sql_generator | Already in discovery.json |
| Extracts FK relationships | ❌ Not used | Already in discovery.json |
| Builds column_index (term → column) | ⚠️ Loaded but not actively used | Merge into sql_templates.py |

### Current State (Redundant)

```
discovery.json (15MB) ──► column_fingerprints.py ──► column_fingerprints.json
                                                           │
                                                           ▼
                                                    (mostly unused)
```

### Proposed: Single File Architecture

```
discovery.json (15MB) ──► sql_templates.py ──► sql_templates.json
                              │
                              ├── templates[] (from Views/RDL)
                              ├── keyword_index{} (for template matching)
                              └── column_index{} (for term→column, MERGED)
```

### Benefits

1. **One file to extract**: `python main.py extract-templates`
2. **One file to load at Q&A**: `sql_templates.json`
3. **Less code to maintain**: Remove entire module
4. **No redundant data**: discovery.json is the source of truth

### Implementation (COMPLETED)

1. ✅ Delete `src/semantic/column_fingerprints.py`
2. ✅ Remove `extract-fingerprints` CLI command
3. ✅ Update `sql_templates.py` to include column_index extraction
4. ✅ Update `sql_generator.py` to only load templates (includes column_index)
5. ✅ Update REFACTORING_PLAN.md
