# 🧠 CRM-Aware Semantic Database RAG System — **OpenAI-Powered NER with Zero Hallucinations**

> A **production-grade CRM-aware** Text-to-SQL system that combines **OpenAI-powered Named Entity Recognition** with **strict capability contracts** and **zero schema hallucinations**. Leverages existing views, foreign keys, and stored procedures to understand relationships and never executes unsafe queries.

---

## 🎯 Core Principles

### **No-Hallucination Operating Rules**
- **Never execute a query unless the schema can prove it supports the request**
- **OpenAI-powered intent parsing and Named Entity Recognition** for accurate CRM entity identification
- **Leverage existing database artifacts**: Foreign keys, views, and stored procedures provide relationship truth
- **Correctness over convenience**: never sum the wrong column or join the wrong dimension
- **Gate every query behind three validations**: Identifier gate, Relationship gate, Capability gate
- **Non-Executable Analysis Reports (NEAR)** when capability is missing—no "best-guess" queries
- **Business-friendly results**: neat columns, explicit units, meaningful labels, reproducible SQL

### **Universal CRM Capability Contract**
For any CRM analytics question, require ALL of the following before generation/execution:

✅ **Grain**: Row grain of the fact table (opportunity, invoice_line, case, activity, campaign_member)  
✅ **Measure(s)**: Numeric column(s) compatible with the asked metric (revenue, duration, count, score)  
✅ **Time**: Usable timestamp/date for filtering and bucketing (CloseDate, CreatedDate, ActivityDate)  
✅ **Entity key(s)**: The CRM entity for grouping (AccountId, ContactId, OwnerId, ProductId)  
✅ **Join path(s)**: Proven joins from fact → dimension via FK, views, or stored procedure patterns  
✅ **Filters & Status**: Columns for explicit filters or status flags (IsWon, IsClosed, Priority, Stage)  
✅ **Quality minima**: Row count > 0, null-rate thresholds, optional data-freshness check  

**If ANY item fails → produce NEAR instead of executing**

---

## 🏗️ Architecture

```
crm-semantic-rag/
├── main.py                         # CLI entry point with three-phase workflow
├── shared/
│   ├── config.py                   # OpenAI + database configuration
│   ├── models.py                   # CRM-enhanced dataclasses with capability contracts
│   └── utils.py                    # SQL validation, formatting, safety utilities
├── discovery/
│   └── discovery.py                # Schema discovery + FK analysis + view/SP parsing
├── semantic/
│   └── analysis.py                 # CRM-aware semantic analysis + capability assessment
├── interactive/
│   └── query_interface.py          # OpenAI NER + intent + capability-gated SQL generation
└── data/
    ├── database_structure.json     # Canonical schema cache (tables + FKs + views/SPs)
    ├── semantic_analysis.json      # CRM semantic classification + capability scores
    └── query_patterns.json         # Validated successful query patterns
```

---

## 🧠 OpenAI-Powered Intent + Named Entity Recognition

### **Structured Intent Analysis**
Replace heuristic pattern-matching with OpenAI structured output using JSON schema:

```json
{
  "type": "object",
  "properties": {
    "task_type": { 
      "enum": ["aggregation", "ranking", "trend", "distribution", "funnel", "comparison", "drill_down"] 
    },
    "metrics": { 
      "type": "array", 
      "items": { "type": "string" },
      "description": "revenue, pipeline, cases, activities, response_time, conversion_rate"
    },
    "crm_entities": { 
      "type": "array", 
      "items": { "type": "string" },
      "description": "account, contact, opportunity, lead, case, campaign, product, territory, owner"
    },
    "time_window": { 
      "type": "string",
      "description": "2025, Q2_2025, last_12_months, YTD, this_quarter"
    },
    "group_by": { 
      "type": "array", 
      "items": { "type": "string" },
      "description": "account, owner, region, product_line, stage, priority"
    },
    "filters": { 
      "type": "array", 
      "items": { "type": "string" },
      "description": "won_opportunities, high_priority, enterprise_accounts, active_campaigns"
    },
    "limit": { "type": "integer", "minimum": 1, "maximum": 500 }
  },
  "required": ["task_type", "metrics", "crm_entities"]
}
```

### **CRM Entity Resolution with Schema Context**
OpenAI receives discovered schema context to map entities to actual tables:

```python
def resolve_crm_entities(llm, question: str, schema_context: dict) -> EntityResolution:
    """
    Uses OpenAI to map user entities to schema objects with confidence scores
    
    Args:
        schema_context: {
            "tables": [{"name": "Opportunities", "columns": [...], "fks": [...]}],
            "views": [{"name": "vw_PipelineMetrics", "referenced_objects": [...]}],
            "synonym_registry": {"pipeline": "Opportunity.Amount", "bookings": "InvoiceLine.NetAmount"}
        }
    
    Returns:
        EntityResolution with ranked candidate facts and supporting dimensions
    """
```

### **Few-Shot CRM Examples**
```python
CRM_EXAMPLES = [
    {
        "question": "top 10 accounts by revenue in Q2 2025",
        "intent": {
            "task_type": "ranking",
            "metrics": ["revenue"],
            "crm_entities": ["account"],
            "time_window": "Q2_2025",
            "limit": 10
        }
    },
    {
        "question": "monthly case volume by priority last 12 months",
        "intent": {
            "task_type": "trend",
            "metrics": ["case_count"],
            "crm_entities": ["case"],
            "group_by": ["priority"],
            "time_window": "last_12_months"
        }
    },
    {
        "question": "average sales cycle length by region for won opportunities",
        "intent": {
            "task_type": "aggregation",
            "metrics": ["sales_cycle_days"],
            "crm_entities": ["opportunity"],
            "group_by": ["region"],
            "filters": ["won_opportunities"]
        }
    }
]
```

---

## 🔍 CRM-Aware Semantic Analysis

### **Enhanced Table Classification**
Beyond basic entity types, classifies CRM-specific patterns:

**CRM Data Types:**
- **Transactional**: Real business events (Opportunities, Cases, Activities, Orders)
- **Master Data**: Core entities (Accounts, Contacts, Products, Users)
- **Configuration**: Settings, picklists, territories, campaigns
- **Analytical**: Views, rollup tables, calculated metrics

**CRM Table Roles:**
- **Fact Tables**: Contains measures and events (OpportunityLineItem, CaseHistory, ActivityLog)
- **Dimension Tables**: Descriptive attributes (Accounts, Products, Territories)
- **Bridge Tables**: Many-to-many relationships (CampaignMembers, OpportunityTeamMembers)
- **Lookup Tables**: Static reference data (Stages, Priorities, RecordTypes)

**CRM Capability Assessment:**
- **Revenue Measures**: Amount, TotalPrice, AnnualRevenue, ExpectedRevenue
- **Activity Measures**: Count, Duration, ResponseTime, ConversionRate
- **CRM Entity Keys**: AccountId, ContactId, OwnerId, ProductId, TerritoryId
- **CRM Time Columns**: CloseDate, CreatedDate, LastActivityDate, DueDate
- **CRM Status Filters**: IsWon, IsClosed, Priority, Stage, RecordType

### **Relationship Discovery via Database Artifacts**
Leverage existing database knowledge sources:

1. **Foreign Key Constraints**: Primary source of truth for relationships
2. **View Definitions**: Parse SELECT statements to find implicit joins
3. **Stored Procedure Analysis**: Extract JOIN patterns from procedure bodies
4. **Naming Conventions**: AccountId → Accounts.Id, ContactId → Contacts.Id

```python
def discover_relationships(connection) -> RelationshipGraph:
    """
    Builds comprehensive relationship graph from multiple sources:
    1. FK constraints (highest confidence)
    2. View JOIN patterns (high confidence)
    3. Stored procedure JOIN patterns (medium confidence) 
    4. Naming convention inference (low confidence, validation required)
    """
```

---

## 🚫 No-Fallback Query Pipeline with OpenAI

### **Stage 1: OpenAI Intent + NER**
```python
async def analyze_intent_with_openai(question: str) -> AnalyticalTask:
    """
    Uses OpenAI structured output to parse natural language into:
    - Task type (ranking, trend, aggregation, etc.)
    - CRM metrics (revenue, pipeline, case_count, response_time)
    - CRM entities (account, opportunity, case, contact)
    - Time windows and filters
    """
```

### **Stage 2: Schema-Aware Entity Resolution**
```python
async def resolve_entities_with_schema(task: AnalyticalTask, schema: DatabaseSchema) -> List[CandidateFact]:
    """
    Uses OpenAI + schema context to map logical entities to physical tables:
    - "revenue" → OpportunityLineItem.TotalPrice or InvoiceLine.NetAmount
    - "account" → Accounts table with proper FK paths
    - "pipeline" → Opportunities with Stage != 'Closed Won'
    """
```

### **Stage 3: Capability Contract Validation**
For each candidate fact table, validate the complete contract:
```python
def validate_capability_contract(fact: CandidateFact, task: AnalyticalTask) -> CapabilityAssessment:
    """
    Validates ALL required capabilities:
    ✓ Grain matches task (opportunity-level for pipeline, line-level for revenue)
    ✓ Measures exist and are numeric (Amount, Quantity, Duration)
    ✓ Time column available for filtering (CloseDate, CreatedDate)
    ✓ Entity keys available for grouping (AccountId, OwnerId)
    ✓ Join paths proven via FK/view/SP analysis
    ✓ Filter columns available (Stage, Priority, IsWon)
    ✓ Quality thresholds met (row count, null rates)
    """
```

### **Stage 4: Evidence-Driven Selection**
Only capability-validated tables are scored:
```python
def score_candidate_facts(candidates: List[CandidateFact]) -> List[ScoredCandidate]:
    """
    Weighted evidence scoring:
    1. CRM Role Match (40%): Transactional fact with measures + time + FKs
    2. Join Strength (25%): FK constraints > view patterns > SP patterns > naming
    3. Measure Fitness (20%): Exact metric match > compatible aggregation
    4. Time Fitness (10%): Exact time semantics > compatible time column
    5. Data Quality (5%): Row count, freshness, null rates
    """
```

### **Stage 5: Validated SQL Generation OR NEAR**
**If capable facts found**: Generate SQL using only proven schema elements  
**If NO capable facts**: Return **Non-Executable Analysis Report** with:

```json
{
  "question": "average deal size by territory in Q2 2025",
  "normalized_task": {
    "task_type": "aggregation",
    "metrics": ["deal_size"],
    "crm_entities": ["opportunity", "territory"],
    "time_window": "Q2_2025"
  },
  "missing_capabilities": [
    "No proven join path from Opportunities to Territories",
    "Deal size measure requires OpportunityLineItem aggregation"
  ],
  "top_candidate_facts": [
    {
      "table": "Opportunities",
      "evidence_score": 0.7,
      "missing": ["territory_join", "line_item_aggregation"]
    }
  ],
  "fix_paths": [
    "Add TerritoryId foreign key to Opportunities table",
    "Use OpportunityLineItem.TotalPrice for accurate deal sizing",
    "Create view vw_OpportunityMetrics with territory joins"
  ],
  "safe_exploration": [
    "SELECT TOP 5 * FROM Opportunities WHERE CloseDate >= '2025-04-01'",
    "SELECT COUNT(*), MIN(Amount), MAX(Amount) FROM Opportunities"
  ]
}
```

---

## 🔗 Leveraging Database Artifacts for Relationships

### **Foreign Key Analysis**
```sql
-- Discover all FK relationships
SELECT 
    fk.name as constraint_name,
    tp.name as parent_table,
    cp.name as parent_column,
    tr.name as referenced_table,
    cr.name as referenced_column
FROM sys.foreign_keys fk
INNER JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
INNER JOIN sys.tables tr ON fk.referenced_object_id = tr.object_id
INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
INNER JOIN sys.columns cp ON fkc.parent_column_id = cp.column_id AND fkc.parent_object_id = cp.object_id
INNER JOIN sys.columns cr ON fkc.referenced_column_id = cr.column_id AND fkc.referenced_object_id = cr.object_id
```

### **View Definition Parsing**
```python
def parse_view_relationships(view_definition: str) -> List[JoinPattern]:
    """
    Parse SQL view definitions to extract JOIN patterns:
    - FROM table1 t1 JOIN table2 t2 ON t1.id = t2.foreign_id
    - Builds relationship graph from observed patterns
    """
    # Use SQLGlot to parse view SQL and extract JOIN conditions
```

### **Stored Procedure Analysis**
```python
def extract_sp_join_patterns(sp_body: str) -> List[JoinPattern]:
    """
    Extract JOIN patterns from stored procedure bodies:
    - Common CRM patterns: Account → Contact → Opportunity
    - Activity rollups: Account → Activities summary
    - Pipeline reports: Opportunity → Product → Territory
    """
```

---

## 🎯 CRM-Specific SQL Generation Examples

### **A) Account Revenue Ranking**
```sql
-- Generated from: "top 10 accounts by revenue in 2025"
SELECT TOP 10
    a.Name AS account_name,
    a.Type AS account_type,
    SUM(oli.TotalPrice) AS total_revenue,
    COUNT(DISTINCT o.Id) AS deal_count,
    COUNT(oli.Id) AS line_item_count
FROM Accounts a
    JOIN Opportunities o ON o.AccountId = a.Id
    JOIN OpportunityLineItems oli ON oli.OpportunityId = o.Id
WHERE o.CloseDate >= '2025-01-01' 
    AND o.CloseDate < '2026-01-01'
    AND o.IsWon = 1
GROUP BY a.Id, a.Name, a.Type
ORDER BY total_revenue DESC;
```

### **B) Case Resolution Trends**
```sql
-- Generated from: "monthly case resolution time by priority last 12 months"
SELECT 
    DATEFROMPARTS(YEAR(c.CreatedDate), MONTH(c.CreatedDate), 1) AS month_start,
    c.Priority,
    COUNT(*) AS case_count,
    AVG(DATEDIFF(hour, c.CreatedDate, c.ClosedDate)) AS avg_resolution_hours
FROM Cases c
WHERE c.CreatedDate >= DATEADD(month, -12, GETDATE())
    AND c.IsClosed = 1
    AND c.ClosedDate IS NOT NULL
GROUP BY DATEFROMPARTS(YEAR(c.CreatedDate), MONTH(c.CreatedDate), 1), c.Priority
ORDER BY month_start, c.Priority;
```

### **C) Sales Pipeline Analysis**
```sql
-- Generated from: "pipeline value by stage and owner for enterprise accounts"
SELECT 
    u.Name AS owner_name,
    o.StageName AS stage,
    COUNT(*) AS opportunity_count,
    SUM(o.Amount) AS total_pipeline_value,
    AVG(DATEDIFF(day, o.CreatedDate, GETDATE())) AS avg_age_days
FROM Opportunities o
    JOIN Users u ON u.Id = o.OwnerId
    JOIN Accounts a ON a.Id = o.AccountId
WHERE o.IsClosed = 0
    AND a.Type = 'Enterprise'
GROUP BY u.Id, u.Name, o.StageName
ORDER BY total_pipeline_value DESC;
```

---

## ⚙️ Configuration

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o  # or gpt-4o-mini for cost optimization
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT_SECS=60
OPENAI_MAX_RETRIES=3

# Database Configuration  
DATABASE_CONNECTION_STRING=your_crm_database_connection
DATABASE_SCHEMA=dbo  # or your CRM schema name

# CRM-Aware Features
ENABLE_OPENAI_NER=true
ENABLE_CAPABILITY_CONTRACTS=true
ENABLE_EVIDENCE_SELECTION=true
DISABLE_QUERY_FALLBACKS=true
USE_DATABASE_ARTIFACTS=true  # FKs, views, SPs for relationships

# Capability Thresholds
MIN_FACT_ROW_COUNT=100
MAX_NULL_RATE_FOR_MEASURES=0.25
EVIDENCE_SCORE_THRESHOLD=0.65
CAPABILITY_COMPLETENESS_THRESHOLD=0.85

# Output Configuration
MAX_RESULTS_DEFAULT=100
DECIMAL_PLACES=2
SHOW_GENERATED_SQL=true
INCLUDE_EXECUTION_STATS=true
```

---

## 🛡️ Three-Gate Security Validation

### **Gate 1: Identifier Validation**
- SQL must reference only discovered, allow-listed objects
- SQLGlot AST parsing with schema object existence checks
- No dynamic identifier construction or SQL injection vectors

### **Gate 2: Relationship Validation**  
- All JOINs must exist in FK graph OR observed in views/SPs
- No arbitrary table joins based on name similarity
- Relationship confidence scoring (FK > view > SP > naming)

### **Gate 3: Capability Validation**
- Selected fact/dimension tables must satisfy metric requirements
- Grain compatibility (opportunity-level vs line-level)
- Required columns present and of correct data types
- Quality thresholds met (row counts, null rates)

---

## 🚀 Getting Started

### **1. Environment Setup**
```bash
# Clone and configure
cp env_example.txt .env
# Edit .env with your OpenAI key and database connection

# Install dependencies
pip install openai pyodbc sqlglot pandas
```

### **2. Three-Phase Workflow**

```bash
python main.py

# Phase 1: Database Discovery
# - Discovers tables, columns, data types
# - Analyzes foreign key relationships  
# - Parses view definitions and stored procedures
# - Caches results in database_structure.json

# Phase 2: CRM Semantic Analysis  
# - Classifies tables by CRM role (fact/dimension/bridge)
# - Identifies measures, entity keys, time columns
# - Assesses capability contracts for each table
# - Caches results in semantic_analysis.json

# Phase 3: Interactive Querying
# - OpenAI-powered intent parsing and NER
# - Evidence-driven table selection
# - Capability-gated SQL generation
# - Business-friendly result formatting
```

### **3. Example Queries**

```
🧠 CRM Analytics Assistant Ready

❓ Query: "What are our top performing sales reps by closed won revenue this quarter?"

🔍 OpenAI Analysis:
   📊 Intent: ranking task, revenue metric, sales_rep entity, this_quarter time
   🏷️  NER: sales_rep → Users (Owner), revenue → OpportunityLineItem.TotalPrice
   
🧠 Capability Validation:
   ✅ OpportunityLineItem (fact): has TotalPrice measure, CreatedDate time
   ✅ Opportunities (bridge): has OwnerId key, CloseDate time, IsWon filter  
   ✅ Users (dimension): has Name for grouping
   ✅ Proven joins: OLI→Opportunity→User via FKs

📋 Generated SQL:
SELECT TOP 10
    u.Name AS sales_rep_name,
    u.Department,
    SUM(oli.TotalPrice) AS total_revenue,
    COUNT(DISTINCT o.Id) AS deals_closed,
    AVG(oli.TotalPrice) AS avg_deal_size
FROM OpportunityLineItems oli
    JOIN Opportunities o ON o.Id = oli.OpportunityId
    JOIN Users u ON u.Id = o.OwnerId  
WHERE o.CloseDate >= '2025-07-01'
    AND o.CloseDate < '2025-10-01'
    AND o.IsWon = 1
GROUP BY u.Id, u.Name, u.Department
ORDER BY total_revenue DESC;

📊 Results: 8 sales reps | Total: $2.3M revenue
```

---

## 🔧 Migration from Basic Text-to-SQL

### **Key Improvements**
1. **Zero Hallucinations**: Only uses discovered schema objects with proven relationships
2. **OpenAI Intelligence**: Sophisticated intent parsing and entity resolution
3. **CRM Awareness**: Understands CRM-specific patterns and metrics
4. **Relationship Truth**: Leverages FKs, views, and SPs for accurate joins
5. **Capability Contracts**: Validates completeness before execution
6. **Business Results**: Formatted output with proper labels and units

### **Breaking Changes**
- Queries now require capability validation - some may return NEAR instead of executing
- Enhanced schema analysis required - re-run discovery and semantic analysis
- OpenAI API key required for NER functionality

---

## 📈 Success Metrics

**Production Quality Indicators:**
- **Schema Coverage**: >90% of CRM tables properly classified
- **Relationship Accuracy**: >95% of generated JOINs use proven paths  
- **Intent Recognition**: >85% accurate OpenAI entity/metric extraction
- **Capability Success**: >80% of queries pass validation and execute
- **Zero Hallucination Rate**: 100% (never reference non-existent objects)
- **Business User Satisfaction**: Readable results with proper context

This system ensures **enterprise-grade CRM analytics** by combining OpenAI's language understanding with rigorous schema validation and relationship discovery from existing database artifacts.