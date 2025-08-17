# 🧠 BI-Aware Semantic Database RAG System — **No-Fallback Production**

> A **business intelligence aware** Text-to-SQL system with **strict capability contracts** and **zero schema hallucinations**. Never executes unsafe queries—returns **Non-Executable Analysis Reports (NER)** when capability checks fail.

---

## 🎯 Core BI-Aware Principles

### **No-Fallback Operating Rules**
- **Never execute a query unless the schema can prove it supports the ask**
- If any required capability is missing, return a **Non-Executable Analysis Report (NER)** with what's missing and fix paths
- **Gate every query behind three validations**: Identifier gate, Relationship gate, Capability gate
- Use **constrained decoding + AST check + EG retry**, but never schema fallback

### **Universal Capability Contract**
For any BI question, require ALL of the following before generation/execution:

✅ **Grain**: Row grain of the fact table (payment, invoice_line, order, event)  
✅ **Measure(s)**: Numeric column(s) compatible with the asked metric (sum/avg/count/distinct)  
✅ **Time**: Usable timestamp/date for filtering and bucketing  
✅ **Entity key(s)**: The entity whose metric is grouped by/over (customer, product, user, store, rep)  
✅ **Join path(s)**: Proven join(s) from fact → dimension(s) via FK or observed view/SP pattern  
✅ **Filters & Status**: Columns for explicit filters or 'is_refund/is_cancel' flags  
✅ **Quality minima**: Row count > 0, null-rate thresholds, optional data-freshness check  

**If ANY item fails → produce NER instead of executing**

---

## 🏗️ Architecture (BI-Enhanced)

```
semantic-db-rag/
├── main.py                         # CLI with three options + BI validation
├── shared/
│   ├── config.py                   # Env + BI feature flags
│   ├── models.py                   # BI-enhanced dataclasses with capability contracts
│   └── utils.py                    # I/O, logging, safety + evidence scoring
├── db/
│   └── discovery.py                # Schema discovery + samples + view/SP analysis (unchanged)
├── semantic/
│   └── analysis.py                 # BI-AWARE: Capability contracts + Evidence-driven selection + NER
├── interactive/
│   └── query_interface.py          # BI-aware 4-stage pipeline with capability gates
└── data/
    ├── database_structure.json     # Canonical cache (schema + samples + views/SP text)
    ├── semantic_analysis.json      # BI-enhanced: Operational/Planning classification + capability scores
    └── query_patterns.json         # Learned successful patterns (capability-validated only)
```

---

## 🧠 BI-Aware Semantic Analysis (NEW)

### **Enhanced Table Classification**
Beyond basic entity types, now classifies:

**Data Types:**
- **Operational**: Real transactions, events, actual business activity
- **Planning**: Targets, goals, budgets, forecasts (often zeros or future dates)
- **Reference**: Lookup tables, codes, categories, static definitions

**BI Roles:**
- **Fact**: Contains measures (amounts, quantities) and foreign keys to dimensions
- **Dimension**: Contains descriptive attributes (names, descriptions, categories)
- **Bridge**: Many-to-many relationships between facts and dimensions

**Capability Assessment:**
- **Measures**: Numeric columns suitable for SUM/AVG/COUNT operations
- **Entity Keys**: Columns for GROUP BY operations and joins
- **Time Columns**: Date/timestamp columns for temporal filtering and trending
- **Filter Columns**: Status, type, region columns for WHERE conditions

### **Evidence-Driven Object Selection**
Tables ranked using weighted evidence score:

1. **Role Match** (high weight): Fact table with operational data + has measures/dates
2. **Join Evidence** (high weight): FK relationships or observed joins from views/SPs
3. **Lexical/Semantic Match** (medium weight): Synonyms in table/column names
4. **Graph Proximity** (medium weight): Hop distance between fact and requested dimensions
5. **Operational Tag** (medium weight): Operational > planning/config tables
6. **Row Count & Freshness** (tie-breaker): Data availability and recency

---

## 🚫 No-Fallback Query Pipeline

### **Stage 1: Intent → Analytical Task**
Normalize natural-language question to structured task:
- **Task type**: aggregation | ranking | trend | distribution | cohort | funnel
- **Metric(s)**: revenue, orders, active users, conversion rate
- **Entity**: customer, product, sales rep, region
- **Time window**: Q2_2025, last_12_months, YTD
- **Grouping**: by_customer, by_rep, by_product
- **Filters**: segments, statuses, geos

### **Stage 2: Capability Gate**
For top-ranked candidate tables, validate capability contract:
```python
✓ Has grain definition (what each row represents)
✓ Has measures OR entity keys for the requested metric
✓ Has time column for filtering (if time-based query)
✓ Has proven join paths to requested dimensions
✓ Passes quality minima (row count > 0, reasonable null rates)
```

### **Stage 3: Evidence-Driven Selection**
Only tables passing capability gate are considered:
- Score candidates using evidence weights
- Select top N tables with complete capability contracts
- Document WHY each table was chosen (role, joins, lexical, operational status)

### **Stage 4: Validated SQL Generation OR NER**
**If capable tables found**: Generate SQL using proven schema elements only  
**If NO capable tables**: Return **Non-Executable Analysis Report** with:
- What you asked (normalized analytical task)
- Missing capabilities (grain, measures, time, joins, quality)
- Top candidate facts/dims with evidence scores
- Fix paths (suggested mappings, schema additions)
- Safe exploratory queries (metadata-only SELECTs)

---

## 📋 Non-Executable Analysis Report (NER)

When capability checks fail, return structured report instead of "safe" query:

```json
{
  "question": "Top 10 customers by revenue in Q2 2025",
  "normalized_task": {
    "task_type": "ranking",
    "metrics": ["revenue"],
    "entity": "customer",
    "time_window": "Q2_2025",
    "top_limit": 10
  },
  "missing_capabilities": [
    "No measure column found for revenue aggregation",
    "No proven join path from transactions to customers"
  ],
  "top_candidate_tables": [
    {
      "table": "[dbo].[Payments]",
      "evidence_score": 0.8,
      "reasoning": "Has amount columns but missing customer join"
    }
  ],
  "fix_paths": [
    "Add foreign key from [Payments].[CustomerID] to [Customers].[ID]",
    "Use [Payments].[Amount] as revenue measure",
    "Use [Payments].[PaymentDate] for Q2 2025 filtering"
  ],
  "suggested_queries": [
    "SELECT TOP 5 * FROM [dbo].[Payments] -- Explore structure",
    "SELECT PaymentDate, Amount, COUNT(*) FROM [dbo].[Payments] GROUP BY PaymentDate, Amount"
  ]
}
```

---

## ⚙️ Configuration (BI-Enhanced)

```env
# BI-Aware Analysis Settings
ENABLE_BI_CAPABILITY_CONTRACTS=true
ENABLE_EVIDENCE_DRIVEN_SELECTION=true
OPERATIONAL_DATA_PRIORITY=true
DISABLE_QUERY_FALLBACKS=true

# Capability Contract Thresholds
MIN_ROW_COUNT_FOR_FACTS=100
MAX_NULL_RATE_FOR_MEASURES=0.3
EVIDENCE_SCORE_THRESHOLD=0.6
CAPABILITY_COMPLETENESS_THRESHOLD=0.8

# NER Generation Settings
MAX_CANDIDATE_TABLES_IN_NER=5
INCLUDE_SAFE_EXPLORATORY_QUERIES=true
DETAILED_FIX_PATH_SUGGESTIONS=true
```

---

## 🔍 Usage Examples

### **✅ Successful Capability Match**
```
❓ Query: "What are our top 10 customers by revenue in Q2 2025?"

🧠 BI Analysis:
   📊 Intent: ranking task, revenue metric, customer entity, Q2_2025 time window
   🔍 Evidence-driven selection: [dbo].[CustomerPayments] (score: 0.92)
   ✅ Capability contract satisfied:
      • Grain: payment transactions
      • Measures: Amount column (numeric, non-null)
      • Time: PaymentDate column available
      • Entity keys: CustomerID for grouping
      • Join paths: FK to [dbo].[Customers]

📋 Generated SQL:
SELECT TOP 10 
    c.CustomerName,
    SUM(p.Amount) as TotalRevenue,
    COUNT(*) as TransactionCount
FROM [dbo].[CustomerPayments] p
JOIN [dbo].[Customers] c ON p.CustomerID = c.CustomerID
WHERE p.PaymentDate >= '2025-04-01' AND p.PaymentDate < '2025-07-01'
GROUP BY c.CustomerID, c.CustomerName
ORDER BY TotalRevenue DESC

📊 Results: 10 rows returned
```

### **❌ Capability Contract Failure → NER**
```
❓ Query: "Show me customer satisfaction trends by region"

🧠 BI Analysis:
   📊 Intent: trend task, satisfaction metric, region entity
   🔍 Evidence-driven selection: [dbo].[Customers] (score: 0.4)
   ❌ Capability contract FAILED:
      • Missing: No satisfaction measure column found
      • Missing: No time column for trend analysis
      • Missing: No region dimension available

📋 Non-Executable Analysis Report:
   ⚠️ Cannot execute - missing required capabilities
   
   🔧 Fix Paths:
      • Add satisfaction score column to customer table
      • Add survey/feedback fact table with satisfaction measures
      • Add region dimension or region column to customer table
      • Ensure temporal data for trend analysis
   
   🔍 Suggested Exploration:
      SELECT TOP 5 * FROM [dbo].[Customers] -- Check available columns
      SELECT COUNT(*) as CustomerCount FROM [dbo].[Customers] -- Data volume
```

---

## 🛡️ Security & Safety (Enhanced)

### **Three-Gate Validation**
1. **Identifier Gate**: SQL must reference only discovered, allow-listed objects (SQLGlot parse + allow-list)
2. **Relationship Gate**: All joins must exist in FK graph or observed in parsed view/SP text
3. **Capability Gate**: Selected fact/dims must satisfy metric template's minimal requirements

### **Enterprise Guardrails**
- Read-only DB principal with strict permissions
- Allow-list SQL forms (SELECT/WITH/EXPLAIN only)
- AST validation via SQLGlot with identifier existence checks
- No dynamic SQL execution from stored procedures
- Prompt isolation and input/output filtering
- Query timeouts and row limits

---

## 🚀 Getting Started (BI-Aware)

1. **Setup Environment**
```bash
# Copy configuration
cp env_example.txt .env

# Set required variables
AZURE_OPENAI_API_KEY=your_key
DATABASE_CONNECTION_STRING=your_connection
ENABLE_BI_CAPABILITY_CONTRACTS=true
DISABLE_QUERY_FALLBACKS=true
```

2. **Install Dependencies**
```bash
pip install pyodbc sqlglot langchain-openai
```

3. **Run BI-Aware Discovery**
```bash
python main.py
# Choose Option 1: Database Discovery (same as before)
```

4. **Run BI-Aware Analysis**
```bash
# Choose Option 2: Semantic Analysis (now BI-enhanced)
# - Classifies operational vs planning data
# - Identifies fact vs dimension tables
# - Assesses capability contracts for each table
# - Builds evidence-driven selection rankings
```

5. **Query with BI Validation**
```bash
# Choose Option 3: Interactive Queries (now capability-gated)
# - Parses intent into analytical tasks
# - Validates capability contracts before execution
# - Returns NER when contracts fail
# - Only executes proven, validated SQL
```

---

## 📊 What's New in BI-Aware Version

### **🔥 Major Enhancements**
- **Capability Contracts**: Universal validation before any query execution
- **Evidence-Driven Selection**: Weighted scoring replaces simple keyword matching
- **BI Table Classification**: Operational/Planning/Reference + Fact/Dimension/Bridge roles
- **Non-Executable Analysis Reports**: Structured guidance when queries can't be safely executed
- **No-Fallback Mode**: Never runs arbitrary "safe" queries on random tables

### **🧠 Intelligence Upgrades**
- **Grain Detection**: Understands what each table row represents
- **Measure Identification**: Finds numeric columns suitable for aggregation
- **Time Dimension Discovery**: Locates columns for temporal analysis
- **Join Path Validation**: Only uses proven relationships from FKs or observed patterns
- **Operational Data Priority**: Distinguishes real transactions from planning/target data

### **🛡️ Safety Improvements**
- **Three-Gate Validation**: Identifier + Relationship + Capability gates
- **Zero Schema Hallucination**: Only references discovered objects with proven capabilities
- **Quality Minima**: Checks row counts, null rates, data freshness
- **Structured Error Reporting**: Actionable guidance instead of generic error messages

---

## 🔧 Migration from Previous Version

If upgrading from the previous version:

1. **Configuration**: Add BI-aware settings to `.env`
2. **Re-run Analysis**: The semantic analysis now includes BI classification
3. **Update Queries**: Queries now go through capability validation
4. **Handle NERs**: Prepare to receive Non-Executable Analysis Reports for unsupported queries

**Breaking Changes**:
- No more fallback queries - unsupported requests return NER
- Enhanced table metadata requires re-running semantic analysis
- Some previously "working" queries may now fail capability validation (this is intentional for safety)

---

## 📈 Success Metrics

**BI-Aware System Quality Indicators**:
- **Capability Contract Coverage**: % of tables with complete contracts
- **Evidence Score Distribution**: Quality of table selection reasoning
- **NER Rate**: % of queries requiring Non-Executable Analysis Reports
- **False Positive Rate**: Queries that pass validation but fail execution
- **Business User Satisfaction**: Quality of generated insights and guidance

**Target Metrics for Production**:
- Capability contract coverage: >80%
- Evidence-driven selection accuracy: >90%
- NER rate: <20% (most queries should be executable)
- False positive rate: <5%
- Zero schema hallucinations: 100%

---

This BI-aware system ensures **enterprise-grade reliability** by never executing queries that cannot be proven safe and effective. Instead of falling back to arbitrary queries, it provides **actionable intelligence** about what's missing and how to fix it.