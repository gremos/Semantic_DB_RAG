# Quick Start Guide

Get up and running with the GPT-5 Semantic Modeling & SQL Q&A System in minutes.

## Prerequisites

- Python 3.11 or higher
- Access to an Azure OpenAI deployment (GPT-5-mini)
- A SQL Server database (or other supported database)
- SQL Server ODBC Driver 17 or higher (for MSSQL)

## Installation

### 1. Clone or Download the Project

```bash
cd /path/to/your/workspace
# Project should be in gpt5-semantic-sql/ directory
```

### 2. Create Virtual Environment

```bash
cd gpt5-semantic-sql
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required configuration in `.env`:

```bash
# Azure OpenAI (Required)
DEPLOYMENT_NAME=gpt-5-mini
API_VERSION=2025-01-01-preview
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-api-key-here

# Database (Required)
DATABASE_CONNECTION_STRING=mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server

# Optional: Adjust paths and settings as needed
```

## Usage

### Phase 1: Discover Your Database

This phase introspects your database, samples data, detects relationships, and caches the results.

```bash
python main.py discovery
```

**Expected Output:**
```
✓ Discovery completed successfully!
✓ Found 3 schemas
✓ Cached at: ./cache/discovery.json

Discovery Summary
• Total Tables: 45
• Inferred Relationships: 23
• Named Assets: 5
```

**What Happens:**
- Connects to your database (read-only)
- Discovers all tables, columns, keys, indexes
- Samples up to 1000 rows per table for statistics
- Stores up to 10 example rows per table
- Detects implicit foreign keys (>80% value overlap)
- Parses any RDL files in `./data_upload/`
- Caches everything to `./cache/discovery.json`

**Cache Duration:** 168 hours (7 days) by default

### Phase 2: Build Semantic Model

This phase builds a business-friendly semantic model from the discovery cache.

```bash
python main.py model
```

**Expected Output:**
```
✓ Semantic model built successfully!
✓ Cached at: ./cache/semantic_model.json

Semantic Model Summary
• Entities: 12
• Dimensions: 5
• Facts: 8
• Relationships: 23
```

**What Happens:**
- Loads discovery cache
- Identifies entities (customers, products, etc.)
- Identifies dimensions (time, geography, etc.)
- Identifies facts (sales, orders, etc.)
- Assigns semantic types and roles to columns
- Defines measures with units and formatting
- Ranks duplicate sources (views preferred over tables)
- Caches model to `./cache/semantic_model.json`

**Note:** Phase 2 requires full LLM integration implementation (currently stub).

**Cache Duration:** 168 hours (7 days) by default

### Phase 3: Ask Questions

Query your data using natural language.

```bash
python main.py question "What were the top 10 customers by revenue last month?"
```

**Expected Output:**
```
Generated SQL:
┌─────────────────────────────────────┐
│ Query 1 (mssql)                     │
├─────────────────────────────────────┤
│ SELECT TOP 10                       │
│   c.CustomerName,                   │
│   SUM(o.TotalAmount) as Revenue     │
│ FROM dbo.Customer c                 │
│ JOIN dbo.Orders o ON c.CustomerID = o.CustomerID │
│ WHERE o.OrderDate >= DATEADD(month, -1, GETDATE()) │
│ GROUP BY c.CustomerName             │
│ ORDER BY Revenue DESC               │
└─────────────────────────────────────┘

First Row Interpretation:
Acme Corp had the highest revenue of $1,234,567.89 in the last month

Results (10 rows):
[...JSON results...]

Suggested follow-up questions:
  1. What products did Acme Corp purchase?
  2. How does this compare to the previous month?
  3. Which sales rep handled Acme Corp's orders?

✓ Answer logged to: ./logs/qa_history.log.jsonl
```

**What Happens:**
- Loads semantic model
- Parses question intent with LLM
- Calculates confidence score
- Generates SQL grounded in semantic model
- Validates SQL (read-only, syntax, grounding)
- Executes query with timeout and row limit
- Formats results with evidence chain
- Logs to Q&A history for analysis

**Note:** Phase 3 requires full implementation (currently stub).

### Clear Caches

Force fresh discovery and model building:

```bash
python main.py cache-clear
```

**Use Cases:**
- Database schema changed
- Want to adjust exclusion rules
- Testing different configurations

## Troubleshooting

### Connection Errors

**Problem:** `Failed to connect to database`

**Solutions:**
1. Verify connection string in `.env`
2. Check database server is accessible
3. Verify credentials are correct
4. Ensure ODBC driver is installed (for MSSQL)
5. Test connection with a database client first

### Discovery Timeout

**Problem:** `Discovery timed out`

**Solutions:**
1. Increase `DISCOVERY_TIMEOUT` in `.env` (default: 300 seconds)
2. Add more exclusions to reduce scope:
   ```bash
   SCHEMA_EXCLUSIONS=sys,information_schema,temp
   TABLE_EXCLUSIONS=temp_,test_,backup_,old_,staging_
   ```
3. Use more aggressive table exclusion patterns

### Azure OpenAI Errors

**Problem:** `Invalid API key` or `Rate limit exceeded`

**Solutions:**
1. Verify `AZURE_OPENAI_API_KEY` is correct
2. Check `AZURE_ENDPOINT` matches your resource
3. Verify `DEPLOYMENT_NAME` matches your deployment
4. Check rate limits in Azure Portal
5. Wait and retry if rate limited

### Cache Issues

**Problem:** `Discovery data not found` when running model

**Solution:** Run discovery first:
```bash
python main.py discovery
python main.py model
```

**Problem:** Stale cache data

**Solution:** Clear cache and re-run:
```bash
python main.py cache-clear
python main.py discovery
python main.py model
```

## File Locations

- **Discovery Cache:** `./cache/discovery.json`
- **Semantic Model Cache:** `./cache/semantic_model.json`
- **Debug Logs:** `./logs/discovery_semantic.log`
- **Q&A History:** `./logs/qa_history.log.jsonl`
- **RDL Files:** `./data_upload/` (place your .rdl files here)

## Configuration Tips

### Exclude Temporary Tables

```bash
TABLE_EXCLUSIONS=temp_,test_,staging_,backup_,old_
TABLE_EXCLUSION_PATTERNS=.*_\d{8}$,.*_archive.*,.*_copy.*
```

### Adjust Batch Sizes for Large Schemas

For databases with hundreds of tables, reduce batch sizes to avoid LLM context limits:

```bash
ENTITY_BATCH_SIZE=1
DIMENSION_BATCH_SIZE=1
FACT_BATCH_SIZE=1
```

### Adjust Confidence Thresholds

Be more strict about answering questions:

```bash
CONFIDENCE_HIGH=0.90
CONFIDENCE_MEDIUM=0.75
CONFIDENCE_LOW=0.60
```

### Longer Cache Duration

Keep caches longer for stable databases:

```bash
DISCOVERY_CACHE_HOURS=720  # 30 days
SEMANTIC_CACHE_HOURS=720   # 30 days
```

## Development Workflow

### Typical Flow

1. **Initial Setup:** Discovery → Model → Test Questions
2. **Iterative:** Adjust exclusions → Clear cache → Re-discover
3. **Daily Use:** Ask questions (uses cached model)
4. **Weekly:** Clear cache and refresh (captures schema changes)

### Adding RDL Files

1. Place .rdl files in `./data_upload/` directory
2. Run discovery: `python main.py discovery`
3. RDL datasets will be included in discovery results

### Monitoring

Check logs for issues:

```bash
# View discovery/semantic logs
tail -f logs/discovery_semantic.log

# View Q&A history
tail -f logs/qa_history.log.jsonl

# Parse Q&A history
cat logs/qa_history.log.jsonl | jq '.'
```

## Next Steps

1. **Explore Discovery Results:**
   ```bash
   cat cache/discovery.json | jq '.schemas[0].tables[0]'
   ```

2. **Examine Semantic Model:**
   ```bash
   cat cache/semantic_model.json | jq '.entities[0]'
   ```

3. **Test Questions:**
   - Start with simple questions
   - Check generated SQL
   - Iterate on semantic model if needed

4. **Customize:**
   - Add exclusions for your database
   - Adjust batch sizes
   - Tune confidence thresholds

## Getting Help

- **Logs:** Check `./logs/discovery_semantic.log` for errors
- **Project Structure:** See `PROJECT_STRUCTURE.md`
- **Configuration:** See `.env.example` for all options
- **Documentation:** See `README.md` for detailed info

## Implementation Status

### ✅ Working Now

- Phase 1: Discovery (fully implemented)
- CLI commands and caching
- Logging and configuration

### 🔨 In Development

- Phase 2: Semantic Model (stub - requires LLM implementation)
- Phase 3: Q&A (stub - requires LLM implementation)
- QuadRails validation (stub)

See `PROJECT_STRUCTURE.md` for detailed implementation status.
