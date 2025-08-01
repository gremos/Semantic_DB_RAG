# 🧠 Enhanced Semantic Database RAG System


1. Database Discovery Module (Option 1: Database Discovery 🔍)
Purpose: Connect to the SQL Server database, read the schema structure and a sample of data, then output a summary JSON (database_structure.json). This gives the LLM (and the user) an overview of what data is available. Key Tasks:
Connect to the database (using credentials from environment, e.g. via a connection string or pyodbc/sqlalchemy).
Retrieve a list of all user tables and views. System tables, temporary tables, or any irrelevant technical tables should be filtered out (e.g. by schema name or using SQL Server’s catalog views to exclude them).
For each table/view: get its column names, data types, and if possible, primary key and foreign key info. This could involve queries against system catalogs like INFORMATION_SCHEMA.COLUMNS and INFORMATION_SCHEMA.TABLE_CONSTRAINTS or the newer catalog views (sys.tables, sys.foreign_keys, etc.).
Also fetch the first 5 rows of data from each table/view as a sample. Using a query like SELECT * FROM [Schema].[Table] OPTION (FAST 5) (or simply a TOP 5) is a good strategy to retrieve a few rows quickly without scanning the whole table. These samples will help the LLM understand the kind of data each column holds (e.g. numeric vs text, example values which hint at meaning).
While looping through tables, update a progress bar (e.g. using a library like tqdm or rich) to show progress, especially if there are many tables. This gives real-time feedback in the CLI.
Compile the gathered information into a structured Python object (dictionary) and then save it as database_structure.json. For clarity, the JSON structure might look like:
json
Copy
Edit
{
  "tables": {
     "Schema.TableName": {
        "columns": {
           "Column1": "INT",
           "Column2": "VARCHAR(100)",
           ... 
        },
        "sample_rows": [
           { "Column1": 1, "Column2": "Alice", ... },
           { "Column1": 2, "Column2": "Bob", ... },
           ...
        ],
        "primary_key": ["Column1"],
        "foreign_keys": [
           { "column": "DeptId", "references": "HR.Department(Id)" }
        ]
     },
     "...": { ... }
  }
}
Ensure proper handling of text encoding so that any international characters (Greek, etc.) in the data are preserved in the JSON. Python’s default UTF-8 encoding should handle this, but be mindful when writing to files or console.
It’s wise to encapsulate this logic in a function or class method (e.g. run_discovery()), which can be called when the user selects option 1. This function can also perform a check like, “if database_structure.json already exists, confirm if the user wants to regenerate it,” to avoid unnecessary re-scanning.
By structuring the discovery phase cleanly, we make the subsequent steps easier. The JSON will serve as a cache of schema info that the later steps can use, which improves performance (no need to query the database schema repeatedly). It also decouples the database access code from the LLM logic.
2. Semantic Analysis Module (Option 2: Semantic Analysis 🧠)
Purpose: Enrich the raw schema/data information with semantic context using the LLM. This module takes the database_structure.json from step 1 and produces a new JSON (semantic_analysis.json) with additional insights: classification of tables, discovered relationships, and domain knowledge. Key Tasks:
Load the database_structure.json produced by the discovery step. This provides the list of tables, columns, and samples.
Entity Classification: For each table or view, use the LLM to determine its business role and type. For example, a table with columns like CustomerName, ContactInfo, Address might be classified as a Customer Master table in a CRM domain, whereas a table with OrderID, CustomerID, TotalAmount, OrderDate might be classified as a Sales Transactions table. The classification could include:
A short description of the table’s purpose. (e.g. "Stores customer contact details")
A category or role (e.g. "Dimension" vs "Fact", or "Master Data" vs "Transaction Records").
Any domain-specific label (e.g. "CRM/Customer data").
Relationship Discovery: Utilize both deterministic logic and the LLM to find relationships:
Explicit relationships: Extract foreign key references from the schema metadata (many might have been found in Option 1). These give table-to-table links (e.g. Orders.CustomerID → Customers.ID). Add these to the semantic JSON.
Implicit relationships: Some tables might not have formal foreign keys but are related by naming conventions or data. For instance, a column named CustomerName in one table might correspond to Name in a Customer table. Or there might be tables that share a common prefix or part of their name (like Orders and OrderItems). The LLM can be prompted to suggest if any tables appear related based on their content and names. It could analyze the sample rows (e.g., see that values in OrderItems.OrderID look like the IDs in Orders.ID). This step is experimental – the LLM can sometimes infer relationships a human might see, but it could also make mistakes, so treat these inferences as suggestions.
The output for relationships can be a list of mappings or a graph-like structure. E.g., "relationships": [ {"from": "Orders.CustomerID", "to": "Customers.ID", "type": "foreign_key"}, {"from": "Orders", "to": "OrderItems", "type": "one-to-many via OrderID"} ].
Business Domain Identification: Prompt the LLM to analyze all table names and descriptions to guess the overall domain/industry. For example, tables named Patient, Diagnosis, Treatment likely indicate a Healthcare domain; Customer, Invoice, Product suggest Retail; Student, Course, Enrollment suggest Education, etc. This domain info can be saved (e.g., "domain": "Retail CRM" or "domain": "Healthcare Management"). It provides context but is mostly for user information or future use in refining queries (the system might use domain to choose certain vocabulary or clarifications with the user).
Use international-friendly processing: If table or column names or data samples are in Greek (or any language), ensure the LLM is instructed to preserve and understand them. Azure’s GPT-4 should handle Unicode text, but you might explicitly tell the model to maintain non-English text in output (to avoid transliteration issues).
Prompt Engineering: Compose clear prompts for the LLM in each of the above tasks. You might combine some tasks (e.g., ask the LLM to produce a JSON containing the table classification and any relationships it sees for that table). Alternatively, iterate table by table for classification, then do another pass for cross-table relationships. A possible approach is:
Table classification prompt: “Here is a database table with its columns and example rows: {…}. Describe the table’s purpose in business terms, classify its role (e.g., dimension/fact), and suggest what real-world entity or concept it represents.”
Cross-table relationship prompt: “Given the following two tables summaries, do you see any logical relationship between them? Table A: {...}, Table B: {...}.” This could be done for every pair or based on name similarity to reduce calls. However, doing an LLM call for every pair might be expensive if there are many tables (this approach doesn’t scale well beyond small schemas). A heuristic to cut down combinations is to only check likely pairs (e.g., where one table’s name or columns contain part of the other’s name, or where a sample value in one appears in another).
Update the user via a progress bar or console messages since this step can be slow (especially if many LLM calls are made). For example: “Classifying table 5/20: Customers…”, “Analyzing relationships for table Orders…”.
Save the resulting semantic info to semantic_analysis.json. This file will augment the original structure JSON with fields like "description", "category", and "relationships" discovered. Essentially it becomes a lightweight knowledge graph of the database.
Structuring this as a separate module or at least a distinct function (run_semantic_analysis()) is recommended. It should load the prior JSON, loop through tables for classification, and build the new JSON. By keeping this logic separate, you can easily swap out the prompt phrasing or handle different model versions (e.g., a future update might use a different model or include few-shot examples to improve output quality). The separation also means if the user re-runs Option 2, you can load existing results and perhaps skip already-processed parts. Feasibility Note: This semantic analysis is an optional enhancement in the sense that a basic text-to-SQL could work with just the raw schema. But adding it greatly improves the LLM’s understanding. Other projects have implemented similar semantic layers, confirming this approach. The challenge is ensuring the LLM’s outputs are accurate and useful. In practice, you might need to review semantic_analysis.json for obvious errors (especially early in development). Over time, fine-tuning prompts or providing a few examples of correct classifications can increase reliability.
3. Interactive Query Module (Option 3: Interactive Queries 💬)
Purpose: Provide a CLI interactive interface where the user can ask natural language questions, and the system will generate and execute SQL queries to answer those questions. This is the end-user-facing part that brings together the previous outputs (schema and semantic context) to actually retrieve data insights. Key Tasks:
Load context: When this module starts, it should load the saved database_structure.json and semantic_analysis.json (if available). Having both allows the system to use raw schema info as well as semantic info. If the semantic file is missing (e.g., user skipped Option 2), the tool could either prompt the user to run that step or proceed with limited context (just schema). However, the best results will be obtained if both steps have been completed.
User Prompt Input: Display a prompt (e.g., > ) for the user to type a question in natural language. Also, show a reminder or example like “Ask a question about the data, e.g., ‘What were the total sales last month per region?’”. This makes it clear what the user is supposed to do.
Identify Relevant Tables/Columns: This is a crucial retrieval step before generating SQL. The system should figure out which parts of the database are likely needed to answer the question. There are multiple strategies to do this, which can be used in combination:
Keyword matching: Parse the user question for keywords that match table names or column names. For instance, if the question mentions "customers" or "customer name", it likely involves the Customers table. If it says "total sales", maybe a Sales or Orders table is relevant. Even plural vs singular variations or common synonyms should be considered (the semantic layer can help here, since it might label a table as “customer info” even if the table name is CLIENT_TBL).
Semantic search: Use the LLM or embeddings to interpret the question and compare it to table descriptions. Since an LLM is available, a simpler approach is to directly ask it: “Given the user question and the following list of tables with descriptions, which tables (and columns) are relevant?” This can be done by providing the model with a summary of each table (from the semantic analysis JSON) and the question, then parsing its answer for the table names. Alternatively, for efficiency, you might embed the table descriptions and the query and do a vector similarity search if using an embedding model. But given the scope (“implemented from scratch” and CLI tool), sticking to an LLM-based reasoning or a simple algorithm is acceptable initially.
Use of relationships: If one relevant table is identified, consider pulling in tables that are directly related to it (via foreign keys or relationships from the semantic analysis). For example, a question about “orders by customer” clearly involves the Orders table, but likely the Customers table too (to get customer names). If the system knows Orders → Customers relationship, it should include both in context.
Compose the Prompt for SQL Generation: Once the candidate set of relevant tables/columns is chosen, the next step is to ask the LLM to generate an SQL query. The prompt to the LLM could look like:
System message: You are an expert SQL generator. Given a user question and relevant database tables, you will produce an SQL query that correctly answers the question.
User message: “User Question: {question}\nRelevant Tables:\n - Table: Orders (columns: OrderID, OrderDate, TotalAmount, CustomerID, ...)\n - Table: Customers (columns: CustomerID, Name, City, ...)\nProvide the SQL query only.”
This prompt provides the context the model needs. We include only the relevant tables’ schemas (not the whole database) to avoid confusion and token overload
reddit.com
. The semantic descriptions from earlier can be included as brief comments if needed, but often listing columns is enough for structure. If the model has been told the high-level domain (e.g., “This is a retail sales database”), that can also be part of the system prompt to guide its assumptions.
Model Considerations: Using the specified Azure OpenAI deployment (gpt-4.1-mini), which presumably is a variant of GPT-4, should yield good results for non-trivial queries. GPT-4 is quite adept at mapping natural language to SQL given schema context. If needed, include a few-shot example in the prompt (a simple Q→SQL pair) to further guide it, though this increases prompt length. In many cases, a well-crafted single prompt will suffice.
SQL Validation: After the LLM returns a query, the system should verify it before execution:
Syntax check: Attempt to run the query with EXPLAIN or a dry run. If the database throws an error (syntax or semantic), catch it. You could either report the error to the user or even feed the error back to the LLM for a second attempt (though automatic retry logic can be complex to implement reliably).
Safety: Ensure the query is read-only (the system should ideally not execute UPDATE/DELETE queries from an NL prompt!). If using a restricted SQL user with read-only permissions, that’s a safe guard. Also, consider automatically appending a row limit if not present, to prevent huge result sets. The user specifically mentioned using OPTION (FAST 5) for example – one strategy is to always limit output to e.g. 100 rows unless the user requests more. This way a broad query doesn’t overwhelm the CLI.
Execute Query: If validation passes, run the query against the database. Fetch the results (e.g., via a cursor) and format them. For CLI display, pretty-print the output in tabular form. Python libraries like tabulate or rich can format tables nicely. If results are too many, show the first N and a note like “… and X more rows”.
Iterative Interaction: After showing results, allow the user to ask another question or refine the query. This could simply loop back to prompt for the next question. A more advanced feature (for later) could maintain context so the user can ask follow-up questions (e.g., “Now show me only last year’s data” after an initial question) – but that requires tracking the conversation state and perhaps re-using the last SQL or results. For now, a simpler stateless loop (each question independently handled) is easier and aligns with a CLI workflow.
Menu Integration: Since this is option 3 in the menu, the program’s main loop should handle switching to this interactive mode. For example, when the user selects "3. Interactive Queries", the code can enter a sub-loop that keeps asking for questions until the user types some exit command or an empty line. Provide instructions like “Type 'exit' or press Ctrl+C to return to main menu.”
By separating this interactive logic, we ensure the main program isn’t cluttered with prompt-building and SQL-handling code. A function like run_interactive_queries() could encapsulate this behavior. Internally it might call helper functions like find_relevant_tables(question) and generate_sql(query, tables_context) to keep things modular. This way, each part can be tested or updated independently (for example, swapping out the method for finding relevant tables to a more sophisticated one later). Progress and Feedback: While each individual query will be relatively fast (just an API call and a DB query), providing some feedback is still helpful. If an LLM call is taking a few seconds, you might show a spinner or message “Generating SQL…”. If the query execution is slow, a progress bar or at least a note “Executing query, please wait…” is good. The user should feel the system is responsive and know that something is happening, even if the operation takes, say, 10 seconds.

we need views analysis for join, 

we need foreign key analysis 

we also need llm to scan database structure to export all available entities 


**Version 2.0** - Simple, Readable, and Maintainable with Enhanced View Handling

# 🧠 Semantic Database RAG System

**Simple, Readable, and Maintainable Python Implementation**

A powerful AI-driven system that provides intelligent database analysis and natural language querying capabilities. Built with Azure OpenAI (MODEL_VERSION=2024-12-01-preview) and designed for real-world business databases with proper Greek text and international character support.

## 🎯 Core Features

### ✅ **Complete Database Discovery**
- **Full Structure Analysis**: Tables, views, columns, data types, primary keys, foreign keys
- **Sample Data Collection**: 5 sample rows per table/view for context understanding
- **View Query Analysis**: Extracts view definitions to understand relationships and business logic
- **Performance Optimized**: Uses `OPTION (FAST 5)` for 2-5x faster data sampling
- **Unicode Support**: Proper handling of Greek text and international characters

### ✅ **Intelligent Entity Classification**  
- **LLM-Powered Analysis**: Uses Azure OpenAI to identify all business entities
- **Pattern Recognition**: Combines rule-based classification with AI enhancement
- **Relationship Discovery**: Finds explicit (foreign keys) and implicit relationships
- **Business Domain Identification**: Automatically detects industry context

### ✅ **Natural Language Querying**
- **Smart Table Selection**: Fuzzy matching to find relevant tables for questions
- **SQL Generation**: Converts natural language to optimized T-SQL queries
- **Query Execution**: Safe execution with proper error handling and result limits
- **Real-time Results**: Interactive session with immediate feedback

## 🏗️ Simple Architecture

```
semantic-db-rag/
├── main.py                     # 🚀 Main entry point with 3 core options
├── shared/                     # 🔄 Shared components
│   ├── config.py              # ⚙️ Configuration with Azure OpenAI settings
│   ├── models.py              # 📊 Data models (TableInfo, BusinessDomain, etc.)
│   └── utils.py               # 🛠️ Utility functions
├── db/
│   └── discovery.py           # 🔍 Database structure discovery
├── semantic/
│   └── analysis.py            # 🧠 LLM-powered entity classification
├── interactive/
│   └── query_interface.py     # 💬 Natural language query interface
└── data/                      # 💾 Cache files
    ├── database_structure.json # Complete DB structure + samples
    └── semantic_analysis.json  # Entity classifications + relationships
```

## 🚀 Quick Start

### 1. **Environment Setup**

Create `.env` file with your configuration:

```env
# Azure OpenAI (Required)
AZURE_OPENAI_API_KEY=your_key_here
AZURE_ENDPOINT=https://gyp-weu-02-res01-prdenv-cog01.openai.azure.com/
DEPLOYMENT_NAME=gpt-4.1-mini
MODEL_VERSION=2024-12-01-preview

# Database Connection (Required)
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=YourDB;Trusted_Connection=yes;

# Optional Performance Settings
DISCOVERY_CACHE_HOURS=24
USE_FAST_QUERIES=true
MAX_RESULTS=100
```

### 2. **Install Dependencies**

```bash
pip install pyodbc python-dotenv tqdm langchain-openai
```

### 3. **Run the System**

```bash
python main.py
```

## 📋 Three Core Options

### **Option 1: Database Discovery 🔍**

Discovers and analyzes your complete database structure:

**What it does:**
- Scans ALL tables and views in your database
- Extracts complete column information (names, types, nullability, defaults)
- Identifies primary keys and foreign key relationships  
- Collects 5 sample rows from each table/view for context
- Analyzes view definitions to understand business logic
- Saves everything to `database_structure.json`

**Output Structure:**
```json
{
  "tables": {
    "dbo.Customers": {
      "columns": {
        "CustomerID": "int PRIMARY KEY",
        "CustomerName": "varchar(100)",
        "Email": "varchar(255)",
        "CreatedDate": "datetime"
      },
      "sample_data": [
        {"CustomerID": 1, "CustomerName": "John Doe", "Email": "john@example.com"},
        {"CustomerID": 2, "CustomerName": "Jane Smith", "Email": "jane@example.com"}
      ],
      "foreign_keys": [
        {"column": "CountryID", "references": "dbo.Countries(CountryID)"}
      ],
      "view_definition": "SELECT ... FROM ... WHERE ..." // For views
    }
  }
}
```

**Performance Features:**
- Uses `OPTION (FAST 5)` for quick sampling
- Parallel processing with adaptive batching
- Intelligent caching system
- Progress bars with real-time updates

### **Option 2: Semantic Analysis 🧠**

Uses LLM to classify all entities and discover relationships:

**What it does:**
- Loads database structure from Step 1
- **LLM Entity Classification**: Uses Azure OpenAI to identify business entities
  - Customers, Orders, Products, Payments, Users, etc.
  - Confidence scores for each classification
  - Business role identification (Core, Supporting, Reference)
- **Relationship Discovery**: Finds connections between entities
  - Foreign key relationships from database schema
  - Implicit relationships through naming patterns and data analysis
  - View-based relationships from JOIN patterns
- **Business Domain Analysis**: Identifies industry context
  - E-Commerce, CRM, Healthcare, Financial Services, etc.
  - Generates relevant sample questions
  - Enables domain-specific query capabilities

**LLM Analysis Process:**
```python
# Example LLM prompt for entity classification
"""
Analyze this table: dbo.CustomerPayments
Columns: CustomerID (int), PaymentAmount (decimal), PaymentDate (datetime), PaymentMethodID (int)
Sample: {"CustomerID": 123, "PaymentAmount": 299.99, "PaymentDate": "2024-01-15", "PaymentMethodID": 1}

Classify as business entity with confidence score.
"""

# LLM Response
{
  "entity_type": "Payment",
  "confidence": 0.95,
  "business_role": "Core",
  "reasoning": "Contains payment amounts and customer references"
}
```

**Output:**
- Entity classifications for all tables/views
- Relationship map between entities
- Business domain identification
- Sample questions for interactive querying

### **Option 3: Interactive Queries 💬**

Natural language to SQL conversion and execution:

**What it does:**
- Loads analyzed database structure and entity classifications
- **Smart Table Selection**: Uses fuzzy matching to find relevant tables
- **LLM SQL Generation**: Converts questions to optimized T-SQL
- **Safe Execution**: Runs queries with proper limits and error handling
- **Result Display**: Shows results in readable format

**Example Session:**
```
❓ Query #1: How many customers have made payments?

🔍 Finding relevant tables...
   📋 Found 2 relevant tables
      • dbo.Customers (Customer entity, confidence: 0.92)
      • dbo.CustomerPayments (Payment entity, confidence: 0.95)

⚡ Generating SQL query...
   💾 Generated: SELECT COUNT(DISTINCT c.CustomerID) FROM [dbo].[Customers] c...

🚀 Executing query...
   📊 Results: 1 rows
   1. {'paying_customers': 1247}

⚡ Execution time: 0.234s
```

**Query Capabilities:**
- Customer analysis: "How many customers do we have?"
- Payment analysis: "What's our total revenue this year?"
- Relationship queries: "Show customers with their recent payments"
- Trend analysis: "Monthly sales growth"
- Complex joins across multiple entities

## 🔧 Configuration Options

### **Database Discovery Settings**
```env
# Performance
USE_FAST_QUERIES=true           # Enable OPTION (FAST 5) optimization
MAX_PARALLEL_WORKERS=12         # Parallel processing threads
QUERY_TIMEOUT_SECONDS=30        # Timeout per table/view

# Data Collection
SAMPLES_PER_OBJECT=5            # Sample rows per table/view
EXCLUDE_BACKUP_TABLES=true      # Skip backup/temp tables

# Caching
DISCOVERY_CACHE_HOURS=24        # Cache database structure
```

### **Semantic Analysis Settings**
```env
# LLM Configuration
SEMANTIC_CACHE_HOURS=48         # Cache entity classifications
AZURE_OPENAI_API_KEY=your_key   # Required for entity classification
DEPLOYMENT_NAME=gpt-4.1-mini    # Azure OpenAI model

# Analysis Options
ENTITY_CONFIDENCE_THRESHOLD=0.7 # Minimum confidence for classification
RELATIONSHIP_DISCOVERY=true     # Enable relationship finding
```

### **Query Interface Settings**
```env
# Query Execution
MAX_RESULTS=100                 # Limit query results
COMMAND_TIMEOUT=10              # SQL execution timeout

# Display Options
SHOW_SQL_QUERIES=true          # Display generated SQL
USE_COLORED_OUTPUT=true        # Colored console output
```

## 🗄️ Database Compatibility

**Supported Databases:**
- SQL Server (all versions)
- Azure SQL Database  
- SQL Server Express
- SQL Server LocalDB

**Required Drivers:**
- ODBC Driver 17 for SQL Server (recommended)
- ODBC Driver 13 for SQL Server (supported)

**Connection Examples:**
```env
# Local SQL Server with Windows Authentication
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=YourDB;Trusted_Connection=yes;

# Azure SQL Database
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=yourserver.database.windows.net;Database=YourDB;UID=username;PWD=password;Encrypt=yes;

# Custom instance with specific port
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=localhost,1433;Database=YourDB;UID=username;PWD=password;
```

## 💡 Key Technical Features

### **Complete Database Structure Analysis**
- **Metadata Extraction**: Uses SQL Server system views (`sys.tables`, `sys.columns`, `sys.foreign_keys`)
- **View Definition Analysis**: Extracts `sys.sql_modules` to understand view logic and relationships
- **Constraint Discovery**: Identifies primary keys, foreign keys, unique constraints
- **Data Type Mapping**: Complete column information with nullability and defaults

### **Advanced View Analysis**
```sql
-- Example: Discovering relationships through view definitions
SELECT 
    v.name as view_name,
    m.definition as view_definition
FROM sys.views v
JOIN sys.sql_modules m ON v.object_id = m.object_id
WHERE m.definition LIKE '%JOIN%'
```

### **LLM Entity Classification**
- **Structured Prompts**: Uses JSON-formatted prompts for consistent entity classification
- **Context-Aware**: Combines table names, column names, and sample data for accurate classification
- **Confidence Scoring**: Each classification includes confidence level (0.0-1.0)
- **Business Role Identification**: Classifies entities as Core, Supporting, or Reference

### **Relationship Discovery Methods**
1. **Foreign Key Analysis**: Direct schema relationships
2. **Naming Pattern Matching**: `customer_id` → `customers.id`
3. **View Join Analysis**: Extracts relationships from view definitions
4. **LLM Inference**: AI-powered relationship discovery from data patterns

## 🧪 Sample Usage Scenarios

### **Scenario 1: New Database Exploration**
```bash
# Discover a new database completely
python main.py
> 1  # Database Discovery
> 2  # Semantic Analysis  
> 3  # Interactive Queries

# Ask questions like:
"What tables contain customer information?"
"How are customers connected to orders?"
"Show me the database structure"
```

### **Scenario 2: Business Intelligence**
```bash
# Generate business insights
python main.py
> 3  # Interactive Queries

# Ask business questions:
"What's our total revenue this month?"
"How many active customers do we have?"
"Show top 10 customers by revenue"
"Which products sell the most?"
```

### **Scenario 3: Data Migration Planning**
```bash
# Understand data relationships for migration
python main.py
> 1  # Discover source database
> 2  # Classify all entities
> Check data/database_structure.json for complete schema
> Check data/semantic_analysis.json for relationships
```

## 🔍 Troubleshooting

### **Common Issues**

**1. Database Connection Errors**
```bash
# Test connection
python -c "import pyodbc; pyodbc.connect('your_connection_string')"

# Check ODBC drivers
odbcinst -j
```

**2. Azure OpenAI Issues**  
```bash
# Verify API key and endpoint
curl -H "api-key: YOUR_KEY" "https://your-endpoint.openai.azure.com/openai/deployments?api-version=2024-12-01-preview"
```

**3. Unicode/Greek Text Issues**
- Ensure UTF-8 encoding in connection string
- Check database collation supports Unicode
- Verify console encoding for display

**4. Performance Issues**
- Enable `USE_FAST_QUERIES=true`
- Reduce `MAX_PARALLEL_WORKERS` if system overloaded  
- Check `DISCOVERY_CACHE_HOURS` to avoid re-scanning

### **Debug Mode**
```env
# Enable detailed logging
LOG_LEVEL=DEBUG
USE_COLORED_LOGS=true
```

## 📚 Example Outputs

### **Database Structure (database_structure.json)**
```json
{
  "tables": {
    "dbo.Customers": {
      "columns": {
        "CustomerID": {"type": "int", "primary_key": true},
        "CustomerName": {"type": "varchar(100)", "nullable": false},
        "Email": {"type": "varchar(255)", "nullable": true}
      },
      "sample_data": [
        {"CustomerID": 1, "CustomerName": "ACME Corp", "Email": "info@acme.com"},
        {"CustomerID": 2, "CustomerName": "Tech Solutions", "Email": "hello@tech.com"}
      ],
      "relationships": [
        {"column": "CountryID", "references": "dbo.Countries(CountryID)"}
      ]
    }
  }
}
```

### **Semantic Analysis (semantic_analysis.json)**
```json
{
  "entity_classifications": {
    "dbo.Customers": {
      "entity_type": "Customer",
      "confidence": 0.95,
      "business_role": "Core"
    },
    "dbo.CustomerPayments": {
      "entity_type": "Payment", 
      "confidence": 0.92,
      "business_role": "Core"
    }
  },
  "relationships": [
    {
      "from": "dbo.CustomerPayments",
      "to": "dbo.Customers", 
      "type": "foreign_key",
      "confidence": 1.0
    }
  ],
  "business_domain": {
    "type": "CRM/Financial",
    "confidence": 0.88,
    "sample_questions": [
      "How many customers have made payments?",
      "What's our total revenue?",
      "Show customer payment history"
    ]
  }
}
```

## 🚀 Getting Started Checklist

- [ ] **Install Requirements**: `pip install pyodbc python-dotenv tqdm langchain-openai`
- [ ] **Configure .env**: Set up Azure OpenAI and database connection
- [ ] **Test Connection**: Verify database access and Azure OpenAI API
- [ ] **Run Discovery**: Execute Option 1 to scan database structure
- [ ] **Run Analysis**: Execute Option 2 to classify entities with LLM
- [ ] **Start Querying**: Execute Option 3 to ask natural language questions

## 🎯 Advanced Features

### **Robust Discovery Engine**
- **Adaptive Performance**: Automatically adjusts parallelism based on dataset size
- **Error Recovery**: Retry logic for failed table analysis
- **Progress Tracking**: Real-time progress bars and detailed metrics
- **Resource Management**: Conservative memory and connection usage

### **Intelligent Caching**
- **Smart Cache Invalidation**: Configurable cache expiration times
- **Incremental Updates**: Only re-analyze changed objects
- **Cache Validation**: Automatic cache health checks

### **International Support**
- **Unicode Handling**: Full UTF-8 support throughout the system
- **Greek Text Processing**: Proper handling of Greek characters and text
- **International Characters**: Support for all Unicode character sets

## 🔬 Technical Implementation

### **Database Discovery Process**
1. **Schema Enumeration**: Query `sys.tables` and `sys.views` for object discovery
2. **Metadata Collection**: Extract column information from `INFORMATION_SCHEMA.COLUMNS`
3. **Relationship Mapping**: Analyze `sys.foreign_keys` for explicit relationships
4. **Sample Collection**: Use `OPTION (FAST 5)` for efficient data sampling
5. **View Analysis**: Parse view definitions from `sys.sql_modules`

### **LLM Integration Architecture**
```python
# Example LLM client usage
llm_client = SimpleLLMClient(config)
response = await llm_client.generate_sql(
    "Generate SQL to find customers with payments > $1000"
)
```

### **Query Processing Pipeline**
1. **Question Analysis**: Parse natural language query
2. **Table Selection**: Use fuzzy matching to find relevant tables
3. **Context Building**: Create structured prompt with table schemas
4. **SQL Generation**: Use LLM to generate T-SQL query
5. **Query Execution**: Safe execution with timeout and result limits
6. **Result Formatting**: Display results in user-friendly format

## 📈 Performance Optimization

### **Database Query Optimization**
- **FAST Queries**: Uses `OPTION (FAST n)` for quick sampling
- **Parallel Processing**: Configurable worker threads for large datasets
- **Connection Pooling**: Efficient database connection management
- **Timeout Management**: Prevents hanging on problematic queries

### **Memory Management**
- **Streaming Results**: Process large datasets without memory overflow
- **Adaptive Batching**: Adjusts batch sizes based on system performance
- **Cache Optimization**: Intelligent caching to reduce redundant operations

## 🛡️ Security Features

### **Safe Query Execution**
- **Read-Only Operations**: No UPDATE/DELETE/DROP operations allowed
- **Result Limits**: Automatic row limits to prevent resource exhaustion
- **SQL Injection Prevention**: Parameterized queries where applicable
- **Timeout Protection**: Query timeouts to prevent runaway operations

### **Data Privacy**
- **Sample Data Limits**: Only collects minimal sample data needed
- **No Data Storage**: LLM prompts don't store sensitive data permanently
- **Local Processing**: All analysis happens on your infrastructure

## 🎯 Ready to Get Started?

1. **Setup**: Configure `.env` with your Azure OpenAI and database settings
2. **Discovery**: Run Option 1 to analyze your complete database structure  
3. **Classification**: Run Option 2 to classify entities and find relationships
4. **Query**: Run Option 3 to start asking natural language questions

**Transform your database into an intelligent, queryable knowledge base!** 🚀

---

**Built with Azure OpenAI GPT-4, LangChain, and optimized for real-world business databases with international character support.**

## 📝 License

MIT License - Feel free to use this project for your own database analysis needs.

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review configuration examples
- Ensure all dependencies are installed correctly
- Verify Azure OpenAI and database connectivity

**Happy querying!** 🎉