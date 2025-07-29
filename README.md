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


**Version 2.0** - Simple, Readable, and Maintainable with Enhanced View Handling

A powerful AI-driven system that provides intelligent database analysis and natural language querying capabilities with improved performance and modular architecture.

## 🎯 Key Features

### ✅ **Enhanced Performance**
- **FAST Option Integration**: `OPTION (FAST n)` queries for 2-5x speed improvement
- **Optimized View Handling**: Fixed views showing 0 rows issue
- **Intelligent Caching**: JSON-based cache system with configurable expiration
- **Progress Tracking**: Real-time progress bars and detailed metrics

### ✅ **Advanced AI Capabilities**
- **Semantic Entity Classification**: Automatically classifies database objects by business role
- **Relationship Discovery**: Finds explicit and implicit relationships between entities
- **Business Domain Analysis**: Identifies industry context and business patterns
- **Natural Language Querying**: Converts questions to optimized SQL queries

### ✅ **International Support**
- **Unicode Handling**: Proper Greek text and international character support
- **UTF-8 Encoding**: Throughout the entire system
- **Localized Analysis**: Context-aware business domain identification

### ✅ **Modular Architecture**
- **Clean Separation**: Maintainable, testable, and scalable code structure
- **Easy Integration**: Plugin-style architecture for easy extensions
- **Component Isolation**: Independent modules with clear interfaces

## 🏗️ Architecture Overview

```
semantic-db-rag/
├── main.py                      # 🚀 Entry point with menu options
├── shared/                      # 🔄 Shared components
│   ├── config.py               # ⚙️ Configuration management
│   ├── models.py               # 📊 Data models and structures
│   ├── utils.py                # 🛠️ Utility functions
│   └── logger.py               # 📝 Logging configuration
├── db/                         # 🗄️ Database operations
│   └── discovery.py            # 🔍 Enhanced table/view scanner
├── semantic/                   # 🧠 AI-powered analysis
│   └── analysis.py             # 🧠 LLM-based classification
├── interactive/                # 💬 User interface
│   └── query_interface.py      # 💬 Natural language interface
└── data/                       # 💾 Cache and output
    ├── database_structure.json # 🏗️ Discovery cache
    └── semantic_analysis.json  # 🧠 Semantic cache
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone or download the project
git clone <repository-url>
cd semantic-db-rag

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### 2. **Configuration**

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required Configuration:**
```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_ENDPOINT=https://gyp-weu-02-res01-prdenv-cog01.openai.azure.com/
DEPLOYMENT_NAME=gpt-4.1-mini
MODEL_VERSION=2024-12-01-preview

# Database Connection
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=YourDB;Trusted_Connection=yes;
```

### 3. **Run the System**

```bash
python main.py
```

## 📋 Usage Guide

### **Option 1: Database Discovery 🔍**
- Scans database tables and views with enhanced performance
- Uses `OPTION (FAST n)` for quick sampling
- Handles Unicode text properly
- Saves results to `database_structure.json`

### **Option 2: Semantic Analysis 🧠**
- Classifies entities using AI
- Discovers relationships between tables
- Identifies business domain and industry
- Saves results to `semantic_analysis.json`

### **Option 3: Interactive Queries 💬**
- Natural language to SQL conversion
- Enhanced table relevance scoring
- Optimized query execution
- Real-time result display

### **Option 4: Full Demo 🚀**
- Runs all steps in sequence
- Shows system status and capabilities
- Comprehensive demonstration

## 🔧 Configuration Options

### **Performance Settings**
```env
# Cache expiration (hours)
DISCOVERY_CACHE_HOURS=24
SEMANTIC_CACHE_HOURS=48

# Database timeouts (seconds)
CONNECTION_TIMEOUT=15
COMMAND_TIMEOUT=10

# Query limits
MAX_RESULTS=100
MAX_BATCH_SIZE=5
```

### **Advanced Features**
```env
# Enable FAST queries (recommended)
USE_FAST_QUERIES=true

# Exclude backup tables (recommended)
EXCLUDE_BACKUP_TABLES=true

# Enhanced logging
LOG_LEVEL=INFO
USE_COLORED_LOGS=true
```

## 💡 Key Improvements in Version 2.0

### **Fixed View Handling**
- **Problem**: Views previously showed 0 rows
- **Solution**: Enhanced estimation using `sys.views` metadata
- **Result**: Views now participate fully in semantic analysis

### **FAST Query Optimization**
- **Implementation**: `OPTION (FAST n)` for quick data sampling
- **Performance**: 2-5x speed improvement for complex views
- **Fallback**: Graceful degradation to standard queries

### **Enhanced Unicode Support**
- **Encoding**: UTF-8 throughout the system
- **Greek Text**: Proper handling and display
- **International**: Supports all Unicode characters

### **Modular Architecture**
- **Maintainability**: Clear separation of concerns
- **Testing**: Component-level testing capability
- **Scalability**: Easy to add new features

## 🗄️ Database Compatibility

### **Supported Databases**
- SQL Server (all versions)
- Azure SQL Database
- SQL Server Express
- SQL Server LocalDB

### **Required Drivers**
- ODBC Driver 17 for SQL Server (recommended)
- ODBC Driver 13 for SQL Server (supported)

### **Connection Examples**
```env
# Local SQL Server with Windows Authentication
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=YourDB;Trusted_Connection=yes;

# Azure SQL Database
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=yourserver.database.windows.net;Database=YourDB;UID=username;PWD=password;Encrypt=yes;

# SQL Server with custom port
DATABASE_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=localhost,1433;Database=YourDB;UID=username;PWD=password;
```

## 🧪 Testing

### **Run Tests**
```bash
# Install test dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test modules
pytest tests/test_discovery.py
pytest tests/test_semantic.py
pytest tests/test_queries.py
```

### **Manual Testing**
```bash
# Test database connection
python -c "from db.discovery import DatabaseDiscovery; from shared.config import Config; d = DatabaseDiscovery(Config()); print('✅ Connection OK')"

# Test Azure OpenAI
python -c "from semantic.analysis import SimpleLLMClient; from shared.config import Config; import asyncio; asyncio.run(SimpleLLMClient(Config()).ask('Hello')); print('✅ AI OK')"
```

## 🐛 Troubleshooting

### **Common Issues**

**1. Connection Errors**
```bash
# Check ODBC drivers
odbcinst -j

# Test connection string
sqlcmd -S "your_server" -d "your_database"
```

**2. Unicode Issues**
- Ensure UTF-8 encoding in connection string
- Check database collation settings
- Verify console encoding

**3. Performance Issues**
- Enable FAST queries in configuration
- Check cache settings
- Review excluded table patterns

**4. Memory Issues**
- Reduce MAX_BATCH_SIZE
- Lower MAX_RESULTS
- Enable result pagination

### **Debug Mode**
```env
LOG_LEVEL=DEBUG
USE_COLORED_LOGS=true
```

## 📚 API Reference

### **Main Classes**

#### `DatabaseDiscovery`
```python
from db.discovery import DatabaseDiscovery
from shared.config import Config

discovery = DatabaseDiscovery(Config())
success = await discovery.discover_database(limit=50)
tables = discovery.get_tables()
```

#### `SemanticAnalyzer`
```python
from semantic.analysis import SemanticAnalyzer
from shared.config import Config

analyzer = SemanticAnalyzer(Config())
success = await analyzer.analyze_semantics(tables)
domain = analyzer.get_domain()
```

#### `QueryInterface`
```python
from interactive.query_interface import QueryInterface
from shared.config import Config

interface = QueryInterface(Config())
await interface.start_interactive_session(tables, domain, relationships)
```

## 🤝 Contributing

### **Development Setup**
```bash
# Clone repository
git clone <repository-url>
cd semantic-db-rag

# Install in development mode
pip install -e .[dev,logging,performance]

# Set up pre-commit hooks
pre-commit install
```

### **Code Style**
```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

### **Adding Features**
1. Create feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with Azure OpenAI and LangChain
- Enhanced with FAST query optimization
- Supports international Unicode text
- Designed for real-world business databases

---

**🎯 Ready to transform your database into an intelligent, queryable knowledge base with enhanced performance and international support!**