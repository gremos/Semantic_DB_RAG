# 🧠 Enhanced Semantic Database RAG System

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