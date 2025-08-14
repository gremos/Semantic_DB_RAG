#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database RAG System - Enhanced Query Interface
Simple, readable, and maintainable implementation with template-based constraints
"""

import json
import re
import pyodbc
import time
import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from contextlib import contextmanager

# Enhanced imports for new features
try:
    import sqlglot
    from sqlglot import parse_one, transpile
    from sqlglot.optimizer import optimize
    SQLGLOT_AVAILABLE = True
except ImportError:
    print("❌ SQLGlot not found. Install with: pip install sqlglot")
    sqlglot = None
    SQLGLOT_AVAILABLE = False

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("⚠️ Embeddings not available. Install with: pip install sentence-transformers numpy")
    np = None
    SentenceTransformer = None
    EMBEDDINGS_AVAILABLE = False

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from tqdm import tqdm

# Import shared modules
try:
    from shared.config import Config
    from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
except ImportError as e:
    print(f"❌ Import error: {e}")
    raise


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('semantic_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Query execution statistics"""
    question: str
    tables_selected: List[str] = field(default_factory=list)
    sql_generated: str = ""
    execution_time: float = 0.0
    result_count: int = 0
    error: Optional[str] = None
    retry_count: int = 0
    validation_failures: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class SchemaNode:
    """Schema graph node"""
    table_name: str
    columns: List[Dict]
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class LiveSchemaValidator:
    """Validates columns against actual database schema"""
    
    def __init__(self, config: Config):
        self.config = config
        self.schema_cache = {}
        self._lock = threading.Lock()
    
    def get_actual_columns(self, table_name: str) -> List[str]:
        """Get actual column names from database"""
        with self._lock:
            if table_name in self.schema_cache:
                return self.schema_cache[table_name]
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract clean table name
                clean_table = table_name.split('.')[-1].replace('[', '').replace(']', '')
                
                # Query actual table schema
                schema_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
                """
                
                cursor.execute(schema_query, (clean_table,))
                columns = [row[0] for row in cursor.fetchall()]
                
                if not columns:
                    # Fallback: try to get columns directly from table
                    logger.warning(f"No columns found in INFORMATION_SCHEMA for {clean_table}, trying direct query")
                    try:
                        cursor.execute(f"SELECT TOP 0 * FROM {table_name}")
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    except:
                        logger.error(f"Direct query also failed for {table_name}")
                        return []
                
                # Cache the result
                with self._lock:
                    self.schema_cache[table_name] = columns
                
                logger.info(f"Retrieved {len(columns)} actual columns for {table_name}: {columns[:5]}...")
                return columns
                
        except Exception as e:
            logger.error(f"Failed to get actual columns for {table_name}: {e}")
            return []
    
    def validate_and_fix_columns(self, business_columns: Dict, table_name: str) -> Dict:
        """Validate business columns against actual database schema"""
        actual_columns = self.get_actual_columns(table_name)
        if not actual_columns:
            logger.warning(f"No actual columns found for {table_name}")
            return business_columns
        
        # Convert to lowercase for comparison
        actual_lower = [col.lower() for col in actual_columns]
        actual_map = {col.lower(): col for col in actual_columns}
        
        validated_columns = {}
        
        for category, cols in business_columns.items():
            validated_cols = []
            
            for col in cols:
                col_lower = col.lower()
                
                # Direct match
                if col_lower in actual_lower:
                    validated_cols.append(actual_map[col_lower])
                    logger.debug(f"Column {col} found as {actual_map[col_lower]}")
                else:
                    # Find similar columns
                    similar = self._find_similar_column(col_lower, actual_columns, category)
                    if similar:
                        validated_cols.append(similar)
                        logger.info(f"Column {col} replaced with similar {similar}")
                    else:
                        logger.warning(f"Column {col} not found in {table_name}")
            
            if validated_cols:
                validated_columns[category] = validated_cols
        
        # Add missing categories by scanning actual columns
        validated_columns.update(self._discover_missing_categories(actual_columns, validated_columns))
        
        logger.info(f"Validated columns for {table_name}: {validated_columns}")
        return validated_columns
    
    def _find_similar_column(self, target_col: str, actual_columns: List[str], category: str) -> Optional[str]:
        """Find similar column in actual schema"""
        target_lower = target_col.lower()
        
        # Category-specific patterns
        patterns = {
            'customer': ['customer', 'client', 'account', 'user', 'owner', 'billing'],
            'amount': ['amount', 'price', 'total', 'value', 'cost', 'fee', 'charge'],
            'date': ['date', 'time', 'created', 'modified', 'signed', 'started'],
            'complaint': ['case', 'ticket', 'issue', 'problem', 'type'],
            'content': ['description', 'content', 'text', 'message', 'comment']
        }
        
        category_patterns = patterns.get(category, [])
        
        # Look for exact partial matches first
        for col in actual_columns:
            col_lower = col.lower()
            if target_lower in col_lower or col_lower in target_lower:
                return col
        
        # Look for pattern matches
        for col in actual_columns:
            col_lower = col.lower()
            for pattern in category_patterns:
                if pattern in col_lower:
                    return col
        
        return None
    
    def _discover_missing_categories(self, actual_columns: List[str], existing: Dict) -> Dict:
        """Discover missing business categories from actual columns"""
        discovered = {}
        
        patterns = {
            'customer': ['customer', 'client', 'account', 'user', 'owner', 'billing'],
            'amount': ['amount', 'price', 'total', 'value', 'cost', 'fee'],
            'date': ['date', 'time', 'created', 'modified', 'signed'],
            'id': ['id', 'key', 'guid']
        }
        
        for category, keywords in patterns.items():
            if category not in existing:
                matches = []
                for col in actual_columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in keywords):
                        matches.append(col)
                
                if matches:
                    discovered[category] = matches[:3]  # Limit to 3
        
        return discovered
    
    @contextmanager
    def _get_connection(self):
        """Get database connection"""
        conn = None
        try:
            conn_string = self.config.get_database_connection_string()
            if 'timeout=' not in conn_string.lower():
                conn_string += ';timeout=10'  # Quick timeout for schema queries
            
            conn = pyodbc.connect(conn_string, autocommit=True)
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
            
            yield conn
        except Exception as e:
            logger.error(f"Schema validation connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    """SQL security validation using AST parsing"""
    
    ALLOWED_STATEMENTS = {'SELECT'}
    BLOCKED_KEYWORDS = {
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 
        'ALTER', 'CREATE', 'EXEC', 'EXECUTE', 'xp_cmdshell'
    }
    
    def __init__(self):
        self.violation_count = 0
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL for security and safety"""
        if not sql or not sql.strip():
            return False, "Empty SQL query"
        
        try:
            # Parse with SQLGlot if available
            if SQLGLOT_AVAILABLE:
                parsed = parse_one(sql, dialect="tsql")
                return self._validate_ast(parsed)
            else:
                return self._validate_regex(sql)
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def _validate_ast(self, parsed) -> Tuple[bool, str]:
        """Validate using AST"""
        try:
            # Check statement type
            if not isinstance(parsed, sqlglot.exp.Select):
                return False, "Only SELECT statements allowed"
            
            # Check for blocked functions/procedures
            for node in parsed.walk():
                if isinstance(node, sqlglot.exp.Anonymous):
                    func_name = str(node.this).upper()
                    if any(blocked in func_name for blocked in self.BLOCKED_KEYWORDS):
                        return False, f"Blocked function: {func_name}"
            
            return True, "Valid"
        except Exception as e:
            return False, f"AST validation error: {str(e)}"
    
    def _validate_regex(self, sql: str) -> Tuple[bool, str]:
        """Fallback regex validation"""
        sql_upper = sql.upper()
        
        # Check for blocked keywords
        for keyword in self.BLOCKED_KEYWORDS:
            if re.search(rf'\b{keyword}\b', sql_upper):
                self.violation_count += 1
                return False, f"Blocked keyword: {keyword}"
        
        # Must start with SELECT
        if not re.match(r'^\s*(WITH|SELECT)\s', sql_upper):
            return False, "Only SELECT statements allowed"
        
        return True, "Valid"


class EnhancedSQLGenerator:
    """Enhanced SQL generator with templates and constraints - No external dependencies"""
    
    def __init__(self):
        self.templates = {
            'customer_count': """
            SELECT COUNT(DISTINCT {customer_column}) as total_customers
            FROM {table}
            WHERE {conditions}
            """,
            
            'customer_list': """
            SELECT TOP {limit} 
                {customer_columns}
            FROM {table}
            WHERE {conditions}
            ORDER BY {order_column} DESC
            """,
            
            'payment_analysis': """
            SELECT 
                {customer_column},
                SUM({amount_column}) as total_paid,
                COUNT(*) as payment_count
            FROM {table}
            WHERE {amount_column} > 0
                AND {date_filter}
            GROUP BY {customer_column}
            ORDER BY total_paid DESC
            """,
            
            'complaint_analysis': """
            SELECT TOP {limit}
                {customer_columns},
                {complaint_columns},
                {date_column}
            FROM {table}
            WHERE {complaint_conditions}
            ORDER BY {date_column} DESC
            """,
            
            'general_query': """
            SELECT TOP {limit}
                {columns}
            FROM {table}
            WHERE {conditions}
            ORDER BY {order_column}
            """
        }
        
        self.validation_patterns = {
            'required': [
                (r'^\s*SELECT\s+', "Must start with SELECT"),
                (r'\bFROM\s+[\w\[\]\.]+', "Must have FROM clause"),
            ],
            'forbidden': [
                (r'\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|CREATE|EXEC)\b', "Only SELECT allowed"),
                (r'\bxp_\w+', "System procedures not allowed"),
            ]
        }
        
        self.intent_keywords = {
            'customer_count': ['how many customers', 'count customers', 'number of customers'],
            'customer_list': ['list customers', 'show customers', 'customers who'],
            'payment_analysis': ['payment', 'paid', 'revenue', 'amount', 'total paid'],
            'complaint_analysis': ['complaint', 'issue', 'problem', 'error', 'ticket']
        }
        
        logger.info("Enhanced SQL generator initialized with templates")
    
    def detect_intent(self, question: str) -> str:
        """Detect query intent from question"""
        q_lower = question.lower()
        
        # Check for count queries first
        if any(word in q_lower for word in ['how many', 'count', 'number of']):
            return 'customer_count'
        
        # Check specific business intents
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in q_lower for keyword in keywords):
                return intent
        
        return 'customer_list'  # Default fallback
    
    def extract_parameters(self, question: str, context: Dict) -> Dict:
        """Extract parameters from question and context"""
        params = {
            'limit': 1000,
            'conditions': '1=1',
            'date_filter': '1=1',
            'order_column': 'ID',
            'table': 'unknown_table',
            'columns': '*'
        }
        
        # Extract limit from question
        limit_match = re.search(r'\btop\s+(\d+)\b|\b(\d+)\s+(?:customers?|rows?|records?)\b', question.lower())
        if limit_match:
            limit_val = limit_match.group(1) or limit_match.group(2)
            params['limit'] = min(int(limit_val), 5000)  # Cap at 5000 for safety
        
        # Extract year filter
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            year = year_match.group(1)
            params['date_filter'] = f"YEAR([date_column]) = {year}"
        
        # Special conditions for paid customers
        if 'paid' in question.lower():
            params['conditions'] = '[amount_column] > 0'
        
        # Extract business columns from context - with fallbacks
        business_cols = context.get('business_columns', {})
        
        # Set customer columns with intelligent fallbacks
        if business_cols.get('customer'):
            params['customer_column'] = business_cols['customer'][0]
            params['customer_columns'] = ', '.join(business_cols['customer'][:3])
        else:
            # Smart fallback: look for ID columns that might represent customers
            all_tables = context.get('tables', [])
            if all_tables:
                for col_name in all_tables[0].get('columns', []):
                    col_lower = col_name.lower()
                    if any(word in col_lower for word in ['customer', 'client', 'account', 'billing']):
                        params['customer_column'] = col_name
                        params['customer_columns'] = col_name
                        break
                else:
                    # Final fallback: use any ID column
                    for col_name in all_tables[0].get('columns', []):
                        if 'id' in col_name.lower():
                            params['customer_column'] = col_name
                            params['customer_columns'] = col_name
                            break
                    else:
                        # Ultimate fallback
                        params['customer_column'] = '*'
                        params['customer_columns'] = '*'
        
        # Set amount columns with fallbacks
        if business_cols.get('amount'):
            params['amount_column'] = business_cols['amount'][0]
        else:
            # Smart fallback: look for amount/price columns
            all_tables = context.get('tables', [])
            if all_tables:
                for col_name in all_tables[0].get('columns', []):
                    col_lower = col_name.lower()
                    if any(word in col_lower for word in ['amount', 'price', 'total', 'value', 'cost']):
                        params['amount_column'] = col_name
                        break
                else:
                    params['amount_column'] = '1'  # Fallback to literal
        
        # Replace placeholder in conditions
        if '[amount_column]' in params['conditions']:
            params['conditions'] = params['conditions'].replace('[amount_column]', params['amount_column'])
        
        # Set date columns with fallbacks
        if business_cols.get('date'):
            params['date_column'] = business_cols['date'][0]
            params['order_column'] = business_cols['date'][0]
        else:
            # Smart fallback: look for date columns
            all_tables = context.get('tables', [])
            if all_tables:
                for col_name in all_tables[0].get('columns', []):
                    col_lower = col_name.lower()
                    if any(word in col_lower for word in ['date', 'time', 'created', 'modified', 'signed']):
                        params['date_column'] = col_name
                        params['order_column'] = col_name
                        break
                else:
                    params['date_column'] = 'GETDATE()'  # Fallback to current date
                    params['order_column'] = params['customer_column']
        
        # Replace placeholder in date_filter
        if '[date_column]' in params['date_filter']:
            params['date_filter'] = params['date_filter'].replace('[date_column]', params['date_column'])
        
        # Set complaint columns
        if business_cols.get('complaint'):
            params['complaint_columns'] = ', '.join(business_cols['complaint'][:3])
            params['complaint_conditions'] = f"{business_cols['complaint'][0]} IS NOT NULL"
        
        # Set table name
        if context.get('tables') and len(context['tables']) > 0:
            params['table'] = context['tables'][0]['name']
        
        # Set general columns
        all_columns = []
        for category, cols in business_cols.items():
            all_columns.extend(cols[:2])  # Take first 2 from each category
        
        if all_columns:
            params['columns'] = ', '.join(all_columns[:10])  # Limit to 10 columns
        elif context.get('tables') and len(context['tables']) > 0:
            # Fallback to first few columns from table
            table_cols = context['tables'][0].get('columns', [])[:5]
            params['columns'] = ', '.join(table_cols) if table_cols else '*'
        
        return params
    
    def generate_from_template(self, question: str, context: Dict) -> Tuple[str, str]:
        """Generate SQL using templates"""
        try:
            # Detect intent
            intent = self.detect_intent(question)
            logger.info(f"Detected intent: {intent}")
            
            # Get template
            template = self.templates.get(intent, self.templates['general_query'])
            
            # Extract parameters
            params = self.extract_parameters(question, context)
            logger.info(f"Extracted parameters: {params}")
            
            # Check for missing critical parameters
            critical_params = ['table', 'customer_column']
            missing_params = [p for p in critical_params if params.get(p) in [None, 'unknown_table', '*']]
            
            if missing_params:
                logger.warning(f"Missing critical parameters: {missing_params}")
                logger.info(f"Available context: {context}")
            
            # Generate SQL from template
            sql = template.format(**params)
            sql = re.sub(r'\s+', ' ', sql.strip())  # Clean whitespace
            
            logger.info(f"Generated SQL from template: {sql}")
            return sql, intent
            
        except KeyError as e:
            logger.error(f"Template parameter missing: {e}")
            logger.error(f"Available parameters: {list(params.keys()) if 'params' in locals() else 'None'}")
            logger.error(f"Template requires: {re.findall(r'{(\w+)}', template)}")
            return "", f"Missing parameter: {e}"
        except Exception as e:
            logger.error(f"Template generation error: {e}")
            return "", f"Generation error: {e}"
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Validate generated SQL"""
        if not sql or not sql.strip():
            return False, "Empty SQL"
        
        sql_upper = sql.upper()
        
        # Check required patterns
        for pattern, message in self.validation_patterns['required']:
            if not re.search(pattern, sql_upper):
                return False, message
        
        # Check forbidden patterns
        for pattern, message in self.validation_patterns['forbidden']:
            if re.search(pattern, sql_upper):
                return False, message
        
        return True, "Valid"
    
    async def generate_constrained_sql(self, question: str, context: Dict, 
                                     llm_fallback=None) -> Tuple[str, int]:
        """Generate SQL with constraints (main interface)"""
        
        # Strategy 1: Template-based generation (preferred)
        sql, intent = self.generate_from_template(question, context)
        
        if sql:
            is_valid, error = self.validate_sql(sql)
            if is_valid:
                logger.info(f"Template generation successful: {intent}")
                return sql, 0  # Success on first try
            else:
                logger.warning(f"Template validation failed: {error}")
        
        # Strategy 2: LLM fallback with constraints
        if llm_fallback:
            logger.info("Falling back to LLM generation")
            return await self._llm_with_constraints(question, context, llm_fallback)
        
        return "", 1
    
    async def _llm_with_constraints(self, question: str, context: Dict, 
                                  llm_client, max_retries: int = 3) -> Tuple[str, int]:
        """LLM generation with validation and retry"""
        
        for attempt in range(max_retries):
            try:
                # Enhanced system prompt with strict constraints
                system_prompt = f"""You are a T-SQL expert. Generate ONLY SELECT statements.

STRICT RULES:
- Start with SELECT TOP [number]
- Use exact column names from context
- Only use tables provided in context
- Include WHERE clause for filtering
- Add ORDER BY for consistency
- NO DDL/DML operations (no DROP, DELETE, UPDATE, INSERT, etc.)

AVAILABLE CONTEXT:
{json.dumps(context, indent=2)}

Return ONLY the SQL query - no explanations."""

                user_prompt = f'Question: "{question}"\n\nGenerate T-SQL SELECT statement:'
                
                # Add error feedback for retries
                if attempt > 0:
                    user_prompt += f"\n\nPrevious attempt failed. Ensure you follow all rules above."
                
                # Generate using LLM
                response = await llm_client._generate_sql_direct(system_prompt, user_prompt)
                sql = self._clean_sql_response(response)
                
                if sql:
                    is_valid, error = self.validate_sql(sql)
                    if is_valid:
                        logger.info(f"LLM generation successful on attempt {attempt + 1}")
                        return sql, attempt
                    else:
                        logger.warning(f"LLM validation failed (attempt {attempt + 1}): {error}")
                
            except Exception as e:
                logger.error(f"LLM generation error (attempt {attempt + 1}): {e}")
        
        return "", max_retries
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean LLM response to extract SQL"""
        if not response:
            return ""
        
        # Remove markdown and comments
        cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = re.sub(r'--.*$', '', cleaned, flags=re.MULTILINE)
        
        # Extract SQL lines
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        sql_lines = []
        
        for line in lines:
            if line.upper().startswith(('SELECT', 'WITH')) or sql_lines:
                if not line.startswith(('--', '/*', 'Note:', 'Here', 'The', 'This')):
                    sql_lines.append(line)
        
        return ' '.join(sql_lines).rstrip(';').strip()


class SchemaEmbedder:
    """Creates and manages schema embeddings (optional)"""
    
    def __init__(self):
        self.model = None
        self.embeddings_cache = {}
        self.available = EMBEDDINGS_AVAILABLE
        
        if self.available:
            self._init_model()
    
    def _init_model(self):
        """Initialize embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            return
        
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Schema embedding model loaded")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.available = False
    
    def embed_schema(self, tables: List[TableInfo]) -> Dict[str, np.ndarray]:
        """Create embeddings for schema elements"""
        if not self.available or not self.model:
            return {}
        
        embeddings = {}
        
        # Progress bar for embedding creation
        with tqdm(total=len(tables), desc="Creating embeddings") as pbar:
            for table in tables:
                # Create table description
                description = self._create_table_description(table)
                cache_key = hashlib.md5(description.encode()).hexdigest()
                
                if cache_key in self.embeddings_cache:
                    embeddings[table.full_name] = self.embeddings_cache[cache_key]
                else:
                    embedding = self.model.encode([description])[0]
                    embeddings[table.full_name] = embedding
                    self.embeddings_cache[cache_key] = embedding
                
                pbar.update(1)
        
        return embeddings
    
    def _create_table_description(self, table: TableInfo) -> str:
        """Create searchable description of table"""
        parts = [
            f"Table: {table.name}",
            f"Type: {table.entity_type}",
            f"Columns: {', '.join([col['name'] for col in table.columns[:10]])}",
        ]
        
        # Add sample data context
        if table.sample_data:
            sample_values = []
            for row in table.sample_data[:3]:
                for value in row.values():
                    if isinstance(value, str) and len(value) > 5:
                        sample_values.append(value[:50])
            if sample_values:
                parts.append(f"Sample content: {', '.join(sample_values[:5])}")
        
        return " | ".join(parts)
    
    def find_similar_tables(self, query: str, embeddings: Dict[str, np.ndarray], 
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """Find tables similar to query using embeddings"""
        if not self.available or not self.model or not embeddings:
            return []
        
        query_embedding = self.model.encode([query])[0]
        similarities = []
        
        for table_name, table_embedding in embeddings.items():
            similarity = np.dot(query_embedding, table_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(table_embedding)
            )
            similarities.append((table_name, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class SchemaGraphBuilder:
    """Builds schema relationship graph from metadata"""
    
    def __init__(self):
        self.graph = {}
        self.pk_fk_relationships = []
    
    def build_graph(self, tables: List[TableInfo]) -> Dict[str, SchemaNode]:
        """Build comprehensive schema graph"""
        logger.info(f"Building schema graph for {len(tables)} tables")
        
        # Create nodes
        nodes = {}
        for table in tables:
            node = SchemaNode(
                table_name=table.full_name,
                columns=table.columns,
                primary_keys=self._extract_primary_keys(table),
                foreign_keys=self._extract_foreign_keys(table)
            )
            nodes[table.full_name] = node
        
        # Discover relationships
        self._discover_relationships(nodes)
        
        logger.info(f"Schema graph built with {len(self.pk_fk_relationships)} relationships")
        return nodes
    
    def _extract_primary_keys(self, table: TableInfo) -> List[str]:
        """Extract primary key columns"""
        pks = []
        for col in table.columns:
            col_name = col['name'].lower()
            # Common PK patterns
            if (col_name.endswith('_id') or col_name == 'id' or 
                col_name.startswith('pk_') or 'primary' in col_name):
                pks.append(col['name'])
        return pks
    
    def _extract_foreign_keys(self, table: TableInfo) -> List[Dict]:
        """Extract foreign key relationships"""
        fks = []
        for col in table.columns:
            col_name = col['name'].lower()
            # Common FK patterns
            if col_name.endswith('_id') and col_name != 'id':
                # Infer target table
                target_table = col_name[:-3]  # Remove '_id'
                fks.append({
                    'column': col['name'],
                    'references_table': target_table,
                    'confidence': 0.8
                })
        return fks
    
    def _discover_relationships(self, nodes: Dict[str, SchemaNode]):
        """Discover relationships between nodes"""
        for table_name, node in nodes.items():
            for fk in node.foreign_keys:
                target_pattern = fk['references_table']
                
                # Find matching tables
                for target_name in nodes.keys():
                    if target_pattern in target_name.lower():
                        relationship = f"{table_name} -> {target_name} (FK: {fk['column']})"
                        node.relationships.append(relationship)
                        self.pk_fk_relationships.append({
                            'from_table': table_name,
                            'to_table': target_name,
                            'join_condition': f"{table_name}.{fk['column']} = {target_name}.id",
                            'confidence': fk['confidence']
                        })
    
    def get_join_path(self, table1: str, table2: str) -> Optional[str]:
        """Find join path between two tables"""
        for rel in self.pk_fk_relationships:
            if ((rel['from_table'] == table1 and rel['to_table'] == table2) or
                (rel['from_table'] == table2 and rel['to_table'] == table1)):
                return rel['join_condition']
        return None


class LLMClient:
    """Enhanced LLM client with template-based constraints and retry logic"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version="2024-12-01-preview",  # Use specified version
            request_timeout=60,
            temperature=0.1,  # Lower temperature for more deterministic SQL
            max_retries=3
        )
        
        self.sql_generator = EnhancedSQLGenerator()  # Replace SQLGrammar
        self.security_validator = SecurityValidator()
        self.generation_stats = defaultdict(int)
    
    async def generate_sql_with_constraints(self, system_prompt: str, user_prompt: str,
                                          context: Dict = None, max_retries: int = 3) -> Tuple[str, int]:
        """Generate SQL with template constraints and retry logic"""
        
        # Extract question from user_prompt
        question = user_prompt
        if '"' in user_prompt:
            # Extract question from quotes
            match = re.search(r'"([^"]+)"', user_prompt)
            if match:
                question = match.group(1)
        
        # Use enhanced SQL generator
        sql, retry_count = await self.sql_generator.generate_constrained_sql(
            question, 
            context or {}, 
            llm_fallback=self
        )
        
        if sql:
            # Final security validation
            is_safe, safety_msg = self.security_validator.validate_sql(sql)
            if not is_safe:
                logger.warning(f"Security validation failed: {safety_msg}")
                self.generation_stats['security_failure'] += 1
                return "", retry_count + 1
            
            # Optimize with SQLGlot if available
            optimized_sql = self._optimize_sql(sql)
            
            self.generation_stats['success'] += 1
            return optimized_sql, retry_count
        
        self.generation_stats['failure'] += 1
        return "", retry_count
    
    async def _generate_sql_direct(self, system_prompt: str, user_prompt: str) -> str:
        """Direct LLM generation for fallback"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Direct LLM generation error: {e}")
            return ""
    
    def _optimize_sql(self, sql: str) -> str:
        """Optimize SQL using SQLGlot"""
        if not SQLGLOT_AVAILABLE or not sql:
            return sql
        
        try:
            # Parse and optimize
            parsed = parse_one(sql, dialect="tsql")
            optimized = optimize(parsed, schema={})
            
            # Transpile back to T-SQL with formatting
            formatted = transpile(str(optimized), read="tsql", write="tsql", pretty=True)
            return formatted[0] if formatted else sql
            
        except Exception as e:
            logger.debug(f"SQL optimization failed: {e}")
            return sql


class SmartTableSelector:
    """Enhanced table selection using embeddings and business logic"""
    
    def __init__(self, tables: List[TableInfo], schema_embedder: SchemaEmbedder,
                 schema_graph: Dict[str, SchemaNode]):
        self.tables = tables
        self.embedder = schema_embedder
        self.schema_graph = schema_graph
        self.embeddings = {}
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize table embeddings"""
        if self.embedder.available and self.embedder.model:
            self.embeddings = self.embedder.embed_schema(self.tables)
            logger.info(f"Initialized embeddings for {len(self.embeddings)} tables")
        else:
            logger.info("Embeddings not available - using keyword-based selection only")
    
    async def select_tables(self, question: str, max_tables: int = 8) -> List[TableInfo]:
        """Select relevant tables using multiple strategies"""
        candidates = []
        
        # Strategy 1: Embedding-based similarity (if available)
        if self.embeddings:
            similar_tables = self.embedder.find_similar_tables(question, self.embeddings, 20)
            embedding_candidates = [(name, score, 'embedding') for name, score in similar_tables]
            candidates.extend(embedding_candidates)
        
        # Strategy 2: Keyword matching with business logic
        keyword_candidates = self._keyword_based_selection(question)
        candidates.extend([(name, score, 'keyword') for name, score in keyword_candidates])
        
        # Strategy 3: Schema graph relationships
        if candidates:
            related_candidates = self._find_related_tables(candidates[:10])
            candidates.extend([(name, score, 'related') for name, score in related_candidates])
        
        # Combine and rank candidates
        ranked_tables = self._rank_and_select(candidates, max_tables)
        
        logger.info(f"Selected {len(ranked_tables)} tables from {len(candidates)} candidates")
        return ranked_tables
    
    def _keyword_based_selection(self, question: str) -> List[Tuple[str, float]]:
        """Select tables based on keyword matching"""
        q_lower = question.lower()
        keywords = set(q_lower.split())
        
        # Expand with business synonyms
        business_synonyms = {
            'customer': ['client', 'account', 'contact', 'user'],
            'payment': ['invoice', 'billing', 'revenue', 'transaction'],
            'complaint': ['issue', 'problem', 'ticket', 'support'],
            'order': ['purchase', 'sale', 'transaction']
        }
        
        expanded_keywords = set(keywords)
        for keyword in keywords:
            if keyword in business_synonyms:
                expanded_keywords.update(business_synonyms[keyword])
        
        candidates = []
        for table in self.tables:
            if table.row_count == 0:
                continue
                
            score = 0
            table_name_lower = table.full_name.lower()
            
            # Table name scoring
            for keyword in expanded_keywords:
                if keyword in table_name_lower:
                    score += 5
            
            # Column name scoring
            for col in table.columns:
                col_name_lower = col['name'].lower()
                for keyword in expanded_keywords:
                    if keyword in col_name_lower:
                        score += 2
            
            # Entity type scoring
            if hasattr(table, 'entity_type') and table.entity_type != 'Unknown':
                entity_lower = table.entity_type.lower()
                for keyword in expanded_keywords:
                    if keyword in entity_lower:
                        score += 3
            
            if score > 0:
                candidates.append((table.full_name, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _find_related_tables(self, primary_candidates: List[Tuple]) -> List[Tuple[str, float]]:
        """Find tables related to primary candidates through schema graph"""
        related = []
        primary_table_names = [name for name, _, _ in primary_candidates[:5]]
        
        for table_name in primary_table_names:
            if table_name in self.schema_graph:
                node = self.schema_graph[table_name]
                for relationship in node.relationships:
                    # Extract related table name from relationship string
                    match = re.search(r'-> ([^\s]+)', relationship)
                    if match:
                        related_table = match.group(1)
                        related.append((related_table, 2.0))  # Lower score for related tables
        
        return related
    
    def _rank_and_select(self, candidates: List[Tuple], max_tables: int) -> List[TableInfo]:
        """Rank and select final tables"""
        # Combine scores by table name
        table_scores = defaultdict(float)
        table_sources = defaultdict(list)
        
        for name, score, source in candidates:
            table_scores[name] += score
            table_sources[name].append(source)
        
        # Bonus for multiple selection strategies
        for name in table_scores:
            if len(set(table_sources[name])) > 1:
                table_scores[name] *= 1.2
        
        # Sort and select
        ranked_names = sorted(table_scores.keys(), key=lambda x: table_scores[x], reverse=True)
        selected_names = ranked_names[:max_tables]
        
        # Return TableInfo objects
        selected_tables = []
        for table in self.tables:
            if table.full_name in selected_names:
                selected_tables.append(table)
        
        return selected_tables


class RelationshipMapper:
    """Maps relationships between selected tables"""
    
    def __init__(self, schema_graph: Dict[str, SchemaNode]):
        self.schema_graph = schema_graph
    
    def find_relationships(self, tables: List[TableInfo]) -> List[Dict]:
        """Find relationships between tables"""
        relationships = []
        table_names = [t.full_name for t in tables]
        
        # Use schema graph for relationships
        for i, table1_name in enumerate(table_names):
            for table2_name in table_names[i+1:]:
                join_condition = self._find_join_condition(table1_name, table2_name)
                if join_condition:
                    relationships.append({
                        'table1': table1_name,
                        'table2': table2_name,
                        'join_condition': join_condition,
                        'source': 'schema_graph'
                    })
        
        logger.info(f"Found {len(relationships)} relationships")
        return relationships
    
    def _find_join_condition(self, table1: str, table2: str) -> Optional[str]:
        """Find join condition between two tables"""
        node1 = self.schema_graph.get(table1)
        node2 = self.schema_graph.get(table2)
        
        if not (node1 and node2):
            return None
        
        # Check foreign key relationships
        for fk in node1.foreign_keys:
            if fk['references_table'] in table2.lower():
                return f"{table1}.{fk['column']} = {table2}.id"
        
        for fk in node2.foreign_keys:
            if fk['references_table'] in table1.lower():
                return f"{table2}.{fk['column']} = {table1}.id"
        
        return None


class SmartSQLGenerator:
    """Enhanced SQL generator with business logic and live schema validation"""
    
    def __init__(self, llm_client: LLMClient, schema_graph: Dict[str, SchemaNode], config: Config):
        self.llm = llm_client
        self.schema_graph = schema_graph
        self.schema_validator = LiveSchemaValidator(config)
    
    async def generate_smart_sql(self, question: str, tables: List[TableInfo]) -> Tuple[str, int]:
        """Generate SQL with smart business logic and live schema validation"""
        
        # Build enhanced context with live validation
        context = self._build_enhanced_context_with_validation(tables, question)
        
        # Debug: Show what context we built
        logger.info(f"Built context for {len(tables)} tables")
        logger.info(f"Business columns found: {list(context.get('business_columns', {}).keys())}")
        
        # Use enhanced LLM client with templates
        sql, retry_count = await self.llm.generate_sql_with_constraints(
            system_prompt="",  # System prompt handled in generator
            user_prompt=question,
            context=context,
            max_retries=3
        )
        
        return sql, retry_count
    
    def _build_enhanced_context_with_validation(self, tables: List[TableInfo], question: str) -> Dict:
        """Build enhanced context with live schema validation"""
        context = {
            'tables': [],
            'business_columns': {},
            'relationships': []
        }
        
        # Analyze tables for business columns with live validation
        all_business_cols = defaultdict(list)
        
        for table in tables:
            table_info = {
                'name': table.full_name,
                'entity_type': getattr(table, 'entity_type', 'Unknown'),
                'columns': [col['name'] for col in table.columns],
                'row_count': table.row_count
            }
            
            # Categorize business columns for this table (from cached analysis)
            cached_business_cols = self._categorize_columns(table.columns)
            
            # Validate against actual database schema
            validated_business_cols = self.schema_validator.validate_and_fix_columns(
                cached_business_cols, table.full_name
            )
            
            # Store per-table business columns
            context['business_columns'][table.full_name] = validated_business_cols
            
            # Aggregate all business columns across tables
            for category, cols in validated_business_cols.items():
                all_business_cols[category].extend(cols)
            
            context['tables'].append(table_info)
        
        # Flatten business columns for template generation
        # Use the most relevant table's columns, with fallbacks from others
        if context['tables']:
            primary_table = context['tables'][0]['name']
            primary_business_cols = context['business_columns'].get(primary_table, {})
            
            # Create flattened structure with primary table's columns first
            flattened_cols = {}
            for category in ['customer', 'amount', 'date', 'complaint', 'content', 'id']:
                # Start with primary table's columns
                cols = primary_business_cols.get(category, [])
                
                # Add from other tables if primary is empty
                if not cols:
                    cols = all_business_cols.get(category, [])
                
                if cols:
                    flattened_cols[category] = cols
            
            # Set as top-level business_columns for template access
            context['business_columns'] = flattened_cols
            
            # Log final validated columns
            logger.info(f"Final validated business columns: {flattened_cols}")
        
        return context
    
    def _categorize_columns(self, columns: List[Dict]) -> Dict:
        """Categorize columns by business purpose"""
        categories = {
            'customer': [],
            'date': [],
            'amount': [],
            'complaint': [],
            'content': [],
            'id': []
        }
        
        for col in columns:
            name_lower = col['name'].lower()
            
            # Customer/Client patterns (more aggressive)
            if any(word in name_lower for word in [
                'customer', 'client', 'account', 'billing', 'contact', 
                'user', 'subscriber', 'member', 'tenant', 'owner'
            ]):
                categories['customer'].append(col['name'])
            
            # Date patterns (more comprehensive)
            elif any(word in name_lower for word in [
                'date', 'time', 'created', 'modified', 'updated', 'signed', 
                'started', 'ended', 'timestamp', 'when'
            ]):
                categories['date'].append(col['name'])
            
            # Amount/Money patterns (more comprehensive)
            elif any(word in name_lower for word in [
                'amount', 'total', 'price', 'value', 'cost', 'fee', 'charge',
                'payment', 'revenue', 'income', 'sum', 'final', 'net', 'gross'
            ]):
                categories['amount'].append(col['name'])
            
            # Complaint/Issue patterns
            elif any(word in name_lower for word in [
                'complaint', 'issue', 'problem', 'error', 'fault', 'defect',
                'ticket', 'case', 'incident', 'alert'
            ]):
                categories['complaint'].append(col['name'])
            
            # Content/Description patterns
            elif any(word in name_lower for word in [
                'description', 'content', 'text', 'message', 'comment', 'note',
                'details', 'info', 'summary', 'subject', 'title'
            ]):
                categories['content'].append(col['name'])
            
            # ID patterns (catch-all for identifiers)
            elif any(word in name_lower for word in ['id', 'key', 'ref', 'guid', 'uuid']):
                categories['id'].append(col['name'])
        
        return {k: v for k, v in categories.items() if v}
    
    def _categorize_columns(self, columns: List[Dict]) -> Dict:
        """Categorize columns by business purpose"""
        categories = {
            'customer': [],
            'date': [],
            'amount': [],
            'complaint': [],
            'content': [],
            'id': []
        }
        
        for col in columns:
            name_lower = col['name'].lower()
            
            # Customer/Client patterns (more aggressive)
            if any(word in name_lower for word in [
                'customer', 'client', 'account', 'billing', 'contact', 
                'user', 'subscriber', 'member', 'tenant'
            ]):
                categories['customer'].append(col['name'])
            
            # Date patterns (more comprehensive)
            elif any(word in name_lower for word in [
                'date', 'time', 'created', 'modified', 'updated', 'signed', 
                'started', 'ended', 'timestamp', 'when'
            ]):
                categories['date'].append(col['name'])
            
            # Amount/Money patterns (more comprehensive)
            elif any(word in name_lower for word in [
                'amount', 'total', 'price', 'value', 'cost', 'fee', 'charge',
                'payment', 'revenue', 'income', 'sum', 'final', 'net', 'gross'
            ]):
                categories['amount'].append(col['name'])
            
            # Complaint/Issue patterns
            elif any(word in name_lower for word in [
                'complaint', 'issue', 'problem', 'error', 'fault', 'defect',
                'ticket', 'case', 'incident', 'alert'
            ]):
                categories['complaint'].append(col['name'])
            
            # Content/Description patterns
            elif any(word in name_lower for word in [
                'description', 'content', 'text', 'message', 'comment', 'note',
                'details', 'info', 'summary', 'subject', 'title'
            ]):
                categories['content'].append(col['name'])
            
            # ID patterns (catch-all for identifiers)
            elif any(word in name_lower for word in ['id', 'key', 'ref', 'guid', 'uuid']):
                categories['id'].append(col['name'])
        
        # Log categorization results for debugging
        non_empty_cats = {k: v for k, v in categories.items() if v}
        logger.debug(f"Column categorization: {non_empty_cats}")
        
        return {k: v for k, v in categories.items() if v}


class QueryExecutor:
    """Enhanced query executor with safety and monitoring"""
    
    def __init__(self, config: Config):
        self.config = config
        self.execution_stats = defaultdict(int)
        self.query_cache = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper configuration"""
        conn = None
        try:
            # Build connection string with timeout
            conn_string = self.config.get_database_connection_string()
            if 'timeout=' not in conn_string.lower():
                conn_string += ';timeout=30'
            
            conn = pyodbc.connect(
                conn_string,
                autocommit=True,
                readonly=True  # Force read-only
            )
            
            # Configure for Greek text support
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8') 
            conn.setencoding(encoding='utf-8')
            
            yield conn
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_with_safety(self, sql: str, timeout: int = 60) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL with comprehensive safety measures"""
        if not sql or not sql.strip():
            return [], "Empty SQL query"
        
        # Check cache first
        cache_key = hashlib.md5(sql.encode()).hexdigest()
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            logger.info("Query result served from cache")
            return cached_result
        
        start_time = time.time()
        
        # Wrapper function for execution with timeout
        def execute_query():
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Execute with monitoring (timeout handled at connection level)
                    logger.info(f"Executing SQL: {sql[:100]}...")
                    
                    cursor.execute(sql)
                    
                    # Process results
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
                        results = []
                        
                        # Fetch with limit for safety
                        for row in cursor.fetchmany(5000):  # Hard limit
                            row_dict = {}
                            for i, value in enumerate(row):
                                if i < len(columns):
                                    row_dict[columns[i]] = self._safe_value(value)
                            results.append(row_dict)
                        
                        return results, None
                    else:
                        return [], None
                        
            except pyodbc.Error as e:
                self.execution_stats['db_error'] += 1
                error_msg = str(e)
                
                # Categorize database errors
                if "Invalid column name" in error_msg:
                    return [], f"Column not found: {self._extract_column_name(error_msg)}"
                elif "Invalid object name" in error_msg:
                    return [], f"Table not found: {self._extract_table_name(error_msg)}"
                elif "timeout" in error_msg.lower() or "query timeout" in error_msg.lower():
                    return [], f"Query timeout after {timeout}s - query too complex"
                elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                    return [], f"Access denied - insufficient permissions"
                else:
                    logger.error(f"Database error: {error_msg}")
                    return [], f"Database error: {error_msg}"
            
            except Exception as e:
                self.execution_stats['system_error'] += 1
                logger.error(f"System error during query execution: {e}")
                return [], f"System error: {str(e)}"
        
        # Execute the query
        try:
            results, error = execute_query()
            
            if error is None and results is not None:
                # Cache successful results
                with self._lock:
                    self.query_cache[cache_key] = (results, None)
                    # Limit cache size
                    if len(self.query_cache) > 100:
                        oldest_key = next(iter(self.query_cache))
                        del self.query_cache[oldest_key]
                
                execution_time = time.time() - start_time
                self.execution_stats['success'] += 1
                self.execution_stats['total_time'] += execution_time
                
                logger.info(f"Query executed successfully: {len(results)} rows in {execution_time:.1f}s")
                return results, None
            else:
                return results or [], error
                
        except Exception as e:
            self.execution_stats['system_error'] += 1
            error_msg = f"Execution wrapper error: {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    def _safe_value(self, value) -> Any:
        """Convert value to safe, serializable format"""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            # Truncate very long strings
            if isinstance(value, str) and len(value) > 500:
                return value[:500] + "... [truncated]"
            return value
        else:
            return str(value)[:200]
    
    def _extract_column_name(self, error_msg: str) -> str:
        """Extract column name from error message"""
        match = re.search(r"Invalid column name '([^']+)'", error_msg)
        return match.group(1) if match else "unknown"
    
    def _extract_table_name(self, error_msg: str) -> str:
        """Extract table name from error message"""
        match = re.search(r"Invalid object name '([^']+)'", error_msg)
        return match.group(1) if match else "unknown"
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        stats = dict(self.execution_stats)
        if stats.get('success', 0) > 0:
            stats['avg_execution_time'] = stats.get('total_time', 0) / stats['success']
        return stats


class SemanticRAG:
    """Main Semantic RAG System - Enhanced with template-based constraints"""
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        
        # Initialize components
        logger.info("Initializing Enhanced Semantic RAG System...")
        
        self.llm_client = LLMClient(config)
        self.embedder = SchemaEmbedder() 
        self.graph_builder = SchemaGraphBuilder()
        self.executor = QueryExecutor(config)
        
        # Load data
        self._load_semantic_data()
        
        # Build enhanced structures
        logger.info("Building enhanced schema structures...")
        self.schema_graph = self.graph_builder.build_graph(self.tables)
        
        # Initialize selectors and generators
        self.table_selector = SmartTableSelector(self.tables, self.embedder, self.schema_graph)
        self.relationship_mapper = RelationshipMapper(self.schema_graph)
        self.sql_generator = SmartSQLGenerator(self.llm_client, self.schema_graph, config)  # Pass config
        
        # Statistics tracking
        self.session_stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0
        }
        
        init_time = time.time() - self.start_time
        logger.info(f"Enhanced Semantic RAG System initialized in {init_time:.1f}s")
        
        # Print initialization summary
        self._print_initialization_summary()
    
    def _load_semantic_data(self):
        """Load semantic analysis data"""
        try:
            semantic_file = self.config.get_cache_path("semantic_analysis.json")
            if not semantic_file.exists():
                raise FileNotFoundError("No semantic analysis found. Run analysis first.")
            
            with open(semantic_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load tables
            self.tables = []
            for table_data in data.get('tables', []):
                table = TableInfo(
                    name=table_data['name'],
                    schema=table_data['schema'], 
                    full_name=table_data['full_name'],
                    object_type=table_data['object_type'],
                    row_count=table_data['row_count'],
                    columns=table_data['columns'],
                    sample_data=table_data['sample_data'],
                    relationships=table_data.get('relationships', [])
                )
                table.entity_type = table_data.get('entity_type', 'Unknown')
                table.confidence = table_data.get('confidence', 0.0)
                self.tables.append(table)
            
            # Load domain
            domain_data = data.get('domain')
            if domain_data:
                self.domain = BusinessDomain(
                    domain_type=domain_data['domain_type'],
                    industry=domain_data['industry'],
                    confidence=domain_data['confidence'],
                    sample_questions=domain_data['sample_questions'],
                    capabilities=domain_data['capabilities']
                )
            else:
                self.domain = None
            
            logger.info(f"Loaded {len(self.tables)} tables from semantic analysis")
            
        except Exception as e:
            logger.error(f"Failed to load semantic data: {e}")
            raise
    
    def _print_initialization_summary(self):
        """Print system initialization summary"""
        print("\n" + "="*70)
        print("🧠 ENHANCED SEMANTIC DATABASE RAG SYSTEM")
        print("="*70)
        print(f"📊 Tables loaded: {len(self.tables):,}")
        print(f"🏗️ Schema relationships: {len(self.graph_builder.pk_fk_relationships)}")
        
        if self.domain:
            print(f"🏢 Business domain: {self.domain.domain_type} ({self.domain.industry})")
        
        # Entity distribution
        entity_counts = defaultdict(int)
        for table in self.tables:
            entity_counts[table.entity_type] += 1
        
        if entity_counts:
            print("📋 Entity types:")
            for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {entity_type}: {count}")
        
        # Capabilities summary
        capabilities = []
        if EMBEDDINGS_AVAILABLE and self.embedder.available:
            capabilities.append("🎯 Semantic search")
        else:
            capabilities.append("🔍 Keyword-based search")
            
        if SQLGLOT_AVAILABLE:
            capabilities.append("⚡ SQL optimization") 
        
        capabilities.append("📝 Template-based generation")
        capabilities.append("🔒 Multi-layer validation")
        capabilities.append("🎯 Live schema validation")  # New capability
        
        if capabilities:
            print("🚀 Available capabilities:")
            for cap in capabilities:
                print(f"   {cap}")
        
        # Show missing optional dependencies
        missing = []
        if not EMBEDDINGS_AVAILABLE:
            missing.append("sentence-transformers (semantic search)")
        if not SQLGLOT_AVAILABLE:
            missing.append("sqlglot (SQL optimization)")
        
        if missing:
            print("⚠️  Optional features disabled:")
            for miss in missing:
                print(f"   • {miss}")
            print("   Install missing dependencies to enable these features")
        
        print("="*70)
        print("✅ Template-based SQL generation ready!")
        print("Ready for queries! Type 'quit' to exit.\n")
    
    async def start_interactive_session(self, tables=None, domain=None, relationships=None):
        """Start enhanced interactive session"""
        print("🎯 Starting interactive query session...")
        
        query_count = 0
        session_start = time.time()
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"\n🚀 Processing query {query_count}...")
                
                # Process with progress indicators
                start_time = time.time()
                result = await self._process_query_with_progress(question)
                result.execution_time = time.time() - start_time
                
                # Update statistics
                self.session_stats['queries_processed'] += 1
                self.session_stats['total_execution_time'] += result.execution_time
                
                if result.error:
                    self.session_stats['failed_queries'] += 1
                else:
                    self.session_stats['successful_queries'] += 1
                
                # Display result
                self._display_enhanced_result(result, query_count)
                
            except KeyboardInterrupt:
                print("\n⏸️ Session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in query {query_count}: {e}")
                print(f"❌ Unexpected error: {e}")
        
        # Session summary
        self._print_session_summary(query_count, time.time() - session_start)
    
    async def _process_query_with_progress(self, question: str) -> QueryResult:
        """Process query with progress indicators"""
        try:
            with tqdm(total=4, desc="Processing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                
                # Step 1: Table selection
                pbar.set_description("Selecting tables")
                selected_tables = await self.table_selector.select_tables(question)
                pbar.update(1)
                
                if not selected_tables:
                    return QueryResult(
                        question=question,
                        sql_query="",
                        results=[],
                        error="No relevant tables found for the question"
                    )
                
                # Step 2: SQL generation  
                pbar.set_description("Generating SQL")
                sql, retry_count = await self.sql_generator.generate_smart_sql(question, selected_tables)
                pbar.update(1)
                
                if not sql:
                    return QueryResult(
                        question=question,
                        sql_query="",
                        results=[],
                        error=f"Failed to generate valid SQL after {retry_count} attempts"
                    )
                
                # Step 3: Query execution
                pbar.set_description("Executing query")
                results, error = self.executor.execute_with_safety(sql)
                pbar.update(1)
                
                # Step 4: Result processing
                pbar.set_description("Processing results")
                query_result = QueryResult(
                    question=question,
                    sql_query=sql,
                    results=results,
                    error=error,
                    tables_used=[t.full_name for t in selected_tables]
                )
                pbar.update(1)
                
                return query_result
                
        except Exception as e:
            logger.error(f"Error in query processing pipeline: {e}")
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Pipeline error: {str(e)}"
            )
    
    def _display_enhanced_result(self, result: QueryResult, query_num: int):
        """Display enhanced query results"""
        print(f"\n📊 QUERY {query_num} RESULTS")
        print("-" * 60)
        print(f"⏱️ Execution time: {result.execution_time:.2f}s")
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.sql_query:
                print(f"\n📝 Generated SQL:")
                print(f"   {result.sql_query}")
        else:
            print(f"✅ Success: {len(result.results)} rows returned")
            
            # Show SQL
            print(f"\n📝 Generated SQL:")
            sql_lines = result.sql_query.split('\n')
            for line in sql_lines:
                print(f"   {line}")
            
            # Show results intelligently
            if result.results:
                self._display_results_smart(result.results)
            else:
                print("📊 No data returned")
            
            # Show table usage
            if result.tables_used:
                print(f"\n📋 Tables queried ({len(result.tables_used)}):")
                for table_name in result.tables_used:
                    table_info = self._get_table_info(table_name)
                    print(f"   • {table_name} ({table_info['entity_type']}) - {table_info['row_count']:,} rows")
        
        print("-" * 60)
    
    def _display_results_smart(self, results: List[Dict]):
        """Smart result display based on data type and volume"""
        if not results:
            return
        
        # Single value results
        if len(results) == 1 and len(results[0]) == 1:
            key, value = next(iter(results[0].items()))
            if isinstance(value, (int, float)):
                print(f"🎯 Result: {value:,} ({key})")
            else:
                print(f"🎯 Result: {value} ({key})")
            return
        
        # Multiple results
        print(f"\n📊 Data preview (showing up to 10 rows):")
        
        # Determine display columns (limit very wide tables)
        sample_row = results[0]
        all_columns = list(sample_row.keys())
        display_columns = all_columns[:8]  # Limit columns for readability
        
        # Format and display
        for i, row in enumerate(results[:10], 1):
            row_display = {}
            for col in display_columns:
                value = row.get(col)
                if isinstance(value, str) and len(value) > 50:
                    row_display[col] = value[:50] + "..."
                elif isinstance(value, (int, float)) and abs(value) > 1000:
                    row_display[col] = f"{value:,}"
                else:
                    row_display[col] = value
            
            print(f"   {i:2d}. {row_display}")
        
        if len(results) > 10:
            print(f"   ... and {len(results) - 10} more rows")
        
        if len(all_columns) > len(display_columns):
            hidden_cols = len(all_columns) - len(display_columns)
            print(f"   (+ {hidden_cols} more columns)")
    
    def _get_table_info(self, table_name: str) -> Dict:
        """Get table information"""
        for table in self.tables:
            if table.full_name == table_name:
                return {
                    'entity_type': getattr(table, 'entity_type', 'Unknown'),
                    'row_count': table.row_count
                }
        return {'entity_type': 'Unknown', 'row_count': 0}
    
    def _print_session_summary(self, query_count: int, session_duration: float):
        """Print session summary"""
        print(f"\n📈 SESSION SUMMARY")
        print("="*50)
        print(f"Queries processed: {query_count}")
        print(f"Successful: {self.session_stats['successful_queries']}")
        print(f"Failed: {self.session_stats['failed_queries']}")
        print(f"Success rate: {(self.session_stats['successful_queries']/max(query_count,1)*100):.1f}%")
        print(f"Session duration: {session_duration:.1f}s")
        
        if query_count > 0:
            avg_time = self.session_stats['total_execution_time'] / query_count
            print(f"Average query time: {avg_time:.2f}s")
        
        # Component statistics
        executor_stats = self.executor.get_stats()
        if executor_stats:
            print(f"\nExecutor stats: {executor_stats}")
        
        llm_stats = self.llm_client.generation_stats
        if llm_stats:
            print(f"LLM stats: {dict(llm_stats)}")
        
        print("="*50)
        print("Thank you for using Enhanced Semantic RAG! 🎉")
    
    async def process_single_query(self, question: str) -> QueryResult:
        """Process a single query - useful for API integration"""
        return await self._process_query_with_progress(question)


# Backward compatibility
QueryInterface = SemanticRAG


async def main():
    """Main entry point for testing"""
    try:
        # Check critical dependencies
        missing_critical = []
        try:
            import sqlglot
        except ImportError:
            missing_critical.append("sqlglot")
        
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            missing_critical.append("langchain-openai")
        
        try:
            import pyodbc
        except ImportError:
            missing_critical.append("pyodbc")
        
        if missing_critical:
            print(f"❌ Critical dependencies missing: {', '.join(missing_critical)}")
            print("Install with: pip install " + " ".join(missing_critical))
            return
        
        # Load configuration
        config = Config()
        
        # Initialize and start the enhanced semantic RAG system
        print("🚀 Starting Enhanced Semantic Database RAG System...")
        
        # Show dependency status
        print(f"✅ Core dependencies loaded")
        if not EMBEDDINGS_AVAILABLE:
            print(f"⚠️  Semantic search disabled (install: pip install sentence-transformers numpy)")
        if not SQLGLOT_AVAILABLE:
            print(f"⚠️  SQL optimization disabled (install: pip install sqlglot)")
        
        print("✅ Template-based SQL generation with live schema validation enabled")
        
        semantic_rag = SemanticRAG(config)
        await semantic_rag.start_interactive_session()
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        print(f"❌ System startup failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Ensure semantic_analysis.json exists (run analysis first)")
        print("   • Check database connection settings")
        print("   • Verify Azure OpenAI configuration")
        print("   • Install core dependencies: pip install sqlglot langchain-openai pyodbc tqdm")


if __name__ == "__main__":
    asyncio.run(main())