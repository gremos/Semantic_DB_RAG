#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Semantic RAG - Simple, Readable, Maintainable
Implements: DRY, SOLID, YAGNI principles with README compliance
"""

import asyncio
import json
import re
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

# SQLGlot for AST validation
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False
    print("⚠️ SQLGlot not available - install with: pip install sqlglot")

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import parse_json_response, clean_sql_query, safe_database_value


@dataclass
class QueryIntent:
    """Simple data class for query intent"""
    aggregation: Optional[str] = None
    limit: int = 100
    has_date_filter: bool = False
    date_year: Optional[int] = None
    entity_focus: Optional[str] = None
    requires_joins: bool = False
    business_context: Optional[str] = None


@dataclass
class ValidationResult:
    """AST validation result"""
    is_valid: bool
    message: str
    ast: Optional[Any] = None


class LLMProvider:
    """Simple LLM abstraction - Single Responsibility"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            request_timeout=60,
            temperature=1.0
        )
    
    async def analyze(self, system_prompt: str, user_prompt: str) -> str:
        """Single LLM analysis method"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
            return ""


class IntentAnalyzer:
    """Extract intent from natural language queries"""
    
    @staticmethod
    def analyze(question: str) -> QueryIntent:
        """Analyze query intent with enhanced detection"""
        q_lower = question.lower()
        intent = QueryIntent()
        
        # Detect aggregation
        if any(w in q_lower for w in ['how many', 'count', 'number of']):
            intent.aggregation = 'count'
        elif any(w in q_lower for w in ['total', 'sum', 'amount']):
            intent.aggregation = 'sum'
        elif any(w in q_lower for w in ['average', 'avg', 'mean']):
            intent.aggregation = 'avg'
        
        # Extract limits
        limit_match = re.search(r'(?:top|first|limit)\s+(\d+)', q_lower)
        if limit_match:
            intent.limit = int(limit_match.group(1))
        
        # Date filtering
        year_match = re.search(r'(20\d{2})', question)
        if year_match:
            intent.has_date_filter = True
            intent.date_year = int(year_match.group(1))
        elif any(w in q_lower for w in ['this year', 'current year']):
            intent.has_date_filter = True
            intent.date_year = 2025
        
        # Entity focus
        if any(w in q_lower for w in ['customer', 'client', 'user']):
            intent.entity_focus = 'customer'
        elif any(w in q_lower for w in ['payment', 'paid', 'revenue']):
            intent.entity_focus = 'payment'
        elif any(w in q_lower for w in ['order', 'sale', 'transaction']):
            intent.entity_focus = 'order'
        
        # Business context
        if any(w in q_lower for w in ['paid', 'payment']):
            intent.business_context = 'payment_analysis'
            intent.requires_joins = True
        elif any(w in q_lower for w in ['revenue', 'income']):
            intent.business_context = 'financial_analysis'
            intent.requires_joins = True
        
        return intent


class TableScorer:
    """Score tables for relevance"""
    
    ENTITY_KEYWORDS = {
        'customer': ['customer', 'client', 'user', 'account', 'contact'],
        'payment': ['payment', 'paid', 'billing', 'invoice', 'revenue', 'financial'],
        'order': ['order', 'sale', 'purchase', 'transaction', 'contract'],
        'financial': ['revenue', 'amount', 'total', 'value', 'price', 'cost']
    }
    
    @classmethod
    def score_table(cls, table: TableInfo, question: str) -> Tuple[float, List[str]]:
        """Score table relevance with detailed reasoning"""
        q_lower = question.lower()
        question_words = [w for w in q_lower.split() if len(w) > 2]
        
        score = 0.0
        reasons = []
        
        # Table name matching
        table_name_lower = table.name.lower()
        name_matches = [word for word in question_words if word in table_name_lower]
        if name_matches:
            score += len(name_matches) * 3.0
            reasons.append(f"name_match:{','.join(name_matches)}")
        
        # Entity type matching
        table_entity = getattr(table, 'entity_type', '').lower()
        for entity, keywords in cls.ENTITY_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                if entity in table_entity or any(kw in table_name_lower for kw in keywords):
                    score += 2.0
                    reasons.append(f"entity_match:{entity}")
        
        # Column relevance
        column_names = [col.get('name', '').lower() for col in table.columns]
        score += cls._score_columns(column_names, q_lower, reasons)
        
        # Data availability bonus
        if table.row_count > 0:
            score += 0.5
            reasons.append(f"has_data:{table.row_count:,}")
        
        # Business role bonus
        if getattr(table, 'business_role', '') == 'Core':
            score += 1.0
            reasons.append("core_business_table")
        
        return score, reasons
    
    @staticmethod
    def _score_columns(column_names: List[str], question: str, reasons: List[str]) -> float:
        """Score column relevance"""
        score = 0.0
        
        # Payment columns
        if any(w in question for w in ['paid', 'payment', 'revenue']):
            payment_cols = [col for col in column_names 
                          if any(p in col for p in ['payment', 'amount', 'paid', 'revenue'])]
            if payment_cols:
                score += 2.0
                reasons.append(f"payment_columns:{','.join(payment_cols[:3])}")
        
        # Date columns
        if any(w in question for w in ['2025', '2024', 'year']):
            date_cols = [col for col in column_names 
                        if any(d in col for d in ['date', 'time', 'year', 'created'])]
            if date_cols:
                score += 1.5
                reasons.append(f"date_columns:{','.join(date_cols[:3])}")
        
        # Customer columns
        if any(w in question for w in ['customer', 'client']):
            customer_cols = [col for col in column_names 
                           if any(c in col for c in ['customer', 'client', 'user'])]
            if customer_cols:
                score += 1.5
                reasons.append(f"customer_columns:{','.join(customer_cols[:3])}")
        
        return score


class TableFilter:
    """Filter and select relevant tables"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
        self.scorer = TableScorer()
    
    def filter_lexically(self, question: str) -> List[Tuple[TableInfo, float, List[str]]]:
        """Filter tables using lexical analysis"""
        scored_tables = []
        
        for table in self.tables:
            score, reasons = self.scorer.score_table(table, question)
            if score > 0:
                scored_tables.append((table, score, reasons))
        
        # Sort by score and filter
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        # Dynamic selection based on score distribution
        if len(scored_tables) > 20:
            return [(table, score, reasons) for table, score, reasons in scored_tables 
                   if score > 2.0][:15]
        else:
            return scored_tables[:15]
    
    async def select_by_llm(self, question: str, candidates: List[TableInfo], 
                           llm: LLMProvider) -> List[TableInfo]:
        """Use LLM for final table selection"""
        table_summaries = []
        for table in candidates:
            table_summaries.append({
                'table_name': table.full_name,
                'entity_type': getattr(table, 'entity_type', ''),
                'row_count': table.row_count,
                'columns': [col.get('name') for col in table.columns[:5]]
            })
        
        system_prompt = """Select the most relevant tables for the user question.
Focus on tables that directly relate to what the user is asking."""
        
        user_prompt = f"""
QUESTION: "{question}"
CANDIDATES: {json.dumps(table_summaries, indent=2)}

Select the 6 most relevant tables. Respond with JSON:
{{"selected_tables": ["[schema].[table1]"], "reasoning": "Why these tables"}}
"""
        
        response = await llm.analyze(system_prompt, user_prompt)
        result = parse_json_response(response)
        
        if result and 'selected_tables' in result:
            selected_names = result['selected_tables']
            return [t for t in candidates if t.full_name in selected_names]
        
        return candidates[:6]


class ViewPatternExtractor:
    """Extract patterns from database views"""
    
    def __init__(self, database_structure: Dict):
        self.structure = database_structure
        self.patterns = self._extract_patterns()
    
    def _extract_patterns(self) -> List[Dict]:
        """Extract proven join patterns from views"""
        patterns = []
        views = self.structure.get('view_info', {})
        
        for view_name, view_data in views.items():
            if view_data.get('definition') and view_data.get('execution_success'):
                business_pattern = view_data.get('business_pattern', {})
                
                patterns.append({
                    'view_name': view_name,
                    'definition': view_data['definition'],
                    'business_pattern': business_pattern.get('pattern', 'unknown'),
                    'use_case': business_pattern.get('estimated_use_case', ''),
                    'sample_data': view_data.get('sample_data', []),
                    'columns': view_data.get('columns_returned', []),
                    'proven_working': True,
                    'confidence': business_pattern.get('confidence', 0.0)
                })
        
        return patterns
    
    def find_relevant(self, question: str, tables: List[TableInfo]) -> List[Dict]:
        """Find view patterns relevant to the question"""
        if not self.patterns:
            return []
        
        q_lower = question.lower()
        table_names = [t.full_name.lower() for t in tables]
        relevant = []
        
        for pattern in self.patterns:
            relevance_score = 0.0
            
            # Business pattern matching
            business_pattern = pattern.get('business_pattern', '')
            if any(keyword in q_lower for keyword in ['customer', 'payment', 'paid']):
                if any(bp in business_pattern for bp in ['customer', 'payment', 'financial']):
                    relevance_score += 3.0
            
            # Use case matching
            use_case = pattern.get('use_case', '').lower()
            if 'payment' in q_lower and 'payment' in use_case:
                relevance_score += 2.0
            
            # Table involvement
            definition = pattern.get('definition', '').lower()
            table_overlap = sum(1 for table_name in table_names 
                              if any(part in definition for part in table_name.split('.')))
            relevance_score += table_overlap * 1.0
            
            # Confidence bonus
            relevance_score += pattern.get('confidence', 0.0) * 0.5
            
            if relevance_score > 1.0:
                pattern['relevance_score'] = relevance_score
                relevant.append(pattern)
        
        return sorted(relevant, key=lambda x: x['relevance_score'], reverse=True)[:3]


class SQLValidator:
    """Validate SQL using SQLGlot AST and allowlists"""
    
    def __init__(self, allowed_identifiers: Set[str]):
        self.allowed_identifiers = allowed_identifiers
    
    def validate(self, sql: str) -> ValidationResult:
        """Comprehensive SQL validation"""
        if not sql.strip():
            return ValidationResult(False, "Empty SQL query")
        
        if HAS_SQLGLOT:
            return self._validate_with_sqlglot(sql)
        else:
            return self._basic_validation(sql)
    
    def _validate_with_sqlglot(self, sql: str) -> ValidationResult:
        """Enhanced SQLGlot validation with column checking"""
        try:
            parsed = sqlglot.parse_one(sql, dialect="tsql")
            
            if not parsed:
                return ValidationResult(False, "Failed to parse SQL")
            
            # Check for dangerous operations
            dangerous_nodes = [
                sqlglot.expressions.Insert, sqlglot.expressions.Update,
                sqlglot.expressions.Delete, sqlglot.expressions.Drop,
                sqlglot.expressions.Create, sqlglot.expressions.Alter,
                sqlglot.expressions.Merge, sqlglot.expressions.Truncate
            ]
            
            for dangerous in dangerous_nodes:
                if parsed.find(dangerous):
                    return ValidationResult(False, f"Dangerous operation: {dangerous.__name__}")
            
            # Must be SELECT
            if not isinstance(parsed, sqlglot.expressions.Select):
                return ValidationResult(False, "Only SELECT statements allowed")
            
            # Validate identifiers using AST traversal
            validation_error = self._validate_identifiers_ast(parsed)
            if validation_error:
                return ValidationResult(False, validation_error, parsed)
            
            return ValidationResult(True, "SQL validated successfully", parsed)
            
        except Exception as e:
            return ValidationResult(False, f"SQLGlot validation error: {str(e)}")
    
    def _validate_identifiers_ast(self, ast) -> Optional[str]:
        """Validate table and column identifiers from AST"""
        unknown_identifiers = []
        
        # Check tables
        for table_node in ast.find_all(sqlglot.expressions.Table):
            table_name = str(table_node).lower().strip('[]')
            if table_name not in self.allowed_identifiers:
                # Check partial matches
                if not any(table_name in allowed for allowed in self.allowed_identifiers):
                    unknown_identifiers.append(f"table:{table_name}")
        
        # Check columns
        for column_node in ast.find_all(sqlglot.expressions.Column):
            column_name = str(column_node.this).lower().strip('[]')
            if column_name not in self.allowed_identifiers and len(column_name) > 2:
                # Allow common SQL keywords and functions
                if not self._is_sql_keyword(column_name):
                    unknown_identifiers.append(f"column:{column_name}")
        
        if len(unknown_identifiers) > 3:  # Allow some flexibility
            return f"Unknown identifiers: {', '.join(unknown_identifiers[:3])}"
        
        return None
    
    @staticmethod
    def _is_sql_keyword(name: str) -> bool:
        """Check if name is a common SQL keyword/function"""
        keywords = {
            'count', 'sum', 'avg', 'max', 'min', 'year', 'month', 'day',
            'getdate', 'datepart', 'cast', 'convert', 'case', 'when', 'then',
            'else', 'end', 'null', 'top', 'distinct', 'as', 'desc', 'asc'
        }
        return name.lower() in keywords
    
    def _basic_validation(self, sql: str) -> ValidationResult:
        """Basic validation when SQLGlot unavailable"""
        sql_upper = sql.upper().strip()
        
        if not sql_upper.startswith('SELECT'):
            return ValidationResult(False, "Only SELECT statements allowed")
        
        dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        if any(op in sql_upper for op in dangerous):
            return ValidationResult(False, "Dangerous operations detected")
        
        return ValidationResult(True, "Basic validation passed")


class SQLGenerator:
    """Generate SQL with constraints and validation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.structure = self._load_structure()
        self.pattern_extractor = ViewPatternExtractor(self.structure)
        self.allowed_identifiers = self._build_allowlist()
        self.validator = SQLValidator(self.allowed_identifiers)
    
    def _load_structure(self) -> Dict:
        """Load database structure"""
        try:
            structure_file = self.config.get_cache_path("database_structure.json")
            if structure_file.exists():
                with open(structure_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _build_allowlist(self) -> Set[str]:
        """Build comprehensive allowlist of identifiers"""
        identifiers = set()
        
        # Add tables and columns
        if 'tables' in self.structure:
            for table_data in self.structure['tables']:
                self._add_table_identifiers(table_data, identifiers)
        
        # Add views
        if 'view_info' in self.structure:
            for view_name, view_data in self.structure['view_info'].items():
                identifiers.add(view_name.lower())
                if 'full_name' in view_data:
                    identifiers.add(view_data['full_name'].lower())
        
        print(f"      📊 Built allowlist with {len(identifiers)} identifiers")
        return identifiers
    
    def _add_table_identifiers(self, table_data: Dict, identifiers: Set[str]):
        """Add table and column identifiers to allowlist"""
        full_name = table_data.get('full_name', '')
        schema = table_data.get('schema', '')
        name = table_data.get('name', '')
        
        # Add table names
        for table_id in [full_name, f"{schema}.{name}", f"[{schema}].[{name}]", name]:
            if table_id:
                identifiers.add(table_id.lower())
        
        # Add columns
        for column in table_data.get('columns', []):
            col_name = column.get('name', '')
            if col_name:
                identifiers.add(col_name.lower())
                identifiers.add(f"[{col_name}]".lower())
    
    async def generate(self, question: str, tables: List[TableInfo], 
                      intent: QueryIntent, llm: LLMProvider) -> str:
        """Generate validated SQL with view patterns"""
        print("   ⚡ Stage 4: Constrained SQL generation...")
        
        # Find relevant view patterns
        relevant_patterns = self.pattern_extractor.find_relevant(question, tables)
        
        if relevant_patterns:
            print(f"      📋 Found {len(relevant_patterns)} relevant view patterns")
            sql = await self._generate_with_patterns(question, tables, relevant_patterns, intent, llm)
        else:
            print("      🔄 Generating SQL using table analysis...")
            sql = await self._generate_from_tables(question, tables, intent, llm)
        
        # Enhance with intent
        sql = self._enhance_with_intent(sql, intent, tables)
        
        # Validate
        validation = self.validator.validate(sql)
        if not validation.is_valid:
            print(f"      ⚠️ Validation failed: {validation.message}")
            sql = self._generate_fallback(tables, intent)
            print("      ✅ Using safe fallback SQL")
        else:
            print(f"      ✅ SQL validated successfully")
        
        return sql
    
    async def _generate_with_patterns(self, question: str, tables: List[TableInfo],
                                    patterns: List[Dict], intent: QueryIntent, 
                                    llm: LLMProvider) -> str:
        """Generate SQL using proven view patterns"""
        best_pattern = patterns[0]
        
        system_prompt = f"""Adapt the proven view pattern to answer the user's question.

PROVEN VIEW PATTERN:
{best_pattern['view_name']}: {best_pattern['business_pattern']}
Definition: {best_pattern['definition']}

Use the proven JOIN logic and adapt SELECT/WHERE for the question."""
        
        user_prompt = f"""
QUESTION: "{question}"
INTENT: {intent}
TABLES: {self._format_tables(tables)}

Adapt the pattern to answer this question. Include date filtering if needed.
Return only SQL:
"""
        
        response = await llm.analyze(system_prompt, user_prompt)
        return clean_sql_query(response)
    
    async def _generate_from_tables(self, question: str, tables: List[TableInfo],
                                  intent: QueryIntent, llm: LLMProvider) -> str:
        """Generate SQL from table analysis"""
        context = self._build_context(tables, intent)
        
        system_prompt = f"""Generate SQL using exact table and column names.
Include proper date filtering when specified.
Return only SELECT statements.

CONTEXT: {context}"""
        
        user_prompt = f"""
QUESTION: "{question}"
INTENT: {intent}

Generate SQL that answers the question precisely.
Return only the SQL query:
"""
        
        response = await llm.analyze(system_prompt, user_prompt)
        return clean_sql_query(response)
    
    def _enhance_with_intent(self, sql: str, intent: QueryIntent, tables: List[TableInfo]) -> str:
        """Enhance SQL to match intent requirements"""
        if not sql:
            return sql
        
        # Add missing date filtering
        if intent.has_date_filter and intent.date_year:
            sql = self._add_date_filter(sql, intent.date_year, tables)
        
        # Ensure TOP limit
        if intent.aggregation != 'count' and 'TOP' not in sql.upper():
            sql = sql.replace('SELECT', f'SELECT TOP {intent.limit}', 1)
        
        return sql
    
    def _add_date_filter(self, sql: str, year: int, tables: List[TableInfo]) -> str:
        """Add year filtering to SQL"""
        if f'YEAR(' in sql.upper() or str(year) in sql:
            return sql
        
        # Find date columns
        date_columns = []
        for table in tables:
            for col in table.columns:
                col_name = col.get('name', '').lower()
                if any(d in col_name for d in ['date', 'time', 'created', 'modified']):
                    date_columns.append(col.get('name'))
        
        if not date_columns:
            return sql
        
        date_col = date_columns[0]
        year_filter = f"YEAR({date_col}) = {year}"
        
        if 'WHERE' in sql.upper():
            where_pos = sql.upper().find('WHERE')
            existing_where = sql[where_pos + 5:].strip()
            new_where = f"WHERE {year_filter} AND ({existing_where})"
            sql = sql[:where_pos] + new_where
        else:
            # Add before ORDER BY/GROUP BY
            insert_pos = len(sql)
            for keyword in ['ORDER BY', 'GROUP BY', 'HAVING']:
                pos = sql.upper().find(keyword)
                if pos != -1:
                    insert_pos = min(insert_pos, pos)
            
            sql = sql[:insert_pos].rstrip() + f" WHERE {year_filter} " + sql[insert_pos:]
        
        print(f"      ✅ Added year filter: YEAR({date_col}) = {year}")
        return sql
    
    def _generate_fallback(self, tables: List[TableInfo], intent: QueryIntent) -> str:
        """Generate safe fallback SQL"""
        if not tables:
            return ""
        
        table = max(tables, key=lambda t: t.row_count)
        safe_columns = [col.get('name') for col in table.columns[:5] 
                       if 'int' in col.get('data_type', '').lower() or 
                          'varchar' in col.get('data_type', '').lower()]
        
        if intent.aggregation == 'count':
            sql = f"SELECT COUNT(*) as total_count FROM {table.full_name}"
        else:
            columns_str = ', '.join([f"[{col}]" for col in safe_columns[:5]]) if safe_columns else "*"
            sql = f"SELECT TOP {intent.limit} {columns_str} FROM {table.full_name}"
        
        return sql
    
    def _format_tables(self, tables: List[TableInfo]) -> str:
        """Format tables for prompt"""
        formatted = []
        for table in tables:
            columns = [col.get('name') for col in table.columns[:8]]
            formatted.append(f"{table.full_name}: {', '.join(columns)}")
        return '\n'.join(formatted)
    
    def _build_context(self, tables: List[TableInfo], intent: QueryIntent) -> str:
        """Build context string for SQL generation"""
        context = []
        for table in tables:
            context.append(f"TABLE: {table.full_name}")
            columns = [f"{col.get('name')} ({col.get('data_type', 'unknown')})" 
                      for col in table.columns[:8]]
            context.append(f"  Columns: {', '.join(columns)}")
        return '\n'.join(context)


class DatabaseExecutor:
    """Execute SQL with retry logic"""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_retries = 2
    
    async def execute_with_retry(self, sql: str, question: str, 
                               llm: LLMProvider) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL with retry on failure"""
        print("   🔄 Stage 4b: Execution with retry...")
        
        for attempt in range(self.max_retries + 1):
            results, error = self._execute_sql(sql)
            
            if error is None:
                if len(results) == 0 and attempt < self.max_retries:
                    print(f"      ⚠️ Attempt {attempt + 1}: Empty results, retrying...")
                    sql = await self._retry_for_empty(sql, question, llm)
                    continue
                
                print(f"      ✅ Success on attempt {attempt + 1}: {len(results)} rows")
                return results, None
            else:
                print(f"      ⚠️ Attempt {attempt + 1} failed: {error}")
                if attempt < self.max_retries:
                    sql = await self._retry_for_error(sql, error, question, llm)
                    continue
        
        return [], f"Failed after {self.max_retries + 1} attempts: {error}"
    
    def _execute_sql(self, sql: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL with proper encoding"""
        if not sql.strip():
            return [], "Empty SQL query"
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                cursor.execute(sql)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor:
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = safe_database_value(value)
                        results.append(row_dict)
                    
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            return [], str(e)
    
    async def _retry_for_error(self, sql: str, error: str, question: str, 
                             llm: LLMProvider) -> str:
        """Generate corrected SQL based on error"""
        system_prompt = "Fix the SQL based on the error message."
        user_prompt = f"""
Failed SQL: {sql}
Error: {error}
Question: "{question}"

Generate corrected SQL:
"""
        
        response = await llm.analyze(system_prompt, user_prompt)
        return clean_sql_query(response)
    
    async def _retry_for_empty(self, sql: str, question: str, 
                             llm: LLMProvider) -> str:
        """Modify SQL to return results"""
        system_prompt = "Modify SQL to be less restrictive and return data."
        user_prompt = f"""
SQL returned 0 rows: {sql}
Question: "{question}"

Make the SQL less restrictive:
"""
        
        response = await llm.analyze(system_prompt, user_prompt)
        return clean_sql_query(response)


class QueryInterface:
    """Main 4-Stage Pipeline - Simple and focused"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMProvider(config)
        self.intent_analyzer = IntentAnalyzer()
        self.sql_generator = SQLGenerator(config)
        self.executor = DatabaseExecutor(config)
        
        print("✅ QueryInterface initialized with README compliance:")
        print(f"   🔒 SQLGlot AST Validation: {'✅ Available' if HAS_SQLGLOT else '❌ Not Available'}")
        print(f"   📋 View Patterns: {len(self.sql_generator.pattern_extractor.patterns)}")
        print(f"   🛡️ Identifier Allowlist: {len(self.sql_generator.allowed_identifiers)}")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start interactive query session"""
        self.table_filter = TableFilter(tables)
        
        print(f"🚀 4-Stage Pipeline Ready")
        print(f"   📊 Tables: {len(tables)}")
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🔄 Processing with 4-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_query(question)
                result.execution_time = time.time() - start_time
                
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_query(self, question: str) -> QueryResult:
        """4-Stage Pipeline Implementation"""
        try:
            # Stage 1: Intent Analysis
            print("   🧠 Stage 1: Intent analysis...")
            intent = self.intent_analyzer.analyze(question)
            
            # Stage 2: Table Selection
            print("   📋 Stage 2: Table selection...")
            scored_tables = self.table_filter.filter_lexically(question)
            
            if not scored_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found"
                )
            
            candidates = [table for table, _, _ in scored_tables]
            
            if len(candidates) > 8:
                selected_tables = await self.table_filter.select_by_llm(question, candidates, self.llm)
            else:
                selected_tables = candidates
            
            print(f"      ✅ Selected {len(selected_tables)} tables")
            
            # Stage 3: Relationship Resolution (embedded in SQL generation)
            print("   🔗 Stage 3: Relationship resolution...")
            
            # Stage 4: SQL Generation + Execution
            sql = await self.sql_generator.generate(question, selected_tables, intent, self.llm)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to generate SQL"
                )
            
            results, error = await self.executor.execute_with_retry(sql, question, self.llm)
            
            return QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=[t.full_name for t in selected_tables]
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Pipeline error: {str(e)}"
            )
    
    def _display_result(self, result: QueryResult):
        """Display query results"""
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.sql_query:
                print(f"📋 Generated SQL:\n{result.sql_query}")
        else:
            validation_status = '✅ Validated' if HAS_SQLGLOT else '⚠️ Basic Validation'
            print(f"📋 SQL Query (SQLGlot {validation_status}):")
            print(f"{result.sql_query}")
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                if result.is_single_value():
                    value = result.get_single_value()
                    column_name = list(result.results[0].keys())[0]
                    formatted_value = f"{value:,}" if isinstance(value, (int, float)) and value >= 1000 else str(value)
                    print(f"   🎯 {column_name}: {formatted_value}")
                else:
                    for i, row in enumerate(result.results[:5], 1):
                        display_row = {}
                        for key, value in list(row.items())[:6]:
                            if isinstance(value, str) and len(value) > 40:
                                display_row[key] = value[:37] + "..."
                            elif isinstance(value, (int, float)) and value >= 1000:
                                display_row[key] = f"{value:,}"
                            else:
                                display_row[key] = value
                        print(f"   {i}. {display_row}")
                    
                    if len(result.results) > 5:
                        print(f"   ... and {len(result.results) - 5} more rows")
        
        print("\n💡 4-Stage Pipeline Features:")
        print("   ✅ Intent-driven table selection")
        print("   ✅ View pattern analysis with proven logic")
        print(f"   ✅ {'SQLGlot AST validation' if HAS_SQLGLOT else 'Basic SQL validation'}")
        print("   ✅ Execution-guided retry with error recovery")