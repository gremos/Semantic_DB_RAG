#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - Enhanced with Fixed SQL Templates & Safety Validation
Architecture: Deterministic templates with LLM fallback, sqlglot safety validation
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
"""

import asyncio
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# SQLGlot for safety validation (Architecture requirement)
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False
    print("⚠️ sqlglot not available - SQL safety validation disabled")

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import parse_json_response, clean_sql_query, safe_database_value

class EnhancedIntentAnalyzer:
    """Enhanced intent analysis with business context"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str, tables: List[TableInfo]) -> Dict[str, Any]:
        """Enhanced intent analysis with cross-industry entity support"""
        print("   🧠 Enhanced intent analysis...")
        
        # Build context with cross-industry entities
        entity_context = self._build_entity_context(tables)
        
        # Get LLM analysis
        intent = await self._get_intent_from_llm(question, entity_context)
        
        if intent:
            print(f"      ✅ Intent: {intent.get('task_type', 'unknown')}")
            print(f"      🎯 Entities: {', '.join(intent.get('entities', []))}")
            return intent
        
        # Enhanced pattern-based fallback
        return self._enhanced_pattern_fallback(question)
    
    def _build_entity_context(self, tables: List[TableInfo]) -> str:
        """Build context with available cross-industry entities"""
        entity_groups = {}
        for table in tables[:25]:  # Show more tables for better context
            entity_type = table.entity_type
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(table)
        
        context_lines = ["AVAILABLE BUSINESS ENTITIES:"]
        
        # Prioritize important entities
        priority_entities = ['Customer', 'Payment', 'Contract', 'Order', 'Employee', 'Product']
        
        for entity_type in priority_entities:
            if entity_type in entity_groups:
                tables_list = entity_groups[entity_type]
                context_lines.append(f"\n{entity_type.upper()}:")
                for table in tables_list[:3]:
                    capabilities = []
                    if getattr(table, 'name_columns', []):
                        capabilities.append(f"Names: {', '.join(table.name_columns[:2])}")
                    if getattr(table, 'measures', []):
                        capabilities.append(f"Amounts: {', '.join(table.measures[:2])}")
                    cap_str = f" ({'; '.join(capabilities)})" if capabilities else ""
                    context_lines.append(f"  - {table.name}{cap_str}")
        
        # Add other entities
        other_entities = [e for e in entity_groups.keys() if e not in priority_entities]
        if other_entities:
            context_lines.append(f"\nOTHER: {', '.join(other_entities)}")
        
        return '\n'.join(context_lines)
    
    async def _get_intent_from_llm(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """Enhanced LLM intent analysis with cross-industry support"""
        try:
            prompt = f"""Analyze this business question to understand what data is needed.

QUESTION: "{question}"

{context}

Determine the user's intent based on available entities:
1. What type of query? (ranking, total/sum, count, list, analysis)
2. Which business entities are needed? (Customer, Payment, Contract, Order, Employee, etc.)
3. What should be displayed? (names, amounts, dates, counts)
4. Any limits or ordering? (top 10, highest, etc.)
5. Time filters? (this year, 2025, last month, recent, etc.)
6. Any filtering terms? (active, approved, paid, etc.)

Respond with JSON only:
{{
  "task_type": "ranking|aggregation|count|list|analysis",
  "entities": ["Customer", "Payment"],
  "show_fields": ["names", "amounts", "dates"],
  "limit": 10,
  "time_filter": "2025|this_year|last_month|null",
  "filters": {{"status": "approved", "type": "paid"}},
  "group_by": "customer|product|month|null",
  "search_terms": ["top", "customers", "paid"]
}}"""

            messages = [
                SystemMessage(content="You are a business intelligence assistant. Analyze questions and respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return parse_json_response(response.content)
            
        except Exception as e:
            print(f"      ⚠️ LLM intent analysis failed: {e}")
            return None
    
    def _enhanced_pattern_fallback(self, question: str) -> Dict[str, Any]:
        """Enhanced pattern-based intent analysis with cross-industry support"""
        q_lower = question.lower()
        
        # Determine task type with better patterns
        task_type = 'list'
        limit = None
        
        if any(word in q_lower for word in ['top', 'highest', 'best', 'largest', 'most']):
            task_type = 'ranking'
            # Extract number more accurately
            import re
            numbers = re.findall(r'top\s*(\d+)|(\d+)\s*(?:top|best|highest)', q_lower)
            if numbers:
                limit = int(numbers[0][0] or numbers[0][1])
            else:
                limit = 10  # Default
        elif any(word in q_lower for word in ['total', 'sum', 'revenue', 'amount']):
            task_type = 'aggregation'
        elif any(word in q_lower for word in ['count', 'how many', 'number of']):
            task_type = 'count'
        
        # Enhanced entity detection with cross-industry support
        entities = []
        search_terms = []
        
        entity_patterns = {
            'Customer': ['customer', 'client', 'account', 'user'],
            'Payment': ['payment', 'paid', 'revenue', 'transaction', 'billing'],
            'Contract': ['contract', 'agreement', 'deal', 'συμβόλαια'],
            'Order': ['order', 'purchase', 'sale', 'quote'],
            'Employee': ['employee', 'staff', 'worker', 'personnel'],
            'Product': ['product', 'item', 'catalog', 'inventory'],
            'Invoice': ['invoice', 'bill', 'receipt'],
            'Project': ['project', 'task', 'work'],
            'Vendor': ['vendor', 'supplier', 'partner']
        }
        
        for entity_type, keywords in entity_patterns.items():
            if any(keyword in q_lower for keyword in keywords):
                entities.append(entity_type)
                search_terms.extend([k for k in keywords if k in q_lower])
        
        # Determine what to show
        show_fields = []
        if any(word in q_lower for word in ['name', 'who', 'which', 'title']):
            show_fields.append('names')
        if any(word in q_lower for word in ['amount', 'revenue', 'total', 'paid', 'value', 'price']):
            show_fields.append('amounts')
        if any(word in q_lower for word in ['date', 'when', 'time', 'signed', 'created']):
            show_fields.append('dates')
        
        # Enhanced time filter detection
        time_filter = None
        if '2025' in question:
            time_filter = '2025'
        elif any(phrase in q_lower for phrase in ['this year', 'current year']):
            time_filter = 'this_year'
        elif any(phrase in q_lower for phrase in ['last month', 'previous month']):
            time_filter = 'last_month'
        elif any(phrase in q_lower for phrase in ['recent', 'lately', 'new']):
            time_filter = 'recent'
        
        # Detect filters
        filters = {}
        if 'approved' in q_lower:
            filters['status'] = 'approved'
        if 'active' in q_lower:
            filters['status'] = 'active'
        if 'paid' in q_lower:
            filters['type'] = 'paid'
        
        return {
            'task_type': task_type,
            'entities': entities,
            'show_fields': show_fields,
            'limit': limit,
            'time_filter': time_filter,
            'filters': filters,
            'search_terms': search_terms
        }

class SmartTableSelector:
    """Smart table selection with enhanced scoring"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
    
    def select_tables(self, intent: Dict[str, Any]) -> List[TableInfo]:
        """Smart table selection with multi-factor scoring"""
        print("   📋 Smart table selection...")
        
        target_entities = intent.get('entities', [])
        show_fields = intent.get('show_fields', [])
        search_terms = intent.get('search_terms', [])
        
        # Score all tables
        scored_tables = []
        for table in self.tables:
            score = self._calculate_smart_score(table, target_entities, show_fields, search_terms)
            if score > 0:
                scored_tables.append((table, score))
        
        # Sort by score and select top candidates
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        selected = [table for table, score in scored_tables[:5]]
        
        if selected:
            print(f"      ✅ Selected {len(selected)} tables:")
            for i, table in enumerate(selected):
                score = scored_tables[i][1] if i < len(scored_tables) else 0
                print(f"         {i+1}. {table.name} ({table.entity_type}, score: {score:.2f})")
        else:
            print("      ❌ No suitable tables found")
        
        return selected
    
    def _calculate_smart_score(self, table: TableInfo, target_entities: List[str], 
                              show_fields: List[str], search_terms: List[str]) -> float:
        """Multi-factor scoring algorithm"""
        score = 0.0
        
        # 1. Entity type match (highest weight)
        if table.entity_type in target_entities:
            score += 10.0
        
        # Partial entity matching
        for entity in target_entities:
            if entity.lower() in table.entity_type.lower():
                score += 6.0
        
        # 2. Table name matching
        table_name_lower = table.name.lower()
        for term in search_terms:
            if term.lower() in table_name_lower:
                score += 3.0
        
        # 3. Business priority boost
        priority_scores = {'high': 5.0, 'medium': 2.0, 'low': 0.5}
        priority = getattr(table, 'business_priority', 'medium')
        score += priority_scores.get(priority, 2.0)
        
        # 4. Field capability matching
        if 'names' in show_fields and getattr(table, 'name_columns', []):
            score += 4.0
        if 'amounts' in show_fields and getattr(table, 'measures', []):
            score += 4.0
        if 'dates' in show_fields and getattr(table, 'time_columns', []):
            score += 2.0
        
        # 5. Data availability
        if table.row_count > 0:
            score += 2.0
        if table.row_count > 1000:
            score += 1.0
        
        # 6. BI role preference
        bi_role = getattr(table, 'bi_role', 'dimension')
        if bi_role == 'fact' and 'amounts' in show_fields:
            score += 3.0
        
        # 7. Data quality
        data_quality = getattr(table, 'data_quality', 'production')
        if data_quality == 'production':
            score += 2.0
        elif data_quality == 'test':
            score *= 0.5
        
        # 8. Penalties
        if any(word in table.name.lower() for word in ['temp', 'test', 'backup', 'sys', 'log']):
            score *= 0.1
        
        return score

class FixedSQLTemplateGenerator:
    """Fixed SQL template generator with deterministic patterns (Architecture requirement)"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def generate_sql(self, question: str, intent: Dict[str, Any], tables: List[TableInfo]) -> Optional[str]:
        """Generate SQL using fixed templates with LLM fallback (Architecture requirement)"""
        print("   ⚡ Fixed SQL template generation...")
        
        if not tables:
            return None
        
        # Try deterministic template generation first
        template_sql = self._generate_template_sql(intent, tables[0])
        if template_sql:
            # Validate with sqlglot
            if self._validate_sql_safety(template_sql):
                print("      ✅ SQL generated using fixed templates")
                return template_sql
            else:
                print("      ⚠️ Template SQL failed safety validation")
        
        # Fallback to LLM generation
        llm_sql = await self._generate_llm_sql(question, intent, tables[0])
        if llm_sql:
            if self._validate_sql_safety(llm_sql):
                print("      ✅ SQL generated using LLM fallback")
                return llm_sql
            else:
                print("      ⚠️ LLM SQL failed safety validation")
        
        print("      ❌ SQL generation failed")
        return None
    
    def _generate_template_sql(self, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """Generate SQL using deterministic templates"""
        task_type = intent.get('task_type', 'list')
        limit = intent.get('limit')
        time_filter = intent.get('time_filter')
        filters = intent.get('filters', {})
        
        # Build SELECT clause deterministically
        select_columns = self._build_select_columns(intent, table)
        if not select_columns:
            return None
        
        select_clause = ', '.join(select_columns)
        from_clause = table.full_name
        
        # Build WHERE clause
        where_conditions = []
        
        # Add time filters
        time_conditions = self._build_time_conditions(time_filter, table)
        where_conditions.extend(time_conditions)
        
        # Add other filters
        filter_conditions = self._build_filter_conditions(filters, table)
        where_conditions.extend(filter_conditions)
        
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        # Generate SQL based on task type (deterministic templates)
        if task_type == 'ranking' and limit:
            order_column = self._get_best_order_column(table)
            if order_column:
                sql = f"SELECT TOP ({limit}) {select_clause} FROM {from_clause}"
                if where_clause:
                    sql += f" {where_clause}"
                sql += f" ORDER BY [{order_column}] DESC"
            else:
                sql = f"SELECT TOP ({limit}) {select_clause} FROM {from_clause}"
                if where_clause:
                    sql += f" {where_clause}"
                    
        elif task_type == 'aggregation':
            measure_column = self._get_best_measure_column(table)
            if measure_column:
                sql = f"SELECT SUM([{measure_column}]) AS Total FROM {from_clause}"
                if where_clause:
                    sql += f" {where_clause}"
            else:
                sql = f"SELECT COUNT(*) AS Count FROM {from_clause}"
                if where_clause:
                    sql += f" {where_clause}"
                    
        elif task_type == 'count':
            sql = f"SELECT COUNT(*) AS Count FROM {from_clause}"
            if where_clause:
                sql += f" {where_clause}"
                
        else:  # list
            top_clause = f"TOP ({limit})" if limit else "TOP (50)"
            sql = f"SELECT {top_clause} {select_clause} FROM {from_clause}"
            if where_clause:
                sql += f" {where_clause}"
            
            # Add ORDER BY for consistent results
            order_column = self._get_best_order_column(table)
            if order_column:
                sql += f" ORDER BY [{order_column}]"
        
        return sql
    
    def _build_select_columns(self, intent: Dict[str, Any], table: TableInfo) -> List[str]:
        """Build SELECT columns deterministically"""
        show_fields = intent.get('show_fields', [])
        columns = []
        
        # Add name columns if needed
        if 'names' in show_fields or not show_fields:
            name_cols = getattr(table, 'name_columns', [])
            for col in name_cols[:3]:  # First 3 name columns
                columns.append(f"[{col}]")
        
        # Add key columns if no names available
        if not columns:
            key_cols = getattr(table, 'entity_keys', [])
            for col in key_cols[:1]:  # First key column
                columns.append(f"[{col}]")
        
        # Add measure columns if requested
        if 'amounts' in show_fields:
            measures = getattr(table, 'measures', [])
            for col in measures[:2]:  # First 2 measures
                columns.append(f"[{col}]")
        
        # Add time columns if requested
        if 'dates' in show_fields:
            time_cols = getattr(table, 'time_columns', [])
            for col in time_cols[:1]:  # First time column
                columns.append(f"[{col}]")
        
        # Fallback: use first few columns from table metadata
        if not columns:
            for col in table.columns[:4]:
                col_name = col.get('name', '')
                if col_name and not col_name.startswith('__'):
                    columns.append(f"[{col_name}]")
        
        return columns
    
    def _build_time_conditions(self, time_filter: str, table: TableInfo) -> List[str]:
        """Build time-based WHERE conditions (fixed implementation)"""
        conditions = []
        
        if not time_filter or time_filter == 'null':
            return conditions
        
        # Get best time column
        time_columns = getattr(table, 'time_columns', [])
        if not time_columns:
            # Look for date-like columns in metadata
            for col in table.columns:
                col_name = col.get('name', '').lower()
                col_type = col.get('data_type', '').lower()
                if any(t in col_type for t in ['date', 'time']) or any(w in col_name for w in ['date', 'created', 'signed']):
                    time_columns = [col.get('name')]
                    break
        
        if not time_columns:
            return conditions
        
        time_col = time_columns[0]
        
        # Fixed time filter implementations
        if time_filter == '2025':
            conditions.append(f"YEAR([{time_col}]) = 2025")
        elif time_filter == 'this_year':
            conditions.append(f"YEAR([{time_col}]) = YEAR(GETDATE())")
        elif time_filter == 'last_month':
            conditions.append(f"[{time_col}] >= DATEADD(month, -1, DATEADD(day, 1-DAY(GETDATE()), GETDATE()))")
            conditions.append(f"[{time_col}] < DATEADD(day, 1-DAY(GETDATE()), GETDATE())")
        elif time_filter == 'recent':
            conditions.append(f"[{time_col}] >= DATEADD(day, -30, GETDATE())")
        
        return conditions
    
    def _build_filter_conditions(self, filters: Dict[str, str], table: TableInfo) -> List[str]:
        """Build filter-based WHERE conditions"""
        conditions = []
        
        for filter_key, filter_value in filters.items():
            # Look for matching columns
            for col in table.columns:
                col_name = col.get('name', '').lower()
                if filter_key.lower() in col_name:
                    conditions.append(f"[{col.get('name')}] = '{filter_value}'")
                    break
        
        return conditions
    
    def _get_best_order_column(self, table: TableInfo) -> Optional[str]:
        """Get best column for ORDER BY"""
        # Prefer measures for ordering
        measures = getattr(table, 'measures', [])
        if measures:
            return measures[0]
        
        # Then time columns
        time_cols = getattr(table, 'time_columns', [])
        if time_cols:
            return time_cols[0]
        
        # Then key columns
        key_cols = getattr(table, 'entity_keys', [])
        if key_cols:
            return key_cols[0]
        
        # Finally, first column
        if table.columns:
            return table.columns[0].get('name')
        
        return None
    
    def _get_best_measure_column(self, table: TableInfo) -> Optional[str]:
        """Get best measure column for aggregation"""
        measures = getattr(table, 'measures', [])
        return measures[0] if measures else None
    
    def _validate_sql_safety(self, sql: str) -> bool:
        """Validate SQL safety using sqlglot (Architecture requirement)"""
        if not sql or len(sql.strip()) < 5:
            return False
        
        # Basic safety checks first
        sql_upper = sql.upper().strip()
        
        # Must start with safe operations
        if not any(sql_upper.startswith(start) for start in ['SELECT', 'WITH']):
            return False
        
        # Check for dangerous operations
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'SP_', 'XP_', 'DBCC', 'GRANT', 
            'REVOKE', 'DENY', 'BULK', 'OPENROWSET', 'OPENDATASOURCE'
        ]
        
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {sql_upper} ' or f' {keyword}(' in f' {sql_upper} ':
                return False
        
        # Enhanced validation with sqlglot if available
        if HAS_SQLGLOT:
            try:
                # Parse with T-SQL dialect
                parsed = sqlglot.parse_one(sql, dialect="tsql")
                if not parsed:
                    return False
                
                # Check that it's a SELECT statement
                if not isinstance(parsed, sqlglot.expressions.Select):
                    return False
                
                # Additional safety checks could be added here
                return True
                
            except Exception as e:
                print(f"      ⚠️ sqlglot validation failed: {e}")
                return False
        
        # If sqlglot not available, rely on basic checks
        return True
    
    async def _generate_llm_sql(self, question: str, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """LLM fallback SQL generation with enhanced constraints"""
        try:
            table_context = self._build_sql_context(table)
            
            prompt = f"""Generate safe T-SQL for this question using the provided table.

QUESTION: "{question}"

INTENT ANALYSIS:
- Task: {intent.get('task_type', 'list')}
- Show fields: {intent.get('show_fields', [])}
- Limit: {intent.get('limit', 'none')}
- Time filter: {intent.get('time_filter', 'none')}
- Filters: {intent.get('filters', {})}

{table_context}

REQUIREMENTS:
- Generate ONLY safe SELECT statements for SQL Server
- Use square brackets around all column/table names: [column_name]
- For rankings: SELECT TOP (N) ... ORDER BY [measure_column] DESC
- For totals: SELECT SUM([measure_column]) AS Total FROM [table]
- For counts: SELECT COUNT(*) AS Count FROM [table]
- Use only columns that exist in the table schema
- Include proper WHERE clauses for time filters
- NO INSERT, UPDATE, DELETE, or other dangerous operations

Return only the T-SQL query:"""

            messages = [
                SystemMessage(content="Generate safe T-SQL SELECT queries only. Use proper SQL Server syntax with square brackets. Return only SQL code."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return clean_sql_query(response.content)
            
        except Exception as e:
            print(f"      ⚠️ LLM SQL generation failed: {e}")
            return None
    
    def _build_sql_context(self, table: TableInfo) -> str:
        """Build context for LLM SQL generation"""
        lines = [f"TABLE: {table.full_name}"]
        lines.append(f"ENTITY TYPE: {table.entity_type}")
        lines.append(f"ROW COUNT: {table.row_count:,}")
        
        # Show columns with business context
        lines.append("\nCOLUMNS:")
        for col in table.columns[:15]:
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            
            # Add business annotations
            annotations = []
            if col_name in getattr(table, 'name_columns', []):
                annotations.append('NAME')
            if col_name in getattr(table, 'measures', []):
                annotations.append('MEASURE')
            if col_name in getattr(table, 'entity_keys', []):
                annotations.append('KEY')
            if col_name in getattr(table, 'time_columns', []):
                annotations.append('TIME')
            
            ann_str = f" -- {', '.join(annotations)}" if annotations else ""
            lines.append(f"  [{col_name}] ({col_type}){ann_str}")
        
        return '\n'.join(lines)

class EnhancedQueryExecutor:
    """Enhanced query executor with better error handling"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute SQL with enhanced error handling and UTF-8 support"""
        print("   🔄 Executing query...")
        print(f"      📝 SQL: {sql[:100]}{'...' if len(sql) > 100 else ''}")
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                # Enhanced UTF-8 support for international characters
                if self.config.utf8_encoding:
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
                    
                    print(f"      ✅ Query executed successfully: {len(results)} rows")
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Query failed: {error_msg}")
            
            # Enhanced error classification and suggestions
            if "Incorrect syntax" in error_msg or "syntax error" in error_msg.lower():
                error_msg += " | Suggestion: Check column names and SQL syntax"
            elif "Invalid column name" in error_msg:
                error_msg += " | Suggestion: Column may not exist in table"
            elif "Invalid object name" in error_msg:
                error_msg += " | Suggestion: Table may not exist or access denied"
            elif "Permission denied" in error_msg:
                error_msg += " | Suggestion: Insufficient database permissions"
            elif "Timeout" in error_msg:
                error_msg += " | Suggestion: Query may be too complex, try simpler query"
            
            return [], error_msg

class QueryInterface:
    """Enhanced Query Interface with fixed templates and safety validation"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            request_timeout=60,
        )
        
        # Initialize enhanced components
        self.intent_analyzer = EnhancedIntentAnalyzer(self.llm)
        self.sql_generator = FixedSQLTemplateGenerator(self.llm)
        self.executor = EnhancedQueryExecutor(config)
        
        print("✅ Enhanced Query Interface initialized")
        print("   🧠 Cross-industry intent analysis")
        print("   📋 Smart multi-factor table selection")
        print("   ⚡ Fixed SQL templates with LLM fallback")
        print(f"   🛡️ Safety validation: {'✅ sqlglot' if HAS_SQLGLOT else '⚠️ basic only'}")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start enhanced interactive session"""
        
        # Initialize smart table selector
        self.table_selector = SmartTableSelector(tables)
        
        # Show system readiness
        self._show_system_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🔄 Processing with enhanced pipeline...")
                
                start_time = time.time()
                result = await self.process_query(question, tables)
                result.execution_time = time.time() - start_time
                
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_query(self, question: str, tables: List[TableInfo]) -> QueryResult:
        """Process query using enhanced 3-stage pipeline"""
        
        try:
            # Stage 1: Enhanced intent analysis
            intent = await self.intent_analyzer.analyze_intent(question, tables)
            
            # Stage 2: Smart table selection
            selected_tables = self.table_selector.select_tables(intent)
            
            if not selected_tables:
                available_entities = list(set(t.entity_type for t in tables[:15]))
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error=f"No suitable tables found. Available entities: {', '.join(available_entities)}",
                    result_type="error"
                )
            
            # Stage 3: Fixed SQL generation with safety validation
            sql = await self.sql_generator.generate_sql(question, intent, selected_tables)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not generate safe SQL for this query. Try rephrasing with specific entity names.",
                    result_type="error"
                )
            
            # Execute validated SQL
            results, error = await self.executor.execute_query(sql)
            
            return QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=[t.full_name for t in selected_tables],
                result_type="data" if results and not error else "error",
                sql_generation_method="fixed_template" if "template" in sql else "llm_fallback"
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Query processing failed: {str(e)}",
                result_type="error"
            )
    
    def _show_system_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show enhanced system readiness with cross-industry entities"""
        # Analyze available entities
        entities = {}
        for table in tables:
            entity = table.entity_type
            entities[entity] = entities.get(entity, 0) + 1
        
        print(f"\n🚀 ENHANCED QUERY SYSTEM READY:")
        print(f"   📊 Total objects: {len(tables)}")
        
        # Show entities with priorities
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
        priority_entities = ['Customer', 'Payment', 'Contract', 'Order', 'Employee']
        
        for entity, count in sorted_entities[:8]:
            priority_emoji = "🔥" if entity in priority_entities else "📋"
            print(f"   {priority_emoji} {entity}: {count} objects")
        
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
            print(f"   🏭 Industry: {domain.industry}")
        
        print(f"\n🔄 ENHANCED 3-STAGE PIPELINE:")
        print(f"   1. Cross-Industry Intent Analysis")
        print(f"   2. Smart Multi-Factor Table Selection")
        print(f"   3. Fixed SQL Templates + Safety Validation")
        
        print(f"\n💡 Example questions:")
        if domain and domain.sample_questions:
            for i, question in enumerate(domain.sample_questions[:3], 1):
                print(f"   {i}. {question}")
        else:
            print("   • Show customer names")
            print("   • Total revenue this year") 
            print("   • Top 10 contracts by amount")
    
    def _display_result(self, result: QueryResult):
        """Display enhanced query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.is_successful() and result.has_results():
            print(f"✅ QUERY SUCCESSFUL")
            print(f"\n📋 Generated SQL ({result.sql_generation_method}):")
            print(f"{result.sql_query}")
            
            if result.tables_used:
                print(f"\n📊 Tables Used: {', '.join([t.split('.')[-1].replace(']', '') for t in result.tables_used])}")
            
            print(f"\n📈 Results ({len(result.results)} rows):")
            self._display_data(result.results)
            
        else:
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
            
            if result.tables_used:
                print(f"   Tables attempted: {', '.join(result.tables_used)}")
            
            print(f"\n💡 Suggestions:")
            print(f"   • Try specific entity names: 'customers', 'payments', 'contracts'")
            print(f"   • Use simple questions: 'show customer names', 'total revenue'")
            print(f"   • Check spelling and try rephrasing")
    
    def _display_data(self, results: List[Dict[str, Any]]):
        """Display query results with enhanced formatting"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result (aggregation)
            key, value = next(iter(results[0].items()))
            if isinstance(value, (int, float)) and abs(value) >= 1000:
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            print(f"   🎯 {key}: {formatted_value}")
            
        else:
            # Multiple rows - show with enhanced formatting
            for i, row in enumerate(results[:20], 1):  # Show first 20 rows
                parts = []
                for j, (key, value) in enumerate(row.items()):
                    if j >= 5:  # Limit to 5 columns for readability
                        break
                    
                    # Smart value formatting
                    if isinstance(value, (int, float)) and abs(value) >= 1000:
                        formatted_value = f"{value:,}"
                    elif isinstance(value, str) and len(value) > 40:
                        formatted_value = value[:40] + "..."
                    else:
                        formatted_value = str(value) if value is not None else ""
                    
                    # Smart key formatting
                    display_key = key.replace('_', ' ').title() if '_' in key else key
                    parts.append(f"{display_key}: {formatted_value}")
                
                print(f"   {i:2d}. {' | '.join(parts)}")
            
            if len(results) > 20:
                print(f"   ... and {len(results) - 20} more rows")