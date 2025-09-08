#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - Fixed to Only Use Discovered Columns
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
FIXED: Only use columns that actually exist in discovered table structure
"""

import asyncio
import pyodbc
import time
from typing import List, Dict, Any, Optional

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# SQLGlot for safety validation (Architecture requirement)
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import parse_json_response, clean_sql_query, safe_database_value

class IntentParser:
    """Parse user intent for deterministic template selection"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def parse_intent(self, question: str) -> Dict[str, Any]:
        """Parse user intent into structured format"""
        print("   🎯 Parsing intent...")
        
        try:
            prompt = f"""Parse this business question into structured intent:

QUESTION: "{question}"

Respond with JSON only:
{{
  "task_type": "list|count|aggregation|ranking",
  "entities": ["Customer", "Payment", "Order", "Employee"],
  "show_fields": ["name", "amount", "date"],
  "limit": 10,
  "time_filter": "this_year|last_month|2025|null",
  "filters": {{"status": "active", "type": "paid"}},
  "group_by": ["customer", "month"]
}}

TASK TYPES:
- list: Show records
- count: Count records  
- aggregation: Sum/total/average
- ranking: Top N by measure

ENTITIES: Customer, Payment, Order, Product, Contract, Employee, Vendor, Project, Asset, Inventory, Event, System, Other

TIME FILTERS:
- this_year, last_month, 2025, between_dates, null

Examples:
"top customers by revenue" → task_type: "ranking", entities: ["Customer"], show_fields: ["name", "revenue"], time_filter: null
"total payments this year" → task_type: "aggregation", entities: ["Payment"], show_fields: ["amount"], time_filter: "this_year"
"customer count" → task_type: "count", entities: ["Customer"]"""

            messages = [
                SystemMessage(content="Parse business questions into structured intent. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            result = parse_json_response(response.content)
            
            if result:
                print(f"      ✅ Task: {result.get('task_type', 'unknown')}")
                print(f"      📋 Entities: {', '.join(result.get('entities', []))}")
                return result
            
            return self._create_fallback_intent(question)
            
        except Exception as e:
            print(f"      ⚠️ Intent parsing failed: {e}")
            return self._create_fallback_intent(question)
    
    def _create_fallback_intent(self, question: str) -> Dict[str, Any]:
        """Create fallback intent from keywords"""
        q_lower = question.lower()
        
        # Simple keyword detection
        if any(word in q_lower for word in ['top', 'highest', 'best', 'ranking']):
            task_type = "ranking"
            limit = 10
        elif any(word in q_lower for word in ['total', 'sum', 'amount']):
            task_type = "aggregation"  
            limit = None
        elif any(word in q_lower for word in ['count', 'how many', 'number']):
            task_type = "count"
            limit = None
        else:
            task_type = "list"
            limit = 20
        
        # Entity detection
        entities = []
        if any(word in q_lower for word in ['customer', 'client']):
            entities.append('Customer')
        if any(word in q_lower for word in ['payment', 'revenue', 'money']):
            entities.append('Payment')
        if any(word in q_lower for word in ['order', 'sale']):
            entities.append('Order')
        if any(word in q_lower for word in ['employee', 'staff']):
            entities.append('Employee')
        
        if not entities:
            entities = ['Customer']  # Default
        
        # Time filter detection
        time_filter = None
        if any(word in q_lower for word in ['2025']):
            time_filter = '2025'
        elif any(word in q_lower for word in ['this year', 'current year']):
            time_filter = 'this_year'
        elif any(word in q_lower for word in ['last month']):
            time_filter = 'last_month'
        
        return {
            'task_type': task_type,
            'entities': entities,
            'show_fields': ['name', 'amount'],
            'limit': limit,
            'time_filter': time_filter,
            'filters': {},
            'group_by': []
        }

class SmartTableSelector:
    """Smart table selection with multi-factor scoring"""
    
    def select_tables(self, intent: Dict[str, Any], tables: List[TableInfo]) -> List[TableInfo]:
        """Select best tables using multi-factor scoring"""
        print("   📊 Smart table selection...")
        
        scored_tables = []
        required_entities = intent.get('entities', [])
        task_type = intent.get('task_type', 'list')
        time_filter = intent.get('time_filter')
        
        for table in tables:
            score = self._score_table(table, required_entities, task_type, time_filter)
            if score > 0.1:  # Minimum threshold
                scored_tables.append((table, score))
        
        # Sort by score and return top tables
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        selected = [table for table, score in scored_tables[:3]]
        
        if selected:
            for table in selected[:2]:
                print(f"      ✅ {table.name}: {table.entity_type} ({table.row_count:,} rows)")
        
        return selected
    
    def _score_table(self, table: TableInfo, entities: List[str], task_type: str, time_filter: str) -> float:
        """Score table using multi-factor analysis"""
        score = 0.0
        
        # 1. Entity fit (30%)
        entity_score = 0.0
        if table.entity_type in entities:
            entity_score = 1.0
        elif any(entity.lower() in table.name.lower() for entity in entities):
            entity_score = 0.7
        score += entity_score * 0.3
        
        # 2. Measure fit (25%) - for aggregation/ranking tasks
        measure_score = 0.0
        if task_type in ['aggregation', 'ranking']:
            if hasattr(table, 'measures') and table.measures:
                measure_score = 1.0
            elif any(col.get('data_type', '').lower() in ['decimal', 'money', 'float', 'numeric'] 
                    for col in table.columns):
                measure_score = 0.8
            elif any(word in col.get('name', '').lower() 
                    for col in table.columns 
                    for word in ['amount', 'total', 'revenue', 'value', 'price']):
                measure_score = 0.9
        else:
            measure_score = 0.5  # Not critical for list/count
        score += measure_score * 0.25
        
        # 3. Time fit (20%) - for time-filtered queries
        time_score = 0.0
        if time_filter:
            if hasattr(table, 'time_columns') and table.time_columns:
                time_score = 1.0
            elif any(col.get('data_type', '').lower() in ['datetime', 'date', 'timestamp']
                    for col in table.columns):
                time_score = 0.9
            elif any(word in col.get('name', '').lower()
                    for col in table.columns
                    for word in ['date', 'time', 'created', 'modified']):
                time_score = 0.8
        else:
            time_score = 0.7  # Neutral if no time filter
        score += time_score * 0.20
        
        # 4. Row estimate (10%) - prefer fact tables for aggregations
        row_score = 0.0
        if table.row_count > 1000:
            row_score = 1.0
        elif table.row_count > 100:
            row_score = 0.8
        elif table.row_count > 10:
            row_score = 0.6
        else:
            row_score = 0.3
        score += row_score * 0.10
        
        # 5. Business priority (10%)
        priority_score = 0.0
        if hasattr(table, 'business_priority'):
            if table.business_priority == 'high':
                priority_score = 1.0
            elif table.business_priority == 'medium':
                priority_score = 0.7
            else:
                priority_score = 0.4
        else:
            priority_score = 0.5
        score += priority_score * 0.10
        
        # 6. Name columns (5%) - for display purposes
        name_score = 0.0
        if hasattr(table, 'name_columns') and table.name_columns:
            name_score = 1.0
        elif any(word in col.get('name', '').lower()
                for col in table.columns
                for word in ['name', 'title', 'description']):
            name_score = 0.8
        score += name_score * 0.05
        
        return min(score, 1.0)

class TemplateGenerator:
    """FIXED: Generate SQL using only discovered columns"""
    
    def generate_sql(self, intent: Dict[str, Any], tables: List[TableInfo]) -> Optional[str]:
        """FIXED: Generate SQL using only discovered columns"""
        if not tables:
            return None
        
        print("   ⚙️ Template-based SQL generation...")
        
        task_type = intent.get('task_type', 'list')
        table = tables[0]  # Use best table
        
        # FIXED: Validate table has columns before proceeding
        if not table.columns:
            print(f"      ❌ No columns discovered for table {table.name}")
            return None
        
        try:
            if task_type == 'ranking':
                return self._generate_ranking_sql(intent, table)
            elif task_type == 'aggregation':
                return self._generate_aggregation_sql(intent, table)
            elif task_type == 'count':
                return self._generate_count_sql(intent, table)
            else:  # list
                return self._generate_list_sql(intent, table)
        except Exception as e:
            print(f"      ⚠️ Template generation failed: {e}")
            return None
    
    def _generate_ranking_sql(self, intent: Dict[str, Any], table: TableInfo) -> str:
        """FIXED: Generate TOP N ranking query using only discovered columns"""
        limit = intent.get('limit', 10)
        
        # FIXED: Find columns that actually exist
        name_col = self._find_actual_name_column(table)
        measure_col = self._find_actual_measure_column(table)
        
        if not measure_col:
            # FIXED: If no measure column, try to find any numeric column
            measure_col = self._find_any_numeric_column(table)
            if not measure_col:
                raise ValueError(f"No measure/numeric column found in {table.name}")
        
        # Build SELECT using only existing columns
        select_cols = []
        if name_col:
            select_cols.append(f"[{name_col}]")
        select_cols.append(f"[{measure_col}]")
        
        # FIXED: Build WHERE for time filter using discovered columns
        where_clause = self._build_where_conditions(intent, table)
        
        sql = f"SELECT TOP ({limit}) {', '.join(select_cols)} FROM {table.full_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += f" ORDER BY [{measure_col}] DESC"
        
        return sql
    
    def _generate_aggregation_sql(self, intent: Dict[str, Any], table: TableInfo) -> str:
        """FIXED: Generate aggregation query using discovered columns"""
        measure_col = self._find_actual_measure_column(table)
        
        if not measure_col:
            # FIXED: Try any numeric column for aggregation
            measure_col = self._find_any_numeric_column(table)
            if not measure_col:
                raise ValueError(f"No measure/numeric column found in {table.name}")
        
        # FIXED: Build WHERE using discovered columns
        where_clause = self._build_where_conditions(intent, table)
        
        sql = f"SELECT SUM([{measure_col}]) AS Total FROM {table.full_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        return sql
    
    def _generate_count_sql(self, intent: Dict[str, Any], table: TableInfo) -> str:
        """Generate count query using discovered columns"""
        where_clause = self._build_where_conditions(intent, table)
        
        sql = f"SELECT COUNT(*) AS Count FROM {table.full_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        return sql
    
    def _generate_list_sql(self, intent: Dict[str, Any], table: TableInfo) -> str:
        """FIXED: Generate list query using only discovered columns"""
        limit = intent.get('limit', 20)
        
        # FIXED: Get display columns that actually exist
        display_cols = self._get_actual_display_columns(table, 4)
        
        if not display_cols:
            raise ValueError(f"No suitable display columns found in {table.name}")
        
        # FIXED: Build WHERE using discovered columns
        where_clause = self._build_where_conditions(intent, table)
        
        sql = f"SELECT TOP ({limit}) {', '.join(f'[{col}]' for col in display_cols)} FROM {table.full_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        # FIXED: Order by best available column
        sort_col = self._find_actual_sort_column(table)
        if sort_col:
            sql += f" ORDER BY [{sort_col}] DESC"
        
        return sql
    
    def _find_actual_name_column(self, table: TableInfo) -> Optional[str]:
        """FIXED: Find name column that actually exists in discovered columns"""
        # First check if semantic analysis provided name columns
        if hasattr(table, 'name_columns') and table.name_columns:
            # Verify the column actually exists
            for name_col in table.name_columns:
                if self._column_exists(table, name_col):
                    return name_col
        
        # Search through actual discovered columns
        for col in table.columns:
            col_name = col.get('name', '')
            if col_name and any(word in col_name.lower() for word in ['name', 'title', 'description']):
                return col_name
        
        # Fallback to first string column
        for col in table.columns:
            col_type = col.get('data_type', '').lower()
            if col_type in ['varchar', 'nvarchar', 'char', 'nchar', 'text']:
                return col.get('name')
        
        return None
    
    def _find_actual_measure_column(self, table: TableInfo) -> Optional[str]:
        """FIXED: Find measure column that actually exists"""
        # First check semantic analysis measures
        if hasattr(table, 'measures') and table.measures:
            for measure_col in table.measures:
                if self._column_exists(table, measure_col):
                    return measure_col
        
        # Look for columns with revenue/amount names AND numeric types
        for col in table.columns:
            col_name = col.get('name', '').lower()
            col_type = col.get('data_type', '').lower()
            
            if (any(word in col_name for word in ['amount', 'total', 'revenue', 'value', 'price']) and
                col_type in ['decimal', 'money', 'float', 'numeric', 'int']):
                return col.get('name')
        
        return None
    
    def _find_any_numeric_column(self, table: TableInfo) -> Optional[str]:
        """FIXED: Find any numeric column that exists"""
        for col in table.columns:
            col_type = col.get('data_type', '').lower()
            if col_type in ['decimal', 'money', 'float', 'numeric', 'int', 'bigint']:
                return col.get('name')
        return None
    
    def _find_actual_sort_column(self, table: TableInfo) -> Optional[str]:
        """FIXED: Find sort column that actually exists"""
        # Prefer measure columns
        measure_col = self._find_actual_measure_column(table)
        if measure_col:
            return measure_col
        
        # Then date columns
        for col in table.columns:
            col_type = col.get('data_type', '').lower()
            if 'date' in col_type or 'time' in col_type:
                return col.get('name')
        
        # Then ID columns
        for col in table.columns:
            col_name = col.get('name', '').lower()
            if col_name.endswith('id') or 'id' in col_name:
                return col.get('name')
        
        # Fallback to first column
        if table.columns:
            return table.columns[0].get('name')
        
        return None
    
    def _get_actual_display_columns(self, table: TableInfo, max_cols: int) -> List[str]:
        """FIXED: Get display columns that actually exist"""
        cols = []
        
        # Name columns first (if they exist)
        name_col = self._find_actual_name_column(table)
        if name_col:
            cols.append(name_col)
        
        # Measure columns (if they exist)
        measure_col = self._find_actual_measure_column(table)
        if measure_col and measure_col not in cols:
            cols.append(measure_col)
        
        # Fill with other existing columns
        for col in table.columns:
            if len(cols) >= max_cols:
                break
            col_name = col.get('name', '')
            if col_name and col_name not in cols:
                cols.append(col_name)
        
        return cols[:max_cols]
    
    def _build_where_conditions(self, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """FIXED: Build WHERE conditions using discovered columns only"""
        time_filter = intent.get('time_filter')
        if not time_filter:
            return None
        
        # FIXED: Find date column that actually exists
        date_col = None
        
        # Check semantic analysis time columns first
        if hasattr(table, 'time_columns') and table.time_columns:
            for time_col in table.time_columns:
                if self._column_exists(table, time_col):
                    date_col = time_col
                    break
        
        # Search discovered columns
        if not date_col:
            for col in table.columns:
                col_type = col.get('data_type', '').lower()
                col_name = col.get('name', '').lower()
                
                if ('date' in col_type or 'time' in col_type or 
                    any(word in col_name for word in ['date', 'time', 'created'])):
                    date_col = col.get('name')
                    break
        
        if not date_col:
            return None
        
        # Build condition based on time filter
        if time_filter == 'this_year':
            return f"YEAR([{date_col}]) = YEAR(GETDATE())"
        elif time_filter == '2025':
            return f"YEAR([{date_col}]) = 2025"
        elif time_filter == 'last_month':
            return f"[{date_col}] >= EOMONTH(GETDATE(), -2) AND [{date_col}] < EOMONTH(GETDATE(), -1)"
        
        return None
    
    def _column_exists(self, table: TableInfo, column_name: str) -> bool:
        """FIXED: Check if column actually exists in discovered columns"""
        if not column_name or not table.columns:
            return False
        
        existing_columns = [col.get('name', '') for col in table.columns]
        return column_name in existing_columns

class SafetyValidator:
    """Fixed SQL safety validation"""
    
    def validate_sql_safety(self, sql: str) -> bool:
        """Enhanced SQL safety validation (FIXED)"""
        if not sql or len(sql.strip()) < 5:
            return False
        
        # FIXED: Don't prepend leading space that breaks startswith
        sql_stripped = sql.strip()
        sql_normalized = sql_stripped.upper()
        
        # FIXED: Check without leading space
        if not sql_normalized.startswith(('SELECT', 'WITH')):
            return False
        
        # Dangerous operations
        dangerous_patterns = [
            r'\b(?:INSERT|UPDATE|DELETE|MERGE|REPLACE)\b',
            r'\b(?:DROP|CREATE|ALTER|TRUNCATE)\b',
            r'\b(?:EXEC|EXECUTE|SP_|XP_|DBCC)\b',
            r'\b(?:GRANT|REVOKE|DENY)\b',
            r'\b(?:BULK|OPENROWSET|OPENDATASOURCE)\b',
            r'\bXP_CMDSHELL\b',
            r'\b(?:BACKUP|RESTORE)\b'
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_normalized, re.IGNORECASE):
                return False
        
        # Check for multiple statements
        sql_clean = sql.strip().rstrip(';')
        if ';' in sql_clean:
            return False
        
        # FIXED: sqlglot validation with CTE support
        if HAS_SQLGLOT:
            try:
                parsed = sqlglot.parse_one(sql, dialect="tsql")
                if not parsed:
                    return False
                
                # FIXED: Accept both SELECT and WITH (CTE)
                if isinstance(parsed, sqlglot.expressions.Select):
                    return True
                elif isinstance(parsed, sqlglot.expressions.With):
                    # CTE - check if it contains a SELECT
                    for node in parsed.walk():
                        if isinstance(node, sqlglot.expressions.Select):
                            return True
                    return False
                else:
                    return False
                
            except Exception:
                return False
        
        return True

class QueryExecutor:
    """Query executor with enhanced error handling"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_query(self, sql: str):
        """Execute SQL with enhanced error handling"""
        print("   🔄 Executing query...")
        print(f"      📝 SQL: {sql[:80]}{'...' if len(sql) > 80 else ''}")
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
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
                    
                    print(f"      ✅ Query executed: {len(results)} rows")
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Query failed: {error_msg}")
            
            # Enhanced error suggestions
            if "Incorrect syntax" in error_msg:
                error_msg += " | Try simpler column names or check table structure"
            elif "Invalid column name" in error_msg:
                error_msg += " | Column may not exist in selected table"
            elif "Invalid object name" in error_msg:
                error_msg += " | Table may not exist or wrong schema"
            elif "Arithmetic overflow" in error_msg:
                error_msg += " | Try using different numeric columns"
            
            return [], error_msg

class QueryInterface:
    """Enhanced Query Interface with fixed safety validation and discovered-column-only approach"""
    
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
        
        # Initialize components
        self.intent_parser = IntentParser(self.llm)
        self.table_selector = SmartTableSelector()
        self.template_generator = TemplateGenerator()
        self.safety_validator = SafetyValidator()
        self.executor = QueryExecutor(config)
        
        print("✅ Enhanced Query Interface initialized")
        print("   🎯 Template-first SQL generation")
        print("   📊 Smart multi-factor table selection")
        print("   🛡️ FIXED: Only use discovered columns")
        print(f"   🛡️ Fixed safety validation: {'✅ sqlglot + CTE' if HAS_SQLGLOT else '⚠️ basic only'}")
    
    async def start_session(self, tables: List[TableInfo], domain: Optional[BusinessDomain], relationships: List[Relationship]):
        """Start enhanced session with template-first approach"""
        
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
                print(f"🔄 Processing...")
                
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
        """FIXED: Process query with discovered-columns-only approach"""
        
        try:
            # Stage 1: Parse intent 
            intent = await self.intent_parser.parse_intent(question)
            
            if not intent or intent.get('task_type') == 'unknown':
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not understand the question. Please try rephrasing.",
                    result_type="error"
                )
            
            # Stage 2: Smart table selection
            selected_tables = self.table_selector.select_tables(intent, tables)
            
            if not selected_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No suitable tables found for this query. Try asking about available data entities.",
                    result_type="error"
                )
            
            # FIXED: Validate selected table has columns
            best_table = selected_tables[0]
            if not best_table.columns:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error=f"Selected table {best_table.name} has no discovered columns. Cannot generate SQL.",
                    result_type="error"
                )
            
            # Stage 3: Template-based SQL generation using only discovered columns
            sql = self.template_generator.generate_sql(intent, selected_tables)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not generate SQL template using discovered columns. Try a simpler query type.",
                    result_type="error"
                )
            
            # FIXED: Normalize SQL first, then validate
            cleaned_sql = clean_sql_query(sql)
            
            # FIXED: Safety validation
            if not self.safety_validator.validate_sql_safety(cleaned_sql):
                return QueryResult(
                    question=question,
                    sql_query=cleaned_sql,
                    results=[],
                    error="Generated SQL failed safety validation. Try a simpler query.",
                    result_type="error"
                )
            
            # Execute validated SQL
            results, error = await self.executor.execute_query(cleaned_sql)
            
            # Extract table names
            tables_used = [table.full_name for table in selected_tables]
            
            return QueryResult(
                question=question,
                sql_query=cleaned_sql,
                results=results,
                error=error,
                tables_used=tables_used,
                result_type="data" if results and not error else "error",
                sql_generation_method="template_discovered_columns",
                intent_confidence=0.9
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
        """Show enhanced system readiness"""
        
        # Analyze available data
        business_areas = {
            'Customer Data': 0,
            'Financial Data': 0, 
            'Sales Data': 0,
            'Employee Data': 0,
            'Contract Data': 0,
            'Other Data': 0
        }
        
        for table in tables:
            entity_type = getattr(table, 'entity_type', 'Other')
            
            if entity_type == 'Customer':
                business_areas['Customer Data'] += 1
            elif entity_type == 'Payment':
                business_areas['Financial Data'] += 1
            elif entity_type in ['Order', 'Sale']:
                business_areas['Sales Data'] += 1
            elif entity_type == 'Employee':
                business_areas['Employee Data'] += 1
            elif entity_type == 'Contract':
                business_areas['Contract Data'] += 1
            else:
                business_areas['Other Data'] += 1
        
        print(f"\n🚀 DISCOVERED-COLUMNS-ONLY QUERY SYSTEM READY:")
        print(f"   📊 Total tables: {len(tables)}")
        
        for area, count in business_areas.items():
            if count > 0:
                emoji = "🔥" if area in ['Customer Data', 'Financial Data', 'Sales Data'] else "📋"
                print(f"   {emoji} {area}: {count} tables")
        
        print(f"\n⚙️ ENHANCED PIPELINE:")
        print(f"   1. 🎯 Intent parsing with structured output")
        print(f"   2. 📊 Multi-factor table scoring")
        print(f"   3. ⚙️ Template-based SQL using ONLY discovered columns")
        print(f"   4. 🛡️ Fixed safety validation + normalization")
        
        print(f"\n💡 Try asking:")
        print(f"   • Who are the top 10 customers by revenue?")
        print(f"   • What is our total revenue this year?")
        print(f"   • Show me customer payment data")
        print(f"   • Count active contracts")
    
    def _display_result(self, result: QueryResult):
        """Display query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 50)
        
        if result.is_successful() and result.has_results():
            print(f"✅ QUERY SUCCESSFUL")
            print(f"\n📋 Generated SQL ({result.sql_generation_method}):")
            print(f"{result.sql_query}")
            
            if result.tables_used:
                # Clean table names for display
                clean_names = []
                for table in result.tables_used:
                    clean_name = table.replace('[', '').replace(']', '').split('.')[-1]
                    clean_names.append(clean_name)
                print(f"\n📊 Tables Used: {', '.join(clean_names)}")
            
            print(f"\n📈 Results ({len(result.results)} rows):")
            self._display_data(result.results)
            
        else:
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
            
            print(f"\n💡 Template-first suggestions:")
            print(f"   • Try: 'top 10 customers by revenue'")
            print(f"   • Try: 'total payments this year'")
            print(f"   • Try: 'count of active customers'")
    
    def _display_data(self, results: List[Dict[str, Any]]):
        """Enhanced data display with business formatting"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            key, value = next(iter(results[0].items()))
            
            # Format based on column name
            if any(word in key.lower() for word in ['revenue', 'amount', 'total', 'value']):
                if isinstance(value, (int, float)) and abs(value) >= 1000:
                    formatted_value = f"${value:,.2f}"
                else:
                    formatted_value = f"${value}"
            elif any(word in key.lower() for word in ['count', 'number', 'qty']):
                formatted_value = f"{value:,}" if isinstance(value, (int, float)) else str(value)
            else:
                formatted_value = str(value)
            
            print(f"   🎯 {key}: {formatted_value}")
            
        else:
            # Multiple rows - enhanced business formatting
            for i, row in enumerate(results[:15], 1):
                parts = []
                for j, (key, value) in enumerate(row.items()):
                    if j >= 4:  # Limit columns
                        break
                    
                    # Enhanced business formatting
                    if isinstance(value, (int, float)):
                        if any(word in key.lower() for word in ['revenue', 'amount', 'total', 'value', 'price']):
                            formatted_value = f"${value:,.2f}" if abs(value) >= 1000 else f"${value}"
                        elif any(word in key.lower() for word in ['count', 'number', 'qty']):
                            formatted_value = f"{value:,}"
                        else:
                            formatted_value = f"{value:,}" if abs(value) >= 1000 else str(value)
                    elif isinstance(value, str):
                        formatted_value = value[:40] + "..." if len(value) > 40 else value
                    else:
                        formatted_value = str(value) if value is not None else ""
                    
                    # Enhanced key formatting
                    if any(word in key.lower() for word in ['revenue', 'amount', 'total']):
                        display_key = "Revenue"
                    elif any(word in key.lower() for word in ['name', 'title']):
                        display_key = key.replace('_', ' ').title()
                    elif any(word in key.lower() for word in ['count', 'number']):
                        display_key = "Count"
                    else:
                        display_key = key.replace('_', ' ').title()
                    
                    parts.append(f"{display_key}: {formatted_value}")
                
                print(f"   {i:2d}. {' | '.join(parts)}")
            
            if len(results) > 15:
                print(f"   ... and {len(results) - 15} more rows")