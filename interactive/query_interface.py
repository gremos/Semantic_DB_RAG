#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - Fixed SQL Templates & Safety Validation
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
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

class IntentAnalyzer:
    """Intent analysis with business context"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str, tables: List[TableInfo]) -> Dict[str, Any]:
        """Analyze intent with cross-industry entity support"""
        print("   🧠 Intent analysis...")
        
        # Build entity context
        entity_context = self._build_entity_context(tables)
        
        # Get LLM analysis
        intent = await self._get_intent_from_llm(question, entity_context)
        
        if intent:
            print(f"      ✅ Intent: {intent.get('task_type', 'unknown')}")
            print(f"      🎯 Entities: {', '.join(intent.get('entities', []))}")
            return intent
        
        # Pattern-based fallback
        return self._pattern_fallback(question)
    
    def _build_entity_context(self, tables: List[TableInfo]) -> str:
        """Build context with available entities"""
        entity_groups = {}
        for table in tables[:20]:
            entity_type = table.entity_type
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(table)
        
        context_lines = ["AVAILABLE ENTITIES:"]
        
        priority_entities = ['Customer', 'Payment', 'Contract', 'Order', 'Employee']
        
        for entity_type in priority_entities:
            if entity_type in entity_groups:
                tables_list = entity_groups[entity_type]
                context_lines.append(f"\n{entity_type.upper()}:")
                for table in tables_list[:2]:
                    context_lines.append(f"  - {table.name}")
        
        other_entities = [e for e in entity_groups.keys() if e not in priority_entities]
        if other_entities:
            context_lines.append(f"\nOTHER: {', '.join(other_entities)}")
        
        return '\n'.join(context_lines)
    
    async def _get_intent_from_llm(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """LLM intent analysis"""
        try:
            prompt = f"""Analyze this business question to understand what data is needed.

QUESTION: "{question}"

{context}

Determine the user's intent:
1. What type of query? (ranking, total, count, list)
2. Which entities are needed? (Customer, Payment, Contract, etc.)
3. What should be displayed? (names, amounts, dates)
4. Any limits? (top 10, etc.)
5. Time filters? (this year, 2025, last month)
6. Any filters? (active, approved, paid)

Respond with JSON only:
{{
  "task_type": "ranking|aggregation|count|list",
  "entities": ["Customer", "Payment"],
  "show_fields": ["names", "amounts", "dates"],
  "limit": 10,
  "time_filter": "2025|this_year|last_month|null",
  "filters": {{"status": "approved"}},
  "search_terms": ["top", "customers"]
}}"""

            messages = [
                SystemMessage(content="Analyze questions and respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return parse_json_response(response.content)
            
        except Exception as e:
            print(f"      ⚠️ LLM intent analysis failed: {e}")
            return None
    
    def _pattern_fallback(self, question: str) -> Dict[str, Any]:
        """Pattern-based intent analysis"""
        q_lower = question.lower()
        
        # Task type
        task_type = 'list'
        limit = None
        
        if any(word in q_lower for word in ['top', 'highest', 'best']):
            task_type = 'ranking'
            import re
            numbers = re.findall(r'top\s*(\d+)|(\d+)\s*(?:top|best)', q_lower)
            if numbers:
                limit = int(numbers[0][0] or numbers[0][1])
            else:
                limit = 10
        elif any(word in q_lower for word in ['total', 'sum', 'revenue']):
            task_type = 'aggregation'
        elif any(word in q_lower for word in ['count', 'how many']):
            task_type = 'count'
        
        # Entity detection
        entities = []
        entity_patterns = {
            'Customer': ['customer', 'client', 'account'],
            'Payment': ['payment', 'paid', 'revenue'],
            'Contract': ['contract', 'agreement', 'συμβόλαια'],
            'Order': ['order', 'purchase', 'sale'],
            'Employee': ['employee', 'staff', 'personnel'],
            'Product': ['product', 'item']
        }
        
        for entity_type, keywords in entity_patterns.items():
            if any(keyword in q_lower for keyword in keywords):
                entities.append(entity_type)
        
        # Show fields
        show_fields = []
        if any(word in q_lower for word in ['name', 'who', 'which']):
            show_fields.append('names')
        if any(word in q_lower for word in ['amount', 'revenue', 'total']):
            show_fields.append('amounts')
        if any(word in q_lower for word in ['date', 'when', 'time']):
            show_fields.append('dates')
        
        # Time filter
        time_filter = None
        if '2025' in question:
            time_filter = '2025'
        elif any(phrase in q_lower for phrase in ['this year', 'current year']):
            time_filter = 'this_year'
        elif any(phrase in q_lower for phrase in ['last month', 'previous month']):
            time_filter = 'last_month'
        
        return {
            'task_type': task_type,
            'entities': entities,
            'show_fields': show_fields,
            'limit': limit,
            'time_filter': time_filter,
            'filters': {},
            'search_terms': q_lower.split()[:5]
        }

class TableSelector:
    """Smart table selection"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
    
    def select_tables(self, intent: Dict[str, Any]) -> List[TableInfo]:
        """Select tables based on intent"""
        print("   📋 Table selection...")
        
        target_entities = intent.get('entities', [])
        show_fields = intent.get('show_fields', [])
        
        # Score tables
        scored_tables = []
        for table in self.tables:
            score = self._calculate_score(table, target_entities, show_fields)
            if score > 0:
                scored_tables.append((table, score))
        
        # Sort and select
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        selected = [table for table, score in scored_tables[:3]]
        
        if selected:
            print(f"      ✅ Selected {len(selected)} tables:")
            for i, table in enumerate(selected):
                score = scored_tables[i][1] if i < len(scored_tables) else 0
                print(f"         {i+1}. {table.name} ({table.entity_type}, score: {score:.2f})")
        
        return selected
    
    def _calculate_score(self, table: TableInfo, target_entities: List[str], show_fields: List[str]) -> float:
        """Calculate table score"""
        score = 0.0
        
        # Entity match (highest weight)
        if table.entity_type in target_entities:
            score += 10.0
        
        # Partial entity matching
        for entity in target_entities:
            if entity.lower() in table.entity_type.lower():
                score += 5.0
        
        # Table name matching
        table_name_lower = table.name.lower()
        for entity in target_entities:
            if entity.lower() in table_name_lower:
                score += 3.0
        
        # Business priority
        priority_scores = {'high': 5.0, 'medium': 2.0, 'low': 0.5}
        priority = getattr(table, 'business_priority', 'medium')
        score += priority_scores.get(priority, 2.0)
        
        # Field capabilities
        if 'names' in show_fields and getattr(table, 'name_columns', []):
            score += 3.0
        if 'amounts' in show_fields and getattr(table, 'measures', []):
            score += 3.0
        if 'dates' in show_fields and getattr(table, 'time_columns', []):
            score += 2.0
        
        # Data availability
        if table.row_count > 0:
            score += 2.0
        
        # Penalties
        if any(word in table.name.lower() for word in ['temp', 'test', 'backup']):
            score *= 0.1
        
        return score

class SQLTemplateGenerator:
    """Fixed SQL template generator (Architecture requirement)"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def generate_sql(self, question: str, intent: Dict[str, Any], tables: List[TableInfo]) -> Optional[str]:
        """Generate SQL using fixed templates with LLM fallback"""
        print("   ⚡ SQL generation...")
        
        if not tables:
            return None
        
        # Try fixed templates first
        template_sql = self._generate_template_sql(intent, tables[0])
        if template_sql:
            if self._validate_sql_safety(template_sql):
                print("      ✅ Generated using fixed templates")
                return template_sql
            else:
                print("      ⚠️ Template SQL failed safety validation")
        
        # LLM fallback
        llm_sql = await self._generate_llm_sql(question, intent, tables[0])
        if llm_sql:
            if self._validate_sql_safety(llm_sql):
                print("      ✅ Generated using LLM fallback")
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
        
        # Build SELECT clause
        select_columns = self._build_select_columns(intent, table)
        if not select_columns:
            return None
        
        select_clause = ', '.join(select_columns)
        from_clause = table.full_name
        
        # Build WHERE clause
        where_conditions = []
        
        # Time filters
        time_conditions = self._build_time_conditions(time_filter, table)
        where_conditions.extend(time_conditions)
        
        # Other filters
        filter_conditions = self._build_filter_conditions(filters, table)
        where_conditions.extend(filter_conditions)
        
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        # Generate SQL by task type
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
            
            # Add ORDER BY
            order_column = self._get_best_order_column(table)
            if order_column:
                sql += f" ORDER BY [{order_column}]"
        
        return sql
    
    def _build_select_columns(self, intent: Dict[str, Any], table: TableInfo) -> List[str]:
        """Build SELECT columns"""
        show_fields = intent.get('show_fields', [])
        columns = []
        
        # Name columns
        if 'names' in show_fields or not show_fields:
            name_cols = getattr(table, 'name_columns', [])
            for col in name_cols[:2]:
                columns.append(f"[{col}]")
        
        # Key columns if no names
        if not columns:
            key_cols = getattr(table, 'entity_keys', [])
            for col in key_cols[:1]:
                columns.append(f"[{col}]")
        
        # Measure columns
        if 'amounts' in show_fields:
            measures = getattr(table, 'measures', [])
            for col in measures[:2]:
                columns.append(f"[{col}]")
        
        # Time columns
        if 'dates' in show_fields:
            time_cols = getattr(table, 'time_columns', [])
            for col in time_cols[:1]:
                columns.append(f"[{col}]")
        
        # Fallback: use first few columns
        if not columns:
            for col in table.columns[:3]:
                col_name = col.get('name', '')
                if col_name and not col_name.startswith('__'):
                    columns.append(f"[{col_name}]")
        
        return columns
    
    def _build_time_conditions(self, time_filter: str, table: TableInfo) -> List[str]:
        """Build time-based WHERE conditions"""
        conditions = []
        
        if not time_filter or time_filter == 'null':
            return conditions
        
        # Get time column
        time_columns = getattr(table, 'time_columns', [])
        if not time_columns:
            # Look for date columns
            for col in table.columns:
                col_type = col.get('data_type', '').lower()
                if 'date' in col_type or 'time' in col_type:
                    time_columns = [col.get('name')]
                    break
        
        if not time_columns:
            return conditions
        
        time_col = time_columns[0]
        
        # Time filter implementations
        if time_filter == '2025':
            conditions.append(f"YEAR([{time_col}]) = 2025")
        elif time_filter == 'this_year':
            conditions.append(f"YEAR([{time_col}]) = YEAR(GETDATE())")
        elif time_filter == 'last_month':
            conditions.append(f"[{time_col}] >= DATEADD(month, -1, DATEADD(day, 1-DAY(GETDATE()), GETDATE()))")
            conditions.append(f"[{time_col}] < DATEADD(day, 1-DAY(GETDATE()), GETDATE())")
        
        return conditions
    
    def _build_filter_conditions(self, filters: Dict[str, str], table: TableInfo) -> List[str]:
        """Build filter-based WHERE conditions"""
        conditions = []
        
        for filter_key, filter_value in filters.items():
            for col in table.columns:
                col_name = col.get('name', '').lower()
                if filter_key.lower() in col_name:
                    conditions.append(f"[{col.get('name')}] = '{filter_value}'")
                    break
        
        return conditions
    
    def _get_best_order_column(self, table: TableInfo) -> Optional[str]:
        """Get best column for ORDER BY"""
        # Prefer measures
        measures = getattr(table, 'measures', [])
        if measures:
            return measures[0]
        
        # Then time columns
        time_cols = getattr(table, 'time_columns', [])
        if time_cols:
            return time_cols[0]
        
        # Then keys
        key_cols = getattr(table, 'entity_keys', [])
        if key_cols:
            return key_cols[0]
        
        # First column
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
        
        sql_upper = sql.upper().strip()
        
        # Must start with safe operations
        if not any(sql_upper.startswith(start) for start in ['SELECT', 'WITH']):
            return False
        
        # Dangerous operations
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'SP_', 'XP_'
        ]
        
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {sql_upper} ':
                return False
        
        # sqlglot validation if available
        if HAS_SQLGLOT:
            try:
                parsed = sqlglot.parse_one(sql, dialect="tsql")
                if not parsed:
                    return False
                
                if not isinstance(parsed, sqlglot.expressions.Select):
                    return False
                
                return True
                
            except Exception:
                return False
        
        return True
    
    async def _generate_llm_sql(self, question: str, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """LLM fallback SQL generation"""
        try:
            table_context = self._build_sql_context(table)
            
            prompt = f"""Generate safe T-SQL for this question using the provided table.

QUESTION: "{question}"

INTENT: {intent}

{table_context}

REQUIREMENTS:
- Generate ONLY safe SELECT statements for SQL Server
- Use square brackets: [column_name]
- For rankings: SELECT TOP (N) ... ORDER BY [measure] DESC
- For totals: SELECT SUM([measure]) AS Total FROM [table]
- For counts: SELECT COUNT(*) AS Count FROM [table]
- Use only columns that exist in the table
- NO dangerous operations

Return only the T-SQL query:"""

            messages = [
                SystemMessage(content="Generate safe T-SQL SELECT queries only. Return only SQL code."),
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
        
        lines.append("\nCOLUMNS:")
        for col in table.columns[:12]:
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            
            annotations = []
            if col_name in getattr(table, 'name_columns', []):
                annotations.append('NAME')
            if col_name in getattr(table, 'measures', []):
                annotations.append('MEASURE')
            if col_name in getattr(table, 'entity_keys', []):
                annotations.append('KEY')
            
            ann_str = f" -- {', '.join(annotations)}" if annotations else ""
            lines.append(f"  [{col_name}] ({col_type}){ann_str}")
        
        return '\n'.join(lines)

class QueryExecutor:
    """Query executor with error handling"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_query(self, sql: str):
        """Execute SQL with error handling"""
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
            
            # Error suggestions
            if "Incorrect syntax" in error_msg:
                error_msg += " | Check column names and SQL syntax"
            elif "Invalid column name" in error_msg:
                error_msg += " | Column may not exist"
            elif "Invalid object name" in error_msg:
                error_msg += " | Table may not exist"
            
            return [], error_msg

class QueryInterface:
    """Enhanced Query Interface with fixed templates"""
    
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
        self.intent_analyzer = IntentAnalyzer(self.llm)
        self.sql_generator = SQLTemplateGenerator(self.llm)
        self.executor = QueryExecutor(config)
        
        print("✅ Query Interface initialized")
        print("   🧠 Cross-industry intent analysis")
        print("   ⚡ Fixed SQL templates with LLM fallback")
        print(f"   🛡️ Safety validation: {'✅ sqlglot' if HAS_SQLGLOT else '⚠️ basic only'}")
    
    async def start_session(self, tables: List[TableInfo], domain: Optional[BusinessDomain], relationships: List[Relationship]):
        """Start interactive session"""
        
        # Initialize table selector
        self.table_selector = TableSelector(tables)
        
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
        """Process query using 3-stage pipeline"""
        
        try:
            # Stage 1: Intent analysis
            intent = await self.intent_analyzer.analyze_intent(question, tables)
            
            # Stage 2: Table selection
            selected_tables = self.table_selector.select_tables(intent)
            
            if not selected_tables:
                available_entities = list(set(t.entity_type for t in tables[:10]))
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error=f"No suitable tables found. Available entities: {', '.join(available_entities)}",
                    result_type="error"
                )
            
            # Stage 3: SQL generation with safety validation
            sql = await self.sql_generator.generate_sql(question, intent, selected_tables)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not generate safe SQL. Try rephrasing.",
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
        """Show system readiness"""
        entities = {}
        for table in tables:
            entity = table.entity_type
            entities[entity] = entities.get(entity, 0) + 1
        
        print(f"\n🚀 QUERY SYSTEM READY:")
        print(f"   📊 Total objects: {len(tables)}")
        
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
        priority_entities = ['Customer', 'Payment', 'Contract', 'Order', 'Employee']
        
        for entity, count in sorted_entities[:6]:
            priority_emoji = "🔥" if entity in priority_entities else "📋"
            print(f"   {priority_emoji} {entity}: {count} objects")
        
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        print(f"\n🔄 3-STAGE PIPELINE:")
        print(f"   1. Cross-Industry Intent Analysis")
        print(f"   2. Smart Table Selection")
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
        """Display query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 50)
        
        if result.is_successful() and result.has_results():
            print(f"✅ QUERY SUCCESSFUL")
            print(f"\n📋 Generated SQL ({result.sql_generation_method}):")
            print(f"{result.sql_query}")
            
            if result.tables_used:
                table_names = [t.split('.')[-1].replace(']', '') for t in result.tables_used]
                print(f"\n📊 Tables Used: {', '.join(table_names)}")
            
            print(f"\n📈 Results ({len(result.results)} rows):")
            self._display_data(result.results)
            
        else:
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
            
            print(f"\n💡 Suggestions:")
            print(f"   • Try specific entity names: 'customers', 'payments', 'contracts'")
            print(f"   • Use simple questions: 'show customer names', 'total revenue'")
    
    def _display_data(self, results: List[Dict[str, Any]]):
        """Display query results"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            key, value = next(iter(results[0].items()))
            if isinstance(value, (int, float)) and abs(value) >= 1000:
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            print(f"   🎯 {key}: {formatted_value}")
            
        else:
            # Multiple rows
            for i, row in enumerate(results[:15], 1):
                parts = []
                for j, (key, value) in enumerate(row.items()):
                    if j >= 4:  # Limit columns
                        break
                    
                    # Format value
                    if isinstance(value, (int, float)) and abs(value) >= 1000:
                        formatted_value = f"{value:,}"
                    elif isinstance(value, str) and len(value) > 30:
                        formatted_value = value[:30] + "..."
                    else:
                        formatted_value = str(value) if value is not None else ""
                    
                    # Format key
                    display_key = key.replace('_', ' ').title() if '_' in key else key
                    parts.append(f"{display_key}: {formatted_value}")
                
                print(f"   {i:2d}. {' | '.join(parts)}")
            
            if len(results) > 15:
                print(f"   ... and {len(results) - 15} more rows")