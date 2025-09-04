#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - Fixed SQL Generation & Enhanced Table Selection
Simple, Readable, Maintainable - Following DRY, SOLID, YAGNI
"""

import asyncio
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import parse_json_response, clean_sql_query, safe_database_value, validate_sql_safety

class IntentAnalyzer:
    """Stage 1: Enhanced intent analysis"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str, tables: List[TableInfo]) -> Dict[str, Any]:
        """Enhanced intent analysis with business context"""
        print("   🧠 Stage 1: Enhanced intent analysis...")
        
        # Build enhanced table context
        table_context = self._build_enhanced_context(tables)
        
        # Get LLM analysis
        intent = await self._get_intent_from_llm(question, table_context)
        
        if intent:
            print(f"      ✅ Intent: {intent.get('task_type', 'unknown')}")
            print(f"      🎯 Entities: {', '.join(intent.get('entities', []))}")
            return intent
        
        # Enhanced fallback
        return self._enhanced_fallback(question)
    
    def _build_enhanced_context(self, tables: List[TableInfo]) -> str:
        """Build enhanced context with entity types and capabilities"""
        context_lines = ["AVAILABLE DATA SOURCES:"]
        
        # Group by entity type
        entity_groups = {}
        for table in tables[:20]:  # Limit for token efficiency
            entity_type = table.entity_type
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(table)
        
        # Show entities with their capabilities
        for entity_type, entity_tables in entity_groups.items():
            context_lines.append(f"\n{entity_type.upper()} DATA:")
            
            for table in entity_tables[:3]:  # Top 3 per entity
                capabilities = []
                
                if getattr(table, 'name_columns', []):
                    capabilities.append(f"Names: {', '.join(table.name_columns[:2])}")
                
                if getattr(table, 'measures', []):
                    capabilities.append(f"Amounts: {', '.join(table.measures[:2])}")
                    
                if getattr(table, 'time_columns', []):
                    capabilities.append(f"Dates: {', '.join(table.time_columns[:1])}")
                
                cap_str = f" ({'; '.join(capabilities)})" if capabilities else ""
                context_lines.append(f"  - {table.full_name}{cap_str}")
        
        return '\n'.join(context_lines)
    
    async def _get_intent_from_llm(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """Enhanced LLM intent analysis"""
        try:
            prompt = f"""Analyze this business question and determine what data is needed.

QUESTION: "{question}"

{context}

Based on the available data, determine:
1. What is the user asking for? (ranking, total, count, list, analysis)
2. Which entity types are needed? (Customer, Payment, Order, Address, etc.)
3. What should be displayed? (names, amounts, dates, counts)
4. Any specific limits? (top 10, etc.)
5. Time period? (this year, 2025, last month, etc.)

Respond with JSON only:
{{
  "task_type": "ranking|aggregation|count|list|analysis",
  "entities": ["Customer", "Payment"],
  "show_fields": ["names", "amounts", "dates"],
  "limit": 10,
  "time_filter": "2025|this_year|last_month|null",
  "search_terms": ["paid", "customers", "top"]
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
    
    def _enhanced_fallback(self, question: str) -> Dict[str, Any]:
        """Enhanced pattern-based intent analysis"""
        q_lower = question.lower()
        
        # Determine task type
        task_type = 'list'
        limit = None
        
        if any(word in q_lower for word in ['top', 'highest', 'best', 'largest']):
            task_type = 'ranking'
            # Extract number
            import re
            numbers = re.findall(r'\b(\d+)\b', question)
            limit = int(numbers[0]) if numbers else 10
        elif any(word in q_lower for word in ['total', 'sum', 'revenue']):
            task_type = 'aggregation'
        elif any(word in q_lower for word in ['count', 'how many', 'number of']):
            task_type = 'count'
        
        # Enhanced entity detection
        entities = []
        search_terms = []
        
        if any(word in q_lower for word in ['customer', 'client', 'account']):
            entities.append('Customer')
            search_terms.extend(['customer', 'client'])
            
        if any(word in q_lower for word in ['payment', 'paid', 'revenue', 'sales', 'transaction']):
            entities.append('Payment')
            search_terms.extend(['payment', 'paid', 'revenue'])
            
        if any(word in q_lower for word in ['order', 'purchase', 'sale']):
            entities.append('Order')
            search_terms.extend(['order', 'purchase'])
        
        # Determine what to show
        show_fields = []
        if any(word in q_lower for word in ['name', 'who', 'which']):
            show_fields.append('names')
        if any(word in q_lower for word in ['amount', 'revenue', 'total', 'paid', 'value']):
            show_fields.append('amounts')
        if any(word in q_lower for word in ['date', 'when', 'time']):
            show_fields.append('dates')
        
        # Time filter detection
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
            'search_terms': search_terms
        }

class EnhancedTableSelector:
    """Stage 2: Enhanced table selection with better scoring"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
    
    def select_tables(self, intent: Dict[str, Any]) -> List[TableInfo]:
        """Enhanced table selection with detailed scoring"""
        print("   📋 Stage 2: Enhanced table selection...")
        
        target_entities = intent.get('entities', [])
        show_fields = intent.get('show_fields', [])
        search_terms = intent.get('search_terms', [])
        
        # Score all tables
        scored_tables = []
        for table in self.tables:
            score = self._calculate_enhanced_score(table, target_entities, show_fields, search_terms)
            if score > 0:
                scored_tables.append((table, score))
                print(f"      📊 {table.name}: {score:.2f} ({table.entity_type})")
        
        # Sort by score and select top candidates
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        selected = [table for table, score in scored_tables[:5]]  # Top 5
        
        if selected:
            print(f"      ✅ Selected {len(selected)} tables:")
            for i, table in enumerate(selected):
                capabilities = []
                if getattr(table, 'name_columns', []):
                    capabilities.append(f"Names: {len(table.name_columns)}")
                if getattr(table, 'measures', []):
                    capabilities.append(f"Measures: {len(table.measures)}")
                cap_str = f" ({', '.join(capabilities)})" if capabilities else ""
                print(f"         {i+1}. {table.name} ({table.entity_type}){cap_str}")
        else:
            print("      ❌ No suitable tables found")
        
        return selected
    
    def _calculate_enhanced_score(self, table: TableInfo, target_entities: List[str], 
                                show_fields: List[str], search_terms: List[str]) -> float:
        """Enhanced scoring with multiple factors"""
        score = 0.0
        
        # Entity type match (highest priority)
        if table.entity_type in target_entities:
            score += 5.0
        
        # Partial entity match
        for entity in target_entities:
            if entity.lower() in table.entity_type.lower():
                score += 3.0
        
        # Search term matching in table name
        table_name_lower = table.name.lower()
        for term in search_terms:
            if term.lower() in table_name_lower:
                score += 2.0
        
        # Business priority bonus
        priority_bonus = {
            'high': 3.0,
            'medium': 1.5,
            'low': 0.5
        }
        score += priority_bonus.get(getattr(table, 'business_priority', 'medium'), 1.5)
        
        # Field capability matching
        if 'names' in show_fields and getattr(table, 'name_columns', []):
            score += 2.0
        if 'amounts' in show_fields and getattr(table, 'measures', []):
            score += 2.0
        if 'dates' in show_fields and getattr(table, 'time_columns', []):
            score += 1.0
        
        # Data availability
        if table.row_count > 0:
            score += 1.0
        if table.row_count > 100:
            score += 0.5
        
        # BI role preference (facts for aggregations)
        bi_role = getattr(table, 'bi_role', 'dimension')
        if bi_role == 'fact' and any(field in show_fields for field in ['amounts']):
            score += 1.5
        
        # Avoid system/temp tables
        if any(word in table.name.lower() for word in ['temp', 'test', 'backup', 'sys', 'log']):
            score *= 0.1
        
        # Object type preference (tables over views for simple queries)
        if table.object_type in ['USER_TABLE', 'TABLE', 'BASE TABLE']:
            score += 0.5
        
        return score

class FixedSQLGenerator:
    """Stage 3: Fixed SQL generation with proper templates"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def generate_sql(self, question: str, intent: Dict[str, Any], tables: List[TableInfo]) -> Optional[str]:
        """Generate SQL with fixed templates and LLM backup"""
        print("   ⚡ Stage 3: Fixed SQL generation...")
        
        if not tables:
            return None
        
        # Try template-based generation first (more reliable)
        template_sql = self._generate_template_sql(intent, tables[0])
        if template_sql and validate_sql_safety(template_sql):
            print("      ✅ SQL generated using fixed templates")
            return template_sql
        
        # Fallback to LLM generation
        llm_sql = await self._generate_llm_sql(question, intent, tables[0])
        if llm_sql and validate_sql_safety(llm_sql):
            print("      ✅ SQL generated using LLM")
            return llm_sql
        
        print("      ❌ SQL generation failed")
        return None
    
    def _generate_template_sql(self, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """Generate SQL using fixed templates"""
        task_type = intent.get('task_type', 'list')
        limit = intent.get('limit')
        time_filter = intent.get('time_filter')
        
        # Build SELECT clause
        select_columns = self._build_select_columns(intent, table)
        if not select_columns:
            return None
        
        # Base SQL
        select_clause = ', '.join(select_columns)
        from_clause = table.full_name
        
        # Build WHERE clause
        where_conditions = self._build_where_conditions(time_filter, table)
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        # Build complete SQL based on task type
        if task_type == 'ranking' and limit:
            # Fixed ranking query
            order_column = self._get_order_column(table)
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
            # Fixed aggregation query
            measure_column = self._get_measure_column(table)
            if measure_column:
                sql = f"SELECT SUM([{measure_column}]) AS Total FROM {from_clause}"
                if where_clause:
                    sql += f" {where_clause}"
            else:
                sql = f"SELECT COUNT(*) AS Count FROM {from_clause}"
                if where_clause:
                    sql += f" {where_clause}"
                    
        elif task_type == 'count':
            # Fixed count query
            sql = f"SELECT COUNT(*) AS Count FROM {from_clause}"
            if where_clause:
                sql += f" {where_clause}"
                
        else:
            # Fixed list query
            top_clause = f"TOP ({limit})" if limit else "TOP (50)"
            sql = f"SELECT {top_clause} {select_clause} FROM {from_clause}"
            if where_clause:
                sql += f" {where_clause}"
        
        return sql
    
    def _build_select_columns(self, intent: Dict[str, Any], table: TableInfo) -> List[str]:
        """Build SELECT columns based on intent and available columns"""
        show_fields = intent.get('show_fields', [])
        columns = []
        
        # Add name columns if requested or available
        if 'names' in show_fields or not show_fields:
            name_cols = getattr(table, 'name_columns', [])
            for col in name_cols[:2]:  # First 2 name columns
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
        
        # Fallback: use first few actual columns
        if not columns:
            for col in table.columns[:3]:
                col_name = col.get('name', '')
                if col_name:
                    columns.append(f"[{col_name}]")
        
        return columns
    
    def _build_where_conditions(self, time_filter: str, table: TableInfo) -> List[str]:
        """Build WHERE conditions"""
        conditions = []
        
        if time_filter and time_filter != 'null':
            time_columns = getattr(table, 'time_columns', [])
            if time_columns:
                time_col = time_columns[0]  # Use first time column
                
                if time_filter == '2025':
                    conditions.append(f"YEAR([{time_col}]) = 2025")
                elif time_filter == 'this_year':
                    conditions.append(f"YEAR([{time_col}]) = YEAR(GETDATE())")
                elif time_filter == 'last_month':
                    conditions.append(f"[{time_col}] >= DATEADD(month, -1, DATEADD(day, 1-DAY(GETDATE()), GETDATE()))")
                    conditions.append(f"[{time_col}] < DATEADD(day, 1-DAY(GETDATE()), GETDATE())")
        
        return conditions
    
    def _get_order_column(self, table: TableInfo) -> Optional[str]:
        """Get column for ORDER BY in ranking queries"""
        # Try measures first
        measures = getattr(table, 'measures', [])
        if measures:
            return measures[0]
        
        # Try numeric columns
        for col in table.columns:
            col_type = col.get('data_type', '').lower()
            if any(t in col_type for t in ['int', 'decimal', 'float', 'money', 'numeric']):
                return col.get('name')
        
        # Fallback to first column
        if table.columns:
            return table.columns[0].get('name')
        
        return None
    
    def _get_measure_column(self, table: TableInfo) -> Optional[str]:
        """Get measure column for aggregation"""
        measures = getattr(table, 'measures', [])
        return measures[0] if measures else None
    
    async def _generate_llm_sql(self, question: str, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """Fallback LLM SQL generation"""
        try:
            table_context = self._build_sql_context(table)
            
            prompt = f"""Generate SQL for this question using the provided table.

QUESTION: "{question}"

INTENT:
- Task: {intent.get('task_type', 'list')}
- Show: {intent.get('show_fields', [])}
- Limit: {intent.get('limit', '')}
- Time: {intent.get('time_filter', '')}

{table_context}

Generate simple, working SQL:
- Use only columns that exist in the table
- For rankings: SELECT TOP (N) columns FROM table ORDER BY measure_column DESC
- For totals: SELECT SUM(measure_column) AS Total FROM table
- For counts: SELECT COUNT(*) AS Count FROM table
- Always use square brackets around column names: [column_name]
- Use proper WHERE clause syntax

Return only the SQL query, no explanations:"""

            messages = [
                SystemMessage(content="Generate simple, correct SQL queries. Use proper SQL Server syntax with square brackets around column names. Return only SQL code."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return clean_sql_query(response.content)
            
        except Exception as e:
            print(f"      ⚠️ LLM SQL generation failed: {e}")
            return None
    
    def _build_sql_context(self, table: TableInfo) -> str:
        """Build context for SQL generation"""
        lines = [f"TABLE: {table.full_name}"]
        
        # Show columns with types and annotations
        lines.append("COLUMNS:")
        for col in table.columns[:10]:
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            
            # Annotations
            annotations = []
            if col_name in getattr(table, 'name_columns', []):
                annotations.append('NAME')
            if col_name in getattr(table, 'measures', []):
                annotations.append('MEASURE')
            if col_name in getattr(table, 'entity_keys', []):
                annotations.append('KEY')
            if col_name in getattr(table, 'time_columns', []):
                annotations.append('DATE')
            
            ann_str = f" [{', '.join(annotations)}]" if annotations else ""
            lines.append(f"  - [{col_name}] ({col_type}){ann_str}")
        
        # Show sample data
        if table.sample_data:
            lines.append("\nSAMPLE DATA:")
            first_row = table.sample_data[0]
            for key, value in list(first_row.items())[:4]:
                if not key.startswith('__'):
                    lines.append(f"  [{key}]: {value}")
        
        return '\n'.join(lines)

class QueryExecutor:
    """Execute SQL queries with enhanced error handling"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute SQL with enhanced error handling"""
        print("   🔄 Executing query...")
        print(f"      📝 SQL: {sql}")
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                # UTF-8 support
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
                    
                    print(f"      ✅ Query executed: {len(results)} rows returned")
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Query failed: {error_msg}")
            
            # Enhanced error reporting
            if "Incorrect syntax" in error_msg:
                error_msg += " | SQL syntax error - check column names and WHERE clause"
            elif "Invalid column name" in error_msg:
                error_msg += " | Column does not exist in table"
            elif "Cannot resolve" in error_msg:
                error_msg += " | Table or column reference issue"
            
            return [], error_msg

class QueryInterface:
    """Enhanced query interface with fixed SQL generation"""
    
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
        
        # Initialize enhanced pipeline components
        self.intent_analyzer = IntentAnalyzer(self.llm)
        self.sql_generator = FixedSQLGenerator(self.llm)
        self.executor = QueryExecutor(config)
        
        print("✅ Enhanced Query Interface initialized")
        print("   🧠 Enhanced intent analysis")
        print("   📋 Smart table selection") 
        print("   ⚡ Fixed SQL generation")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start enhanced interactive session"""
        
        # Initialize enhanced table selector
        self.table_selector = EnhancedTableSelector(tables)
        
        # Show system readiness
        self._show_enhanced_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🔄 Processing with enhanced 3-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_query(question, tables)
                result.execution_time = time.time() - start_time
                
                self._display_enhanced_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_query(self, question: str, tables: List[TableInfo]) -> QueryResult:
        """Process query using enhanced pipeline"""
        
        try:
            # Stage 1: Enhanced intent analysis
            intent = await self.intent_analyzer.analyze_intent(question, tables)
            
            # Stage 2: Enhanced table selection
            selected_tables = self.table_selector.select_tables(intent)
            
            if not selected_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No suitable tables found for this query. Available entities: " + 
                          ", ".join(set(t.entity_type for t in tables[:10])),
                    result_type="error"
                )
            
            # Stage 3: Fixed SQL generation
            sql = await self.sql_generator.generate_sql(question, intent, selected_tables)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not generate SQL for this query. Try rephrasing.",
                    result_type="error"
                )
            
            # Execute SQL
            results, error = await self.executor.execute_query(sql)
            
            return QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=[t.full_name for t in selected_tables],
                result_type="data" if results and not error else "error"
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Query processing error: {str(e)}",
                result_type="error"
            )
    
    def _show_enhanced_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show enhanced system readiness"""
        # Entity analysis
        entities = {}
        for table in tables:
            entity = table.entity_type
            entities[entity] = entities.get(entity, 0) + 1
        
        print(f"\n🚀 ENHANCED BI SYSTEM READY:")
        print(f"   📊 Total objects: {len(tables)}")
        
        # Show top entities
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
        for entity, count in sorted_entities[:5]:
            print(f"   🎯 {entity}: {count} objects")
        
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
            print(f"   💡 Capabilities: {', '.join(domain.capabilities.keys())}")
        
        print(f"\n🔄 ENHANCED PIPELINE:")
        print(f"   1. Enhanced Intent Analysis (Business context)")
        print(f"   2. Smart Table Selection (Multi-factor scoring)")
        print(f"   3. Fixed SQL Generation (Templates + LLM)")
    
    def _display_enhanced_result(self, result: QueryResult):
        """Display enhanced query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.is_successful() and result.has_results():
            print(f"✅ QUERY SUCCESSFUL")
            print(f"\n📋 SQL Generated:")
            print(f"{result.sql_query}")
            
            if result.tables_used:
                print(f"\n📊 Tables Used: {', '.join(result.tables_used)}")
            
            print(f"\n📈 Results ({len(result.results)} rows):")
            self._display_enhanced_data(result.results)
            
        else:
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
            
            # Enhanced suggestions
            if result.tables_used:
                print(f"   Tables attempted: {', '.join(result.tables_used)}")
            
            print(f"\n💡 Suggestions:")
            print(f"   • Try simpler questions like 'show customer names'")
            print(f"   • Check available entities and their data")
            print(f"   • Use specific terms like 'customers', 'payments', 'orders'")
    
    def _display_enhanced_data(self, results: List[Dict[str, Any]]):
        """Display query results with enhanced formatting"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            key, value = next(iter(results[0].items()))
            if isinstance(value, (int, float)) and abs(value) >= 1000:
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            print(f"   🎯 {key}: {formatted_value}")
            
        else:
            # Multiple rows with enhanced formatting
            headers = list(results[0].keys()) if results else []
            
            for i, row in enumerate(results[:15], 1):  # Show first 15 rows
                parts = []
                for j, (key, value) in enumerate(row.items()):
                    if j >= 4:  # Limit to 4 columns
                        break
                        
                    # Format value
                    if isinstance(value, (int, float)) and abs(value) >= 1000:
                        formatted_value = f"{value:,}"
                    elif isinstance(value, str) and len(value) > 35:
                        formatted_value = value[:35] + "..."
                    else:
                        formatted_value = str(value) if value is not None else ""
                    
                    parts.append(f"{key}: {formatted_value}")
                
                print(f"   {i:2d}. {' | '.join(parts)}")
            
            if len(results) > 15:
                print(f"   ... and {len(results) - 15} more rows")