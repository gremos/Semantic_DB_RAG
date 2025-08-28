#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - Simplified 3-Stage Pipeline
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
    """Stage 1: Understand user intent - Single Responsibility"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str, tables: List[TableInfo]) -> Dict[str, Any]:
        """Analyze user intent and identify relevant entities"""
        print("   🧠 Stage 1: Intent analysis...")
        
        # Create table context for LLM
        table_context = self._build_table_context(tables)
        
        # Get LLM analysis
        intent = await self._get_intent_from_llm(question, table_context)
        
        if intent:
            print(f"      ✅ Intent: {intent.get('task_type', 'unknown')}")
            return intent
        
        # Fallback to simple pattern matching
        return self._simple_intent_fallback(question)
    
    def _build_table_context(self, tables: List[TableInfo]) -> str:
        """Build table context for LLM"""
        context_lines = ["AVAILABLE TABLES:"]
        
        # Show most relevant tables first (customers, payments)
        priority_tables = []
        other_tables = []
        
        for table in tables[:15]:  # Limit to avoid token limits
            table_info = f"- {table.full_name}: {table.entity_type}"
            if getattr(table, 'name_columns', []):
                table_info += f" (Names: {', '.join(table.name_columns[:2])})"
            if getattr(table, 'measures', []):
                table_info += f" (Amounts: {', '.join(table.measures[:2])})"
            
            if table.entity_type in ['Customer', 'Payment']:
                priority_tables.append(table_info)
            else:
                other_tables.append(table_info)
        
        context_lines.extend(priority_tables)
        context_lines.extend(other_tables[:8])  # Limit other tables
        
        return '\n'.join(context_lines)
    
    async def _get_intent_from_llm(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """Get intent analysis from LLM"""
        try:
            prompt = f"""Analyze this business question and map to available data:

QUESTION: "{question}"

{context}

Identify:
1. What the user wants (ranking, total, count, list)
2. Which entity types are needed (Customer, Payment, Order, etc.)
3. What should be shown (names, amounts, dates)
4. Any limits (top 10, etc.)

Respond with JSON only:
{{
  "task_type": "ranking|aggregation|count|list",
  "entities": ["Customer", "Payment"],
  "show_fields": ["names", "amounts"],
  "limit": 10,
  "time_filter": "this_year|last_month|null"
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
    
    def _simple_intent_fallback(self, question: str) -> Dict[str, Any]:
        """Simple pattern-based intent analysis as fallback"""
        q_lower = question.lower()
        
        # Determine task type
        if any(word in q_lower for word in ['top', 'highest', 'best']):
            task_type = 'ranking'
            limit = 10
        elif any(word in q_lower for word in ['total', 'sum']):
            task_type = 'aggregation'
            limit = None
        elif any(word in q_lower for word in ['count', 'how many']):
            task_type = 'count'
            limit = None
        else:
            task_type = 'list'
            limit = 20
        
        # Detect entities
        entities = []
        if any(word in q_lower for word in ['customer', 'client']):
            entities.append('Customer')
        if any(word in q_lower for word in ['payment', 'revenue', 'sales', 'paid']):
            entities.append('Payment')
        
        # Detect what to show
        show_fields = []
        if 'name' in q_lower:
            show_fields.append('names')
        if any(word in q_lower for word in ['amount', 'revenue', 'total', 'paid']):
            show_fields.append('amounts')
        
        return {
            'task_type': task_type,
            'entities': entities,
            'show_fields': show_fields,
            'limit': limit,
            'time_filter': None
        }

class TableSelector:
    """Stage 2: Select best tables for query - Single Responsibility"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
    
    def select_tables(self, intent: Dict[str, Any]) -> List[TableInfo]:
        """Select best tables based on intent"""
        print("   📋 Stage 2: Table selection...")
        
        target_entities = intent.get('entities', [])
        show_fields = intent.get('show_fields', [])
        
        # Score tables based on intent
        scored_tables = []
        for table in self.tables:
            score = self._calculate_table_score(table, target_entities, show_fields)
            if score > 0:
                scored_tables.append((table, score))
        
        # Sort by score and select top candidates
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        selected = [table for table, score in scored_tables[:3]]
        
        if selected:
            print(f"      ✅ Selected {len(selected)} tables")
            for i, table in enumerate(selected):
                print(f"         {i+1}. {table.name} ({table.entity_type})")
        else:
            print("      ❌ No suitable tables found")
        
        return selected
    
    def _calculate_table_score(self, table: TableInfo, target_entities: List[str], show_fields: List[str]) -> float:
        """Calculate table relevance score"""
        score = 0.0
        
        # Entity type match (highest priority)
        if table.entity_type in target_entities:
            score += 3.0
        
        # Business priority
        priority_bonus = {
            'high': 1.0,
            'medium': 0.5,
            'low': 0.0
        }
        score += priority_bonus.get(getattr(table, 'business_priority', 'medium'), 0.5)
        
        # Check for required fields
        if 'names' in show_fields and getattr(table, 'name_columns', []):
            score += 1.0
        if 'amounts' in show_fields and getattr(table, 'measures', []):
            score += 1.0
        
        # Data availability
        if table.row_count > 0:
            score += 0.5
        
        # Avoid system/temp tables
        if any(word in table.name.lower() for word in ['temp', 'test', 'backup', 'sys']):
            score = 0.0
        
        return score

class SQLGenerator:
    """Stage 3: Generate SQL from intent and tables - Single Responsibility"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def generate_sql(self, question: str, intent: Dict[str, Any], tables: List[TableInfo]) -> Optional[str]:
        """Generate SQL based on intent and selected tables"""
        print("   ⚡ Stage 3: SQL generation...")
        
        if not tables:
            return None
        
        # Use primary table (best scored)
        primary_table = tables[0]
        
        # Create SQL generation prompt
        sql = await self._generate_sql_with_llm(question, intent, primary_table)
        
        if sql and validate_sql_safety(sql):
            print("      ✅ SQL generated and validated")
            return sql
        
        # Fallback: Simple template-based SQL
        return self._generate_simple_sql(intent, primary_table)
    
    async def _generate_sql_with_llm(self, question: str, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """Generate SQL using LLM"""
        try:
            # Build context about the table
            table_context = self._build_sql_context(table)
            
            prompt = f"""Generate SQL for this question using the provided table.

QUESTION: "{question}"

INTENT:
- Task: {intent.get('task_type', 'list')}
- Show: {intent.get('show_fields', [])}
- Limit: {intent.get('limit', '')}

{table_context}

Generate simple, working SQL:
- Use only columns that exist
- For rankings: SELECT TOP ({intent.get('limit', 10)}) ... ORDER BY amount DESC
- For totals: SELECT SUM(amount) FROM table
- For counts: SELECT COUNT(*) FROM table
- Include name columns when possible

Return only the SQL query:"""

            messages = [
                SystemMessage(content="You are a SQL expert. Generate simple, correct SQL queries. Return only SQL code."),
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
        
        # Show relevant columns
        lines.append("COLUMNS:")
        for col in table.columns[:12]:  # Show first 12 columns
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            
            # Highlight important columns
            notes = []
            if col_name in getattr(table, 'name_columns', []):
                notes.append('NAME')
            if col_name in getattr(table, 'measures', []):
                notes.append('AMOUNT')
            if col_name in getattr(table, 'time_columns', []):
                notes.append('DATE')
            
            note_str = f" [{', '.join(notes)}]" if notes else ""
            lines.append(f"  - {col_name} ({col_type}){note_str}")
        
        # Show sample data
        if table.sample_data:
            lines.append("\nSAMPLE DATA:")
            first_row = table.sample_data[0]
            for key, value in list(first_row.items())[:4]:
                if not key.startswith('__'):
                    lines.append(f"  {key}: {value}")
        
        return '\n'.join(lines)
    
    def _generate_simple_sql(self, intent: Dict[str, Any], table: TableInfo) -> Optional[str]:
        """Generate simple SQL using templates"""
        task_type = intent.get('task_type', 'list')
        limit = intent.get('limit')
        
        # Build SELECT clause
        select_parts = []
        
        # Add name columns if available
        name_columns = getattr(table, 'name_columns', [])
        if name_columns:
            select_parts.extend(name_columns[:2])  # First 2 name columns
        
        # Add measure columns if requested
        if 'amounts' in intent.get('show_fields', []):
            measures = getattr(table, 'measures', [])
            if measures:
                select_parts.extend(measures[:2])  # First 2 measure columns
        
        # Fallback: use first few columns
        if not select_parts:
            for col in table.columns[:3]:
                select_parts.append(col.get('name', ''))
        
        # Generate SQL based on task type
        if task_type == 'ranking' and limit:
            sql = f"SELECT TOP ({limit}) {', '.join(select_parts)} FROM {table.full_name}"
            if getattr(table, 'measures', []):
                order_col = getattr(table, 'measures', [])[0]
                sql += f" ORDER BY {order_col} DESC"
        elif task_type == 'aggregation':
            measures = getattr(table, 'measures', [])
            if measures:
                sql = f"SELECT SUM({measures[0]}) AS Total FROM {table.full_name}"
            else:
                sql = f"SELECT COUNT(*) AS Count FROM {table.full_name}"
        elif task_type == 'count':
            sql = f"SELECT COUNT(*) AS Count FROM {table.full_name}"
        else:
            # List query
            top_clause = f"TOP ({limit})" if limit else "TOP (50)"
            sql = f"SELECT {top_clause} {', '.join(select_parts)} FROM {table.full_name}"
        
        return sql if sql else None

class QueryExecutor:
    """Execute SQL queries - Single Responsibility"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute SQL and return results"""
        print("   🔄 Executing query...")
        
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
                    
                    print(f"      ✅ Query executed: {len(results)} rows")
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Query failed: {error_msg}")
            return [], error_msg

class QueryInterface:
    """Main query interface - Simplified 3-stage pipeline"""
    
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
        
        # Initialize pipeline components
        self.intent_analyzer = IntentAnalyzer(self.llm)
        self.sql_generator = SQLGenerator(self.llm)
        self.executor = QueryExecutor(config)
        
        print("✅ Simplified Query Interface initialized")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start interactive query session"""
        
        # Initialize table selector
        self.table_selector = TableSelector(tables)
        
        # Show system readiness
        self._show_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🔄 Processing with simplified 3-stage pipeline...")
                
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
        """Process query using simplified 3-stage pipeline"""
        
        try:
            # Stage 1: Analyze intent
            intent = await self.intent_analyzer.analyze_intent(question, tables)
            
            # Stage 2: Select tables
            selected_tables = self.table_selector.select_tables(intent)
            
            if not selected_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No suitable tables found for this query",
                    result_type="error"
                )
            
            # Stage 3: Generate and execute SQL
            sql = await self.sql_generator.generate_sql(question, intent, selected_tables)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not generate SQL for this query",
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
                result_type="data" if results else "error"
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Query processing error: {str(e)}",
                result_type="error"
            )
    
    def _show_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show system readiness"""
        customer_tables = len([t for t in tables if t.entity_type == 'Customer'])
        payment_tables = len([t for t in tables if t.entity_type == 'Payment'])
        
        print(f"\n🚀 SIMPLIFIED BI SYSTEM READY:")
        print(f"   📊 Total tables: {len(tables)}")
        print(f"   👥 Customer tables: {customer_tables}")
        print(f"   💳 Payment tables: {payment_tables}")
        
        if domain:
            print(f"   🎯 Domain: {domain.domain_type}")
        
        print(f"\n🔄 SIMPLIFIED PIPELINE:")
        print(f"   1. Intent Analysis (LLM)")
        print(f"   2. Table Selection (Scoring)")
        print(f"   3. SQL Generation (LLM)")
    
    def _display_result(self, result: QueryResult):
        """Display query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 50)
        
        if result.is_successful() and result.has_results():
            print(f"✅ QUERY SUCCESSFUL")
            print(f"\n📋 SQL:")
            print(f"{result.sql_query}")
            
            print(f"\n📊 Results ({len(result.results)} rows):")
            self._display_data(result.results)
        else:
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
    
    def _display_data(self, results: List[Dict[str, Any]]):
        """Display query results in a readable format"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            key, value = next(iter(results[0].items()))
            formatted_value = f"{value:,}" if isinstance(value, (int, float)) and abs(value) >= 1000 else str(value)
            print(f"   🎯 {key}: {formatted_value}")
        else:
            # Multiple rows - show first 10
            for i, row in enumerate(results[:10], 1):
                parts = []
                for key, value in list(row.items())[:3]:  # First 3 columns
                    if isinstance(value, (int, float)) and abs(value) >= 1000:
                        value = f"{value:,}"
                    elif isinstance(value, str) and len(value) > 30:
                        value = value[:30] + "..."
                    parts.append(f"{key}: {value}")
                
                print(f"   {i:2d}. {' | '.join(parts)}")
            
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more rows")