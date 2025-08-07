#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple 4-Stage Query Pipeline - Clean and Maintainable
Uses LLM + actual sample data for accurate table identification
"""

import json
import pyodbc
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult

class LLMClient:
    """Simple LLM client for all analysis"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,
            request_timeout=60
        )
    
    async def ask(self, system_prompt: str, user_prompt: str) -> str:
        """Simple LLM query"""
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

class QueryInterface:
    """Simple 4-Stage Query Pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.classified_tables = []  # Will store semantic analysis results
        self.domain = None
        self.relationships = []
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship]):
        """Start interactive query session"""
        
        # Use the semantic analysis results (already classified tables)
        self.classified_tables = tables  # These have entity_type already set
        self.domain = domain
        self.relationships = relationships
        
        print(f"🚀 4-Stage Pipeline Ready with {len(tables)} classified tables")
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        # Show what entities we have
        entity_counts = {}
        for table in tables:
            if hasattr(table, 'entity_type') and table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   📊 Available entities: {dict(list(entity_counts.items())[:5])}")  # Show top 5
        
        query_count = 0
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif not question:
                    continue
                
                query_count += 1
                print(f"🚀 Processing with 4-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_4_stage_pipeline(question)
                result.execution_time = time.time() - start_time
                
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Session summary: {query_count} queries processed")
    
    async def process_4_stage_pipeline(self, question: str) -> QueryResult:
        """Execute 4-stage pipeline"""
        
        try:
            # Stage 1: Understand Intent
            print("   🎯 Stage 1: Understanding intent...")
            intent = await self.analyze_intent(question)
            if not intent:
                return QueryResult(question=question, sql_query="", results=[], 
                                 error="Failed to understand question")
            
            # Stage 2: Find Relevant Tables Using Real Data
            print("   📋 Stage 2: Finding tables with real data...")
            selected_tables = await self.select_tables_with_data(question, intent)
            if not selected_tables:
                return QueryResult(question=question, sql_query="", results=[],
                                 error="No relevant tables found")
            
            print(f"      ✅ Found {len(selected_tables)} relevant tables")
            
            # Stage 3: Determine Relationships
            print("   🔗 Stage 3: Analyzing relationships...")
            relationships = await self.find_relationships(selected_tables)
            
            # Stage 4: Generate and Execute SQL
            print("   ⚡ Stage 4: Generating SQL...")
            sql = await self.generate_sql(question, intent, selected_tables, relationships)
            if not sql:
                return QueryResult(question=question, sql_query="", results=[],
                                 error="Failed to generate SQL")
            
            print(f"      💾 Generated: {sql[:60]}...")
            
            # Execute
            results, error = self.execute_sql(sql)
            
            return QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=[t['full_name'] for t in selected_tables]
            )
            
        except Exception as e:
            return QueryResult(question=question, sql_query="", results=[],
                             error=f"Pipeline failed: {str(e)}")
    
    async def analyze_intent(self, question: str) -> Dict[str, Any]:
        """Stage 1: Analyze what user wants"""
        
        system_prompt = """You analyze business questions to understand intent.
Extract the key information needed to answer the question.
Respond with JSON only."""
        
        user_prompt = f"""
Question: "{question}"

What does the user want to know? Extract:
1. Action: count, list, show, calculate, etc.
2. Subject: what they're asking about (customers, payments, orders, etc.)
3. Filters: time periods, conditions, etc.
4. Expected result type: single number, list, summary

JSON format:
{{
  "action": "count",
  "subject": "customers who made payments",
  "filters": ["2025", "paid status"],
  "result_type": "single_number"
}}
"""
        
        response = await self.llm.ask(system_prompt, user_prompt)
        return self.parse_json(response)
    
    async def select_tables_with_data(self, question: str, intent: Dict) -> List[Dict]:
        """Stage 2: Select tables using semantic analysis + sample data"""
        
        # Use the already classified tables from semantic analysis
        relevant_tables = []
        
        for table in self.classified_tables:
            # Skip tables with no data or unknown entity type
            if not table.has_data() or table.entity_type == 'Unknown':
                continue
            
            # Create table summary with semantic classification + sample data
            sample_preview = ""
            if table.sample_data:
                first_row = table.sample_data[0]
                sample_items = []
                for key, value in list(first_row.items())[:4]:
                    if value is not None:
                        sample_items.append(f"{key}: {value}")
                sample_preview = ", ".join(sample_items)
            
            table_summary = {
                'full_name': table.full_name,
                'name': table.name,
                'entity_type': table.entity_type,  # Use semantic classification!
                'confidence': getattr(table, 'confidence', 0.0),
                'row_count': table.row_count,
                'columns': [col['name'] for col in table.columns[:8]],
                'sample_data': sample_preview
            }
            relevant_tables.append(table_summary)
        
        # Sort by confidence and row count
        relevant_tables.sort(key=lambda t: (t['confidence'], t['row_count']), reverse=True)
        
        # Take top candidates
        top_candidates = relevant_tables[:40]  # Top 40 candidates
        
        system_prompt = """You are a data analyst. Select tables based on their ENTITY CLASSIFICATION and sample data.
The semantic analysis has already classified these tables by business entity type.
Use both the entity type and sample data to make the best selection.
Respond with JSON only."""
        
        user_prompt = f"""
Question: "{question}"
Intent: {json.dumps(intent)}

CLASSIFIED TABLES (semantic analysis already done):
{json.dumps(top_candidates, indent=2)}

The entity_type field shows what each table represents (Customer, Payment, Order, etc.).
The sample_data shows actual values from the database.

Based on the ENTITY TYPES and SAMPLE DATA, select 3-6 tables that contain the data needed to answer this question.

JSON format:
{{
  "selected_tables": ["[dbo].[Customers]", "[dbo].[Payments]"],
  "reasoning": "Selected Customer tables (entity_type=Customer) with real customer data and Payment tables (entity_type=Payment) with payment amounts and dates."
}}
"""
        
        response = await self.llm.ask(system_prompt, user_prompt)
        result = self.parse_json(response)
        
        if result and 'selected_tables' in result:
            # Return full table data for selected tables
            selected = []
            table_lookup = {t.full_name: t for t in self.classified_tables}
            
            for table_name in result['selected_tables']:
                if table_name in table_lookup:
                    table = table_lookup[table_name]
                    selected.append({
                        'full_name': table.full_name,
                        'name': table.name,
                        'entity_type': table.entity_type,
                        'columns': table.columns,
                        'sample_data': table.sample_data,
                        'row_count': table.row_count
                    })
            return selected
        
        return []
    
    async def find_relationships(self, tables: List[Dict]) -> List[Dict]:
        """Stage 3: Find relationships between tables"""
        
        if len(tables) <= 1:
            return []
        
        # Simple relationship discovery using column names
        relationships = []
        
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                # Look for common column patterns
                cols1 = [col['name'].lower() for col in table1.get('columns', [])]
                cols2 = [col['name'].lower() for col in table2.get('columns', [])]
                
                # Find potential join columns
                for col1 in cols1:
                    for col2 in cols2:
                        if (col1 == col2 and 'id' in col1) or \
                           (col1.replace('_id', '') in table2['name'].lower()) or \
                           (col2.replace('_id', '') in table1['name'].lower()):
                            
                            relationships.append({
                                'table1': table1['full_name'],
                                'table2': table2['full_name'],
                                'column1': col1,
                                'column2': col2,
                                'join_condition': f"{col1} = {col2}"
                            })
        
        return relationships[:3]  # Limit relationships
    
    async def generate_sql(self, question: str, intent: Dict, tables: List[Dict], relationships: List[Dict]) -> str:
        """Stage 4: Generate SQL using semantic classifications + real schemas"""
        
        # Prepare complete context with entity types and schemas
        table_schemas = {}
        for table in tables:
            table_schemas[table['full_name']] = {
                'entity_type': table.get('entity_type', 'Unknown'),
                'columns': [f"{col['name']} ({col['data_type']})" for col in table.get('columns', [])],
                'sample_data': table.get('sample_data', [{}])[0] if table.get('sample_data') else {}
            }
        
        system_prompt = """You are an expert SQL developer. Generate SQL Server T-SQL queries using entity types and exact table/column names.
The tables have been classified by semantic analysis. Use this information to understand what data each table contains.
Generate working SQL only."""
        
        user_prompt = f"""
Question: "{question}"
Intent: {json.dumps(intent)}

CLASSIFIED TABLES WITH SCHEMAS:
{json.dumps(table_schemas, indent=2)}

RELATIONSHIPS:
{json.dumps(relationships, indent=2)}

Generate a complete T-SQL query using:
1. The entity_type tells you what each table contains (Customer, Payment, Order, etc.)
2. EXACT table names and column names from schemas above
3. Proper JOINs based on relationships and entity types
4. Appropriate WHERE clauses for the question
5. Square bracket syntax for SQL Server

For "count paid customers", look for Customer entity tables and Payment entity tables, then COUNT DISTINCT customers with payments.

Generate ONLY the SQL query, nothing else:
"""
        
        response = await self.llm.ask(system_prompt, user_prompt)
        return self.clean_sql(response)
    
    def execute_sql(self, sql: str) -> tuple:
        """Execute SQL and return results"""
        
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
                    
                    for row in cursor.fetchmany(100):  # Limit results
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = self.safe_value(value)
                        results.append(row_dict)
                    
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            return [], str(e)
    
    def parse_json(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            import re
            # Remove markdown
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            # Extract JSON
            cleaned = re.sub(r'^[^{]*', '', cleaned)
            cleaned = re.sub(r'[^}]*$', '', cleaned)
            return json.loads(cleaned)
        except:
            return {}
    
    def clean_sql(self, response: str) -> str:
        """Clean SQL from LLM response"""
        import re
        
        # Remove markdown
        cleaned = re.sub(r'```sql\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Extract SQL statement
        lines = cleaned.strip().split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT') or sql_lines:
                sql_lines.append(line)
                if line.endswith(';'):
                    break
        
        if sql_lines:
            return '\n'.join(sql_lines).rstrip(';')
        
        return cleaned.strip()
    
    def safe_value(self, value):
        """Convert database value to safe format"""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)[:200]
    
    def display_result(self, result: QueryResult):
        """Display query result"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.tables_used:
                print(f"💡 Tables used: {', '.join(result.tables_used)}")
        else:
            print(f"📋 Generated SQL:")
            print(f"   {result.sql_query}")
            
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                if len(result.results) == 1 and len(result.results[0]) == 1:
                    # Single value result
                    value = list(result.results[0].values())[0]
                    column_name = list(result.results[0].keys())[0]
                    if isinstance(value, (int, float)):
                        print(f"   🎯 {column_name}: {value:,}")
                    else:
                        print(f"   🎯 {column_name}: {value}")
                else:
                    # Multiple results
                    for i, row in enumerate(result.results[:5], 1):
                        display_row = {}
                        for key, value in list(row.items())[:6]:
                            if isinstance(value, str) and len(value) > 30:
                                display_row[key] = value[:30] + "..."
                            elif isinstance(value, (int, float)) and value > 1000:
                                display_row[key] = f"{value:,}"
                            else:
                                display_row[key] = value
                        print(f"   {i}. {display_row}")
                    
                    if len(result.results) > 5:
                        print(f"   ... and {len(result.results) - 5} more rows")
            
            if result.tables_used:
                print(f"📋 Tables used:")
                # Show entity types for better understanding
                table_lookup = {t.full_name: t for t in self.classified_tables}
                for table_name in result.tables_used:
                    if table_name in table_lookup:
                        table = table_lookup[table_name]
                        entity_type = getattr(table, 'entity_type', 'Unknown')
                        print(f"      • {table_name} ({entity_type})")
                    else:
                        print(f"      • {table_name}")
            
            print("✅ Query completed successfully using semantic classifications")