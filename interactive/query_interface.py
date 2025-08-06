#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced 4-Stage Automated Query Pipeline - Fixed Schema Reading
Reads actual table schemas from database_structure.json instead of guessing
"""

import json
import pyodbc
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult

class EnhancedQueryLLMClient:
    """Enhanced LLM client with accurate schema information"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.05,
            request_timeout=60
        )
    
    async def analyze_intent(self, question: str, domain_info: str) -> Dict[str, Any]:
        """Stage 1: Analyze business intent"""
        prompt = f"""
Analyze this business question and extract the intent:

QUESTION: "{question}"
BUSINESS DOMAIN: {domain_info}

Determine:
1. Primary entities needed (Customer, Payment, Order, Product, User, etc.)
2. Operation type (count, sum, list, analyze, etc.)
3. Filters or conditions (date ranges, specific criteria)
4. Expected result type (number, list, summary, etc.)

Respond with JSON only:
{{
  "primary_entities": ["Customer", "Payment"],
  "operation_type": "count",
  "filters": ["date_2025"],
  "result_type": "number",
  "business_intent": "Count customers who made payments in 2025"
}}
"""
        
        try:
            messages = [
                SystemMessage(content="You are a business analyst. Extract business intent from questions. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return self._parse_json_response(response.content)
        except Exception as e:
            print(f"   ⚠️ Intent analysis failed: {e}")
            return {}
    
    async def select_tables(self, intent: Dict[str, Any], available_tables: List[Dict]) -> List[str]:
        """Stage 2: Select relevant tables using exact table information"""
        
        # Format tables with complete information
        table_descriptions = []
        for table in available_tables:
            table_desc = f"""
Table: {table['full_name']}
Entity Type: {table['entity_type']} (confidence: {table['confidence']:.2f})
Row Count: {table['row_count']}
Key Columns: {', '.join(table['key_columns'][:8])}
Sample Data: {table['sample_preview']}
"""
            table_descriptions.append(table_desc)
        
        prompt = f"""
Based on this business intent, select the most relevant tables:

BUSINESS INTENT: {intent}

AVAILABLE TABLES:
{chr(10).join(table_descriptions)}

Select 2-5 most relevant tables that can answer the question.
Consider entity types, column names, and sample data.
Return the EXACT table names as they appear above.

Respond with JSON only:
{{
  "selected_tables": ["[schema].[table1]", "[schema].[table2]"],
  "reasoning": "brief explanation of selection"
}}
"""
        
        try:
            messages = [
                SystemMessage(content="You are a database expert. Select relevant tables for queries. Use EXACT table names. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            result = self._parse_json_response(response.content)
            return result.get('selected_tables', [])
        except Exception as e:
            print(f"   ⚠️ Table selection failed: {e}")
            return []
    
    async def resolve_relationships(self, selected_tables: List[str], table_schemas: Dict[str, Dict], relationships: List[Dict]) -> List[Dict]:
        """Stage 3: Resolve table relationships using actual schemas"""
        
        # Prepare relationship information with actual column names
        relevant_relationships = []
        for rel in relationships:
            if any(table_name in rel['from_table'] or table_name in rel['to_table'] for table_name in selected_tables):
                relevant_relationships.append(rel)
        
        prompt = f"""
Determine how to join these selected tables using their actual schemas:

SELECTED TABLES WITH SCHEMAS:
{json.dumps({name: schema for name, schema in table_schemas.items() if name in selected_tables}, indent=2)}

KNOWN RELATIONSHIPS:
{json.dumps(relevant_relationships, indent=2)}

Determine the optimal JOIN strategy using the EXACT column names from the schemas above.

Respond with JSON only:
{{
  "joins": [
    {{
      "from_table": "[schema].[table1]", 
      "to_table": "[schema].[table2]",
      "join_condition": "t1.actual_column_id = t2.actual_id_column",
      "join_type": "INNER JOIN"
    }}
  ],
  "join_order": ["table1", "table2", "table3"]
}}
"""
        
        try:
            messages = [
                SystemMessage(content="You are a SQL expert. Design optimal table joins using EXACT column names from schemas. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            result = self._parse_json_response(response.content)
            return result.get('joins', [])
        except Exception as e:
            print(f"   ⚠️ Relationship resolution failed: {e}")
            return []
    
    async def generate_sql(self, intent: Dict[str, Any], table_schemas: Dict[str, Dict], joins: List[Dict]) -> str:
        """Stage 4: Generate SQL using exact table schemas"""
        
        # Format complete table schemas for SQL generation
        schema_descriptions = []
        for table_name, schema in table_schemas.items():
            columns_list = []
            for col in schema['columns']:
                col_name = col['name']
                col_type = col['data_type']
                is_pk = " [PRIMARY KEY]" if col.get('is_primary_key', False) else ""
                columns_list.append(f"  {col_name} ({col_type}){is_pk}")
            
            schema_desc = f"""
TABLE: {table_name}
COLUMNS:
{chr(10).join(columns_list)}
SAMPLE DATA: {schema.get('sample_preview', 'No sample data')}
ROW COUNT: {schema.get('row_count', 0)}
"""
            schema_descriptions.append(schema_desc)
        
        prompt = f"""
Generate SQL Server T-SQL query for this business question using the EXACT table and column names provided:

BUSINESS INTENT: {intent}

EXACT TABLE SCHEMAS:
{chr(10).join(schema_descriptions)}

REQUIRED JOINS:
{json.dumps(joins, indent=2)}

IMPORTANT REQUIREMENTS:
1. Use the EXACT table names and column names from the schemas above
2. Use proper SQL Server syntax with square brackets for table/column names
3. Implement the required joins correctly using the exact column names
4. Include appropriate WHERE clauses for filters
5. Use TOP 100 unless counting/summing
6. Use meaningful column aliases

Generate ONLY the complete, executable T-SQL query using the exact names provided above:
"""
        
        try:
            messages = [
                SystemMessage(content="You are an expert SQL Server developer. Generate correct T-SQL queries using EXACT table and column names provided. Return only the SQL query."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return self._clean_sql_response(response.content)
        except Exception as e:
            print(f"   ⚠️ SQL generation failed: {e}")
            return ""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            import re
            
            cleaned = response.strip()
            
            # Remove markdown code blocks
            if '```json' in cleaned:
                match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            elif '```' in cleaned:
                match = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            return json.loads(cleaned)
            
        except Exception:
            return {}
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean SQL response"""
        import re
        
        # Remove markdown
        cleaned = re.sub(r'```sql\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Extract SQL query
        lines = cleaned.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT') or sql_lines:
                sql_lines.append(line)
                if line.endswith(';'):
                    break
        
        if sql_lines:
            return '\n'.join(sql_lines).rstrip(';')
        
        # Fallback
        if 'SELECT' in response.upper():
            return response.strip().rstrip(';')
        
        return ""

class EnhancedQueryInterface:
    """Enhanced 4-Stage Automated Query Pipeline with accurate schema reading"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = EnhancedQueryLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.database_schema: Dict[str, Dict] = {}  # Complete database schema
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship]):
        """Start enhanced 4-stage automated query session"""
        
        self.tables = tables
        self.domain = domain  
        self.relationships = relationships
        
        # Load complete database schema from cache
        await self._load_database_schema()
        
        # Show system capabilities
        self._show_system_capabilities()
        
        query_count = 0
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    self._show_help()
                    continue
                elif question.lower() == 'examples':
                    self._show_examples()
                    continue  
                elif question.lower() == 'debug':
                    self._show_debug_info()
                    continue
                elif not question:
                    continue
                
                query_count += 1
                print(f"🚀 Processing with enhanced 4-stage automated pipeline...")
                
                start_time = time.time()
                result = await self._process_enhanced_4_stage_pipeline(question)
                result.execution_time = time.time() - start_time
                
                print(f"⏱️ Completed in {result.execution_time:.1f}s")
                print("-" * 60)
                
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Session summary: {query_count} queries processed")
        print("👋 Thanks for using the Enhanced 4-Stage Automated Query Pipeline!")
    
    async def _load_database_schema(self):
        """Load complete database schema from database_structure.json"""
        
        cache_file = self.config.get_cache_path("database_structure.json")
        
        if not cache_file.exists():
            print("⚠️ database_structure.json not found. Please run discovery first.")
            return
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process tables from discovery data
            if 'tables' in data:
                for table_data in data['tables']:
                    full_name = table_data['full_name']
                    self.database_schema[full_name] = {
                        'name': table_data['name'],
                        'schema': table_data['schema'],
                        'full_name': full_name,
                        'object_type': table_data['object_type'],
                        'columns': table_data['columns'],
                        'sample_data': table_data['sample_data'],
                        'row_count': table_data['row_count'],
                        'relationships': table_data.get('relationships', [])
                    }
            
            print(f"✅ Loaded complete schema for {len(self.database_schema)} database objects")
            
        except Exception as e:
            print(f"⚠️ Failed to load database schema: {e}")
    
    async def _process_enhanced_4_stage_pipeline(self, question: str) -> QueryResult:
        """Execute the enhanced 4-stage automated pipeline with accurate schema reading"""
        
        try:
            # Stage 1: Business Intent Analysis
            print("   🎯 Stage 1: Business Intent Analysis...")
            domain_info = f"{self.domain.domain_type} - {self.domain.industry}" if self.domain else "Business Operations"
            intent = await self.llm_client.analyze_intent(question, domain_info)
            
            if not intent:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to analyze business intent",
                    tables_used=[]
                )
            
            print(f"      📊 Intent: {intent.get('business_intent', 'Unknown')}")
            print(f"      🎯 Entities: {', '.join(intent.get('primary_entities', []))}")
            
            # Stage 2: Smart Table Selection using complete schema
            print("   📋 Stage 2: Smart Table Selection...")
            available_tables = self._prepare_tables_with_complete_schema()
            selected_table_names = await self.llm_client.select_tables(intent, available_tables)
            
            if not selected_table_names:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found for the question",
                    tables_used=[]
                )
            
            # Get actual table schemas for selected tables
            selected_schemas = self._get_complete_table_schemas(selected_table_names)
            print(f"      ✅ Selected {len(selected_schemas)} tables:")
            for table_name in selected_schemas.keys():
                entity_type = self._get_table_entity_type(table_name)
                print(f"         • {table_name} ({entity_type})")
            
            # Stage 3: Relationship Resolution using actual schemas
            print("   🔗 Stage 3: Relationship Resolution...")
            available_relationships = self._prepare_relationships_for_resolution(selected_table_names)
            joins = await self.llm_client.resolve_relationships(selected_table_names, selected_schemas, available_relationships)
            
            print(f"      ✅ Resolved {len(joins)} joins")
            
            # Stage 4: Enhanced SQL Generation using exact schemas
            print("   ⚡ Stage 4: Enhanced SQL Generation...")
            sql_query = await self.llm_client.generate_sql(intent, selected_schemas, joins)
            
            if not sql_query:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to generate SQL query",
                    tables_used=list(selected_schemas.keys())
                )
            
            print(f"      💾 Generated SQL: {sql_query[:100]}...")
            
            # Execute query with validation
            print("   🚀 Executing query...")
            results, error = self._execute_query_with_validation(sql_query, intent)
            
            return QueryResult(
                question=question,
                sql_query=sql_query,
                results=results,
                error=error,
                tables_used=list(selected_schemas.keys())
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Enhanced 4-stage pipeline failed: {str(e)}",
                tables_used=[]
            )
    
    def _prepare_tables_with_complete_schema(self) -> List[Dict]:
        """Prepare table information with complete schema details"""
        
        available_tables = []
        
        for table in self.tables:
            if table.full_name in self.database_schema:
                schema_info = self.database_schema[table.full_name]
                
                # Get key columns (first 8 columns)
                key_columns = [col['name'] for col in schema_info['columns'][:8]]
                
                # Create sample preview
                sample_preview = "No sample data"
                if schema_info['sample_data']:
                    sample_row = schema_info['sample_data'][0]
                    sample_items = []
                    for key, value in list(sample_row.items())[:3]:
                        if value is not None:
                            sample_items.append(f"{key}: {str(value)[:30]}")
                    sample_preview = ", ".join(sample_items)
                
                table_info = {
                    'full_name': table.full_name,
                    'entity_type': getattr(table, 'entity_type', 'Unknown'),
                    'confidence': getattr(table, 'confidence', 0.0),
                    'row_count': schema_info['row_count'],
                    'key_columns': key_columns,
                    'sample_preview': sample_preview,
                    'object_type': schema_info['object_type']
                }
                available_tables.append(table_info)
        
        # Sort by entity type confidence and row count
        available_tables.sort(key=lambda x: (x['confidence'], x['row_count']), reverse=True)
        
        return available_tables
    
    def _get_complete_table_schemas(self, selected_table_names: List[str]) -> Dict[str, Dict]:
        """Get complete schema information for selected tables"""
        
        selected_schemas = {}
        
        for table_name in selected_table_names:
            if table_name in self.database_schema:
                schema_info = self.database_schema[table_name]
                
                # Create sample preview
                sample_preview = "No sample data"
                if schema_info['sample_data']:
                    sample_row = schema_info['sample_data'][0]
                    sample_items = []
                    for key, value in list(sample_row.items())[:3]:
                        if value is not None:
                            sample_items.append(f"{key}: {str(value)[:50]}")
                    sample_preview = ", ".join(sample_items)
                
                selected_schemas[table_name] = {
                    'columns': schema_info['columns'],
                    'sample_preview': sample_preview,
                    'row_count': schema_info['row_count'],
                    'object_type': schema_info['object_type']
                }
            else:
                print(f"      ⚠️ Schema not found for table: {table_name}")
        
        return selected_schemas
    
    def _get_table_entity_type(self, table_name: str) -> str:
        """Get entity type for a table"""
        for table in self.tables:
            if table.full_name == table_name:
                return getattr(table, 'entity_type', 'Unknown')
        return 'Unknown'
    
    def _prepare_relationships_for_resolution(self, selected_table_names: List[str]) -> List[Dict]:
        """Prepare relationship information for resolution stage"""
        
        relevant_relationships = []
        for rel in self.relationships:
            # Check if relationship involves selected tables
            if any(table_name in rel.from_table or table_name in rel.to_table for table_name in selected_table_names):
                rel_info = {
                    'from_table': rel.from_table,
                    'to_table': rel.to_table,
                    'relationship_type': rel.relationship_type,
                    'confidence': rel.confidence,
                    'description': rel.description
                }
                relevant_relationships.append(rel_info)
        
        return relevant_relationships
    
    def _execute_query_with_validation(self, sql_query: str, intent: Dict[str, Any]) -> Tuple[List[Dict], Optional[str]]:
        """Execute query with business validation"""
        
        try:
            connection_string = self.config.get_database_connection_string()
            
            with pyodbc.connect(connection_string) as conn:
                # Set UTF-8 encoding
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                # Get results
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor.fetchmany(self.config.max_results):
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = self._safe_database_value(value)
                        results.append(row_dict)
                    
                    # Business validation
                    validation_error = self._validate_business_result(results, intent)
                    if validation_error:
                        return results, validation_error
                    
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages
            if 'invalid object name' in error_msg.lower():
                error_msg = f"Table not found: {error_msg[:200]}"
            elif 'invalid column name' in error_msg.lower():
                error_msg = f"Column not found: {error_msg[:200]}"
            elif 'syntax error' in error_msg.lower():
                error_msg = f"SQL syntax error: {error_msg[:200]}"
            
            return [], error_msg
    
    def _validate_business_result(self, results: List[Dict], intent: Dict[str, Any]) -> Optional[str]:
        """Validate that results make business sense"""
        
        if not results:
            return None
        
        result_type = intent.get('result_type', '')
        operation_type = intent.get('operation_type', '')
        entities = intent.get('primary_entities', [])
        
        # Validate count operations
        if operation_type == 'count' and result_type == 'number':
            if len(results) == 1 and len(results[0]) == 1:
                count_value = list(results[0].values())[0]
                if isinstance(count_value, (int, float)):
                    # Business validation rules
                    if 'Customer' in entities and count_value == 0:
                        return "Warning: Query returned 0 customers. This may indicate a data issue or incorrect query logic."
                    elif 'Payment' in entities and count_value == 0:
                        return "Warning: Query returned 0 payments. This may indicate a data issue or incorrect date filters."
        
        return None
    
    def _safe_database_value(self, value) -> Any:
        """Convert database value to safe format"""
        from datetime import datetime
        
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)[:200]
    
    def _show_debug_info(self):
        """Show debug information about loaded schemas"""
        print(f"\n🔧 DEBUG INFORMATION:")
        print(f"   📊 Total tables loaded: {len(self.tables)}")
        print(f"   💾 Schema cache loaded: {len(self.database_schema)}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        if self.database_schema:
            print(f"   📋 Sample tables in schema:")
            for i, (table_name, schema) in enumerate(list(self.database_schema.items())[:5]):
                columns = [col['name'] for col in schema['columns'][:5]]
                print(f"      {i+1}. {table_name}: {', '.join(columns)}")
        
        if self.tables:
            print(f"   🧠 Sample classified tables:")
            for i, table in enumerate(self.tables[:5]):
                entity_type = getattr(table, 'entity_type', 'Unknown')
                confidence = getattr(table, 'confidence', 0.0)
                print(f"      {i+1}. {table.full_name}: {entity_type} ({confidence:.2f})")
    
    def _show_system_capabilities(self):
        """Show enhanced system capabilities"""
        
        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        classified_count = sum(1 for t in self.tables if hasattr(t, 'entity_type') and t.entity_type != 'Unknown')
        
        print(f"✅ ENHANCED 4-STAGE AUTOMATED PIPELINE READY:")
        print(f"   📊 Database: {table_count} tables, {view_count} views")
        print(f"   💾 Complete schemas loaded: {len(self.database_schema)} objects")
        print(f"   🧠 Classified: {classified_count} business entities")
        print(f"   🔗 Relationships: {len(self.relationships)} discovered")
        
        if self.domain:
            print(f"   🏢 Domain: {self.domain.domain_type}")
            
            # Show entity counts
            entity_counts = {}
            for table in self.tables:
                if hasattr(table, 'entity_type') and table.entity_type != 'Unknown':
                    entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
            
            if entity_counts:
                print(f"   🏢 Business Entities:")
                for entity_type, count in sorted(entity_counts.items()):
                    print(f"      • {entity_type}: {count}")
        
        print("\n🚀 Enhanced 4-Stage Pipeline Process:")
        print("   🎯 Stage 1: Business Intent Analysis (2-3s)")
        print("   📋 Stage 2: Smart Table Selection with Complete Schemas (2-3s)")  
        print("   🔗 Stage 3: Relationship Resolution using Actual Column Names (2s)")
        print("   ⚡ Stage 4: SQL Generation with Exact Table/Column Names (2-3s)")
        print("   💡 Total time: 10-15 seconds with accurate schema reading")
        
        print("\n💡 Type 'help' for commands, 'examples' for sample questions, 'debug' for schema info")
    
    def _show_help(self):
        """Show help information"""
        
        print("\n🔧 COMMANDS:")
        print("   • 'help' - Show this help")
        print("   • 'examples' - Show sample questions")
        print("   • 'debug' - Show schema debug information")
        print("   • 'quit' or 'exit' - Exit pipeline")
        
        print("\n🚀 ENHANCED 4-STAGE PIPELINE FEATURES:")
        print("   • Reads exact table schemas from database_structure.json")
        print("   • Uses actual column names and data types in SQL generation")
        print("   • AI-powered table selection with complete schema context")
        print("   • Smart relationship resolution using real foreign keys")
        print("   • Validated SQL generation with exact table/column references")
        print("   • Business logic validation of results")
        
        print("\n💡 QUERY TIPS:")
        print("   • Ask in natural language: 'How many customers made payments in 2025?'")
        print("   • Be specific about time periods: 'total revenue this year'")
        print("   • Ask for comparisons: 'monthly growth compared to last year'")
        print("   • Complex analysis: 'top 10 customers by order value'")
    
    def _show_examples(self):
        """Show example questions"""
        
        print("\n💡 SAMPLE QUESTIONS:")
        
        if self.domain and self.domain.sample_questions:
            for i, question in enumerate(self.domain.sample_questions[:10], 1):
                print(f"   {i}. {question}")
        else:
            # Default examples
            print("   1. How many customers do we have?")
            print("   2. What is our total revenue for 2025?")
            print("   3. Count total paid customers")
            print("   4. Show customer payment information")
            print("   5. How many orders were placed this year?")
            print("   6. List customers with recent payments")
            print("   7. What's our average order value?")
            print("   8. Show top 10 customers by revenue")
        
        print("\n🔥 ADVANCED QUERIES:")
        print("   • 'Monthly revenue growth compared to last year'")
        print("   • 'Customers who haven't made payments in 6 months'")
        print("   • 'Product performance by customer segment'")
        print("   • 'Customer lifetime value analysis'")
        
        print("\n⚡ The enhanced pipeline uses exact database schemas for accurate results!")
    
    def _display_result(self, result: QueryResult):
        """Display query result with enhanced formatting"""
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.tables_used:
                print(f"💡 Tables used: {', '.join(result.tables_used)}")
        else:
            print(f"📋 Generated SQL:")
            print(f"   {result.sql_query}")
            
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                # Smart result display
                if len(result.results) == 1 and len(result.results[0]) == 1:
                    # Single value result (likely a count or sum)
                    value = list(result.results[0].values())[0]
                    column_name = list(result.results[0].keys())[0]
                    print(f"   🎯 {column_name}: {value:,}" if isinstance(value, (int, float)) else f"   🎯 {column_name}: {value}")
                else:
                    # Multiple results
                    for i, row in enumerate(result.results[:5], 1):
                        # Limit display width
                        display_row = {}
                        for key, value in list(row.items())[:6]:  # Max 6 columns
                            if isinstance(value, str) and len(value) > 30:
                                display_row[key] = value[:30] + "..."
                            elif isinstance(value, (int, float)) and value > 1000:
                                display_row[key] = f"{value:,}"  # Add commas for large numbers
                            else:
                                display_row[key] = value
                        print(f"   {i}. {display_row}")
                    
                    if len(result.results) > 5:
                        print(f"   ... and {len(result.results) - 5} more rows")
            
            if result.tables_used:
                print(f"📋 Tables used: {', '.join(result.tables_used)}")
            
            if result.execution_time > 0:
                pipeline_stages = "🎯📋🔗⚡"  # Icons for the 4 stages
                print(f"⚡ Enhanced pipeline time: {result.execution_time:.1f}s {pipeline_stages}")
            
            # Success indicator for business queries
            if not result.error and result.results:
                print("✅ Query completed successfully with schema-accurate SQL generation")

# For backward compatibility
QueryInterface = EnhancedQueryInterface
QueryLLMClient = EnhancedQueryLLMClient