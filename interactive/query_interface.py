#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data-Driven 4-Stage Query Pipeline - Generic and Sample-Data Based
Uses actual table content instead of entity classifications
"""

import json
import pyodbc
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import re

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult

class DataDrivenLLMClient:
    """LLM client that uses actual data content instead of entity types"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.05,
            request_timeout=60
        )
    
    async def analyze_business_intent(self, question: str, domain_info: str) -> Dict[str, Any]:
        """Stage 1: Extract business intent from natural language question"""
        
        prompt = f"""
Analyze this business question and extract the core intent:

QUESTION: "{question}"
BUSINESS DOMAIN: {domain_info}

Extract:
1. Main Action: What does the user want to do? (count, list, show, calculate, analyze, etc.)
2. Target Subject: What are they asking about? (customers, payments, orders, people, transactions, etc.)
3. Filters/Conditions: Any date ranges, status filters, amount thresholds, etc.
4. Expected Result: Single number, list of records, summary, etc.

Think about what KIND OF DATA would be needed to answer this question, not specific entity types.

Respond with JSON only:
{{
  "action": "count",
  "subject": "people who made payments", 
  "filters": ["year 2025", "paid transactions"],
  "result_type": "single_number",
  "data_requirements": ["person/customer identifiers", "payment/transaction records", "date fields"],
  "business_question": "How many unique people made payments in 2025?"
}}
"""
        
        try:
            messages = [
                SystemMessage(content="You are a business analyst. Extract intent from questions. Think about DATA CONTENT, not entity types. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return self._parse_json_response(response.content)
        except Exception as e:
            print(f"   ⚠️ Intent analysis failed: {e}")
            return {}
    
    async def select_tables_by_content(self, intent: Dict[str, Any], table_analysis: List[Dict]) -> List[str]:
        """Stage 2: Select tables based on actual data content analysis"""
        
        # Create rich table descriptions with sample data analysis
        table_descriptions = []
        for table in table_analysis:
            
            # Analyze what this table actually contains based on sample data
            content_summary = table['content_analysis']
            
            table_desc = f"""
Table: {table['full_name']}
Rows: {table['row_count']}
Data Content Analysis: {content_summary}
Key Columns: {', '.join(table['key_columns'][:8])}
Sample Data: {table['sample_preview']}
Column Patterns: {table['column_patterns']}
Data Types Present: {table['data_type_summary']}
"""
            table_descriptions.append(table_desc)
        
        prompt = f"""
Based on this business intent, select tables that contain the ACTUAL DATA needed:

BUSINESS INTENT: {json.dumps(intent, indent=2)}

AVAILABLE TABLES WITH ACTUAL CONTENT ANALYSIS:
{chr(10).join(table_descriptions)}

Select 3-6 tables that actually contain the data needed to answer the question.
Focus on:
1. Tables that have data matching the subject (people, payments, transactions, etc.)
2. Tables with relevant date fields for time-based filters
3. Tables with identifiers that can be used for counting/linking

Return EXACT table names as they appear above.

Respond with JSON only:
{{
  "selected_tables": ["[schema].[table1]", "[schema].[table2]"],
  "reasoning": "Table1 contains customer/people data with IDs. Table2 contains payment/transaction data with dates and amounts. These can be linked to count unique people who paid."
}}
"""
        
        try:
            messages = [
                SystemMessage(content="You are a data analyst. Select tables based on ACTUAL DATA CONTENT, not table names or classifications. Use EXACT table names. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            result = self._parse_json_response(response.content)
            return result.get('selected_tables', [])
        except Exception as e:
            print(f"   ⚠️ Table selection failed: {e}")
            return []
    
    async def resolve_table_relationships(self, selected_tables: List[str], table_schemas: Dict[str, Dict], 
                                         sample_data_analysis: Dict[str, Dict]) -> List[Dict]:
        """Stage 3: Find relationships between tables using actual data analysis"""
        
        # Prepare relationship analysis
        relationship_info = []
        
        for i, table1 in enumerate(selected_tables):
            for table2 in selected_tables[i+1:]:
                if table1 in table_schemas and table2 in table_schemas:
                    schema1 = table_schemas[table1]
                    schema2 = table_schemas[table2]
                    
                    # Analyze potential relationships
                    rel_analysis = self._analyze_table_relationship(
                        table1, schema1, sample_data_analysis.get(table1, {}),
                        table2, schema2, sample_data_analysis.get(table2, {})
                    )
                    
                    if rel_analysis:
                        relationship_info.append(rel_analysis)
        
        prompt = f"""
Analyze these tables and determine how to join them using ACTUAL COLUMN NAMES and DATA:

SELECTED TABLES WITH SCHEMAS:
{json.dumps({name: schema for name, schema in table_schemas.items() if name in selected_tables}, indent=2)}

POTENTIAL RELATIONSHIPS FOUND:
{json.dumps(relationship_info, indent=2)}

Determine the optimal JOIN strategy using:
1. EXACT column names from the schemas above
2. Data patterns and common values found in sample data
3. ID/identifier fields that appear to link tables

Respond with JSON only:
{{
  "joins": [
    {{
      "from_table": "[schema].[table1]", 
      "to_table": "[schema].[table2]",
      "join_condition": "t1.ActualColumnName = t2.ActualColumnName",
      "join_type": "INNER JOIN",
      "confidence": 0.9,
      "reasoning": "Found matching ID values in sample data"
    }}
  ],
  "join_order": ["table1", "table2", "table3"]
}}
"""
        
        try:
            messages = [
                SystemMessage(content="You are a database expert. Design JOINs using EXACT column names from schemas and data analysis. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            result = self._parse_json_response(response.content)
            return result.get('joins', [])
        except Exception as e:
            print(f"   ⚠️ Relationship resolution failed: {e}")
            return []
    
    async def generate_data_driven_sql(self, intent: Dict[str, Any], table_schemas: Dict[str, Dict], 
                                      joins: List[Dict], sample_data_analysis: Dict[str, Dict]) -> str:
        """Stage 4: Generate SQL using actual data understanding"""
        
        # Prepare complete context
        sql_context = {
            'intent': intent,
            'table_schemas': table_schemas,
            'joins': joins,
            'sample_data_insights': sample_data_analysis
        }
        
        prompt = f"""
Generate SQL Server T-SQL query based on actual data analysis:

COMPLETE CONTEXT:
{json.dumps(sql_context, indent=2, default=str)}

REQUIREMENTS:
1. Use EXACT table and column names from schemas
2. Implement the business intent based on actual data content
3. Use appropriate JOINs based on the relationship analysis
4. Apply filters based on actual column names and data patterns
5. Use proper SQL Server syntax with square brackets
6. Return meaningful results based on what the data actually contains

For the business question: "{intent.get('business_question', 'Unknown question')}"

Generate ONLY the complete, executable T-SQL query:
"""
        
        try:
            messages = [
                SystemMessage(content="You are an expert SQL developer. Generate queries based on ACTUAL DATA CONTENT and EXACT column names. Return only the SQL query."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return self._clean_sql_response(response.content)
        except Exception as e:
            print(f"   ⚠️ SQL generation failed: {e}")
            return ""
    
    def _analyze_table_relationship(self, table1: str, schema1: Dict, sample1: Dict,
                                   table2: str, schema2: Dict, sample2: Dict) -> Optional[Dict]:
        """Analyze potential relationship between two tables using actual data"""
        
        cols1 = {col['name'].lower(): col for col in schema1.get('columns', [])}
        cols2 = {col['name'].lower(): col for col in schema2.get('columns', [])}
        
        potential_links = []
        
        # Look for common column names
        common_cols = set(cols1.keys()).intersection(set(cols2.keys()))
        for col_name in common_cols:
            if 'id' in col_name or col_name in ['code', 'key', 'number']:
                potential_links.append({
                    'type': 'common_column',
                    'column': col_name,
                    'table1_col': cols1[col_name]['name'],
                    'table2_col': cols2[col_name]['name'],
                    'confidence': 0.8
                })
        
        # Look for ID reference patterns
        for col1_name, col1_info in cols1.items():
            if col1_name.endswith('_id') or col1_name.endswith('id'):
                entity_name = col1_name.replace('_id', '').replace('id', '')
                
                # Check if table2 might be the referenced entity
                if entity_name in table2.lower() or any(entity_name in col for col in cols2.keys()):
                    # Look for ID column in table2
                    if 'id' in cols2:
                        potential_links.append({
                            'type': 'foreign_key_pattern',
                            'from_column': col1_info['name'],
                            'to_column': cols2['id']['name'],
                            'confidence': 0.7
                        })
        
        # Analyze sample data for common values (simplified)
        if sample1.get('id_values') and sample2.get('id_values'):
            overlap = set(sample1['id_values']).intersection(set(sample2['id_values']))
            if len(overlap) >= 2:
                potential_links.append({
                    'type': 'data_overlap',
                    'common_values': len(overlap),
                    'confidence': min(0.9, len(overlap) / 10)
                })
        
        if potential_links:
            return {
                'table1': table1,
                'table2': table2,
                'potential_links': potential_links
            }
        
        return None
    
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

class DataDrivenQueryInterface:
    """Data-driven 4-Stage Query Pipeline using actual table content analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = DataDrivenLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.database_schema: Dict[str, Dict] = {}  # Complete database schema
        self.table_content_analysis: Dict[str, Dict] = {}  # Content analysis cache
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship]):
        """Start data-driven 4-stage automated query session"""
        
        self.tables = tables
        self.domain = domain  
        self.relationships = relationships
        
        # Load complete database schema from cache
        await self._load_database_schema()
        
        # Analyze table content using sample data
        print("🔍 Analyzing table content using sample data...")
        await self._analyze_table_content()
        
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
                elif question.lower() == 'analyze':
                    await self._show_content_analysis()
                    continue
                elif not question:
                    continue
                
                query_count += 1
                print(f"🚀 Processing with data-driven 4-stage pipeline...")
                
                start_time = time.time()
                result = await self._process_data_driven_pipeline(question)
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
        print("👋 Thanks for using the Data-Driven 4-Stage Query Pipeline!")
    
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
    
    async def _analyze_table_content(self):
        """Analyze what each table actually contains using sample data"""
        
        print(f"   🔍 Analyzing content of {len(self.database_schema)} tables...")
        
        for table_name, schema in self.database_schema.items():
            self.table_content_analysis[table_name] = self._analyze_single_table_content(schema)
        
        print(f"   ✅ Content analysis completed")
    
    def _analyze_single_table_content(self, schema: Dict) -> Dict:
        """Analyze what a single table contains based on its schema and sample data"""
        
        columns = schema.get('columns', [])
        sample_data = schema.get('sample_data', [])
        
        analysis = {
            'has_people_data': False,
            'has_payment_data': False,
            'has_date_fields': False,
            'has_amount_fields': False,
            'has_id_fields': False,
            'has_name_fields': False,
            'content_summary': '',
            'key_columns': [],
            'column_patterns': [],
            'data_type_summary': {},
            'sample_preview': '',
            'id_values': []
        }
        
        # Analyze columns
        for col in columns:
            col_name = col.get('name', '').lower()
            data_type = col.get('data_type', '').lower()
            
            analysis['key_columns'].append(col.get('name', ''))
            
            # Data type summary
            analysis['data_type_summary'][data_type] = analysis['data_type_summary'].get(data_type, 0) + 1
            
            # Pattern detection
            if 'id' in col_name:
                analysis['has_id_fields'] = True
                analysis['column_patterns'].append('identifiers')
            
            if any(word in col_name for word in ['name', 'first', 'last', 'full', 'company']):
                analysis['has_name_fields'] = True
                analysis['column_patterns'].append('names')
            
            if any(word in col_name for word in ['date', 'time', 'created', 'modified', 'updated']):
                analysis['has_date_fields'] = True
                analysis['column_patterns'].append('dates')
            
            if any(word in col_name for word in ['amount', 'price', 'total', 'cost', 'value', 'sum']):
                analysis['has_amount_fields'] = True
                analysis['column_patterns'].append('amounts')
            
            if any(word in col_name for word in ['email', 'phone', 'address', 'contact']):
                analysis['has_people_data'] = True
                analysis['column_patterns'].append('contact_info')
            
            if any(word in col_name for word in ['payment', 'transaction', 'billing', 'charge']):
                analysis['has_payment_data'] = True
                analysis['column_patterns'].append('payment_info')
        
        # Analyze sample data
        if sample_data:
            first_row = sample_data[0]
            
            # Create sample preview
            sample_items = []
            id_values = []
            
            for key, value in list(first_row.items())[:4]:  # First 4 columns
                if value is not None:
                    value_str = str(value)[:30]
                    sample_items.append(f"{key}: {value_str}")
                    
                    # Collect ID-like values
                    if 'id' in key.lower() and str(value).isdigit():
                        id_values.extend([str(row.get(key, '')) for row in sample_data[:3]])
            
            analysis['sample_preview'] = ", ".join(sample_items)
            analysis['id_values'] = [v for v in id_values if v and v.isdigit()]
            
            # Enhanced content detection based on sample data
            all_values = ' '.join(str(v) for row in sample_data for v in row.values() if v is not None).lower()
            
            if any(word in all_values for word in ['customer', 'client', '@', 'email']):
                analysis['has_people_data'] = True
            
            if any(word in all_values for word in ['payment', 'transaction', '$', '€', 'amount']):
                analysis['has_payment_data'] = True
        
        # Create content summary
        content_parts = []
        
        if analysis['has_people_data']:
            content_parts.append("people/customer data")
        if analysis['has_payment_data']:
            content_parts.append("payment/transaction data")
        if analysis['has_id_fields']:
            content_parts.append("identifier fields")
        if analysis['has_date_fields']:
            content_parts.append("date/time tracking")
        if analysis['has_amount_fields']:
            content_parts.append("amount/financial values")
        if analysis['has_name_fields']:
            content_parts.append("name/title fields")
        
        if content_parts:
            analysis['content_summary'] = f"Contains {', '.join(content_parts)}"
        else:
            analysis['content_summary'] = "General data table"
        
        # Remove duplicates from patterns
        analysis['column_patterns'] = list(set(analysis['column_patterns']))
        
        return analysis
    
    async def _process_data_driven_pipeline(self, question: str) -> QueryResult:
        """Execute the data-driven 4-stage pipeline"""
        
        try:
            # Stage 1: Business Intent Analysis
            print("   🎯 Stage 1: Business Intent Analysis...")
            domain_info = f"{self.domain.domain_type}" if self.domain else "Business Operations"
            intent = await self.llm_client.analyze_business_intent(question, domain_info)
            
            if not intent:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to analyze business intent",
                    tables_used=[]
                )
            
            print(f"      📊 Intent: {intent.get('business_question', 'Unknown')}")
            print(f"      🎯 Action: {intent.get('action', 'Unknown')}")
            print(f"      📝 Subject: {intent.get('subject', 'Unknown')}")
            
            # Stage 2: Content-Based Table Selection
            print("   📋 Stage 2: Content-Based Table Selection...")
            
            # Prepare table analysis for selection
            table_analysis = []
            for table_name, content_analysis in self.table_content_analysis.items():
                if table_name in self.database_schema:
                    schema = self.database_schema[table_name]
                    table_analysis.append({
                        'full_name': table_name,
                        'row_count': schema.get('row_count', 0),
                        'content_analysis': content_analysis['content_summary'],
                        'key_columns': content_analysis['key_columns'][:8],
                        'sample_preview': content_analysis['sample_preview'],
                        'column_patterns': content_analysis['column_patterns'],
                        'data_type_summary': content_analysis['data_type_summary']
                    })
            
            # Sort by relevance (tables with more data first)
            table_analysis.sort(key=lambda x: x['row_count'], reverse=True)
            
            selected_table_names = await self.llm_client.select_tables_by_content(intent, table_analysis[:50])  # Top 50 tables
            
            if not selected_table_names:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found for the question",
                    tables_used=[]
                )
            
            # Get actual table schemas for selected tables
            selected_schemas = self._get_selected_table_schemas(selected_table_names)
            print(f"      ✅ Selected {len(selected_schemas)} tables:")
            for table_name in selected_schemas.keys():
                content = self.table_content_analysis.get(table_name, {}).get('content_summary', 'Unknown')
                print(f"         • {table_name} ({content})")
            
            # Stage 3: Data-Driven Relationship Resolution
            print("   🔗 Stage 3: Data-Driven Relationship Resolution...")
            sample_analysis = {name: self.table_content_analysis.get(name, {}) for name in selected_table_names}
            joins = await self.llm_client.resolve_table_relationships(selected_table_names, selected_schemas, sample_analysis)
            
            print(f"      ✅ Resolved {len(joins)} joins based on data analysis")
            
            # Stage 4: Data-Driven SQL Generation
            print("   ⚡ Stage 4: Data-Driven SQL Generation...")
            sql_query = await self.llm_client.generate_data_driven_sql(intent, selected_schemas, joins, sample_analysis)
            
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
                error=f"Data-driven 4-stage pipeline failed: {str(e)}",
                tables_used=[]
            )
    
    def _get_selected_table_schemas(self, selected_table_names: List[str]) -> Dict[str, Dict]:
        """Get complete schema information for selected tables"""
        
        selected_schemas = {}
        
        for table_name in selected_table_names:
            if table_name in self.database_schema:
                schema_info = self.database_schema[table_name]
                
                # Create comprehensive schema info
                selected_schemas[table_name] = {
                    'columns': schema_info['columns'],
                    'sample_data': schema_info['sample_data'],
                    'row_count': schema_info['row_count'],
                    'object_type': schema_info['object_type']
                }
            else:
                print(f"      ⚠️ Schema not found for table: {table_name}")
        
        return selected_schemas
    
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
                    
                    # Validate results make sense
                    validation_warning = self._validate_results(results, intent, sql_query)
                    
                    return results, validation_warning
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
    
    def _validate_results(self, results: List[Dict], intent: Dict[str, Any], sql_query: str) -> Optional[str]:
        """Validate that results make business sense"""
        
        if not results:
            return "Query returned no results. This might indicate no matching data or incorrect logic."
        
        action = intent.get('action', '').lower()
        result_type = intent.get('result_type', '')
        subject = intent.get('subject', '').lower()
        
        # Validate count operations
        if action == 'count' and result_type == 'single_number':
            if len(results) == 1 and len(results[0]) == 1:
                count_value = list(results[0].values())[0]
                if isinstance(count_value, (int, float)):
                    if count_value == 0:
                        return f"Query returned 0 {subject}. This may indicate no matching data or incorrect table selection/joins."
                    elif 'customer' in subject or 'people' in subject:
                        if count_value > 1000000:  # Suspiciously high
                            return f"Count of {count_value:,} seems very high for {subject}. Please verify the query logic."
        
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
    
    async def _show_content_analysis(self):
        """Show content analysis for debugging"""
        print(f"\n🔍 TABLE CONTENT ANALYSIS:")
        print(f"   📊 Total tables analyzed: {len(self.table_content_analysis)}")
        
        # Show top tables by different criteria
        people_tables = [(name, analysis) for name, analysis in self.table_content_analysis.items() 
                        if analysis['has_people_data']]
        payment_tables = [(name, analysis) for name, analysis in self.table_content_analysis.items() 
                         if analysis['has_payment_data']]
        
        print(f"\n   👥 Tables with people/customer data ({len(people_tables)}):")
        for name, analysis in people_tables[:5]:
            row_count = self.database_schema.get(name, {}).get('row_count', 0)
            print(f"      • {name}: {analysis['content_summary']} ({row_count:,} rows)")
        
        print(f"\n   💰 Tables with payment/financial data ({len(payment_tables)}):")
        for name, analysis in payment_tables[:5]:
            row_count = self.database_schema.get(name, {}).get('row_count', 0)
            print(f"      • {name}: {analysis['content_summary']} ({row_count:,} rows)")
    
    def _show_debug_info(self):
        """Show debug information"""
        print(f"\n🔧 DEBUG INFORMATION:")
        print(f"   📊 Total tables loaded: {len(self.tables)}")
        print(f"   💾 Schema cache loaded: {len(self.database_schema)}")
        print(f"   🔍 Content analysis completed: {len(self.table_content_analysis)}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        # Show content analysis summary
        people_count = sum(1 for analysis in self.table_content_analysis.values() if analysis['has_people_data'])
        payment_count = sum(1 for analysis in self.table_content_analysis.values() if analysis['has_payment_data'])
        
        print(f"\n   📋 Content Analysis Summary:")
        print(f"      • Tables with people data: {people_count}")
        print(f"      • Tables with payment data: {payment_count}")
        print(f"      • Tables with dates: {sum(1 for a in self.table_content_analysis.values() if a['has_date_fields'])}")
        print(f"      • Tables with amounts: {sum(1 for a in self.table_content_analysis.values() if a['has_amount_fields'])}")
    
    def _show_system_capabilities(self):
        """Show data-driven system capabilities"""
        
        people_tables = sum(1 for analysis in self.table_content_analysis.values() if analysis['has_people_data'])
        payment_tables = sum(1 for analysis in self.table_content_analysis.values() if analysis['has_payment_data'])
        date_tables = sum(1 for analysis in self.table_content_analysis.values() if analysis['has_date_fields'])
        
        print(f"✅ DATA-DRIVEN 4-STAGE PIPELINE READY:")
        print(f"   📊 Database: {len(self.database_schema)} objects analyzed")
        print(f"   🔍 Content Analysis: {len(self.table_content_analysis)} tables")
        print(f"   👥 People/Customer Data: {people_tables} tables")
        print(f"   💰 Payment/Financial Data: {payment_tables} tables")
        print(f"   📅 Date/Time Tracking: {date_tables} tables")
        
        if self.domain:
            print(f"   🏢 Domain: {self.domain.domain_type}")
        
        print(f"   🔗 Relationships: {len(self.relationships)} discovered")
        
        print("\n🚀 Data-Driven 4-Stage Process:")
        print("   🎯 Stage 1: Business Intent Analysis (extracts what you're really asking)")
        print("   📋 Stage 2: Content-Based Table Selection (uses actual data content)")  
        print("   🔗 Stage 3: Data-Driven Relationship Resolution (analyzes sample data)")
        print("   ⚡ Stage 4: Context-Aware SQL Generation (uses real column names)")
        print("   💡 Total time: 10-15 seconds with data-driven accuracy")
        
        print("\n💡 Type 'help' for commands, 'examples' for questions, 'analyze' for content analysis")
    
    def _show_help(self):
        """Show help information"""
        
        print("\n🔧 COMMANDS:")
        print("   • 'help' - Show this help")
        print("   • 'examples' - Show sample questions")
        print("   • 'debug' - Show system debug information")
        print("   • 'analyze' - Show table content analysis")
        print("   • 'quit' or 'exit' - Exit pipeline")
        
        print("\n🚀 DATA-DRIVEN PIPELINE FEATURES:")
        print("   • Analyzes actual table content using sample data")
        print("   • No hardcoded entity types - discovers data dynamically")
        print("   • Uses real column names and data patterns")
        print("   • Finds relationships through data analysis")
        print("   • Generates SQL based on actual table structure")
        print("   • Validates results against business logic")
        
        print("\n💡 QUERY TIPS:")
        print("   • Ask about actual data: 'count people who paid in 2025'")
        print("   • Be specific about time: 'payments made this year'")
        print("   • Ask for data that exists: system will find the right tables")
        print("   • Complex questions work: 'customers with recent high-value orders'")
    
    def _show_examples(self):
        """Show example questions based on actual data content"""
        
        people_tables = sum(1 for analysis in self.table_content_analysis.values() if analysis['has_people_data'])
        payment_tables = sum(1 for analysis in self.table_content_analysis.values() if analysis['has_payment_data'])
        
        print("\n💡 SAMPLE QUESTIONS (based on your actual data):")
        
        if people_tables > 0:
            print("   👥 People/Customer Questions:")
            print("      • How many people/customers are in the system?")
            print("      • Show me customer information")
            print("      • List people with contact details")
        
        if payment_tables > 0:
            print("   💰 Payment/Financial Questions:")
            print("      • Count payments made in 2025")
            print("      • What is the total amount of payments?")
            print("      • Show payment transactions")
        
        if people_tables > 0 and payment_tables > 0:
            print("   🔄 Combined Questions:")
            print("      • How many people made payments in 2025?")
            print("      • Show customers with their payments")
            print("      • Count unique people who paid this year")
        
        print("\n🔥 ADVANCED QUESTIONS:")
        print("   • 'Find people who made multiple payments'")
        print("   • 'Show monthly payment trends'")
        print("   • 'List high-value transactions'")
        print("   • 'Customers who haven't paid recently'")
        
        print(f"\n⚡ The pipeline analyzes {len(self.table_content_analysis)} tables to find the right data!")
    
    def _display_result(self, result: QueryResult):
        """Display query result with enhanced formatting"""
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.tables_used:
                print(f"💡 Tables analyzed: {', '.join(result.tables_used)}")
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
                pipeline_stages = "🎯🔍🔗⚡"  # Icons for the 4 stages
                print(f"⚡ Data-driven pipeline time: {result.execution_time:.1f}s {pipeline_stages}")
            
            # Success indicator
            if not result.error and result.results:
                print("✅ Query completed successfully with data-driven table selection")

# For backward compatibility
QueryInterface = DataDrivenQueryInterface
QueryLLMClient = DataDrivenLLMClient