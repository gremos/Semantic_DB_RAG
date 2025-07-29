#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED Query Interface - Smart Business Query Processing
Focuses on accurate business query understanding and SQL generation
"""

import pyodbc
import time
import re
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import safe_database_value, clean_sql_response, extract_json_from_response

# Interactive LLM client for query interface
class InteractiveLLMClient:
    """LLM client specifically for interactive query operations"""
    
    def __init__(self, config: Config):
        from langchain_openai import AzureChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,
            request_timeout=120
        )
    
    async def ask(self, prompt: str, system_message: str = "You are a helpful database expert.") -> str:
        """Ask LLM a question with retry logic"""
        from langchain.schema import HumanMessage, SystemMessage
        import asyncio
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(self.llm.invoke, messages)
                return response.content
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

class SmartBusinessQueryProcessor:
    """Smart processor that understands business queries and generates accurate SQL"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = InteractiveLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.business_analysis: Dict[str, Any] = {}
        
        # Business query intelligence
        self.business_query_patterns = self._init_business_query_intelligence()
    
    def _init_business_query_intelligence(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intelligent business query processing patterns"""
        return {
            'paid_customers': {
                'keywords': ['paid', 'customer', 'payment', 'transaction'],
                'intent': 'Find customers who have made payments',
                'required_entities': ['Customer', 'Payment'],
                'sql_strategy': 'customer_payment_join',
                'business_logic': 'Join customer tables with payment tables on customer ID'
            },
            'customer_count': {
                'keywords': ['how many customers', 'count customers', 'number of customers'],
                'intent': 'Count total customers',
                'required_entities': ['Customer'],
                'sql_strategy': 'simple_count',
                'business_logic': 'Count records in primary customer table'
            },
            'revenue_total': {
                'keywords': ['total revenue', 'revenue', 'income', 'earnings'],
                'intent': 'Calculate total revenue',
                'required_entities': ['Payment', 'Order'],
                'sql_strategy': 'sum_amounts',
                'business_logic': 'Sum payment amounts or order totals'
            },
            'order_count': {
                'keywords': ['how many orders', 'count orders', 'number of orders'],
                'intent': 'Count total orders',
                'required_entities': ['Order'],
                'sql_strategy': 'simple_count',
                'business_logic': 'Count records in primary order table'
            }
        }
    
    def analyze_business_query(self, question: str) -> Dict[str, Any]:
        """Analyze business query to understand intent and requirements"""
        question_lower = question.lower()
        
        # Identify query pattern
        for pattern_name, pattern_info in self.business_query_patterns.items():
            if all(keyword in question_lower for keyword in pattern_info['keywords'][:2]):  # Match at least 2 keywords
                return {
                    'pattern': pattern_name,
                    'intent': pattern_info['intent'],
                    'required_entities': pattern_info['required_entities'],
                    'sql_strategy': pattern_info['sql_strategy'],
                    'business_logic': pattern_info['business_logic']
                }
        
        # Default analysis for unmatched queries
        if any(word in question_lower for word in ['customer', 'client']):
            required_entities = ['Customer']
            if any(word in question_lower for word in ['paid', 'payment', 'revenue']):
                required_entities.append('Payment')
        elif any(word in question_lower for word in ['payment', 'revenue', 'income']):
            required_entities = ['Payment']
        elif any(word in question_lower for word in ['order', 'sale']):
            required_entities = ['Order']
        else:
            required_entities = ['Customer', 'Payment', 'Order']
        
        return {
            'pattern': 'general_business',
            'intent': 'General business query',
            'required_entities': required_entities,
            'sql_strategy': 'adaptive',
            'business_logic': 'Determine based on available data'
        }
    
    def find_relevant_tables_smart(self, query_analysis: Dict[str, Any]) -> List[TableInfo]:
        """Smart table selection based on business analysis and relationships"""
        
        required_entities = query_analysis['required_entities']
        pattern = query_analysis['pattern']
        
        # Get tables by entity type from business analysis
        entity_tables = defaultdict(list)
        for table in self.tables:
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                confidence = table.semantic_profile.confidence
                
                # Only include high-confidence tables for critical queries
                if confidence >= 0.7 or pattern == 'general_business':
                    entity_tables[entity_type].append((table, confidence))
        
        selected_tables = []
        
        # Select best tables for each required entity
        for entity_type in required_entities:
            tables_for_entity = entity_tables.get(entity_type, [])
            if tables_for_entity:
                # Sort by confidence and data availability
                tables_for_entity.sort(key=lambda x: (x[1], len(x[0].sample_data)), reverse=True)
                
                # Select top 2 tables for this entity
                for table, confidence in tables_for_entity[:2]:
                    if table not in selected_tables:
                        selected_tables.append(table)
        
        # For paid customer queries, ensure we have both customer and payment tables
        if pattern == 'paid_customers':
            customer_tables = [t for t in selected_tables if t.semantic_profile.entity_type == 'Customer']
            payment_tables = [t for t in selected_tables if t.semantic_profile.entity_type == 'Payment']
            
            if not customer_tables or not payment_tables:
                print(f"   ⚠️ Paid customer query missing required tables!")
                print(f"      Customer tables: {len(customer_tables)}, Payment tables: {len(payment_tables)}")
                
                # Try to find additional tables
                for table in self.tables:
                    if table.semantic_profile:
                        if table.semantic_profile.entity_type == 'Customer' and not customer_tables:
                            selected_tables.append(table)
                        elif table.semantic_profile.entity_type == 'Payment' and not payment_tables:
                            selected_tables.append(table)
        
        # Add related tables based on relationships
        related_tables = self._find_related_tables(selected_tables)
        for table in related_tables:
            if table not in selected_tables and len(selected_tables) < 5:
                selected_tables.append(table)
        
        return selected_tables[:5]  # Limit to top 5 tables
    
    def _find_related_tables(self, primary_tables: List[TableInfo]) -> List[TableInfo]:
        """Find tables related to primary tables through relationships"""
        related_tables = []
        
        primary_table_names = {table.full_name for table in primary_tables}
        
        for relationship in self.relationships:
            # If one table is in primary, add the other
            if relationship.from_table in primary_table_names:
                related_table = self._find_table_by_full_name(relationship.to_table)
                if related_table and related_table not in related_tables:
                    related_tables.append(related_table)
            elif relationship.to_table in primary_table_names:
                related_table = self._find_table_by_full_name(relationship.from_table)
                if related_table and related_table not in related_tables:
                    related_tables.append(related_table)
        
        return related_tables
    
    def _find_table_by_full_name(self, full_name: str) -> Optional[TableInfo]:
        """Find table by full name"""
        for table in self.tables:
            if table.full_name == full_name:
                return table
        return None
    
    async def generate_smart_sql(self, question: str, query_analysis: Dict[str, Any], 
                               relevant_tables: List[TableInfo]) -> Optional[str]:
        """Generate SQL using smart business understanding"""
        
        pattern = query_analysis['pattern']
        sql_strategy = query_analysis['sql_strategy']
        
        # Prepare enhanced table context for LLM
        table_context = []
        for table in relevant_tables:
            context = {
                'table_name': table.name,
                'full_name': table.full_name,
                'entity_type': table.semantic_profile.entity_type if table.semantic_profile else 'Unknown',
                'business_purpose': table.semantic_profile.primary_purpose if table.semantic_profile else 'Unknown',
                'confidence': table.semantic_profile.confidence if table.semantic_profile else 0.0,
                'columns': [
                    {
                        'name': col['name'],
                        'type': col['data_type'],
                        'is_id': col['name'].lower().endswith('id'),
                        'is_amount': any(word in col['name'].lower() for word in ['amount', 'total', 'price', 'cost', 'value']),
                        'is_date': any(word in col['name'].lower() for word in ['date', 'time', 'created', 'modified'])
                    } for col in table.columns[:15]
                ],
                'sample_data': table.sample_data[:2] if table.sample_data else []
            }
            table_context.append(context)
        
        # Get relationship context
        relationship_context = []
        table_names = {table.full_name for table in relevant_tables}
        for rel in self.relationships:
            if rel.from_table in table_names and rel.to_table in table_names:
                relationship_context.append({
                    'from_table': rel.from_table,
                    'to_table': rel.to_table,
                    'link_column': rel.column,
                    'relationship_type': rel.relationship_type,
                    'description': rel.description
                })
        
        # Create specialized prompt based on query pattern
        if pattern == 'paid_customers':
            prompt = self._create_paid_customers_prompt(question, table_context, relationship_context)
        elif pattern == 'customer_count':
            prompt = self._create_customer_count_prompt(question, table_context)
        elif pattern == 'revenue_total':
            prompt = self._create_revenue_prompt(question, table_context)
        else:
            prompt = self._create_general_prompt(question, table_context, relationship_context, query_analysis)
        
        try:
            system_message = """You are an expert SQL developer specializing in business queries. 
Generate accurate SQL Server T-SQL that correctly answers business questions.
Focus on proper table joins and business logic.
Respond with ONLY the SQL query - no explanations."""
            
            response = await self.llm.ask(prompt, system_message)
            cleaned_sql = clean_sql_response(response)
            return cleaned_sql
            
        except Exception as e:
            print(f"⚠️ Smart SQL generation failed: {e}")
            return None
    
    def _create_paid_customers_prompt(self, question: str, table_context: List[Dict], 
                                    relationship_context: List[Dict]) -> str:
        """Create specialized prompt for paid customer queries"""
        
        customer_tables = [t for t in table_context if t['entity_type'] == 'Customer']
        payment_tables = [t for t in table_context if t['entity_type'] == 'Payment']
        
        prompt = f"""
BUSINESS QUERY: "{question}"

This is a PAID CUSTOMER query. You need to find customers who have made payments.

CUSTOMER TABLES:
{json.dumps(customer_tables, indent=2)}

PAYMENT TABLES:  
{json.dumps(payment_tables, indent=2)}

RELATIONSHIPS:
{json.dumps(relationship_context, indent=2)}

PAID CUSTOMER SQL RULES:
1. JOIN customer tables with payment tables
2. Use DISTINCT COUNT to avoid counting same customer multiple times
3. Look for common ID columns (CustomerID, ClientID, BusinessPointID, AccountID)
4. Filter by payment amounts > 0 or payment status = 'completed'
5. Apply date filters if specified (e.g., year 2025)
6. Return COUNT of DISTINCT customers who have payment records

SQL PATTERN FOR PAID CUSTOMERS:
```sql
SELECT COUNT(DISTINCT c.CustomerID) as PaidCustomerCount
FROM CustomerTable c
INNER JOIN PaymentTable p ON c.CustomerID = p.CustomerID  
WHERE p.Amount > 0 
  AND YEAR(p.PaymentDate) = 2025  -- if year specified
```

Generate the SQL query for this specific database:
"""
        return prompt
    
    def _create_customer_count_prompt(self, question: str, table_context: List[Dict]) -> str:
        """Create prompt for customer count queries"""
        
        customer_tables = [t for t in table_context if t['entity_type'] == 'Customer']
        
        prompt = f"""
BUSINESS QUERY: "{question}"

This is a CUSTOMER COUNT query.

CUSTOMER TABLES:
{json.dumps(customer_tables, indent=2)}

Generate SQL to count total customers:
- Use the primary customer table (highest confidence)
- Use COUNT(*) or COUNT(DISTINCT CustomerID) 
- Apply any date filters if mentioned
- Return simple count result

Generate the SQL query:
"""
        return prompt
    
    def _create_revenue_prompt(self, question: str, table_context: List[Dict]) -> str:
        """Create prompt for revenue queries"""
        
        payment_tables = [t for t in table_context if t['entity_type'] in ['Payment', 'Order']]
        
        prompt = f"""
BUSINESS QUERY: "{question}"

This is a REVENUE/INCOME query.

PAYMENT/ORDER TABLES:
{json.dumps(payment_tables, indent=2)}

Generate SQL to calculate revenue:
- SUM payment amounts or order totals
- Look for columns like Amount, Total, Price, Value
- Apply date filters if specified
- Handle NULL values appropriately

Generate the SQL query:
"""
        return prompt
    
    def _create_general_prompt(self, question: str, table_context: List[Dict], 
                             relationship_context: List[Dict], query_analysis: Dict) -> str:
        """Create general prompt for other business queries"""
        
        prompt = f"""
BUSINESS QUERY: "{question}"

QUERY ANALYSIS:
- Intent: {query_analysis['intent']}
- Required Entities: {query_analysis['required_entities']}
- Business Logic: {query_analysis['business_logic']}

AVAILABLE TABLES:
{json.dumps(table_context, indent=2)}

TABLE RELATIONSHIPS:
{json.dumps(relationship_context, indent=2)}

BUSINESS SQL RULES:
1. Use proper table joins based on relationships
2. Apply appropriate filters for business logic
3. Use meaningful column aliases
4. Handle date filters correctly
5. Limit results if needed (TOP 100)

Generate accurate SQL Server T-SQL query:
"""
        return prompt


class EnhancedQueryInterface:
    """Enhanced query interface with smart business query processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.query_processor = SmartBusinessQueryProcessor(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.business_analysis: Dict[str, Any] = {}
    
    def get_database_connection(self):
        """Get database connection for query execution"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # Set connection to handle Unicode properly
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
        return conn
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                       domain: Optional[BusinessDomain], 
                                       relationships: List[Relationship],
                                       business_analysis: Dict[str, Any] = None):
        """Start enhanced interactive query session"""
        self.tables = tables
        self.domain = domain
        self.relationships = relationships
        self.business_analysis = business_analysis or {}
        
        # Configure query processor
        self.query_processor.tables = tables
        self.query_processor.domain = domain
        self.query_processor.relationships = relationships
        self.query_processor.business_analysis = business_analysis
        
        # Enhanced system status
        self._show_enhanced_system_status()
        
        query_count = 0
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    self._show_enhanced_help()
                    continue
                elif question.lower() == 'status':
                    self._show_system_status()
                    continue
                elif question.lower() == 'entities':
                    self._show_entity_breakdown()
                    continue
                elif question.lower() == 'relationships':
                    self._show_relationships()
                    continue
                elif not question:
                    continue
                
                query_count += 1
                print(f"🔍 Processing query #{query_count} with SMART business logic...")
                start_time = time.time()
                
                result = await self._process_question_smart(question)
                elapsed = time.time() - start_time
                
                print(f"⏱️ Completed in {elapsed:.1f}s")
                print("-" * 50)
                
                self._display_enhanced_query_result(result, query_count)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Session summary: {query_count} queries processed")
        print("👋 Thanks for using the ENHANCED Semantic Database RAG System!")
    
    async def _process_question_smart(self, question: str) -> QueryResult:
        """Process question using smart business logic"""
        try:
            # Step 1: Analyze business query
            print(f"   🎯 Analyzing business intent...")
            query_analysis = self.query_processor.analyze_business_query(question)
            print(f"   📋 Intent: {query_analysis['intent']}")
            print(f"   📋 Required entities: {query_analysis['required_entities']}")
            
            # Step 2: Find relevant tables using smart selection
            print(f"   🔍 Smart table selection...")
            relevant_tables = self.query_processor.find_relevant_tables_smart(query_analysis)
            
            if not relevant_tables:
                return QueryResult(
                    question=question,
                    relevant_tables=[],
                    sql_query="",
                    results=[],
                    results_count=0,
                    execution_error=f'Could not find relevant tables for: {", ".join(query_analysis["required_entities"])}'
                )
            
            # Step 3: Show selected tables with business context
            table_names = [t.name for t in relevant_tables]
            print(f"   📋 Selected tables: {', '.join(table_names)}")
            
            for table in relevant_tables:
                entity_type = table.semantic_profile.entity_type if table.semantic_profile else "Unknown"
                confidence = table.semantic_profile.confidence if table.semantic_profile else 0.0
                data_status = "has data" if table.sample_data else "no data"
                print(f"      • {table.name}: {entity_type} (confidence: {confidence:.2f}, {data_status})")
            
            # Step 4: Generate SQL using smart business logic
            print(f"   🧠 Generating SQL with smart business understanding...")
            sql_query = await self.query_processor.generate_smart_sql(question, query_analysis, relevant_tables)
            
            if not sql_query:
                return QueryResult(
                    question=question,
                    relevant_tables=table_names,
                    sql_query="",
                    results=[],
                    results_count=0,
                    execution_error='Could not generate appropriate SQL query'
                )
            
            print(f"   ⚡ Generated query: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
            
            # Step 5: Execute SQL
            print(f"   💾 Executing query...")
            start_time = time.time()
            execution_result = await self._execute_sql_enhanced(sql_query)
            execution_time = time.time() - start_time
            
            if execution_result.get('error'):
                print(f"   ❌ SQL execution failed: {execution_result.get('error')}")
            else:
                print(f"   ✅ Query executed successfully")
            
            return QueryResult(
                question=question,
                relevant_tables=table_names,
                sql_query=sql_query,
                results=execution_result.get('data', []),
                results_count=execution_result.get('count', 0),
                execution_error=execution_result.get('error'),
                execution_time=execution_time
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                relevant_tables=[],
                sql_query="",
                results=[],
                results_count=0,
                execution_error=f'Processing failed: {str(e)}'
            )
    
    async def _execute_sql_enhanced(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL with enhanced error handling"""
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                if not cursor.description:
                    return {'data': [], 'count': 0, 'error': None}
                
                columns = [col[0] for col in cursor.description]
                results = []
                
                for i, row in enumerate(cursor.fetchall()):
                    if i >= self.config.max_results:
                        break
                    
                    row_data = {}
                    for j, value in enumerate(row):
                        if j < len(columns):
                            row_data[columns[j]] = safe_database_value(value)
                    results.append(row_data)
                
                return {'data': results, 'count': len(results), 'error': None}
                
        except Exception as e:
            error_msg = str(e)
            
            # Enhanced error classification
            if 'invalid object name' in error_msg.lower():
                error_msg = f"Table/view not found: {error_msg[:100]}"
            elif 'invalid column name' in error_msg.lower():
                error_msg = f"Column not found: {error_msg[:100]}"
            elif 'timeout' in error_msg.lower():
                error_msg = f"Query timeout: {error_msg[:100]}"
            else:
                error_msg = error_msg[:150]
            
            return {'data': [], 'count': 0, 'error': error_msg}
    
    def _show_enhanced_system_status(self):
        """Show enhanced system status with business context"""
        table_count = sum(1 for t in self.tables if t.object_type == 'BASE TABLE')
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        views_with_data = sum(1 for t in self.tables if t.object_type == 'VIEW' and t.sample_data)
        
        # Entity breakdown from business analysis
        validation_results = self.business_analysis.get('validation_results', {})
        entity_counts = validation_results.get('entity_counts', {})
        
        print(f"✅ ENHANCED system ready! Domain: {self.domain.domain_type if self.domain else 'Unknown'}")
        print(f"📊 Available: {views_with_data}/{view_count} views, {table_count} tables")
        
        # Show business entity status
        core_entities = ['Customer', 'Payment', 'Order', 'Product', 'Invoice']
        core_status = []
        missing_core = []
        
        for entity in core_entities:
            count = entity_counts.get(entity, 0)
            if count > 0:
                core_status.append(f"{entity}: {count}")
            else:
                missing_core.append(entity)
        
        if core_status:
            print(f"🎯 Core entities: {', '.join(core_status)}")
        
        if missing_core:
            print(f"⚠️  Missing entities: {', '.join(missing_core)}")
        
        # Show relationship status
        customer_payment_links = validation_results.get('customer_payment_links', 0)
        if customer_payment_links > 0:
            print(f"🔗 Customer-payment links: {customer_payment_links}")
        else:
            print(f"❌ No customer-payment relationships found!")
            print(f"   This may cause 'paid customer' queries to fail")
        
        print("💡 Type 'help' for examples, 'entities' for breakdown, 'relationships' for links")
    
    def _show_enhanced_help(self):
        """Show enhanced help with smart query examples"""
        validation_results = self.business_analysis.get('validation_results', {})
        entity_counts = validation_results.get('entity_counts', {})
        
        print("\n💡 ENHANCED HELP - Smart Business Queries:")
        
        if entity_counts.get('Customer', 0) > 0:
            print("\n👥 CUSTOMER QUERIES:")
            print("   • How many customers do we have?")
            print("   • Show me customer details")
            print("   • List customers by registration date")
            
            if entity_counts.get('Payment', 0) > 0:
                print("   • How many customers have paid? (SMART: joins customer+payment tables)")
                print("   • Count paid customers for 2025")
        
        if entity_counts.get('Payment', 0) > 0:
            print("\n💰 PAYMENT QUERIES:")
            print("   • What is our total revenue?")
            print("   • Show recent payments")
            print("   • Count payments this year")
        
        if entity_counts.get('Order', 0) > 0:
            print("\n📦 ORDER QUERIES:")
            print("   • How many orders this year?")
            print("   • Show recent orders")
            print("   • What is our average order value?")
        
        print("\n🧠 SMART FEATURES:")
        print("   • Automatic entity detection")
        print("   • Intelligent table relationships")
        print("   • Business-focused SQL generation")
        print("   • Enhanced error handling")
        
        print("\n📋 COMMANDS:")
        print("   • 'entities' - Show entity breakdown")
        print("   • 'relationships' - Show table relationships")
        print("   • 'status' - Show system status")
    
    def _show_entity_breakdown(self):
        """Show detailed entity breakdown with business context"""
        validation_results = self.business_analysis.get('validation_results', {})
        high_confidence = validation_results.get('high_confidence_entities', {})
        
        print("\n📊 BUSINESS ENTITY BREAKDOWN:")
        
        for entity_type, tables_list in high_confidence.items():
            if tables_list:
                print(f"\n{entity_type.upper()} ({len(tables_list)} tables):")
                for table_info in tables_list[:5]:  # Show top 5
                    name = table_info['name']
                    confidence = table_info['confidence']
                    purpose = table_info.get('purpose', 'No description')[:60]
                    print(f"   • {name} (confidence: {confidence:.2f}) - {purpose}")
                if len(tables_list) > 5:
                    print(f"   ... and {len(tables_list) - 5} more")
    
    def _show_relationships(self):
        """Show discovered relationships"""
        print("\n🔗 DISCOVERED RELATIONSHIPS:")
        
        if not self.relationships:
            print("   No relationships found")
            return
        
        # Group by relationship type
        by_type = defaultdict(list)
        for rel in self.relationships:
            by_type[rel.relationship_type].append(rel)
        
        for rel_type, rels in by_type.items():
            print(f"\n{rel_type.upper()}:")
            for rel in rels[:5]:  # Show top 5
                print(f"   • {rel.from_table} → {rel.to_table}")
                print(f"     via {rel.column} (confidence: {rel.confidence:.2f})")
            if len(rels) > 5:
                print(f"   ... and {len(rels) - 5} more")
    
    def _show_system_status(self):
        """Show system status"""
        self._show_enhanced_system_status()
    
    def _display_enhanced_query_result(self, result: QueryResult, query_number: int):
        """Display enhanced query result with business interpretation"""
        if result.execution_error:
            print(f"❌ Error: {result.execution_error}")
            
            # Provide helpful business context for errors
            if "not found" in result.execution_error.lower():
                print("💡 Business Analysis:")
                print("   • Tables may not be properly classified")
                print("   • Run semantic analysis again for better entity recognition")
                print("   • Check if required business entities were identified")
        else:
            print(f"📋 Generated SQL:")
            print(f"   {result.sql_query}")
            
            count = result.results_count
            print(f"📊 Results: {count} rows")
            
            if result.results:
                # Enhanced result display with business context
                for i, row in enumerate(result.results[:5], 1):
                    display_row = {}
                    for key, value in list(row.items())[:6]:
                        if isinstance(value, str) and len(value) > 30:
                            display_row[key] = value[:30] + "..."
                        else:
                            display_row[key] = value
                    print(f"   {i}. {display_row}")
                
                if count > 5:
                    print(f"   ... and {count - 5} more rows")
                
                # Business interpretation of results
                self._provide_business_interpretation(result)
            
            # Show execution time
            if result.execution_time > 0:
                print(f"⚡ Execution time: {result.execution_time:.3f}s")
            
            # Show tables used
            if result.relevant_tables:
                print(f"📋 Used tables: {', '.join(result.relevant_tables)}")
    
    def _provide_business_interpretation(self, result: QueryResult):
        """Provide business interpretation of query results"""
        question_lower = result.question.lower()
        count = result.results_count
        
        if 'paid customer' in question_lower and count == 0:
            print("\n💡 BUSINESS INTERPRETATION:")
            print("   ❌ Zero paid customers found - This suggests:")
            print("      • Customer and payment tables may not be properly linked")
            print("      • Payment data may not exist for the specified period")
            print("      • Table relationships need verification")
            
            print("\n🔧 RECOMMENDED ACTIONS:")
            print("   1. Type 'relationships' to verify customer-payment links")
            print("   2. Check payment data: 'Show me recent payments'")
            print("   3. Verify customer data: 'How many customers do we have?'")
            
        elif 'customer' in question_lower and count > 0:
            print(f"\n💡 BUSINESS INTERPRETATION:")
            print(f"   ✅ Found {count} customers - Business looks healthy!")
            
        elif 'revenue' in question_lower or 'payment' in question_lower:
            if count > 0 and result.results:
                # Try to extract monetary values
                first_result = result.results[0]
                for key, value in first_result.items():
                    if isinstance(value, (int, float)) and value > 0:
                        print(f"\n💡 BUSINESS INTERPRETATION:")
                        print(f"   💰 {key}: {value:,.2f}")
                        break


# Update the main query interface class
QueryInterface = EnhancedQueryInterface