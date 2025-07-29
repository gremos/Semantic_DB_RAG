#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED Interactive Query Interface Module
Superior table selection and SQL generation for business queries
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
from semantic.analysis import SimpleLLMClient

class EnhancedQueryInterface:
    """ENHANCED Interactive Query Interface with superior business logic"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = SimpleLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        
        # Business query patterns for better table selection
        self.business_query_patterns = self._init_business_query_patterns()
    
    def _init_business_query_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize business query patterns for intelligent table selection"""
        return {
            'customer_queries': {
                'keywords': [
                    'customer', 'client', 'account', 'user', 'buyer', 'subscriber',
                    'contact', 'person', 'member', 'businesspoint'
                ],
                'required_entity_types': ['Customer'],
                'preferred_entity_types': ['Customer', 'Order', 'Payment'],
                'avoid_entity_types': ['Support', 'System', 'Reference']
            },
            'payment_queries': {
                'keywords': [
                    'paid', 'payment', 'transaction', 'invoice', 'billing', 'revenue',
                    'money', 'amount', 'financial', 'charge', 'receipt'
                ],
                'required_entity_types': ['Payment'],
                'preferred_entity_types': ['Payment', 'Customer', 'Order', 'Invoice'],
                'avoid_entity_types': ['Support', 'Marketing', 'System']
            },
            'order_queries': {
                'keywords': [
                    'order', 'sale', 'purchase', 'booking', 'reservation',
                    'buy', 'sold', 'transaction'
                ],
                'required_entity_types': ['Order'],
                'preferred_entity_types': ['Order', 'Customer', 'Payment', 'Product'],
                'avoid_entity_types': ['Support', 'System']
            },
            'revenue_queries': {
                'keywords': [
                    'revenue', 'income', 'sales', 'earnings', 'profit',
                    'total', 'sum', 'amount'
                ],
                'required_entity_types': ['Payment', 'Order'],
                'preferred_entity_types': ['Payment', 'Order', 'Customer', 'Invoice'],
                'avoid_entity_types': ['Support', 'Marketing', 'System']
            }
        }
    
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
                                       relationships: List[Relationship]):
        """Start enhanced interactive query session"""
        self.tables = tables
        self.domain = domain
        self.relationships = relationships
        
        # Enhanced system status with entity breakdown
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
                elif not question:
                    continue
                
                query_count += 1
                print(f"🔍 Processing query #{query_count} with ENHANCED business logic...")
                start_time = time.time()
                
                result = await self._process_question_enhanced_business(question)
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
    
    def _show_enhanced_system_status(self):
        """Show enhanced system status with entity breakdown"""
        table_count = sum(1 for t in self.tables if t.object_type == 'BASE TABLE')
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        views_with_data = sum(1 for t in self.tables if t.object_type == 'VIEW' and t.sample_data)
        
        # Entity breakdown
        entity_counts = defaultdict(int)
        for table in self.tables:
            if table.semantic_profile:
                entity_counts[table.semantic_profile.entity_type] += 1
        
        print(f"✅ ENHANCED system ready! Domain: {self.domain.domain_type if self.domain else 'Unknown'}")
        print(f"📊 Available: {views_with_data}/{view_count} views, {table_count} tables")
        
        # Show critical entity status
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
            print("💡 This may affect query accuracy - run semantic analysis again")
        
        print("💡 Type 'help' for examples, 'entities' for entity breakdown, 'quit' to exit")
    
    def _show_enhanced_help(self):
        """Show enhanced help with entity-based examples"""
        print("\n💡 ENHANCED HELP - Business Entity Queries:")
        
        # Show examples based on available entities
        entity_counts = defaultdict(int)
        for table in self.tables:
            if table.semantic_profile:
                entity_counts[table.semantic_profile.entity_type] += 1
        
        if entity_counts.get('Customer', 0) > 0:
            print("\n👥 CUSTOMER QUERIES:")
            print("   • How many customers do we have?")
            print("   • Show me active customers")
            print("   • List customers registered in 2025")
        
        if entity_counts.get('Payment', 0) > 0:
            print("\n💰 PAYMENT QUERIES:")
            print("   • Count total paid customers for 2025")
            print("   • What is our total revenue?")
            print("   • Show recent payments")
        
        if entity_counts.get('Order', 0) > 0:
            print("\n📦 ORDER QUERIES:")
            print("   • How many orders this year?")
            print("   • Show recent orders")
            print("   • What is our average order value?")
        
        if entity_counts.get('Product', 0) > 0:
            print("\n🛍️ PRODUCT QUERIES:")
            print("   • List our top products")
            print("   • Show product categories")
            print("   • What products are most popular?")
        
        print("\n🎯 BUSINESS ANALYSIS:")
        print("   • Total revenue for 2025")
        print("   • Customer growth trends")
        print("   • Monthly sales summary")
        
        print("\n📋 SPECIAL COMMANDS:")
        print("   • 'entities' - Show entity breakdown")
        print("   • 'status' - Show system status")
        print("   • 'help' - Show this help")
    
    def _show_entity_breakdown(self):
        """Show detailed entity breakdown"""
        print("\n📊 ENTITY BREAKDOWN:")
        
        entity_tables = defaultdict(list)
        for table in self.tables:
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                confidence = table.semantic_profile.confidence
                entity_tables[entity_type].append((table.name, confidence))
        
        for entity_type, tables_list in sorted(entity_tables.items()):
            print(f"\n{entity_type.upper()} ({len(tables_list)} tables):")
            # Sort by confidence
            tables_list.sort(key=lambda x: x[1], reverse=True)
            for table_name, confidence in tables_list[:5]:  # Show top 5
                print(f"   • {table_name} (confidence: {confidence:.2f})")
            if len(tables_list) > 5:
                print(f"   ... and {len(tables_list) - 5} more")
    
    async def _process_question_enhanced_business(self, question: str) -> QueryResult:
        """ENHANCED question processing with superior business logic"""
        try:
            # Step 1: Identify query type and required entities
            print(f"   🎯 Analyzing query type for: '{question}'")
            query_type, required_entities = self._identify_query_type(question)
            print(f"   📋 Query type: {query_type}, Required entities: {required_entities}")
            
            # Step 2: Find tables using ENHANCED business logic
            print(f"   🔍 Finding tables with ENHANCED business logic...")
            relevant_tables = self._find_relevant_tables_enhanced_business(question, required_entities)
            
            if not relevant_tables:
                return QueryResult(
                    question=question,
                    relevant_tables=[],
                    sql_query="",
                    results=[],
                    results_count=0,
                    execution_error=f'Could not find relevant {", ".join(required_entities)} tables for this query'
                )
            
            # Step 3: Show selected tables with reasoning
            table_names = [t.name for t in relevant_tables]
            print(f"   📋 Selected tables: {', '.join(table_names)}")
            
            for table in relevant_tables:
                entity_type = table.semantic_profile.entity_type if table.semantic_profile else "Unknown"
                confidence = table.semantic_profile.confidence if table.semantic_profile else 0.0
                data_status = "has data" if table.sample_data else "no data"
                print(f"      • {table.name}: {entity_type} (confidence: {confidence:.2f}, {data_status})")
            
            # Step 4: Generate SQL with ENHANCED business context
            print(f"   🧠 Generating SQL with ENHANCED business context...")
            sql_query = await self._generate_sql_business_enhanced(question, relevant_tables, query_type)
            
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
    
    def _identify_query_type(self, question: str) -> tuple[str, List[str]]:
        """Identify query type and required entity types"""
        question_lower = question.lower()
        
        # Check against business query patterns
        for query_type, pattern in self.business_query_patterns.items():
            if any(keyword in question_lower for keyword in pattern['keywords']):
                return query_type, pattern['required_entity_types']
        
        # Default analysis for unmatched queries
        if any(word in question_lower for word in ['customer', 'client', 'account', 'user']):
            return 'customer_queries', ['Customer']
        elif any(word in question_lower for word in ['paid', 'payment', 'revenue', 'money']):
            return 'payment_queries', ['Payment']
        elif any(word in question_lower for word in ['order', 'sale', 'purchase']):
            return 'order_queries', ['Order']
        else:
            return 'general_queries', ['Customer', 'Payment', 'Order']
    
    def _find_relevant_tables_enhanced_business(self, question: str, required_entities: List[str]) -> List[TableInfo]:
        """ENHANCED table finding with business entity prioritization"""
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_tables = []
        
        for table in self.tables:
            score = 0
            reasons = []
            
            # PRIORITY 1: Required entity types (CRITICAL)
            if table.semantic_profile and table.semantic_profile.entity_type in required_entities:
                score += 1000  # Very high priority for required entities
                confidence = table.semantic_profile.confidence
                score += confidence * 500  # Boost by confidence
                reasons.append(f"required entity: {table.semantic_profile.entity_type} (conf: {confidence:.2f})")
            
            # PRIORITY 2: Entity type relevance for business queries
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                
                # Special business logic for paid customer queries
                if 'paid' in question_lower and 'customer' in question_lower:
                    if entity_type == 'Payment':
                        score += 800  # Payment tables are crucial for paid customer queries
                        reasons.append("payment table for paid customers")
                    elif entity_type == 'Customer':
                        score += 600  # Customer tables needed for customer identification
                        reasons.append("customer table for customer identification")
                    elif entity_type == 'Order':
                        score += 400  # Orders might link customers to payments
                        reasons.append("order table for customer-payment link")
                
                # Revenue/financial queries
                if any(word in question_lower for word in ['revenue', 'total', 'amount', 'money']):
                    if entity_type in ['Payment', 'Invoice', 'Order']:
                        score += 700
                        reasons.append(f"financial entity: {entity_type}")
            
            # PRIORITY 3: Name matching (still important)
            table_name_words = set(re.findall(r'\w+', table.name.lower()))
            name_matches = question_words.intersection(table_name_words)
            if name_matches:
                score += len(name_matches) * 200
                reasons.append(f"name matches: {', '.join(name_matches)}")
            
            # PRIORITY 4: Column relevance
            relevant_columns = 0
            for col in table.columns:
                col_name_lower = col['name'].lower()
                
                # Look for business-critical columns
                if 'paid' in question_lower and any(term in col_name_lower for term in ['amount', 'payment', 'total', 'price']):
                    relevant_columns += 3
                    reasons.append(f"payment column: {col['name']}")
                elif 'customer' in question_lower and any(term in col_name_lower for term in ['customer', 'client', 'account']):
                    relevant_columns += 2
                    reasons.append(f"customer column: {col['name']}")
                elif question_words.intersection(set(col_name_lower.split('_'))):
                    relevant_columns += 1
            
            score += relevant_columns * 50
            
            # PRIORITY 5: Data availability bonus
            if table.sample_data:
                score += 100
                reasons.append("has sample data")
                
                # Check sample data for relevant business indicators
                for row in table.sample_data[:2]:
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and 'paid' in question_lower:
                            if any(term in key.lower() for term in ['amount', 'total', 'price', 'payment']):
                                score += 150
                                reasons.append("contains monetary data")
                                break
            
            # PENALTY: Avoid irrelevant entity types
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                if entity_type in ['System', 'Reference'] and 'paid customer' in question_lower:
                    score -= 200  # Penalize system/reference tables for business queries
                    reasons.append(f"penalized: {entity_type} table")
            
            if score > 0:
                scored_tables.append((table, score, reasons))
        
        # Sort by score and return top tables
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure we have required entity types
        selected_tables = []
        required_entity_found = {entity: False for entity in required_entities}
        
        # First pass: Get tables of required entity types
        for table, score, reasons in scored_tables:
            if table.semantic_profile and table.semantic_profile.entity_type in required_entities:
                selected_tables.append(table)
                required_entity_found[table.semantic_profile.entity_type] = True
                if len(selected_tables) >= 5:  # Limit to top 5
                    break
        
        # Second pass: Add high-scoring tables if we need more
        if len(selected_tables) < 3:
            for table, score, reasons in scored_tables[:10]:
                if table not in selected_tables:
                    selected_tables.append(table)
                    if len(selected_tables) >= 3:
                        break
        
        # Check if we found required entities
        missing_entities = [entity for entity, found in required_entity_found.items() if not found]
        if missing_entities:
            print(f"   ⚠️  Missing required entities: {', '.join(missing_entities)}")
            print(f"       Query results may be inaccurate!")
        
        return selected_tables[:5]  # Return top 5 tables
    
    async def _generate_sql_business_enhanced(self, question: str, tables: List[TableInfo], query_type: str) -> Optional[str]:
        """Generate SQL with ENHANCED business context and logic"""
        
        # Prepare detailed table information for LLM
        table_info = []
        for table in tables:
            # More comprehensive table analysis
            columns_detail = []
            for col in table.columns[:20]:  # Include more columns
                col_detail = {
                    'name': col['name'],
                    'type': col['data_type'],
                    'nullable': col['nullable']
                }
                columns_detail.append(col_detail)
            
            info = {
                'name': table.name,
                'schema': table.schema,
                'full_name': table.full_name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': columns_detail,
                'sample_data': table.sample_data[:2] if table.sample_data else [],
                'entity_type': table.semantic_profile.entity_type if table.semantic_profile else 'Unknown',
                'business_role': table.semantic_profile.business_role if table.semantic_profile else 'Unknown',
                'confidence': table.semantic_profile.confidence if table.semantic_profile else 0.0
            }
            table_info.append(info)
        
        # Enhanced business context
        domain_context = f"{self.domain.domain_type} system" if self.domain else "Business system"
        
        # Special handling for paid customer queries
        if 'paid' in question.lower() and 'customer' in question.lower():
            business_context = """
CRITICAL: This is a PAID CUSTOMER query. You must:
1. Find tables that contain PAYMENT/TRANSACTION data (entity_type: Payment)
2. Find tables that contain CUSTOMER data (entity_type: Customer)  
3. JOIN these tables to identify customers who have made payments
4. Count DISTINCT customers who have payment records
5. Apply date filters if specified (e.g., year 2025)

PAID CUSTOMER LOGIC:
- A paid customer is someone who appears in payment/transaction records
- Look for tables with monetary amounts, payment dates, transaction IDs
- JOIN customer tables with payment tables on customer ID
- Use DISTINCT COUNT to avoid counting same customer multiple times
"""
        else:
            business_context = f"Business query in {domain_context} context."
        
        prompt = f"""
You are an expert SQL developer for business systems. Generate a precise SQL query for this business question.

BUSINESS CONTEXT:
{business_context}

Question: "{question}"
Query Type: {query_type}
Domain: {domain_context}

AVAILABLE TABLES WITH BUSINESS CONTEXT:
{json.dumps(table_info, indent=2, default=str)}

SQL GENERATION RULES:
1. Use proper SQL Server syntax with [schema].[table] format
2. For PAID CUSTOMER queries:
   - Join Customer and Payment tables
   - Count DISTINCT customers who have payment records
   - Apply appropriate date filters
3. For REVENUE queries: Sum amounts from Payment/Invoice tables
4. For COUNT queries: Use COUNT(*) or COUNT(DISTINCT) as appropriate
5. Apply date filters correctly (e.g., >= '2025-01-01' AND < '2026-01-01')
6. Use meaningful aliases and column names
7. Include TOP clause for large result sets
8. Handle NULL values appropriately

BUSINESS LOGIC PATTERNS:
- Customer identification: Use CustomerID, ClientID, BusinessPointID, AccountID
- Payment identification: Use PaymentID, TransactionID, InvoiceID, Amount columns
- Date filtering: Use PaymentDate, TransactionDate, CreatedDate, CompletedDate
- Monetary amounts: Sum Amount, Total, Price, Value columns

IMPORTANT: 
- Only use columns that exist in the provided table schemas
- Generate syntactically correct SQL Server T-SQL
- Focus on business logic accuracy over complexity

Respond with ONLY the SQL query, no explanations:
"""
        
        try:
            system_message = """You are a senior SQL developer specializing in business intelligence queries. 
Generate accurate, business-focused SQL queries that answer the specific question asked.
Pay special attention to paid customer queries - these require joining customer and payment data.
Respond with only valid SQL Server T-SQL syntax."""
            
            response = await self.llm.ask(prompt, system_message)
            cleaned_sql = clean_sql_response(response)
            return cleaned_sql
                
        except Exception as e:
            print(f"⚠️ Enhanced SQL generation failed: {e}")
            return None
    
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
    
    def _display_enhanced_query_result(self, result: QueryResult, query_number: int):
        """Display enhanced query result with business context"""
        if result.execution_error:
            print(f"❌ Error: {result.execution_error}")
            
            # Provide helpful suggestions for common errors
            if "not found" in result.execution_error.lower():
                print("💡 Suggestions:")
                print("   • Check if the required entity tables exist")
                print("   • Run semantic analysis again to improve entity recognition")
                print("   • Try a simpler query to test table access")
        else:
            print(f"📋 Generated SQL:")
            print(f"   {result.sql_query}")
            
            count = result.results_count
            print(f"📊 Results: {count} rows")
            
            if result.results:
                # Enhanced result display
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
                if count == 0 and 'paid customer' in result.question.lower():
                    print("\n💡 BUSINESS ANALYSIS:")
                    print("   ❌ No paid customers found - this suggests:")
                    print("      • Payment tables may not be properly identified")
                    print("      • Customer-Payment relationships may be incorrect")
                    print("      • Date filters may be too restrictive")
                    print("      • Table joins may need adjustment")
                    print("\n🔧 RECOMMENDED ACTIONS:")
                    print("   1. Type 'entities' to verify Payment and Customer tables were found")
                    print("   2. Check if payment data exists for the specified period")
                    print("   3. Try a simpler query like 'how many payments in 2025'")
            
            # Show execution time
            if result.execution_time > 0:
                print(f"⚡ Execution time: {result.execution_time:.3f}s")
            
            # Show tables used
            if result.relevant_tables:
                print(f"📋 Used tables: {', '.join(result.relevant_tables)}")

# Update the main query interface class
QueryInterface = EnhancedQueryInterface