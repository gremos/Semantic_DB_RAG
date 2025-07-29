#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Query Interface Module
Handles natural language to SQL conversion and query execution
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

class QueryInterface:
    """Enhanced Interactive Query Interface with improved SQL generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = SimpleLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
    
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
        """Start interactive query session"""
        self.tables = tables
        self.domain = domain
        self.relationships = relationships
        
        # Enhanced system status
        table_count = sum(1 for t in tables if t.object_type == 'BASE TABLE')
        view_count = sum(1 for t in tables if t.object_type == 'VIEW')
        classified_count = sum(1 for t in tables if t.semantic_profile)
        views_available = sum(1 for t in tables if t.object_type == 'VIEW' and t.sample_data)
        
        print(f"✅ Enhanced system ready! Domain: {domain.domain_type if domain else 'Unknown'}")
        print(f"📊 Available for queries: {views_available}/{view_count} views (improved), {table_count} tables")
        print("💡 Type 'help' for examples, 'status' for system info, 'quit' to exit")
        
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
                elif not question:
                    continue
                
                query_count += 1
                print(f"🔍 Processing query #{query_count} with enhanced system...")
                start_time = time.time()
                
                result = await self._process_question_enhanced(question)
                elapsed = time.time() - start_time
                
                print(f"⏱️ Completed in {elapsed:.1f}s")
                print("-" * 50)
                
                self._display_query_result(result, query_count)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Session summary: {query_count} queries processed")
        print("👋 Thanks for using the Enhanced Semantic Database RAG System!")
    
    def _show_enhanced_help(self):
        """Show enhanced help with better examples"""
        print("\n💡 ENHANCED HELP - Example Questions:")
        
        if self.domain and self.domain.sample_questions:
            print("🎯 Suggested questions for your system:")
            for i, q in enumerate(self.domain.sample_questions[:5], 1):
                print(f"   {i}. {q}")
        
        print("\n🔧 General examples:")
        print("   • How many customers do we have?")
        print("   • Show me recent orders")
        print("   • What are our top products?")
        print("   • List active users")
        print("   • Show data from the last month")
        
        if self.domain:
            print(f"\n🏢 Your system context:")
            print(f"   Domain: {self.domain.domain_type}")
            print(f"   Industry: {self.domain.industry}")
            if self.domain.customer_definition:
                print(f"   Customer info: {self.domain.customer_definition}")
        
        print("\n📝 Tips:")
        print("   • Be specific about what data you want")
        print("   • Mention time ranges if relevant") 
        print("   • Use business terms from your domain")
        print("   • Ask about counts, totals, or recent activity")
    
    def _show_system_status(self):
        """Show current system status"""
        print("\n📊 CURRENT SYSTEM STATUS")
        print("=" * 40)
        
        if self.tables:
            table_count = sum(1 for t in self.tables if t.object_type == 'BASE TABLE')
            view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
            views_with_data = sum(1 for t in self.tables if t.object_type == 'VIEW' and t.sample_data)
            
            print(f"📋 Available Objects: {len(self.tables)}")
            print(f"   Tables: {table_count}")
            print(f"   Views: {view_count} ({views_with_data} with data)")
            
            classified_count = sum(1 for t in self.tables if t.semantic_profile)
            print(f"   🧠 Classified: {classified_count}")
        
        if self.domain:
            print(f"\n🏢 Business Context:")
            print(f"   Domain: {self.domain.domain_type}")
            print(f"   Industry: {self.domain.industry}")
            print(f"   Confidence: {self.domain.confidence:.2f}")
        
        if self.relationships:
            print(f"\n🔗 Relationships: {len(self.relationships)}")
    
    async def _process_question_enhanced(self, question: str) -> QueryResult:
        """Enhanced question processing with better table selection and SQL generation"""
        try:
            # Enhanced table relevance finding
            print(f"   🔍 Finding relevant objects (including views) for: '{question}'")
            relevant_tables = self._find_relevant_tables_enhanced(question)
            
            if not relevant_tables:
                return QueryResult(
                    question=question,
                    relevant_tables=[],
                    sql_query="",
                    results=[],
                    results_count=0,
                    execution_error='Could not find relevant tables for this question'
                )
            
            table_names = [t.name for t in relevant_tables]
            print(f"   📋 Selected tables: {', '.join(table_names)}")
            
            # Show why these tables were selected
            for table in relevant_tables:
                reasons = []
                if table.semantic_profile:
                    reasons.append(f"role: {table.semantic_profile.business_role}")
                if table.sample_data:
                    reasons.append("has data")
                if table.object_type == 'VIEW':
                    reasons.append("view/report")
                
                reason_str = ", ".join(reasons) if reasons else "name match"
                print(f"      • {table.name}: {reason_str}")
            
            # Enhanced SQL generation
            print(f"   🧠 Generating SQL with enhanced AI context...")
            sql_query = await self._generate_sql_enhanced(question, relevant_tables)
            
            if not sql_query:
                return QueryResult(
                    question=question,
                    relevant_tables=table_names,
                    sql_query="",
                    results=[],
                    results_count=0,
                    execution_error='Could not generate SQL query'
                )
            
            print(f"   ⚡ Generated query: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
            
            # Execute SQL with enhanced error handling
            print(f"   💾 Executing query...")
            start_time = time.time()
            execution_result = await self._execute_sql_enhanced(sql_query)
            execution_time = time.time() - start_time
            
            if execution_result.get('error'):
                print(f"   ❌ SQL execution failed")
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
    
    def _find_relevant_tables_enhanced(self, question: str) -> List[TableInfo]:
        """Enhanced table relevance scoring with better business logic"""
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_tables = []
        
        for table in self.tables:
            score = 0
            reasons = []
            
            # 1. Direct name matching (highest priority)
            table_name_words = set(re.findall(r'\w+', table.name.lower()))
            name_matches = question_words.intersection(table_name_words)
            if name_matches:
                score += len(name_matches) * 3
                reasons.append(f"name matches: {', '.join(name_matches)}")
            
            # 2. Semantic profile matching (very important)
            if table.semantic_profile:
                role = table.semantic_profile.business_role.lower()
                role_words = set(re.findall(r'\w+', role))
                role_matches = question_words.intersection(role_words)
                if role_matches:
                    score += len(role_matches) * 4
                    reasons.append(f"business role matches: {', '.join(role_matches)}")
                
                # Special business term matching
                business_mappings = {
                    'customer': ['customer', 'client', 'account', 'user', 'businesspoint'],
                    'order': ['order', 'purchase', 'transaction', 'sale'],
                    'product': ['product', 'item', 'service', 'offering'],
                    'payment': ['payment', 'invoice', 'billing', 'financial'],
                    'employee': ['employee', 'staff', 'worker', 'user'],
                    'campaign': ['campaign', 'marketing', 'promotion', 'advertising']
                }
                
                for business_term, keywords in business_mappings.items():
                    if any(keyword in question_lower for keyword in keywords):
                        if business_term in role or any(keyword in role for keyword in keywords):
                            score += 5
                            reasons.append(f"business term: {business_term}")
            
            # 3. Column name matching
            column_matches = 0
            for col in table.columns:
                col_name_words = set(re.findall(r'\w+', col['name'].lower()))
                if question_words.intersection(col_name_words):
                    column_matches += 1
            
            if column_matches > 0:
                score += column_matches
                reasons.append(f"{column_matches} column matches")
            
            # 4. Data availability bonus (prefer tables/views with actual data)
            if table.sample_data:
                score += 2
                reasons.append("has sample data")
            
            # 5. Row count consideration (prefer tables with reasonable data)
            if table.row_count > 100:
                score += 1
            elif table.row_count > 10000:
                score += 2  # Medium-sized tables often most useful
            
            # 6. Object type considerations
            if 'report' in question_lower or 'summary' in question_lower:
                if table.object_type == 'VIEW':
                    score += 2
                    reasons.append("view for reporting")
            
            # 7. Schema considerations (prefer main business schemas)
            if table.schema.lower() == 'dbo':
                score += 1
            
            if score > 0:
                scored_tables.append((table, score, reasons))
        
        # Sort by score and return top tables
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3-4 tables, but ensure we have at least one with data
        selected_tables = []
        tables_with_data = []
        
        for table, score, reasons in scored_tables[:6]:  # Look at top 6
            if table.sample_data:
                tables_with_data.append(table)
            selected_tables.append(table)
            
            if len(selected_tables) >= 3 and len(tables_with_data) >= 1:
                break
        
        # Ensure we return at least one table with data if available
        if not tables_with_data and scored_tables:
            # Add the highest scoring table with data
            for table, score, reasons in scored_tables:
                if table.sample_data:
                    if table not in selected_tables:
                        selected_tables.append(table)
                    break
        
        return selected_tables[:4]  # Limit to 4 tables maximum
    
    async def _generate_sql_enhanced(self, question: str, tables: List[TableInfo]) -> Optional[str]:
        """Enhanced SQL generation with improved context and error handling"""
        # Prepare enhanced table information for LLM
        table_info = []
        for table in tables:
            # Include more detailed column information
            columns_detail = []
            for col in table.columns[:12]:  # Include more columns
                col_detail = {
                    'name': col['name'],
                    'type': col['data_type'],
                    'nullable': col['nullable'],
                    'is_pk': col['is_primary_key'],
                    'is_fk': col['is_foreign_key']
                }
                columns_detail.append(col_detail)
            
            info = {
                'name': table.name,
                'schema': table.schema,
                'full_name': table.full_name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': columns_detail,
                'sample_data': table.sample_data[:1] if table.sample_data else [],
                'business_role': table.semantic_profile.business_role if table.semantic_profile else 'Unknown',
                'primary_purpose': table.semantic_profile.primary_purpose if table.semantic_profile else '',
                'has_data': len(table.sample_data) > 0
            }
            table_info.append(info)
        
        domain_context = f"{self.domain.domain_type} system in {self.domain.industry}" if self.domain else "Business system"
        customer_context = self.domain.customer_definition if self.domain and self.domain.customer_definition else ""
        
        prompt = f"""
Generate SQL query to answer this business question accurately.

BUSINESS CONTEXT:
Question: "{question}"
Domain: {domain_context}
Customer Definition: {customer_context}

AVAILABLE TABLES:
{json.dumps(table_info, indent=2, default=str)}

SQL GENERATION RULES:
1. Use proper SQL Server syntax with [schema].[table] format
2. Include appropriate WHERE clauses for filtering
3. Use JOINs when relationships are clear
4. Use TOP clause to limit results (TOP 100 for counts, TOP 20 for detail queries)
5. Handle NULL values appropriately
6. Prefer tables/views that have sample data
7. Use meaningful column aliases
8. Consider the business context for filtering (active records, recent data, etc.)

COMMON PATTERNS:
- For counts: SELECT COUNT(*) as TotalCount FROM [schema].[table] WHERE conditions
- For recent data: Use date filters like WHERE DateColumn >= DATEADD(month, -1, GETDATE())
- For top results: ORDER BY relevant_column DESC
- For customer queries: Focus on tables with 'customer', 'business', or 'account' context

IMPORTANT:
- Only use columns that exist in the provided table schemas
- Ensure all referenced tables are in the available tables list
- Generate syntactically correct SQL Server T-SQL
- Consider the business meaning behind the question

Respond with ONLY the SQL query, no explanations:
"""
        
        try:
            response = await self.llm.ask(prompt, "You are an expert SQL developer specializing in business intelligence queries. Generate only valid SQL Server T-SQL.")
            
            # Enhanced response cleaning
            cleaned_sql = clean_sql_response(response)
            return cleaned_sql
                
        except Exception as e:
            print(f"⚠️ Enhanced SQL generation failed: {e}")
            return None
    
    async def _execute_sql_enhanced(self, sql_query: str) -> Dict[str, Any]:
        """Enhanced SQL execution with better error handling and result processing"""
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                
                # Add query timeout for safety
                cursor.execute(sql_query)
                
                if not cursor.description:
                    return {'data': [], 'count': 0, 'error': None}
                
                columns = [col[0] for col in cursor.description]
                results = []
                
                # Enhanced result processing with better limits
                for i, row in enumerate(cursor.fetchall()):
                    if i >= self.config.max_results:  # Use config limit
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
            if 'timeout' in error_msg.lower():
                error_msg = f"Query timeout: {error_msg[:100]}"
            elif 'invalid object name' in error_msg.lower():
                error_msg = f"Table/view not found: {error_msg[:100]}"
            elif 'invalid column name' in error_msg.lower():
                error_msg = f"Column not found: {error_msg[:100]}"
            elif 'permission' in error_msg.lower():
                error_msg = f"Access denied: {error_msg[:100]}"
            else:
                error_msg = error_msg[:150]
            
            return {'data': [], 'count': 0, 'error': error_msg}
    
    def _display_query_result(self, result: QueryResult, query_number: int):
        """Display query result with enhanced formatting"""
        if result.execution_error:
            print(f"❌ Error: {result.execution_error}")
        else:
            print(f"📋 Generated SQL:")
            print(f"   {result.sql_query}")
            
            # Show if views were used
            relevant_tables = result.relevant_tables
            views_used = [t for t in relevant_tables if any(table.name == t and table.object_type == 'VIEW' for table in self.tables)]
            if views_used:
                print(f"📊 Views utilized: {', '.join(views_used)} (enhanced capability)")
            
            count = result.results_count
            print(f"📊 Results: {count} rows")
            
            if result.results:
                # Enhanced result display
                for i, row in enumerate(result.results[:5], 1):
                    # Show more columns but limit length
                    display_row = {}
                    for key, value in list(row.items())[:6]:
                        if isinstance(value, str) and len(value) > 30:
                            display_row[key] = value[:30] + "..."
                        else:
                            display_row[key] = value
                    print(f"   {i}. {display_row}")
                
                if count > 5:
                    print(f"   ... and {count - 5} more rows")
            
            # Show execution time if available
            if result.execution_time > 0:
                print(f"⚡ Execution time: {result.execution_time:.3f}s")
            
            # Show relevant tables used
            if relevant_tables:
                print(f"📋 Used tables: {', '.join(relevant_tables)}")