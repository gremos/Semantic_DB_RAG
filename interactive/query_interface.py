#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Query Interface - Readable and Maintainable
Focus on getting queries to work reliably
"""

import pyodbc
import time
from typing import List, Dict, Any, Optional

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import find_related_tables_fuzzy, generate_simple_sql_prompt, clean_sql_query, safe_database_value

class SimpleLLMClient:
    """Simple LLM client for SQL generation"""
    
    def __init__(self, config: Config):
        from langchain_openai import AzureChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.05,
            request_timeout=60
        )
    
    async def generate_sql(self, prompt: str) -> str:
        """Generate SQL from prompt"""
        from langchain.schema import HumanMessage, SystemMessage
        import asyncio
        
        try:
            messages = [
                SystemMessage(content="You are an expert SQL developer. Generate only SQL Server T-SQL queries. No explanations."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ SQL generation failed: {e}")
            return ""

class SimpleQueryInterface:
    """Simplified query interface that actually works"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = SimpleLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship],
                                      business_analysis: Dict[str, Any] = None):
        """Start interactive query session"""
        
        self.tables = tables
        self.domain = domain
        self.relationships = relationships
        
        # Show system status
        self._show_system_status()
        
        query_count = 0
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    self._show_help()
                    continue
                elif question.lower() == 'status':
                    self._show_system_status()
                    continue
                elif question.lower() == 'entities':
                    self._show_entities()
                    continue
                elif question.lower() == 'tables':
                    self._show_tables()
                    continue
                elif not question:
                    continue
                
                query_count += 1
                print(f"🔍 Processing query #{query_count}...")
                
                start_time = time.time()
                result = await self._process_query(question)
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
        print("👋 Thanks for using the Simplified Semantic Database RAG System!")
    
    async def _process_query(self, question: str) -> QueryResult:
        """Process a natural language query"""
        
        try:
            # Step 1: Find relevant tables
            print("   🔍 Finding relevant tables...")
            relevant_tables = find_related_tables_fuzzy(question, self.tables)
            
            if not relevant_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found. Try asking about customers, payments, orders, or products.",
                    tables_used=[]
                )
            
            print(f"   📋 Found {len(relevant_tables)} relevant tables")
            for table in relevant_tables:
                print(f"      • {table.name} ({table.entity_type}, confidence: {table.confidence:.2f})")
            
            # Step 2: Generate SQL
            print("   ⚡ Generating SQL query...")
            sql_prompt = generate_simple_sql_prompt(question, relevant_tables)
            
            sql_response = await self.llm_client.generate_sql(sql_prompt)
            sql_query = clean_sql_query(sql_response)
            
            if not sql_query:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to generate SQL query",
                    tables_used=[t.name for t in relevant_tables]
                )
            
            print(f"   💾 Generated: {sql_query[:100]}...")
            
            # Step 3: Execute query
            print("   🚀 Executing query...")
            results, error = self._execute_query(sql_query)
            
            return QueryResult(
                question=question,
                sql_query=sql_query,
                results=results,
                error=error,
                tables_used=[t.name for t in relevant_tables]
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Query processing failed: {str(e)}",
                tables_used=[]
            )
    
    def _execute_query(self, sql_query: str) -> tuple:
        """Execute SQL query and return results"""
        
        try:
            connection_string = self.config.get_database_connection_string()
            
            with pyodbc.connect(connection_string) as conn:
                # Set UTF-8 encoding
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                # Get column names
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    
                    # Fetch results
                    results = []
                    row_count = 0
                    
                    for row in cursor.fetchall():
                        if row_count >= self.config.max_results:
                            break
                        
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = safe_database_value(value)
                        
                        results.append(row_dict)
                        row_count += 1
                    
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages
            if 'invalid object name' in error_msg.lower():
                error_msg = f"Table not found: {error_msg[:150]}"
            elif 'invalid column name' in error_msg.lower():
                error_msg = f"Column not found: {error_msg[:150]}"
            elif 'syntax error' in error_msg.lower():
                error_msg = f"SQL syntax error: {error_msg[:150]}"
            else:
                error_msg = error_msg[:200]
            
            return [], error_msg
    
    def _show_system_status(self):
        """Show system status"""
        
        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        classified_count = sum(1 for t in self.tables if t.entity_type != 'Unknown')
        
        print(f"✅ System Status:")
        print(f"   📊 Database: {table_count} tables, {view_count} views")
        print(f"   🧠 Classified: {classified_count} entities")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        # Show entity counts
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   🏢 Business Entities:")
            for entity_type, count in sorted(entity_counts.items()):
                print(f"      • {entity_type}: {count}")
        
        # Show capabilities
        if self.domain and self.domain.capabilities:
            enabled_caps = [cap for cap, enabled in self.domain.capabilities.items() if enabled]
            if enabled_caps:
                print(f"   🎯 Available queries: {', '.join(cap.replace('_', ' ') for cap in enabled_caps)}")
        
        print("💡 Type 'help' for examples, 'entities' for entity list, 'tables' for table list")
    
    def _show_help(self):
        """Show help with sample questions"""
        
        print("\n💡 SAMPLE QUESTIONS:")
        
        if self.domain and self.domain.sample_questions:
            for i, question in enumerate(self.domain.sample_questions[:10], 1):
                print(f"   {i}. {question}")
        else:
            # Default examples
            print("   1. How many customers do we have?")
            print("   2. Show customer information")
            print("   3. What is our total revenue?")
            print("   4. How many orders do we have?")
            print("   5. Show payment information")
            print("   6. Count total paid customers")
            print("   7. List recent transactions")
            print("   8. Show product information")
        
        print("\n🔧 COMMANDS:")
        print("   • 'help' - Show this help")
        print("   • 'status' - Show system status")
        print("   • 'entities' - Show classified entities")
        print("   • 'tables' - Show available tables")
        print("   • 'quit' - Exit")
    
    def _show_entities(self):
        """Show classified entities"""
        
        print("\n🧠 CLASSIFIED ENTITIES:")
        
        entity_groups = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                if table.entity_type not in entity_groups:
                    entity_groups[table.entity_type] = []
                entity_groups[table.entity_type].append(table)
        
        if not entity_groups:
            print("   No entities classified yet. Run semantic analysis first.")
            return
        
        for entity_type, tables_list in sorted(entity_groups.items()):
            print(f"   🏢 {entity_type} ({len(tables_list)} tables):")
            for table in sorted(tables_list, key=lambda x: x.confidence, reverse=True)[:5]:
                print(f"      • {table.name} (confidence: {table.confidence:.2f})")
            if len(tables_list) > 5:
                print(f"      ... and {len(tables_list) - 5} more")
    
    def _show_tables(self):
        """Show available tables"""
        
        print(f"\n📋 AVAILABLE TABLES ({len(self.tables)}):")
        
        # Group by entity type
        by_entity = {}
        for table in self.tables:
            entity = table.entity_type if table.entity_type != 'Unknown' else 'Unclassified'
            if entity not in by_entity:
                by_entity[entity] = []
            by_entity[entity].append(table)
        
        for entity_type, tables_list in sorted(by_entity.items()):
            print(f"   🏢 {entity_type} ({len(tables_list)}):")
            for table in sorted(tables_list, key=lambda x: x.row_count, reverse=True)[:3]:
                row_info = f"{table.row_count} rows" if table.row_count > 0 else "no data"
                print(f"      • {table.name} ({row_info})")
            if len(tables_list) > 3:
                print(f"      ... and {len(tables_list) - 3} more")
    
    def _display_result(self, result: QueryResult):
        """Display query result"""
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.tables_used:
                print(f"💡 Tables considered: {', '.join(result.tables_used)}")
        else:
            print(f"📋 SQL Query:")
            print(f"   {result.sql_query}")
            
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                # Show first few results
                for i, row in enumerate(result.results[:5], 1):
                    # Limit display width
                    display_row = {}
                    for key, value in list(row.items())[:6]:  # Max 6 columns
                        if isinstance(value, str) and len(value) > 30:
                            display_row[key] = value[:30] + "..."
                        else:
                            display_row[key] = value
                    print(f"   {i}. {display_row}")
                
                if len(result.results) > 5:
                    print(f"   ... and {len(result.results) - 5} more rows")
            
            if result.tables_used:
                print(f"📋 Tables used: {', '.join(result.tables_used)}")
            
            if result.execution_time > 0:
                print(f"⚡ Execution time: {result.execution_time:.3f}s")

# For backward compatibility
IntelligentQueryInterface = SimpleQueryInterface
EnhancedQueryInterface = SimpleQueryInterface
QueryInterface = SimpleQueryInterface