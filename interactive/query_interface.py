#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXED Query Interface - Method Name Fixes and Performance Improvements
Resolves the 'start_interactive_session' method name issue
"""

import pyodbc
import time
import re
import json
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import safe_database_value, clean_sql_response, extract_json_from_response

class IntelligentLLMClient:
    """Enhanced LLM client for intelligent query operations with relationship context"""
    
    def __init__(self, config: Config):
        from langchain_openai import AzureChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.05,  # Very low temperature for consistent SQL generation
            request_timeout=180  # Longer timeout for complex analysis
        )
    
    async def ask(self, prompt: str, system_message: str = "You are an expert database architect and SQL developer.") -> str:
        """Ask LLM with enhanced retry logic and relationship context awareness"""
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
                    await asyncio.sleep(3 ** attempt)
                else:
                    print(f"⚠️ LLM request failed after 3 attempts: {e}")
                    raise e

class IntelligentQueryProcessor:
    """Intelligent query processor with comprehensive relationship awareness"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = IntelligentLLMClient(config)
    
    async def process_intelligent_query(self, question: str, 
                                      tables: List[TableInfo],
                                      domain: Optional[BusinessDomain],
                                      relationships: List[Relationship],
                                      comprehensive_analysis: Dict[str, Any]) -> QueryResult:
        """Process query with intelligent table selection and relationship awareness"""
        
        try:
            # Step 1: Intelligent table selection based on entity types
            print(f"   🧠 Selecting tables based on question analysis...")
            selected_tables = self._select_tables_intelligent(question, tables, comprehensive_analysis)
            
            if not selected_tables:
                return self._create_error_result(question, "No suitable tables found for query")
            
            print(f"   📋 Selected {len(selected_tables)} relevant tables")
            
            # Step 2: Generate SQL with relationship context
            print(f"   ⚡ Generating SQL with relationship intelligence...")
            sql_query = await self._generate_intelligent_sql(question, selected_tables, relationships)
            
            if not sql_query:
                return self._create_error_result(question, "Failed to generate SQL query")
            
            print(f"   💾 Generated query: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
            
            # Step 3: Execute with enhanced error handling
            print(f"   🚀 Executing query...")
            execution_result = await self._execute_sql_intelligent(sql_query)
            
            # Step 4: Create enhanced result
            return self._create_intelligent_result(
                question, selected_tables, sql_query, execution_result, relationships
            )
            
        except Exception as e:
            print(f"   ❌ Query processing failed: {e}")
            return self._create_error_result(question, f"Processing failed: {str(e)}")
    
    def _select_tables_intelligent(self, question: str, tables: List[TableInfo], 
                                 comprehensive_analysis: Dict[str, Any]) -> List[TableInfo]:
        """Intelligent table selection based on question analysis and entity types"""
        
        question_lower = question.lower()
        selected_tables = []
        
        # Get entity analysis from comprehensive analysis
        entity_analysis = comprehensive_analysis.get('entity_analysis', {})
        high_confidence_entities = entity_analysis.get('high_confidence_entities', {})
        
        # Question pattern analysis
        if any(word in question_lower for word in ['paid', 'customer', 'payment']):
            # Customer-payment related query
            customer_tables = high_confidence_entities.get('Customer', [])
            payment_tables = high_confidence_entities.get('Payment', [])
            
            if customer_tables:
                # Find best customer table
                best_customer = max(customer_tables, key=lambda x: x.get('confidence', 0))
                customer_table = self._find_table_by_name(tables, best_customer['table_name'])
                if customer_table:
                    selected_tables.append(customer_table)
            
            if payment_tables:
                # Find best payment table
                best_payment = max(payment_tables, key=lambda x: x.get('confidence', 0))
                payment_table = self._find_table_by_name(tables, best_payment['table_name'])
                if payment_table:
                    selected_tables.append(payment_table)
        
        elif any(word in question_lower for word in ['customer', 'client', 'account']):
            # Customer-only query
            customer_tables = high_confidence_entities.get('Customer', [])
            if customer_tables:
                best_customer = max(customer_tables, key=lambda x: x.get('confidence', 0))
                customer_table = self._find_table_by_name(tables, best_customer['table_name'])
                if customer_table:
                    selected_tables.append(customer_table)
        
        elif any(word in question_lower for word in ['revenue', 'payment', 'transaction', 'total']):
            # Payment/revenue query
            payment_tables = high_confidence_entities.get('Payment', [])
            if payment_tables:
                best_payment = max(payment_tables, key=lambda x: x.get('confidence', 0))
                payment_table = self._find_table_by_name(tables, best_payment['table_name'])
                if payment_table:
                    selected_tables.append(payment_table)
        
        elif any(word in question_lower for word in ['order', 'sale', 'purchase']):
            # Order query
            order_tables = high_confidence_entities.get('Order', [])
            if order_tables:
                best_order = max(order_tables, key=lambda x: x.get('confidence', 0))
                order_table = self._find_table_by_name(tables, best_order['table_name'])
                if order_table:
                    selected_tables.append(order_table)
        
        # Fallback: if no specific tables selected, try to find any business-relevant tables
        if not selected_tables:
            for table in tables[:20]:  # Check first 20 tables
                if table.semantic_profile and table.semantic_profile.entity_type in ['Customer', 'Payment', 'Order']:
                    selected_tables.append(table)
                    if len(selected_tables) >= 3:
                        break
        
        return selected_tables
    
    def _find_table_by_name(self, tables: List[TableInfo], full_name: str) -> Optional[TableInfo]:
        """Find table by full name"""
        for table in tables:
            if table.full_name == full_name:
                return table
        return None
    
    async def _generate_intelligent_sql(self, question: str, selected_tables: List[TableInfo], 
                                      relationships: List[Relationship]) -> Optional[str]:
        """Generate SQL using intelligent analysis of tables and relationships"""
        
        # Build table context for LLM
        table_context = []
        for table in selected_tables:
            entity_type = 'Unknown'
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
            
            context = {
                'table_name': table.name,
                'full_name': table.full_name,
                'entity_type': entity_type,
                'row_count': table.row_count,
                'columns': [
                    {
                        'name': col['name'],
                        'type': col['data_type'],
                        'is_id': col['name'].lower().endswith('id'),
                        'is_amount': any(word in col['name'].lower() for word in ['amount', 'total', 'price', 'cost', 'value']),
                        'is_date': any(word in col['name'].lower() for word in ['date', 'time', 'created', 'modified']),
                        'is_name': any(word in col['name'].lower() for word in ['name', 'title', 'description'])
                    } for col in table.columns[:15]  # Limit columns for context
                ],
                'sample_data': table.sample_data[:2] if table.sample_data else []
            }
            table_context.append(context)
        
        # Find relevant relationships
        table_names = {table.full_name for table in selected_tables}
        relevant_relationships = []
        for rel in relationships:
            if rel.from_table in table_names and rel.to_table in table_names:
                relevant_relationships.append({
                    'from_table': rel.from_table,
                    'to_table': rel.to_table,
                    'from_column': rel.column,
                    'relationship_type': rel.relationship_type,
                    'confidence': rel.confidence
                })
        
        # Create intelligent prompt
        prompt = self._create_intelligent_sql_prompt(question, table_context, relevant_relationships)
        
        try:
            system_message = """You are an expert SQL architect specializing in business intelligence queries.
Generate accurate SQL Server T-SQL that answers the business question using the provided tables and relationships.
Focus on business logic and proper JOINs. Respond with ONLY the SQL query - no explanations or markdown."""
            
            response = await self.llm.ask(prompt, system_message)
            cleaned_sql = clean_sql_response(response)
            return cleaned_sql
            
        except Exception as e:
            print(f"⚠️ SQL generation failed: {e}")
            return None
    
    def _create_intelligent_sql_prompt(self, question: str, table_context: List[Dict], 
                                     relationships: List[Dict]) -> str:
        """Create intelligent SQL generation prompt"""
        
        prompt = f"""
BUSINESS QUESTION: "{question}"

AVAILABLE TABLES:
{json.dumps(table_context, indent=2)}

RELATIONSHIPS DISCOVERED:
{json.dumps(relationships, indent=2)}

SQL GENERATION RULES:
1. Use proper table and column names with square brackets
2. If multiple tables are involved, use appropriate JOINs based on relationships
3. For customer-payment queries, use INNER JOIN to exclude customers without payments
4. For count queries, use COUNT(DISTINCT ...) to avoid duplicates
5. Add reasonable LIMIT (TOP 100) unless the question asks for totals/counts
6. Use business-appropriate WHERE clauses (e.g., exclude NULL values for amounts)
7. Format results with meaningful column aliases

Generate the optimal SQL query for this business question:
"""
        return prompt
    
    async def _execute_sql_intelligent(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL with intelligent error handling and recovery"""
        try:
            with self._get_connection() as conn:
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
            
            # Intelligent error classification
            if 'invalid object name' in error_msg.lower():
                error_msg = f"Table/view not found: {error_msg[:100]}"
            elif 'invalid column name' in error_msg.lower():
                error_msg = f"Column not found: {error_msg[:100]}"
            elif 'timeout' in error_msg.lower():
                error_msg = f"Query timeout: {error_msg[:100]}"
            else:
                error_msg = error_msg[:200]
            
            return {'data': [], 'count': 0, 'error': error_msg}
    
    def _create_intelligent_result(self, question: str, selected_tables: List[TableInfo],
                                 sql_query: str, execution_result: Dict[str, Any],
                                 relationships: List[Relationship]) -> QueryResult:
        """Create enhanced query result with business intelligence"""
        
        table_names = [t.name for t in selected_tables]
        entity_types = [t.semantic_profile.entity_type if t.semantic_profile else 'Unknown' for t in selected_tables]
        relationships_used = [rel.relationship_type for rel in relationships if rel.from_table in {t.full_name for t in selected_tables} and rel.to_table in {t.full_name for t in selected_tables}]
        
        # Determine query complexity
        complexity = 'Simple'
        if len(selected_tables) > 2:
            complexity = 'Medium'
        if len(relationships_used) > 1 or 'JOIN' in sql_query.upper():
            complexity = 'Complex'
        
        # Generate business interpretation
        business_interpretation = self._generate_business_interpretation(
            question, execution_result, entity_types
        )
        
        return QueryResult(
            question=question,
            relevant_tables=table_names,
            sql_query=sql_query,
            results=execution_result.get('data', []),
            results_count=execution_result.get('count', 0),
            execution_error=execution_result.get('error'),
            execution_time=0.0,  # Will be set by caller
            query_complexity=complexity,
            business_significance='High' if any(et in ['Customer', 'Payment'] for et in entity_types) else 'Medium',
            entities_involved=entity_types,
            relationships_used=relationships_used,
            business_interpretation=business_interpretation
        )
    
    def _generate_business_interpretation(self, question: str, execution_result: Dict[str, Any],
                                        entity_types: List[str]) -> str:
        """Generate intelligent business interpretation of results"""
        
        results = execution_result.get('data', [])
        count = execution_result.get('count', 0)
        error = execution_result.get('error')
        
        if error:
            return f"Query execution failed: {error}"
        
        if count == 0:
            return f"No results found for {', '.join(entity_types)} analysis. Data may be missing or filtering too restrictive."
        
        # Generate positive interpretations
        question_lower = question.lower()
        if 'paid' in question_lower and 'customer' in question_lower:
            return f"Found {count} paid customers. Customer-payment relationships are working correctly."
        elif 'customer' in question_lower:
            return f"Customer analysis returned {count} records. Customer data is accessible."
        elif 'revenue' in question_lower or 'payment' in question_lower:
            if results and isinstance(list(results[0].values())[0], (int, float)):
                return f"Revenue analysis shows {count} records with financial data."
        
        return f"Analysis of {', '.join(entity_types)} entities returned {count} records with business-relevant data."
    
    def _create_error_result(self, question: str, error_message: str) -> QueryResult:
        """Create error result with helpful context"""
        return QueryResult(
            question=question,
            relevant_tables=[],
            sql_query="",
            results=[],
            results_count=0,
            execution_error=error_message,
            execution_time=0.0,
            query_complexity="Unknown",
            business_significance="Unknown",
            entities_involved=[],
            relationships_used=[],
            business_interpretation=f"Query processing failed: {error_message}"
        )
    
    def _get_connection(self):
        """Get database connection"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # Set UTF-8 encoding
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
        return conn


class IntelligentQueryInterface:
    """FIXED: Intelligent query interface with correct method names"""
    
    def __init__(self, config: Config):
        self.config = config
        self.query_processor = IntelligentQueryProcessor(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.comprehensive_analysis: Dict[str, Any] = {}
    
    async def start_intelligent_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship],
                                      comprehensive_analysis: Dict[str, Any] = None):
        """FIXED: Start intelligent interactive query session with relationship awareness"""
        
        self.tables = tables
        self.domain = domain
        self.relationships = relationships
        self.comprehensive_analysis = comprehensive_analysis or {}
        
        # Show intelligent system status
        self._show_intelligent_system_status()
        
        query_count = 0
        while True:
            try:
                question = input(f"\n❓ Intelligent Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    self._show_intelligent_help()
                    continue
                elif question.lower() == 'status':
                    self._show_intelligent_system_status()
                    continue
                elif question.lower() == 'relationships':
                    self._show_relationship_intelligence()
                    continue
                elif question.lower() == 'entities':
                    self._show_entity_intelligence()
                    continue
                elif not question:
                    continue
                
                query_count += 1
                print(f"🧠 Processing intelligent query #{query_count} with relationship awareness...")
                start_time = time.time()
                
                result = await self.query_processor.process_intelligent_query(
                    question, self.tables, self.domain, self.relationships, self.comprehensive_analysis
                )
                
                elapsed = time.time() - start_time
                result.execution_time = elapsed
                
                print(f"⏱️ Completed in {elapsed:.1f}s")
                print("-" * 60)
                
                self._display_intelligent_query_result(result, query_count)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Intelligent session summary: {query_count} queries processed")
        print("👋 Thanks for using the INTELLIGENT Semantic Database RAG System!")
    
    # FIXED: Add backward compatibility method
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship],
                                      comprehensive_analysis: Dict[str, Any] = None):
        """BACKWARD COMPATIBILITY: Redirect to start_intelligent_session"""
        await self.start_intelligent_session(tables, domain, relationships, comprehensive_analysis)
    
    def _show_intelligent_system_status(self):
        """Show intelligent system status with relationship context"""
        
        table_count = sum(1 for t in self.tables if t.object_type == 'BASE TABLE')
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        relationship_count = len(self.relationships)
        
        print(f"✅ INTELLIGENT system ready with relationship intelligence!")
        print(f"📊 Database: {table_count} tables, {view_count} views, {relationship_count} relationships")
        
        # Show business intelligence status
        if self.comprehensive_analysis:
            business_insights = self.comprehensive_analysis.get('business_insights', {})
            if business_insights:
                readiness = business_insights.get('business_readiness', {})
                capabilities = business_insights.get('query_capabilities', {})
                
                rating = readiness.get('rating', 'Unknown')
                score = readiness.get('score', 0)
                print(f"🎯 Business Intelligence: {rating} ({score}/100)")
                
                # Show enabled capabilities
                enabled_caps = [cap.replace('_', ' ').title() for cap, enabled in capabilities.items() if enabled]
                if enabled_caps:
                    print(f"💬 Query Capabilities: {', '.join(enabled_caps[:4])}")
                    if len(enabled_caps) > 4:
                        print(f"   + {len(enabled_caps) - 4} more capabilities")
        
        print("💡 Type 'help' for examples, 'relationships' for network, 'entities' for classification")
    
    def _show_intelligent_help(self):
        """Show intelligent help with relationship-aware examples"""
        
        print("\n💡 INTELLIGENT HELP - Relationship-Aware Business Queries:")
        
        if self.comprehensive_analysis:
            business_insights = self.comprehensive_analysis.get('business_insights', {})
            capabilities = business_insights.get('query_capabilities', {})
            entity_analysis = self.comprehensive_analysis.get('entity_analysis', {})
            entity_counts = entity_analysis.get('entity_counts', {})
            
            if capabilities.get('customer_queries', False):
                print("\n👥 CUSTOMER INTELLIGENCE:")
                print("   • How many customers do we have?")
                print("   • Show me customer details")
                
                if capabilities.get('paid_customer_analysis', False):
                    print("   • How many customers have made payments? (INTELLIGENT: uses relationship discovery)")
                    print("   • Which customers are our top spenders?")
                    print("   • Show paid customers for 2025")
            
            if capabilities.get('payment_queries', False):
                print("\n💰 REVENUE INTELLIGENCE:")
                print("   • What is our total revenue?")
                print("   • Show payment trends")
                print("   • Calculate average transaction value")
            
            if entity_counts.get('Order', 0) > 0:
                print("\n📦 ORDER INTELLIGENCE:")
                print("   • How many orders do we have?")
                print("   • Show recent orders")
                print("   • What is our average order value?")
        
        print("\n🧠 INTELLIGENT FEATURES:")
        print("   • Automatic relationship-aware table selection")
        print("   • Business context understanding")
        print("   • Optimal SQL generation using discovered relationships")
        print("   • Query complexity assessment and optimization")
        
        print("\n📋 INTELLIGENT COMMANDS:")
        print("   • 'relationships' - Show relationship network intelligence")
        print("   • 'entities' - Show entity classification intelligence")
        print("   • 'status' - Show intelligent system status")
    
    def _show_relationship_intelligence(self):
        """Show relationship intelligence network"""
        
        print("\n🕸️ RELATIONSHIP INTELLIGENCE NETWORK:")
        
        if not self.relationships:
            print("   No relationships discovered")
            return
        
        # Group by relationship type
        rel_types = defaultdict(list)
        for rel in self.relationships:
            rel_types[rel.relationship_type].append(f"{rel.from_table.split('.')[-1]} → {rel.to_table.split('.')[-1]}")
        
        for rel_type, relationships in rel_types.items():
            print(f"   🔗 {rel_type.replace('_', ' ').title()}: {len(relationships)}")
            for i, rel_desc in enumerate(relationships[:3]):
                print(f"      • {rel_desc}")
            if len(relationships) > 3:
                print(f"      ... and {len(relationships) - 3} more")
        
        print(f"\n   📊 Total: {len(self.relationships)} relationships discovered")
    
    def _show_entity_intelligence(self):
        """Show entity classification intelligence"""
        
        print("\n🧠 ENTITY CLASSIFICATION INTELLIGENCE:")
        
        if not self.comprehensive_analysis:
            print("   No entity analysis available")
            return
        
        entity_analysis = self.comprehensive_analysis.get('entity_analysis', {})
        high_confidence = entity_analysis.get('high_confidence_entities', {})
        entity_counts = entity_analysis.get('entity_counts', {})
        
        print(f"   📊 BUSINESS ENTITIES DISCOVERED:")
        for entity_type, count in entity_counts.items():
            high_conf_count = len(high_confidence.get(entity_type, []))
            confidence_ratio = f"({high_conf_count} high confidence)" if high_conf_count > 0 else "(low confidence)"
            print(f"      • {entity_type}: {count} tables {confidence_ratio}")
        
        # Show best examples
        core_entities = ['Customer', 'Payment', 'Order', 'Product']
        for entity in core_entities:
            if entity in high_confidence and high_confidence[entity]:
                best = high_confidence[entity][0]
                table_name = best['table_name'].split('.')[-1].replace('[', '').replace(']', '')
                print(f"   🎯 Best {entity} table: {table_name} (confidence: {best['confidence']:.2f})")
    
    def _display_intelligent_query_result(self, result: QueryResult, query_number: int):
        """Display intelligent query result with business context"""
        
        if result.execution_error:
            print(f"❌ Error: {result.execution_error}")
            print(f"💡 Business Context: {result.business_interpretation}")
        else:
            print(f"📋 Generated SQL ({result.query_complexity} complexity):")
            print(f"   {result.sql_query}")
            
            count = result.results_count
            print(f"📊 Results: {count} rows ({result.business_significance} business significance)")
            
            if result.results:
                # Enhanced result display with business context
                for i, row in enumerate(result.results[:5], 1):
                    display_row = {}
                    for key, value in list(row.items())[:6]:
                        if isinstance(value, str) and len(value) > 35:
                            display_row[key] = value[:35] + "..."
                        else:
                            display_row[key] = value
                    print(f"   {i}. {display_row}")
                
                if count > 5:
                    print(f"   ... and {count - 5} more rows")
            
            # Show intelligent analysis
            print(f"\n🧠 Intelligent Analysis:")
            print(f"   • Entities involved: {', '.join(result.entities_involved) if result.entities_involved else 'None'}")
            if result.relationships_used:
                print(f"   • Relationships used: {', '.join(result.relationships_used)}")
            print(f"   • Business interpretation: {result.business_interpretation}")
            
            # Show execution metrics
            if result.execution_time > 0:
                print(f"⚡ Execution: {result.execution_time:.3f}s")
            
            # Show tables used
            if result.relevant_tables:
                print(f"📋 Tables: {', '.join(result.relevant_tables)}")

    def _select_tables_comprehensive(self, question: str, tables: List[TableInfo], 
                                   comprehensive_analysis: Dict[str, Any]) -> List[TableInfo]:
        """Enhanced table selection using comprehensive analysis results"""
        
        question_lower = question.lower()
        selected_tables = []
        
        # Get comprehensive entity analysis
        business_intelligence = comprehensive_analysis.get('business_intelligence', {})
        entity_distribution = business_intelligence.get('entity_distribution', {})
        
        # Use comprehensive relationship graph for better selection
        comprehensive_graph = comprehensive_analysis.get('comprehensive_graph', {})
        nodes = {node['id']: node for node in comprehensive_graph.get('nodes', [])}
        edges = comprehensive_graph.get('edges', [])
        
        # Question pattern analysis with comprehensive entity types
        if any(word in question_lower for word in ['paid', 'customer', 'payment']):
            # Find high-confidence customer and payment entities
            customer_tables = [t for t in tables 
                             if nodes.get(t.full_name, {}).get('entity_type') == 'Customer'
                             and nodes.get(t.full_name, {}).get('confidence', 0) > 0.6]
            
            payment_tables = [t for t in tables 
                            if nodes.get(t.full_name, {}).get('entity_type') == 'Payment'
                            and nodes.get(t.full_name, {}).get('confidence', 0) > 0.6]
            
            # Add tables with validated relationships
            for customer_table in customer_tables[:2]:  # Top 2 customer tables
                selected_tables.append(customer_table)
                
                # Find connected payment tables via relationships
                for edge in edges:
                    if (edge['source'] == customer_table.full_name and 
                        edge['target'] in [pt.full_name for pt in payment_tables]):
                        target_table = next(t for t in tables if t.full_name == edge['target'])
                        if target_table not in selected_tables:
                            selected_tables.append(target_table)
        
        # Enhanced selection logic using comprehensive relationship graph
        if not selected_tables:
            # Fallback to highest confidence entities
            high_confidence_tables = [
                t for t in tables 
                if nodes.get(t.full_name, {}).get('confidence', 0) > 0.7
                and nodes.get(t.full_name, {}).get('entity_type') != 'Unknown'
            ]
            selected_tables.extend(high_confidence_tables[:5])
        
        return selected_tables

    async def _generate_comprehensive_sql(self, question: str, selected_tables: List[TableInfo], 
                                        relationships: List[Relationship],
                                        comprehensive_analysis: Dict[str, Any]) -> Optional[str]:
        """Generate SQL using comprehensive relationship analysis"""
        
        # Get comprehensive relationship information
        comprehensive_graph = comprehensive_analysis.get('comprehensive_graph', {})
        fk_relationships = comprehensive_analysis.get('foreign_key_relationships', [])
        
        # Build enhanced table context with relationship metadata
        table_context = []
        for table in selected_tables:
            # Get comprehensive entity information
            node_info = next((node for node in comprehensive_graph.get('nodes', []) 
                            if node['id'] == table.full_name), {})
            
            # Find all relationships for this table
            table_relationships = []
            for edge in comprehensive_graph.get('edges', []):
                if edge['source'] == table.full_name or edge['target'] == table.full_name:
                    table_relationships.append({
                        'type': edge.get('relationship_type', 'unknown'),
                        'confidence': edge.get('confidence', 0.0),
                        'business_context': edge.get('business_type', 'unknown')
                    })
            
            context = {
                'table_name': table.name,
                'full_name': table.full_name,
                'entity_type': node_info.get('entity_type', 'Unknown'),
                'business_role': node_info.get('business_role', 'Unknown'),
                'confidence': node_info.get('confidence', 0.0),
                'row_count': table.row_count,
                'relationships': table_relationships,
                'columns': [
                    {
                        'name': col['name'],
                        'type': col['data_type'],
                        'is_key': any('id' in col['name'].lower() for rel in fk_relationships
                                    if rel['parent_table'] == table.full_name 
                                    and rel['parent_column'] == col['name']),
                        'is_business_significant': any(word in col['name'].lower() 
                                                     for word in ['amount', 'total', 'name', 'date'])
                    } for col in table.columns[:15]
                ],
                'sample_data': table.sample_data[:2] if table.sample_data else []
            }
            table_context.append(context)
        
        # Create comprehensive SQL generation prompt
        prompt = self._create_comprehensive_sql_prompt(question, table_context, 
                                                     fk_relationships, comprehensive_analysis)
        
        try:
            system_message = """You are an expert SQL architect with deep understanding of business relationships and entity modeling.

Generate accurate SQL Server T-SQL that leverages comprehensive relationship analysis including:
- Foreign key constraints (highest reliability)
- View-based relationships (business logic insights)  
- LLM-discovered entity patterns (business context)

Focus on business-meaningful JOINs and proper relationship utilization. 
Respond with ONLY the SQL query - no explanations or markdown."""
            
            response = await self.llm.ask(prompt, system_message)
            cleaned_sql = clean_sql_response(response)
            return cleaned_sql
            
        except Exception as e:
            print(f"⚠️ Comprehensive SQL generation failed: {e}")
            return None
    
    def _create_comprehensive_sql_prompt(self, question: str, table_context: List[Dict],
                                       fk_relationships: List[Dict],
                                       comprehensive_analysis: Dict[str, Any]) -> str:
        """Create comprehensive SQL generation prompt with full relationship context"""
        
        business_intelligence = comprehensive_analysis.get('business_intelligence', {})
        
        prompt = f"""
BUSINESS QUESTION: "{question}"

COMPREHENSIVE TABLE ANALYSIS:
{json.dumps(table_context, indent=2)}

VALIDATED FOREIGN KEY RELATIONSHIPS (Highest Confidence):
{json.dumps([fk for fk in fk_relationships 
            if any(fk['parent_table'] == tc['full_name'] or fk['referenced_table'] == tc['full_name'] 
                  for tc in table_context)], indent=2)}

BUSINESS INTELLIGENCE CONTEXT:
{json.dumps(business_intelligence, indent=2)}

COMPREHENSIVE SQL GENERATION RULES:
1. Use validated foreign key relationships first (confidence = 1.0)
2. Consider view-based relationships for complex business logic
3. Leverage entity type information for appropriate JOINs
4. Use business role context to determine JOIN necessity
5. For customer-payment queries, prioritize validated FK relationships
6. Add meaningful column aliases based on business context
7. Include appropriate WHERE clauses based on entity patterns
8. Use INNER JOINs for validated relationships, LEFT JOINs for optional ones

Generate optimal SQL leveraging comprehensive relationship analysis:
"""
        return prompt


# Export classes with consistent naming for backward compatibility
EnhancedQueryInterface = IntelligentQueryInterface
QueryInterface = IntelligentQueryInterface
InteractiveLLMClient = IntelligentLLMClient

# Make all classes available at module level
__all__ = ['IntelligentQueryInterface', 'EnhancedQueryInterface', 'QueryInterface', 
           'IntelligentQueryProcessor', 'IntelligentLLMClient', 'InteractiveLLMClient']