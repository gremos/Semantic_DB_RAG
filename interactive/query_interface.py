#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTELLIGENT Query Interface - Relationship-Aware Business Query Processing
Advanced table selection and SQL generation using comprehensive relationship intelligence
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

class RelationshipIntelligenceEngine:
    """Engine for intelligent relationship-based table selection and query planning"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.comprehensive_analysis: Dict[str, Any] = {}
        self.relationship_graph = nx.Graph()
        
        # Build relationship intelligence patterns
        self.query_patterns = self._init_intelligent_query_patterns()
    
    def _init_intelligent_query_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intelligent query processing patterns with relationship awareness"""
        return {
            'paid_customers': {
                'keywords': ['paid', 'customer', 'payment', 'revenue', 'transaction'],
                'required_entities': ['Customer', 'Payment'],
                'relationship_requirements': ['customer_payment', 'customer_order'],
                'sql_strategy': 'inner_join_with_aggregation',
                'business_logic': 'Find customers with payment records using relationship intelligence',
                'confidence_boost': 0.3
            },
            'customer_analysis': {
                'keywords': ['customer', 'client', 'account', 'how many customers'],
                'required_entities': ['Customer'],
                'relationship_requirements': [],
                'sql_strategy': 'simple_aggregation',
                'business_logic': 'Analyze customer data using best customer entity',
                'confidence_boost': 0.2
            },
            'revenue_analysis': {
                'keywords': ['revenue', 'income', 'total', 'payment', 'earnings'],
                'required_entities': ['Payment'],
                'relationship_requirements': ['customer_payment'],
                'sql_strategy': 'sum_with_optional_joins',
                'business_logic': 'Calculate revenue using payment entities and customer context',
                'confidence_boost': 0.25
            },
            'order_analysis': {
                'keywords': ['order', 'purchase', 'sale', 'booking'],
                'required_entities': ['Order'],
                'relationship_requirements': ['customer_order', 'order_payment'],
                'sql_strategy': 'order_aggregation_with_joins',
                'business_logic': 'Analyze orders with customer and payment context',
                'confidence_boost': 0.2
            },
            'user_activity': {
                'keywords': ['user', 'task', 'assignment', 'work', 'activity'],
                'required_entities': ['User', 'Task'],
                'relationship_requirements': ['user_task'],
                'sql_strategy': 'user_task_aggregation',
                'business_logic': 'Analyze user activity using task assignments',
                'confidence_boost': 0.2
            }
        }
    
    def initialize_relationship_intelligence(self, tables: List[TableInfo], 
                                           relationships: List[Relationship],
                                           comprehensive_analysis: Dict[str, Any]):
        """Initialize the relationship intelligence engine"""
        self.tables = tables
        self.relationships = relationships
        self.comprehensive_analysis = comprehensive_analysis
        
        # Build relationship graph for intelligent navigation
        self._build_relationship_graph()
        
        print(f"🧠 Relationship Intelligence Initialized:")
        print(f"   • Tables: {len(self.tables)}")
        print(f"   • Relationships: {len(self.relationships)}")
        print(f"   • Graph nodes: {self.relationship_graph.number_of_nodes()}")
        print(f"   • Graph edges: {self.relationship_graph.number_of_edges()}")
    
    def _build_relationship_graph(self):
        """Build networkx graph for intelligent relationship navigation"""
        self.relationship_graph = nx.Graph()
        
        # Add nodes (tables) with entity information
        for table in self.tables:
            entity_type = 'Unknown'
            confidence = 0.0
            
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                confidence = table.semantic_profile.confidence
            
            self.relationship_graph.add_node(
                table.full_name,
                name=table.name,
                entity_type=entity_type,
                confidence=confidence,
                row_count=table.row_count,
                has_data=bool(table.sample_data)
            )
        
        # Add edges (relationships) with business context
        for rel in self.relationships:
            if (self.relationship_graph.has_node(rel.from_table) and 
                self.relationship_graph.has_node(rel.to_table)):
                
                business_significance = getattr(rel, 'business_significance', 0.5)
                
                self.relationship_graph.add_edge(
                    rel.from_table,
                    rel.to_table,
                    relationship_type=rel.relationship_type,
                    confidence=rel.confidence,
                    business_significance=business_significance,
                    column=rel.column,
                    description=rel.description
                )
    
    def analyze_query_intelligence(self, question: str) -> Dict[str, Any]:
        """Perform intelligent analysis of the query using relationship context"""
        question_lower = question.lower()
        
        # Find matching query patterns
        best_match = None
        best_score = 0
        
        for pattern_name, pattern_info in self.query_patterns.items():
            score = 0
            
            # Score based on keyword matches
            keywords = pattern_info['keywords']
            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
            
            # Normalize score
            if keywords:
                score = score / len(keywords)
            
            if score > best_score:
                best_score = score
                best_match = pattern_name
        
        # Get comprehensive analysis context
        entity_analysis = self.comprehensive_analysis.get('entity_analysis', {})
        business_insights = self.comprehensive_analysis.get('business_insights', {})
        
        if best_match and best_score > 0.3:
            pattern_info = self.query_patterns[best_match]
            
            return {
                'query_pattern': best_match,
                'confidence': best_score + pattern_info.get('confidence_boost', 0),
                'required_entities': pattern_info['required_entities'],
                'relationship_requirements': pattern_info['relationship_requirements'],
                'sql_strategy': pattern_info['sql_strategy'],
                'business_logic': pattern_info['business_logic'],
                'entity_context': entity_analysis,
                'business_context': business_insights
            }
        else:
            # General analysis for unmatched queries
            return {
                'query_pattern': 'general_business',
                'confidence': 0.5,
                'required_entities': self._infer_entities_from_question(question_lower),
                'relationship_requirements': [],
                'sql_strategy': 'adaptive',
                'business_logic': 'General business query requiring entity and relationship analysis',
                'entity_context': entity_analysis,
                'business_context': business_insights
            }
    
    def _infer_entities_from_question(self, question_lower: str) -> List[str]:
        """Infer required entities from question text"""
        entities = []
        
        entity_keywords = {
            'Customer': ['customer', 'client', 'account', 'businesspoint'],
            'Payment': ['payment', 'transaction', 'revenue', 'income', 'billing'],
            'Order': ['order', 'purchase', 'sale', 'booking'],
            'User': ['user', 'employee', 'staff', 'person'],
            'Task': ['task', 'assignment', 'work', 'activity'],
            'Product': ['product', 'item', 'service']
        }
        
        for entity, keywords in entity_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                entities.append(entity)
        
        return entities if entities else ['Customer', 'Payment', 'Order']  # Default fallback
    
    def find_optimal_tables_with_relationships(self, query_analysis: Dict[str, Any]) -> Tuple[List[TableInfo], Dict[str, Any]]:
        """Find optimal tables using relationship intelligence"""
        
        required_entities = query_analysis['required_entities']
        relationship_requirements = query_analysis['relationship_requirements']
        
        # Get entity classifications
        entity_context = query_analysis.get('entity_context', {})
        classifications = entity_context.get('classifications', {})
        high_confidence_entities = entity_context.get('high_confidence_entities', {})
        
        selected_tables = []
        relationship_context = {}
        
        # Step 1: Select best table for each required entity
        for entity_type in required_entities:
            best_table = self._find_best_table_for_entity(entity_type, high_confidence_entities, classifications)
            if best_table:
                selected_tables.append(best_table)
                print(f"   📋 Selected {best_table.name} for {entity_type} entity")
        
        # Step 2: Add relationship-required tables
        if relationship_requirements:
            additional_tables = self._find_relationship_required_tables(
                selected_tables, relationship_requirements
            )
            for table in additional_tables:
                if table not in selected_tables:
                    selected_tables.append(table)
                    print(f"   🔗 Added {table.name} for relationship requirements")
        
        # Step 3: Use graph analysis to find connecting tables
        if len(selected_tables) > 1:
            connecting_tables = self._find_connecting_tables(selected_tables)
            for table in connecting_tables:
                if table not in selected_tables and len(selected_tables) < 5:
                    selected_tables.append(table)
                    print(f"   🌉 Added {table.name} as connecting table")
        
        # Step 4: Build relationship context for SQL generation
        relationship_context = self._build_query_relationship_context(selected_tables)
        
        return selected_tables, relationship_context
    
    def _find_best_table_for_entity(self, entity_type: str, 
                                   high_confidence_entities: Dict[str, List],
                                   classifications: Dict[str, Dict]) -> Optional[TableInfo]:
        """Find the best table for a specific entity type"""
        
        # First try high confidence entities
        entity_candidates = high_confidence_entities.get(entity_type, [])
        if entity_candidates:
            # Sort by confidence and relationship score
            best_candidate = max(entity_candidates, key=lambda x: (
                x.get('confidence', 0),
                x.get('relationship_score', 0)
            ))
            
            # Find the actual table object
            candidate_name = best_candidate['table_name']
            for table in self.tables:
                if table.full_name == candidate_name:
                    return table
        
        # Fallback: search all classifications
        for table_name, classification in classifications.items():
            if classification.get('entity_type') == entity_type:
                for table in self.tables:
                    if table.full_name == table_name:
                        return table
        
        return None
    
    def _find_relationship_required_tables(self, selected_tables: List[TableInfo],
                                         relationship_requirements: List[str]) -> List[TableInfo]:
        """Find additional tables required for specific relationships"""
        
        additional_tables = []
        selected_table_names = {table.full_name for table in selected_tables}
        
        for req_relationship in relationship_requirements:
            # Find relationships of this type
            matching_relationships = [
                rel for rel in self.relationships
                if rel.relationship_type == req_relationship or 
                   req_relationship in rel.relationship_type
            ]
            
            for rel in matching_relationships:
                # Add missing tables from relationships
                if rel.from_table not in selected_table_names:
                    table = self._find_table_by_name(rel.from_table)
                    if table:
                        additional_tables.append(table)
                
                if rel.to_table not in selected_table_names:
                    table = self._find_table_by_name(rel.to_table)
                    if table:
                        additional_tables.append(table)
        
        return additional_tables
    
    def _find_connecting_tables(self, selected_tables: List[TableInfo]) -> List[TableInfo]:
        """Find tables that connect the selected tables through relationships"""
        
        connecting_tables = []
        selected_names = {table.full_name for table in selected_tables}
        
        if len(selected_names) < 2:
            return connecting_tables
        
        # Use graph analysis to find shortest paths
        for i, table1 in enumerate(selected_tables):
            for table2 in selected_tables[i+1:]:
                try:
                    if (self.relationship_graph.has_node(table1.full_name) and 
                        self.relationship_graph.has_node(table2.full_name)):
                        
                        path = nx.shortest_path(
                            self.relationship_graph, 
                            table1.full_name, 
                            table2.full_name
                        )
                        
                        # Add intermediate tables in the path
                        for table_name in path[1:-1]:  # Exclude start and end
                            if table_name not in selected_names:
                                table = self._find_table_by_name(table_name)
                                if table and table not in connecting_tables:
                                    connecting_tables.append(table)
                
                except nx.NetworkXNoPath:
                    continue  # No path found, skip
        
        return connecting_tables
    
    def _find_table_by_name(self, full_name: str) -> Optional[TableInfo]:
        """Find table by full name"""
        for table in self.tables:
            if table.full_name == full_name:
                return table
        return None
    
    def _build_query_relationship_context(self, selected_tables: List[TableInfo]) -> Dict[str, Any]:
        """Build relationship context for SQL generation"""
        
        table_names = {table.full_name for table in selected_tables}
        relevant_relationships = []
        
        # Find relationships between selected tables
        for rel in self.relationships:
            if rel.from_table in table_names and rel.to_table in table_names:
                relevant_relationships.append({
                    'from_table': rel.from_table,
                    'to_table': rel.to_table,
                    'column': rel.column,
                    'relationship_type': rel.relationship_type,
                    'confidence': rel.confidence,
                    'business_significance': getattr(rel, 'business_significance', 0.5),
                    'description': rel.description
                })
        
        # Build join recommendations
        join_recommendations = []
        for rel in relevant_relationships:
            if rel['confidence'] > 0.6:  # Only confident relationships
                join_recommendations.append({
                    'tables': [rel['from_table'], rel['to_table']],
                    'join_condition': rel['column'],
                    'join_type': 'INNER' if rel['business_significance'] > 0.7 else 'LEFT',
                    'business_reason': rel['description']
                })
        
        return {
            'relationships': relevant_relationships,
            'join_recommendations': join_recommendations,
            'table_priorities': self._calculate_table_priorities(selected_tables),
            'entity_mapping': self._build_entity_mapping(selected_tables)
        }
    
    def _calculate_table_priorities(self, tables: List[TableInfo]) -> Dict[str, float]:
        """Calculate priority scores for tables in query context"""
        priorities = {}
        
        for table in tables:
            priority = 0.5  # Base priority
            
            # Boost for core business entities
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                if entity_type in ['Customer', 'Payment', 'Order']:
                    priority += 0.3
                
                # Boost for high confidence
                priority += table.semantic_profile.confidence * 0.2
            
            # Boost for tables with data
            if table.sample_data:
                priority += 0.1
            
            # Boost for central tables (high relationship count)
            if self.relationship_graph.has_node(table.full_name):
                degree = self.relationship_graph.degree(table.full_name)
                priority += min(0.2, degree * 0.05)
            
            priorities[table.full_name] = min(1.0, priority)
        
        return priorities
    
    def _build_entity_mapping(self, tables: List[TableInfo]) -> Dict[str, str]:
        """Build mapping of entity types to table names"""
        mapping = {}
        
        for table in tables:
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                if entity_type not in mapping:
                    mapping[entity_type] = table.name
        
        return mapping


class IntelligentQueryProcessor:
    """Intelligent query processor with comprehensive relationship awareness"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = IntelligentLLMClient(config)
        self.relationship_engine = RelationshipIntelligenceEngine(config)
    
    async def process_intelligent_query(self, question: str, 
                                      tables: List[TableInfo],
                                      domain: Optional[BusinessDomain],
                                      relationships: List[Relationship],
                                      comprehensive_analysis: Dict[str, Any]) -> QueryResult:
        """Process query with full relationship intelligence"""
        
        try:
            # Initialize relationship intelligence
            self.relationship_engine.initialize_relationship_intelligence(
                tables, relationships, comprehensive_analysis
            )
            
            # Step 1: Intelligent query analysis
            print(f"   🧠 Analyzing query with relationship intelligence...")
            query_analysis = self.relationship_engine.analyze_query_intelligence(question)
            
            pattern = query_analysis['query_pattern']
            confidence = query_analysis['confidence']
            
            print(f"   📋 Query pattern: {pattern} (confidence: {confidence:.2f})")
            print(f"   🎯 Required entities: {', '.join(query_analysis['required_entities'])}")
            
            # Step 2: Intelligent table selection with relationships
            print(f"   🔍 Selecting optimal tables with relationship context...")
            selected_tables, relationship_context = self.relationship_engine.find_optimal_tables_with_relationships(query_analysis)
            
            if not selected_tables:
                return self._create_error_result(question, "No suitable tables found for query")
            
            # Step 3: Generate intelligent SQL with relationship awareness
            print(f"   ⚡ Generating SQL with relationship intelligence...")
            sql_query = await self._generate_intelligent_sql(
                question, query_analysis, selected_tables, relationship_context
            )
            
            if not sql_query:
                return self._create_error_result(question, "Failed to generate SQL query")
            
            print(f"   💾 Generated query: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
            
            # Step 4: Execute with enhanced error handling
            print(f"   🚀 Executing intelligent query...")
            execution_result = await self._execute_sql_intelligent(sql_query)
            
            # Step 5: Create enhanced result with business context
            return self._create_intelligent_result(
                question, selected_tables, sql_query, execution_result, 
                query_analysis, relationship_context
            )
            
        except Exception as e:
            print(f"   ❌ Intelligent query processing failed: {e}")
            return self._create_error_result(question, f"Processing failed: {str(e)}")
    
    async def _generate_intelligent_sql(self, question: str, 
                                      query_analysis: Dict[str, Any],
                                      selected_tables: List[TableInfo],
                                      relationship_context: Dict[str, Any]) -> Optional[str]:
        """Generate SQL using comprehensive relationship intelligence"""
        
        sql_strategy = query_analysis['sql_strategy']
        pattern = query_analysis['query_pattern']
        
        # Prepare comprehensive context for LLM
        table_context = []
        for table in selected_tables:
            entity_type = 'Unknown'
            business_purpose = 'Unknown'
            
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                business_purpose = table.semantic_profile.primary_purpose
            
            context = {
                'table_name': table.name,
                'full_name': table.full_name,
                'entity_type': entity_type,
                'business_purpose': business_purpose,
                'row_count': table.row_count,
                'columns': [
                    {
                        'name': col['name'],
                        'type': col['data_type'],
                        'is_id': col['name'].lower().endswith('id'),
                        'is_amount': any(word in col['name'].lower() for word in ['amount', 'total', 'price', 'cost', 'value']),
                        'is_date': any(word in col['name'].lower() for word in ['date', 'time', 'created', 'modified']),
                        'is_name': any(word in col['name'].lower() for word in ['name', 'title', 'description'])
                    } for col in table.columns[:20]
                ],
                'sample_data': table.sample_data[:3] if table.sample_data else []
            }
            table_context.append(context)
        
        # Create specialized prompt based on query pattern and relationships
        if pattern == 'paid_customers':
            prompt = self._create_intelligent_paid_customers_prompt(
                question, table_context, relationship_context
            )
        elif pattern == 'customer_analysis':
            prompt = self._create_intelligent_customer_prompt(
                question, table_context, relationship_context
            )
        elif pattern == 'revenue_analysis':
            prompt = self._create_intelligent_revenue_prompt(
                question, table_context, relationship_context
            )
        elif pattern == 'user_activity':
            prompt = self._create_intelligent_user_activity_prompt(
                question, table_context, relationship_context
            )
        else:
            prompt = self._create_intelligent_general_prompt(
                question, table_context, relationship_context, query_analysis
            )
        
        try:
            system_message = """You are an expert SQL architect specializing in relationship-aware query generation.
Generate accurate SQL Server T-SQL that leverages relationship intelligence for optimal business queries.
Use the provided relationship context to create proper JOINs and business logic.
Respond with ONLY the SQL query - no explanations or markdown."""
            
            response = await self.llm.ask(prompt, system_message)
            cleaned_sql = clean_sql_response(response)
            return cleaned_sql
            
        except Exception as e:
            print(f"⚠️ Intelligent SQL generation failed: {e}")
            return None
    
    def _create_intelligent_paid_customers_prompt(self, question: str, 
                                                table_context: List[Dict],
                                                relationship_context: Dict[str, Any]) -> str:
        """Create intelligent prompt for paid customer queries with relationship awareness"""
        
        customer_tables = [t for t in table_context if t['entity_type'] == 'Customer']
        payment_tables = [t for t in table_context if t['entity_type'] == 'Payment']
        join_recommendations = relationship_context.get('join_recommendations', [])
        
        prompt = f"""
INTELLIGENT BUSINESS QUERY: "{question}"

This is a PAID CUSTOMER analysis query requiring relationship intelligence.

CUSTOMER ENTITIES:
{json.dumps(customer_tables, indent=2)}

PAYMENT ENTITIES:
{json.dumps(payment_tables, indent=2)}

RELATIONSHIP INTELLIGENCE:
{json.dumps(join_recommendations, indent=2)}

INTELLIGENT SQL GENERATION RULES:

1. **Relationship-Aware JOINs**:
   - Use provided join recommendations for optimal table connections
   - Prefer INNER JOINs for paid customer analysis (exclude non-paying customers)
   - Use relationship context to determine correct join columns

2. **Business Logic Intelligence**:
   - Count DISTINCT customers to avoid duplicates from multiple payments
   - Filter for meaningful payment amounts (> 0, NOT NULL)
   - Consider date filters if year/period mentioned in question

3. **Query Optimization**:
   - Use table priorities to determine primary table for FROM clause
   - Leverage entity mapping for consistent column references
   - Apply business significance weighting for JOIN order

4. **Result Formatting**:
   - Return business-meaningful column names
   - Include relevant business context (payment amounts, dates, etc.)
   - Limit results appropriately (TOP 100 unless otherwise specified)

EXAMPLE INTELLIGENT PATTERN:
```sql
SELECT COUNT(DISTINCT c.CustomerID) as PaidCustomerCount
FROM [HighestPriorityCustomerTable] c
INNER JOIN [HighestPriorityPaymentTable] p ON [OptimalJoinCondition]
WHERE p.Amount > 0 
  AND p.PaymentDate >= '2025-01-01'  -- if year specified
```

Generate the optimal SQL query using relationship intelligence:
"""
        return prompt
    
    def _create_intelligent_customer_prompt(self, question: str,
                                          table_context: List[Dict],
                                          relationship_context: Dict[str, Any]) -> str:
        """Create intelligent prompt for customer analysis queries"""
        
        customer_tables = [t for t in table_context if t['entity_type'] == 'Customer']
        entity_mapping = relationship_context.get('entity_mapping', {})
        
        prompt = f"""
INTELLIGENT BUSINESS QUERY: "{question}"

This is a CUSTOMER ANALYSIS query using relationship intelligence.

CUSTOMER ENTITIES:
{json.dumps(customer_tables, indent=2)}

ENTITY MAPPING:
{json.dumps(entity_mapping, indent=2)}

INTELLIGENT ANALYSIS STRATEGY:
- Use the highest priority customer table as primary source
- Apply appropriate aggregation (COUNT, DISTINCT, etc.)
- Consider business context for filtering (active customers, registered customers, etc.)
- Use meaningful business column names in results

Generate optimal customer analysis SQL:
"""
        return prompt
    
    def _create_intelligent_revenue_prompt(self, question: str,
                                         table_context: List[Dict],
                                         relationship_context: Dict[str, Any]) -> str:
        """Create intelligent prompt for revenue analysis queries"""
        
        payment_tables = [t for t in table_context if t['entity_type'] == 'Payment']
        join_recommendations = relationship_context.get('join_recommendations', [])
        
        prompt = f"""
INTELLIGENT BUSINESS QUERY: "{question}"

This is a REVENUE ANALYSIS query requiring financial intelligence.

PAYMENT/FINANCIAL ENTITIES:
{json.dumps(payment_tables, indent=2)}

RELATIONSHIP CONTEXT:
{json.dumps(join_recommendations, indent=2)}

FINANCIAL INTELLIGENCE RULES:
- SUM payment amounts with proper NULL handling
- Use appropriate date filtering for period analysis
- Consider currency/denomination consistency
- Apply business logic for valid payments (positive amounts, completed status)
- Include meaningful grouping if analysis requires breakdowns

Generate optimal revenue analysis SQL:
"""
        return prompt
    
    def _create_intelligent_user_activity_prompt(self, question: str,
                                               table_context: List[Dict],
                                               relationship_context: Dict[str, Any]) -> str:
        """Create intelligent prompt for user activity queries"""
        
        user_tables = [t for t in table_context if t['entity_type'] == 'User']
        task_tables = [t for t in table_context if t['entity_type'] == 'Task']
        join_recommendations = relationship_context.get('join_recommendations', [])
        
        prompt = f"""
INTELLIGENT BUSINESS QUERY: "{question}"

This is a USER ACTIVITY analysis query with workflow intelligence.

USER ENTITIES:
{json.dumps(user_tables, indent=2)}

TASK/WORK ENTITIES:
{json.dumps(task_tables, indent=2)}

WORKFLOW RELATIONSHIPS:
{json.dumps(join_recommendations, indent=2)}

ACTIVITY INTELLIGENCE RULES:
- Use user-task relationships for activity analysis
- Consider assignment dates, completion status, task types
- Apply appropriate aggregation for activity metrics
- Include meaningful activity indicators (task counts, completion rates, etc.)

Generate optimal user activity analysis SQL:
"""
        return prompt
    
    def _create_intelligent_general_prompt(self, question: str,
                                         table_context: List[Dict],
                                         relationship_context: Dict[str, Any],
                                         query_analysis: Dict[str, Any]) -> str:
        """Create intelligent prompt for general business queries"""
        
        business_logic = query_analysis['business_logic']
        relationships = relationship_context.get('relationships', [])
        
        prompt = f"""
INTELLIGENT BUSINESS QUERY: "{question}"

BUSINESS LOGIC: {business_logic}

AVAILABLE ENTITIES:
{json.dumps(table_context, indent=2)}

RELATIONSHIP INTELLIGENCE:
{json.dumps(relationships, indent=2)}

INTELLIGENT QUERY STRATEGY:
1. Use relationship context to determine optimal table joins
2. Apply business logic appropriate to the question
3. Consider entity priorities and data quality
4. Generate business-meaningful results with proper aggregation
5. Use relationship intelligence to avoid cartesian products

Generate optimal business query SQL using available intelligence:
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
            
            # Intelligent error classification and recovery suggestions
            if 'invalid object name' in error_msg.lower():
                error_msg = f"Table/view not found. Relationship intelligence may need recalibration: {error_msg[:100]}"
            elif 'invalid column name' in error_msg.lower():
                error_msg = f"Column not found. Check entity-to-column mapping: {error_msg[:100]}"
            elif 'timeout' in error_msg.lower():
                error_msg = f"Query timeout. Consider relationship optimization: {error_msg[:100]}"
            elif 'join' in error_msg.lower():
                error_msg = f"JOIN error. Relationship context may be incorrect: {error_msg[:100]}"
            else:
                error_msg = error_msg[:200]
            
            return {'data': [], 'count': 0, 'error': error_msg}
    
    def _create_intelligent_result(self, question: str, selected_tables: List[TableInfo],
                                 sql_query: str, execution_result: Dict[str, Any],
                                 query_analysis: Dict[str, Any],
                                 relationship_context: Dict[str, Any]) -> QueryResult:
        """Create enhanced query result with business intelligence"""
        
        table_names = [t.name for t in selected_tables]
        entity_types = [t.semantic_profile.entity_type if t.semantic_profile else 'Unknown' for t in selected_tables]
        relationships_used = [rel['relationship_type'] for rel in relationship_context.get('relationships', [])]
        
        # Determine query complexity
        complexity = 'Simple'
        if len(selected_tables) > 2:
            complexity = 'Medium'
        if len(relationships_used) > 2 or 'JOIN' in sql_query.upper():
            complexity = 'Complex'
        
        # Determine business significance
        pattern = query_analysis.get('query_pattern', 'general')
        if pattern in ['paid_customers', 'revenue_analysis']:
            significance = 'High'
        elif pattern in ['customer_analysis', 'order_analysis']:
            significance = 'Medium'
        else:
            significance = 'Low'
        
        # Generate business interpretation
        business_interpretation = self._generate_business_interpretation(
            question, execution_result, query_analysis, entity_types
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
            business_significance=significance,
            entities_involved=entity_types,
            relationships_used=relationships_used,
            business_interpretation=business_interpretation
        )
    
    def _generate_business_interpretation(self, question: str, execution_result: Dict[str, Any],
                                        query_analysis: Dict[str, Any], entity_types: List[str]) -> str:
        """Generate intelligent business interpretation of results"""
        
        results = execution_result.get('data', [])
        count = execution_result.get('count', 0)
        error = execution_result.get('error')
        pattern = query_analysis.get('query_pattern', 'general')
        
        if error:
            return f"Query execution failed: {error}. This may indicate relationship mapping issues or data quality problems."
        
        if count == 0:
            if pattern == 'paid_customers':
                return "No paid customers found. This suggests either no payment data exists, or customer-payment relationships need verification."
            elif pattern == 'revenue_analysis':
                return "No revenue data found. Check payment/transaction tables and data availability."
            else:
                return f"No results found for {', '.join(entity_types)} analysis. Data may be missing or filtering too restrictive."
        
        # Generate positive interpretations
        if pattern == 'paid_customers' and results:
            first_result = results[0]
            if 'count' in str(first_result).lower():
                return f"Found {count} paid customers. This indicates healthy customer-payment relationship data."
        
        elif pattern == 'revenue_analysis' and results:
            first_result = results[0]
            for key, value in first_result.items():
                if isinstance(value, (int, float)) and value > 0:
                    return f"Revenue analysis shows {key}: {value:,.2f}. Financial data appears well-structured."
        
        elif pattern == 'customer_analysis':
            return f"Customer analysis returned {count} records. {', '.join(entity_types)} entities are properly accessible."
        
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
    """Intelligent query interface with comprehensive relationship awareness"""
    
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
        """Start intelligent interactive query session with relationship awareness"""
        
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
        
        print("💡 Type 'help' for intelligent examples, 'relationships' for network, 'entities' for classification")
    
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
                print("   • Show me customer details and activity")
                
                if capabilities.get('paid_customer_analysis', False):
                    print("   • How many customers have made payments? (INTELLIGENT: uses relationship discovery)")
                    print("   • Which customers are our top spenders?")
                    print("   • Show paid customers for 2025")
            
            if capabilities.get('payment_queries', False):
                print("\n💰 REVENUE INTELLIGENCE:")
                print("   • What is our total revenue?")
                print("   • Show payment trends with customer context")
                print("   • Calculate average transaction value per customer")
            
            if entity_counts.get('Task', 0) > 0 and entity_counts.get('User', 0) > 0:
                print("\n👷 WORKFLOW INTELLIGENCE:")
                print("   • How many tasks are assigned to users?")
                print("   • Which users have the highest workload?")
                print("   • Show task completion rates by user")
        
        print("\n🧠 INTELLIGENT FEATURES:")
        print("   • Automatic relationship-aware table selection")
        print("   • Business context understanding")
        print("   • Optimal JOIN generation using discovered relationships")
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
        
        # Group by relationship type and business significance
        high_significance = []
        medium_significance = []
        low_significance = []
        
        for rel in self.relationships:
            significance = getattr(rel, 'business_significance', 0.5)
            rel_info = f"{rel.from_table.split('.')[-1]} → {rel.to_table.split('.')[-1]} ({rel.relationship_type})"
            
            if significance >= 0.8:
                high_significance.append(rel_info)
            elif significance >= 0.5:
                medium_significance.append(rel_info)
            else:
                low_significance.append(rel_info)
        
        if high_significance:
            print(f"   🔥 HIGH BUSINESS SIGNIFICANCE:")
            for rel in high_significance[:5]:
                print(f"      • {rel}")
        
        if medium_significance:
            print(f"   ⚡ MEDIUM BUSINESS SIGNIFICANCE:")
            for rel in medium_significance[:5]:
                print(f"      • {rel}")
        
        if low_significance:
            print(f"   📝 SUPPORTING RELATIONSHIPS:")
            for rel in low_significance[:3]:
                print(f"      • {rel}")
        
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
        core_entities = ['Customer', 'Payment', 'Order', 'User', 'Task']
        for entity in core_entities:
            if entity in high_confidence and high_confidence[entity]:
                best = high_confidence[entity][0]
                print(f"   🎯 Best {entity} table: {best['table_name']} (confidence: {best['confidence']:.2f})")
    
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
            print(f"   • Entities involved: {', '.join(result.entities_involved)}")
            if result.relationships_used:
                print(f"   • Relationships used: {', '.join(result.relationships_used)}")
            print(f"   • Business interpretation: {result.business_interpretation}")
            
            # Show execution metrics
            if result.execution_time > 0:
                print(f"⚡ Execution: {result.execution_time:.3f}s")
            
            # Show tables used
            if result.relevant_tables:
                print(f"📋 Tables: {', '.join(result.relevant_tables)}")


# Export the intelligent interface as the main class
QueryInterface = IntelligentQueryInterface

# Make all classes available at module level
__all__ = ['IntelligentQueryInterface', 'QueryInterface', 'IntelligentQueryProcessor', 
           'RelationshipIntelligenceEngine', 'IntelligentLLMClient']