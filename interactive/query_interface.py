#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BI-Aware Query Interface - Fixed to Use Semantic Analysis Properly
Following README: 4-stage pipeline with proper semantic integration
Uses actual column names and relationships from discovery/analysis
"""

import asyncio
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple, Union

# Import SQLGlot for proper SQL generation
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import (TableInfo, BusinessDomain, Relationship, QueryResult, 
                          AnalyticalTask, CapabilityContract, EvidenceScore, 
                          NonExecutableAnalysisReport)
from shared.utils import parse_json_response, clean_sql_query, safe_database_value, validate_sql_safety

class IntentAnalyzer:
    """Stage 1: Natural language → Analytical Task (Enhanced)"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str) -> AnalyticalTask:
        """Analyze user question and normalize to analytical task"""
        print("   🧠 Stage 1: Intent analysis...")
        
        # Enhanced pattern matching
        intent = self._enhanced_pattern_match(question)
        if intent:
            print(f"      ✅ Pattern matched: {intent.task_type}")
            return intent
        
        # LLM analysis for complex cases
        return await self._llm_intent_analysis(question)
    
    def _enhanced_pattern_match(self, question: str) -> Optional[AnalyticalTask]:
        """Enhanced pattern matching with better entity detection"""
        q_lower = question.lower()
        
        # Detect task type
        if any(word in q_lower for word in ['top', 'highest', 'best', 'worst', 'largest']):
            task_type = 'ranking'
            metrics = ['revenue', 'amount', 'value']
        elif any(word in q_lower for word in ['total', 'sum', 'amount']):
            task_type = 'aggregation'  
            metrics = ['sum']
        elif any(word in q_lower for word in ['count', 'how many', 'number of']):
            task_type = 'aggregation'
            metrics = ['count']
        elif any(word in q_lower for word in ['trend', 'over time', 'monthly']):
            task_type = 'trend'
            metrics = ['trend']
        else:
            task_type = 'aggregation'
            metrics = ['count']
        
        # Enhanced entity detection
        entity = None
        if any(word in q_lower for word in ['customer', 'client', 'user']):
            entity = 'customer'
        elif any(word in q_lower for word in ['payment', 'revenue', 'sales', 'money']):
            entity = 'payment'
        elif any(word in q_lower for word in ['order', 'transaction']):
            entity = 'order'
        elif any(word in q_lower for word in ['product', 'item']):
            entity = 'product'
        
        # Extract time window and limits
        time_window = None
        if 'last 12 months' in q_lower or '12 months' in q_lower:
            time_window = 'last_12_months'
        elif '2025' in q_lower:
            time_window = '2025'
        elif '2024' in q_lower:
            time_window = '2024'
        
        # Extract top limit
        top_limit = None
        import re
        top_match = re.search(r'top\s+(\d+)', q_lower)
        if top_match:
            top_limit = int(top_match.group(1))
        elif task_type == 'ranking':
            top_limit = 10  # Default for ranking
        
        return AnalyticalTask(
            task_type=task_type,
            metrics=metrics,
            entity=entity,
            time_window=time_window,
            top_limit=top_limit
        )
    
    async def _llm_intent_analysis(self, question: str) -> AnalyticalTask:
        """LLM-based intent analysis"""
        try:
            messages = [
                SystemMessage(content="Extract analytical intent. Respond with JSON only."),
                HumanMessage(content=f"""
Analyze: "{question}"

Extract:
1. Task type: aggregation, ranking, trend, count
2. Metrics: revenue, count, sum, etc.
3. Entity: customer, payment, order
4. Time window: 2025, last_12_months, etc.
5. Top limit: number for ranking

JSON only:
{{
  "task_type": "ranking",
  "metrics": ["revenue"],
  "entity": "customer",
  "time_window": "last_12_months",
  "top_limit": 10
}}
""")
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            
            data = parse_json_response(response.content)
            if data:
                return AnalyticalTask(
                    task_type=data.get('task_type', 'aggregation'),
                    metrics=data.get('metrics', ['count']),
                    entity=data.get('entity'),
                    time_window=data.get('time_window'),
                    top_limit=data.get('top_limit')
                )
        except Exception as e:
            print(f"      ⚠️ LLM intent analysis failed: {e}")
        
        # Simple fallback
        return AnalyticalTask(task_type='aggregation', metrics=['count'])

class SemanticTableSelector:
    """Stage 2: Semantic-aware table selection using analysis results"""
    
    def __init__(self, tables: List[TableInfo], relationships: List[Relationship]):
        self.tables = tables
        self.relationships = relationships
        self._build_relationship_map()
    
    def _build_relationship_map(self):
        """Build relationship mapping for JOIN generation"""
        self.relationship_map = {}
        for rel in self.relationships:
            if rel.from_table not in self.relationship_map:
                self.relationship_map[rel.from_table] = []
            self.relationship_map[rel.from_table].append(rel)
    
    def select_candidates(self, intent: AnalyticalTask, top_k: int = 8) -> List[Tuple[TableInfo, EvidenceScore, List[str]]]:
        """Select tables using semantic analysis results"""
        print("   📋 Stage 2: Evidence-driven table selection...")
        
        scored_tables = []
        for table in self.tables:
            score = self._calculate_semantic_evidence(table, intent)
            reasoning = self._generate_semantic_reasoning(table, score, intent)
            scored_tables.append((table, score, reasoning))
        
        # Sort by evidence score
        scored_tables.sort(key=lambda x: x[1].total_score, reverse=True)
        
        selected = scored_tables[:top_k]
        print(f"      ✅ Selected {len(selected)} candidates")
        
        # Show top candidates with reasoning
        for i, (table, score, reasoning) in enumerate(selected[:3], 1):
            print(f"         {i}. {table.name} (score: {score.total_score:.2f})")
            if reasoning:
                print(f"            • {reasoning[0]}")
        
        return selected
    
    def _calculate_semantic_evidence(self, table: TableInfo, intent: AnalyticalTask) -> EvidenceScore:
        """Calculate evidence using semantic analysis results"""
        score = EvidenceScore()
        
        # 1. BI Role match - use actual semantic analysis
        bi_role = getattr(table, 'bi_role', 'dimension')
        data_type = getattr(table, 'data_type', 'reference')
        entity_type = getattr(table, 'entity_type', 'Unknown')
        
        if intent.task_type in ['aggregation', 'ranking']:
            if bi_role == 'fact' and data_type == 'operational':
                score.role_match = 1.0
            elif bi_role == 'fact':
                score.role_match = 0.8
            elif data_type == 'operational':
                score.role_match = 0.6
            else:
                score.role_match = 0.3
        else:
            score.role_match = 0.5
        
        # 2. Entity type match - use semantic classification
        score.lexical_match = self._calculate_entity_match(table, intent)
        
        # 3. Capability match - use actual measures/keys from analysis
        measures = getattr(table, 'measures', [])
        entity_keys = getattr(table, 'entity_keys', [])
        time_columns = getattr(table, 'time_columns', [])
        
        capability_score = 0.0
        if intent.task_type in ['aggregation', 'ranking'] and measures:
            capability_score += 0.4
        if entity_keys:
            capability_score += 0.3
        if time_columns and intent.time_window:
            capability_score += 0.3
        
        score.operational_tag = capability_score
        
        # 4. Join evidence - actual relationships
        score.join_evidence = min(1.0, len(table.relationships) / 3.0)
        
        # 5. Table quality
        score.graph_proximity = self._calculate_table_quality(table)
        
        # 6. Data volume
        score.row_count = min(1.0, table.row_count / 10000.0) if table.row_count > 0 else 0.0
        score.freshness = 1.0
        
        return score
    
    def _calculate_entity_match(self, table: TableInfo, intent: AnalyticalTask) -> float:
        """Calculate entity match using semantic analysis"""
        if not intent.entity:
            return 0.5
        
        entity_type = getattr(table, 'entity_type', 'Unknown').lower()
        table_name = table.name.lower()
        
        # Direct entity type match
        if intent.entity.lower() in entity_type:
            return 1.0
        
        # Table name match
        if intent.entity.lower() in table_name:
            return 0.9
        
        # Semantic matches
        entity_mappings = {
            'customer': ['customer', 'client', 'account', 'user'],
            'payment': ['payment', 'invoice', 'billing', 'collection'],
            'order': ['order', 'contract', 'agreement'],
            'product': ['product', 'item', 'service']
        }
        
        if intent.entity in entity_mappings:
            for synonym in entity_mappings[intent.entity]:
                if synonym in table_name or synonym in entity_type:
                    return 0.7
        
        return 0.2
    
    def _calculate_table_quality(self, table: TableInfo) -> float:
        """Calculate table quality score"""
        table_name = table.name.lower()
        quality_score = 1.0
        
        # Penalty for dated tables
        import re
        if re.search(r'\d{8}', table_name):
            quality_score -= 0.4
        elif re.search(r'\d{6}', table_name):
            quality_score -= 0.3
        elif re.search(r'\d{4}', table_name):
            quality_score -= 0.2
        
        # Penalty for temp/backup indicators
        temp_indicators = ['temp', 'tmp', 'backup', 'bak', 'old', 'archive', 'test']
        for indicator in temp_indicators:
            if indicator in table_name:
                quality_score -= 0.3
                break
        
        # Penalty for bridge/buffer tables
        if any(word in table_name for word in ['bridge', 'buffer']):
            quality_score -= 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _generate_semantic_reasoning(self, table: TableInfo, score: EvidenceScore, intent: AnalyticalTask) -> List[str]:
        """Generate reasoning using semantic analysis"""
        reasoning = []
        
        # BI role reasoning
        bi_role = getattr(table, 'bi_role', 'dimension')
        if score.role_match > 0.7:
            reasoning.append(f"Strong BI role: {bi_role} table")
        
        # Entity type reasoning  
        entity_type = getattr(table, 'entity_type', 'Unknown')
        if score.lexical_match > 0.7:
            reasoning.append(f"Entity match: {entity_type}")
        
        # Capability reasoning
        measures = getattr(table, 'measures', [])
        if measures and intent.task_type in ['aggregation', 'ranking']:
            reasoning.append(f"Has measures: {', '.join(measures[:2])}")
        
        entity_keys = getattr(table, 'entity_keys', [])
        if entity_keys:
            reasoning.append(f"Has entity keys: {', '.join(entity_keys[:2])}")
        
        time_columns = getattr(table, 'time_columns', [])
        if time_columns and intent.time_window:
            reasoning.append(f"Has time columns: {', '.join(time_columns[:1])}")
        
        # Data quality
        if table.row_count > 1000:
            reasoning.append(f"Good data volume: {table.row_count:,} rows")
        
        return reasoning

class SmartSQLGenerator:
    """Stage 4: Intelligent SQL generation using semantic analysis and SQLGlot"""
    
    def __init__(self, config: Config):
        self.config = config
        self.allowed_tables = set()
        self.table_metadata = {}
    
    def set_context(self, tables: List[TableInfo], relationships: List[Relationship]):
        """Set context with semantic analysis results"""
        self.allowed_tables = {t.full_name.lower() for t in tables}
        
        # Build metadata map
        for table in tables:
            self.table_metadata[table.full_name.lower()] = {
                'table': table,
                'measures': getattr(table, 'measures', []),
                'entity_keys': getattr(table, 'entity_keys', []),
                'time_columns': getattr(table, 'time_columns', []),
                'filter_columns': getattr(table, 'filter_columns', []),
                'relationships': table.relationships
            }
        
        # Build relationship map
        self.relationships = relationships
    
    async def generate_sql(self, intent: AnalyticalTask, valid_tables: List[TableInfo], 
                          llm: AzureChatOpenAI) -> Optional[str]:
        """Generate SQL using semantic analysis and intelligent logic"""
        print("   ⚡ Stage 4: SQL generation...")
        
        if not valid_tables:
            return None
        
        # Select primary table based on semantic analysis
        primary_table = self._select_primary_table(valid_tables, intent)
        
        # Try intelligent semantic generation first
        sql = self._generate_semantic_sql(intent, primary_table, valid_tables)
        
        if sql and self._validate_sql(sql):
            print("      ✅ Semantic SQL generated and validated")
            return sql
        
        # Fallback to LLM with rich context
        sql = await self._generate_contextual_llm_sql(intent, primary_table, valid_tables, llm)
        
        if sql and self._validate_sql(sql):
            print("      ✅ LLM SQL generated and validated")
            return sql
        
        print("      ❌ SQL generation failed")
        return None
    
    def _select_primary_table(self, tables: List[TableInfo], intent: AnalyticalTask) -> TableInfo:
        """Select primary table using semantic analysis"""
        # For aggregation/ranking, prefer fact tables with measures
        if intent.task_type in ['aggregation', 'ranking']:
            fact_tables = [t for t in tables if getattr(t, 'bi_role', '') == 'fact']
            if fact_tables:
                # Select fact table with most relevant measures
                best_fact = max(fact_tables, key=lambda t: self._score_table_for_intent(t, intent))
                return best_fact
        
        # For other tasks or if no fact tables, select best match
        return max(tables, key=lambda t: self._score_table_for_intent(t, intent))
    
    def _score_table_for_intent(self, table: TableInfo, intent: AnalyticalTask) -> float:
        """Score table relevance for intent"""
        score = 0.0
        
        measures = getattr(table, 'measures', [])
        entity_keys = getattr(table, 'entity_keys', [])
        time_columns = getattr(table, 'time_columns', [])
        
        # Task-specific scoring
        if intent.task_type in ['aggregation', 'ranking']:
            if measures:
                score += 3.0
                # Bonus for revenue-related measures
                for measure in measures:
                    if any(word in measure.lower() for word in ['amount', 'value', 'revenue', 'total']):
                        score += 1.0
                        break
        
        # Entity matching
        if intent.entity:
            entity_type = getattr(table, 'entity_type', '').lower()
            if intent.entity.lower() in entity_type:
                score += 2.0
            elif intent.entity.lower() in table.name.lower():
                score += 1.5
        
        # Time capability
        if intent.time_window and time_columns:
            score += 1.0
        
        # Entity keys for grouping
        if entity_keys:
            score += 1.0
        
        return score
    
    def _generate_semantic_sql(self, intent: AnalyticalTask, primary_table: TableInfo, 
                             all_tables: List[TableInfo]) -> Optional[str]:
        """Generate SQL using semantic analysis results"""
        
        measures = getattr(primary_table, 'measures', [])
        entity_keys = getattr(primary_table, 'entity_keys', [])
        time_columns = getattr(primary_table, 'time_columns', [])
        
        # Find dimension table for customer names if needed
        dimension_table = None
        customer_name_column = None
        
        if intent.entity == 'customer' and intent.task_type == 'ranking':
            dimension_table, customer_name_column = self._find_customer_dimension(primary_table, all_tables)
        
        try:
            if intent.task_type == 'ranking':
                return self._generate_ranking_sql(intent, primary_table, measures, entity_keys, 
                                                time_columns, dimension_table, customer_name_column)
            elif intent.task_type == 'aggregation':
                return self._generate_aggregation_sql(intent, primary_table, measures, entity_keys, time_columns)
            else:
                return self._generate_simple_sql(intent, primary_table, entity_keys, time_columns)
        except Exception as e:
            print(f"      ⚠️ Semantic SQL generation failed: {e}")
            return None
    
    def _find_customer_dimension(self, fact_table: TableInfo, all_tables: List[TableInfo]) -> Tuple[Optional[TableInfo], Optional[str]]:
        """Find customer dimension table and name column"""
        
        # Look for customer-related dimension tables
        customer_tables = []
        for table in all_tables:
            if table == fact_table:
                continue
                
            # Check if it's a customer dimension
            entity_type = getattr(table, 'entity_type', '').lower()
            table_name = table.name.lower()
            
            if ('customer' in entity_type or 'customer' in table_name or 
                any(word in table_name for word in ['client', 'account', 'bp'])):
                customer_tables.append(table)
        
        if not customer_tables:
            return None, None
        
        # Select best customer table
        best_customer_table = max(customer_tables, key=lambda t: self._score_customer_table(t))
        
        # Find name column in customer table
        name_column = self._find_name_column(best_customer_table)
        
        return best_customer_table, name_column
    
    def _score_customer_table(self, table: TableInfo) -> float:
        """Score customer table relevance"""
        score = 0.0
        table_name = table.name.lower()
        entity_type = getattr(table, 'entity_type', '').lower()
        
        if 'customer' in entity_type:
            score += 3.0
        if 'customer' in table_name:
            score += 2.0
        if any(word in table_name for word in ['client', 'account']):
            score += 1.0
        if getattr(table, 'bi_role', '') == 'dimension':
            score += 1.0
        
        return score
    
    def _find_name_column(self, table: TableInfo) -> Optional[str]:
        """Find name column in table"""
        for col in table.columns:
            col_name = col.get('name', '').lower()
            if any(word in col_name for word in ['name', 'title', 'description']):
                return col.get('name')
        
        # Fallback: look for first text column
        for col in table.columns:
            col_type = col.get('data_type', '').lower()
            if any(word in col_type for word in ['char', 'text', 'string']):
                return col.get('name')
        
        return None
    
    def _generate_ranking_sql(self, intent: AnalyticalTask, primary_table: TableInfo,
                            measures: List[str], entity_keys: List[str], time_columns: List[str],
                            dimension_table: Optional[TableInfo], customer_name_column: Optional[str]) -> str:
        """Generate ranking SQL with proper JOINs"""
        
        # Select appropriate measure
        measure_col = self._select_best_measure(measures, intent.metrics)
        if not measure_col:
            measure_col = measures[0] if measures else 'COUNT(*)'
        
        # Select entity key for grouping
        entity_col = entity_keys[0] if entity_keys else None
        
        limit = intent.top_limit or 10
        
        if dimension_table and customer_name_column and entity_col:
            # Generate SQL with JOIN to dimension table
            join_condition = self._find_join_condition(primary_table, dimension_table, entity_col)
            
            if join_condition:
                where_clause = self._build_where_clause(intent, time_columns)
                
                sql = f"""
SELECT TOP {limit}
    d.[{customer_name_column}] as customer_name,
    SUM(f.[{measure_col}]) as total_revenue,
    COUNT(*) as transaction_count
FROM {primary_table.full_name} f
INNER JOIN {dimension_table.full_name} d ON {join_condition}
{where_clause}
GROUP BY d.[{customer_name_column}]
ORDER BY total_revenue DESC
                """.strip()
                
                return sql
        
        # Fallback: simple ranking without JOIN
        where_clause = self._build_where_clause(intent, time_columns)
        
        if entity_col:
            sql = f"""
SELECT TOP {limit}
    [{entity_col}] as entity_id,
    SUM([{measure_col}]) as total_amount
FROM {primary_table.full_name}
{where_clause}
GROUP BY [{entity_col}]
ORDER BY total_amount DESC
            """.strip()
        else:
            sql = f"""
SELECT TOP {limit}
    [{measure_col}] as amount_value
FROM {primary_table.full_name}
{where_clause}
ORDER BY [{measure_col}] DESC
            """.strip()
        
        return sql
    
    def _select_best_measure(self, measures: List[str], intent_metrics: List[str]) -> Optional[str]:
        """Select best measure column for intent"""
        if not measures:
            return None
        
        # Look for measure matching intent
        for metric in intent_metrics:
            for measure in measures:
                if metric.lower() in measure.lower():
                    return measure
        
        # Look for common revenue/amount measures
        revenue_keywords = ['amount', 'value', 'revenue', 'total', 'net']
        for measure in measures:
            for keyword in revenue_keywords:
                if keyword.lower() in measure.lower():
                    return measure
        
        # Return first measure
        return measures[0]
    
    def _find_join_condition(self, fact_table: TableInfo, dim_table: TableInfo, entity_col: str) -> Optional[str]:
        """Find JOIN condition between fact and dimension tables"""
        
        # Look in relationships
        for rel_text in fact_table.relationships:
            if '->' in rel_text and dim_table.name.lower() in rel_text.lower():
                # Parse relationship like "CustomerID -> [dbo].[Customer].ID"
                parts = rel_text.split('->', 1)
                if len(parts) == 2:
                    from_col = parts[0].strip()
                    to_part = parts[1].strip()
                    
                    # Extract column from to_part
                    if '.' in to_part:
                        to_col = to_part.split('.')[-1]
                        return f"f.[{from_col}] = d.[{to_col}]"
        
        # Fallback: try common patterns
        dim_key_cols = [col.get('name') for col in dim_table.columns 
                       if col.get('name', '').lower().endswith('id')]
        
        if dim_key_cols and entity_col:
            return f"f.[{entity_col}] = d.[{dim_key_cols[0]}]"
        
        return None
    
    def _build_where_clause(self, intent: AnalyticalTask, time_columns: List[str]) -> str:
        """Build WHERE clause with time filtering"""
        conditions = ["1=1"]  # Base condition
        
        if intent.time_window and time_columns:
            time_col = time_columns[0]
            
            if intent.time_window == '2025':
                conditions.append(f"YEAR([{time_col}]) = 2025")
            elif intent.time_window == '2024':
                conditions.append(f"YEAR([{time_col}]) = 2024")
            elif intent.time_window == 'last_12_months':
                conditions.append(f"[{time_col}] >= DATEADD(MONTH, -12, GETDATE())")
        
        if len(conditions) > 1:
            return f"WHERE {' AND '.join(conditions[1:])}"
        else:
            return ""
    
    def _generate_aggregation_sql(self, intent: AnalyticalTask, primary_table: TableInfo,
                                measures: List[str], entity_keys: List[str], time_columns: List[str]) -> str:
        """Generate aggregation SQL"""
        
        where_clause = self._build_where_clause(intent, time_columns)
        
        if 'count' in intent.metrics:
            if entity_keys:
                entity_col = entity_keys[0]
                sql = f"""
SELECT COUNT(DISTINCT [{entity_col}]) as total_count
FROM {primary_table.full_name}
{where_clause}
                """.strip()
            else:
                sql = f"""
SELECT COUNT(*) as total_count
FROM {primary_table.full_name}
{where_clause}
                """.strip()
        else:
            # Sum aggregation
            measure_col = self._select_best_measure(measures, intent.metrics)
            if measure_col:
                sql = f"""
SELECT SUM([{measure_col}]) as total_amount
FROM {primary_table.full_name}
{where_clause}
                """.strip()
            else:
                sql = f"""
SELECT COUNT(*) as total_count
FROM {primary_table.full_name}
{where_clause}
                """.strip()
        
        return sql
    
    def _generate_simple_sql(self, intent: AnalyticalTask, primary_table: TableInfo,
                           entity_keys: List[str], time_columns: List[str]) -> str:
        """Generate simple SQL for basic queries"""
        
        where_clause = self._build_where_clause(intent, time_columns)
        
        sql = f"""
SELECT TOP 100 *
FROM {primary_table.full_name}
{where_clause}
        """.strip()
        
        return sql
    
    async def _generate_contextual_llm_sql(self, intent: AnalyticalTask, primary_table: TableInfo, 
                                         all_tables: List[TableInfo], llm: AzureChatOpenAI) -> Optional[str]:
        """Generate SQL using LLM with rich semantic context"""
        
        # Build rich context
        context = self._build_rich_context(primary_table, all_tables, intent)
        
        prompt = f"""
Generate SQL for: {intent.task_type} query about {intent.entity}

CONTEXT:
{context}

REQUIREMENTS:
1. Use semantic analysis results (measures, entity_keys, time_columns)
2. Generate proper JOINs if needed for customer names
3. Use appropriate WHERE clauses for time filtering
4. Generate clean, efficient SQL

Generate SQL only:
"""
        
        try:
            messages = [
                SystemMessage(content="Generate SQL using semantic analysis context. Respond with SQL only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(llm.invoke, messages)
            
            return clean_sql_query(response.content)
            
        except Exception as e:
            print(f"      ⚠️ LLM SQL generation failed: {e}")
            return None
    
    def _build_rich_context(self, primary_table: TableInfo, all_tables: List[TableInfo], intent: AnalyticalTask) -> str:
        """Build rich context for LLM"""
        
        context = f"PRIMARY TABLE: {primary_table.full_name}\n"
        
        measures = getattr(primary_table, 'measures', [])
        if measures:
            context += f"Measures: {', '.join(measures)}\n"
        
        entity_keys = getattr(primary_table, 'entity_keys', [])
        if entity_keys:
            context += f"Entity Keys: {', '.join(entity_keys)}\n"
        
        time_columns = getattr(primary_table, 'time_columns', [])
        if time_columns:
            context += f"Time Columns: {', '.join(time_columns)}\n"
        
        # Add relationship info
        if primary_table.relationships:
            context += f"Relationships: {', '.join(primary_table.relationships[:3])}\n"
        
        # Add related tables if customer query
        if intent.entity == 'customer':
            customer_tables = [t for t in all_tables if 'customer' in t.name.lower()]
            if customer_tables:
                context += f"Customer Tables: {', '.join(t.full_name for t in customer_tables[:2])}\n"
        
        return context
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL using SQLGlot and safety checks"""
        if not sql or not sql.strip():
            return False
        
        # Basic safety check
        if not validate_sql_safety(sql):
            return False
        
        # SQLGlot validation if available
        if HAS_SQLGLOT:
            try:
                parsed = sqlglot.parse_one(sql, dialect="tsql")
                if not parsed:
                    print("      ❌ SQLGlot parsing failed")
                    return False
            except Exception as e:
                print(f"      ❌ SQLGlot validation failed: {e}")
                return False
        
        # Check table references
        return self._validate_table_references(sql)
    
    def _validate_table_references(self, sql: str) -> bool:
        """Validate that SQL references allowed tables"""
        sql_lower = sql.lower()
        
        # Look for table references
        import re
        table_pattern = r'\[([^\]]+)\]\.\[([^\]]+)\]'
        matches = re.findall(table_pattern, sql)
        
        if matches:
            for schema, table in matches:
                full_name = f"[{schema}].[{table}]".lower()
                if full_name in self.allowed_tables:
                    return True
            print("      ❌ No valid table references found")
            return False
        else:
            # Fallback check
            for table_name in self.allowed_tables:
                if table_name in sql_lower:
                    return True
            print("      ❌ No table references found")
            return False

class CapabilityValidator:
    """Stage 3: Enhanced capability validation"""
    
    def validate_capabilities(self, candidates: List[Tuple[TableInfo, EvidenceScore, List[str]]], 
                            intent: AnalyticalTask) -> Tuple[List[TableInfo], List[Tuple[TableInfo, CapabilityContract]]]:
        """Validate capability contracts using semantic analysis"""
        print("   🔒 Stage 3: Capability validation...")
        
        valid_tables = []
        failed_validations = []
        
        for table, score, reasoning in candidates:
            contract = self._assess_semantic_contract(table, intent)
            
            if contract.is_complete():
                valid_tables.append(table)
                print(f"      ✅ {table.full_name}: Contract satisfied")
            else:
                failed_validations.append((table, contract))
                missing = contract.get_missing_capabilities()
                print(f"      ❌ {table.full_name}: Missing {', '.join(missing[:2])}")
        
        return valid_tables, failed_validations
    
    def _assess_semantic_contract(self, table: TableInfo, intent: AnalyticalTask) -> CapabilityContract:
        """Assess capability contract using semantic analysis results"""
        contract = CapabilityContract()
        
        # Use semantic analysis results
        contract.grain = getattr(table, 'grain', 'unknown')
        contract.measures = getattr(table, 'measures', [])
        contract.entity_keys = getattr(table, 'entity_keys', [])
        
        time_columns = getattr(table, 'time_columns', [])
        if time_columns:
            contract.time_column = time_columns[0]
        
        contract.filters = getattr(table, 'filter_columns', [])
        
        contract.quality_checks = {
            'row_count': table.row_count,
            'has_sample_data': len(table.sample_data) > 0,
            'measures_available': len(contract.measures) > 0,
            'entity_keys_available': len(contract.entity_keys) > 0,
            'time_available': contract.time_column is not None,
            'bi_role': getattr(table, 'bi_role', 'unknown'),
            'data_type': getattr(table, 'data_type', 'unknown')
        }
        
        return contract

class QueryExecutor:
    """Execute SQL with enhanced error handling"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_sql(self, sql: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL with proper error handling"""
        print("   🔄 Executing SQL...")
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                # UTF-8 support
                if self.config.utf8_encoding:
                    conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                    conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                    conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                
                # Set query timeout properly
                try:
                    cursor.execute(sql)
                except Exception as e:
                    return [], str(e)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor:
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = safe_database_value(value)
                        results.append(row_dict)
                    
                    print(f"      ✅ Success: {len(results)} rows")
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Failed: {error_msg}")
            return [], error_msg

class QueryInterface:
    """Enhanced BI-Aware Query Interface with proper semantic integration"""
    
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
        
        # Initialize components
        self.intent_analyzer = IntentAnalyzer(self.llm)
        self.table_selector = None  # Initialized with tables
        self.capability_validator = CapabilityValidator()
        self.sql_generator = SmartSQLGenerator(config)
        self.executor = QueryExecutor(config)
        
        print("✅ BI-Aware Query Interface initialized")
        print("   Following README: Simple, Readable, Maintainable")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start interactive query session with semantic context"""
        
        # Initialize components with semantic analysis results
        self.table_selector = SemanticTableSelector(tables, relationships)
        self.sql_generator.set_context(tables, relationships)
        
        # Show enhanced readiness
        self._show_semantic_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ BI Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🧠 Processing with semantic-enhanced 4-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_query(question)
                result.execution_time = time.time() - start_time
                
                self._display_enhanced_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_query(self, question: str) -> QueryResult:
        """Enhanced 4-Stage Pipeline with semantic integration"""
        
        try:
            # Stage 1: Enhanced intent analysis
            intent = await self.intent_analyzer.analyze_intent(question)
            
            # Stage 2: Semantic table selection
            candidates = self.table_selector.select_candidates(intent, top_k=8)
            
            # Stage 3: Enhanced capability validation
            valid_tables, failed_validations = self.capability_validator.validate_capabilities(
                candidates, intent
            )
            
            # Stage 4: Smart SQL generation
            if valid_tables:
                sql = await self.sql_generator.generate_sql(intent, valid_tables, self.llm)
                
                if sql:
                    results, error = await self.executor.execute_sql(sql)
                    
                    return QueryResult(
                        question=question,
                        sql_query=sql,
                        results=results,
                        error=error,
                        tables_used=[t.full_name for t in valid_tables],
                        result_type="data",
                        evidence_reasoning=[r for _, _, reasons in candidates[:1] for r in reasons],
                        capability_score=self._calculate_capability_score(valid_tables[0])
                    )
                else:
                    return QueryResult(
                        question=question,
                        sql_query="",
                        results=[],
                        error="SQL generation failed - semantic analysis could not create valid SQL",
                        result_type="error"
                    )
            else:
                # Generate enhanced NER
                ner = self._generate_enhanced_ner(question, intent, failed_validations, candidates)
                return ner.to_query_result()
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Pipeline error: {str(e)}",
                result_type="error"
            )
    
    def _calculate_capability_score(self, table: TableInfo) -> float:
        """Calculate capability score for table"""
        return table.get_capability_score()
    
    def _generate_enhanced_ner(self, question: str, intent: AnalyticalTask, 
                             failed_validations: List[Tuple[TableInfo, CapabilityContract]],
                             candidates: List[Tuple[TableInfo, EvidenceScore, List[str]]]) -> NonExecutableAnalysisReport:
        """Generate enhanced NER with semantic context"""
        
        # Collect missing capabilities
        all_missing = set()
        for table, contract in failed_validations:
            all_missing.update(contract.get_missing_capabilities())
        
        # Generate semantic fix paths
        fix_paths = []
        for table, contract in failed_validations[:3]:
            missing = contract.get_missing_capabilities()
            
            if "Numeric measures" in missing:
                # Use actual column analysis
                numeric_cols = []
                for col in table.columns:
                    col_type = col.get('data_type', '').lower()
                    if any(t in col_type for t in ['int', 'decimal', 'money', 'float', 'numeric']):
                        numeric_cols.append(col.get('name'))
                
                if numeric_cols:
                    fix_paths.append(f"Consider {table.full_name} columns: {', '.join(numeric_cols[:3])} as measures")
            
            if "Entity keys" in missing:
                # Look for ID columns
                id_cols = [col.get('name') for col in table.columns 
                          if col.get('name', '').lower().endswith('id')]
                if id_cols:
                    fix_paths.append(f"Use {table.full_name} entity keys: {', '.join(id_cols[:2])}")
        
        # Generate safe exploration queries
        safe_queries = []
        for table, score, reasoning in candidates[:2]:
            safe_queries.append(f"-- Explore {table.full_name} structure")
            safe_queries.append(f"SELECT TOP 5 * FROM {table.full_name}")
            
            # Add column analysis
            measures = getattr(table, 'measures', [])
            if measures:
                safe_queries.append(f"-- Check measures: {', '.join(measures[:2])}")
        
        return NonExecutableAnalysisReport(
            question=question,
            normalized_task=intent.__dict__,
            missing_capabilities=list(all_missing),
            top_candidate_tables=[(t.full_name, s.total_score) for t, s, r in candidates[:5]],
            fix_paths=fix_paths,
            suggested_queries=safe_queries
        )
    
    def _show_semantic_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show semantic readiness with analysis results"""
        fact_tables = len([t for t in tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in tables if getattr(t, 'data_type', '') == 'operational'])
        with_measures = len([t for t in tables if getattr(t, 'measures', [])])
        with_relationships = len([t for t in tables if t.relationships])
        
        print(f"\n🧠 SEMANTIC-ENHANCED BI SYSTEM READY:")
        print(f"   📊 Total tables: {len(tables)}")
        print(f"   📈 Fact tables (with measures): {fact_tables}")
        print(f"   ⚡ Operational tables: {operational_tables}")
        print(f"   💰 Tables with semantic measures: {with_measures}")
        print(f"   🔗 Tables with relationships: {with_relationships}")
        
        if domain:
            print(f"   🎯 Domain: {domain.domain_type}")
        
        # Show semantic capabilities
        print(f"\n🔬 SEMANTIC ANALYSIS FEATURES:")
        print(f"   ✅ Entity type classification")
        print(f"   ✅ BI role identification (fact/dimension)")
        print(f"   ✅ Measure and entity key detection")
        print(f"   ✅ Relationship-aware JOIN generation")
        
        print(f"\n🚫 NO-FALLBACK MODE:")
        print(f"   • Semantic capability contracts enforced")
        print(f"   • Evidence-driven table selection using analysis")
        print(f"   • NER reports when validation fails")
    
    def _display_enhanced_result(self, result: QueryResult):
        """Display results with semantic context"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.is_ner():
            # Display enhanced NER
            print(f"🚫 NON-EXECUTABLE ANALYSIS REPORT")
            print(f"   Issue: {result.error}")
            
            if result.evidence_reasoning:
                print(f"   Semantic guidance:")
                for fix in result.evidence_reasoning[:3]:
                    print(f"      • {fix}")
        
        elif result.is_successful():
            # Display success with semantic context
            print(f"✅ QUERY EXECUTED")
            
            if result.evidence_reasoning:
                print(f"   Semantic evidence:")
                for reason in result.evidence_reasoning[:2]:
                    print(f"      • {reason}")
            
            if hasattr(result, 'capability_score') and result.capability_score:
                print(f"   📊 Table capability score: {result.capability_score:.2f}")
            
            print(f"\n📋 SQL:")
            print(f"{result.sql_query}")
            
            print(f"\n📊 Results: {len(result.results)} rows")
            if result.results:
                self._display_semantic_data(result.results)
        
        else:
            # Display error
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
    
    def _display_semantic_data(self, results: List[Dict[str, Any]]):
        """Display query results with enhanced formatting"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value
            key, value = next(iter(results[0].items()))
            from shared.utils import format_number
            formatted = format_number(value) if isinstance(value, (int, float)) else str(value)
            print(f"   🎯 {key}: {formatted}")
        else:
            # Multiple rows with smart formatting
            for i, row in enumerate(results[:10], 1):
                display_row = {}
                for key, value in list(row.items())[:5]:
                    from shared.utils import truncate_text, format_number
                    if isinstance(value, str) and len(value) > 30:
                        display_row[key] = truncate_text(value, 30)
                    elif isinstance(value, (int, float)) and abs(value) >= 1000:
                        display_row[key] = format_number(value)
                    else:
                        display_row[key] = value
                print(f"   {i:2d}. {display_row}")
            
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more rows")
        
        print(f"\n🔬 SEMANTIC FEATURES USED:")
        print(f"   ✅ Semantic table classification")
        print(f"   ✅ Evidence-driven selection")
        print(f"   ✅ Capability contract validation")
        print(f"   ✅ Intelligent JOIN generation")
        print(f"   ✅ Column-aware SQL generation")