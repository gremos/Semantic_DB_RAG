#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - Enhanced with LLM Entity Resolution
Following README: OpenAI-powered NER with schema context
Simple, Readable, Maintainable
"""

import asyncio
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import (TableInfo, BusinessDomain, Relationship, QueryResult, 
                          AnalyticalTask, CapabilityContract, EvidenceScore)
from shared.utils import parse_json_response, clean_sql_query, safe_database_value, validate_sql_safety

class EnhancedIntentAnalyzer:
    """Enhanced intent analysis with LLM entity resolution"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str, schema_context: Dict[str, Any]) -> AnalyticalTask:
        """Enhanced intent analysis with schema-aware entity resolution"""
        print("   🧠 Stage 1: Enhanced intent analysis with LLM...")
        
        # Build schema context for LLM
        context = self._build_schema_context(schema_context)
        
        # Use LLM for structured intent extraction
        intent_data = await self._llm_intent_analysis(question, context)
        
        if intent_data:
            print(f"      ✅ LLM Analysis: {intent_data.get('task_type', 'unknown')}")
            return AnalyticalTask(
                task_type=intent_data.get('task_type', 'aggregation'),
                metrics=intent_data.get('metrics', []),
                entity=intent_data.get('entity'),
                crm_entities=intent_data.get('crm_entities', []),
                time_window=intent_data.get('time_window'),
                grouping=intent_data.get('group_by', []),
                filters=intent_data.get('filters', []),
                top_limit=intent_data.get('limit')
            )
        
        # Fallback to pattern matching
        return self._pattern_fallback(question)
    
    def _build_schema_context(self, schema_context: Dict[str, Any]) -> str:
        """Build schema context for LLM"""
        context_parts = []
        
        # Add table summaries
        if 'tables' in schema_context:
            context_parts.append("AVAILABLE TABLES:")
            for table in schema_context['tables'][:20]:  # Limit to top 20 tables
                table_name = table.get('full_name', table.get('name', 'Unknown'))
                entity_type = table.get('entity_type', 'Unknown')
                columns = [col.get('name', '') for col in table.get('columns', [])[:8]]
                sample = table.get('sample_preview', '')
                
                context_parts.append(f"- {table_name}: {entity_type}")
                context_parts.append(f"  Columns: {', '.join(columns)}")
                if sample:
                    context_parts.append(f"  Sample: {sample}")
        
        return '\n'.join(context_parts)
    
    async def _llm_intent_analysis(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """LLM-based intent analysis with schema context"""
        try:
            system_prompt = """
You are a BI analyst. Extract analytical intent from natural language and map to database schema.

Respond with JSON only using this structure:
{
  "task_type": "ranking|aggregation|trend|count",
  "metrics": ["revenue", "amount", "count", "quantity"],
  "entity": "customer|payment|order|product",
  "crm_entities": ["customer", "payment"],
  "group_by": ["customer_name", "region"],
  "filters": ["paid", "active", "won"],
  "time_window": "this_year|last_month|2025",
  "limit": 10
}

ENTITY MAPPING RULES:
- "customers" → customer entity, look for Customer/Client tables
- "paid/revenue/sales" → payment/transaction tables with amount columns  
- "top X" → ranking task with limit X
- "names" → include name/title columns in grouping
"""

            user_prompt = f"""
QUESTION: "{question}"

SCHEMA CONTEXT:
{context}

Map the question to available tables and extract intent as JSON:
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return parse_json_response(response.content)
            
        except Exception as e:
            print(f"      ⚠️ LLM intent analysis failed: {e}")
            return None
    
    def _pattern_fallback(self, question: str) -> AnalyticalTask:
        """Enhanced pattern matching fallback"""
        q_lower = question.lower()
        
        # Detect task type
        if any(word in q_lower for word in ['top', 'highest', 'best', 'worst', 'most']):
            task_type = 'ranking'
        elif any(word in q_lower for word in ['count', 'how many', 'number of']):
            task_type = 'count'
        elif any(word in q_lower for word in ['total', 'sum']):
            task_type = 'aggregation'
        else:
            task_type = 'aggregation'
        
        # Detect metrics
        metrics = []
        if any(word in q_lower for word in ['paid', 'payment', 'revenue', 'sales', 'amount']):
            metrics = ['amount', 'revenue']
        elif 'count' in q_lower or 'number' in q_lower:
            metrics = ['count']
        else:
            metrics = ['amount']
        
        # Detect entities
        crm_entities = []
        if any(word in q_lower for word in ['customer', 'client', 'account']):
            crm_entities.append('customer')
        if any(word in q_lower for word in ['payment', 'paid', 'revenue', 'sales']):
            crm_entities.append('payment')
        
        # Extract limit
        import re
        top_match = re.search(r'top\s+(\d+)', q_lower)
        limit = int(top_match.group(1)) if top_match else (10 if task_type == 'ranking' else None)
        
        # Detect grouping
        grouping = []
        if 'name' in q_lower:
            grouping = ['name']
        
        return AnalyticalTask(
            task_type=task_type,
            metrics=metrics,
            crm_entities=crm_entities,
            grouping=grouping,
            top_limit=limit
        )

class EnhancedTableSelector:
    """Enhanced table selection with entity-based scoring"""
    
    def __init__(self, tables: List[TableInfo], relationships: List[Relationship]):
        self.tables = tables
        self.relationships = relationships
    
    def select_candidates(self, intent: AnalyticalTask, top_k: int = 5) -> List[Tuple[TableInfo, EvidenceScore]]:
        """Enhanced evidence-based table selection"""
        print("   📋 Stage 2: Enhanced entity-aware table selection...")
        
        scored_tables = []
        for table in self.tables:
            score = self._calculate_enhanced_evidence(table, intent)
            if score.total_score > 0.1:  # Only consider tables with some relevance
                scored_tables.append((table, score))
        
        # Sort by evidence score
        scored_tables.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Show reasoning for top candidates
        for i, (table, score) in enumerate(scored_tables[:3]):
            explanations = score.get_explanation()
            print(f"      #{i+1} {table.name} (score: {score.total_score:.2f}): {explanations[0] if explanations else 'No strong evidence'}")
        
        selected = scored_tables[:top_k]
        print(f"      ✅ Selected {len(selected)} candidates")
        
        return selected
    
    def _calculate_enhanced_evidence(self, table: TableInfo, intent: AnalyticalTask) -> EvidenceScore:
        """Enhanced evidence calculation with entity matching"""
        score = EvidenceScore()
        
        # Entity matching - primary scoring factor
        score.lexical_match = self._calculate_entity_match(table, intent)
        
        # BI role and data type scoring
        score.role_match = self._calculate_bi_role_match(table, intent)
        
        # Table quality (avoid temp/test/system tables)
        score.graph_proximity = self._calculate_table_quality(table)
        
        # Operational preference
        score.operational_tag = 1.0 if getattr(table, 'data_type', '') == 'operational' else 0.3
        
        # Join connectivity
        score.join_evidence = min(1.0, len(table.relationships) / 2.0)
        
        # Data availability
        score.row_count = min(1.0, table.row_count / 1000.0) if table.row_count > 0 else 0.0
        score.freshness = 1.0  # Assume fresh for now
        
        return score
    
    def _calculate_entity_match(self, table: TableInfo, intent: AnalyticalTask) -> float:
        """Calculate entity matching score"""
        if not intent.crm_entities:
            return 0.5
        
        table_name = table.name.lower()
        entity_type = getattr(table, 'entity_type', '').lower()
        columns = [col.get('name', '').lower() for col in table.columns]
        
        max_match = 0.0
        
        for entity in intent.crm_entities:
            entity_lower = entity.lower()
            
            # Direct table name match
            if entity_lower in table_name:
                max_match = max(max_match, 1.0)
            
            # Entity type match
            elif entity_lower in entity_type:
                max_match = max(max_match, 0.9)
            
            # Column name matches
            elif any(entity_lower in col for col in columns):
                max_match = max(max_match, 0.7)
            
            # Semantic matches
            elif self._semantic_entity_match(entity_lower, table_name, columns):
                max_match = max(max_match, 0.6)
        
        return max_match
    
    def _semantic_entity_match(self, entity: str, table_name: str, columns: List[str]) -> bool:
        """Semantic entity matching"""
        semantic_mappings = {
            'customer': ['client', 'account', 'buyer', 'consumer'],
            'payment': ['transaction', 'invoice', 'billing', 'revenue', 'sales'],
            'order': ['purchase', 'transaction', 'sale'],
            'product': ['item', 'goods', 'service']
        }
        
        if entity in semantic_mappings:
            synonyms = semantic_mappings[entity]
            return (any(syn in table_name for syn in synonyms) or 
                   any(any(syn in col for syn in synonyms) for col in columns))
        
        return False
    
    def _calculate_bi_role_match(self, table: TableInfo, intent: AnalyticalTask) -> float:
        """Calculate BI role match score"""
        bi_role = getattr(table, 'bi_role', 'dimension')
        
        if intent.task_type in ['ranking', 'aggregation'] and bi_role == 'fact':
            return 1.0
        elif bi_role == 'fact':
            return 0.8
        elif intent.task_type == 'count' and bi_role == 'dimension':
            return 0.7
        else:
            return 0.4
    
    def _calculate_table_quality(self, table: TableInfo) -> float:
        """Calculate table quality score"""
        table_name = table.name.lower()
        
        # High quality indicators
        quality = 1.0
        
        # Penalize temp/test tables
        low_quality_indicators = [
            'temp', 'test', 'backup', 'old', 'archive', 'delete', 
            'staging', 'work', 'scratch', 'debug'
        ]
        
        for indicator in low_quality_indicators:
            if indicator in table_name:
                quality -= 0.5
                break
        
        # Penalize dated tables (with timestamps)
        import re
        if re.search(r'\d{8}|\d{4}_\d{2}_\d{2}', table_name):
            quality -= 0.3
        elif re.search(r'\d{4}', table_name):
            quality -= 0.1
        
        # Boost main business tables
        business_indicators = ['customer', 'payment', 'order', 'product', 'invoice', 'transaction']
        if any(indicator in table_name for indicator in business_indicators):
            quality += 0.2
        
        return max(0.0, min(1.0, quality))

class EnhancedSQLGenerator:
    """Enhanced SQL generation with intent-driven approach"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def generate_sql(self, intent: AnalyticalTask, valid_tables: List[TableInfo], 
                          llm: AzureChatOpenAI) -> Optional[str]:
        """Generate SQL with enhanced intent understanding"""
        print("   ⚡ Stage 4: Enhanced SQL generation...")
        
        if not valid_tables:
            return None
        
        # Select best table combination
        primary_table, supporting_tables = self._select_table_combination(valid_tables, intent)
        
        # Generate SQL with LLM using enhanced context
        sql = await self._generate_intent_driven_sql(intent, primary_table, supporting_tables, llm)
        
        if sql and self._validate_sql(sql):
            print("      ✅ Enhanced SQL generated and validated")
            return sql
        
        print("      ❌ Enhanced SQL generation failed")
        return None
    
    def _select_table_combination(self, tables: List[TableInfo], intent: AnalyticalTask) -> Tuple[TableInfo, List[TableInfo]]:
        """Select optimal table combination for the query"""
        # For customer queries, prefer customer-related tables
        if 'customer' in intent.crm_entities:
            customer_tables = [t for t in tables if 'customer' in t.name.lower() or 
                             'customer' in getattr(t, 'entity_type', '').lower()]
            if customer_tables:
                primary = customer_tables[0]
                supporting = [t for t in tables if 'payment' in t.name.lower() or 
                            'transaction' in t.name.lower() or 'invoice' in t.name.lower()]
                return primary, supporting
        
        # For payment queries, prefer fact tables with measures
        if 'payment' in intent.crm_entities:
            payment_tables = [t for t in tables if getattr(t, 'measures', []) and 
                            ('payment' in t.name.lower() or 'transaction' in t.name.lower())]
            if payment_tables:
                primary = payment_tables[0]
                supporting = [t for t in tables if 'customer' in t.name.lower()]
                return primary, supporting
        
        # Default: use first valid table
        return tables[0], tables[1:3] if len(tables) > 1 else []
    
    async def _generate_intent_driven_sql(self, intent: AnalyticalTask, primary_table: TableInfo, 
                                        supporting_tables: List[TableInfo], llm: AzureChatOpenAI) -> Optional[str]:
        """Generate SQL based on intent understanding"""
        
        # Build enhanced context
        context = self._build_enhanced_context(intent, primary_table, supporting_tables)
        
        # Create intent-specific prompt
        prompt = self._create_intent_prompt(intent, context)
        
        try:
            messages = [
                SystemMessage(content="""
You are a SQL expert. Generate simple, correct SQL for business intelligence queries.

RULES:
1. Use only columns that exist in the provided schema
2. No variables or parameters - use literal values
3. For "top N" queries: SELECT TOP (N) ...
4. For customer names: look for Name, CustomerName, Title columns  
5. For amounts: look for Amount, Total, Price, Revenue columns
6. Use simple JOINs only when necessary
7. GROUP BY when aggregating
8. ORDER BY for rankings (DESC for highest values)

Generate clean SQL only - no explanations.
"""),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(llm.invoke, messages)
            sql = clean_sql_query(response.content)
            
            # Post-process to ensure simplicity
            sql = self._simplify_sql(sql, intent)
            
            return sql
            
        except Exception as e:
            print(f"      ⚠️ SQL generation failed: {e}")
            return None
    
    def _build_enhanced_context(self, intent: AnalyticalTask, primary_table: TableInfo, 
                              supporting_tables: List[TableInfo]) -> str:
        """Build enhanced context for SQL generation"""
        context_parts = []
        
        # Primary table context
        context_parts.append(f"PRIMARY TABLE: {primary_table.full_name}")
        context_parts.append("COLUMNS:")
        
        # Show relevant columns based on intent
        for col in primary_table.columns[:15]:
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            
            # Highlight relevant columns
            relevance = ""
            if any(word in col_name.lower() for word in ['name', 'title', 'customer']):
                relevance = " [CUSTOMER NAME]"
            elif any(word in col_name.lower() for word in ['amount', 'total', 'price', 'revenue']):
                relevance = " [AMOUNT]"
            elif any(word in col_name.lower() for word in ['date', 'time']):
                relevance = " [DATE/TIME]"
            
            context_parts.append(f"  - {col_name} ({col_type}){relevance}")
        
        # Sample data for understanding
        if primary_table.sample_data:
            context_parts.append("\nSAMPLE DATA (first row):")
            first_row = primary_table.sample_data[0]
            for k, v in list(first_row.items())[:6]:
                if not k.startswith('__'):
                    context_parts.append(f"  {k}: {v}")
        
        # Supporting tables if any
        if supporting_tables:
            context_parts.append(f"\nSUPPORTING TABLES:")
            for table in supporting_tables[:2]:
                context_parts.append(f"- {table.full_name}")
                key_cols = [col.get('name', '') for col in table.columns[:5]]
                context_parts.append(f"  Key columns: {', '.join(key_cols)}")
        
        return '\n'.join(context_parts)
    
    def _create_intent_prompt(self, intent: AnalyticalTask, context: str) -> str:
        """Create intent-specific prompt"""
        task_templates = {
            'ranking': f"Generate SQL for: Get top {intent.top_limit or 10} records ranked by {intent.metrics[0] if intent.metrics else 'value'}",
            'aggregation': f"Generate SQL for: Calculate {intent.metrics[0] if intent.metrics else 'total'} aggregated by groups",
            'count': f"Generate SQL for: Count records matching criteria"
        }
        
        task_description = task_templates.get(intent.task_type, f"Generate SQL for: {intent.task_type} query")
        
        return f"""
{task_description}

REQUIREMENTS:
- Task: {intent.task_type}
- Entities: {', '.join(intent.crm_entities) if intent.crm_entities else 'general'}
- Metrics: {', '.join(intent.metrics) if intent.metrics else 'count'}
- Grouping: {', '.join(intent.grouping) if intent.grouping else 'none'}
- Limit: {intent.top_limit or 'none'}

SCHEMA CONTEXT:
{context}

Generate simple, working SQL:
"""
    
    def _simplify_sql(self, sql: str, intent: AnalyticalTask) -> str:
        """Simplify SQL to remove unnecessary complexity"""
        if not sql:
            return sql
        
        import re
        
        # Remove unnecessary ROW_NUMBER() constructs for simple ranking
        if intent.task_type == 'ranking' and 'ROW_NUMBER()' in sql and intent.top_limit:
            # Try to simplify to basic ORDER BY with TOP
            sql = re.sub(r'SELECT .* FROM \([^)]+ROW_NUMBER\(\)[^)]+\) AS \w+ WHERE \w+ <= \d+', 
                        f"SELECT TOP ({intent.top_limit})", sql)
        
        # Clean up excessive whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def _validate_sql(self, sql: str) -> bool:
        """Enhanced SQL validation"""
        return validate_sql_safety(sql)

class QueryInterface:
    """Enhanced query interface with LLM entity resolution"""
    
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
        
        # Initialize enhanced components
        self.intent_analyzer = EnhancedIntentAnalyzer(self.llm)
        self.capability_validator = CapabilityValidator()
        self.sql_generator = EnhancedSQLGenerator(config)
        self.executor = QueryExecutor(config)
        
        print("✅ Enhanced Query Interface initialized with LLM entity resolution")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start enhanced interactive query session"""
        
        # Initialize enhanced table selector
        self.table_selector = EnhancedTableSelector(tables, relationships)
        
        # Build schema context for LLM
        self.schema_context = self._build_schema_context(tables)
        
        # Show enhanced readiness
        self._show_enhanced_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🧠 Processing with enhanced 4-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_enhanced_query(question)
                result.execution_time = time.time() - start_time
                
                self._display_enhanced_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _build_schema_context(self, tables: List[TableInfo]) -> Dict[str, Any]:
        """Build schema context for LLM entity resolution"""
        schema_tables = []
        
        for table in tables:
            # Get sample preview for context
            sample_preview = ""
            if table.sample_data:
                first_row = table.sample_data[0]
                preview_parts = []
                for k, v in list(first_row.items())[:3]:
                    if not k.startswith('__'):
                        preview_parts.append(f"{k}={v}")
                sample_preview = " | ".join(preview_parts)
            
            schema_tables.append({
                'name': table.name,
                'full_name': table.full_name,
                'entity_type': table.entity_type,
                'columns': table.columns,
                'bi_role': getattr(table, 'bi_role', 'dimension'),
                'measures': getattr(table, 'measures', []),
                'sample_preview': sample_preview
            })
        
        return {'tables': schema_tables}
    
    async def process_enhanced_query(self, question: str) -> QueryResult:
        """Enhanced 4-Stage Pipeline with LLM entity resolution"""
        
        try:
            # Stage 1: Enhanced intent analysis with schema context
            intent = await self.intent_analyzer.analyze_intent(question, self.schema_context)
            
            # Stage 2: Enhanced entity-aware table selection
            candidates = self.table_selector.select_candidates(intent, top_k=5)
            
            # Stage 3: Capability validation
            valid_tables = self.capability_validator.validate_capabilities(candidates, intent)
            
            # Stage 4: Enhanced SQL generation and execution
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
                        result_type="data"
                    )
                else:
                    return QueryResult(
                        question=question,
                        sql_query="",
                        results=[],
                        error="Enhanced SQL generation failed",
                        result_type="error"
                    )
            else:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No tables passed capability validation",
                    result_type="ner"
                )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Enhanced pipeline error: {str(e)}",
                result_type="error"
            )
    
    def _show_enhanced_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show enhanced system readiness"""
        fact_tables = len([t for t in tables if getattr(t, 'bi_role', '') == 'fact'])
        customer_tables = len([t for t in tables if 'customer' in t.name.lower()])
        payment_tables = len([t for t in tables if any(word in t.name.lower() 
                              for word in ['payment', 'transaction', 'invoice'])])
        
        print(f"\n🧠 ENHANCED BI SYSTEM READY:")
        print(f"   📊 Total tables: {len(tables)}")
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   👥 Customer tables: {customer_tables}")
        print(f"   💳 Payment tables: {payment_tables}")
        
        if domain:
            print(f"   🎯 Domain: {domain.domain_type}")
        
        print(f"\n🧠 ENHANCED FEATURES:")
        print(f"   • LLM-powered entity resolution")
        print(f"   • Schema-aware intent analysis") 
        print(f"   • Enhanced table selection")
        print(f"   • Intent-driven SQL generation")
    
    def _display_enhanced_result(self, result: QueryResult):
        """Display enhanced query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.is_successful():
            print(f"✅ ENHANCED QUERY EXECUTED")
            print(f"\n📋 SQL:")
            print(f"{result.sql_query}")
            
            print(f"\n📊 Results: {len(result.results)} rows")
            if result.results:
                self._display_enhanced_data(result.results)
        
        else:
            print(f"❌ ENHANCED QUERY FAILED")
            print(f"   Error: {result.error}")
            print(f"   💡 Try more specific entity names or check table availability")
    
    def _display_enhanced_data(self, results: List[Dict[str, Any]]):
        """Display enhanced query results with better formatting"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value
            key, value = next(iter(results[0].items()))
            from shared.utils import format_number
            formatted = format_number(value) if isinstance(value, (int, float)) else str(value)
            print(f"   🎯 {key}: {formatted}")
        else:
            # Multiple rows - show in a more business-friendly format
            for i, row in enumerate(results[:10], 1):
                display_parts = []
                for key, value in list(row.items())[:4]:  # Show first 4 columns
                    if isinstance(value, str) and len(value) > 25:
                        value = value[:25] + "..."
                    elif isinstance(value, (int, float)) and abs(value) >= 1000:
                        from shared.utils import format_number
                        value = format_number(value)
                    display_parts.append(f"{key}: {value}")
                
                print(f"   {i:2d}. {' | '.join(display_parts)}")
            
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more rows")


# Keep existing CapabilityValidator and QueryExecutor classes unchanged
class CapabilityValidator:
    """Stage 3: Capability validation"""
    
    def validate_capabilities(self, candidates: List[Tuple[TableInfo, EvidenceScore]], 
                            intent: AnalyticalTask) -> List[TableInfo]:
        """Validate capability contracts"""
        print("   🔒 Stage 3: Capability validation...")
        
        valid_tables = []
        
        for table, score in candidates:
            contract = self._assess_contract(table, intent)
            
            if contract.is_complete():
                valid_tables.append(table)
                print(f"      ✅ {table.name}: Contract satisfied")
            else:
                missing = contract.get_missing_capabilities()
                print(f"      ❌ {table.name}: Missing {', '.join(missing[:2])}")
        
        return valid_tables
    
    def _assess_contract(self, table: TableInfo, intent: AnalyticalTask) -> CapabilityContract:
        """Assess capability contract"""
        contract = CapabilityContract()
        
        contract.grain = getattr(table, 'grain', 'unknown')
        contract.measures = getattr(table, 'measures', [])
        contract.entity_keys = getattr(table, 'entity_keys', [])
        
        time_columns = getattr(table, 'time_columns', [])
        if time_columns:
            contract.time_column = time_columns[0]
        
        contract.quality_checks = {
            'row_count': table.row_count,
            'has_measures': len(contract.measures) > 0,
            'has_entity_keys': len(contract.entity_keys) > 0,
            'has_time': contract.time_column is not None
        }
        
        return contract


class QueryExecutor:
    """Execute SQL with error handling"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_sql(self, sql: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL and return results"""
        print("   🔄 Executing SQL...")
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                # UTF-8 support
                if self.config.utf8_encoding:
                    conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                    conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                    conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                cursor.execute(sql)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = [
                        {columns[i]: safe_database_value(row[i]) for i in range(len(columns))}
                        for row in cursor
                    ]
                    
                    print(f"      ✅ Success: {len(results)} rows")
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Failed: {error_msg}")
            return [], error_msg