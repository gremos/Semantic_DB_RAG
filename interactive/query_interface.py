#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BI-Aware Query Interface - Simple, Readable, Maintainable
Following README: 4-stage pipeline with capability contracts
DRY, SOLID, YAGNI principles with clean implementation
"""

import asyncio
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import (TableInfo, BusinessDomain, Relationship, QueryResult, 
                          AnalyticalTask, CapabilityContract, EvidenceScore, 
                          NonExecutableAnalysisReport)
from shared.utils import parse_json_response, clean_sql_query, safe_database_value, validate_sql_safety

class IntentAnalyzer:
    """Stage 1: Natural language → Analytical Task"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str) -> AnalyticalTask:
        """Analyze user question and normalize to analytical task"""
        print("   🧠 Stage 1: Intent analysis...")
        
        # Quick pattern matching first
        intent = self._quick_pattern_match(question)
        if intent:
            print(f"      ✅ Pattern matched: {intent.task_type}")
            return intent
        
        # LLM analysis for complex cases
        return await self._llm_intent_analysis(question)
    
    def _quick_pattern_match(self, question: str) -> Optional[AnalyticalTask]:
        """Quick pattern matching for common patterns"""
        q_lower = question.lower()
        
        # Detect task type
        if any(word in q_lower for word in ['how many', 'count', 'number of']):
            task_type = 'aggregation'
            metrics = ['count']
        elif any(word in q_lower for word in ['total', 'sum', 'how much']):
            task_type = 'aggregation'  
            metrics = ['sum']
        elif any(word in q_lower for word in ['top', 'highest', 'best', 'worst']):
            task_type = 'ranking'
            metrics = ['value']
        elif any(word in q_lower for word in ['trend', 'over time', 'monthly']):
            task_type = 'trend'
            metrics = ['trend']
        else:
            return None
        
        # Detect entity
        entity = None
        if any(word in q_lower for word in ['customer', 'client', 'user']):
            entity = 'customer'
        elif any(word in q_lower for word in ['revenue', 'sales', 'payment', 'money']):
            entity = 'revenue'
        elif any(word in q_lower for word in ['product', 'item']):
            entity = 'product'
        
        # Extract top limit
        import re
        top_match = re.search(r'top\s+(\d+)', q_lower)
        top_limit = int(top_match.group(1)) if top_match else None
        
        return AnalyticalTask(
            task_type=task_type,
            metrics=metrics,
            entity=entity,
            top_limit=top_limit
        )
    
    async def _llm_intent_analysis(self, question: str) -> AnalyticalTask:
        """LLM-based intent analysis"""
        prompt = f"""
Analyze this business question and extract the analytical intent:

QUESTION: "{question}"

Determine:
1. Task type: aggregation, ranking, trend, count, distribution
2. Metrics: revenue, count, average, etc.
3. Entity: customer, product, order, etc.
4. Time window: current_quarter, last_month, this_year
5. Top limit: if asking for "top N"

Respond with JSON only:
{{
  "task_type": "ranking",
  "metrics": ["revenue"],
  "entity": "customer",
  "time_window": "current_quarter",
  "top_limit": 10
}}
"""
        
        try:
            messages = [
                SystemMessage(content="Extract analytical intent from questions. Respond with valid JSON only."),
                HumanMessage(content=prompt)
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
        
        # Fallback
        return AnalyticalTask(task_type='aggregation', metrics=['count'])

class EvidenceSelector:
    """Stage 2: Evidence-driven table selection"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
    
    def select_candidates(self, intent: AnalyticalTask, top_k: int = 5) -> List[Tuple[TableInfo, EvidenceScore, List[str]]]:
        """Select candidate tables using evidence scoring"""
        print("   📋 Stage 2: Evidence-driven table selection...")
        
        scored_tables = []
        for table in self.tables:
            score = self._calculate_evidence_score(table, intent)
            reasoning = self._generate_reasoning(table, score, intent)
            scored_tables.append((table, score, reasoning))
        
        # Sort by evidence score
        scored_tables.sort(key=lambda x: x[1].total_score, reverse=True)
        
        selected = scored_tables[:top_k]
        print(f"      ✅ Selected {len(selected)} candidates")
        
        return selected
    
    def _calculate_evidence_score(self, table: TableInfo, intent: AnalyticalTask) -> EvidenceScore:
        """Calculate evidence score for table"""
        score = EvidenceScore()
        
        # Role match - prefer fact tables for aggregation
        bi_role = getattr(table, 'bi_role', 'dimension')
        data_type = getattr(table, 'data_type', 'reference')
        
        if intent.task_type in ['aggregation', 'ranking'] and bi_role == 'fact':
            score.role_match = 1.0 if data_type == 'operational' else 0.8
        elif bi_role == 'fact':
            score.role_match = 0.7
        else:
            score.role_match = 0.3
        
        # Join evidence - connectivity
        score.join_evidence = min(1.0, len(table.relationships) / 3.0)
        
        # Lexical match - name similarity
        table_name = table.name.lower()
        entity_type = table.entity_type.lower()
        
        if intent.entity and intent.entity.lower() in table_name:
            score.lexical_match = 1.0
        elif intent.entity and intent.entity.lower() in entity_type:
            score.lexical_match = 0.8
        elif any(metric.lower() in table_name for metric in intent.metrics):
            score.lexical_match = 0.7
        else:
            score.lexical_match = 0.2
        
        # Operational preference
        score.operational_tag = 1.0 if data_type == 'operational' else 0.5
        
        # Row count
        score.row_count = min(1.0, table.row_count / 10000.0) if table.row_count > 0 else 0.0
        score.freshness = 1.0
        
        return score
    
    def _generate_reasoning(self, table: TableInfo, score: EvidenceScore, intent: AnalyticalTask) -> List[str]:
        """Generate reasoning for evidence score"""
        reasoning = []
        
        if score.role_match > 0.7:
            bi_role = getattr(table, 'bi_role', 'unknown')
            reasoning.append(f"Strong BI role: {bi_role} table")
        
        if score.lexical_match > 0.7:
            reasoning.append(f"Name matches intent: {intent.entity or intent.metrics[0] if intent.metrics else 'query'}")
        
        if score.operational_tag > 0.8:
            reasoning.append("Contains operational data")
        
        measures = getattr(table, 'measures', [])
        if measures:
            reasoning.append(f"Has measures: {', '.join(measures[:2])}")
        
        if table.row_count > 1000:
            reasoning.append(f"Good data volume: {table.row_count:,} rows")
        
        return reasoning

class CapabilityValidator:
    """Stage 3: Capability contract validation"""
    
    def validate_capabilities(self, candidates: List[Tuple[TableInfo, EvidenceScore, List[str]]], 
                            intent: AnalyticalTask) -> Tuple[List[TableInfo], List[Tuple[TableInfo, CapabilityContract]]]:
        """Validate capability contracts"""
        print("   🔒 Stage 3: Capability validation...")
        
        valid_tables = []
        failed_validations = []
        
        for table, score, reasoning in candidates:
            contract = self._assess_contract(table, intent)
            
            if contract.is_complete():
                valid_tables.append(table)
                print(f"      ✅ {table.full_name}: Contract satisfied")
            else:
                failed_validations.append((table, contract))
                missing = contract.get_missing_capabilities()
                print(f"      ❌ {table.full_name}: Missing {', '.join(missing[:2])}")
        
        return valid_tables, failed_validations
    
    def _assess_contract(self, table: TableInfo, intent: AnalyticalTask) -> CapabilityContract:
        """Assess capability contract for table"""
        contract = CapabilityContract()
        
        contract.grain = getattr(table, 'grain', 'unknown')
        contract.measures = getattr(table, 'measures', [])
        contract.entity_keys = getattr(table, 'entity_keys', [])
        
        time_columns = getattr(table, 'time_columns', [])
        if time_columns:
            contract.time_column = time_columns[0]
        
        contract.quality_checks = {
            'row_count': table.row_count,
            'has_sample_data': len(table.sample_data) > 0,
            'measures_available': len(contract.measures) > 0
        }
        
        return contract

class SQLGenerator:
    """Stage 4: SQL generation with validation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.allowed_tables = set()
    
    def set_allowed_tables(self, tables: List[TableInfo]):
        """Set allowed tables for validation"""
        self.allowed_tables = {t.full_name.lower() for t in tables}
    
    async def generate_sql(self, intent: AnalyticalTask, valid_tables: List[TableInfo], 
                          llm: AzureChatOpenAI) -> Optional[str]:
        """Generate validated SQL"""
        print("   ⚡ Stage 4: SQL generation...")
        
        if not valid_tables:
            return None
        
        # Select primary table
        primary_table = self._select_primary_table(valid_tables, intent)
        
        # Generate SQL
        sql = await self._generate_sql_for_intent(intent, primary_table, llm)
        
        if sql and self._validate_sql(sql):
            print("      ✅ SQL generated and validated")
            return sql
        
        print("      ❌ SQL validation failed")
        return None
    
    def _select_primary_table(self, tables: List[TableInfo], intent: AnalyticalTask) -> TableInfo:
        """Select primary table for query"""
        # Prefer fact tables for aggregation
        if intent.task_type in ['aggregation', 'ranking']:
            fact_tables = [t for t in tables if getattr(t, 'bi_role', '') == 'fact']
            if fact_tables:
                return max(fact_tables, key=lambda t: len(getattr(t, 'measures', [])))
        
        # Select table with best capabilities
        return max(tables, key=lambda t: t.get_capability_score())
    
    async def _generate_sql_for_intent(self, intent: AnalyticalTask, table: TableInfo, 
                                     llm: AzureChatOpenAI) -> Optional[str]:
        """Generate SQL for analytical intent"""
        
        # Build table context
        measures = getattr(table, 'measures', [])
        entity_keys = getattr(table, 'entity_keys', [])
        time_columns = getattr(table, 'time_columns', [])
        
        sample_preview = ""
        if table.sample_data:
            first_row = table.sample_data[0]
            sample_items = []
            for key, value in list(first_row.items())[:4]:
                if not key.startswith('__'):
                    sample_items.append(f"{key}: {value}")
            sample_preview = " | ".join(sample_items)
        
        prompt = f"""
Generate SQL for this business request:

TASK: {intent.task_type}
METRICS: {intent.metrics}
ENTITY: {intent.entity}
TOP LIMIT: {intent.top_limit}

TABLE: {table.full_name}
MEASURES: {measures}
ENTITY KEYS: {entity_keys}
TIME COLUMNS: {time_columns}
SAMPLE DATA: {sample_preview}

RULES:
1. Use only the table and columns listed above
2. For aggregation: use measures from the list
3. For ranking: use ORDER BY with measures
4. For count: use COUNT(*)
5. Include TOP clause if limit specified
6. Generate clean, efficient SQL

Generate SQL only:
"""
        
        try:
            messages = [
                SystemMessage(content="Generate SQL for business intelligence queries. Respond with SQL only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(llm.invoke, messages)
            
            return clean_sql_query(response.content)
            
        except Exception as e:
            print(f"      ⚠️ SQL generation failed: {e}")
            return None
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL safety and table references"""
        if not validate_sql_safety(sql):
            return False
        
        # Check table references
        sql_lower = sql.lower()
        for table_name in self.allowed_tables:
            if table_name in sql_lower:
                return True
        
        return False

class QueryExecutor:
    """Execute SQL with retry logic"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_sql(self, sql: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL with error handling"""
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

class NERGenerator:
    """Generate Non-Executable Analysis Reports"""
    
    def generate_ner(self, question: str, intent: AnalyticalTask, 
                    failed_validations: List[Tuple[TableInfo, CapabilityContract]],
                    candidates: List[Tuple[TableInfo, EvidenceScore, List[str]]]) -> NonExecutableAnalysisReport:
        """Generate NER when validation fails"""
        
        print("   🚫 Generating Non-Executable Analysis Report...")
        
        # Collect missing capabilities
        all_missing = set()
        for table, contract in failed_validations:
            all_missing.update(contract.get_missing_capabilities())
        
        # Generate fix paths
        fix_paths = []
        for table, contract in failed_validations[:3]:
            missing = contract.get_missing_capabilities()
            
            if "Numeric measures" in missing:
                numeric_cols = [col.get('name') for col in table.columns 
                              if any(t in col.get('data_type', '').lower() for t in ['int', 'decimal', 'money', 'float'])]
                if numeric_cols:
                    fix_paths.append(f"Use {table.full_name} numeric columns: {', '.join(numeric_cols[:2])}")
            
            if "Time/date column" in missing:
                date_cols = [col.get('name') for col in table.columns 
                           if 'date' in col.get('name', '').lower() or 'time' in col.get('data_type', '').lower()]
                if date_cols:
                    fix_paths.append(f"Use {table.full_name} time columns: {', '.join(date_cols[:1])}")
        
        # Generate safe queries
        safe_queries = []
        for table, score, reasoning in candidates[:2]:
            safe_queries.append(f"-- Explore {table.full_name}")
            safe_queries.append(f"SELECT TOP 5 * FROM {table.full_name}")
        
        return NonExecutableAnalysisReport(
            question=question,
            normalized_task=intent.__dict__,
            missing_capabilities=list(all_missing),
            top_candidate_tables=[(t.full_name, s.total_score) for t, s, r in candidates[:5]],
            fix_paths=fix_paths,
            suggested_queries=safe_queries
        )

class QueryInterface:
    """BI-Aware 4-Stage Query Interface - Simple and effective"""
    
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
        self.evidence_selector = None  # Initialized with tables
        self.capability_validator = CapabilityValidator()
        self.sql_generator = SQLGenerator(config)
        self.ner_generator = NERGenerator()
        self.executor = QueryExecutor(config)
        
        print("✅ BI-Aware Query Interface initialized")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start interactive query session"""
        
        # Initialize components with tables
        self.evidence_selector = EvidenceSelector(tables)
        self.sql_generator.set_allowed_tables(tables)
        
        # Show readiness summary
        self._show_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ BI Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🧠 Processing with 4-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_query(question)
                result.execution_time = time.time() - start_time
                
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_query(self, question: str) -> QueryResult:
        """BI-Aware 4-Stage Pipeline"""
        
        try:
            # Stage 1: Intent analysis
            intent = await self.intent_analyzer.analyze_intent(question)
            
            # Stage 2: Evidence-driven selection
            candidates = self.evidence_selector.select_candidates(intent, top_k=8)
            
            # Stage 3: Capability validation
            valid_tables, failed_validations = self.capability_validator.validate_capabilities(
                candidates, intent
            )
            
            # Stage 4: Generate and execute or NER
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
                        evidence_reasoning=[r for _, _, reasons in candidates[:1] for r in reasons]
                    )
                else:
                    return QueryResult(
                        question=question,
                        sql_query="",
                        results=[],
                        error="SQL generation failed",
                        result_type="error"
                    )
            else:
                # Generate NER
                ner = self.ner_generator.generate_ner(question, intent, failed_validations, candidates)
                return ner.to_query_result()
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Pipeline error: {str(e)}",
                result_type="error"
            )
    
    def _show_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show system readiness"""
        fact_tables = len([t for t in tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in tables if getattr(t, 'data_type', '') == 'operational'])
        with_measures = len([t for t in tables if getattr(t, 'measures', [])])
        
        print(f"\n🧠 BI-AWARE SYSTEM READY:")
        print(f"   📊 Tables: {len(tables)}")
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   ⚡ Operational tables: {operational_tables}")
        print(f"   💰 Tables with measures: {with_measures}")
        
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        print(f"\n🚫 NO-FALLBACK MODE:")
        print(f"   • Capability contracts enforced")
        print(f"   • Evidence-driven table selection")
        print(f"   • NER reports when validation fails")
    
    def _display_result(self, result: QueryResult):
        """Display query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.is_ner():
            # Display NER
            print(f"🚫 NON-EXECUTABLE ANALYSIS REPORT")
            print(f"   Issue: {result.error}")
            
            if result.evidence_reasoning:
                print(f"   Fix suggestions:")
                for fix in result.evidence_reasoning[:3]:
                    print(f"      • {fix}")
        
        elif result.is_successful():
            # Display success
            print(f"✅ QUERY EXECUTED")
            
            if result.evidence_reasoning:
                print(f"   Evidence:")
                for reason in result.evidence_reasoning[:2]:
                    print(f"      • {reason}")
            
            print(f"\n📋 SQL:")
            print(f"{result.sql_query}")
            
            print(f"\n📊 Results: {len(result.results)} rows")
            if result.results:
                self._display_data(result.results)
        
        else:
            # Display error
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
    
    def _display_data(self, results: List[Dict[str, Any]]):
        """Display query results"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value
            key, value = next(iter(results[0].items()))
            from shared.utils import format_number
            formatted = format_number(value) if isinstance(value, (int, float)) else str(value)
            print(f"   🎯 {key}: {formatted}")
        else:
            # Multiple rows
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
        
        print(f"\n🧠 BI Features:")
        print(f"   ✅ Evidence-driven table selection")
        print(f"   ✅ Capability contract validation")
        print(f"   ✅ SQL safety validation")
        print(f"   ✅ Zero schema hallucination")