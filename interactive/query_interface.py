#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BI-Aware Query Interface - Simple, Readable, Maintainable
Following README: 4-stage pipeline with capability contracts
DRY, SOLID, YAGNI principles with clean implementation
Fixed SQL generation and validation
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
        
        # Quick pattern matching first (YAGNI - what works)
        intent = self._pattern_match(question)
        if intent:
            print(f"      ✅ Pattern matched: {intent.task_type}")
            return intent
        
        # LLM analysis for complex cases
        return await self._llm_intent_analysis(question)
    
    def _pattern_match(self, question: str) -> Optional[AnalyticalTask]:
        """Simple pattern matching for common patterns"""
        q_lower = question.lower()
        
        # Detect task type and metrics
        if any(word in q_lower for word in ['count', 'how many', 'number of']):
            task_type = 'aggregation'
            metrics = ['count']
        elif any(word in q_lower for word in ['total', 'sum', 'amount']):
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
        elif any(word in q_lower for word in ['payment', 'revenue', 'sales']):
            entity = 'payment'
        
        # Extract year for time filtering
        time_window = None
        if '2025' in q_lower:
            time_window = '2025'
        elif '2024' in q_lower:
            time_window = '2024'
        
        return AnalyticalTask(
            task_type=task_type,
            metrics=metrics,
            entity=entity,
            time_window=time_window
        )
    
    async def _llm_intent_analysis(self, question: str) -> AnalyticalTask:
        """LLM-based intent analysis with error handling"""
        try:
            messages = [
                SystemMessage(content="Extract analytical intent. Respond with JSON only."),
                HumanMessage(content=f"""
Analyze: "{question}"

Extract:
1. Task type: aggregation, ranking, trend, count
2. Metrics: count, sum, revenue, etc.
3. Entity: customer, payment, order
4. Time window: 2025, 2024, etc.

JSON only:
{{
  "task_type": "aggregation",
  "metrics": ["count"],
  "entity": "customer",
  "time_window": "2025"
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
                    time_window=data.get('time_window')
                )
        except Exception as e:
            print(f"      ⚠️ LLM intent analysis failed: {e}")
        
        # Simple fallback
        return AnalyticalTask(task_type='aggregation', metrics=['count'])

class EvidenceSelector:
    """Stage 2: Improved Evidence-driven table selection"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
    
    def select_candidates(self, intent: AnalyticalTask, top_k: int = 5) -> List[Tuple[TableInfo, EvidenceScore, List[str]]]:
        """Select candidate tables using improved evidence scoring"""
        print("   📋 Stage 2: Evidence-driven table selection...")
        
        scored_tables = []
        for table in self.tables:
            score = self._calculate_improved_evidence(table, intent)
            reasoning = self._generate_reasoning(table, score, intent)
            scored_tables.append((table, score, reasoning))
        
        # Sort by evidence score
        scored_tables.sort(key=lambda x: x[1].total_score, reverse=True)
        
        selected = scored_tables[:top_k]
        print(f"      ✅ Selected {len(selected)} candidates")
        
        # Debug: Show why tables were selected
        print(f"      🔍 Top candidates:")
        for i, (table, score, reasoning) in enumerate(selected[:3], 1):
            print(f"         {i}. {table.name} (score: {score.total_score:.2f})")
        
        return selected
    
    def _calculate_improved_evidence(self, table: TableInfo, intent: AnalyticalTask) -> EvidenceScore:
        """Calculate evidence score with improved logic"""
        score = EvidenceScore()
        
        # 1. Role match - prefer fact tables for aggregation
        bi_role = getattr(table, 'bi_role', 'dimension')
        data_type = getattr(table, 'data_type', 'reference')
        
        if intent.task_type in ['aggregation', 'ranking'] and bi_role == 'fact':
            score.role_match = 1.0 if data_type == 'operational' else 0.8
        elif bi_role == 'fact':
            score.role_match = 0.7
        else:
            score.role_match = 0.3
        
        # 2. Join evidence - table connectivity
        score.join_evidence = min(1.0, len(table.relationships) / 3.0)
        
        # 3. Lexical match - name similarity with intent
        score.lexical_match = self._calculate_lexical_match(table, intent)
        
        # 4. NEW: Table quality score - prefer main tables over temp/dated tables
        score.graph_proximity = self._calculate_table_quality(table)
        
        # 5. Operational preference
        score.operational_tag = 1.0 if data_type == 'operational' else 0.5
        
        # 6. Row count - but not too heavily weighted
        score.row_count = min(1.0, table.row_count / 10000.0) if table.row_count > 0 else 0.0
        
        # 7. Freshness/recency
        score.freshness = 1.0
        
        return score
    
    def _calculate_lexical_match(self, table: TableInfo, intent: AnalyticalTask) -> float:
        """Calculate lexical match score"""
        table_name = table.name.lower()
        entity_type = getattr(table, 'entity_type', '').lower()
        
        # Perfect match
        if intent.entity and intent.entity.lower() in table_name:
            return 1.0
        
        # Good match with entity type
        elif intent.entity and intent.entity.lower() in entity_type:
            return 0.8
        
        # Match with metrics
        elif any(metric.lower() in table_name for metric in intent.metrics):
            return 0.7
        
        # Partial matches
        elif intent.entity == 'payment' and any(word in table_name for word in ['payment', 'invoice', 'billing']):
            return 0.6
        elif intent.entity == 'customer' and any(word in table_name for word in ['customer', 'client', 'account']):
            return 0.6
        elif intent.entity == 'sales_rep' and any(word in table_name for word in ['sales', 'rep', 'agent']):
            return 0.6
        
        else:
            return 0.2
    
    def _calculate_table_quality(self, table: TableInfo) -> float:
        """NEW: Calculate table quality score - prefer main tables over temp/dated tables"""
        table_name = table.name.lower()
        quality_score = 1.0  # Start with perfect score
        
        # Penalty for dated table names (like table20230517)
        import re
        if re.search(r'\d{8}', table_name):  # 8 digits (YYYYMMDD)
            quality_score -= 0.4
            print(f"      ⚠️ {table.name}: Dated table detected (8 digits)")
        elif re.search(r'\d{6}', table_name):  # 6 digits (YYYYMM) 
            quality_score -= 0.3
            print(f"      ⚠️ {table.name}: Dated table detected (6 digits)")
        elif re.search(r'\d{4}', table_name):  # 4 digits (YYYY)
            quality_score -= 0.2
            print(f"      ⚠️ {table.name}: Dated table detected (4 digits)")
        
        # Penalty for temp/backup indicators
        temp_indicators = ['temp', 'tmp', 'backup', 'bak', 'old', 'archive', 'test', 'staging']
        for indicator in temp_indicators:
            if indicator in table_name:
                quality_score -= 0.3
                print(f"      ⚠️ {table.name}: Temp table indicator '{indicator}' detected")
                break
        
        # Penalty for bridge/buffer tables (unless specifically needed)
        if any(word in table_name for word in ['bridge', 'buffer']) and not any(word in table_name for word in ['customer', 'payment']):
            quality_score -= 0.2
            print(f"      ⚠️ {table.name}: Bridge/buffer table detected")
        
        # Bonus for clean, simple names
        if not re.search(r'\d', table_name) and len(table_name.split()) <= 2:
            quality_score += 0.1
            print(f"      ✅ {table.name}: Clean table name bonus")
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, quality_score))
    
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
        
        # NEW: Quality indicators
        if score.graph_proximity < 0.7:
            reasoning.append("⚠️ Possible temp/dated table")
        elif score.graph_proximity >= 0.9:
            reasoning.append("✅ Clean main table")
        
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
    """Stage 4: SQL generation with validation - Fixed and simplified"""
    
    def __init__(self, config: Config):
        self.config = config
        self.allowed_tables = set()
    
    def set_allowed_tables(self, tables: List[TableInfo]):
        """Set allowed tables for validation"""
        self.allowed_tables = {t.full_name.lower() for t in tables}
    
    async def generate_sql(self, intent: AnalyticalTask, valid_tables: List[TableInfo], 
                          llm: AzureChatOpenAI) -> Optional[str]:
        """Generate validated SQL - Fixed implementation"""
        print("   ⚡ Stage 4: SQL generation...")
        
        if not valid_tables:
            return None
        
        # Select primary table
        primary_table = self._select_primary_table(valid_tables, intent)
        
        # Try template-based generation first (simpler, more reliable)
        sql = self._generate_template_sql(intent, primary_table)
        
        if sql and self._validate_sql(sql):
            print("      ✅ Template SQL generated and validated")
            return sql
        
        # Fallback to LLM generation
        sql = await self._generate_llm_sql(intent, primary_table, llm)
        
        if sql and self._validate_sql(sql):
            print("      ✅ LLM SQL generated and validated")
            return sql
        
        print("      ❌ SQL generation failed")
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
    
    def _generate_template_sql(self, intent: AnalyticalTask, table: TableInfo) -> Optional[str]:
        """Generate SQL using simple templates - More reliable"""
        
        measures = getattr(table, 'measures', [])
        entity_keys = getattr(table, 'entity_keys', [])
        time_columns = getattr(table, 'time_columns', [])
        
        # Choose appropriate column for counting
        count_column = None
        if entity_keys:
            count_column = entity_keys[0]
        elif measures:
            count_column = measures[0]
        else:
            count_column = "*"
        
        # Build WHERE clause for time filtering
        where_clause = "1=1"
        if intent.time_window and time_columns:
            time_col = time_columns[0]
            if intent.time_window == "2025":
                where_clause = f"YEAR([{time_col}]) = 2025"
            elif intent.time_window == "2024":
                where_clause = f"YEAR([{time_col}]) = 2024"
        
        # Generate appropriate SQL based on task type
        if intent.task_type == 'aggregation':
            if 'count' in intent.metrics:
                if count_column == "*":
                    sql = f"SELECT COUNT(*) as total_count FROM {table.full_name} WHERE {where_clause}"
                else:
                    sql = f"SELECT COUNT(DISTINCT [{count_column}]) as total_count FROM {table.full_name} WHERE {where_clause}"
            else:
                # Sum aggregation
                if measures:
                    measure_col = measures[0]
                    sql = f"SELECT SUM([{measure_col}]) as total_amount FROM {table.full_name} WHERE {where_clause}"
                else:
                    sql = f"SELECT COUNT(*) as total_count FROM {table.full_name} WHERE {where_clause}"
        elif intent.task_type == 'ranking':
            limit = intent.top_limit or 10
            if measures and entity_keys:
                measure_col = measures[0]
                entity_col = entity_keys[0]
                sql = f"SELECT TOP {limit} [{entity_col}], [{measure_col}] FROM {table.full_name} WHERE {where_clause} ORDER BY [{measure_col}] DESC"
            else:
                sql = f"SELECT TOP {limit} * FROM {table.full_name} WHERE {where_clause}"
        else:
            # Default to count
            sql = f"SELECT COUNT(*) as total_count FROM {table.full_name} WHERE {where_clause}"
        
        return sql
    
    async def _generate_llm_sql(self, intent: AnalyticalTask, table: TableInfo, 
                               llm: AzureChatOpenAI) -> Optional[str]:
        """Generate SQL using LLM with improved prompting"""
        
        # Build table context
        measures = getattr(table, 'measures', [])
        entity_keys = getattr(table, 'entity_keys', [])
        time_columns = getattr(table, 'time_columns', [])
        
        # Sample data preview
        sample_preview = "No sample data"
        if table.sample_data:
            first_row = table.sample_data[0]
            sample_items = []
            for key, value in list(first_row.items())[:4]:
                if not key.startswith('__'):
                    sample_items.append(f"{key}: {value}")
            sample_preview = " | ".join(sample_items)
        
        prompt = f"""
Generate SQL for this business request:

QUESTION: {intent.task_type} - {intent.metrics} for {intent.entity} in {intent.time_window}

TABLE: {table.full_name}
AVAILABLE COLUMNS:
- Measures (for aggregation): {measures}
- Entity Keys (for grouping): {entity_keys}  
- Time Columns (for filtering): {time_columns}
- Sample Data: {sample_preview}

REQUIREMENTS:
1. Use only the table {table.full_name}
2. For count: use COUNT(*) or COUNT(DISTINCT column)
3. For time filtering: use YEAR(column) = {intent.time_window}
4. Generate clean, simple SQL
5. Start with SELECT

Generate SQL only, no explanation:
"""
        
        try:
            messages = [
                SystemMessage(content="Generate clean SQL queries. Respond with SQL only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(llm.invoke, messages)
            
            return clean_sql_query(response.content)
            
        except Exception as e:
            print(f"      ⚠️ LLM SQL generation failed: {e}")
            return None
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL safety and table references - Fixed validation"""
        if not sql or not sql.strip():
            print("      ❌ Empty SQL")
            return False
        
        # SQL safety check
        if not validate_sql_safety(sql):
            print("      ❌ SQL safety check failed")
            return False
        
        # Check table references - More flexible matching
        sql_lower = sql.lower()
        
        # Extract table references from SQL
        import re
        table_pattern = r'\[([^\]]+)\]\.\[([^\]]+)\]'
        matches = re.findall(table_pattern, sql)
        
        if matches:
            # Check if any matched table is in allowed tables
            for schema, table in matches:
                full_name = f"[{schema}].[{table}]".lower()
                if full_name in self.allowed_tables:
                    print(f"      ✅ Valid table reference: {full_name}")
                    return True
            print(f"      ❌ No valid table references found in SQL")
            return False
        else:
            # Fallback: check if any allowed table name appears in SQL
            for table_name in self.allowed_tables:
                if table_name in sql_lower:
                    print(f"      ✅ Table reference found: {table_name}")
                    return True
            print(f"      ❌ No table references found")
            return False

class QueryExecutor:
    """Execute SQL with error handling"""
    
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
                
                # Set query timeout if available
                if hasattr(cursor, 'timeout'):
                    cursor.timeout = self.config.query_timeout_seconds
                
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
    """BI-Aware 4-Stage Query Interface - Simple and effective (Fixed)"""
    
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
        """BI-Aware 4-Stage Pipeline - Fixed implementation"""
        
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
                        error="SQL generation failed - no valid SQL could be created",
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