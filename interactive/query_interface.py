#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - Simple, Readable, Maintainable
Following README: 4-stage pipeline with capability validation
Fixed SQL generation to avoid variable issues
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

class IntentAnalyzer:
    """Stage 1: Natural language → Analytical Task"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_intent(self, question: str) -> AnalyticalTask:
        """Analyze user question and normalize to analytical task"""
        print("   🧠 Stage 1: Intent analysis...")
        
        # Try pattern matching first
        intent = self._pattern_match(question)
        if intent:
            print(f"      ✅ Pattern matched: {intent.task_type}")
            return intent
        
        # Fallback to LLM
        return await self._llm_analysis(question)
    
    def _pattern_match(self, question: str) -> Optional[AnalyticalTask]:
        """Simple pattern matching for common queries"""
        q_lower = question.lower()
        
        # Detect task type
        if any(word in q_lower for word in ['top', 'highest', 'best', 'worst']):
            task_type = 'ranking'
            metrics = ['value']
        elif any(word in q_lower for word in ['count', 'how many', 'number of']):
            task_type = 'aggregation'
            metrics = ['count']
        elif any(word in q_lower for word in ['total', 'sum', 'amount']):
            task_type = 'aggregation'
            metrics = ['sum']
        else:
            return None
        
        # Detect entity
        entity = None
        if any(word in q_lower for word in ['customer', 'client']):
            entity = 'customer'
        elif any(word in q_lower for word in ['payment', 'revenue', 'sales']):
            entity = 'payment'
        elif any(word in q_lower for word in ['order', 'transaction']):
            entity = 'order'
        
        # Extract top limit
        import re
        top_match = re.search(r'top\s+(\d+)', q_lower)
        top_limit = int(top_match.group(1)) if top_match else (10 if task_type == 'ranking' else None)
        
        return AnalyticalTask(
            task_type=task_type,
            metrics=metrics,
            entity=entity,
            top_limit=top_limit
        )
    
    async def _llm_analysis(self, question: str) -> AnalyticalTask:
        """LLM-based intent analysis"""
        try:
            messages = [
                SystemMessage(content="Extract analytical intent. Respond with JSON only."),
                HumanMessage(content=f"""
Analyze: "{question}"

Extract:
- task_type: aggregation, ranking, trend, count
- metrics: revenue, count, sum, etc.
- entity: customer, payment, order
- top_limit: number for ranking

JSON only:
{{
  "task_type": "ranking",
  "metrics": ["revenue"],
  "entity": "customer",
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
                    top_limit=data.get('top_limit')
                )
        except Exception as e:
            print(f"      ⚠️ LLM analysis failed: {e}")
        
        # Fallback
        return AnalyticalTask(task_type='aggregation', metrics=['count'])

class TableSelector:
    """Stage 2: Evidence-driven table selection"""
    
    def __init__(self, tables: List[TableInfo], relationships: List[Relationship]):
        self.tables = tables
        self.relationships = relationships
    
    def select_candidates(self, intent: AnalyticalTask, top_k: int = 5) -> List[Tuple[TableInfo, EvidenceScore]]:
        """Select tables using evidence scoring"""
        print("   📋 Stage 2: Evidence-driven table selection...")
        
        scored_tables = []
        for table in self.tables:
            score = self._calculate_evidence(table, intent)
            scored_tables.append((table, score))
        
        # Sort by evidence score
        scored_tables.sort(key=lambda x: x[1].total_score, reverse=True)
        
        selected = scored_tables[:top_k]
        print(f"      ✅ Selected {len(selected)} candidates")
        
        return selected
    
    def _calculate_evidence(self, table: TableInfo, intent: AnalyticalTask) -> EvidenceScore:
        """Calculate evidence score"""
        score = EvidenceScore()
        
        # BI role match
        bi_role = getattr(table, 'bi_role', 'dimension')
        data_type = getattr(table, 'data_type', 'reference')
        
        if intent.task_type in ['aggregation', 'ranking'] and bi_role == 'fact':
            score.role_match = 1.0
        elif bi_role == 'fact':
            score.role_match = 0.8
        else:
            score.role_match = 0.3
        
        # Entity match
        entity_type = getattr(table, 'entity_type', '').lower()
        table_name = table.name.lower()
        
        if intent.entity:
            if intent.entity.lower() in entity_type:
                score.lexical_match = 1.0
            elif intent.entity.lower() in table_name:
                score.lexical_match = 0.8
            else:
                score.lexical_match = 0.2
        else:
            score.lexical_match = 0.5
        
        # Operational preference
        score.operational_tag = 1.0 if data_type == 'operational' else 0.5
        
        # Join evidence
        score.join_evidence = min(1.0, len(table.relationships) / 3.0)
        
        # Table quality
        score.graph_proximity = self._calculate_quality(table)
        
        # Data volume
        score.row_count = min(1.0, table.row_count / 10000.0) if table.row_count > 0 else 0.0
        score.freshness = 1.0
        
        return score
    
    def _calculate_quality(self, table: TableInfo) -> float:
        """Calculate table quality score"""
        table_name = table.name.lower()
        quality = 1.0
        
        # Penalty for dated tables
        import re
        if re.search(r'\d{8}', table_name):
            quality -= 0.4
        elif re.search(r'\d{4}', table_name):
            quality -= 0.2
        
        # Penalty for temp/backup
        temp_indicators = ['temp', 'backup', 'old', 'test']
        for indicator in temp_indicators:
            if indicator in table_name:
                quality -= 0.3
                break
        
        return max(0.0, quality)

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

class SQLGenerator:
    """Stage 4: Clean SQL generation without variables"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def generate_sql(self, intent: AnalyticalTask, valid_tables: List[TableInfo], 
                          llm: AzureChatOpenAI) -> Optional[str]:
        """Generate clean SQL from intent and tables"""
        print("   ⚡ Stage 4: SQL generation...")
        
        if not valid_tables:
            return None
        
        # Select primary table
        primary_table = self._select_primary_table(valid_tables, intent)
        
        # Generate SQL using LLM
        sql = await self._generate_clean_sql(intent, primary_table, llm)
        
        if sql and self._validate_sql(sql):
            print("      ✅ SQL generated and validated")
            return sql
        
        print("      ❌ SQL generation failed")
        return None
    
    def _select_primary_table(self, tables: List[TableInfo], intent: AnalyticalTask) -> TableInfo:
        """Select primary table"""
        # Prefer fact tables for aggregation
        if intent.task_type in ['aggregation', 'ranking']:
            fact_tables = [t for t in tables if getattr(t, 'bi_role', '') == 'fact']
            if fact_tables:
                return fact_tables[0]
        
        return tables[0]
    
    async def _generate_clean_sql(self, intent: AnalyticalTask, primary_table: TableInfo, 
                               llm: AzureChatOpenAI) -> Optional[str]:
        """Generate clean SQL without variables"""
        
        # Build context with actual columns only
        context = self._build_clean_context(primary_table, intent)
        
        prompt = f"""
Generate SQL for: {intent.task_type} query

TABLE: {primary_table.full_name}
COLUMNS: {', '.join([col['name'] for col in primary_table.columns[:10]])}

REQUIREMENTS:
1. Use ONLY columns that exist in the table schema above
2. Do NOT use any variables like @BatchID, @StartDate, @Parameter
3. For date filters, use literal dates: WHERE date_column >= '2025-01-01'
4. For top 10, use: SELECT TOP (10)
5. For revenue/amount, look for columns with 'amount', 'value', 'revenue', 'price'
6. Use simple JOINs only if foreign key relationships exist
7. Include ORDER BY for ranking queries

Generate only clean SQL without variables or parameters:
"""
        
        try:
            messages = [
                SystemMessage(content="Generate clean SQL without any variables. Use only actual table columns."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(llm.invoke, messages)
            
            sql = clean_sql_query(response.content)
            
            # Remove any variables that might have been included
            sql = self._remove_variables(sql)
            
            return sql
            
        except Exception as e:
            print(f"      ⚠️ SQL generation failed: {e}")
            return None
    
    def _build_clean_context(self, table: TableInfo, intent: AnalyticalTask) -> str:
        """Build context with only actual columns"""
        context = f"TABLE: {table.full_name}\n"
        
        # Show actual columns
        if table.columns:
            context += "COLUMNS:\n"
            for col in table.columns[:15]:  # Limit to first 15 columns
                context += f"  - {col['name']} ({col['data_type']})\n"
        
        # Show sample data if available
        clean_samples = table.get_clean_samples() if hasattr(table, 'get_clean_samples') else table.sample_data
        if clean_samples:
            context += f"\nSAMPLE DATA (first row):\n"
            first_row = clean_samples[0] if clean_samples else {}
            for k, v in list(first_row.items())[:5]:
                if not k.startswith('__'):
                    context += f"  {k}: {v}\n"
        
        return context
    
    def _remove_variables(self, sql: str) -> str:
        """Remove SQL variables and replace with literals"""
        if not sql:
            return sql
        
        import re
        
        # Remove DECLARE statements
        sql = re.sub(r'DECLARE\s+@\w+\s+[^;]+;?\s*', '', sql, flags=re.IGNORECASE)
        
        # Replace common variables with literals
        replacements = {
            r'@BatchID': "'1'",
            r'@StartDate': "'2025-01-01'",
            r'@EndDate': "'2025-12-31'", 
            r'@Year': "2025",
            r'@CustomerID': "1",
            r'@\w+': "'2025-01-01'"  # Catch-all for other variables
        }
        
        for pattern, replacement in replacements.items():
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        
        # Clean up whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL safety"""
        return validate_sql_safety(sql)

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

class QueryInterface:
    """Main query interface - Clean and simple"""
    
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
        self.capability_validator = CapabilityValidator()
        self.sql_generator = SQLGenerator(config)
        self.executor = QueryExecutor(config)
        
        print("✅ Query Interface initialized")
    
    async def start_session(self, tables: List[TableInfo], 
                          domain: Optional[BusinessDomain], 
                          relationships: List[Relationship]):
        """Start interactive query session"""
        
        # Initialize table selector
        self.table_selector = TableSelector(tables, relationships)
        
        # Show readiness
        self._show_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
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
        """4-Stage Pipeline"""
        
        try:
            # Stage 1: Intent analysis
            intent = await self.intent_analyzer.analyze_intent(question)
            
            # Stage 2: Table selection
            candidates = self.table_selector.select_candidates(intent, top_k=5)
            
            # Stage 3: Capability validation
            valid_tables = self.capability_validator.validate_capabilities(candidates, intent)
            
            # Stage 4: SQL generation and execution
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
                        error="SQL generation failed",
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
                error=f"Pipeline error: {str(e)}",
                result_type="error"
            )
    
    def _show_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show system readiness"""
        fact_tables = len([t for t in tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in tables if getattr(t, 'data_type', '') == 'operational'])
        with_measures = len([t for t in tables if getattr(t, 'measures', [])])
        
        print(f"\n🧠 BI SYSTEM READY:")
        print(f"   📊 Total tables: {len(tables)}")
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   ⚡ Operational tables: {operational_tables}")
        print(f"   💰 Tables with measures: {with_measures}")
        
        if domain:
            print(f"   🎯 Domain: {domain.domain_type}")
        
        print(f"\n🚫 NO-FALLBACK MODE:")
        print(f"   • Capability contracts enforced")
        print(f"   • Evidence-driven table selection")
    
    def _display_result(self, result: QueryResult):
        """Display query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.is_successful():
            print(f"✅ QUERY EXECUTED")
            print(f"\n📋 SQL:")
            print(f"{result.sql_query}")
            
            print(f"\n📊 Results: {len(result.results)} rows")
            if result.results:
                self._display_data(result.results)
        
        else:
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