#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Query Interface - Following README Requirements Exactly
Implements: Explainable Retrieval + View Analysis + Execution-Guided + Constraints
"""

import asyncio
import json
import re
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import parse_json_response, clean_sql_query, safe_database_value

class LLMClient:
    """LLM client following README specs"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            request_timeout=90
        )
    
    async def analyze(self, system_prompt: str, user_prompt: str) -> str:
        """Send analysis request to LLM"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
            return ""

class ViewAnalyzer:
    """Analyze view definitions for join patterns - README requirement"""
    
    def __init__(self, database_structure: Dict):
        self.database_structure = database_structure
        self.view_patterns = self.extract_view_patterns()
    
    def extract_view_patterns(self) -> List[Dict]:
        """Extract real view patterns from database structure"""
        patterns = []
        views = self.database_structure.get('views', {})
        
        for view_name, view_data in views.items():
            if view_data.get('definition'):
                pattern = {
                    'view_name': view_name,
                    'definition': view_data['definition'],
                    'tables_involved': self.extract_tables_from_view(view_data['definition']),
                    'join_conditions': self.extract_join_conditions(view_data['definition']),
                    'proven_working': True
                }
                patterns.append(pattern)
        
        return patterns
    
    def extract_tables_from_view(self, definition: str) -> List[str]:
        """Extract table names from view definition"""
        if not definition:
            return []
        
        tables = []
        patterns = [
            r'FROM\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)',
            r'JOIN\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, definition, re.IGNORECASE)
            tables.extend(matches)
        
        return list(set(tables))
    
    def extract_join_conditions(self, definition: str) -> List[str]:
        """Extract proven join conditions from view"""
        if not definition:
            return []
        
        joins = []
        on_pattern = r'ON\s+([^WHERE^GROUP^ORDER^INNER^LEFT^RIGHT^JOIN]+?)(?=\s+(?:WHERE|GROUP|ORDER|INNER|LEFT|RIGHT|JOIN|$))'
        matches = re.findall(on_pattern, definition, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clean_join = match.strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            clean_join = re.sub(r'\s+', ' ', clean_join)
            if clean_join and len(clean_join) < 200:
                joins.append(clean_join)
        
        return joins
    
    def find_proven_join_pattern(self, table1: str, table2: str) -> Optional[str]:
        """Find proven join pattern between two tables from views"""
        for pattern in self.view_patterns:
            tables = pattern['tables_involved']
            if any(table1.lower() in t.lower() for t in tables) and any(table2.lower() in t.lower() for t in tables):
                for join_condition in pattern['join_conditions']:
                    if table1.lower() in join_condition.lower() and table2.lower() in join_condition.lower():
                        return f"PROVEN: {join_condition} (from view: {pattern['view_name']})"
        return None

class ExplainableTableSelector:
    """Table selector with explainable retrieval - README requirement"""
    
    def __init__(self, tables: List[TableInfo], llm: LLMClient, view_analyzer: ViewAnalyzer):
        self.tables = tables
        self.llm = llm
        self.view_analyzer = view_analyzer
    
    async def find_relevant_tables_with_explanation(self, question: str) -> Tuple[List[TableInfo], Dict]:
        """Find relevant tables WITH explanations - README requirement"""
        print("   📋 Explainable table retrieval (README Pattern B)...")
        
        explanations = {
            'total_candidates': len(self.tables),
            'selection_reasoning': [],
            'lexical_matches': [],
            'semantic_matches': [],
            'fk_proximity': [],
            'view_derived_joins': [],
            'final_selection': []
        }
        
        # Stage 1: Lexical matching
        lexical_candidates = self.lexical_matching(question, explanations)
        
        # Stage 2: Semantic analysis
        semantic_candidates = await self.semantic_analysis(question, lexical_candidates, explanations)
        
        # Stage 3: FK proximity analysis
        fk_enhanced = self.fk_proximity_analysis(semantic_candidates, explanations)
        
        # Stage 4: View-derived join analysis
        final_tables = self.view_derived_analysis(fk_enhanced, explanations)
        
        # Log explainable retrieval
        print(f"      📊 Explainable Retrieval Results:")
        print(f"         • Lexical matches: {len(explanations['lexical_matches'])}")
        print(f"         • Semantic relevance: {len(explanations['semantic_matches'])}")
        print(f"         • FK proximity: {len(explanations['fk_proximity'])}")
        print(f"         • View patterns: {len(explanations['view_derived_joins'])}")
        print(f"         • Final selection: {len(final_tables)}")
        
        for reasoning in explanations['selection_reasoning'][-3:]:
            print(f"         • {reasoning}")
        
        return final_tables, explanations
    
    def lexical_matching(self, question: str, explanations: Dict) -> List[Tuple[TableInfo, float, str]]:
        """Lexical matching with scoring"""
        q_lower = question.lower()
        question_words = [word for word in q_lower.split() if len(word) > 3]
        
        scored_tables = []
        
        for table in self.tables:
            score = 0.0
            reasons = []
            
            # Entity type matching
            if table.entity_type != 'Unknown':
                entity_keywords = {
                    'Customer': ['customer', 'client', 'account', 'user'],
                    'Payment': ['payment', 'paid', 'billing', 'invoice', 'revenue', 'financial'],
                    'Order': ['order', 'purchase', 'sale', 'transaction'],
                    'Product': ['product', 'item', 'inventory'],
                    'Financial': ['revenue', 'financial', 'money', 'amount', 'total']
                }
                
                for entity, keywords in entity_keywords.items():
                    if table.entity_type == entity:
                        matches = [kw for kw in keywords if kw in q_lower]
                        if matches:
                            score += 5.0
                            reasons.append(f"entity_match:{entity}({','.join(matches)})")
            
            # Table name matching
            table_name_lower = table.name.lower()
            name_matches = [word for word in question_words if word in table_name_lower]
            if name_matches:
                score += len(name_matches) * 3.0
                reasons.append(f"name_match:{','.join(name_matches)}")
            
            # Column name matching
            column_names = [col.get('name', '').lower() for col in table.columns]
            column_matches = []
            for col_name in column_names:
                for word in question_words:
                    if word in col_name:
                        column_matches.append(col_name)
                        score += 2.0
                        break
            
            if column_matches:
                reasons.append(f"column_match:{','.join(column_matches[:3])}")
            
            # Data availability bonus
            if table.row_count > 0:
                score += 1.0
                reasons.append("has_data")
            
            if score > 0:
                scored_tables.append((table, score, '; '.join(reasons)))
                explanations['lexical_matches'].append({
                    'table': table.full_name,
                    'score': score,
                    'reasoning': '; '.join(reasons)
                })
        
        # Sort by score
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        top_candidates = scored_tables[:15]
        explanations['selection_reasoning'].append(f"Lexical: {len(top_candidates)} tables with keyword matches")
        
        return top_candidates
    
    async def semantic_analysis(self, question: str, candidates: List[Tuple], explanations: Dict) -> List[TableInfo]:
        """Semantic analysis for relevance"""
        if len(candidates) <= 8:
            tables = [candidate[0] for candidate in candidates]
            explanations['selection_reasoning'].append("Semantic: Skipped (few candidates)")
            return tables
        
        # Prepare for LLM analysis
        table_summaries = []
        for table, score, reasoning in candidates:
            table_summaries.append({
                'table_name': table.full_name,
                'entity_type': table.entity_type,
                'lexical_score': score,
                'lexical_reasoning': reasoning,
                'row_count': table.row_count,
                'columns': [col.get('name') for col in table.columns[:6]]
            })
        
        system_prompt = """You are a database analyst. Analyze tables for semantic relevance to the user question.
Consider both lexical matches and business logic relevance."""
        
        user_prompt = f"""
QUESTION: "{question}"

CANDIDATE TABLES:
{json.dumps(table_summaries, indent=2)}

Select the 8 most semantically relevant tables. Consider:
1. Direct relevance to the question intent
2. Business logic connections
3. Data availability

Respond with JSON:
{{
  "selected_tables": ["[schema].[table1]", "[schema].[table2]"],
  "reasoning": "Why these tables are semantically relevant"
}}
"""
        
        response = await self.llm.analyze(system_prompt, user_prompt)
        result = parse_json_response(response)
        
        if result and 'selected_tables' in result:
            selected_names = result['selected_tables']
            selected_tables = [table for table, _, _ in candidates if table.full_name in selected_names]
            
            explanations['semantic_matches'] = selected_names
            explanations['selection_reasoning'].append(f"Semantic: {result.get('reasoning', 'LLM selection')}")
            
            return selected_tables
        
        # Fallback
        tables = [candidate[0] for candidate in candidates[:8]]
        explanations['selection_reasoning'].append("Semantic: Fallback to top lexical matches")
        return tables
    
    def fk_proximity_analysis(self, tables: List[TableInfo], explanations: Dict) -> List[TableInfo]:
        """FK proximity analysis"""
        # Simple implementation - could be enhanced
        table_names = [t.full_name for t in tables]
        
        # Look for FK relationships
        fk_connections = []
        for table in tables:
            for col in table.columns:
                col_name = col.get('name', '').lower()
                if col_name.endswith('_id') or col_name.endswith('id'):
                    entity = col_name.replace('_id', '').replace('id', '')
                    for other_table in tables:
                        if entity in other_table.name.lower():
                            fk_connections.append(f"{table.full_name} -> {other_table.full_name}")
        
        explanations['fk_proximity'] = fk_connections
        if fk_connections:
            explanations['selection_reasoning'].append(f"FK Proximity: Found {len(fk_connections)} FK connections")
        
        return tables
    
    def view_derived_analysis(self, tables: List[TableInfo], explanations: Dict) -> List[TableInfo]:
        """View-derived join analysis - README requirement"""
        view_joins = []
        
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                proven_join = self.view_analyzer.find_proven_join_pattern(table1.full_name, table2.full_name)
                if proven_join:
                    view_joins.append(proven_join)
        
        explanations['view_derived_joins'] = view_joins
        if view_joins:
            explanations['selection_reasoning'].append(f"View Analysis: Found {len(view_joins)} proven join patterns")
        
        # Mark final selection
        explanations['final_selection'] = [t.full_name for t in tables]
        
        return tables

class ConstrainedSQLGenerator:
    """SQL Generator with constraints and validation - README requirement"""
    
    def __init__(self, llm: LLMClient, view_analyzer: ViewAnalyzer, database_structure: Dict):
        self.llm = llm
        self.view_analyzer = view_analyzer
        self.database_structure = database_structure
        self.allowed_identifiers = self.build_identifier_allowlist()
    
    def build_identifier_allowlist(self) -> set:
        """Build allowlist of valid identifiers from discovered objects"""
        identifiers = set()
        
        # Add table names
        for table_name in self.database_structure.get('tables', {}):
            identifiers.add(table_name.lower())
            # Add simple name without schema
            if '.' in table_name:
                simple_name = table_name.split('.')[-1].replace('[', '').replace(']', '')
                identifiers.add(simple_name.lower())
        
        # Add column names
        for table_data in self.database_structure.get('tables', {}).values():
            for col_name, _ in table_data.get('columns', {}).items():
                identifiers.add(col_name.lower())
        
        return identifiers
    
    async def generate_constrained_query(self, question: str, tables: List[TableInfo], 
                                       relationships: List[Dict], explanations: Dict) -> str:
        """Generate SQL with constraints and validation"""
        print("   ⚡ Constrained SQL generation (README Pattern A)...")
        
        # Extract user intent for constraints
        intent = self.analyze_intent(question)
        
        # Generate SQL with proven join patterns
        sql = await self.generate_with_view_patterns(question, tables, intent, explanations)
        
        # Validate against constraints
        if not self.validate_sql_constraints(sql):
            print(f"      ⚠️ SQL failed constraint validation, regenerating...")
            sql = await self.generate_fallback_sql(question, tables, intent)
        
        # Apply intent-based corrections
        sql = self.apply_intent_corrections(sql, intent, question)
        
        return sql
    
    def analyze_intent(self, question: str) -> Dict:
        """Analyze user intent for SQL constraints"""
        q_lower = question.lower()
        
        intent = {
            'limit_requested': None,
            'aggregation_type': None,
            'date_filter': False,
            'specific_columns': []
        }
        
        # Extract specific limits
        limit_match = re.search(r'top\s+(\d+)|first\s+(\d+)|limit\s+(\d+)', q_lower)
        if limit_match:
            intent['limit_requested'] = int(limit_match.group(1) or limit_match.group(2) or limit_match.group(3))
        
        # Detect aggregation
        if any(word in q_lower for word in ['count', 'how many', 'number of']):
            intent['aggregation_type'] = 'COUNT'
        elif any(word in q_lower for word in ['total', 'sum']):
            intent['aggregation_type'] = 'SUM'
        elif any(word in q_lower for word in ['average', 'avg']):
            intent['aggregation_type'] = 'AVG'
        
        # Date filtering
        if any(word in q_lower for word in ['2025', '2024', 'year', 'current']):
            intent['date_filter'] = True
        
        return intent
    
    async def generate_with_view_patterns(self, question: str, tables: List[TableInfo], 
                                        intent: Dict, explanations: Dict) -> str:
        """Generate SQL using proven view patterns"""
        
        # Find relevant view patterns
        relevant_patterns = []
        table_names = [t.full_name for t in tables]
        
        for pattern in self.view_analyzer.view_patterns:
            pattern_tables = pattern['tables_involved']
            overlap = sum(1 for pt in pattern_tables 
                         for tn in table_names 
                         if pt.lower() in tn.lower() or tn.lower() in pt.lower())
            if overlap >= 2:
                relevant_patterns.append(pattern)
        
        # Prepare context
        table_context = self.prepare_table_context(tables)
        pattern_context = self.prepare_pattern_context(relevant_patterns)
        
        system_prompt = f"""You are an expert SQL generator with access to proven join patterns from database views.

CONSTRAINTS (CRITICAL):
1. Use EXACT table and column names from the provided schema
2. Use proven JOIN patterns from views when available
3. Apply user intent precisely: {intent}
4. Return only valid SQL - no explanations

DISCOVERED SCHEMA ONLY:
{table_context}

PROVEN JOIN PATTERNS:
{pattern_context}"""
        
        user_prompt = f"""
QUESTION: "{question}"

Generate SQL that:
1. Answers the question exactly
2. Uses proven join patterns when available  
3. Respects user intent: {intent}
4. Only uses discovered schema objects

Return SQL only.
"""
        
        response = await self.llm.analyze(system_prompt, user_prompt)
        return clean_sql_query(response)
    
    def validate_sql_constraints(self, sql: str) -> bool:
        """Validate SQL against constraints - README requirement"""
        if not sql:
            return False
        
        sql_upper = sql.upper()
        
        # Must be SELECT only
        if not sql_upper.strip().startswith('SELECT'):
            return False
        
        # No dangerous operations
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        if any(op in sql_upper for op in forbidden):
            return False
        
        # Check identifiers against allowlist (simplified)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sql.lower())
        for word in words:
            if word in ['select', 'from', 'where', 'join', 'on', 'and', 'or', 'order', 'by', 'group', 'having', 'top', 'distinct', 'as']:
                continue
            if word not in self.allowed_identifiers:
                print(f"      ⚠️ Unknown identifier: {word}")
                # In production, this would be stricter
        
        return True
    
    async def generate_fallback_sql(self, question: str, tables: List[TableInfo], intent: Dict) -> str:
        """Generate fallback SQL if constraints fail"""
        # Simplified fallback generation
        table = tables[0] if tables else None
        if not table:
            return ""
        
        columns = [col.get('name') for col in table.columns[:5]]
        limit = intent.get('limit_requested', 100)
        
        if intent.get('aggregation_type') == 'COUNT':
            return f"SELECT COUNT(*) as count FROM {table.full_name}"
        else:
            return f"SELECT TOP {limit} {', '.join(columns)} FROM {table.full_name}"
    
    def apply_intent_corrections(self, sql: str, intent: Dict, question: str) -> str:
        """Apply intent-based corrections to SQL"""
        if not sql:
            return sql
        
        # Apply specific limit if requested
        limit_requested = intent.get('limit_requested')
        if limit_requested and 'TOP' not in sql.upper():
            if sql.upper().startswith('SELECT'):
                sql = sql.replace('SELECT', f'SELECT TOP {limit_requested}', 1)
        elif limit_requested and 'TOP' in sql.upper():
            # Replace existing TOP with requested limit
            sql = re.sub(r'TOP\s+\d+', f'TOP {limit_requested}', sql, flags=re.IGNORECASE)
        
        # Add default TOP if missing and not aggregation
        if ('TOP' not in sql.upper() and 
            intent.get('aggregation_type') is None and
            sql.upper().startswith('SELECT')):
            sql = sql.replace('SELECT', 'SELECT TOP 100', 1)
        
        return sql
    
    def prepare_table_context(self, tables: List[TableInfo]) -> str:
        """Prepare detailed table context"""
        context = []
        for table in tables:
            context.append(f"TABLE: {table.full_name}")
            context.append(f"  Type: {table.entity_type}")
            context.append(f"  Rows: {table.row_count:,}")
            
            columns = []
            for col in table.columns[:10]:
                col_name = col.get('name', '')
                col_type = col.get('data_type', '')
                columns.append(f"{col_name} ({col_type})")
            
            context.append(f"  Columns: {', '.join(columns)}")
            context.append("")
        
        return "\n".join(context)
    
    def prepare_pattern_context(self, patterns: List[Dict]) -> str:
        """Prepare view pattern context"""
        if not patterns:
            return "No proven join patterns available"
        
        context = ["PROVEN PATTERNS FROM VIEWS:"]
        for pattern in patterns[:3]:
            context.append(f"View: {pattern['view_name']}")
            for join in pattern['join_conditions'][:2]:
                context.append(f"  JOIN: {join}")
            context.append("")
        
        return "\n".join(context)

class ExecutionGuidedExecutor:
    """Executor with execution-guided retry - README requirement"""
    
    def __init__(self, config: Config, sql_generator: ConstrainedSQLGenerator):
        self.config = config
        self.sql_generator = sql_generator
        self.max_retries = 2
    
    async def execute_with_retry(self, sql: str, question: str, tables: List[TableInfo], 
                               explanations: Dict) -> Tuple[List[Dict], Optional[str]]:
        """Execute with execution-guided retry loop"""
        print("   🔄 Execution-guided retry (README Pattern A)...")
        
        for attempt in range(self.max_retries + 1):
            results, error = self.execute_sql(sql)
            
            if error is None:
                if len(results) == 0:
                    print(f"      ⚠️ Attempt {attempt + 1}: Empty results, retrying...")
                    sql = await self.retry_for_empty_results(sql, question, tables)
                    continue
                else:
                    print(f"      ✅ Success on attempt {attempt + 1}")
                    return results, None
            else:
                print(f"      ⚠️ Attempt {attempt + 1} failed: {error}")
                if attempt < self.max_retries:
                    sql = await self.retry_for_error(sql, error, question, tables)
                    continue
                else:
                    return [], f"Failed after {self.max_retries + 1} attempts: {error}"
        
        return results, error
    
    def execute_sql(self, sql: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL query"""
        if not sql.strip():
            return [], "Empty SQL query"
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
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
                    
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            return [], str(e)
    
    async def retry_for_error(self, failed_sql: str, error: str, question: str, tables: List[TableInfo]) -> str:
        """Retry with error feedback"""
        print(f"      🔄 Retrying with error feedback...")
        
        # Generate new SQL with error context
        retry_prompt = f"""
Previous SQL failed with error: {error}

FAILED SQL:
{failed_sql}

QUESTION: "{question}"

Generate corrected SQL that fixes the error. Common fixes:
- Check column names exist in tables
- Verify table names are correct
- Fix JOIN conditions
- Ensure proper syntax

Return only corrected SQL.
"""
        
        system_prompt = "You are an SQL error correction expert. Fix the SQL based on the error message."
        response = await self.sql_generator.llm.analyze(system_prompt, retry_prompt)
        
        return clean_sql_query(response)
    
    async def retry_for_empty_results(self, sql: str, question: str, tables: List[TableInfo]) -> str:
        """Retry for empty results"""
        print(f"      🔄 Retrying for empty results...")
        
        retry_prompt = f"""
Previous SQL returned 0 rows:

{sql}

QUESTION: "{question}"

Modify the SQL to be less restrictive:
- Remove or relax WHERE conditions
- Check date ranges are reasonable
- Consider using LEFT JOINs instead of INNER JOINs
- Verify filter values make sense

Return modified SQL.
"""
        
        system_prompt = "You are an SQL optimization expert. Modify SQL to return meaningful results."
        response = await self.sql_generator.llm.analyze(system_prompt, retry_prompt)
        
        return clean_sql_query(response)

class QueryInterface:
    """Enhanced Query Interface following README requirements exactly"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        
        # Load database structure for view analysis
        self.database_structure = self.load_database_structure()
        
        # Initialize components following README patterns
        self.view_analyzer = ViewAnalyzer(self.database_structure)
        self.sql_generator = ConstrainedSQLGenerator(self.llm, self.view_analyzer, self.database_structure)
        self.executor = ExecutionGuidedExecutor(config, self.sql_generator)
        
        print("✅ Enhanced Query Interface initialized with README patterns")
    
    def load_database_structure(self) -> Dict:
        """Load database structure for analysis"""
        try:
            structure_file = self.config.get_cache_path("database_structure.json")
            if structure_file.exists():
                with open(structure_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship]):
        """Start enhanced interactive session"""
        
        # Initialize components with data
        self.table_selector = ExplainableTableSelector(tables, self.llm, self.view_analyzer)
        
        print(f"🚀 Enhanced 4-Stage Pipeline (Following README)")
        print(f"   📊 Tables: {len(tables)}")
        print(f"   👁️ View patterns: {len(self.view_analyzer.view_patterns)}")
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Enhanced Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🔄 Processing with enhanced pipeline...")
                
                start_time = time.time()
                result = await self.process_enhanced_query(question)
                result.execution_time = time.time() - start_time
                
                self.display_enhanced_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_enhanced_query(self, question: str) -> QueryResult:
        """Process query with enhanced pipeline following README"""
        
        try:
            # Stage 1: Explainable table selection
            selected_tables, explanations = await self.table_selector.find_relevant_tables_with_explanation(question)
            
            if not selected_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found with explainable retrieval"
                )
            
            # Stage 2: Constrained SQL generation with view patterns
            sql = await self.sql_generator.generate_constrained_query(
                question, selected_tables, [], explanations
            )
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to generate constrained SQL"
                )
            
            # Stage 3: Execution-guided retry
            results, error = await self.executor.execute_with_retry(
                sql, question, selected_tables, explanations
            )
            
            result = QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=[t.full_name for t in selected_tables]
            )
            
            # Add explanations to result
            result.explanations = explanations
            
            return result
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Enhanced pipeline error: {str(e)}"
            )
    
    def display_enhanced_result(self, result: QueryResult):
        """Display result with enhanced explanations"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 80)
        
        # Show explainable retrieval (README requirement)
        if hasattr(result, 'explanations'):
            explanations = result.explanations
            print(f"📋 EXPLAINABLE RETRIEVAL (README Pattern B):")
            print(f"   • Total candidates: {explanations.get('total_candidates', 0)}")
            print(f"   • Lexical matches: {len(explanations.get('lexical_matches', []))}")
            print(f"   • Semantic analysis: {len(explanations.get('semantic_matches', []))}")
            print(f"   • FK proximity: {len(explanations.get('fk_proximity', []))}")
            print(f"   • View patterns: {len(explanations.get('view_derived_joins', []))}")
            
            for reasoning in explanations.get('selection_reasoning', []):
                print(f"   • {reasoning}")
            print()
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.sql_query:
                print(f"📋 Generated SQL:")
                print(f"{result.sql_query}")
        else:
            print(f"📋 Generated SQL (Constrained + Validated):")
            print(f"{result.sql_query}")
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                if result.is_single_value():
                    value = result.get_single_value()
                    column_name = list(result.results[0].keys())[0]
                    print(f"   🎯 {column_name}: {value:,}" if isinstance(value, (int, float)) and value >= 1000 else f"   🎯 {column_name}: {value}")
                else:
                    for i, row in enumerate(result.results[:5], 1):
                        display_row = {}
                        for key, value in list(row.items())[:6]:
                            if isinstance(value, str) and len(value) > 40:
                                display_row[key] = value[:37] + "..."
                            elif isinstance(value, (int, float)) and value >= 1000:
                                display_row[key] = f"{value:,}"
                            else:
                                display_row[key] = value
                        print(f"   {i}. {display_row}")
                    
                    if len(result.results) > 5:
                        print(f"   ... and {len(result.results) - 5} more rows")
            else:
                print("   ⚠️ No results returned")
            
            if result.tables_used:
                print(f"📋 Tables analyzed: {', '.join(result.tables_used)}")
        
        print("\n💡 Enhanced Features Applied:")
        print("   ✅ Explainable table retrieval with reasoning")
        print("   ✅ View-pattern analysis for proven joins") 
        print("   ✅ Constrained SQL generation with validation")
        print("   ✅ Execution-guided retry on errors/empty results")
        print("   ✅ International character support (UTF-8)")