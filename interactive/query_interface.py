#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Query Interface - Using Real View Patterns as LLM Examples
Leverages actual view definitions and patterns from discovery for improved accuracy
"""

import json
import re
import pyodbc
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult


class LLMClient:
    """Enhanced LLM client for SQL generation using view patterns"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            request_timeout=60
        )
    
    async def generate_sql(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL from prompts"""
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


class DataLoader:
    """Enhanced data loader that includes view patterns"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables = []
        self.relationships = []
        self.domain = None
        self.database_structure = {}
        self.view_patterns = []  # NEW: Executable view patterns
        self.business_templates = {}  # NEW: Business query templates
        
    def load_data(self) -> bool:
        """Load all cached data including view patterns"""
        try:
            # Load database structure with view patterns
            db_file = self.config.get_cache_path("database_structure.json")
            if db_file.exists():
                with open(db_file, 'r', encoding='utf-8') as f:
                    self.database_structure = json.load(f)
                
                # Extract view patterns from discovery
                self._extract_view_patterns()
            
            # Load semantic analysis
            semantic_file = self.config.get_cache_path("semantic_analysis.json")
            if semantic_file.exists():
                with open(semantic_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._load_tables(data)
                    self._load_relationships(data)
                    self._load_domain(data)
            
            print(f"   📊 Loaded {len(self.tables)} tables, {len(self.view_patterns)} view patterns")
            return len(self.tables) > 0
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def _extract_view_patterns(self):
        """Extract executable view patterns as LLM examples"""
        view_info = self.database_structure.get('view_info', {})
        self.view_patterns = []
        self.business_templates = {}
        
        for view_name, view_data in view_info.items():
            if view_data.get('execution_success') and view_data.get('sample_data'):
                pattern = {
                    'view_name': view_name,
                    'definition': view_data.get('definition', ''),
                    'business_pattern': view_data.get('business_pattern', {}),
                    'sample_data': view_data.get('sample_data', [])[:3],  # First 3 rows
                    'use_case': view_data.get('business_pattern', {}).get('estimated_use_case', ''),
                    'pattern_type': view_data.get('business_pattern', {}).get('pattern', 'unknown'),
                    'tables_involved': self._extract_tables_from_view(view_data.get('definition', '')),
                    'join_patterns': self._extract_join_patterns(view_data.get('definition', '')),
                    'business_logic': self._extract_business_logic(view_data.get('definition', ''))
                }
                self.view_patterns.append(pattern)
                
                # Group by pattern type for business templates
                pattern_type = pattern['pattern_type']
                if pattern_type not in self.business_templates:
                    self.business_templates[pattern_type] = []
                self.business_templates[pattern_type].append(pattern)
        
        print(f"      ✅ Extracted {len(self.view_patterns)} executable view patterns")
        print(f"      📋 Business patterns: {list(self.business_templates.keys())}")
    
    def _extract_tables_from_view(self, definition: str) -> List[str]:
        """Extract table names mentioned in view definition"""
        if not definition:
            return []
        
        tables = []
        # Look for FROM and JOIN patterns
        patterns = [
            r'FROM\s+(\[?[\w_]+\]?\.\[?[\w_]+\]?)',
            r'JOIN\s+(\[?[\w_]+\]?\.\[?[\w_]+\]?)',
            r'INNER\s+JOIN\s+(\[?[\w_]+\]?\.\[?[\w_]+\]?)',
            r'LEFT\s+JOIN\s+(\[?[\w_]+\]?\.\[?[\w_]+\]?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, definition, re.IGNORECASE)
            tables.extend(matches)
        
        return list(set(tables))
    
    def _extract_join_patterns(self, definition: str) -> List[str]:
        """Extract join patterns from view definition"""
        if not definition:
            return []
        
        joins = []
        # Look for ON clauses
        on_pattern = r'ON\s+([^WHERE^GROUP^ORDER^)]+?)(?=\s+(?:WHERE|GROUP|ORDER|INNER|LEFT|RIGHT|$))'
        matches = re.findall(on_pattern, definition, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clean_join = match.strip().replace('\n', ' ').replace('\r', '')
            if clean_join and len(clean_join) < 200:  # Reasonable length
                joins.append(clean_join)
        
        return joins
    
    def _extract_business_logic(self, definition: str) -> Dict[str, Any]:
        """Extract business logic patterns from view definition"""
        logic = {
            'has_aggregations': False,
            'has_date_filtering': False,
            'has_status_filtering': False,
            'has_calculations': False,
            'complexity': 'simple'
        }
        
        if not definition:
            return logic
        
        def_lower = definition.lower()
        
        # Check for aggregations
        if any(func in def_lower for func in ['sum(', 'count(', 'avg(', 'max(', 'min(']):
            logic['has_aggregations'] = True
        
        # Check for date filtering
        if any(func in def_lower for func in ['dateadd', 'datediff', 'getdate', 'year(', 'month(']):
            logic['has_date_filtering'] = True
        
        # Check for status filtering
        if any(word in def_lower for word in ['status', 'active', 'cancelled', 'expired']):
            logic['has_status_filtering'] = True
        
        # Check for calculations
        if any(op in def_lower for op in ['case when', 'isnull', 'coalesce', '*', '/', '+', '-']):
            logic['has_calculations'] = True
        
        # Determine complexity
        complexity_score = 0
        if logic['has_aggregations']: complexity_score += 1
        if logic['has_date_filtering']: complexity_score += 1
        if logic['has_status_filtering']: complexity_score += 1
        if logic['has_calculations']: complexity_score += 1
        if def_lower.count('join') > 2: complexity_score += 1
        
        if complexity_score >= 3:
            logic['complexity'] = 'complex'
        elif complexity_score >= 1:
            logic['complexity'] = 'medium'
        
        return logic
    
    def _load_tables(self, data: Dict):
        """Load table information"""
        self.tables = []
        for table_data in data.get('tables', []):
            table = TableInfo(
                name=table_data['name'],
                schema=table_data['schema'],
                full_name=table_data['full_name'],
                object_type=table_data['object_type'],
                row_count=table_data['row_count'],
                columns=table_data['columns'],
                sample_data=table_data['sample_data'],
                relationships=table_data.get('relationships', [])
            )
            table.entity_type = table_data.get('entity_type', 'Unknown')
            table.confidence = table_data.get('confidence', 0.0)
            self.tables.append(table)
    
    def _load_relationships(self, data: Dict):
        """Load relationships"""
        self.relationships = []
        for rel_data in data.get('relationships', []):
            self.relationships.append(Relationship(
                from_table=rel_data['from_table'],
                to_table=rel_data['to_table'],
                relationship_type=rel_data['relationship_type'],
                confidence=rel_data['confidence'],
                description=rel_data.get('description', '')
            ))
    
    def _load_domain(self, data: Dict):
        """Load domain information"""
        domain_data = data.get('domain')
        if domain_data:
            self.domain = BusinessDomain(
                domain_type=domain_data['domain_type'],
                industry=domain_data['industry'],
                confidence=domain_data['confidence'],
                sample_questions=domain_data['sample_questions'],
                capabilities=domain_data['capabilities']
            )
    
    def get_view_patterns_by_intent(self, intent: str) -> List[Dict]:
        """Get view patterns matching business intent"""
        matching_patterns = []
        
        for pattern in self.view_patterns:
            pattern_type = pattern.get('pattern_type', '').lower()
            use_case = pattern.get('use_case', '').lower()
            
            # Match by pattern type or use case
            if (intent.lower() in pattern_type or 
                any(word in use_case for word in intent.lower().split()) or
                any(word in pattern_type for word in intent.lower().split())):
                matching_patterns.append(pattern)
        
        return matching_patterns[:5]  # Top 5 matches


class TableSelector:
    """Enhanced table selector using view patterns as examples"""
    
    def __init__(self, tables: List[TableInfo], llm: LLMClient, view_patterns: List[Dict]):
        self.tables = tables
        self.llm = llm
        self.view_patterns = view_patterns
    
    async def find_relevant_tables(self, question: str) -> List[TableInfo]:
        """Find tables using business intent and view pattern examples"""
        
        # Analyze business intent
        intent = self._analyze_business_intent(question)
        
        # Find relevant view patterns as examples
        relevant_patterns = self._find_relevant_view_patterns(intent, question)
        
        # Create enhanced table summaries
        table_summaries = []
        for table in self.tables:
            sample_preview = self._create_sample_preview(table.sample_data)
            business_indicators = self._get_business_indicators(table, question)
            view_usage = self._check_view_usage(table.full_name, relevant_patterns)
            
            table_summaries.append({
                'full_name': table.full_name,
                'entity_type': table.entity_type,
                'confidence': table.confidence,
                'row_count': table.row_count,
                'columns': [col['name'] for col in table.columns[:10]],
                'sample_preview': sample_preview,
                'has_data': len(table.sample_data) > 0,
                'business_score': business_indicators['score'],
                'business_reason': business_indicators['reason'],
                'used_in_views': view_usage['count'],
                'view_examples': view_usage['examples']
            })
        
        # Sort by business relevance and view usage
        table_summaries.sort(key=lambda t: (t['business_score'], t['used_in_views'], t['confidence']), reverse=True)
        
        # Create system prompt with view pattern examples
        system_prompt = self._create_enhanced_system_prompt(relevant_patterns)
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

BUSINESS INTENT: {intent}

PROVEN VIEW PATTERNS FOR SIMILAR QUESTIONS:
{json.dumps(relevant_patterns, indent=2)}

AVAILABLE TABLES (sorted by relevance):
{json.dumps(table_summaries[:20], indent=2)}

Based on the PROVEN VIEW PATTERNS above, select 4-8 tables that contain the actual data needed.
The view patterns show you exactly how similar business questions have been answered successfully.

Look at:
1. Which tables are used in successful view patterns
2. What join patterns work for this type of question  
3. Sample data that matches the business intent
4. Tables with proven relationships from view examples

JSON format:
{{
  "selected_tables": ["[schema].[table1]", "[schema].[table2]"],
  "reasoning": "Based on view pattern examples, these tables contain the data needed",
  "view_pattern_guidance": "How the view patterns influenced this selection"
}}
"""
        
        response = await self.llm.generate_sql(system_prompt, user_prompt)
        result = self._parse_json(response)
        
        if result and 'selected_tables' in result:
            selected_names = result['selected_tables']
            selected_tables = [t for t in self.tables if t.full_name in selected_names]
            
            print(f"      🎯 Selected based on intent: {intent}")
            if result.get('view_pattern_guidance'):
                print(f"      👁️ View guidance: {result['view_pattern_guidance'][:100]}...")
            
            return selected_tables
        
        # Enhanced fallback using view patterns
        return self._view_pattern_fallback(question, intent)
    
    def _find_relevant_view_patterns(self, intent: str, question: str) -> List[Dict]:
        """Find view patterns relevant to the question"""
        relevant = []
        q_lower = question.lower()
        intent_lower = intent.lower()
        
        for pattern in self.view_patterns:
            relevance_score = 0
            
            pattern_type = pattern.get('pattern_type', '').lower()
            use_case = pattern.get('use_case', '').lower()
            
            # Score by pattern type match
            if any(word in pattern_type for word in intent_lower.split()):
                relevance_score += 3
            
            # Score by use case match
            if any(word in use_case for word in q_lower.split() if len(word) > 3):
                relevance_score += 2
            
            # Score by business logic match
            business_logic = pattern.get('business_logic', {})
            if 'revenue' in q_lower and business_logic.get('has_aggregations'):
                relevance_score += 2
            if any(word in q_lower for word in ['quarter', 'month', 'year']) and business_logic.get('has_date_filtering'):
                relevance_score += 2
            
            if relevance_score > 0:
                pattern_with_score = pattern.copy()
                pattern_with_score['relevance_score'] = relevance_score
                relevant.append(pattern_with_score)
        
        # Sort by relevance and return top matches
        relevant.sort(key=lambda p: p['relevance_score'], reverse=True)
        return relevant[:3]  # Top 3 most relevant patterns
    
    def _check_view_usage(self, table_name: str, view_patterns: List[Dict]) -> Dict[str, Any]:
        """Check how often a table is used in view patterns"""
        count = 0
        examples = []
        
        for pattern in view_patterns:
            tables_involved = pattern.get('tables_involved', [])
            if any(table_name.lower() in table.lower() for table in tables_involved):
                count += 1
                examples.append({
                    'view_name': pattern.get('view_name', ''),
                    'use_case': pattern.get('use_case', '')[:50]
                })
        
        return {'count': count, 'examples': examples[:3]}
    
    def _create_enhanced_system_prompt(self, view_patterns: List[Dict]) -> str:
        """Create system prompt enhanced with view pattern examples"""
        
        base_prompt = """You are a business intelligence expert with access to PROVEN VIEW PATTERNS.

These view patterns show you EXACTLY how similar business questions have been answered successfully in this database.

CRITICAL GUIDANCE:
- Use the view patterns as your primary guide for table selection
- If a view pattern successfully answers similar questions, prefer those tables
- View patterns show proven join relationships and business logic
- Tables used in multiple view patterns are likely core business tables

"""
        
        if view_patterns:
            base_prompt += "PROVEN SUCCESSFUL PATTERNS:\n"
            for i, pattern in enumerate(view_patterns, 1):
                base_prompt += f"\n{i}. Pattern: {pattern.get('pattern_type', 'unknown')}"
                base_prompt += f"\n   Use Case: {pattern.get('use_case', '')[:100]}"
                base_prompt += f"\n   Tables Used: {', '.join(pattern.get('tables_involved', [])[:3])}"
                if pattern.get('join_patterns'):
                    base_prompt += f"\n   Join Logic: {pattern['join_patterns'][0][:100]}"
                base_prompt += "\n"
        
        return base_prompt
    
    def _view_pattern_fallback(self, question: str, intent: str) -> List[TableInfo]:
        """Enhanced fallback using view patterns"""
        # Get all tables mentioned in view patterns
        pattern_tables = set()
        for pattern in self.view_patterns:
            pattern_tables.update(pattern.get('tables_involved', []))
        
        # Score tables by view usage and business relevance
        scored_tables = []
        for table in self.tables:
            score = 0
            
            # Bonus for being used in view patterns
            if any(table.full_name.lower() in pt.lower() for pt in pattern_tables):
                score += 3
            
            # Business relevance score
            indicators = self._get_business_indicators(table, question)
            score += indicators['score']
            
            if score > 0:
                scored_tables.append((table, score))
        
        # Sort and return top tables
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        return [table for table, score in scored_tables[:6]]
    
    def _analyze_business_intent(self, question: str) -> str:
        """Analyze what the business question is asking for"""
        q = question.lower()
        
        intents = []
        
        # Revenue/Financial analysis
        if any(word in q for word in ['revenue', 'income', 'sales', 'profit', 'amount', 'value', 'financial']):
            intents.append("financial_analysis")
        
        # Time-based analysis
        if any(word in q for word in ['quarter', 'month', 'year', 'next', 'last', 'this', 'period']):
            intents.append("time_based_analysis")
        
        # Risk/Renewal analysis
        if any(word in q for word in ['risk', 'renewal', 'expire', 'cancel', 'churn', 'retention']):
            intents.append("risk_analysis")
        
        # Customer analysis
        if any(word in q for word in ['customer', 'client', 'account', 'subscriber']):
            intents.append("customer_analysis")
        
        # Contract/Subscription analysis
        if any(word in q for word in ['contract', 'subscription', 'agreement', 'deal']):
            intents.append("contract_analysis")
        
        return " + ".join(intents) if intents else "general_query"
    
    def _get_business_indicators(self, table: TableInfo, question: str) -> Dict[str, Any]:
        """Score table relevance for business question"""
        score = 0
        reasons = []
        
        table_name = table.full_name.lower()
        column_names = [col['name'].lower() for col in table.columns]
        q = question.lower()
        
        # Revenue/Financial indicators
        if any(word in q for word in ['revenue', 'amount', 'financial', 'money']):
            if any(col in column_names for col in ['amount', 'price', 'total', 'revenue', 'value']):
                score += 3
                reasons.append("has_financial_columns")
            if any(word in table_name for word in ['payment', 'invoice', 'billing', 'financial']):
                score += 2
                reasons.append("financial_table")
        
        # Renewal/Contract indicators  
        if any(word in q for word in ['renewal', 'contract', 'subscription', 'expire']):
            if any(col in column_names for col in ['renewaldate', 'expirationdate', 'contractdate', 'subscriptiondate']):
                score += 3
                reasons.append("has_renewal_dates")
            if any(word in table_name for word in ['contract', 'subscription', 'renewal']):
                score += 2
                reasons.append("contract_table")
        
        # Time-based indicators
        if any(word in q for word in ['quarter', 'next', 'period', 'time']):
            date_columns = [col for col in column_names if 'date' in col or 'time' in col]
            if date_columns:
                score += 2
                reasons.append(f"has_date_columns: {date_columns[:3]}")
        
        # Risk indicators
        if any(word in q for word in ['risk', 'why', 'reason']):
            if any(col in column_names for col in ['status', 'active', 'cancelled', 'reason']):
                score += 2
                reasons.append("has_status_columns")
        
        # Data quality bonus
        if table.sample_data and len(table.sample_data) > 0:
            score += 1
            reasons.append("has_sample_data")
        
        return {
            'score': score,
            'reason': ', '.join(reasons) if reasons else 'no_specific_indicators'
        }
    
    def _create_sample_preview(self, sample_data: List[Dict]) -> str:
        """Create preview of sample data"""
        if not sample_data:
            return "No sample data"
        
        first_row = sample_data[0]
        preview_items = []
        
        for key, value in list(first_row.items())[:4]:
            if value is not None:
                value_str = str(value)[:20]
                preview_items.append(f"{key}={value_str}")
        
        return ", ".join(preview_items) if preview_items else "Empty data"
    
    def _parse_json(self, response: str) -> Dict:
        """Parse JSON from response"""
        try:
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = re.sub(r'^[^{]*', '', cleaned)
            cleaned = re.sub(r'[^}]*$', '', cleaned)
            return json.loads(cleaned)
        except:
            return {}


class RelationshipMapper:
    """Enhanced relationship mapper using view join patterns"""
    
    def __init__(self, database_structure: Dict, view_patterns: List[Dict]):
        self.database_structure = database_structure
        self.view_patterns = view_patterns
        self.foreign_keys = self._build_foreign_key_map()
        self.view_join_patterns = self._extract_view_join_patterns()
    
    def _build_foreign_key_map(self) -> Dict[str, List[Dict]]:
        """Build foreign key mapping from database structure"""
        fk_map = {}
        
        # From explicit relationships
        for rel in self.database_structure.get('relationships', []):
            from_table = rel.get('from_table', '')
            if from_table:
                fk_map.setdefault(from_table, []).append(rel)
        
        # From table-level relationships
        for table_data in self.database_structure.get('tables', []):
            table_name = table_data.get('full_name', '')
            for fk_info in table_data.get('relationships', []):
                if '->' in fk_info:
                    # Example: "CustomerId -> dbo.Customers.CustomerId"
                    try:
                        from_col, to_info = [x.strip() for x in fk_info.split('->', 1)]
                        m = re.match(r'(?i)(\w+)\.(\w+)\.(\w+)', to_info)  # schema.table.column
                        if m:
                            sch, tbl, col = m.groups()
                            fk_map.setdefault(table_name, []).append({
                                'from_table': table_name,
                                'from_column': from_col,
                                'to_table': f'[{sch}].[{tbl}]',
                                'to_column': col
                            })
                    except Exception:
                        continue
                    
                    return fk_map
    
    def _extract_view_join_patterns(self) -> Dict[str, List[Dict]]:
        """Extract join patterns from successful views"""
        join_patterns = {}
        
        for pattern in self.view_patterns:
            joins = pattern.get('join_patterns', [])
            tables = pattern.get('tables_involved', [])
            
            for join in joins:
                # Extract table pairs from join condition
                table_pairs = self._extract_table_pairs(join, tables)
                
                for pair in table_pairs:
                    key = f"{pair['table1']}:{pair['table2']}"
                    if key not in join_patterns:
                        join_patterns[key] = []
                    
                    join_patterns[key].append({
                        'join_condition': join,
                        'source_view': pattern.get('view_name', ''),
                        'business_pattern': pattern.get('pattern_type', ''),
                        'success_verified': True
                    })
        
        return join_patterns
    
    def _extract_table_pairs(self, join_condition: str, tables: List[str]) -> List[Dict]:
        """Extract table pairs from join condition"""
        pairs = []
        
        # Simple pattern matching for table references in join
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                table1_simple = table1.split('.')[-1] if '.' in table1 else table1
                table2_simple = table2.split('.')[-1] if '.' in table2 else table2
                
                # Check if both tables are referenced in the join condition
                if (table1_simple.lower() in join_condition.lower() and 
                    table2_simple.lower() in join_condition.lower()):
                    pairs.append({
                        'table1': table1,
                        'table2': table2,
                        'join_condition': join_condition
                    })
        
        return pairs
    
    def find_relationships(self, tables: List[TableInfo]) -> List[Dict]:
        """Find relationships using both foreign keys and view patterns"""
        table_names = [t.full_name for t in tables]
        relationships = []
        
        # First, use traditional foreign key relationships
        for table_name in table_names:
            for fk in self.foreign_keys.get(table_name, []):
                target_table = fk.get('to_table', '')
                
                if any(target_table in t.full_name for t in tables):
                    relationships.append({
                        'from_table': table_name,
                        'to_table': target_table,
                        'from_column': fk.get('from_column', ''),
                        'to_column': fk.get('to_column', ''),
                        'join_type': 'INNER JOIN',
                        'source': 'foreign_key',
                        'confidence': 0.9
                    })
        
        # Enhanced: Use proven view join patterns
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                join_key1 = f"{table1.full_name}:{table2.full_name}"
                join_key2 = f"{table2.full_name}:{table1.full_name}"
                
                view_pattern = (self.view_join_patterns.get(join_key1) or 
                               self.view_join_patterns.get(join_key2))
                
                if view_pattern:
                    # Use the proven join pattern from views
                    pattern = view_pattern[0]  # Use the first (most common) pattern
                    
                    relationships.append({
                        'from_table': table1.full_name,
                        'to_table': table2.full_name,
                        'join_condition': pattern['join_condition'],
                        'join_type': 'INNER JOIN',  # Can be enhanced to detect LEFT/RIGHT
                        'source': 'view_pattern',
                        'source_view': pattern['source_view'],
                        'business_pattern': pattern['business_pattern'],
                        'confidence': 0.95  # High confidence as it's proven in views
                    })
        
        print(f"      🔗 Found {len(relationships)} relationships ({sum(1 for r in relationships if r['source'] == 'view_pattern')} from view patterns)")
        return relationships


class SQLGenerator:
    """Enhanced SQL generator using view patterns as templates"""
    
    def __init__(self, llm: LLMClient, view_patterns: List[Dict]):
        self.llm = llm
        self.view_patterns = view_patterns
    
    async def generate_query(self, question: str, tables: List[TableInfo], 
                           relationships: List[Dict]) -> str:
        """Generate SQL using view patterns as examples"""
        
        # Find the most relevant view patterns as templates
        relevant_patterns = self._find_template_patterns(question, tables, relationships)
        
        # Analyze business context
        business_context = self._analyze_sql_context(question, tables)
        
        # Prepare enhanced table context
        table_context = []
        for table in tables:
            columns_info = []
            important_columns = {
                'date_columns': [],
                'amount_columns': [],
                'status_columns': [],
                'id_columns': []
            }
            
            for col in table.columns[:15]:
                col_name = col['name']
                col_type = col['data_type']
                columns_info.append(f"{col_name} ({col_type})")
                
                # Categorize columns
                col_lower = col_name.lower()
                if any(word in col_lower for word in ['date', 'time']):
                    important_columns['date_columns'].append(col_name)
                if any(word in col_lower for word in ['amount', 'price', 'total', 'revenue', 'value']):
                    important_columns['amount_columns'].append(col_name)
                if any(word in col_lower for word in ['status', 'active', 'cancelled', 'renewed']):
                    important_columns['status_columns'].append(col_name)
                if col_lower.endswith('id') or col_lower.endswith('_id'):
                    important_columns['id_columns'].append(col_name)
            
            # Sample data analysis
            sample_analysis = self._analyze_sample_data(table.sample_data)
            
            table_context.append({
                'table_name': table.full_name,
                'entity_type': table.entity_type,
                'columns': columns_info,
                'important_columns': important_columns,
                'sample_analysis': sample_analysis,
                'row_count': table.row_count
            })
        
        # Create enhanced system prompt with view pattern templates
        system_prompt = self._create_enhanced_system_prompt(business_context, relevant_patterns)
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

PROVEN QUERY TEMPLATES FOR SIMILAR QUESTIONS:
{self._format_view_templates(relevant_patterns)}

AVAILABLE TABLES WITH FULL CONTEXT:
{json.dumps(table_context, indent=2)}

VERIFIED RELATIONSHIPS AND JOIN PATTERNS:
{json.dumps(relationships, indent=2)}

BUSINESS REQUIREMENTS:
{business_context['requirements']}

Generate a SQL query that follows the PROVEN TEMPLATES above while answering the specific business question.
Use the exact join patterns and business logic shown in the successful view examples.
"""
        
        response = await self.llm.generate_sql(system_prompt, user_prompt)
        return self._clean_sql(response)
    
    def _find_template_patterns(self, question: str, tables: List[TableInfo], 
                               relationships: List[Dict]) -> List[Dict]:
        """Find view patterns that can serve as templates"""
        templates = []
        q_lower = question.lower()
        table_names = [t.full_name for t in tables]
        
        for pattern in self.view_patterns:
            relevance_score = 0
            
            # Score by table overlap
            pattern_tables = pattern.get('tables_involved', [])
            table_overlap = sum(1 for pt in pattern_tables 
                              if any(pt.lower() in tn.lower() for tn in table_names))
            relevance_score += table_overlap * 2
            
            # Score by business pattern match
            pattern_type = pattern.get('pattern_type', '').lower()
            if any(word in q_lower for word in pattern_type.split('_')):
                relevance_score += 3
            
            # Score by use case similarity
            use_case = pattern.get('use_case', '').lower()
            common_words = set(q_lower.split()) & set(use_case.split())
            relevance_score += len(common_words)
            
            # Score by business logic match
            business_logic = pattern.get('business_logic', {})
            if 'revenue' in q_lower and business_logic.get('has_aggregations'):
                relevance_score += 2
            if any(word in q_lower for word in ['quarter', 'month', 'year']) and business_logic.get('has_date_filtering'):
                relevance_score += 2
            
            if relevance_score > 0:
                template = pattern.copy()
                template['template_score'] = relevance_score
                templates.append(template)
        
        # Sort by relevance and return top templates
        templates.sort(key=lambda t: t['template_score'], reverse=True)
        return templates[:3]  # Top 3 most relevant templates
    
    def _format_view_templates(self, templates: List[Dict]) -> str:
        """Format view templates for the LLM prompt"""
        if not templates:
            return "No similar view patterns found."
        
        formatted = "SUCCESSFUL QUERY PATTERNS FROM YOUR DATABASE:\n\n"
        
        for i, template in enumerate(templates, 1):
            formatted += f"TEMPLATE {i}: {template.get('pattern_type', 'Unknown')}\n"
            formatted += f"Use Case: {template.get('use_case', '')}\n"
            formatted += f"Tables: {', '.join(template.get('tables_involved', []))}\n"
            
            if template.get('join_patterns'):
                formatted += f"Join Pattern: {template['join_patterns'][0]}\n"
            
            # Show relevant parts of the view definition
            definition = template.get('definition', '')
            if definition:
                # Extract key parts (SELECT clause, main JOINs, WHERE clause)
                key_parts = self._extract_key_sql_parts(definition)
                if key_parts:
                    formatted += f"Key SQL Logic:\n{key_parts}\n"
            
            if template.get('sample_data'):
                formatted += f"Expected Output: {json.dumps(template['sample_data'][0], default=str)}\n"
            
            formatted += "\n" + "-"*60 + "\n\n"
        
        return formatted
    
    def _extract_key_sql_parts(self, definition: str) -> str:
        """Extract key parts of SQL definition for templates"""
        if not definition:
            return ""
        
        lines = definition.split('\n')
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_upper = line.upper()
            
            # Include important SQL clauses
            if any(keyword in line_upper for keyword in [
                'SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY',
                'SUM(', 'COUNT(', 'AVG(', 'CASE WHEN', 'DATEADD', 'GETDATE'
            ]):
                key_lines.append(line)
                
                # Limit to prevent too long examples
                if len(key_lines) >= 10:
                    break
        
        return '\n'.join(key_lines)
    
    def _create_enhanced_system_prompt(self, business_context: Dict, templates: List[Dict]) -> str:
        """Create system prompt enhanced with business context and templates"""
        
        base_prompt = f"""
You are a business intelligence SQL expert specializing in {business_context['domain']}.

You have access to PROVEN QUERY TEMPLATES from successful views in this exact database.
These templates show you EXACTLY how similar business questions have been answered successfully.

BUSINESS CONTEXT: {business_context['intent']}

CRITICAL RULES:
1. FOLLOW THE PROVEN TEMPLATES - they contain the exact business logic needed
2. Use the exact JOIN patterns shown in the templates - they are verified to work
3. Copy the aggregation and calculation patterns from templates
4. Use similar WHERE clause logic for date filtering and business rules
5. Return TOP 100 to limit results
6. Use business-friendly column aliases
7. NO variables (@var), use inline expressions only

TEMPLATE GUIDANCE:
"""
        
        if templates:
            base_prompt += f"You have {len(templates)} proven templates that solve similar problems. "
            base_prompt += "Study their SQL patterns, join logic, and business calculations. "
            base_prompt += "Adapt their proven approaches to answer the current question.\n\n"
        else:
            base_prompt += "No exact templates found, but use standard business intelligence patterns.\n\n"
        
        base_prompt += """
SQL GENERATION STRATEGY:
1. Start with the SELECT clause using template patterns
2. Use proven FROM and JOIN patterns from templates
3. Apply similar WHERE conditions for filtering
4. Use template aggregation patterns for calculations
5. Add appropriate GROUP BY and ORDER BY clauses

Return ONLY the SQL query that directly answers the business question.
"""
        
        return base_prompt
    
    def _analyze_sql_context(self, question: str, tables: List[TableInfo]) -> Dict[str, str]:
        """Analyze business context for SQL generation"""
        q = question.lower()
        
        # Determine domain
        entity_types = [t.entity_type for t in tables]
        if 'Financial' in entity_types or 'Payment' in entity_types:
            domain = "Financial/Revenue Analysis"
        elif 'Customer' in entity_types:
            domain = "Customer Relationship Management"
        else:
            domain = "Business Operations"
        
        # Determine intent and requirements
        if 'revenue' in q and 'risk' in q:
            intent = "Revenue Risk Analysis"
            requirements = [
                "Calculate total revenue amounts at risk",
                "Identify time period (next quarter)",
                "Include risk reasons/explanations",
                "Group by relevant business dimensions"
            ]
        elif 'renewal' in q:
            intent = "Renewal Analysis"
            requirements = [
                "Focus on renewal/expiration dates",
                "Include contract/subscription values",
                "Calculate time-based filtering",
                "Identify renewal status and risks"
            ]
        elif any(word in q for word in ['revenue', 'sales', 'financial']):
            intent = "Financial Analysis"
            requirements = [
                "Sum/aggregate financial amounts",
                "Apply date filtering for time periods",
                "Include business-friendly groupings"
            ]
        else:
            intent = "General Business Query"
            requirements = [
                "Focus on business metrics",
                "Include relevant filters and groupings"
            ]
        
        return {
            'domain': domain,
            'intent': intent,
            'requirements': '; '.join(requirements)
        }
    
    def _analyze_sample_data(self, sample_data: List[Dict]) -> str:
        """Analyze sample data for SQL context"""
        if not sample_data:
            return "No sample data available"
        
        analysis = []
        first_row = sample_data[0]
        
        # Look for key business indicators
        for key, value in first_row.items():
            if value is not None:
                key_lower = key.lower()
                
                # Financial indicators
                if any(word in key_lower for word in ['amount', 'price', 'total', 'revenue']):
                    analysis.append(f"Financial: {key}={value}")
                
                # Date indicators  
                elif any(word in key_lower for word in ['date', 'time']):
                    analysis.append(f"Date: {key}={value}")
                
                # Status indicators
                elif any(word in key_lower for word in ['status', 'active', 'cancelled']):
                    analysis.append(f"Status: {key}={value}")
        
        return "; ".join(analysis[:5]) if analysis else "General data fields"
    
    def _clean_sql(self, response: str) -> str:
        """Clean SQL from response"""
        # Remove markdown
        cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Extract SQL
        lines = cleaned.strip().split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'WITH')) or sql_lines:
                sql_lines.append(line)
        
        return '\n'.join(sql_lines).rstrip(';').strip()


class QueryExecutor:
    """Execute SQL queries safely"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def execute_query(self, sql: str) -> tuple:
        """Execute SQL query and return results"""
        if not sql or not sql.strip():
            return [], "Empty SQL query"
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                # Set UTF-8 encoding for Greek text
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                cursor.execute(sql)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor.fetchmany(100):  # Limit to 100 rows
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = self._safe_value(value)
                        results.append(row_dict)
                    
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            if "Invalid column name" in error_msg:
                return [], f"Column not found: {error_msg}"
            elif "Invalid object name" in error_msg:
                return [], f"Table not found: {error_msg}"
            else:
                return [], f"SQL Error: {error_msg}"
    
    def _safe_value(self, value):
        """Convert value to safe format"""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)[:200]


class QueryInterface:
    """Enhanced query interface using view patterns as examples"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.data_loader = DataLoader(config)
        self.executor = QueryExecutor(config)
        
        # Load cached data including view patterns
        if not self.data_loader.load_data():
            raise ValueError("Failed to load cached data. Run discovery and analysis first.")
        
        # Initialize components with view patterns
        self.table_selector = TableSelector(
            self.data_loader.tables, 
            self.llm, 
            self.data_loader.view_patterns
        )
        self.relationship_mapper = RelationshipMapper(
            self.data_loader.database_structure, 
            self.data_loader.view_patterns
        )
        self.sql_generator = SQLGenerator(
            self.llm, 
            self.data_loader.view_patterns
        )
        
        print(f"   📊 Loaded {len(self.data_loader.tables)} tables")
        print(f"   👁️ View patterns: {len(self.data_loader.view_patterns)}")
        print(f"   📋 Business templates: {len(self.data_loader.business_templates)}")
        print(f"   🔗 Database relationships: {len(self.relationship_mapper.foreign_keys)}")
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship]):
        """Start enhanced interactive query session"""
        
        print(f"🚀 Enhanced 4-Stage Pipeline with View Pattern Intelligence")
        print(f"   📊 Classified tables: {len(tables)}")
        print(f"   👁️ View pattern examples: {len(self.data_loader.view_patterns)}")
        
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        # Show entity distribution
        entity_counts = {}
        for table in tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   📊 Available entities: {dict(list(entity_counts.items())[:5])}")
        
        # Show business patterns available
        if self.data_loader.business_templates:
            print(f"   📋 Business patterns: {list(self.data_loader.business_templates.keys())[:5]}")
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🚀 Processing with view pattern-enhanced 4-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_query(question)
                result.execution_time = time.time() - start_time
                
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Session summary: {query_count} queries processed")
    
    async def process_query(self, question: str) -> QueryResult:
        """Process query through enhanced 4-stage pipeline with view pattern intelligence"""
        
        try:
            # Stage 1: Enhanced intent understanding with view pattern context
            print("   🎯 Stage 1: Understanding intent with view pattern context...")
            
            # Stage 2: Find relevant tables using view pattern examples
            print("   📋 Stage 2: Finding tables using view pattern examples...")
            selected_tables = await self.table_selector.find_relevant_tables(question)
            
            if not selected_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found"
                )
            
            print(f"      ✅ Found {len(selected_tables)} relevant tables")
            
            # Stage 3: Enhanced relationship discovery using view join patterns
            print("   🔗 Stage 3: Analyzing relationships with view join patterns...")
            relationships = self.relationship_mapper.find_relationships(selected_tables)
            view_relationships = sum(1 for r in relationships if r.get('source') == 'view_pattern')
            print(f"      🔗 Found {len(relationships)} relationships ({view_relationships} from proven view patterns)")
            
            # Stage 4: Enhanced SQL generation using view templates
            print("   ⚡ Stage 4: Generating SQL using proven view templates...")
            
            sql = await self.sql_generator.generate_query(question, selected_tables, relationships)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to generate SQL query"
                )
            
            # Enhanced validation using view pattern knowledge
            validation_result = self._validate_business_logic_with_patterns(question, sql, selected_tables)
            if validation_result['issues']:
                print(f"      ⚠️ Pattern-based validation issues: {', '.join(validation_result['issues'])}")
                print("      🔄 Regenerating with view pattern guidance...")
                
                # Regenerate with pattern-based feedback
                sql = await self._regenerate_with_pattern_context(
                    question, selected_tables, relationships, validation_result['issues']
                )
            
            # Execute query
            results, error = self.executor.execute_query(sql)
            
            # Enhanced result validation with pattern expectations
            if not error:
                result_analysis = await self._analyze_result_quality_with_patterns(
                    question, sql, results, selected_tables
                )
                
                if result_analysis['needs_retry']:
                    print(f"      🔍 Pattern-based analysis: {result_analysis['issue']}")
                    print("      🔄 Auto-retrying with view pattern improvements...")
                    
                    # Retry with pattern-based improvements
                    improved_sql = await self._retry_with_pattern_analysis(
                        question, selected_tables, relationships, sql, results, result_analysis
                    )
                    
                    if improved_sql and improved_sql != sql:
                        print("      ✅ Generated improved query using view patterns")
                        retry_results, retry_error = self.executor.execute_query(improved_sql)
                        
                        if not retry_error and len(retry_results) > 0:
                            print(f"      🎯 Pattern-guided retry successful: {len(retry_results)} rows vs {len(results)} rows")
                            sql = improved_sql
                            results = retry_results
                            error = retry_error
                        else:
                            print(f"      ⚠️ Pattern retry didn't improve results, using original")
            
            return QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=[t.full_name for t in selected_tables]
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Enhanced pipeline failed: {str(e)}"
            )
    
    def _validate_business_logic_with_patterns(self, question: str, sql: str, 
                                             tables: List[TableInfo]) -> Dict[str, Any]:
        """Enhanced validation using view pattern knowledge"""
        q = question.lower()
        sql_lower = sql.lower()
        issues = []
        
        # Find relevant patterns for validation
        relevant_patterns = []
        for pattern in self.data_loader.view_patterns:
            pattern_type = pattern.get('pattern_type', '').lower()
            if any(word in q for word in pattern_type.split('_')):
                relevant_patterns.append(pattern)
        
        # Basic business logic validation
        if 'revenue' in q and not any(func in sql_lower for func in ['sum(', 'sum ', 'total']):
            issues.append("Missing revenue calculation (SUM)")
        
        if any(period in q for period in ['quarter', 'month', 'year', 'next', 'last']):
            if not any(func in sql_lower for func in ['dateadd', 'getdate', 'where']):
                issues.append("Missing time period filtering")
        
        # Pattern-based validation
        for pattern in relevant_patterns:
            business_logic = pattern.get('business_logic', {})
            
            # If pattern uses aggregations, query should too
            if business_logic.get('has_aggregations') and 'group by' not in sql_lower:
                issues.append(f"Pattern '{pattern.get('pattern_type')}' typically uses GROUP BY aggregations")
            
            # If pattern uses date filtering, query should too
            if business_logic.get('has_date_filtering') and not any(func in sql_lower for func in ['dateadd', 'getdate']):
                issues.append(f"Pattern '{pattern.get('pattern_type')}' typically includes date filtering")
        
        return {
            'issues': issues,
            'has_issues': len(issues) > 0,
            'pattern_guidance': relevant_patterns[:2]  # Top 2 relevant patterns
        }
    
    async def _regenerate_with_pattern_context(self, question: str, tables: List[TableInfo], 
                                             relationships: List[Dict], issues: List[str]) -> str:
        """Regenerate SQL with pattern-based feedback"""
        
        # Find the most relevant patterns as correction examples
        relevant_patterns = []
        q_lower = question.lower()
        
        for pattern in self.data_loader.view_patterns:
            pattern_type = pattern.get('pattern_type', '').lower()
            use_case = pattern.get('use_case', '').lower()
            
            if (any(word in q_lower for word in pattern_type.split('_')) or
                any(word in use_case for word in q_lower.split() if len(word) > 3)):
                relevant_patterns.append(pattern)
        
        # Sort by relevance and take top patterns
        relevant_patterns = relevant_patterns[:2]
        
        pattern_corrections = []
        for pattern in relevant_patterns:
            business_logic = pattern.get('business_logic', {})
            
            if business_logic.get('has_aggregations'):
                pattern_corrections.append("Use SUM(), COUNT(), or GROUP BY aggregations like the proven patterns")
            
            if business_logic.get('has_date_filtering'):
                pattern_corrections.append("Include DATEADD() or date filtering like successful view patterns")
            
            if pattern.get('join_patterns'):
                pattern_corrections.append(f"Use proven join pattern: {pattern['join_patterns'][0][:100]}")
        
        system_prompt = f"""
You are fixing a SQL query based on PROVEN VIEW PATTERNS from this database.

VALIDATION ISSUES: {', '.join(issues)}

PATTERN-BASED CORRECTIONS:
{chr(10).join([f"- {correction}" for correction in pattern_corrections])}

PROVEN SUCCESSFUL PATTERNS:
{self._format_correction_patterns(relevant_patterns)}

Generate corrected SQL that follows the proven patterns and fixes all validation issues.
Return ONLY the SQL query.
"""
        
        table_context = [
            {
                'table': t.full_name,
                'entity': t.entity_type,
                'important_columns': [col['name'] for col in t.columns 
                                    if any(word in col['name'].lower() 
                                          for word in ['amount', 'date', 'status', 'id'])],
                'rows': t.row_count
            }
            for t in tables
        ]
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

VALIDATION ISSUES TO FIX: {', '.join(issues)}

AVAILABLE TABLES:
{json.dumps(table_context, indent=2)}

RELATIONSHIPS:
{json.dumps(relationships, indent=2)}

Generate corrected SQL that follows the proven pattern examples and fixes all issues.
"""
        
        response = await self.llm.generate_sql(system_prompt, user_prompt)
        return self.sql_generator._clean_sql(response)
    
    def _format_correction_patterns(self, patterns: List[Dict]) -> str:
        """Format patterns for correction guidance"""
        if not patterns:
            return "No specific patterns found for guidance."
        
        formatted = ""
        for i, pattern in enumerate(patterns, 1):
            formatted += f"\nPATTERN {i}: {pattern.get('pattern_type', 'Unknown')}\n"
            formatted += f"Use Case: {pattern.get('use_case', '')}\n"
            
            if pattern.get('join_patterns'):
                formatted += f"Join Logic: {pattern['join_patterns'][0]}\n"
            
            business_logic = pattern.get('business_logic', {})
            if business_logic.get('has_aggregations'):
                formatted += "Uses: GROUP BY aggregations\n"
            if business_logic.get('has_date_filtering'):
                formatted += "Uses: Date filtering with DATEADD/GETDATE\n"
            
            formatted += "-" * 50 + "\n"
        
        return formatted
    
    async def _analyze_result_quality_with_patterns(self, question: str, sql: str, 
                                                   results: List[Dict], tables: List[TableInfo]) -> Dict[str, Any]:
        """Enhanced result analysis using pattern expectations"""
        
        q = question.lower()
        issues = []
        needs_retry = False
        
        # Basic result quality checks
        if len(results) == 0:
            if any(word in q for word in ['revenue', 'customer', 'payment', 'order', 'contract']):
                issues.append("Empty results for business question")
                needs_retry = True
        
        # Pattern-based expectations
        relevant_patterns = [p for p in self.data_loader.view_patterns 
                           if any(word in q for word in p.get('pattern_type', '').lower().split('_'))]
        
        for pattern in relevant_patterns:
            pattern_sample = pattern.get('sample_data', [])
            if pattern_sample and len(results) > 0:
                # Compare result structure with pattern expectations
                pattern_columns = set(pattern_sample[0].keys())
                result_columns = set(results[0].keys())
                
                # Check if we're missing important columns that the pattern typically returns
                if 'amount' in str(pattern_columns).lower() and 'amount' not in str(result_columns).lower():
                    if 'revenue' in q or 'total' in q:
                        issues.append("Missing amount/revenue column - pattern suggests this should be included")
                        needs_retry = True
        
        return {
            'needs_retry': needs_retry,
            'issue': '; '.join(issues) if issues else 'Results meet pattern expectations',
            'result_count': len(results),
            'has_data': len(results) > 0,
            'pattern_guidance': relevant_patterns[:1]
        }
    
    async def _retry_with_pattern_analysis(self, question: str, tables: List[TableInfo], 
                                         relationships: List[Dict], original_sql: str, 
                                         original_results: List[Dict], analysis: Dict) -> str:
        """Generate improved SQL using pattern analysis"""
        
        # Get pattern guidance from analysis
        pattern_guidance = analysis.get('pattern_guidance', [])
        
        corrections = []
        if len(original_results) == 0:
            corrections.append("Switch to LEFT JOINs to preserve base table records")
            corrections.append("Broaden date filters if they're too restrictive")
            
            # Add pattern-specific corrections
            for pattern in pattern_guidance:
                join_patterns = pattern.get('join_patterns', [])
                if join_patterns:
                    corrections.append(f"Try proven join pattern: {join_patterns[0]}")
        
        system_prompt = f"""
You are improving a SQL query using PROVEN VIEW PATTERN GUIDANCE.

ORIGINAL ISSUE: {analysis['issue']}

PATTERN-GUIDED CORRECTIONS:
{chr(10).join([f"- {correction}" for correction in corrections])}

SUCCESSFUL PATTERN EXAMPLES:
{self._format_retry_patterns(pattern_guidance)}

Generate improved SQL that addresses the issues using proven pattern approaches.
Return ONLY the SQL query.
"""
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

ORIGINAL SQL THAT HAD ISSUES:
{original_sql}

ANALYSIS: {analysis['issue']}

AVAILABLE TABLES:
{json.dumps([{'table': t.full_name, 'entity': t.entity_type} for t in tables], indent=2)}

RELATIONSHIPS:
{json.dumps(relationships, indent=2)}

Generate improved SQL using the pattern guidance to fix the issues.
"""
        
        response = await self.llm.generate_sql(system_prompt, user_prompt)
        return self.sql_generator._clean_sql(response)
    
    def _format_retry_patterns(self, patterns: List[Dict]) -> str:
        """Format patterns for retry guidance"""
        if not patterns:
            return "No pattern guidance available."
        
        formatted = ""
        for pattern in patterns:
            formatted += f"Pattern: {pattern.get('pattern_type', 'Unknown')}\n"
            
            if pattern.get('definition'):
                # Show key parts of successful definition
                key_parts = self.sql_generator._extract_key_sql_parts(pattern['definition'])
                if key_parts:
                    formatted += f"Successful SQL Logic:\n{key_parts}\n"
            
            if pattern.get('sample_data'):
                formatted += f"Expected Output Structure: {list(pattern['sample_data'][0].keys())}\n"
        
        return formatted
    
    def display_result(self, result: QueryResult):
        """Enhanced result display with pattern context"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.sql_query:
                print(f"📋 Generated SQL:\n   {result.sql_query}")
        else:
            print(f"📋 Generated SQL:\n   {result.sql_query}")
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                # Handle single value results
                if result.is_single_value():
                    value = result.get_single_value()
                    column_name = list(result.results[0].keys())[0]
                    
                    if isinstance(value, (int, float)):
                        print(f"   🎯 {column_name}: {value:,}")
                    else:
                        print(f"   🎯 {column_name}: {value}")
                else:
                    # Handle multiple rows
                    for i, row in enumerate(result.results[:5], 1):
                        display_row = {}
                        for key, value in list(row.items())[:6]:
                            if isinstance(value, str) and len(value) > 30:
                                display_row[key] = value[:30] + "..."
                            elif isinstance(value, (int, float)) and value > 1000:
                                display_row[key] = f"{value:,}"
                            else:
                                display_row[key] = value
                        print(f"   {i}. {display_row}")
                    
                    if len(result.results) > 5:
                        print(f"   ... and {len(result.results) - 5} more rows")
            else:
                print("   ⚠️ No results returned")
                print("   💡 This might indicate:")
                print("      • Date filters are too restrictive")
                print("      • Table joins don't match data")
                print("      • Selected tables don't contain relevant data")
                
                # Enhanced suggestions using view patterns
                print("   🔍 View pattern suggestions:")
                print("      • Check if similar view patterns use different table combinations")
                print("      • Try broader date ranges based on successful view examples")
                print("      • Consider using LEFT JOINs like proven view patterns")
            
            # Enhanced table usage display
            if result.tables_used:
                print(f"📋 Tables used:")
                for table_name in result.tables_used:
                    entity_type = "Unknown"
                    row_count = 0
                    pattern_usage = 0
                    
                    # Find table info
                    for table in self.data_loader.tables:
                        if table.full_name == table_name:
                            entity_type = table.entity_type
                            row_count = table.row_count
                            break
                    
                    # Check view pattern usage
                    for pattern in self.data_loader.view_patterns:
                        if any(table_name.lower() in t.lower() for t in pattern.get('tables_involved', [])):
                            pattern_usage += 1
                    
                    usage_info = f" [Used in {pattern_usage} view patterns]" if pattern_usage > 0 else ""
                    print(f"      • {table_name} ({entity_type}) - {row_count:,} rows{usage_info}")
            
            # Show relationship and pattern intelligence
            has_joins = 'JOIN' in (result.sql_query or '').upper()
            if has_joins:
                print("✅ Query used verified database relationships")
                # Check if view patterns were used
                view_pattern_relationships = sum(1 for r in self.relationship_mapper.view_join_patterns)
                if view_pattern_relationships > 0:
                    print("👁️ Enhanced with proven view join patterns")
            else:
                print("ℹ️ Single-table query (no joins needed)")
        
        # Enhanced debugging help with pattern context
        print("\n💡 Enhanced Debugging Help:")
        print("   • database_structure.json contains view patterns and proven query templates")
        print("   • semantic_analysis.json has entity classifications")
        print(f"   • {len(self.data_loader.view_patterns)} view patterns available as query examples")
        
        if not result.error and len(result.results) == 0:
            print("   🔍 For empty results, try:")
            print("      • Study similar view patterns in your database")
            print("      • Use broader date ranges based on view pattern examples")
            print("      • Check view pattern join logic for table relationship guidance")
            
        # Enhanced business question suggestions based on patterns
        if hasattr(result, 'tables_used') and result.tables_used:
            print("   💼 Try questions based on your view patterns:")
            
            # Find patterns that use similar tables
            similar_patterns = []
            for pattern in self.data_loader.view_patterns:
                pattern_tables = pattern.get('tables_involved', [])
                if any(table in result.tables_used[0] for table in pattern_tables):
                    similar_patterns.append(pattern)
            
            suggestions = []
            for pattern in similar_patterns[:3]:
                use_case = pattern.get('use_case', '')
                if use_case and len(use_case) < 100:
                    suggestions.append(f"'{use_case}'")
            
            if suggestions:
                print(f"      Based on view patterns: {', '.join(suggestions[:2])}")
            else:
                # Fallback to entity-based suggestions
                entity_types = set()
                for table_name in result.tables_used:
                    for table in self.data_loader.tables:
                        if table.full_name == table_name:
                            entity_types.add(table.entity_type)
                            break
                
                fallback_suggestions = []
                if 'Customer' in entity_types:
                    fallback_suggestions.append("'How many customers do we have?'")
                if 'Payment' in entity_types or 'Financial' in entity_types:
                    fallback_suggestions.append("'What is our total revenue this year?'")
                
                if fallback_suggestions:
                    print(f"      {', '.join(fallback_suggestions[:2])}")