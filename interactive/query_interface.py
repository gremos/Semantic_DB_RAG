#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Interactive Query Interface for Real Database Tables
Handles real business data with proper column analysis and relationship discovery
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
    """Enhanced LLM client for real database analysis"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            # temperature=0.1,
            request_timeout=90
        )
    
    async def analyze(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompt to LLM and get response"""
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
    """Load and analyze real cached database structure"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables = []
        self.relationships = []
        self.domain = None
        self.database_structure = {}
        self.view_patterns = []
        
    def load_data(self) -> bool:
        """Load all cached data with real database information"""
        try:
            # Load database structure with real table information
            db_file = self.config.get_cache_path("database_structure.json")
            if db_file.exists():
                with open(db_file, 'r', encoding='utf-8') as f:
                    self.database_structure = json.load(f)
                    self._extract_view_patterns()
            
            # Load semantic analysis with real business context
            semantic_file = self.config.get_cache_path("semantic_analysis.json")
            if semantic_file.exists():
                with open(semantic_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._load_tables(data)
                    self._load_relationships(data)
                    self._load_domain(data)
            
            print(f"   📊 Loaded {len(self.tables)} real tables, {len(self.relationships)} relationships")
            print(f"   👁️ Extracted {len(self.view_patterns)} view patterns from real database")
            return len(self.tables) > 0
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def _extract_view_patterns(self):
        """Extract real view patterns from database structure"""
        self.view_patterns = []
        view_info = self.database_structure.get('view_info', {})
        
        for view_name, view_data in view_info.items():
            if view_data.get('execution_success') and view_data.get('sample_data'):
                pattern = {
                    'view_name': view_name,
                    'definition': view_data.get('definition', ''),
                    'sample_data': view_data.get('sample_data', [])[:3],
                    'tables_involved': self._extract_tables_from_view(view_data.get('definition', '')),
                    'join_patterns': self._extract_join_patterns(view_data.get('definition', '')),
                    'business_context': self._analyze_business_context(view_name, view_data)
                }
                self.view_patterns.append(pattern)
    
    def _extract_tables_from_view(self, definition: str) -> List[str]:
        """Extract real table names from view definitions"""
        if not definition:
            return []
        
        tables = []
        # Enhanced patterns for real database table references
        patterns = [
            r'FROM\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)',
            r'JOIN\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)',
            r'INNER\s+JOIN\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)',
            r'LEFT\s+JOIN\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, definition, re.IGNORECASE)
            tables.extend(matches)
        
        return list(set(tables))
    
    def _extract_join_patterns(self, definition: str) -> List[str]:
        """Extract real join patterns from view definitions"""
        if not definition:
            return []
        
        joins = []
        on_pattern = r'ON\s+([^WHERE^GROUP^ORDER^INNER^LEFT^RIGHT^JOIN]+?)(?=\s+(?:WHERE|GROUP|ORDER|INNER|LEFT|RIGHT|JOIN|$))'
        matches = re.findall(on_pattern, definition, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clean_join = match.strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            # Clean up multiple spaces
            clean_join = re.sub(r'\s+', ' ', clean_join)
            if clean_join and len(clean_join) < 300:
                joins.append(clean_join)
        
        return joins
    
    def _analyze_business_context(self, view_name: str, view_data: Dict) -> Dict:
        """Analyze business context from real view data"""
        context = {
            'likely_business_function': 'unknown',
            'data_complexity': 'simple',
            'contains_aggregations': False,
            'contains_financial_data': False,
            'contains_customer_data': False
        }
        
        view_name_lower = view_name.lower()
        definition = view_data.get('definition', '').lower()
        
        # Analyze business function from real view names
        if any(word in view_name_lower for word in ['customer', 'client', 'account']):
            context['likely_business_function'] = 'customer_management'
            context['contains_customer_data'] = True
        elif any(word in view_name_lower for word in ['payment', 'invoice', 'billing', 'financial']):
            context['likely_business_function'] = 'financial_management'
            context['contains_financial_data'] = True
        elif any(word in view_name_lower for word in ['order', 'purchase', 'sales']):
            context['likely_business_function'] = 'order_management'
        elif any(word in view_name_lower for word in ['product', 'inventory', 'catalog']):
            context['likely_business_function'] = 'product_management'
        
        # Analyze complexity from definition
        if any(func in definition for func in ['sum(', 'count(', 'avg(', 'max(', 'min(']):
            context['contains_aggregations'] = True
            context['data_complexity'] = 'complex'
        
        if definition.count('join') > 2:
            context['data_complexity'] = 'complex'
        
        return context
    
    def _load_tables(self, data: Dict):
        """Load real table information from semantic analysis"""
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
            table.business_role = table_data.get('business_role', 'Unknown')
            self.tables.append(table)
    
    def _load_relationships(self, data: Dict):
        """Load real relationships from semantic analysis"""
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
        """Load real business domain information"""
        domain_data = data.get('domain')
        if domain_data:
            self.domain = BusinessDomain(
                domain_type=domain_data['domain_type'],
                industry=domain_data['industry'],
                confidence=domain_data['confidence'],
                sample_questions=domain_data['sample_questions'],
                capabilities=domain_data['capabilities']
            )


class TableSelector:
    """Smart table selector for real database tables"""
    
    def __init__(self, tables: List[TableInfo], llm: LLMClient, view_patterns: List[Dict]):
        self.tables = tables
        self.llm = llm
        self.view_patterns = view_patterns
    
    async def find_relevant_tables(self, question: str) -> List[TableInfo]:
        """Find relevant real tables using business intelligence"""
        print("   📋 Analyzing real database tables for relevance...")
        
        # Stage 1: Business intent analysis
        business_intent = self._analyze_business_intent(question)
        print(f"      🎯 Business intent: {business_intent}")
        
        # Stage 2: Smart table filtering
        candidate_tables = self._smart_table_filtering(question, business_intent)
        print(f"      📊 Found {len(candidate_tables)} candidate tables")
        
        # Stage 3: LLM-powered selection for complex cases
        if len(candidate_tables) > 8:
            final_tables = await self._llm_table_selection(question, candidate_tables[:15], business_intent)
        else:
            final_tables = candidate_tables
        
        print(f"      ✅ Selected {len(final_tables)} tables: {[t.name for t in final_tables]}")
        return final_tables
    
    def _analyze_business_intent(self, question: str) -> Dict[str, Any]:
        """Analyze business intent from question"""
        q_lower = question.lower()
        
        intent = {
            'primary_entity': 'unknown',
            'operation_type': 'query',
            'time_dimension': False,
            'aggregation_needed': False,
            'financial_focus': False,
            'customer_focus': False
        }
        
        # Detect primary entity
        if any(word in q_lower for word in ['customer', 'client', 'account', 'user']):
            intent['primary_entity'] = 'customer'
            intent['customer_focus'] = True
        elif any(word in q_lower for word in ['payment', 'paid', 'billing', 'invoice', 'revenue', 'financial']):
            intent['primary_entity'] = 'financial'
            intent['financial_focus'] = True
        elif any(word in q_lower for word in ['order', 'purchase', 'sale', 'transaction']):
            intent['primary_entity'] = 'order'
        elif any(word in q_lower for word in ['product', 'item', 'inventory']):
            intent['primary_entity'] = 'product'
        
        # Detect operation type
        if any(word in q_lower for word in ['count', 'number', 'how many']):
            intent['operation_type'] = 'count'
            intent['aggregation_needed'] = True
        elif any(word in q_lower for word in ['total', 'sum', 'revenue', 'amount']):
            intent['operation_type'] = 'sum'
            intent['aggregation_needed'] = True
        elif any(word in q_lower for word in ['average', 'avg', 'mean']):
            intent['operation_type'] = 'average'
            intent['aggregation_needed'] = True
        
        # Detect time dimension
        if any(word in q_lower for word in ['2025', '2024', 'year', 'month', 'quarter', 'last', 'this', 'period']):
            intent['time_dimension'] = True
        
        return intent
    
    def _smart_table_filtering(self, question: str, business_intent: Dict) -> List[TableInfo]:
        """Smart filtering using real table characteristics"""
        q_lower = question.lower()
        scored_tables = []
        
        for table in self.tables:
            score = 0
            reasons = []
            
            # Score by entity type match
            entity_match_score = self._calculate_entity_match_score(table, business_intent)
            score += entity_match_score['score']
            if entity_match_score['score'] > 0:
                reasons.append(entity_match_score['reason'])
            
            # Score by table name relevance
            name_score = self._calculate_name_relevance_score(table, q_lower)
            score += name_score['score']
            if name_score['score'] > 0:
                reasons.append(name_score['reason'])
            
            # Score by column relevance
            column_score = self._calculate_column_relevance_score(table, q_lower, business_intent)
            score += column_score['score']
            if column_score['score'] > 0:
                reasons.append(column_score['reason'])
            
            # Score by sample data relevance
            data_score = self._calculate_sample_data_score(table, q_lower, business_intent)
            score += data_score['score']
            if data_score['score'] > 0:
                reasons.append(data_score['reason'])
            
            # Score by view pattern usage
            view_score = self._calculate_view_pattern_score(table, business_intent)
            score += view_score['score']
            if view_score['score'] > 0:
                reasons.append(view_score['reason'])
            
            # Bonus for tables with actual data
            if table.row_count > 0 and table.sample_data:
                score += 1
                reasons.append("has_real_data")
            
            if score > 0:
                scored_tables.append({
                    'table': table,
                    'score': score,
                    'reasons': reasons
                })
        
        # Sort by score and return top tables
        scored_tables.sort(key=lambda x: x['score'], reverse=True)
        return [item['table'] for item in scored_tables[:12]]
    
    def _calculate_entity_match_score(self, table: TableInfo, business_intent: Dict) -> Dict:
        """Calculate entity type matching score"""
        primary_entity = business_intent.get('primary_entity', '')
        
        if primary_entity == 'customer' and table.entity_type == 'Customer':
            return {'score': 5, 'reason': 'customer_entity_match'}
        elif primary_entity == 'financial' and table.entity_type == 'Financial':
            return {'score': 5, 'reason': 'financial_entity_match'}
        elif primary_entity == 'order' and table.entity_type in ['Order', 'Transaction']:
            return {'score': 5, 'reason': 'order_entity_match'}
        elif primary_entity == 'product' and table.entity_type == 'Product':
            return {'score': 5, 'reason': 'product_entity_match'}
        elif table.entity_type in ['Customer', 'Financial', 'Order', 'Product'] and primary_entity != 'unknown':
            return {'score': 2, 'reason': 'related_entity'}
        
        return {'score': 0, 'reason': ''}
    
    def _calculate_name_relevance_score(self, table: TableInfo, question: str) -> Dict:
        """Calculate table name relevance score"""
        table_name_lower = table.name.lower()
        full_name_lower = table.full_name.lower()
        
        question_words = [word for word in question.split() if len(word) > 3]
        
        score = 0
        matched_words = []
        
        for word in question_words:
            word_lower = word.lower()
            if word_lower in table_name_lower:
                score += 3
                matched_words.append(word)
            elif word_lower in full_name_lower:
                score += 2
                matched_words.append(word)
        
        if matched_words:
            return {'score': score, 'reason': f'name_match:{",".join(matched_words[:3])}'}
        
        return {'score': 0, 'reason': ''}
    
    def _calculate_column_relevance_score(self, table: TableInfo, question: str, business_intent: Dict) -> Dict:
        """Calculate column relevance score using real column names"""
        column_names = [col.get('name', '').lower() for col in table.columns]
        question_words = question.lower().split()
        
        score = 0
        matched_columns = []
        
        # Direct column name matches
        for col_name in column_names:
            for word in question_words:
                if len(word) > 3 and word in col_name:
                    score += 2
                    matched_columns.append(col_name)
                    break
        
        # Business intent specific columns
        if business_intent.get('financial_focus'):
            financial_columns = [col for col in column_names 
                               if any(fin_word in col for fin_word in ['amount', 'price', 'cost', 'total', 'revenue', 'payment'])]
            score += len(financial_columns) * 2
            matched_columns.extend(financial_columns[:3])
        
        if business_intent.get('customer_focus'):
            customer_columns = [col for col in column_names 
                              if any(cust_word in col for cust_word in ['customer', 'client', 'account', 'user'])]
            score += len(customer_columns) * 2
            matched_columns.extend(customer_columns[:3])
        
        if business_intent.get('time_dimension'):
            date_columns = [col for col in column_names 
                          if any(date_word in col for date_word in ['date', 'time', 'created', 'updated'])]
            score += len(date_columns) * 1
            matched_columns.extend(date_columns[:3])
        
        if matched_columns:
            return {'score': min(score, 10), 'reason': f'column_match:{",".join(matched_columns[:3])}'}
        
        return {'score': 0, 'reason': ''}
    
    def _calculate_sample_data_score(self, table: TableInfo, question: str, business_intent: Dict) -> Dict:
        """Calculate relevance score from real sample data"""
        if not table.sample_data:
            return {'score': 0, 'reason': ''}
        
        sample_text = json.dumps(table.sample_data).lower()
        question_words = [word for word in question.lower().split() if len(word) > 3]
        
        score = 0
        matched_values = []
        
        # Look for question words in sample data
        for word in question_words:
            if word in sample_text:
                score += 1
                matched_values.append(word)
        
        # Look for business context indicators in sample data
        if business_intent.get('financial_focus'):
            if any(indicator in sample_text for indicator in ['€', '$', '£', 'amount', 'total', 'price']):
                score += 2
                matched_values.append('financial_data')
        
        if business_intent.get('time_dimension'):
            if any(indicator in sample_text for indicator in ['2025', '2024', '2023', 'date']):
                score += 2
                matched_values.append('time_data')
        
        if matched_values:
            return {'score': min(score, 5), 'reason': f'sample_match:{",".join(matched_values[:3])}'}
        
        return {'score': 0, 'reason': ''}
    
    def _calculate_view_pattern_score(self, table: TableInfo, business_intent: Dict) -> Dict:
        """Calculate score based on real view pattern usage"""
        score = 0
        view_usage_count = 0
        
        table_name = table.full_name
        
        for pattern in self.view_patterns:
            tables_involved = pattern.get('tables_involved', [])
            if any(table_name.lower() in t.lower() or t.lower() in table_name.lower() for t in tables_involved):
                view_usage_count += 1
                
                # Bonus for business context match
                business_context = pattern.get('business_context', {})
                if business_intent.get('customer_focus') and business_context.get('contains_customer_data'):
                    score += 2
                elif business_intent.get('financial_focus') and business_context.get('contains_financial_data'):
                    score += 2
                else:
                    score += 1
        
        if view_usage_count > 0:
            return {'score': min(score, 5), 'reason': f'used_in_{view_usage_count}_views'}
        
        return {'score': 0, 'reason': ''}
    
    async def _llm_table_selection(self, question: str, candidate_tables: List[TableInfo], 
                                 business_intent: Dict) -> List[TableInfo]:
        """Use LLM for final table selection with real table analysis"""
        
        # Prepare enhanced table summaries with real data analysis
        table_summaries = []
        for table in candidate_tables:
            # Real column analysis
            important_columns = self._analyze_important_columns(table, business_intent)
            
            # Real sample data analysis
            sample_analysis = self._analyze_real_sample_data(table, business_intent)
            
            table_summaries.append({
                'table_name': table.full_name,
                'entity_type': table.entity_type,
                'confidence': table.confidence,
                'row_count': table.row_count,
                'important_columns': important_columns,
                'sample_analysis': sample_analysis,
                'business_relevance': self._assess_business_relevance(table, business_intent)
            })
        
        system_prompt = f"""You are analyzing a real {business_intent.get('primary_entity', 'business')} database for E-Commerce/Order Management.
Select the most relevant tables for answering this specific business question.

BUSINESS CONTEXT:
- Primary Entity: {business_intent.get('primary_entity', 'unknown')}
- Operation: {business_intent.get('operation_type', 'query')}
- Needs Aggregation: {business_intent.get('aggregation_needed', False)}
- Time Dimension: {business_intent.get('time_dimension', False)}
- Financial Focus: {business_intent.get('financial_focus', False)}

Focus on tables that:
1. Contain the actual data needed for the question
2. Have meaningful sample data showing real business information
3. Have columns that match the business intent
4. Are used in proven view patterns for similar queries

Return 4-8 tables maximum."""
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

CANDIDATE TABLES WITH REAL DATA ANALYSIS:
{json.dumps(table_summaries, indent=2)}

Select tables that contain the actual data needed to answer this question.
Consider the sample data analysis and business relevance scores.

JSON format:
{{
  "selected_tables": ["[schema].[table1]", "[schema].[table2]"],
  "reasoning": "Why these specific tables contain the needed data"
}}
"""
        
        response = await self.llm.analyze(system_prompt, user_prompt)
        result = self._parse_json(response)
        
        if result and 'selected_tables' in result:
            selected_names = result['selected_tables']
            selected_tables = [t for t in candidate_tables if t.full_name in selected_names]
            return selected_tables
        
        # Fallback to top scoring tables
        return candidate_tables[:6]
    
    def _analyze_important_columns(self, table: TableInfo, business_intent: Dict) -> Dict:
        """Analyze important columns based on real column names and business intent"""
        important_cols = {
            'id_columns': [],
            'date_columns': [],
            'amount_columns': [],
            'status_columns': [],
            'name_columns': [],
            'business_specific': []
        }
        
        for col in table.columns:
            col_name = col.get('name', '').lower()
            
            # ID columns
            if col_name.endswith('id') or col_name.endswith('_id') or col_name == 'id':
                important_cols['id_columns'].append(col['name'])
            
            # Date columns
            elif any(date_word in col_name for date_word in ['date', 'time', 'created', 'updated', 'modified']):
                important_cols['date_columns'].append(col['name'])
            
            # Amount/Financial columns
            elif any(fin_word in col_name for fin_word in ['amount', 'price', 'cost', 'total', 'revenue', 'value', 'sum']):
                important_cols['amount_columns'].append(col['name'])
            
            # Status columns
            elif any(status_word in col_name for status_word in ['status', 'state', 'active', 'enabled', 'cancelled']):
                important_cols['status_columns'].append(col['name'])
            
            # Name columns
            elif any(name_word in col_name for name_word in ['name', 'title', 'description', 'label']):
                important_cols['name_columns'].append(col['name'])
            
            # Business-specific columns based on intent
            elif business_intent.get('customer_focus') and any(cust_word in col_name for cust_word in ['customer', 'client', 'account', 'user']):
                important_cols['business_specific'].append(col['name'])
            elif business_intent.get('financial_focus') and any(fin_word in col_name for fin_word in ['payment', 'invoice', 'billing', 'transaction']):
                important_cols['business_specific'].append(col['name'])
        
        return important_cols
    
    def _analyze_real_sample_data(self, table: TableInfo, business_intent: Dict) -> Dict:
        """Analyze real sample data for business relevance"""
        analysis = {
            'has_meaningful_data': False,
            'data_patterns': [],
            'business_indicators': [],
            'data_quality': 'unknown'
        }
        
        if not table.sample_data:
            return analysis
        
        try:
            first_row = table.sample_data[0]
            analysis['has_meaningful_data'] = True
            
            # Analyze data patterns
            for key, value in first_row.items():
                if value is not None:
                    value_str = str(value).lower()
                    
                    # Financial data patterns
                    if any(char in value_str for char in ['€', '$', '£']) or (isinstance(value, (int, float)) and value > 0):
                        if any(fin_word in key.lower() for fin_word in ['amount', 'price', 'cost', 'total']):
                            analysis['data_patterns'].append('financial_amounts')
                    
                    # Date patterns
                    if isinstance(value, str) and ('2025' in value_str or '2024' in value_str):
                        analysis['data_patterns'].append('recent_dates')
                    
                    # ID patterns
                    if key.lower().endswith('id') and isinstance(value, (int, str)):
                        analysis['data_patterns'].append('valid_ids')
                    
                    # Customer data patterns
                    if business_intent.get('customer_focus') and any(cust_word in key.lower() for cust_word in ['customer', 'client', 'name']):
                        analysis['business_indicators'].append('customer_data')
            
            # Assess data quality
            non_null_values = sum(1 for v in first_row.values() if v is not None)
            total_values = len(first_row)
            
            if non_null_values / total_values > 0.8:
                analysis['data_quality'] = 'high'
            elif non_null_values / total_values > 0.5:
                analysis['data_quality'] = 'medium'
            else:
                analysis['data_quality'] = 'low'
                
        except Exception:
            analysis['data_quality'] = 'error'
        
        return analysis
    
    def _assess_business_relevance(self, table: TableInfo, business_intent: Dict) -> Dict:
        """Assess overall business relevance of the table"""
        relevance = {
            'score': 0,
            'primary_reason': 'unknown',
            'confidence': table.confidence
        }
        
        # Entity type relevance
        entity_mapping = {
            'customer': ['Customer'],
            'financial': ['Financial', 'Payment'],
            'order': ['Order', 'Transaction'],
            'product': ['Product']
        }
        
        primary_entity = business_intent.get('primary_entity', '')
        if primary_entity in entity_mapping:
            if table.entity_type in entity_mapping[primary_entity]:
                relevance['score'] += 5
                relevance['primary_reason'] = f'{primary_entity}_entity_match'
        
        # Data quality relevance
        if table.row_count > 1000 and table.sample_data:
            relevance['score'] += 2
        
        # Business role relevance
        if hasattr(table, 'business_role') and table.business_role == 'Core':
            relevance['score'] += 1
        
        return relevance
    
    def _parse_json(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = re.sub(r'^[^{]*', '', cleaned)
            cleaned = re.sub(r'[^}]*$', '', cleaned)
            return json.loads(cleaned)
        except:
            return {}


class RelationshipFinder:
    """Find relationships between real database tables"""
    
    def __init__(self, database_structure: Dict, relationships: List[Relationship], view_patterns: List[Dict]):
        self.database_structure = database_structure
        self.global_relationships = relationships
        self.view_patterns = view_patterns
        self.foreign_keys = self._build_foreign_key_map()
        self.view_join_patterns = self._extract_view_join_patterns()
    
    def _build_foreign_key_map(self) -> Dict[str, List[Dict]]:
        """Build foreign key mapping from real database structure"""
        fk_map = {}
        
        # Extract from database structure relationships
        for rel in self.global_relationships:
            from_table = rel.from_table
            if from_table:
                fk_map.setdefault(from_table, []).append({
                    'from_table': rel.from_table,
                    'to_table': rel.to_table,
                    'relationship_type': rel.relationship_type,
                    'confidence': rel.confidence
                })
        
        # Extract from table-level relationship information
        tables_info = self.database_structure.get('tables', {})
        for table_name, table_data in tables_info.items():
            relationships = table_data.get('relationships', [])
            for rel_str in relationships:
                if '->' in rel_str:
                    try:
                        from_col, to_info = [x.strip() for x in rel_str.split('->', 1)]
                        # Parse foreign key relationships
                        fk_map.setdefault(table_name, []).append({
                            'from_table': table_name,
                            'from_column': from_col,
                            'to_info': to_info,
                            'confidence': 0.8
                        })
                    except Exception:
                        continue
        
        return fk_map
    
    def _extract_view_join_patterns(self) -> Dict[str, List[Dict]]:
        """Extract proven join patterns from real views"""
        join_patterns = {}
        
        for pattern in self.view_patterns:
            joins = pattern.get('join_patterns', [])
            tables = pattern.get('tables_involved', [])
            
            for join in joins:
                # Extract table pairs and join conditions
                for i, table1 in enumerate(tables):
                    for table2 in tables[i+1:]:
                        if self._tables_referenced_in_join(join, table1, table2):
                            key = f"{table1}:{table2}"
                            if key not in join_patterns:
                                join_patterns[key] = []
                            
                            join_patterns[key].append({
                                'join_condition': join,
                                'source_view': pattern.get('view_name', ''),
                                'business_context': pattern.get('business_context', {}),
                                'proven_working': True
                            })
        
        return join_patterns
    
    def _tables_referenced_in_join(self, join_condition: str, table1: str, table2: str) -> bool:
        """Check if both tables are referenced in the join condition"""
        join_lower = join_condition.lower()
        
        # Extract simple table names for matching
        table1_simple = table1.split('.')[-1].replace('[', '').replace(']', '').lower()
        table2_simple = table2.split('.')[-1].replace('[', '').replace(']', '').lower()
        
        return table1_simple in join_lower and table2_simple in join_lower
    
    def find_relationships(self, tables: List[TableInfo]) -> List[Dict]:
        """Find relationships between selected real tables"""
        print("   🔗 Analyzing real table relationships...")
        
        table_names = [t.full_name for t in tables]
        relationships = []
        
        # Method 1: Use proven view join patterns (highest confidence)
        view_relationships = self._find_view_pattern_relationships(table_names)
        relationships.extend(view_relationships)
        
        # Method 2: Use explicit foreign key relationships
        fk_relationships = self._find_foreign_key_relationships(table_names)
        relationships.extend(fk_relationships)
        
        # Method 3: Infer relationships from column patterns
        inferred_relationships = self._find_inferred_relationships(tables)
        relationships.extend(inferred_relationships)
        
        # Remove duplicates and prioritize by confidence
        unique_relationships = self._deduplicate_relationships(relationships)
        
        print(f"      ✅ Found {len(unique_relationships)} relationships:")
        print(f"         • View patterns: {len(view_relationships)}")
        print(f"         • Foreign keys: {len(fk_relationships)}")
        print(f"         • Inferred: {len(inferred_relationships)}")
        
        return unique_relationships
    
    def _find_view_pattern_relationships(self, table_names: List[str]) -> List[Dict]:
        """Find relationships using proven view patterns"""
        relationships = []
        
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                # Check both directions
                for pattern_key in [f"{table1}:{table2}", f"{table2}:{table1}"]:
                    if pattern_key in self.view_join_patterns:
                        patterns = self.view_join_patterns[pattern_key]
                        best_pattern = max(patterns, key=lambda p: 1.0)  # All view patterns are equally valid
                        
                        relationships.append({
                            'from_table': table1,
                            'to_table': table2,
                            'join_condition': best_pattern['join_condition'],
                            'join_type': 'INNER JOIN',
                            'source': 'view_pattern',
                            'source_view': best_pattern['source_view'],
                            'confidence': 0.95,
                            'proven_working': True
                        })
        
        return relationships
    
    def _find_foreign_key_relationships(self, table_names: List[str]) -> List[Dict]:
        """Find explicit foreign key relationships"""
        relationships = []
        
        for table_name in table_names:
            if table_name in self.foreign_keys:
                for fk_info in self.foreign_keys[table_name]:
                    target_table = fk_info.get('to_table', '')
                    
                    # Check if target table is in our selected tables
                    if any(target_table in selected_table for selected_table in table_names):
                        from_col = fk_info.get('from_column', '')
                        
                        relationships.append({
                            'from_table': table_name,
                            'to_table': target_table,
                            'from_column': from_col,
                            'join_condition': f"{table_name}.{from_col} = {target_table}.ID",
                            'join_type': 'INNER JOIN',
                            'source': 'foreign_key',
                            'confidence': fk_info.get('confidence', 0.9)
                        })
        
        return relationships
    
    def _find_inferred_relationships(self, tables: List[TableInfo]) -> List[Dict]:
        """Infer relationships from real column patterns"""
        relationships = []
        
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                join_condition = self._infer_join_condition(table1, table2)
                if join_condition:
                    relationships.append({
                        'from_table': table1.full_name,
                        'to_table': table2.full_name,
                        'join_condition': join_condition,
                        'join_type': 'INNER JOIN',
                        'source': 'inferred',
                        'confidence': 0.7
                    })
        
        return relationships
    
    def _infer_join_condition(self, table1: TableInfo, table2: TableInfo) -> Optional[str]:
        """Infer join condition between two real tables"""
        table1_columns = [(col.get('name', ''), col.get('data_type', '')) for col in table1.columns]
        table2_columns = [(col.get('name', ''), col.get('data_type', '')) for col in table2.columns]
        
        # Method 1: Exact column name and type match
        for col1_name, col1_type in table1_columns:
            for col2_name, col2_type in table2_columns:
                if (col1_name.lower() == col2_name.lower() and 
                    col1_type.lower() == col2_type.lower() and
                    ('id' in col1_name.lower() or col1_name.lower().endswith('_id'))):
                    return f"{table1.full_name}.{col1_name} = {table2.full_name}.{col2_name}"
        
        # Method 2: Foreign key pattern matching
        for col1_name, col1_type in table1_columns:
            if col1_name.lower().endswith('_id') or col1_name.lower().endswith('id'):
                # Extract entity name
                entity_name = col1_name.lower().replace('_id', '').replace('id', '')
                
                # Look for matching table or ID column
                if entity_name in table2.name.lower():
                    # Find ID column in table2
                    for col2_name, col2_type in table2_columns:
                        if col2_name.lower() in ['id', 'primaryid', f'{entity_name}id']:
                            return f"{table1.full_name}.{col1_name} = {table2.full_name}.{col2_name}"
        
        # Method 3: Check if table names suggest relationship
        table1_simple = table1.name.lower()
        table2_simple = table2.name.lower()
        
        if table1_simple in table2_simple or table2_simple in table1_simple:
            # Look for common ID patterns
            for col1_name, _ in table1_columns:
                for col2_name, _ in table2_columns:
                    if (col1_name.lower() == col2_name.lower() and 
                        col1_name.lower() in ['id', 'key', 'primarykey']):
                        return f"{table1.full_name}.{col1_name} = {table2.full_name}.{col2_name}"
        
        return None
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships and prioritize by confidence"""
        seen_pairs = set()
        unique_relationships = []
        
        # Sort by confidence (highest first)
        sorted_relationships = sorted(relationships, key=lambda r: r.get('confidence', 0), reverse=True)
        
        for rel in sorted_relationships:
            from_table = rel.get('from_table', '')
            to_table = rel.get('to_table', '')
            
            # Create normalized pair key
            pair1 = f"{from_table}:{to_table}"
            pair2 = f"{to_table}:{from_table}"
            
            if pair1 not in seen_pairs and pair2 not in seen_pairs:
                seen_pairs.add(pair1)
                unique_relationships.append(rel)
        
        return unique_relationships


class SQLGenerator:
    """Generate SQL queries for real database tables"""
    
    def __init__(self, llm: LLMClient, view_patterns: List[Dict]):
        self.llm = llm
        self.view_patterns = view_patterns
    
    async def generate_query(self, question: str, tables: List[TableInfo], 
                           relationships: List[Dict]) -> str:
        """Generate SQL query using real table and column information"""
        print("   ⚡ Generating SQL for real database tables...")
        
        # Analyze query requirements
        query_analysis = self._analyze_query_requirements(question, tables)
        
        # Find relevant view patterns as examples
        relevant_patterns = self._find_relevant_view_patterns(question, tables)
        
        # Prepare comprehensive context
        table_context = self._prepare_detailed_table_context(tables, query_analysis)
        join_context = self._prepare_join_context(relationships, relevant_patterns)
        
        # Create enhanced system prompt
        system_prompt = self._create_system_prompt(query_analysis, relevant_patterns)
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

QUERY ANALYSIS:
{json.dumps(query_analysis, indent=2)}

AVAILABLE REAL TABLES:
{table_context}

PROVEN RELATIONSHIPS & JOIN PATTERNS:
{join_context}

SUCCESSFUL VIEW EXAMPLES:
{self._format_view_examples(relevant_patterns)}

Generate SQL that answers this business question using the real table structure provided.
Use proven join patterns from the view examples where applicable.
Return only the SQL query.
"""
        
        response = await self.llm.analyze(system_prompt, user_prompt)
        sql = self._clean_sql(response)
        
        print(f"      ✅ Generated SQL using real table structure")
        return sql
    
    def _analyze_query_requirements(self, question: str, tables: List[TableInfo]) -> Dict:
        """Analyze what the query needs based on question and real tables"""
        q_lower = question.lower()
        
        analysis = {
            'operation_type': 'select',
            'aggregation_needed': False,
            'aggregation_functions': [],
            'time_filtering': False,
            'grouping_needed': False,
            'entity_focus': [],
            'likely_result_columns': [],
            'complexity': 'simple'
        }
        
        # Determine operation type
        if any(word in q_lower for word in ['count', 'number of', 'how many']):
            analysis['operation_type'] = 'count'
            analysis['aggregation_needed'] = True
            analysis['aggregation_functions'].append('COUNT')
        
        if any(word in q_lower for word in ['total', 'sum']):
            analysis['aggregation_needed'] = True
            analysis['aggregation_functions'].append('SUM')
        
        if any(word in q_lower for word in ['average', 'avg', 'mean']):
            analysis['aggregation_needed'] = True
            analysis['aggregation_functions'].append('AVG')
        
        # Time filtering detection
        if any(word in q_lower for word in ['2025', '2024', 'year', 'month', 'last', 'this', 'period']):
            analysis['time_filtering'] = True
        
        # Entity focus
        for table in tables:
            if table.entity_type != 'Unknown':
                analysis['entity_focus'].append(table.entity_type)
        
        # Complexity assessment
        if len(tables) > 1:
            analysis['complexity'] = 'complex'
        elif analysis['aggregation_needed']:
            analysis['complexity'] = 'medium'
        
        # Predict likely result columns based on question
        if 'count' in q_lower or 'number' in q_lower:
            analysis['likely_result_columns'] = ['Count']
        elif 'total' in q_lower or 'sum' in q_lower:
            analysis['likely_result_columns'] = ['Total']
        else:
            # Use table columns to predict
            for table in tables:
                for col in table.columns[:5]:
                    col_name = col.get('name', '')
                    if any(word in col_name.lower() for word in q_lower.split() if len(word) > 3):
                        analysis['likely_result_columns'].append(col_name)
        
        return analysis
    
    def _find_relevant_view_patterns(self, question: str, tables: List[TableInfo]) -> List[Dict]:
        """Find view patterns relevant to the current query"""
        relevant = []
        q_lower = question.lower()
        table_names = [t.full_name for t in tables]
        
        for pattern in self.view_patterns:
            relevance_score = 0
            
            # Score by table overlap
            pattern_tables = pattern.get('tables_involved', [])
            table_overlap = 0
            for pt in pattern_tables:
                for tn in table_names:
                    if pt.lower() in tn.lower() or tn.lower() in pt.lower():
                        table_overlap += 1
                        break
            
            relevance_score += table_overlap * 3
            
            # Score by business context
            business_context = pattern.get('business_context', {})
            if 'customer' in q_lower and business_context.get('contains_customer_data'):
                relevance_score += 2
            if 'payment' in q_lower and business_context.get('contains_financial_data'):
                relevance_score += 2
            if 'total' in q_lower and business_context.get('contains_aggregations'):
                relevance_score += 2
            
            # Score by view name similarity
            view_name = pattern.get('view_name', '').lower()
            question_words = [word for word in q_lower.split() if len(word) > 3]
            for word in question_words:
                if word in view_name:
                    relevance_score += 1
            
            if relevance_score > 0:
                pattern_with_score = pattern.copy()
                pattern_with_score['relevance_score'] = relevance_score
                relevant.append(pattern_with_score)
        
        # Sort by relevance and return top patterns
        relevant.sort(key=lambda p: p['relevance_score'], reverse=True)
        return relevant[:3]
    
    def _prepare_detailed_table_context(self, tables: List[TableInfo], query_analysis: Dict) -> str:
        """Prepare detailed context about real tables"""
        context_lines = []
        
        for table in tables:
            context_lines.append(f"TABLE: {table.full_name}")
            context_lines.append(f"  Entity Type: {table.entity_type}")
            context_lines.append(f"  Rows: {table.row_count:,}")
            
            # Show important columns based on query analysis
            important_columns = []
            for col in table.columns:
                col_name = col.get('name', '')
                col_type = col.get('data_type', '')
                
                # Highlight relevant columns
                if query_analysis.get('time_filtering') and 'date' in col_name.lower():
                    important_columns.append(f"⭐ {col_name} ({col_type}) [DATE COLUMN]")
                elif 'id' in col_name.lower():
                    important_columns.append(f"🔑 {col_name} ({col_type}) [ID COLUMN]")
                elif any(word in col_name.lower() for word in ['amount', 'total', 'price', 'revenue']):
                    important_columns.append(f"💰 {col_name} ({col_type}) [FINANCIAL]")
                else:
                    important_columns.append(f"   {col_name} ({col_type})")
            
            context_lines.append(f"  Key Columns: {', '.join(important_columns[:8])}")
            
            # Show sample data insights
            if table.sample_data:
                sample_insights = self._extract_sample_insights(table.sample_data[0])
                if sample_insights:
                    context_lines.append(f"  Sample Data: {sample_insights}")
            
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def _extract_sample_insights(self, sample_row: Dict) -> str:
        """Extract meaningful insights from sample data"""
        insights = []
        
        for key, value in list(sample_row.items())[:4]:
            if value is not None:
                if isinstance(value, str) and len(str(value)) > 50:
                    insights.append(f"{key}=[TEXT]")
                elif 'date' in key.lower() and ('2025' in str(value) or '2024' in str(value)):
                    insights.append(f"{key}={value} [RECENT]")
                elif isinstance(value, (int, float)) and value > 1000:
                    insights.append(f"{key}={value:,}")
                else:
                    insights.append(f"{key}={value}")
        
        return ", ".join(insights)
    
    def _prepare_join_context(self, relationships: List[Dict], view_patterns: List[Dict]) -> str:
        """Prepare context about joins with proven patterns"""
        context_lines = []
        
        if relationships:
            context_lines.append("AVAILABLE RELATIONSHIPS:")
            for rel in relationships:
                source = rel.get('source', 'unknown')
                confidence = rel.get('confidence', 0)
                join_condition = rel.get('join_condition', 'No condition specified')
                
                if source == 'view_pattern':
                    context_lines.append(f"  ✅ PROVEN: {join_condition} (from view: {rel.get('source_view', 'unknown')})")
                elif source == 'foreign_key':
                    context_lines.append(f"  🔑 FK: {join_condition} (confidence: {confidence:.1f})")
                else:
                    context_lines.append(f"  💡 INFERRED: {join_condition} (confidence: {confidence:.1f})")
        else:
            context_lines.append("NO RELATIONSHIPS FOUND - Use single table queries")
        
        if view_patterns:
            context_lines.append("\nPROVEN JOIN PATTERNS FROM SUCCESSFUL VIEWS:")
            for pattern in view_patterns:
                joins = pattern.get('join_patterns', [])
                if joins:
                    context_lines.append(f"  Pattern from {pattern.get('view_name', 'unknown')}:")
                    for join in joins[:2]:  # Show top 2 join patterns
                        context_lines.append(f"    {join}")
        
        return "\n".join(context_lines)
    
    def _format_view_examples(self, view_patterns: List[Dict]) -> str:
        """Format view patterns as examples for SQL generation"""
        if not view_patterns:
            return "No relevant view patterns found."
        
        examples = []
        for pattern in view_patterns:
            example = f"VIEW: {pattern.get('view_name', 'Unknown')}"
            
            if pattern.get('definition'):
                # Extract key parts of the view definition
                definition = pattern['definition']
                key_lines = []
                
                for line in definition.split('\n'):
                    line = line.strip()
                    if any(keyword in line.upper() for keyword in ['SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP BY']):
                        key_lines.append(line)
                        if len(key_lines) >= 6:  # Limit example length
                            break
                
                if key_lines:
                    example += "\nKey SQL Logic:\n" + "\n".join(key_lines)
            
            if pattern.get('business_context'):
                bc = pattern['business_context']
                if bc.get('likely_business_function') != 'unknown':
                    example += f"\nBusiness Function: {bc['likely_business_function']}"
            
            examples.append(example)
        
        return "\n\n" + "\n---\n".join(examples)
    
    def _create_system_prompt(self, query_analysis: Dict, view_patterns: List[Dict]) -> str:
        """Create enhanced system prompt for real database SQL generation"""
        
        base_prompt = """You are an expert SQL generator for a real E-Commerce/Order Management database.
Generate accurate, efficient SQL queries that work with the actual table structure provided.

IMPORTANT RULES:
1. Use actual table names and column names EXACTLY as provided
2. Use proper table aliases for readability (t1, t2, etc.)
3. Include TOP 100 to limit results unless counting
4. Use meaningful column aliases for business users
5. Handle international characters properly (UTF-8)
6. Use proven JOIN patterns from successful views when available
7. Return only the SQL query, no explanations"""
        
        # Add query-specific guidance
        if query_analysis.get('aggregation_needed'):
            base_prompt += f"""

AGGREGATION REQUIREMENTS:
- This query needs {', '.join(query_analysis.get('aggregation_functions', ['COUNT']))}
- Use appropriate GROUP BY clauses
- Provide meaningful aggregate column names"""
        
        if query_analysis.get('time_filtering'):
            base_prompt += """

TIME FILTERING:
- Look for date columns in the provided tables
- Use appropriate date filtering (YEAR, DATEPART, etc.)
- Consider date ranges if specific years mentioned"""
        
        if view_patterns:
            base_prompt += f"""

VIEW PATTERN GUIDANCE:
- You have {len(view_patterns)} proven view patterns as examples
- These patterns show successful SQL that works in this database
- Use their JOIN logic and business patterns as guidance"""
        
        return base_prompt
    
    def _clean_sql(self, response: str) -> str:
        """Clean and format SQL from LLM response"""
        # Remove markdown
        cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Extract SQL statement
        lines = cleaned.strip().split('\n')
        sql_lines = []
        found_select = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'WITH')) or found_select:
                found_select = True
                sql_lines.append(line)
                if line.endswith(';'):
                    break
        
        if sql_lines:
            sql = '\n'.join(sql_lines).rstrip(';').strip()
            
            # Add TOP clause if not present and not a COUNT query
            if ('TOP ' not in sql.upper() and 
                'COUNT(' not in sql.upper() and 
                sql.upper().startswith('SELECT')):
                sql = sql.replace('SELECT', 'SELECT TOP 100', 1)
            
            return sql
        
        return cleaned.strip()


class QueryExecutor:
    """Execute SQL queries safely on real database"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def execute_query(self, sql: str) -> tuple:
        """Execute SQL query on real database and return results"""
        if not sql or not sql.strip():
            return [], "Empty SQL query"
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                # Set UTF-8 encoding for international characters
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                
                # Set query timeout for safety
                cursor.timeout = 30
                
                cursor.execute(sql)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    # Fetch results with safety limit
                    row_count = 0
                    for row in cursor:
                        if row_count >= 100:  # Safety limit
                            break
                        
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = self._safe_value(value)
                        results.append(row_dict)
                        row_count += 1
                    
                    return results, None
                else:
                    return [], None
                    
        except pyodbc.Error as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "Invalid column name" in error_msg:
                return [], f"Column not found in database: {error_msg}"
            elif "Invalid object name" in error_msg:
                return [], f"Table not found in database: {error_msg}"
            elif "Timeout" in error_msg:
                return [], "Query timeout - try a simpler query or add more filters"
            elif "Permission" in error_msg:
                return [], "Database permission error - check access rights"
            else:
                return [], f"Database error: {error_msg}"
        
        except Exception as e:
            return [], f"Unexpected error: {str(e)}"
    
    def _safe_value(self, value):
        """Convert database value to safe display format"""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif hasattr(value, 'decode'):  # Handle binary data
            try:
                return value.decode('utf-8')[:200]
            except:
                return "[Binary Data]"
        else:
            return str(value)[:200]


class QueryInterface:
    """Enhanced interactive query interface for real database"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.data_loader = DataLoader(config)
        self.executor = QueryExecutor(config)
        
        # Load real cached data
        if not self.data_loader.load_data():
            raise ValueError("Failed to load cached data. Run discovery and analysis first.")
        
        # Initialize components with real data
        self.table_selector = TableSelector(
            self.data_loader.tables, 
            self.llm, 
            self.data_loader.view_patterns
        )
        self.relationship_finder = RelationshipFinder(
            self.data_loader.database_structure, 
            self.data_loader.relationships,
            self.data_loader.view_patterns
        )
        self.sql_generator = SQLGenerator(
            self.llm, 
            self.data_loader.view_patterns
        )
        
        print(f"   📊 Loaded {len(self.data_loader.tables)} real tables")
        print(f"   🔗 Database relationships: {len(self.data_loader.relationships)}")
        print(f"   👁️ View patterns for examples: {len(self.data_loader.view_patterns)}")
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship]):
        """Start interactive session with real database"""
        
        print(f"🚀 Enhanced Query Interface for Real Database")
        print(f"   📊 Real tables available: {len(tables)}")
        
        if domain:
            print(f"   🏢 Business Domain: {domain.domain_type}")
            if domain.sample_questions:
                print(f"   💡 Try asking: {domain.sample_questions[0]}")
        
        # Show real entity distribution
        entity_counts = {}
        for table in tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   📊 Available entities: {dict(list(entity_counts.items())[:5])}")
            
            # Show business capabilities
            print(f"   💼 You can ask about:")
            if entity_counts.get('Customer', 0) > 0:
                print(f"      • Customer analysis ({entity_counts['Customer']} customer tables)")
            if entity_counts.get('Financial', 0) > 0:
                print(f"      • Financial analysis ({entity_counts['Financial']} financial tables)")
            if entity_counts.get('Product', 0) > 0:
                print(f"      • Product analysis ({entity_counts['Product']} product tables)")
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🔄 Processing query with real database analysis...")
                
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
        print(f"   🔍 Used real database with {len(self.data_loader.tables)} tables")
        print(f"   👁️ Leveraged {len(self.data_loader.view_patterns)} proven view patterns")
    
    async def process_query(self, question: str) -> QueryResult:
        """Process query using real database intelligence"""
        
        try:
            # Stage 1: Intelligent table selection using real data
            selected_tables = await self.table_selector.find_relevant_tables(question)
            
            if not selected_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found in real database"
                )
            
            # Stage 2: Find real relationships (including view patterns)
            relationships = self.relationship_finder.find_relationships(selected_tables)
            
            # Stage 3: Generate SQL using real table structure
            sql = await self.sql_generator.generate_query(question, selected_tables, relationships)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to generate SQL for real database"
                )
            
            # Stage 4: Execute on real database
            results, error = self.executor.execute_query(sql)
            
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
                error=f"Real database query processing failed: {str(e)}"
            )
    
    def display_result(self, result: QueryResult):
        """Display results with real database context"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.sql_query:
                print(f"📋 Generated SQL:")
                print(f"```sql\n{result.sql_query}\n```")
                
                # Provide helpful debugging for real database
                print("\n🔍 Real Database Debugging:")
                print("   • Check that table and column names exist in your database")
                print("   • Verify you have read permissions on the selected tables")
                print("   • Consider if the tables contain the expected data")
        else:
            print(f"📋 Generated SQL:")
            print(f"```sql\n{result.sql_query}\n```")
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                # Handle single value results (like counts)
                if result.is_single_value():
                    value = result.get_single_value()
                    column_name = list(result.results[0].keys())[0]
                    
                    if isinstance(value, (int, float)) and value >= 1000:
                        print(f"   🎯 {column_name}: {value:,}")
                    else:
                        print(f"   🎯 {column_name}: {value}")
                else:
                    # Handle multiple rows with smart formatting
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
                print("   ⚠️ No results returned from real database")
                print("   💡 This could mean:")
                print("      • No data matches your criteria")
                print("      • Date filters are too restrictive") 
                print("      • Tables don't contain the expected data")
                print("      • Column values don't match your question")
            
            # Show real tables used with context
            if result.tables_used:
                print(f"📋 Real tables analyzed:")
                for table_name in result.tables_used:
                    # Find table info for context
                    table_info = next((t for t in self.data_loader.tables if t.full_name == table_name), None)
                    if table_info:
                        entity_type = table_info.entity_type
                        row_count = table_info.row_count
                        print(f"      • {table_name} ({entity_type}) - {row_count:,} rows")
                    else:
                        print(f"      • {table_name}")
        
        # Enhanced troubleshooting for real database
        print("\n💡 Real Database Troubleshooting:")
        print("   • Database structure: data/database_structure.json")
        print("   • Semantic analysis: data/semantic_analysis.json")
        
        if result.error:
            print("   🔧 For errors, try:")
            print("      • Simpler questions to test table accessibility")
            print("      • Questions about tables you know contain data")
            print("      • Checking database permissions")
        elif len(result.results) == 0:
            print("   🔍 For empty results, try:")
            print("      • Broader time ranges (remove year filters)")
            print("      • Different entity types from your domain")
            print("      • Questions about data you know exists")
        
        # Show sample business questions for this real database
        if hasattr(self.data_loader, 'domain') and self.data_loader.domain:
            domain = self.data_loader.domain
            if domain.sample_questions:
                print(f"   💼 Try these {domain.domain_type} questions:")
                for question in domain.sample_questions[:2]:
                    print(f"      • \"{question}\"")