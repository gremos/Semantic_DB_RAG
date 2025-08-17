#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BI-Aware Semantic Analysis - No-Fallback Operating Rules
Following BI Requirements: Capability contracts, evidence-driven selection, NER
DRY, SOLID, YAGNI principles with strict validation gates
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
from dataclasses import dataclass, field

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import (TableInfo, BusinessDomain, Relationship, NonExecutableAnalysisReport, 
                          AnalyticalTask, EvidenceScore, CapabilityContract)
from shared.utils import parse_json_response

class LLMAnalyzer:
    """LLM communication for BI-aware analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            request_timeout=60,
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

class BIAwareTableClassifier:
    """BI-aware table classification with operational/planning/reference detection"""
    
    def __init__(self, llm_analyzer: LLMAnalyzer):
        self.llm_analyzer = llm_analyzer
    
    async def classify_tables(self, tables: List[TableInfo]) -> int:
        """Classify tables with BI-aware categories"""
        print(f"🏷️ BI-aware classification of {len(tables)} tables...")
        
        batch_size = 6  # Smaller batches for more detailed analysis
        classified = 0
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            
            # Create BI-aware classification prompt
            prompt = self._create_bi_classification_prompt(batch)
            
            # Get LLM response
            response = await self.llm_analyzer.analyze(
                "You are a business intelligence analyst. Classify database tables for BI capability assessment. Respond with valid JSON only.",
                prompt
            )
            
            # Apply classifications
            classified += self._apply_bi_classifications(response, batch)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        print(f"   ✅ BI-classified {classified} tables")
        return classified
    
    def _create_bi_classification_prompt(self, tables: List[TableInfo]) -> str:
        """Create BI-aware classification prompt"""
        
        table_summaries = []
        for table in tables:
            sample_preview = self._get_sample_preview(table)
            column_analysis = self._analyze_columns_for_bi(table)
            
            table_summaries.append({
                'table_name': table.full_name,
                'columns': column_analysis,
                'sample_data': sample_preview,
                'row_count': table.row_count
            })
        
        return f"""
Analyze these database tables for BUSINESS INTELLIGENCE capabilities. Focus on identifying:

1. DATA TYPE: Operational (real transactions), Planning (targets/budgets), or Reference (lookups)
2. BI ROLE: Fact (measures/metrics), Dimension (entities/attributes), or Bridge (relationships)
3. GRAIN: What each row represents (customer, transaction, order, etc.)
4. MEASURES: Numeric columns suitable for aggregation
5. ENTITY KEYS: Columns for grouping/filtering
6. TIME DIMENSION: Date/timestamp columns for temporal analysis

TABLES TO ANALYZE:
{json.dumps(table_summaries, indent=2)}

CLASSIFICATION RULES:
- Operational data: Real transactions, events, actual business activity
- Planning data: Targets, goals, budgets, forecasts (often with zeros or future dates)
- Reference data: Lookup tables, codes, categories, static definitions

- Fact tables: Contain measures (amounts, quantities) and foreign keys
- Dimension tables: Contain descriptive attributes (names, descriptions)
- Bridge tables: Many-to-many relationships

Look at actual sample data values to determine:
- Are amounts real transaction values or planning targets?
- Are dates historical (operational) or future (planning)?
- Do rows represent events/transactions or reference items?

Respond with JSON only:
{{
  "classifications": [
    {{
      "table_name": "[schema].[table]",
      "data_type": "operational|planning|reference",
      "bi_role": "fact|dimension|bridge",
      "grain": "customer|transaction|order|product|event",
      "entity_type": "Customer|Payment|Order|Product|Reference",
      "measures": ["amount_column", "quantity_column"],
      "entity_keys": ["customer_id", "product_id"],
      "time_columns": ["created_date", "transaction_date"],
      "filter_columns": ["status", "type", "region"],
      "confidence": 0.9,
      "reasoning": "Sample data shows real transaction amounts and historical dates"
    }}
  ]
}}
"""
    
    def _analyze_columns_for_bi(self, table: TableInfo) -> Dict[str, Any]:
        """Analyze columns for BI capabilities"""
        analysis = {
            'measures': [],
            'entity_keys': [],
            'time_columns': [],
            'filter_columns': [],
            'total_columns': len(table.columns)
        }
        
        for col in table.columns:
            col_name = col.get('name', '').lower()
            col_type = col.get('data_type', '').lower()
            
            # Identify measures (numeric columns for aggregation)
            if any(numeric_type in col_type for numeric_type in ['decimal', 'money', 'float', 'numeric', 'int']):
                if any(measure_word in col_name for measure_word in ['amount', 'value', 'price', 'cost', 'revenue', 'total', 'quantity', 'count']):
                    analysis['measures'].append(col.get('name'))
            
            # Identify entity keys
            if col_name.endswith('id') or 'key' in col_name:
                analysis['entity_keys'].append(col.get('name'))
            
            # Identify time columns
            if ('date' in col_type or 'time' in col_type or 
                any(time_word in col_name for time_word in ['date', 'time', 'created', 'modified', 'updated'])):
                analysis['time_columns'].append(col.get('name'))
            
            # Identify filter columns
            if any(filter_word in col_name for filter_word in ['status', 'type', 'category', 'region', 'channel', 'active']):
                analysis['filter_columns'].append(col.get('name'))
        
        return analysis
    
    def _get_sample_preview(self, table: TableInfo) -> str:
        """Get meaningful sample data preview for BI analysis"""
        if not table.sample_data:
            return "No sample data"
        
        # Look for meaningful patterns in sample data
        sample_analysis = []
        first_row = table.sample_data[0]
        
        for key, value in list(first_row.items())[:6]:
            if key.startswith('__'):  # Skip metadata
                continue
                
            # Analyze value patterns
            if isinstance(value, (int, float)) and value > 0:
                sample_analysis.append(f"{key}: {value} (positive numeric)")
            elif isinstance(value, (int, float)) and value == 0:
                sample_analysis.append(f"{key}: 0 (zero - planning?)")
            elif isinstance(value, str) and len(str(value)) > 0:
                sample_analysis.append(f"{key}: '{value}' (text)")
            else:
                sample_analysis.append(f"{key}: {value}")
        
        return " | ".join(sample_analysis)
    
    def _apply_bi_classifications(self, response: str, batch: List[TableInfo]) -> int:
        """Apply BI-aware classifications to tables"""
        data = parse_json_response(response)
        if not data or 'classifications' not in data:
            return 0
        
        count = 0
        for classification in data['classifications']:
            table_name = classification.get('table_name', '')
            
            # Find and update table
            for table in batch:
                if table.full_name == table_name:
                    # Standard classifications
                    table.entity_type = classification.get('entity_type', 'Unknown')
                    table.confidence = float(classification.get('confidence', 0.0))
                    
                    # BI-specific classifications
                    table.data_type = classification.get('data_type', 'reference')
                    table.bi_role = classification.get('bi_role', 'dimension')
                    table.grain = classification.get('grain', 'unknown')
                    table.business_role = classification.get('bi_role', 'Supporting')
                    
                    # BI capabilities
                    table.measures = classification.get('measures', [])
                    table.entity_keys = classification.get('entity_keys', [])
                    table.time_columns = classification.get('time_columns', [])
                    table.filter_columns = classification.get('filter_columns', [])
                    
                    count += 1
                    break
        
        return count

class CapabilityAnalyzer:
    """Analyze tables for BI capability contracts"""
    
    def __init__(self):
        self.business_synonyms = {
            'customer': ['customer', 'client', 'account', 'user', 'subscriber'],
            'revenue': ['amount', 'value', 'price', 'cost', 'revenue', 'total', 'payment'],
            'transaction': ['payment', 'order', 'sale', 'purchase', 'transaction'],
            'product': ['product', 'item', 'inventory', 'sku', 'catalog'],
            'time': ['date', 'time', 'created', 'modified', 'updated', 'timestamp']
        }
    
    def assess_capability_contract(self, table: TableInfo, intent: AnalyticalTask) -> CapabilityContract:
        """Assess if table satisfies capability contract for intent"""
        contract = CapabilityContract()
        
        # Determine grain from BI classification
        contract.grain = getattr(table, 'grain', 'unknown')
        
        # Find measures
        contract.measures = getattr(table, 'measures', [])
        
        # Find time columns
        time_columns = getattr(table, 'time_columns', [])
        if time_columns:
            contract.time_column = time_columns[0]  # Use first available
        
        # Find entity keys
        contract.entity_keys = getattr(table, 'entity_keys', [])
        
        # Assess quality
        contract.quality_checks = {
            'row_count': table.row_count,
            'has_sample_data': len(table.sample_data) > 0,
            'measures_available': len(contract.measures) > 0,
            'time_available': contract.time_column is not None
        }
        
        return contract
    
    def generate_fix_paths(self, table: TableInfo, missing_capabilities: List[str]) -> List[str]:
        """Generate suggested fix paths for missing capabilities"""
        fixes = []
        
        if "Row grain identification" in missing_capabilities:
            fixes.append(f"Define grain for {table.full_name} - what does each row represent?")
        
        if "Numeric measures" in missing_capabilities:
            numeric_cols = [col.get('name') for col in table.columns 
                          if col.get('data_type', '').lower() in ['decimal', 'money', 'float', 'numeric', 'int']]
            if numeric_cols:
                fixes.append(f"Use numeric columns as measures: {', '.join(numeric_cols[:3])}")
            else:
                fixes.append(f"No numeric columns found in {table.full_name} for aggregation")
        
        if "Time/date column" in missing_capabilities:
            date_cols = [col.get('name') for col in table.columns 
                        if 'date' in col.get('name', '').lower() or 'time' in col.get('data_type', '').lower()]
            if date_cols:
                fixes.append(f"Use time columns: {', '.join(date_cols[:2])}")
            else:
                fixes.append(f"No time/date columns found in {table.full_name}")
        
        return fixes

class EvidenceScorer:
    """Evidence-driven object selection with weighted scoring"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
        self.table_map = {t.full_name: t for t in tables}
    
    def score_table_for_intent(self, table: TableInfo, intent: AnalyticalTask) -> EvidenceScore:
        """Score table based on evidence for analytical intent"""
        score = EvidenceScore()
        
        # Role match (high weight)
        score.role_match = self._score_role_match(table, intent)
        
        # Join evidence (high weight)  
        score.join_evidence = self._score_join_evidence(table)
        
        # Lexical/semantic match (medium weight)
        score.lexical_match = self._score_lexical_match(table, intent)
        
        # Operational vs planning (medium weight)
        score.operational_tag = self._score_operational_data(table)
        
        # Row count and freshness (tie-breaker)
        score.row_count = min(1.0, table.row_count / 10000.0) if table.row_count > 0 else 0.0
        score.freshness = 1.0  # Assume fresh for now
        
        return score
    
    def _score_role_match(self, table: TableInfo, intent: AnalyticalTask) -> float:
        """Score based on BI role appropriateness"""
        bi_role = getattr(table, 'bi_role', 'dimension')
        data_type = getattr(table, 'data_type', 'reference')
        
        # For aggregation tasks, prefer fact tables with operational data
        if intent.task_type in ['aggregation', 'ranking', 'trend']:
            if bi_role == 'fact' and data_type == 'operational':
                return 1.0
            elif bi_role == 'fact':
                return 0.8
            else:
                return 0.3
        
        # For other tasks, dimension tables are also valuable
        return 0.7 if bi_role in ['fact', 'dimension'] else 0.3
    
    def _score_join_evidence(self, table: TableInfo) -> float:
        """Score based on join relationships"""
        # Count foreign key relationships
        fk_count = len(table.relationships)
        
        # Score based on connectivity
        if fk_count >= 3:
            return 1.0
        elif fk_count >= 1:
            return 0.7
        else:
            return 0.3
    
    def _score_lexical_match(self, table: TableInfo, intent: AnalyticalTask) -> float:
        """Score based on lexical/semantic matching"""
        table_name = table.name.lower()
        entity_type = table.entity_type.lower()
        
        # Match against intent entity
        if intent.entity and intent.entity.lower() in table_name:
            return 1.0
        
        # Match against metrics
        for metric in intent.metrics:
            if metric.lower() in table_name or metric.lower() in entity_type:
                return 0.8
        
        return 0.2
    
    def _score_operational_data(self, table: TableInfo) -> float:
        """Score operational data higher than planning"""
        data_type = getattr(table, 'data_type', 'reference')
        
        if data_type == 'operational':
            return 1.0
        elif data_type == 'reference':
            return 0.6
        else:  # planning
            return 0.3
    
    def rank_candidates(self, intent: AnalyticalTask, top_k: int = 5) -> List[Tuple[TableInfo, EvidenceScore]]:
        """Rank candidate tables by evidence score"""
        scored_tables = []
        
        for table in self.tables:
            score = self.score_table_for_intent(table, intent)
            scored_tables.append((table, score))
        
        # Sort by total score descending
        scored_tables.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return scored_tables[:top_k]

class NERGenerator:
    """Non-Executable Analysis Report generator"""
    
    def __init__(self):
        pass
    
    def generate_ner(self, question: str, intent: AnalyticalTask, 
                    capability_failures: List[Tuple[TableInfo, CapabilityContract]],
                    top_candidates: List[Tuple[TableInfo, EvidenceScore]]) -> NonExecutableAnalysisReport:
        """Generate comprehensive NER when capability checks fail"""
        
        # Collect all missing capabilities
        all_missing = set()
        for table, contract in capability_failures:
            all_missing.update(contract.get_missing_capabilities())
        
        # Generate fix paths
        fix_paths = []
        capability_analyzer = CapabilityAnalyzer()
        
        for table, contract in capability_failures[:3]:  # Top 3 candidates
            missing = contract.get_missing_capabilities()
            table_fixes = capability_analyzer.generate_fix_paths(table, missing)
            fix_paths.extend([f"{table.full_name}: {fix}" for fix in table_fixes])
        
        # Generate safe exploratory queries
        safe_queries = []
        for table, score in top_candidates[:2]:
            safe_queries.append(f"-- Explore {table.full_name} structure\nSELECT TOP 5 * FROM {table.full_name}")
            
            if hasattr(table, 'measures') and table.measures:
                measures_str = ', '.join(table.measures[:2])
                safe_queries.append(f"-- Check {table.full_name} measures\nSELECT {measures_str}, COUNT(*) as row_count FROM {table.full_name} GROUP BY {measures_str}")
        
        return NonExecutableAnalysisReport(
            question=question,
            normalized_task=intent,
            missing_capabilities=list(all_missing),
            top_candidate_tables=[(t.full_name, s) for t, s in top_candidates],
            fix_paths=fix_paths,
            suggested_queries=safe_queries
        )

class BISemanticAnalyzer:
    """BI-Aware Semantic Analyzer with capability contracts and NER"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize BI-aware components
        self.llm_analyzer = LLMAnalyzer(config)
        self.table_classifier = BIAwareTableClassifier(self.llm_analyzer)
        self.capability_analyzer = CapabilityAnalyzer()
        self.ner_generator = NERGenerator()
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        self.evidence_scorer: Optional[EvidenceScorer] = None
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """BI-aware analysis with capability assessment"""
        print("🧠 BI-Aware Semantic Analysis (No-Fallback)")
        print("Following capability contracts and evidence-driven selection")
        
        try:
            # Copy input tables
            self.tables = [table for table in tables]
            
            # Step 1: BI-aware classification
            await self.table_classifier.classify_tables(self.tables)
            
            # Step 2: Initialize evidence scorer
            self.evidence_scorer = EvidenceScorer(self.tables)
            
            # Step 3: Enhance relationships with BI context
            self.relationships = self._enhance_bi_relationships()
            
            # Step 4: Determine business domain with BI focus
            self.domain = await self._determine_bi_domain()
            
            # Show BI-aware summary
            self._show_bi_summary()
            return True
            
        except Exception as e:
            print(f"❌ BI-aware analysis failed: {e}")
            return False
    
    def _enhance_bi_relationships(self) -> List[Relationship]:
        """Enhance relationships with BI awareness"""
        relationships = []
        
        # Extract existing FK relationships
        for table in self.tables:
            for rel_info in table.relationships:
                if '->' in rel_info:
                    try:
                        parts = rel_info.split('->', 1)
                        from_col = parts[0].strip()
                        to_ref = parts[1].strip()
                        
                        relationships.append(Relationship(
                            from_table=table.full_name,
                            to_table=to_ref.split('.')[0] if '.' in to_ref else to_ref,
                            relationship_type='foreign_key',
                            confidence=0.95,
                            description=f"FK: {from_col} -> {to_ref}"
                        ))
                    except Exception:
                        continue
        
        # Add BI-specific relationships (fact-to-dimension)
        fact_tables = [t for t in self.tables if getattr(t, 'bi_role', '') == 'fact']
        dimension_tables = [t for t in self.tables if getattr(t, 'bi_role', '') == 'dimension']
        
        for fact in fact_tables:
            fact_keys = getattr(fact, 'entity_keys', [])
            for dim in dimension_tables:
                dim_keys = [col.get('name', '') for col in dim.columns if 'id' in col.get('name', '').lower()]
                
                # Look for matching key patterns
                for fact_key in fact_keys:
                    for dim_key in dim_keys:
                        if self._keys_match(fact_key, dim_key, dim.entity_type):
                            relationships.append(Relationship(
                                from_table=fact.full_name,
                                to_table=dim.full_name,
                                relationship_type='fact_dimension',
                                confidence=0.8,
                                description=f"BI relationship: {fact_key} -> {dim_key}"
                            ))
        
        return relationships
    
    def _keys_match(self, fact_key: str, dim_key: str, entity_type: str) -> bool:
        """Check if fact and dimension keys match"""
        fact_lower = fact_key.lower()
        dim_lower = dim_key.lower()
        entity_lower = entity_type.lower()
        
        # Direct match
        if fact_lower == dim_lower:
            return True
        
        # Entity type match
        if entity_lower in fact_lower and 'id' in fact_lower:
            return True
        
        return False
    
    async def _determine_bi_domain(self) -> Optional[BusinessDomain]:
        """Determine business domain with BI focus"""
        # Count BI roles and data types
        bi_distribution = {
            'fact_tables': len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact']),
            'dimension_tables': len([t for t in self.tables if getattr(t, 'bi_role', '') == 'dimension']),
            'operational_tables': len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational']),
            'planning_tables': len([t for t in self.tables if getattr(t, 'data_type', '') == 'planning']),
        }
        
        # Determine domain based on BI patterns
        if bi_distribution['fact_tables'] >= 3 and bi_distribution['operational_tables'] >= 5:
            domain_type = "Operational BI System"
            confidence = 0.9
        elif bi_distribution['planning_tables'] >= bi_distribution['operational_tables']:
            domain_type = "Planning & Analytics System"
            confidence = 0.8
        else:
            domain_type = "Business System"
            confidence = 0.6
        
        capabilities = {
            'customer_analysis': any('customer' in t.entity_type.lower() for t in self.tables),
            'financial_analysis': any('payment' in t.entity_type.lower() or 'financial' in t.entity_type.lower() for t in self.tables),
            'operational_reporting': bi_distribution['operational_tables'] > 0,
            'planning_analysis': bi_distribution['planning_tables'] > 0,
            'trend_analysis': any(getattr(t, 'time_columns', []) for t in self.tables)
        }
        
        return BusinessDomain(
            domain_type=domain_type,
            industry="Business Intelligence",
            confidence=confidence,
            sample_questions=[
                "What is our total revenue this quarter?",
                "Who are our top 10 customers by value?",
                "How many active customers do we have?",
                "What is the trend in monthly sales?"
            ],
            capabilities=capabilities
        )
    
    def _show_bi_summary(self):
        """Show BI-aware analysis summary"""
        # Count classifications
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        dimension_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'dimension'])
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        planning_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'planning'])
        
        # Count capabilities
        with_measures = len([t for t in self.tables if getattr(t, 'measures', [])])
        with_time = len([t for t in self.tables if getattr(t, 'time_columns', [])])
        
        print(f"\n📊 BI-AWARE ANALYSIS SUMMARY:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   📊 Fact tables: {fact_tables}")
        print(f"   📚 Dimension tables: {dimension_tables}")
        print(f"   ⚡ Operational data: {operational_tables}")
        print(f"   📋 Planning data: {planning_tables}")
        print(f"   📈 Tables with measures: {with_measures}")
        print(f"   ⏰ Tables with time columns: {with_time}")
        print(f"   🔗 BI relationships: {len(self.relationships)}")
        
        if self.domain:
            print(f"   🎯 BI Domain: {self.domain.domain_type} (confidence: {self.domain.confidence:.2f})")
        
        print(f"   ✅ Capability contracts ready for validation")
        print(f"   🚫 No-fallback mode: Only validated queries will execute")
    
    # Public interface - enhanced for BI
    def assess_query_capability(self, question: str) -> Union[List[TableInfo], NonExecutableAnalysisReport]:
        """Assess if query can be executed based on capability contracts"""
        if not self.evidence_scorer:
            return NonExecutableAnalysisReport(
                question=question,
                normalized_task=AnalyticalTask(task_type="unknown"),
                missing_capabilities=["System not properly initialized"],
                top_candidate_tables=[],
                fix_paths=["Run BI-aware analysis first"]
            )
        
        # Parse intent (simplified for now)
        intent = self._parse_intent(question)
        
        # Rank candidates by evidence
        candidates = self.evidence_scorer.rank_candidates(intent, top_k=5)
        
        # Check capability contracts
        valid_tables = []
        capability_failures = []
        
        for table, score in candidates:
            contract = self.capability_analyzer.assess_capability_contract(table, intent)
            
            if contract.is_complete():
                valid_tables.append(table)
            else:
                capability_failures.append((table, contract))
        
        # Return valid tables or NER
        if valid_tables:
            print(f"✅ Found {len(valid_tables)} tables satisfying capability contract")
            return valid_tables
        else:
            print(f"❌ No tables satisfy capability contract - generating NER")
            return self.ner_generator.generate_ner(question, intent, capability_failures, candidates)
    
    def _parse_intent(self, question: str) -> AnalyticalTask:
        """Parse user question into analytical task (simplified)"""
        q_lower = question.lower()
        
        # Detect task type
        if any(word in q_lower for word in ['how many', 'count', 'number']):
            task_type = 'aggregation'
            metrics = ['count']
        elif any(word in q_lower for word in ['total', 'sum']):
            task_type = 'aggregation'
            metrics = ['sum']
        elif any(word in q_lower for word in ['top', 'highest', 'best']):
            task_type = 'ranking'
            metrics = ['value']
        else:
            task_type = 'aggregation'
            metrics = ['count']
        
        # Detect entity
        entity = None
        if any(word in q_lower for word in ['customer', 'client']):
            entity = 'customer'
        elif any(word in q_lower for word in ['revenue', 'payment', 'money']):
            entity = 'revenue'
        
        return AnalyticalTask(
            task_type=task_type,
            metrics=metrics,
            entity=entity,
            time_window='current'
        )
    
    def get_tables(self) -> List[TableInfo]:
        """Get analyzed tables"""
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        """Get BI-enhanced relationships"""
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        """Get BI-aware business domain"""
        return self.domain
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get BI-aware analysis statistics"""
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        with_measures = len([t for t in self.tables if getattr(t, 'measures', [])])
        
        return {
            'total_tables': len(self.tables),
            'fact_tables': fact_tables,
            'operational_tables': operational_tables,
            'tables_with_measures': with_measures,
            'total_relationships': len(self.relationships),
            'domain_type': self.domain.domain_type if self.domain else None,
            'bi_ready': fact_tables > 0 and operational_tables > 0,
            'no_fallback_mode': True
        }

# Maintain compatibility with existing interface
SemanticAnalyzer = BISemanticAnalyzer