#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BI-Aware Semantic Analysis - Simple, Readable, Maintainable
Following README: Capability contracts, evidence-driven selection, NER
DRY, SOLID, YAGNI principles with clean cache support
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import (TableInfo, BusinessDomain, Relationship, NonExecutableAnalysisReport, 
                          AnalyticalTask, EvidenceScore, CapabilityContract)
from shared.utils import parse_json_response

class CacheManager:
    """Simple cache management for semantic analysis - Single responsibility (SOLID)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_analysis_cache(self, tables: List[TableInfo], domain: Optional[BusinessDomain], 
                           relationships: List[Relationship]):
        """Save analysis results to cache"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'tables': [self._table_to_dict(t) for t in tables],
            'domain': self._domain_to_dict(domain) if domain else None,
            'relationships': [self._relationship_to_dict(r) for r in relationships],
            'analyzed': datetime.now().isoformat(),
            'version': '2.0-bi-aware'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Saved analysis to cache: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to save analysis cache: {e}")
    
    def load_analysis_cache(self) -> Tuple[List[TableInfo], Optional[BusinessDomain], List[Relationship]]:
        """Load analysis results from cache if fresh"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        if not cache_file.exists():
            return [], None, []
        
        try:
            # Check cache age
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.config.semantic_cache_hours * 3600):
                return [], None, []
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tables = [self._dict_to_table(t) for t in data.get('tables', [])]
            domain = self._dict_to_domain(data.get('domain')) if data.get('domain') else None
            relationships = [self._dict_to_relationship(r) for r in data.get('relationships', [])]
            
            return tables, domain, relationships
            
        except Exception as e:
            print(f"   ⚠️ Failed to load analysis cache: {e}")
            return [], None, []
    
    def _table_to_dict(self, table: TableInfo) -> Dict:
        """Convert TableInfo to dictionary for cache"""
        return {
            'name': table.name,
            'schema': table.schema,
            'full_name': table.full_name,
            'object_type': table.object_type,
            'row_count': table.row_count,
            'columns': table.columns,
            'sample_data': table.sample_data,
            'relationships': table.relationships,
            'entity_type': table.entity_type,
            'business_role': table.business_role,
            'confidence': table.confidence,
            'data_type': getattr(table, 'data_type', 'reference'),
            'bi_role': getattr(table, 'bi_role', 'dimension'),
            'grain': getattr(table, 'grain', 'unknown'),
            'measures': getattr(table, 'measures', []),
            'entity_keys': getattr(table, 'entity_keys', []),
            'time_columns': getattr(table, 'time_columns', []),
            'filter_columns': getattr(table, 'filter_columns', [])
        }
    
    def _dict_to_table(self, data: Dict) -> TableInfo:
        """Convert dictionary to TableInfo from cache"""
        table = TableInfo(
            name=data['name'],
            schema=data['schema'],
            full_name=data['full_name'],
            object_type=data['object_type'],
            row_count=data['row_count'],
            columns=data['columns'],
            sample_data=data['sample_data'],
            relationships=data.get('relationships', []),
            entity_type=data.get('entity_type', 'Unknown'),
            business_role=data.get('business_role', 'Unknown'),
            confidence=data.get('confidence', 0.0)
        )
        
        # BI-aware properties
        table.data_type = data.get('data_type', 'reference')
        table.bi_role = data.get('bi_role', 'dimension')
        table.grain = data.get('grain', 'unknown')
        table.measures = data.get('measures', [])
        table.entity_keys = data.get('entity_keys', [])
        table.time_columns = data.get('time_columns', [])
        table.filter_columns = data.get('filter_columns', [])
        
        return table
    
    def _domain_to_dict(self, domain: BusinessDomain) -> Dict:
        """Convert BusinessDomain to dictionary"""
        return {
            'domain_type': domain.domain_type,
            'industry': domain.industry,
            'confidence': domain.confidence,
            'sample_questions': domain.sample_questions,
            'capabilities': domain.capabilities,
            'bi_maturity': getattr(domain, 'bi_maturity', 'Basic'),
            'analytical_patterns': getattr(domain, 'analytical_patterns', []),
            'data_quality_score': getattr(domain, 'data_quality_score', 0.0)
        }
    
    def _dict_to_domain(self, data: Dict) -> BusinessDomain:
        """Convert dictionary to BusinessDomain"""
        domain = BusinessDomain(
            domain_type=data['domain_type'],
            industry=data['industry'],
            confidence=data['confidence'],
            sample_questions=data['sample_questions'],
            capabilities=data.get('capabilities', {})
        )
        
        domain.bi_maturity = data.get('bi_maturity', 'Basic')
        domain.analytical_patterns = data.get('analytical_patterns', [])
        domain.data_quality_score = data.get('data_quality_score', 0.0)
        
        return domain
    
    def _relationship_to_dict(self, rel: Relationship) -> Dict:
        """Convert Relationship to dictionary"""
        return {
            'from_table': rel.from_table,
            'to_table': rel.to_table,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'description': rel.description,
            'cardinality': getattr(rel, 'cardinality', 'unknown'),
            'join_strength': getattr(rel, 'join_strength', 'weak'),
            'bi_pattern': getattr(rel, 'bi_pattern', 'unknown')
        }
    
    def _dict_to_relationship(self, data: Dict) -> Relationship:
        """Convert dictionary to Relationship"""
        rel = Relationship(
            from_table=data['from_table'],
            to_table=data['to_table'],
            relationship_type=data['relationship_type'],
            confidence=data['confidence'],
            description=data.get('description', '')
        )
        
        rel.cardinality = data.get('cardinality', 'unknown')
        rel.join_strength = data.get('join_strength', 'weak')
        rel.bi_pattern = data.get('bi_pattern', 'unknown')
        
        return rel

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

class BITableClassifier:
    """BI-aware table classification - Simple and effective"""
    
    def __init__(self, llm_analyzer: LLMAnalyzer):
        self.llm_analyzer = llm_analyzer
    
    async def classify_tables(self, tables: List[TableInfo]) -> int:
        """Classify tables with BI-aware categories"""
        print(f"🏷️ BI-aware classification of {len(tables)} tables...")
        
        batch_size = 6  # Process in small batches
        classified = 0
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            
            # Create classification prompt
            prompt = self._create_classification_prompt(batch)
            
            # Get LLM response
            response = await self.llm_analyzer.analyze(
                "You are a BI analyst. Classify database tables for business intelligence. Respond with valid JSON only.",
                prompt
            )
            
            # Apply classifications
            classified += self._apply_classifications(response, batch)
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        print(f"   ✅ BI-classified {classified} tables")
        return classified
    
    def _create_classification_prompt(self, tables: List[TableInfo]) -> str:
        """Create BI classification prompt"""
        
        table_summaries = []
        for table in tables:
            # Analyze columns for BI patterns
            column_analysis = self._analyze_columns(table)
            sample_preview = self._get_sample_preview(table)
            
            table_summaries.append({
                'table_name': table.full_name,
                'columns': column_analysis,
                'sample_data': sample_preview,
                'row_count': table.row_count
            })
        
        return f"""
Analyze these tables for BUSINESS INTELLIGENCE capabilities:

1. DATA TYPE: operational (real transactions), planning (targets), reference (lookups)
2. BI ROLE: fact (measures), dimension (attributes), bridge (relationships)
3. GRAIN: what each row represents
4. CAPABILITIES: measures, entity keys, time columns, filters

TABLES:
{json.dumps(table_summaries, indent=2)}

Respond with JSON only:
{{
  "classifications": [
    {{
      "table_name": "[schema].[table]",
      "data_type": "operational|planning|reference",
      "bi_role": "fact|dimension|bridge",
      "grain": "customer|transaction|order|product",
      "entity_type": "Customer|Payment|Order|Product",
      "measures": ["amount_col", "quantity_col"],
      "entity_keys": ["customer_id", "product_id"],
      "time_columns": ["date_col"],
      "filter_columns": ["status", "type"],
      "confidence": 0.9
    }}
  ]
}}
"""
    
    def _analyze_columns(self, table: TableInfo) -> Dict[str, Any]:
        """Analyze columns for BI capabilities"""
        analysis = {
            'measures': [],
            'entity_keys': [],
            'time_columns': [],
            'filter_columns': []
        }
        
        for col in table.columns:
            col_name = col.get('name', '').lower()
            col_type = col.get('data_type', '').lower()
            
            # Identify measures
            if any(t in col_type for t in ['decimal', 'money', 'float', 'numeric', 'int']):
                if any(w in col_name for w in ['amount', 'value', 'price', 'total', 'quantity']):
                    analysis['measures'].append(col.get('name'))
            
            # Identify entity keys
            if col_name.endswith('id') or 'key' in col_name:
                analysis['entity_keys'].append(col.get('name'))
            
            # Identify time columns
            if 'date' in col_type or 'time' in col_type or any(w in col_name for w in ['date', 'time', 'created']):
                analysis['time_columns'].append(col.get('name'))
            
            # Identify filter columns
            if any(w in col_name for w in ['status', 'type', 'category', 'region']):
                analysis['filter_columns'].append(col.get('name'))
        
        return analysis
    
    def _get_sample_preview(self, table: TableInfo) -> str:
        """Get sample data preview"""
        if not table.sample_data:
            return "No sample data"
        
        # Show first row with meaningful values
        first_row = table.sample_data[0]
        preview = []
        
        for key, value in list(first_row.items())[:4]:
            if not key.startswith('__'):
                preview.append(f"{key}={value}")
        
        return " | ".join(preview)
    
    def _apply_classifications(self, response: str, batch: List[TableInfo]) -> int:
        """Apply classifications to tables"""
        data = parse_json_response(response)
        if not data or 'classifications' not in data:
            return 0
        
        count = 0
        for classification in data['classifications']:
            table_name = classification.get('table_name', '')
            
            # Find and update table
            for table in batch:
                if table.full_name == table_name:
                    # Apply BI classifications
                    table.entity_type = classification.get('entity_type', 'Unknown')
                    table.confidence = float(classification.get('confidence', 0.0))
                    table.data_type = classification.get('data_type', 'reference')
                    table.bi_role = classification.get('bi_role', 'dimension')
                    table.grain = classification.get('grain', 'unknown')
                    table.measures = classification.get('measures', [])
                    table.entity_keys = classification.get('entity_keys', [])
                    table.time_columns = classification.get('time_columns', [])
                    table.filter_columns = classification.get('filter_columns', [])
                    count += 1
                    break
        
        return count

class CapabilityAnalyzer:
    """Analyze tables for BI capability contracts"""
    
    def assess_capability_contract(self, table: TableInfo, intent: AnalyticalTask) -> CapabilityContract:
        """Assess if table satisfies capability contract"""
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
            'measures_available': len(contract.measures) > 0,
            'time_available': contract.time_column is not None
        }
        
        return contract

class EvidenceScorer:
    """Evidence-driven object selection"""
    
    def __init__(self, tables: List[TableInfo]):
        self.tables = tables
    
    def score_table_for_intent(self, table: TableInfo, intent: AnalyticalTask) -> EvidenceScore:
        """Score table based on evidence for analytical intent"""
        score = EvidenceScore()
        
        # Role match (fact tables for aggregation)
        bi_role = getattr(table, 'bi_role', 'dimension')
        data_type = getattr(table, 'data_type', 'reference')
        
        if intent.task_type in ['aggregation', 'ranking'] and bi_role == 'fact' and data_type == 'operational':
            score.role_match = 1.0
        elif bi_role == 'fact':
            score.role_match = 0.8
        else:
            score.role_match = 0.3
        
        # Join evidence
        score.join_evidence = min(1.0, len(table.relationships) / 3.0)
        
        # Lexical match
        table_name = table.name.lower()
        if intent.entity and intent.entity.lower() in table_name:
            score.lexical_match = 1.0
        elif any(metric.lower() in table_name for metric in intent.metrics):
            score.lexical_match = 0.8
        else:
            score.lexical_match = 0.2
        
        # Operational preference
        score.operational_tag = 1.0 if data_type == 'operational' else 0.5
        
        # Row count
        score.row_count = min(1.0, table.row_count / 10000.0) if table.row_count > 0 else 0.0
        score.freshness = 1.0
        
        return score
    
    def rank_candidates(self, intent: AnalyticalTask, top_k: int = 5) -> List[Tuple[TableInfo, EvidenceScore]]:
        """Rank candidate tables by evidence score"""
        scored_tables = []
        
        for table in self.tables:
            score = self.score_table_for_intent(table, intent)
            scored_tables.append((table, score))
        
        scored_tables.sort(key=lambda x: x[1].total_score, reverse=True)
        return scored_tables[:top_k]

class NERGenerator:
    """Generate Non-Executable Analysis Reports"""
    
    def generate_ner(self, question: str, intent: AnalyticalTask, 
                    capability_failures: List[Tuple[TableInfo, CapabilityContract]],
                    top_candidates: List[Tuple[TableInfo, EvidenceScore]]) -> NonExecutableAnalysisReport:
        """Generate NER when capability checks fail"""
        
        # Collect missing capabilities
        all_missing = set()
        for table, contract in capability_failures:
            all_missing.update(contract.get_missing_capabilities())
        
        # Generate fix paths
        fix_paths = []
        for table, contract in capability_failures[:3]:
            missing = contract.get_missing_capabilities()
            if "Numeric measures" in missing:
                numeric_cols = [col.get('name') for col in table.columns 
                              if 'int' in col.get('data_type', '').lower() or 'decimal' in col.get('data_type', '').lower()]
                if numeric_cols:
                    fix_paths.append(f"{table.full_name}: Use {', '.join(numeric_cols[:2])} as measures")
            
            if "Time/date column" in missing:
                date_cols = [col.get('name') for col in table.columns if 'date' in col.get('name', '').lower()]
                if date_cols:
                    fix_paths.append(f"{table.full_name}: Use {', '.join(date_cols[:1])} for time filtering")
        
        # Generate safe queries
        safe_queries = []
        for table, score in top_candidates[:2]:
            safe_queries.append(f"SELECT TOP 5 * FROM {table.full_name}")
        
        return NonExecutableAnalysisReport(
            question=question,
            normalized_task=intent.__dict__,
            missing_capabilities=list(all_missing),
            top_candidate_tables=[(t.full_name, s.total_score) for t, s in top_candidates],
            fix_paths=fix_paths,
            suggested_queries=safe_queries
        )

class SemanticAnalyzer:
    """BI-Aware Semantic Analyzer - Simple and maintainable"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.cache_manager = CacheManager(config)
        self.llm_analyzer = LLMAnalyzer(config)
        self.table_classifier = BITableClassifier(self.llm_analyzer)
        self.capability_analyzer = CapabilityAnalyzer()
        self.ner_generator = NERGenerator()
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        self.evidence_scorer: Optional[EvidenceScorer] = None
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """BI-aware analysis with capability assessment"""
        print("🧠 BI-Aware Semantic Analysis")
        print("Following capability contracts and evidence-driven selection")
        
        try:
            # Copy input tables
            self.tables = [table for table in tables]
            
            # BI-aware classification
            await self.table_classifier.classify_tables(self.tables)
            
            # Initialize evidence scorer
            self.evidence_scorer = EvidenceScorer(self.tables)
            
            # Enhance relationships
            self.relationships = self._enhance_relationships()
            
            # Determine business domain
            self.domain = self._determine_domain()
            
            # Save to cache
            self.cache_manager.save_analysis_cache(self.tables, self.domain, self.relationships)
            
            # Show summary
            self._show_summary()
            return True
            
        except Exception as e:
            print(f"❌ BI-aware analysis failed: {e}")
            return False
    
    def load_from_cache(self) -> bool:
        """Load analysis from cache"""
        tables, domain, relationships = self.cache_manager.load_analysis_cache()
        if tables:
            self.tables = tables
            self.domain = domain
            self.relationships = relationships
            self.evidence_scorer = EvidenceScorer(self.tables)
            return True
        return False
    
    def _enhance_relationships(self) -> List[Relationship]:
        """Enhance relationships with BI awareness"""
        relationships = []
        
        # Extract FK relationships
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
        
        return relationships
    
    def _determine_domain(self) -> Optional[BusinessDomain]:
        """Determine business domain from BI patterns"""
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        
        if fact_tables >= 3 and operational_tables >= 5:
            domain_type = "Operational BI System"
            confidence = 0.9
        else:
            domain_type = "Business System"
            confidence = 0.6
        
        return BusinessDomain(
            domain_type=domain_type,
            industry="Business Intelligence",
            confidence=confidence,
            sample_questions=[
                "What is our total revenue this quarter?",
                "Who are our top 10 customers?",
                "How many active customers do we have?"
            ],
            capabilities={
                'customer_analysis': any('customer' in t.entity_type.lower() for t in self.tables),
                'financial_analysis': any('payment' in t.entity_type.lower() for t in self.tables),
                'operational_reporting': operational_tables > 0
            }
        )
    
    def _show_summary(self):
        """Show BI analysis summary"""
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        with_measures = len([t for t in self.tables if getattr(t, 'measures', [])])
        
        print(f"\n📊 BI-AWARE ANALYSIS SUMMARY:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   📊 Fact tables: {fact_tables}")
        print(f"   ⚡ Operational tables: {operational_tables}")
        print(f"   📈 Tables with measures: {with_measures}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        if self.domain:
            print(f"   🎯 Domain: {self.domain.domain_type}")
        
        print(f"   ✅ Ready for capability validation")
    
    def assess_query_capability(self, question: str) -> Union[List[TableInfo], NonExecutableAnalysisReport]:
        """Assess if query can be executed based on capability contracts"""
        if not self.evidence_scorer:
            return NonExecutableAnalysisReport(
                question=question,
                normalized_task={"task_type": "unknown"},
                missing_capabilities=["System not initialized"],
                top_candidate_tables=[],
                fix_paths=["Run analysis first"]
            )
        
        # Parse intent (simplified)
        intent = self._parse_intent(question)
        
        # Rank candidates
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
            return valid_tables
        else:
            return self.ner_generator.generate_ner(question, intent, capability_failures, candidates)
    
    def _parse_intent(self, question: str) -> AnalyticalTask:
        """Parse question into analytical task"""
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['how many', 'count']):
            task_type = 'aggregation'
            metrics = ['count']
        elif any(word in q_lower for word in ['total', 'sum']):
            task_type = 'aggregation'
            metrics = ['sum']
        elif any(word in q_lower for word in ['top', 'highest']):
            task_type = 'ranking'
            metrics = ['value']
        else:
            task_type = 'aggregation'
            metrics = ['count']
        
        entity = None
        if any(word in q_lower for word in ['customer', 'client']):
            entity = 'customer'
        elif any(word in q_lower for word in ['revenue', 'payment']):
            entity = 'revenue'
        
        return AnalyticalTask(
            task_type=task_type,
            metrics=metrics,
            entity=entity
        )
    
    # Public interface - clean API
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        
        return {
            'total_tables': len(self.tables),
            'fact_tables': fact_tables,
            'operational_tables': operational_tables,
            'total_relationships': len(self.relationships),
            'domain_type': self.domain.domain_type if self.domain else None,
            'bi_ready': fact_tables > 0 and operational_tables > 0
        }