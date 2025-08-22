#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Analysis - Simple, Readable, Maintainable
Following README: Capability contracts, evidence-driven selection
DRY, SOLID, YAGNI principles
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import (TableInfo, BusinessDomain, Relationship, 
                          AnalyticalTask, EvidenceScore, CapabilityContract)
from shared.utils import parse_json_response

class LLMAnalyzer:
    """Simple LLM communication"""
    
    def __init__(self, config: Config):
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

class TableClassifier:
    """Simple table classification for BI awareness"""
    
    def __init__(self, llm_analyzer: LLMAnalyzer):
        self.llm_analyzer = llm_analyzer
    
    async def classify_tables(self, tables: List[TableInfo]) -> int:
        """Classify tables with BI awareness"""
        print(f"🏷️ Classifying {len(tables)} tables...")
        
        batch_size = 6
        classified = 0
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            
            # Create prompt
            prompt = self._create_prompt(batch)
            
            # Get LLM response
            response = await self.llm_analyzer.analyze(
                "You are a BI analyst. Classify database tables. Respond with valid JSON only.",
                prompt
            )
            
            # Apply classifications
            classified += self._apply_classifications(response, batch)
            await asyncio.sleep(0.5)  # Rate limiting
        
        print(f"   ✅ Classified {classified} tables")
        return classified
    
    def _create_prompt(self, tables: List[TableInfo]) -> str:
        """Create classification prompt"""
        table_summaries = []
        for table in tables:
            # Analyze columns for patterns
            column_analysis = self._analyze_columns(table)
            
            table_summaries.append({
                'table_name': table.full_name,
                'columns': column_analysis,
                'row_count': table.row_count,
                'sample': self._get_sample_preview(table)
            })
        
        return f"""
Analyze these tables for BUSINESS INTELLIGENCE:

1. DATA TYPE: operational (transactions), planning (targets), reference (lookups)
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
    
    def _analyze_columns(self, table: TableInfo) -> Dict[str, List[str]]:
        """Analyze columns for BI patterns"""
        analysis = {'measures': [], 'entity_keys': [], 'time_columns': [], 'filter_columns': []}
        
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

class DomainAnalyzer:
    """Simple business domain analysis"""
    
    def determine_domain(self, tables: List[TableInfo]) -> Optional[BusinessDomain]:
        """Determine business domain from patterns"""
        fact_tables = len([t for t in tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in tables if getattr(t, 'data_type', '') == 'operational'])
        
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
                'customer_analysis': any('customer' in t.entity_type.lower() for t in tables),
                'financial_analysis': any('payment' in t.entity_type.lower() for t in tables),
                'operational_reporting': operational_tables > 0
            }
        )

class CacheManager:
    """Simple cache management"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_cache(self, tables: List[TableInfo], domain: Optional[BusinessDomain], 
                   relationships: List[Relationship]):
        """Save analysis results"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'tables': [self._table_to_dict(t) for t in tables],
            'domain': self._domain_to_dict(domain) if domain else None,
            'relationships': [self._relationship_to_dict(r) for r in relationships],
            'analyzed': datetime.now().isoformat(),
            'version': '2.0-simple'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Saved analysis cache")
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def load_cache(self) -> Tuple[List[TableInfo], Optional[BusinessDomain], List[Relationship]]:
        """Load analysis from cache"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        if not cache_file.exists():
            return [], None, []
        
        try:
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.config.semantic_cache_hours * 3600):
                return [], None, []
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tables = [self._dict_to_table(t) for t in data.get('tables', [])]
            domain = self._dict_to_domain(data.get('domain')) if data.get('domain') else None
            relationships = [self._dict_to_relationship(r) for r in data.get('relationships', [])]
            
            return tables, domain, relationships
        except Exception:
            return [], None, []
    
    def _table_to_dict(self, table: TableInfo) -> Dict:
        """Convert TableInfo to dict"""
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
        """Convert dict to TableInfo"""
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
        """Convert BusinessDomain to dict"""
        return {
            'domain_type': domain.domain_type,
            'industry': domain.industry,
            'confidence': domain.confidence,
            'sample_questions': domain.sample_questions,
            'capabilities': domain.capabilities
        }
    
    def _dict_to_domain(self, data: Dict) -> BusinessDomain:
        """Convert dict to BusinessDomain"""
        return BusinessDomain(
            domain_type=data['domain_type'],
            industry=data['industry'],
            confidence=data['confidence'],
            sample_questions=data['sample_questions'],
            capabilities=data.get('capabilities', {})
        )
    
    def _relationship_to_dict(self, rel: Relationship) -> Dict:
        """Convert Relationship to dict"""
        return {
            'from_table': rel.from_table,
            'to_table': rel.to_table,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'description': rel.description
        }
    
    def _dict_to_relationship(self, data: Dict) -> Relationship:
        """Convert dict to Relationship"""
        return Relationship(
            from_table=data['from_table'],
            to_table=data['to_table'],
            relationship_type=data['relationship_type'],
            confidence=data['confidence'],
            description=data.get('description', '')
        )

class SemanticAnalyzer:
    """Main semantic analyzer - Simple and maintainable"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_analyzer = LLMAnalyzer(config)
        self.table_classifier = TableClassifier(self.llm_analyzer)
        self.domain_analyzer = DomainAnalyzer()
        self.cache_manager = CacheManager(config)
        
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """Main analysis with BI awareness"""
        print("🧠 Semantic Analysis")
        print("Following capability contracts and evidence-driven selection")
        
        try:
            # Copy input tables
            self.tables = [table for table in tables]
            
            # Classify tables
            await self.table_classifier.classify_tables(self.tables)
            
            # Enhance relationships
            self.relationships = self._enhance_relationships()
            
            # Determine domain
            self.domain = self.domain_analyzer.determine_domain(self.tables)
            
            # Save cache
            self.cache_manager.save_cache(self.tables, self.domain, self.relationships)
            
            # Show summary
            self._show_summary()
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
    
    def load_from_cache(self) -> bool:
        """Load from cache"""
        tables, domain, relationships = self.cache_manager.load_cache()
        if tables:
            self.tables = tables
            self.domain = domain
            self.relationships = relationships
            return True
        return False
    
    def _enhance_relationships(self) -> List[Relationship]:
        """Extract relationships from foreign keys"""
        relationships = []
        
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
    
    def _show_summary(self):
        """Show analysis summary"""
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        with_measures = len([t for t in self.tables if getattr(t, 'measures', [])])
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   📊 Fact tables: {fact_tables}")
        print(f"   ⚡ Operational tables: {operational_tables}")
        print(f"   📈 Tables with measures: {with_measures}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        if self.domain:
            print(f"   🎯 Domain: {self.domain.domain_type}")
    
    # Public interface - Clean API
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
            'domain_type': self.domain.domain_type if self.domain else None
        }