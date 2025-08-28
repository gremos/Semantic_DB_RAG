#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Analysis - Pure LLM Approach
Simple, Readable, Maintainable - Following DRY, SOLID, YAGNI
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
from shared.models import TableInfo, BusinessDomain, Relationship
from shared.utils import parse_json_response

class LLMTableAnalyzer:
    """Pure LLM-based table analysis - Single Responsibility"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_tables(self, tables: List[TableInfo]) -> List[TableInfo]:
        """Analyze tables using pure LLM approach"""
        print(f"🧠 Analyzing {len(tables)} tables with pure LLM...")
        
        # Process in batches for efficiency
        batch_size = 3
        analyzed_tables = []
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            batch_result = await self._analyze_batch(batch)
            analyzed_tables.extend(batch_result)
            
            if i + batch_size < len(tables):
                await asyncio.sleep(0.5)  # Rate limiting
        
        print(f"   ✅ Analyzed {len(analyzed_tables)} tables")
        return analyzed_tables
    
    async def _analyze_batch(self, tables: List[TableInfo]) -> List[TableInfo]:
        """Analyze a batch of tables with LLM"""
        try:
            prompt = self._create_analysis_prompt(tables)
            response = await self._get_llm_response(prompt)
            return self._apply_analysis(response, tables)
        except Exception as e:
            print(f"   ⚠️ Batch analysis failed: {e}")
            return tables  # Return original tables if analysis fails
    
    def _create_analysis_prompt(self, tables: List[TableInfo]) -> str:
        """Create focused analysis prompt for LLM"""
        table_info = []
        
        for table in tables:
            # Get essential table information
            columns = [col.get('name', '') for col in table.columns[:10]]  # First 10 columns
            sample = self._get_sample_preview(table)
            
            table_info.append({
                'name': table.full_name,
                'row_count': table.row_count,
                'columns': columns,
                'sample': sample
            })
        
        return f"""Analyze these database tables for business intelligence use.

TABLES:
{json.dumps(table_info, indent=2)}

For each table, determine:
1. ENTITY_TYPE: Customer, Payment, Order, Product, Campaign, or describe the business entity
2. BI_ROLE: fact (contains measures/transactions) or dimension (contains attributes)
3. MEASURES: numeric columns suitable for aggregation (amounts, quantities, counts)
4. ENTITY_KEYS: key columns for joining/grouping (IDs, foreign keys)
5. NAME_COLUMNS: columns with names/titles for display
6. TIME_COLUMNS: date/time columns for filtering
7. BUSINESS_PRIORITY: high (customer/payment data), medium, or low

Respond with JSON only:
{{
  "tables": [
    {{
      "table_name": "[schema].[table]",
      "entity_type": "Customer",
      "bi_role": "dimension",
      "measures": ["amount", "quantity"],
      "entity_keys": ["customer_id", "id"],
      "name_columns": ["name", "title"],
      "time_columns": ["created_date", "modified_date"],
      "business_priority": "high"
    }}
  ]
}}"""
    
    def _get_sample_preview(self, table: TableInfo) -> str:
        """Get simple sample preview"""
        if not table.sample_data:
            return "No sample data"
        
        first_row = table.sample_data[0]
        preview = []
        
        for key, value in list(first_row.items())[:4]:
            if not key.startswith('__'):
                preview.append(f"{key}={value}")
        
        return " | ".join(preview)
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response with error handling"""
        try:
            system_msg = "You are a business intelligence expert. Analyze database tables and respond with valid JSON only."
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"      ⚠️ LLM request failed: {e}")
            return ""
    
    def _apply_analysis(self, response: str, tables: List[TableInfo]) -> List[TableInfo]:
        """Apply LLM analysis results to tables"""
        data = parse_json_response(response)
        if not data or 'tables' not in data:
            return tables
        
        # Create lookup for easy matching
        analysis_map = {
            item.get('table_name', ''): item 
            for item in data['tables']
        }
        
        # Apply analysis to tables
        for table in tables:
            analysis = analysis_map.get(table.full_name, {})
            if analysis:
                # Update table properties
                table.entity_type = analysis.get('entity_type', 'Unknown')
                table.bi_role = analysis.get('bi_role', 'dimension')
                table.measures = analysis.get('measures', [])
                table.entity_keys = analysis.get('entity_keys', [])
                table.name_columns = analysis.get('name_columns', [])
                table.time_columns = analysis.get('time_columns', [])
                table.business_priority = analysis.get('business_priority', 'medium')
                table.confidence = 0.9  # High confidence from LLM analysis
                
                # Set data type based on BI role
                table.data_type = 'operational' if table.bi_role == 'fact' else 'reference'
                table.grain = self._determine_grain(table.entity_type)
        
        return tables
    
    def _determine_grain(self, entity_type: str) -> str:
        """Determine table grain from entity type"""
        grain_map = {
            'Customer': 'customer',
            'Payment': 'transaction',
            'Order': 'order',
            'Product': 'product',
            'Campaign': 'campaign'
        }
        return grain_map.get(entity_type, entity_type.lower())

class DomainAnalyzer:
    """Simple domain analysis - Single Responsibility"""
    
    def determine_domain(self, tables: List[TableInfo]) -> Optional[BusinessDomain]:
        """Determine business domain from analyzed tables"""
        if not tables:
            return None
        
        # Count entity types
        entity_counts = {}
        for table in tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Determine domain based on entity mix
        customer_tables = entity_counts.get('Customer', 0)
        payment_tables = entity_counts.get('Payment', 0)
        
        if customer_tables >= 1 and payment_tables >= 1:
            return BusinessDomain(
                domain_type="Customer Analytics",
                industry="Business Intelligence",
                confidence=0.9,
                sample_questions=[
                    "Who are our top paying customers?",
                    "What is our total revenue?",
                    "Show customer payment trends"
                ],
                capabilities={
                    'customer_analysis': True,
                    'payment_analysis': True,
                    'revenue_reporting': True
                }
            )
        else:
            return BusinessDomain(
                domain_type="Business Analytics",
                industry="General",
                confidence=0.7,
                sample_questions=[
                    "Show me the data summary",
                    "What information is available?"
                ],
                capabilities={
                    'basic_reporting': True,
                    'data_exploration': True
                }
            )

class CacheManager:
    """Simple cache management - Single Responsibility"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_analysis(self, tables: List[TableInfo], domain: Optional[BusinessDomain], 
                     relationships: List[Relationship]):
        """Save analysis results to cache"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'version': '3.0-pure-llm',
                'table_count': len(tables)
            },
            'tables': [self._table_to_dict(t) for t in tables],
            'domain': self._domain_to_dict(domain) if domain else None,
            'relationships': [self._relationship_to_dict(r) for r in relationships]
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Analysis cached successfully")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def load_analysis(self) -> Tuple[List[TableInfo], Optional[BusinessDomain], List[Relationship]]:
        """Load analysis from cache"""
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
        except Exception:
            return [], None, []
    
    def _table_to_dict(self, table: TableInfo) -> Dict:
        """Convert TableInfo to dictionary"""
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
            'bi_role': getattr(table, 'bi_role', 'dimension'),
            'measures': getattr(table, 'measures', []),
            'entity_keys': getattr(table, 'entity_keys', []),
            'name_columns': getattr(table, 'name_columns', []),
            'time_columns': getattr(table, 'time_columns', []),
            'business_priority': getattr(table, 'business_priority', 'medium'),
            'confidence': table.confidence
        }
    
    def _dict_to_table(self, data: Dict) -> TableInfo:
        """Convert dictionary to TableInfo"""
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
        
        # Apply LLM analysis results
        table.bi_role = data.get('bi_role', 'dimension')
        table.measures = data.get('measures', [])
        table.entity_keys = data.get('entity_keys', [])
        table.name_columns = data.get('name_columns', [])
        table.time_columns = data.get('time_columns', [])
        table.business_priority = data.get('business_priority', 'medium')
        
        return table
    
    def _domain_to_dict(self, domain: BusinessDomain) -> Dict:
        """Convert BusinessDomain to dictionary"""
        return {
            'domain_type': domain.domain_type,
            'industry': domain.industry,
            'confidence': domain.confidence,
            'sample_questions': domain.sample_questions,
            'capabilities': domain.capabilities
        }
    
    def _dict_to_domain(self, data: Dict) -> BusinessDomain:
        """Convert dictionary to BusinessDomain"""
        return BusinessDomain(
            domain_type=data['domain_type'],
            industry=data['industry'],
            confidence=data['confidence'],
            sample_questions=data['sample_questions'],
            capabilities=data.get('capabilities', {})
        )
    
    def _relationship_to_dict(self, rel: Relationship) -> Dict:
        """Convert Relationship to dictionary"""
        return {
            'from_table': rel.from_table,
            'to_table': rel.to_table,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'description': rel.description
        }
    
    def _dict_to_relationship(self, data: Dict) -> Relationship:
        """Convert dictionary to Relationship"""
        return Relationship(
            from_table=data['from_table'],
            to_table=data['to_table'],
            relationship_type=data['relationship_type'],
            confidence=data['confidence'],
            description=data.get('description', '')
        )

class SemanticAnalyzer:
    """Main semantic analyzer - Pure LLM approach"""
    
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
        self.table_analyzer = LLMTableAnalyzer(self.llm)
        self.domain_analyzer = DomainAnalyzer()
        self.cache_manager = CacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        
        print("✅ Pure LLM Semantic Analyzer initialized")
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """Main analysis method using pure LLM approach"""
        print("🧠 Pure LLM Semantic Analysis")
        print("=" * 40)
        
        try:
            start_time = time.time()
            
            # Pure LLM table analysis
            self.tables = await self.table_analyzer.analyze_tables(tables)
            
            # Simple domain analysis
            self.domain = self.domain_analyzer.determine_domain(self.tables)
            
            # Extract relationships from table metadata
            self.relationships = self._extract_relationships()
            
            # Save to cache
            self.cache_manager.save_analysis(self.tables, self.domain, self.relationships)
            
            # Show summary
            self._show_summary(time.time() - start_time)
            
            return True
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
    
    def load_from_cache(self) -> bool:
        """Load analysis from cache"""
        tables, domain, relationships = self.cache_manager.load_analysis()
        if tables:
            self.tables = tables
            self.domain = domain
            self.relationships = relationships
            return True
        return False
    
    def _extract_relationships(self) -> List[Relationship]:
        """Extract relationships from table foreign keys"""
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
                            confidence=0.9,
                            description=f"FK: {from_col} -> {to_ref}"
                        ))
                    except Exception:
                        continue
        
        return relationships
    
    def _show_summary(self, elapsed_time: float):
        """Show analysis summary"""
        customer_tables = len([t for t in self.tables if t.entity_type == 'Customer'])
        payment_tables = len([t for t in self.tables if t.entity_type == 'Payment'])
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        tables_with_names = len([t for t in self.tables if getattr(t, 'name_columns', [])])
        
        print(f"\n✅ PURE LLM ANALYSIS COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Total tables: {len(self.tables)}")
        print(f"   👥 Customer tables: {customer_tables}")
        print(f"   💳 Payment tables: {payment_tables}")
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   📝 Tables with names: {tables_with_names}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        if self.domain:
            print(f"   🎯 Domain: {self.domain.domain_type}")
    
    # Public API
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        customer_tables = len([t for t in self.tables if t.entity_type == 'Customer'])
        payment_tables = len([t for t in self.tables if t.entity_type == 'Payment'])
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        
        return {
            'total_tables': len(self.tables),
            'customer_tables': customer_tables,
            'payment_tables': payment_tables,
            'fact_tables': fact_tables,
            'total_relationships': len(self.relationships),
            'domain_type': self.domain.domain_type if self.domain else None,
            'analysis_method': 'pure_llm'
        }