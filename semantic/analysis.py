#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Analysis - Pure LLM Approach
Simple, Readable, Maintainable - Following DRY, SOLID, YAGNI
Enhanced to analyze tables+columns together and include views
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
    """Pure LLM-based table analysis with enhanced column understanding"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_tables_and_views(self, tables: List[TableInfo], view_info: Dict = None) -> List[TableInfo]:
        """Analyze tables and views together using pure LLM approach"""
        print(f"🧠 Analyzing {len(tables)} tables + views with enhanced column analysis...")
        
        # Combine tables and views for comprehensive analysis
        all_objects = []
        
        # Add tables
        for table in tables:
            all_objects.append({
                'object': table,
                'type': 'table',
                'view_definition': None
            })
        
        # Add views if available
        if view_info:
            for view_name, view_data in view_info.items():
                # Create TableInfo for views
                view_table = self._create_view_table_info(view_name, view_data)
                if view_table:
                    all_objects.append({
                        'object': view_table,
                        'type': 'view', 
                        'view_definition': view_data.get('definition', '')
                    })
        
        print(f"   📊 Total objects to analyze: {len(all_objects)} (tables + views)")
        
        # Process in batches for efficiency
        batch_size = 2  # Smaller batches for more detailed analysis
        analyzed_objects = []
        
        for i in range(0, len(all_objects), batch_size):
            batch = all_objects[i:i+batch_size]
            batch_result = await self._analyze_enhanced_batch(batch)
            analyzed_objects.extend(batch_result)
            
            if i + batch_size < len(all_objects):
                await asyncio.sleep(0.7)  # Rate limiting
        
        print(f"   ✅ Enhanced analysis completed: {len(analyzed_objects)} objects")
        return [obj['object'] for obj in analyzed_objects]
    
    def _create_view_table_info(self, view_name: str, view_data: Dict) -> Optional[TableInfo]:
        """Convert view info to TableInfo for analysis"""
        try:
            # Extract schema and name
            if '.' in view_name:
                parts = view_name.replace('[', '').replace(']', '').split('.')
                schema = parts[0] if len(parts) > 1 else 'dbo'
                name = parts[1] if len(parts) > 1 else parts[0]
            else:
                schema = 'dbo'
                name = view_name
            
            # Create basic view TableInfo
            return TableInfo(
                name=name,
                schema=schema,
                full_name=view_name,
                object_type='VIEW',
                row_count=0,  # Views don't have direct row counts
                columns=[],   # Will be populated by analysis
                sample_data=[],
                relationships=[]
            )
        except Exception:
            return None
    
    async def _analyze_enhanced_batch(self, batch: List[Dict]) -> List[Dict]:
        """Enhanced batch analysis with deep column understanding"""
        try:
            prompt = self._create_enhanced_analysis_prompt(batch)
            response = await self._get_llm_response(prompt)
            return self._apply_enhanced_analysis(response, batch)
        except Exception as e:
            print(f"   ⚠️ Enhanced batch analysis failed: {e}")
            return batch  # Return original if analysis fails
    
    def _create_enhanced_analysis_prompt(self, batch: List[Dict]) -> str:
        """Create enhanced analysis prompt focusing on table+column relationships"""
        objects_info = []
        
        for item in batch:
            obj = item['object']
            obj_type = item['type']
            view_def = item.get('view_definition')
            
            # Get detailed column information
            columns_detail = self._get_detailed_columns(obj)
            sample_detail = self._get_enhanced_sample(obj)
            
            obj_info = {
                'name': obj.full_name,
                'type': obj_type,
                'row_count': obj.row_count,
                'columns_with_types': columns_detail,
                'sample_data_analysis': sample_detail,
                'relationships': obj.relationships
            }
            
            # Add view definition if available
            if view_def:
                obj_info['view_definition'] = view_def[:500]  # Limit length
            
            objects_info.append(obj_info)
        
        return f"""Analyze these database objects (tables/views) for business intelligence.
Focus on understanding what business data each object contains by analyzing columns + sample data together.

OBJECTS TO ANALYZE:
{json.dumps(objects_info, indent=2, default=str)}

For each object, provide detailed analysis:

1. ENTITY_TYPE: What business entity does this represent?
   - Customer: customer master data, accounts, contacts
   - Payment: transactions, payments, invoices, financial data  
   - Order: sales orders, purchases, transactions
   - Product: items, catalog, inventory
   - Campaign: marketing campaigns, promotions
   - Address: location/geography data
   - System: logging, configuration, technical tables
   - Other: describe the specific business entity

2. BUSINESS_CONTEXT: What specific business information does this contain?
   (e.g., "Customer contact information with addresses", "Payment transactions with amounts")

3. COLUMN_ANALYSIS: Analyze columns by purpose:
   - MEASURES: numeric columns for aggregation (amounts, quantities, totals, counts)
   - ENTITY_KEYS: primary keys, IDs, foreign keys for joining
   - NAME_COLUMNS: descriptive text for display (names, titles, descriptions)  
   - TIME_COLUMNS: date/time columns for filtering/grouping
   - CATEGORY_COLUMNS: classification/grouping columns (status, type, category)

4. BI_ROLE: Based on column analysis:
   - fact: contains measures/transactions (quantitative data)
   - dimension: contains attributes/descriptions (qualitative data)

5. BUSINESS_PRIORITY: Based on business value:
   - high: core business entities (Customer, Payment, Order)
   - medium: supporting entities (Product, Address) 
   - low: system/technical entities

6. DATA_QUALITY: Based on sample data:
   - production: real business data
   - test: test/sample data
   - archive: historical/backup data

Respond with JSON only:
{{
  "analysis": [
    {{
      "object_name": "[schema].[table]",
      "entity_type": "Customer",
      "business_context": "Customer master data with contact information",
      "column_analysis": {{
        "measures": ["total_spent", "order_count"],
        "entity_keys": ["customer_id", "account_id"],
        "name_columns": ["customer_name", "company_name"],
        "time_columns": ["created_date", "last_order_date"],
        "category_columns": ["status", "customer_type"]
      }},
      "bi_role": "dimension",
      "business_priority": "high",
      "data_quality": "production"
    }}
  ]
}}"""
    
    def _get_detailed_columns(self, obj: TableInfo) -> List[Dict]:
        """Get detailed column information with types"""
        columns = []
        for col in obj.columns[:15]:  # First 15 columns
            col_info = {
                'name': col.get('name', ''),
                'type': col.get('data_type', ''),
                'nullable': col.get('is_nullable', True)
            }
            columns.append(col_info)
        return columns
    
    def _get_enhanced_sample(self, obj: TableInfo) -> Dict:
        """Get enhanced sample data analysis"""
        if not obj.sample_data:
            return {'status': 'no_sample_data', 'insights': []}
        
        insights = []
        first_row = obj.sample_data[0]
        
        # Analyze sample values
        for key, value in list(first_row.items())[:6]:
            if key.startswith('__'):
                continue
                
            value_type = type(value).__name__
            if isinstance(value, (int, float)) and abs(value) > 0:
                insights.append(f"{key}: numeric value ({value}) - potential measure")
            elif isinstance(value, str) and len(value) > 2:
                if any(word in key.lower() for word in ['name', 'title', 'description']):
                    insights.append(f"{key}: text value - likely name column")
                else:
                    insights.append(f"{key}: text value - potential category")
        
        return {
            'status': 'has_sample_data',
            'row_count': len(obj.sample_data),
            'insights': insights
        }
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response with error handling"""
        try:
            system_msg = """You are a business intelligence expert specializing in database analysis.
Analyze tables and views by understanding the relationship between column names, data types, and sample values.
Focus on identifying what business data each object contains. Respond with valid JSON only."""
            
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"      ⚠️ LLM request failed: {e}")
            return ""
    
    def _apply_enhanced_analysis(self, response: str, batch: List[Dict]) -> List[Dict]:
        """Apply enhanced LLM analysis results"""
        data = parse_json_response(response)
        if not data or 'analysis' not in data:
            print("      ⚠️ No valid analysis in LLM response")
            return batch
        
        # Create lookup for analysis results
        analysis_map = {
            item.get('object_name', ''): item 
            for item in data['analysis']
        }
        
        # Apply enhanced analysis
        for item in batch:
            obj = item['object']
            analysis = analysis_map.get(obj.full_name, {})
            
            if analysis:
                # Core properties
                obj.entity_type = analysis.get('entity_type', 'Unknown')
                obj.business_role = analysis.get('business_context', 'Unknown')
                obj.confidence = 0.9  # High confidence from LLM
                
                # Enhanced column analysis
                col_analysis = analysis.get('column_analysis', {})
                obj.measures = col_analysis.get('measures', [])
                obj.entity_keys = col_analysis.get('entity_keys', [])
                obj.name_columns = col_analysis.get('name_columns', [])
                obj.time_columns = col_analysis.get('time_columns', [])
                
                # Set category columns (new property)
                if hasattr(obj, '__dict__'):
                    obj.category_columns = col_analysis.get('category_columns', [])
                
                # BI properties
                obj.bi_role = analysis.get('bi_role', 'dimension')
                obj.business_priority = analysis.get('business_priority', 'medium')
                obj.data_type = 'operational' if obj.bi_role == 'fact' else 'reference'
                obj.grain = self._determine_grain(obj.entity_type)
                
                # Data quality
                if hasattr(obj, '__dict__'):
                    obj.data_quality = analysis.get('data_quality', 'production')
                
                print(f"      ✅ {obj.name}: {obj.entity_type} ({obj.bi_role})")
            else:
                print(f"      ⚠️ No analysis for {obj.full_name}")
        
        return batch
    
    def _determine_grain(self, entity_type: str) -> str:
        """Determine table grain from entity type"""
        grain_map = {
            'Customer': 'customer',
            'Payment': 'transaction', 
            'Order': 'order',
            'Product': 'product',
            'Campaign': 'campaign',
            'Address': 'location'
        }
        return grain_map.get(entity_type, entity_type.lower())

class DomainAnalyzer:
    """Enhanced domain analysis including views"""
    
    def determine_domain(self, tables: List[TableInfo]) -> Optional[BusinessDomain]:
        """Determine business domain from analyzed tables and views"""
        if not tables:
            return None
        
        # Count entity types
        entity_counts = {}
        high_priority_tables = []
        
        for table in tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            # Track high priority tables
            if getattr(table, 'business_priority', 'medium') == 'high':
                high_priority_tables.append(table)
        
        # Enhanced domain determination
        customer_objects = entity_counts.get('Customer', 0)
        payment_objects = entity_counts.get('Payment', 0)
        order_objects = entity_counts.get('Order', 0)
        
        # Determine capabilities
        capabilities = {
            'data_exploration': True,
            'basic_reporting': len(tables) > 0
        }
        
        if customer_objects >= 1:
            capabilities['customer_analysis'] = True
            
        if payment_objects >= 1:
            capabilities['payment_analysis'] = True
            capabilities['revenue_reporting'] = True
            
        if customer_objects >= 1 and payment_objects >= 1:
            capabilities['customer_revenue_analysis'] = True
            
        # Create appropriate domain
        if customer_objects >= 1 and payment_objects >= 1:
            return BusinessDomain(
                domain_type="Customer & Payment Analytics",
                industry="Business Intelligence",
                confidence=0.95,
                sample_questions=[
                    "Who are our top paying customers?",
                    "What is our total revenue this year?",
                    "Show customer payment trends",
                    "How many customers do we have?",
                    "What's the average payment amount?"
                ],
                capabilities=capabilities
            )
        elif customer_objects >= 1:
            return BusinessDomain(
                domain_type="Customer Analytics", 
                industry="CRM/Business",
                confidence=0.85,
                sample_questions=[
                    "List our customers",
                    "How many customers do we have?",
                    "Show customer information",
                    "Find customers by name"
                ],
                capabilities=capabilities
            )
        else:
            # Generic business domain
            dominant_entity = max(entity_counts.items(), key=lambda x: x[1])[0] if entity_counts else "Business"
            
            return BusinessDomain(
                domain_type=f"{dominant_entity} Analytics",
                industry="Business Intelligence",
                confidence=0.7,
                sample_questions=[
                    f"Show {dominant_entity.lower()} data",
                    "What information is available?",
                    "List the main data"
                ],
                capabilities=capabilities
            )

class CacheManager:
    """Enhanced cache management"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_analysis(self, tables: List[TableInfo], domain: Optional[BusinessDomain], 
                     relationships: List[Relationship]):
        """Save enhanced analysis results"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'version': '3.1-enhanced-columns',
                'table_count': len(tables),
                'includes_views': any(t.object_type == 'VIEW' for t in tables)
            },
            'tables': [self._table_to_dict(t) for t in tables],
            'domain': self._domain_to_dict(domain) if domain else None,
            'relationships': [self._relationship_to_dict(r) for r in relationships]
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Enhanced analysis cached successfully")
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
        """Convert TableInfo to dictionary with enhanced properties"""
        data = {
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
            'bi_role': getattr(table, 'bi_role', 'dimension'),
            'measures': getattr(table, 'measures', []),
            'entity_keys': getattr(table, 'entity_keys', []),
            'name_columns': getattr(table, 'name_columns', []),
            'time_columns': getattr(table, 'time_columns', []),
            'business_priority': getattr(table, 'business_priority', 'medium'),
            'data_type': getattr(table, 'data_type', 'reference'),
            'grain': getattr(table, 'grain', 'unknown')
        }
        
        # Add enhanced properties if available
        if hasattr(table, 'category_columns'):
            data['category_columns'] = table.category_columns
        if hasattr(table, 'data_quality'):
            data['data_quality'] = table.data_quality
        
        return data
    
    def _dict_to_table(self, data: Dict) -> TableInfo:
        """Convert dictionary to TableInfo with enhanced properties"""
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
        
        # Apply enhanced analysis results
        table.bi_role = data.get('bi_role', 'dimension')
        table.measures = data.get('measures', [])
        table.entity_keys = data.get('entity_keys', [])
        table.name_columns = data.get('name_columns', [])
        table.time_columns = data.get('time_columns', [])
        table.business_priority = data.get('business_priority', 'medium')
        table.data_type = data.get('data_type', 'reference')
        table.grain = data.get('grain', 'unknown')
        
        # Enhanced properties
        if 'category_columns' in data:
            table.category_columns = data['category_columns']
        if 'data_quality' in data:
            table.data_quality = data['data_quality']
        
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
    """Main semantic analyzer with enhanced table+column+view analysis"""
    
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
        
        # Initialize enhanced components
        self.table_analyzer = LLMTableAnalyzer(self.llm)
        self.domain_analyzer = DomainAnalyzer()
        self.cache_manager = CacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        
        print("✅ Enhanced LLM Semantic Analyzer initialized")
        print("   🧠 Deep table+column analysis")
        print("   👁️ View integration")
    
    async def analyze_tables(self, tables: List[TableInfo], discovery=None) -> bool:
        """Enhanced analysis including tables and views"""
        print("🧠 Enhanced Semantic Analysis")
        print("=" * 40)
        
        try:
            start_time = time.time()
            
            # Get view information if discovery available
            view_info = {}
            if discovery and hasattr(discovery, 'get_view_info'):
                view_info = discovery.get_view_info()
                print(f"   👁️ Including {len(view_info)} views in analysis")
            
            # Enhanced LLM analysis with views
            self.tables = await self.table_analyzer.analyze_tables_and_views(tables, view_info)
            
            # Enhanced domain analysis
            self.domain = self.domain_analyzer.determine_domain(self.tables)
            
            # Extract relationships
            self.relationships = self._extract_relationships()
            
            # Save enhanced cache
            self.cache_manager.save_analysis(self.tables, self.domain, self.relationships)
            
            # Show enhanced summary
            self._show_enhanced_summary(time.time() - start_time)
            
            return True
        except Exception as e:
            print(f"❌ Enhanced analysis failed: {e}")
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
    
    def _show_enhanced_summary(self, elapsed_time: float):
        """Show enhanced analysis summary"""
        customer_objects = len([t for t in self.tables if t.entity_type == 'Customer'])
        payment_objects = len([t for t in self.tables if t.entity_type == 'Payment'])
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        view_objects = len([t for t in self.tables if t.object_type == 'VIEW'])
        tables_with_names = len([t for t in self.tables if getattr(t, 'name_columns', [])])
        tables_with_measures = len([t for t in self.tables if getattr(t, 'measures', [])])
        
        print(f"\n✅ ENHANCED SEMANTIC ANALYSIS COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Total objects: {len(self.tables)} (tables + views)")
        print(f"   📋 Tables: {len(self.tables) - view_objects}")
        print(f"   👁️ Views: {view_objects}")
        print(f"   👥 Customer objects: {customer_objects}")
        print(f"   💳 Payment objects: {payment_objects}")
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   📝 Objects with names: {tables_with_names}")
        print(f"   📊 Objects with measures: {tables_with_measures}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        if self.domain:
            print(f"   🎯 Domain: {self.domain.domain_type}")
            print(f"   💡 Capabilities: {len(self.domain.capabilities)} features")
    
    # Public API
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        customer_objects = len([t for t in self.tables if t.entity_type == 'Customer'])
        payment_objects = len([t for t in self.tables if t.entity_type == 'Payment'])
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        view_objects = len([t for t in self.tables if t.object_type == 'VIEW'])
        
        return {
            'total_objects': len(self.tables),
            'total_tables': len(self.tables) - view_objects,
            'total_views': view_objects,
            'customer_tables': customer_objects,
            'payment_tables': payment_objects,
            'fact_tables': fact_tables,
            'total_relationships': len(self.relationships),
            'domain_type': self.domain.domain_type if self.domain else None,
            'analysis_method': 'enhanced_llm_with_views'
        }