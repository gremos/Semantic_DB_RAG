#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Analysis - Enhanced for Better Column Discovery and Classification
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
Enhanced: Better column analysis to ensure discovered columns are properly classified
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship
from shared.utils import parse_json_response

class EntityAnalyzer:
    """Enhanced entity analyzer with better column discovery and classification"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        
        # Enhanced entity taxonomy with better classification
        self.entity_taxonomy = {
            # Fact tables (transactional data)
            'CustomerRevenue': ['customer_payment', 'customer_invoice', 'sales_transaction'],
            'Payment': ['payment', 'transaction', 'invoice', 'billing', 'revenue_detail'],
            'Order': ['order', 'purchase', 'sale', 'quote', 'contract_payment'],
            
            # Core business entities
            'Customer': ['customer', 'client', 'account', 'contact', 'business_point'],
            'Contract': ['contract', 'agreement', 'deal', 'συμβόλαια'],
            'Product': ['product', 'item', 'catalog', 'inventory'],
            
            # Reference/lookup tables (avoid for revenue queries)
            'PaymentMethod': ['payment_method', 'advanced_payment_method'],
            'Category': ['category', 'classification', 'type', 'status'],
            'Lookup': ['lookup', 'reference', 'master', 'code'],
            
            # Other entities
            'Employee': ['employee', 'staff', 'worker', 'personnel'],
            'System': ['system', 'config', 'setting', 'log'],
            'Other': []
        }
    
    async def analyze_tables(self, tables: List[TableInfo], rdl_info: Dict = None) -> List[TableInfo]:
        """Enhanced table analysis with better column discovery"""
        print(f"🧠 Enhanced table analysis: {len(tables)} objects...")
        
        # First pass: Analyze columns locally for each table
        for table in tables:
            self._analyze_table_columns_locally(table)
        
        batch_size = 3
        analyzed_objects = []
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            batch_result = await self._analyze_batch(batch, rdl_info)
            analyzed_objects.extend(batch_result)
            
            if i + batch_size < len(tables):
                await asyncio.sleep(0.5)
        
        print(f"   ✅ Analysis completed: {len(analyzed_objects)} objects")
        return analyzed_objects
    
    def _analyze_table_columns_locally(self, table: TableInfo):
        """Analyze table columns locally to ensure proper classification"""
        if not table.columns:
            return
        
        # Initialize column classifications
        measures = []
        name_columns = []
        entity_keys = []
        time_columns = []
        
        for col in table.columns:
            col_name = col.get('name', '').lower()
            col_type = col.get('data_type', '').lower()
            
            # Measure columns (amount, revenue, numeric)
            if (any(word in col_name for word in ['amount', 'total', 'revenue', 'value', 'price', 'cost', 'sum']) and
                col_type in ['decimal', 'money', 'float', 'numeric', 'int', 'bigint']):
                measures.append(col.get('name'))
            
            # Name/display columns
            elif any(word in col_name for word in ['name', 'title', 'description', 'label']):
                name_columns.append(col.get('name'))
            
            # Entity key columns
            elif (col_name.endswith('id') or col_name.endswith('_id') or 
                  col_name.startswith('id') or col_name == 'id' or
                  col.get('is_primary_key', False)):
                entity_keys.append(col.get('name'))
            
            # Time columns
            elif ('date' in col_type or 'time' in col_type or 
                  any(word in col_name for word in ['date', 'time', 'created', 'modified', 'updated'])):
                time_columns.append(col.get('name'))
        
        # Set the analyzed columns on the table
        table.measures = measures[:5]  # Limit to avoid overflow
        table.name_columns = name_columns[:3]
        table.entity_keys = entity_keys[:5]
        table.time_columns = time_columns[:3]
        
        print(f"      📊 {table.name}: measures={len(measures)}, names={len(name_columns)}, keys={len(entity_keys)}, times={len(time_columns)}")
    
    async def _analyze_batch(self, batch: List[TableInfo], rdl_info: Dict = None) -> List[TableInfo]:
        """Analyze batch with enhanced revenue detection"""
        try:
            prompt = self._create_enhanced_analysis_prompt(batch, rdl_info)
            response = await self._get_llm_response(prompt)
            return self._apply_enhanced_analysis(response, batch)
        except Exception as e:
            print(f"   ⚠️ Batch analysis failed: {e}")
            # Apply local analysis as fallback
            for table in batch:
                self._enhanced_fallback_classification(table)
            return batch
    
    def _create_enhanced_analysis_prompt(self, batch: List[TableInfo], rdl_info: Dict = None) -> str:
        """Create enhanced analysis prompt with actual discovered columns"""
        objects_info = []
        
        for table in batch:
            # Enhanced column analysis with actual column details
            actual_columns = []
            measure_columns = []
            name_columns = []
            key_columns = []
            
            for col in table.columns[:15]:  # Limit to avoid prompt overflow
                col_info = {
                    'name': col.get('name', ''),
                    'type': col.get('data_type', ''),
                    'is_pk': col.get('is_primary_key', False)
                }
                actual_columns.append(col_info)
                
                col_name = col.get('name', '').lower()
                col_type = col.get('data_type', '').lower()
                
                # Categorize columns
                if col_type in ['decimal', 'money', 'float', 'numeric']:
                    measure_columns.append(col.get('name'))
                if any(word in col_name for word in ['name', 'title', 'description']):
                    name_columns.append(col.get('name'))
                if col_name.endswith('id') or col.get('is_primary_key', False):
                    key_columns.append(col.get('name'))
            
            obj_info = {
                'name': table.full_name,
                'row_count': table.row_count,
                'actual_columns': actual_columns,
                'discovered_measures': measure_columns,
                'discovered_names': name_columns,
                'discovered_keys': key_columns,
                'total_columns': len(table.columns)
            }
            
            objects_info.append(obj_info)
        
        rdl_context = ""
        if rdl_info:
            rdl_tables = rdl_info.get('referenced_tables', [])
            rdl_context = f"\nRDL BUSINESS CONTEXT: {', '.join(rdl_tables[:5])}"
        
        return f"""Analyze these database objects for CUSTOMER REVENUE ANALYTICS using ONLY the discovered columns.

CRITICAL: Use ONLY the columns listed in 'actual_columns' for each table. Do not assume any columns exist.

For "top customers by revenue" queries, we need FACT TABLES with:
1. Customer identifiers (from discovered_keys)
2. Amount/revenue columns (from discovered_measures) 
3. Transaction-level data (NOT lookup tables)

DATABASE OBJECTS WITH DISCOVERED COLUMNS:
{json.dumps(objects_info, indent=2)}

{rdl_context}

For each object, classify using ONLY the discovered columns:

ENTITY TYPES:
- CustomerRevenue: Tables with customer+amount data (BEST for revenue queries)
- Payment: Payment transaction tables
- Customer: Customer master data
- PaymentMethod: Payment method lookup (AVOID for revenue)
- Category: Classification lookup (AVOID for revenue)  
- Contract: Contract/agreement data
- System: System/configuration tables
- Other: Everything else

BI_ROLE:
- fact: Transaction data with measures (PREFER for revenue)
- dimension: Reference/lookup data
- lookup: Small reference tables (AVOID for revenue)

Respond with JSON using ONLY discovered column names:
{{
  "analysis": [
    {{
      "object_name": "[schema].[table]",
      "entity_type": "CustomerRevenue",
      "bi_role": "fact",
      "business_priority": "high",
      "revenue_capability": "excellent|good|poor|none",
      "customer_capability": "excellent|good|poor|none",
      "measures": ["actual_column_name"],
      "entity_keys": ["actual_column_name"],
      "name_columns": ["actual_column_name"],
      "time_columns": ["actual_column_name"]
    }}
  ]
}}

IMPORTANT: Only use column names that exist in the 'actual_columns' list for each table."""
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response"""
        try:
            system_msg = """You are a business intelligence analyst specializing in customer revenue analytics.
Classify tables to ensure revenue queries select FACT TABLES with customer+amount data, NOT lookup tables.
Use ONLY the discovered column names provided - do not assume any columns exist.
Respond with valid JSON only."""
            
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM request failed: {e}")
            return ""
    
    def _apply_enhanced_analysis(self, response: str, batch: List[TableInfo]) -> List[TableInfo]:
        """Apply enhanced analysis with better classification"""
        data = parse_json_response(response)
        if not data or 'analysis' not in data:
            return self._fallback_classification(batch)
        
        analysis_map = {
            item.get('object_name', ''): item 
            for item in data['analysis']
        }
        
        for table in batch:
            analysis = analysis_map.get(table.full_name, {})
            
            if analysis:
                # Core classification
                table.entity_type = analysis.get('entity_type', 'Other')
                table.bi_role = analysis.get('bi_role', 'dimension')
                table.business_priority = analysis.get('business_priority', 'medium')
                table.confidence = 0.9
                
                # Enhanced revenue scoring
                revenue_cap = analysis.get('revenue_capability', 'none')
                customer_cap = analysis.get('customer_capability', 'none')
                
                # Calculate revenue readiness score
                if revenue_cap == 'excellent' and customer_cap in ['excellent', 'good']:
                    table.revenue_readiness = 1.0
                elif revenue_cap == 'good' and customer_cap in ['excellent', 'good']:
                    table.revenue_readiness = 0.8
                elif revenue_cap in ['excellent', 'good']:
                    table.revenue_readiness = 0.6
                else:
                    table.revenue_readiness = 0.1
                
                # CRITICAL: Only use columns that actually exist in discovered structure
                llm_measures = analysis.get('measures', [])
                llm_keys = analysis.get('entity_keys', [])
                llm_names = analysis.get('name_columns', [])
                llm_times = analysis.get('time_columns', [])
                
                existing_columns = [col.get('name') for col in table.columns]
                
                # Filter to only existing columns, combine with local analysis
                verified_measures = [col for col in llm_measures if col in existing_columns]
                verified_keys = [col for col in llm_keys if col in existing_columns]
                verified_names = [col for col in llm_names if col in existing_columns]
                verified_times = [col for col in llm_times if col in existing_columns]
                
                # Combine with local analysis (prefer local analysis)
                table.measures = list(set(table.measures + verified_measures))[:5]
                table.entity_keys = list(set(table.entity_keys + verified_keys))[:5]
                table.name_columns = list(set(table.name_columns + verified_names))[:3]
                table.time_columns = list(set(table.time_columns + verified_times))[:3]
                
                # Set business role based on analysis
                if table.entity_type in ['CustomerRevenue', 'Payment', 'Order']:
                    table.business_role = "Revenue Analytics"
                elif table.entity_type == 'Customer':
                    table.business_role = "Customer Master Data"
                elif table.entity_type in ['PaymentMethod', 'Category']:
                    table.business_role = "Lookup Data"
                    table.business_priority = 'low'  # Downgrade lookup tables
                
                print(f"   ✅ {table.name}: {table.entity_type} (revenue={revenue_cap}, measures={len(table.measures)})")
            else:
                # Enhanced fallback
                self._enhanced_fallback_classification(table)
        
        return batch
    
    def _fallback_classification(self, batch: List[TableInfo]) -> List[TableInfo]:
        """Enhanced fallback classification"""
        for table in batch:
            self._enhanced_fallback_classification(table)
        return batch
    
    def _enhanced_fallback_classification(self, table: TableInfo) -> None:
        """Enhanced fallback with revenue detection using discovered columns"""
        name_lower = table.name.lower()
        
        # Analyze discovered columns for classification
        has_customer = any(word in name_lower for word in ['customer', 'client', 'business'])
        
        # Check discovered columns for amount/revenue indicators
        has_amount = len(table.measures) > 0
        has_numeric = any(col.get('data_type', '').lower() in ['decimal', 'money', 'float'] 
                         for col in table.columns)
        
        if has_customer and (has_amount or has_numeric) and table.row_count > 100:
            table.entity_type = 'CustomerRevenue'
            table.bi_role = 'fact'
            table.business_priority = 'high'
            table.revenue_readiness = 0.9
        elif 'payment' in name_lower and not 'method' in name_lower:
            table.entity_type = 'Payment'
            table.bi_role = 'fact'
            table.business_priority = 'high'
            table.revenue_readiness = 0.8
        elif any(word in name_lower for word in ['method', 'category', 'classification', 'type']):
            table.entity_type = 'Category'
            table.bi_role = 'lookup'
            table.business_priority = 'low'
            table.revenue_readiness = 0.1
        elif 'customer' in name_lower:
            table.entity_type = 'Customer'
            table.bi_role = 'dimension'
            table.business_priority = 'high'
            table.revenue_readiness = 0.5
        else:
            table.entity_type = 'Other'
            table.bi_role = 'dimension'
            table.business_priority = 'medium'
            table.revenue_readiness = 0.3
        
        # Set default business role
        if table.entity_type in ['CustomerRevenue', 'Payment', 'Order']:
            table.business_role = "Revenue Analytics"
        elif table.entity_type == 'Customer':
            table.business_role = "Customer Master Data"
        elif table.entity_type in ['PaymentMethod', 'Category']:
            table.business_role = "Lookup Data"
        else:
            table.business_role = "Business Data"

class DomainAnalyzer:
    """Enhanced domain analysis"""
    
    def determine_domain(self, tables: List[TableInfo], rdl_info: Dict = None) -> Optional[BusinessDomain]:
        """Determine business domain with enhanced capabilities"""
        if not tables:
            return None
        
        # Enhanced entity counting
        entity_counts = {}
        revenue_ready_tables = 0
        
        for table in tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            if getattr(table, 'revenue_readiness', 0) >= 0.7:
                revenue_ready_tables += 1
        
        # Enhanced capabilities
        capabilities = {
            'data_exploration': True,
            'basic_reporting': len(tables) > 0,
            'revenue_analytics': revenue_ready_tables > 0,
            'customer_analytics': entity_counts.get('Customer', 0) > 0
        }
        
        # Special revenue capabilities
        if (entity_counts.get('CustomerRevenue', 0) > 0 or 
            (entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0)):
            capabilities['customer_revenue_analysis'] = True
            capabilities['top_customers_by_revenue'] = True
        
        # Determine domain type
        if capabilities.get('customer_revenue_analysis'):
            domain_type = "Customer Revenue Analytics"
            sample_questions = [
                "Who are our top customers by revenue?",
                "What is our total revenue this year?",
                "Show customer payment trends",
                "List customers with highest spending"
            ]
        elif entity_counts.get('Customer', 0) > 0:
            domain_type = "Customer Analytics" 
            sample_questions = [
                "Show customer information",
                "List all customers",
                "Count active customers"
            ]
        else:
            domain_type = "Business Analytics"
            sample_questions = [
                "Show business data",
                "List available information",
                "Display data summary"
            ]
        
        return BusinessDomain(
            domain_type=domain_type,
            industry="Business Intelligence",
            confidence=0.9 if revenue_ready_tables > 0 else 0.7,
            sample_questions=sample_questions,
            capabilities=capabilities
        )

class CacheManager:
    """Simple cache management"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_analysis(self, tables: List[TableInfo], domain: Optional[BusinessDomain], relationships: List[Relationship]):
        """Save semantic analysis"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'metadata': {
                'analyzed': datetime.now().isoformat(),
                'version': '3.2-enhanced-columns',
                'analysis_method': 'enhanced_column_discovery'
            },
            'tables': [self._table_to_dict(t) for t in tables],
            'domain': self._domain_to_dict(domain) if domain else None,
            'relationships': [self._relationship_to_dict(r) for r in relationships]
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Enhanced analysis cached: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def load_analysis(self):
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
            'business_role': table.business_role,
            'confidence': table.confidence,
            'business_priority': getattr(table, 'business_priority', 'medium'),
            'bi_role': getattr(table, 'bi_role', 'dimension'),
            'measures': getattr(table, 'measures', []),
            'name_columns': getattr(table, 'name_columns', []),
            'entity_keys': getattr(table, 'entity_keys', []),
            'time_columns': getattr(table, 'time_columns', []),
            'revenue_readiness': getattr(table, 'revenue_readiness', 0.0)
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
            relationships=data.get('relationships', [])
        )
        
        # Set enhanced properties
        table.entity_type = data.get('entity_type', 'Unknown')
        table.business_role = data.get('business_role', 'Unknown')
        table.confidence = data.get('confidence', 0.0)
        table.business_priority = data.get('business_priority', 'medium')
        table.bi_role = data.get('bi_role', 'dimension')
        table.measures = data.get('measures', [])
        table.name_columns = data.get('name_columns', [])
        table.entity_keys = data.get('entity_keys', [])
        table.time_columns = data.get('time_columns', [])
        table.revenue_readiness = data.get('revenue_readiness', 0.0)
        
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
    
    def _relationship_to_dict(self, relationship: Relationship) -> Dict:
        """Convert Relationship to dictionary"""
        return {
            'from_table': relationship.from_table,
            'to_table': relationship.to_table,
            'relationship_type': relationship.relationship_type,
            'confidence': relationship.confidence,
            'description': relationship.description
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
    """Enhanced semantic analyzer with better column discovery and classification"""
    
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
        self.entity_analyzer = EntityAnalyzer(self.llm)
        self.domain_analyzer = DomainAnalyzer()
        self.cache_manager = CacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        
        print("✅ Enhanced Semantic Analyzer initialized")
        print("   🎯 Column discovery and classification enabled")
        print("   📊 Only discovered columns will be used")
    
    async def analyze_tables(self, tables: List[TableInfo], discovery=None) -> bool:
        """Enhanced analysis with column discovery"""
        print("🧠 ENHANCED COLUMN-DISCOVERY ANALYSIS")
        print("=" * 50)
        
        try:
            start_time = time.time()
            
            # Get RDL context
            rdl_info = {}
            if discovery and hasattr(discovery, 'get_rdl_info'):
                rdl_info = discovery.get_rdl_info()
                if rdl_info:
                    print(f"   📋 RDL context: {rdl_info.get('report_count', 0)} reports")
            
            # Enhanced entity analysis with column discovery
            self.tables = await self.entity_analyzer.analyze_tables(tables, rdl_info)
            
            # Domain analysis
            self.domain = self.domain_analyzer.determine_domain(self.tables, rdl_info)
            
            # Extract relationships
            self.relationships = self._extract_relationships()
            
            # Save cache
            self.cache_manager.save_analysis(self.tables, self.domain, self.relationships)
            
            # Show summary
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
                            to_table=to_ref.split('.')[0] + '.' + to_ref.split('.')[1] if '.' in to_ref else to_ref,
                            relationship_type='foreign_key',
                            confidence=0.9,
                            description=f"FK: {from_col} -> {to_ref}"
                        ))
                    except Exception:
                        continue
        
        return relationships
    
    def _show_enhanced_summary(self, elapsed_time: float):
        """Show enhanced analysis summary"""
        entity_counts = {}
        revenue_ready = 0
        fact_tables = 0
        total_measures = 0
        total_names = 0
        
        for table in self.tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            if getattr(table, 'revenue_readiness', 0) >= 0.7:
                revenue_ready += 1
            
            if getattr(table, 'bi_role', '') == 'fact':
                fact_tables += 1
            
            total_measures += len(getattr(table, 'measures', []))
            total_names += len(getattr(table, 'name_columns', []))
        
        print(f"\n✅ ENHANCED COLUMN-DISCOVERY ANALYSIS COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Total objects: {len(self.tables)}")
        print(f"   💰 Revenue-ready tables: {revenue_ready}")
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   🔢 Discovered measures: {total_measures}")
        print(f"   🏷️ Discovered name columns: {total_names}")
        
        # Show revenue capabilities
        if entity_counts.get('CustomerRevenue', 0) > 0:
            print(f"   🔥 CustomerRevenue tables: {entity_counts['CustomerRevenue']}")
        if entity_counts.get('Payment', 0) > 0:
            print(f"   💳 Payment tables: {entity_counts['Payment']}")
        if entity_counts.get('Customer', 0) > 0:
            print(f"   👥 Customer tables: {entity_counts['Customer']}")
        
        # Warn about lookup tables
        lookup_count = entity_counts.get('PaymentMethod', 0) + entity_counts.get('Category', 0)
        if lookup_count > 0:
            print(f"   ⚠️ Lookup tables (low priority): {lookup_count}")
        
        if self.domain:
            print(f"   🏢 Domain: {self.domain.domain_type}")
            if self.domain.capabilities.get('customer_revenue_analysis'):
                print(f"   ✅ Revenue analytics enabled")
    
    # Public API
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        entity_counts = {}
        revenue_ready = 0
        total_measures = 0
        
        for table in self.tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            if getattr(table, 'revenue_readiness', 0) >= 0.7:
                revenue_ready += 1
            
            total_measures += len(getattr(table, 'measures', []))
        
        return {
            'total_objects': len(self.tables),
            'revenue_ready_tables': revenue_ready,
            'customer_revenue_tables': entity_counts.get('CustomerRevenue', 0),
            'payment_tables': entity_counts.get('Payment', 0),
            'customer_tables': entity_counts.get('Customer', 0),
            'lookup_tables': entity_counts.get('PaymentMethod', 0) + entity_counts.get('Category', 0),
            'entity_types_found': len(entity_counts),
            'total_relationships': len(self.relationships),
            'domain_type': self.domain.domain_type if self.domain else None,
            'revenue_analytics_enabled': self.domain.capabilities.get('customer_revenue_analysis', False) if self.domain else False,
            'analysis_method': 'enhanced_column_discovery',
            'total_discovered_measures': total_measures
        }