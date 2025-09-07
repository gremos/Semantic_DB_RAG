#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Analysis - Cross-Industry Entity Recognition
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship
from shared.utils import parse_json_response

class EntityAnalyzer:
    """Cross-industry entity analyzer"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        
        # Cross-industry entity taxonomy (Architecture requirement)
        self.entity_taxonomy = {
            # Core business entities
            'Customer': ['customer', 'client', 'account', 'contact'],
            'Payment': ['payment', 'transaction', 'invoice', 'billing'],
            'Order': ['order', 'purchase', 'sale', 'quote'],
            'Product': ['product', 'item', 'catalog', 'inventory'],
            'Contract': ['contract', 'agreement', 'deal', 'συμβόλαια'],
            'Address': ['address', 'location', 'geography'],
            
            # Cross-industry extensions
            'Employee': ['employee', 'staff', 'worker', 'personnel'],
            'Vendor': ['vendor', 'supplier', 'partner'],
            'Project': ['project', 'task', 'work', 'job'],
            'Asset': ['asset', 'equipment', 'resource'],
            'Inventory': ['inventory', 'stock', 'warehouse'],
            'Event': ['event', 'log', 'audit', 'activity'],
            
            # System entities
            'System': ['system', 'config', 'setting', 'log'],
            'Other': []
        }
    
    async def analyze_tables(self, tables: List[TableInfo], rdl_info: Dict = None) -> List[TableInfo]:
        """Analyze tables with cross-industry taxonomy"""
        print(f"🧠 Cross-industry analysis: {len(tables)} objects...")
        
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
    
    async def _analyze_batch(self, batch: List[TableInfo], rdl_info: Dict = None) -> List[TableInfo]:
        """Analyze batch of tables"""
        try:
            prompt = self._create_analysis_prompt(batch, rdl_info)
            response = await self._get_llm_response(prompt)
            return self._apply_analysis(response, batch)
        except Exception as e:
            print(f"   ⚠️ Batch analysis failed: {e}")
            return batch
    
    def _create_analysis_prompt(self, batch: List[TableInfo], rdl_info: Dict = None) -> str:
        """Create analysis prompt"""
        objects_info = []
        
        for table in batch:
            columns_detail = []
            for col in table.columns[:15]:
                columns_detail.append({
                    'name': col.get('name', ''),
                    'type': col.get('data_type', ''),
                    'nullable': col.get('is_nullable', True)
                })
            
            sample_detail = self._get_sample_insights(table)
            
            obj_info = {
                'name': table.full_name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': columns_detail,
                'sample_insights': sample_detail
            }
            
            objects_info.append(obj_info)
        
        rdl_context = ""
        if rdl_info:
            rdl_context = f"\nRDL CONTEXT: Report references: {', '.join(rdl_info.get('referenced_tables', [])[:5])}"
        
        return f"""Analyze these database objects for business intelligence using cross-industry entity recognition.

CROSS-INDUSTRY ENTITY TAXONOMY:
- Customer: customer master data, accounts, contacts
- Payment: financial transactions, payments, invoices
- Order: sales orders, purchases, quotes  
- Product: items, catalog, inventory
- Contract: agreements, deals, contracts
- Employee: staff data, personnel, workers
- Vendor: supplier data, partners
- Project: project management, tasks
- Asset: equipment, resources, property
- Inventory: stock data, warehouse
- Event: activity logs, audit trails
- System: configuration, settings
- Other: specify if none fit

DATABASE OBJECTS:
{json.dumps(objects_info, indent=2, default=str)}

{rdl_context}

For each object, provide analysis:

1. ENTITY_TYPE: Select from taxonomy above
2. BUSINESS_CONTEXT: Specific business purpose
3. COLUMN_ANALYSIS: Categorize by function:
   - MEASURES: numeric columns for aggregation
   - ENTITY_KEYS: identifiers for joining
   - NAME_COLUMNS: descriptive text for display
   - TIME_COLUMNS: dates and timestamps
   - CATEGORY_COLUMNS: classification data
4. BI_ROLE: fact (transactional) or dimension (reference)
5. BUSINESS_PRIORITY: high (core), medium (supporting), low (system)
6. DATA_QUALITY: production, test, or archive

Respond with JSON only:
{{
  "analysis": [
    {{
      "object_name": "[schema].[table]",
      "entity_type": "Customer",
      "business_context": "Customer master data",
      "column_analysis": {{
        "measures": ["amount", "total"],
        "entity_keys": ["customer_id"],
        "name_columns": ["customer_name"],
        "time_columns": ["created_date"],
        "category_columns": ["status"]
      }},
      "bi_role": "dimension",
      "business_priority": "high",
      "data_quality": "production"
    }}
  ]
}}"""
    
    def _get_sample_insights(self, table: TableInfo) -> Dict:
        """Get sample data insights"""
        if not table.sample_data:
            return {'status': 'no_data'}
        
        insights = []
        first_row = table.sample_data[0]
        
        for key, value in list(first_row.items())[:6]:
            if key.startswith('__'):
                continue
            
            insight = self._analyze_column_value(key, value)
            if insight:
                insights.append(insight)
        
        return {
            'status': 'has_data',
            'insights': insights
        }
    
    def _analyze_column_value(self, column_name: str, value: Any) -> str:
        """Analyze column value for insights"""
        col_lower = column_name.lower()
        
        if isinstance(value, (int, float)) and abs(value) > 0:
            if any(word in col_lower for word in ['amount', 'total', 'price']):
                return f"{column_name}: monetary value - likely measure"
            elif any(word in col_lower for word in ['count', 'quantity']):
                return f"{column_name}: numeric value - likely measure"
            else:
                return f"{column_name}: numeric - potential identifier"
        
        elif isinstance(value, str) and len(value) > 2:
            if any(word in col_lower for word in ['name', 'title', 'description']):
                return f"{column_name}: text - likely display name"
            elif any(word in col_lower for word in ['status', 'type', 'category']):
                return f"{column_name}: text - likely classification"
            else:
                return f"{column_name}: text - potential name or category"
        
        return ""
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response"""
        try:
            system_msg = """You are a business intelligence analyst. Analyze database objects by understanding business purpose from table names, column patterns, and sample values.
Use cross-industry entity taxonomy to classify business data accurately.
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
    
    def _apply_analysis(self, response: str, batch: List[TableInfo]) -> List[TableInfo]:
        """Apply LLM analysis results"""
        data = parse_json_response(response)
        if not data or 'analysis' not in data:
            return batch
        
        analysis_map = {
            item.get('object_name', ''): item 
            for item in data['analysis']
        }
        
        for table in batch:
            analysis = analysis_map.get(table.full_name, {})
            
            if analysis:
                # Core classification
                table.entity_type = analysis.get('entity_type', 'Other')
                table.business_role = analysis.get('business_context', 'Unknown')
                table.confidence = 0.9
                
                # Column analysis
                col_analysis = analysis.get('column_analysis', {})
                table.measures = col_analysis.get('measures', [])
                table.entity_keys = col_analysis.get('entity_keys', [])
                table.name_columns = col_analysis.get('name_columns', [])
                table.time_columns = col_analysis.get('time_columns', [])
                
                # BI properties
                table.bi_role = analysis.get('bi_role', 'dimension')
                table.business_priority = analysis.get('business_priority', 'medium')
                table.data_type = 'operational' if table.bi_role == 'fact' else 'reference'
                
                print(f"   ✅ {table.name}: {table.entity_type} ({table.bi_role})")
            else:
                # Fallback
                table.entity_type = self._classify_by_name(table.name)
        
        return batch
    
    def _classify_by_name(self, table_name: str) -> str:
        """Fallback classification by name"""
        name_lower = table_name.lower()
        
        for entity_type, keywords in self.entity_taxonomy.items():
            if any(keyword in name_lower for keyword in keywords):
                return entity_type
        
        # Additional patterns
        if any(word in name_lower for word in ['συμβόλαια', 'contract']):
            return 'Contract'
        elif 'business' in name_lower:
            return 'Customer'
        
        return 'Other'

class DomainAnalyzer:
    """Business domain analysis"""
    
    def determine_domain(self, tables: List[TableInfo], rdl_info: Dict = None) -> Optional[BusinessDomain]:
        """Determine business domain"""
        if not tables:
            return None
        
        # Count entity types
        entity_counts = {}
        for table in tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Build capabilities
        capabilities = {
            'data_exploration': True,
            'basic_reporting': len(tables) > 0
        }
        
        for entity_type, count in entity_counts.items():
            if count >= 1:
                capability_name = f"{entity_type.lower()}_analysis"
                capabilities[capability_name] = True
        
        # Special combinations
        if entity_counts.get('Customer', 0) >= 1 and entity_counts.get('Payment', 0) >= 1:
            capabilities['customer_revenue_analysis'] = True
        
        if entity_counts.get('Employee', 0) >= 1:
            capabilities['hr_analytics'] = True
        
        if entity_counts.get('Contract', 0) >= 1:
            capabilities['contract_management'] = True
        
        # Determine domain type
        domain_confidence = 0.8
        if rdl_info and rdl_info.get('business_priority_signals'):
            domain_confidence = 0.95
        
        # Create domain based on entity mix
        primary_entity = max(entity_counts.items(), key=lambda x: x[1])[0] if entity_counts else 'Business'
        
        if entity_counts.get('Contract', 0) >= 1 and rdl_info:
            domain_type = "Contract Analytics"
            sample_questions = [
                "Show contract details",
                "What is our total contract value?",
                "List contracts by customer"
            ]
        elif entity_counts.get('Customer', 0) >= 1 and entity_counts.get('Payment', 0) >= 1:
            domain_type = "Customer & Financial Analytics"
            sample_questions = [
                "Who are our top customers?",
                "What is our total revenue?",
                "Show customer payment trends"
            ]
        elif entity_counts.get('Employee', 0) >= 1:
            domain_type = "HR Analytics"
            sample_questions = [
                "Show employee headcount",
                "List departments and staff",
                "Show employee data"
            ]
        else:
            domain_type = f"{primary_entity} Analytics"
            sample_questions = [
                f"Show {primary_entity.lower()} data",
                f"List all {primary_entity.lower()} records",
                "Show data summary"
            ]
        
        return BusinessDomain(
            domain_type=domain_type,
            industry=self._determine_industry(entity_counts, rdl_info),
            confidence=domain_confidence,
            sample_questions=sample_questions,
            capabilities=capabilities
        )
    
    def _determine_industry(self, entity_counts: Dict, rdl_info: Dict = None) -> str:
        """Determine industry from entity patterns"""
        if rdl_info:
            report_title = rdl_info.get('report_title', '').lower()
            if 'absentees' in report_title:
                return "Human Resources"
            if 'contract' in report_title:
                return "Contract Management"
        
        if entity_counts.get('Employee', 0) > 2:
            return "Human Resources"
        elif entity_counts.get('Contract', 0) > 0:
            return "Legal/Contracts"
        elif entity_counts.get('Customer', 0) > 0:
            return "Business/Commerce"
        else:
            return "Business Intelligence"

class CacheManager:
    """Cache management for semantic analysis"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_analysis(self, tables: List[TableInfo], domain: Optional[BusinessDomain], relationships: List[Relationship]):
        """Save semantic analysis to cache"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'metadata': {
                'analyzed': datetime.now().isoformat(),
                'version': '3.0-cross-industry',
                'analysis_method': 'cross_industry_llm'
            },
            'tables': [self._table_to_dict(t) for t in tables],
            'domain': self._domain_to_dict(domain) if domain else None,
            'relationships': [self._relationship_to_dict(r) for r in relationships]
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Semantic analysis cached: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def load_analysis(self):
        """Load semantic analysis from cache"""
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
            'time_columns': getattr(table, 'time_columns', [])
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
    """Enhanced semantic analyzer with cross-industry support"""
    
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
        
        print("✅ Cross-Industry Semantic Analyzer initialized")
    
    async def analyze_tables(self, tables: List[TableInfo], discovery=None) -> bool:
        """Enhanced analysis with cross-industry entities"""
        print("🧠 CROSS-INDUSTRY SEMANTIC ANALYSIS")
        print("=" * 40)
        
        try:
            start_time = time.time()
            
            # Get RDL context if available
            rdl_info = {}
            if discovery and hasattr(discovery, 'get_rdl_info'):
                rdl_info = discovery.get_rdl_info()
                if rdl_info:
                    print(f"   📋 RDL context: {rdl_info.get('report_count', 0)} reports")
            
            # Entity analysis
            self.tables = await self.entity_analyzer.analyze_tables(tables, rdl_info)
            
            # Domain analysis
            self.domain = self.domain_analyzer.determine_domain(self.tables, rdl_info)
            
            # Extract relationships
            self.relationships = self._extract_relationships()
            
            # Save cache
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
    
    def _show_summary(self, elapsed_time: float):
        """Show analysis summary"""
        entity_counts = {}
        for table in self.tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        high_priority = len([t for t in self.tables if getattr(t, 'business_priority', '') == 'high'])
        
        print(f"\n✅ CROSS-INDUSTRY ANALYSIS COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Total objects: {len(self.tables)}")
        print(f"   🌐 Entity types: {len(entity_counts)}")
        
        # Show top entities
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        for entity_type, count in sorted_entities[:6]:
            priority_emoji = "🔥" if entity_type in ['Customer', 'Payment', 'Contract'] else "📋"
            print(f"   {priority_emoji} {entity_type}: {count}")
        
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   🎯 High priority: {high_priority}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        if self.domain:
            print(f"   🏢 Domain: {self.domain.domain_type}")
            print(f"   🏭 Industry: {self.domain.industry}")
    
    # Public API
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        entity_counts = {}
        for table in self.tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        high_priority = len([t for t in self.tables if getattr(t, 'business_priority', '') == 'high'])
        
        return {
            'total_objects': len(self.tables),
            'total_tables': len([t for t in self.tables if t.object_type != 'VIEW']),
            'total_views': len([t for t in self.tables if t.object_type == 'VIEW']),
            'entity_types_found': len(entity_counts),
            'customer_tables': entity_counts.get('Customer', 0),
            'payment_tables': entity_counts.get('Payment', 0),
            'contract_tables': entity_counts.get('Contract', 0),
            'employee_tables': entity_counts.get('Employee', 0),
            'fact_tables': fact_tables,
            'high_priority_tables': high_priority,
            'total_relationships': len(self.relationships),
            'domain_type': self.domain.domain_type if self.domain else None,
            'industry': self.domain.industry if self.domain else None,
            'analysis_method': 'cross_industry_llm'
        }