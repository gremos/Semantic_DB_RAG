#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Analysis - Enhanced Cross-Industry Entity Recognition
Architecture: Generalized beyond CRM, enhanced taxonomy, RDL integration
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
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

class CacheManager:
    """Cache manager for semantic analysis results"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_analysis(self, tables: List[TableInfo], domain: Optional[BusinessDomain], 
                     relationships: List[Relationship]):
        """Save semantic analysis results to cache"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'metadata': {
                'analyzed': datetime.now().isoformat(),
                'version': '3.0-cross-industry-rdl',
                'analysis_method': 'enhanced_cross_industry_llm',
                'total_objects': len(tables)
            },
            'analysis_summary': {
                'total_tables': len([t for t in tables if t.object_type != 'VIEW']),
                'total_views': len([t for t in tables if t.object_type == 'VIEW']),
                'high_priority_tables': len([t for t in tables if getattr(t, 'business_priority', '') == 'high']),
                'fact_tables': len([t for t in tables if getattr(t, 'bi_role', '') == 'fact'])
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
    
    def load_analysis(self) -> Tuple[List[TableInfo], Optional[BusinessDomain], List[Relationship]]:
        """Load semantic analysis results from cache"""
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
            'business_role': table.business_role,
            'confidence': table.confidence,
            'business_priority': getattr(table, 'business_priority', 'medium'),
            'bi_role': getattr(table, 'bi_role', 'dimension'),
            'measures': getattr(table, 'measures', []),
            'name_columns': getattr(table, 'name_columns', []),
            'key_columns': getattr(table, 'key_columns', []),
            'time_columns': getattr(table, 'time_columns', []),
            'entity_keys': getattr(table, 'entity_keys', []),
            'data_type': getattr(table, 'data_type', 'reference'),
            'grain': getattr(table, 'grain', 'unknown')
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
        table.key_columns = data.get('key_columns', [])
        table.time_columns = data.get('time_columns', [])
        table.entity_keys = data.get('entity_keys', [])
        table.data_type = data.get('data_type', 'reference')
        table.grain = data.get('grain', 'unknown')
        
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

class EnhancedEntityAnalyzer:
    """Enhanced entity analyzer with cross-industry taxonomy (Architecture requirement)"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        
        # Enhanced entity taxonomy - extensible beyond CRM
        self.entity_taxonomy = {
            # Core business entities
            'Customer': ['customer', 'client', 'account', 'contact', 'user'],
            'Payment': ['payment', 'transaction', 'invoice', 'billing', 'revenue'],
            'Order': ['order', 'purchase', 'sale', 'quote', 'request'],
            'Product': ['product', 'item', 'catalog', 'inventory', 'sku'],
            'Campaign': ['campaign', 'marketing', 'promotion', 'advertising'],
            'Address': ['address', 'location', 'geography', 'region', 'territory'],
            
            # Cross-industry extensions (Architecture requirement)
            'Employee': ['employee', 'staff', 'worker', 'personnel', 'user'],
            'Invoice': ['invoice', 'bill', 'receipt', 'statement'],
            'Journal': ['journal', 'ledger', 'accounting', 'gl', 'financial'],
            'Vendor': ['vendor', 'supplier', 'partner', 'merchant'],
            'Project': ['project', 'task', 'work', 'job', 'initiative'],
            'Ticket': ['ticket', 'case', 'issue', 'incident', 'support'],
            'Asset': ['asset', 'equipment', 'resource', 'property'],
            'Inventory': ['inventory', 'stock', 'warehouse', 'supply'],
            'Shipment': ['shipment', 'delivery', 'shipping', 'logistics'],
            'Subscription': ['subscription', 'plan', 'service', 'membership'],
            'Contract': ['contract', 'agreement', 'deal', 'terms'],
            'Event': ['event', 'log', 'audit', 'activity', 'history'],
            
            # System entities
            'System': ['system', 'config', 'setting', 'parameter', 'log'],
            'Other': []  # Catchall
        }
    
    async def analyze_tables_with_enhanced_taxonomy(self, tables: List[TableInfo], 
                                                   rdl_info: Dict = None) -> List[TableInfo]:
        """Analyze tables using enhanced cross-industry taxonomy"""
        print(f"🧠 Enhanced cross-industry analysis: {len(tables)} objects...")
        
        # Include RDL context in analysis
        rdl_context = self._build_rdl_context(rdl_info) if rdl_info else ""
        
        # Process in batches for detailed analysis
        batch_size = 3  # Smaller batches for more detailed analysis
        analyzed_objects = []
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            batch_result = await self._analyze_enhanced_batch(batch, rdl_context)
            analyzed_objects.extend(batch_result)
            
            if i + batch_size < len(tables):
                await asyncio.sleep(0.5)  # Rate limiting
        
        print(f"   ✅ Enhanced analysis completed: {len(analyzed_objects)} objects")
        return analyzed_objects
    
    def _build_rdl_context(self, rdl_info: Dict) -> str:
        """Build RDL context for enhanced analysis"""
        if not rdl_info:
            return ""
        
        context_lines = ["\nRDL BUSINESS CONTEXT:"]
        
        report_title = rdl_info.get('report_title', '')
        if report_title:
            context_lines.append(f"Report: {report_title}")
        
        referenced_tables = rdl_info.get('referenced_tables', set())
        if referenced_tables:
            context_lines.append(f"RDL References: {', '.join(list(referenced_tables)[:5])}")
        
        business_signals = rdl_info.get('business_priority_signals', [])
        if business_signals:
            context_lines.append(f"Business Context: {', '.join(business_signals)}")
        
        return '\n'.join(context_lines)
    
    async def _analyze_enhanced_batch(self, batch: List[TableInfo], rdl_context: str = "") -> List[TableInfo]:
        """Enhanced batch analysis with cross-industry entity recognition"""
        try:
            prompt = self._create_enhanced_analysis_prompt(batch, rdl_context)
            response = await self._get_llm_response(prompt)
            return self._apply_enhanced_analysis(response, batch)
        except Exception as e:
            print(f"   ⚠️ Enhanced batch analysis failed: {e}")
            return batch
    
    def _create_enhanced_analysis_prompt(self, batch: List[TableInfo], rdl_context: str = "") -> str:
        """Create enhanced analysis prompt with cross-industry focus"""
        objects_info = []
        
        for table in batch:
            # Get detailed information
            columns_detail = self._get_detailed_columns(table)
            sample_detail = self._get_enhanced_sample(table)
            
            obj_info = {
                'name': table.full_name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns_with_types': columns_detail,
                'sample_data_insights': sample_detail,
                'foreign_key_relationships': table.relationships
            }
            
            objects_info.append(obj_info)
        
        return f"""Analyze these database objects for business intelligence using cross-industry entity recognition.
Determine what business data each object contains by analyzing table names, column names, data types, and sample values.

ENHANCED ENTITY TAXONOMY (choose the most appropriate):

CORE BUSINESS ENTITIES:
- Customer: customer master data, accounts, contacts, users
- Payment: financial transactions, payments, invoices, billing data
- Order: sales orders, purchases, quotes, requests
- Product: items, catalog, inventory, SKUs
- Campaign: marketing campaigns, promotions, advertising
- Address: location data, geography, regions

CROSS-INDUSTRY ENTITIES:
- Employee/HR: staff data, personnel, workers, users
- Invoice: billing documents, receipts, statements
- Journal/GL: accounting data, ledger entries, financial records
- Vendor/Supplier: supplier data, partners, merchants
- Project: project management, tasks, work items, jobs
- Ticket/Case: support tickets, issues, incidents, cases
- Asset: equipment, resources, property, fixed assets
- Inventory: stock data, warehouse, supply chain
- Shipment: delivery data, logistics, shipping
- Subscription: plans, services, memberships, recurring
- Contract: agreements, deals, terms, legal documents
- Event/Log: activity logs, audit trails, events, history

SYSTEM ENTITIES:
- System: configuration, settings, parameters, technical data
- Other: specify the specific business entity if none above fit

DATABASE OBJECTS TO ANALYZE:
{json.dumps(objects_info, indent=2, default=str)}

{rdl_context}

For each object, provide comprehensive analysis:

1. ENTITY_TYPE: Select from the enhanced taxonomy above
2. BUSINESS_CONTEXT: Specific business purpose (e.g., "Customer contact management", "Financial transaction processing")
3. COLUMN_ANALYSIS: Categorize columns by business function:
   - MEASURES: numeric columns for aggregation (amounts, quantities, totals, counts, prices)
   - ENTITY_KEYS: identifiers for joining (IDs, keys, codes, references)
   - NAME_COLUMNS: descriptive text for display (names, titles, descriptions, labels)
   - TIME_COLUMNS: temporal data (dates, timestamps, created/modified times)
   - CATEGORY_COLUMNS: classification data (status, type, category, flags)
4. BI_ROLE: Based on data patterns:
   - fact: contains transactional/quantitative data with measures
   - dimension: contains reference/descriptive data for context
5. BUSINESS_PRIORITY: Importance level:
   - high: core operational entities (Customer, Payment, Order, Contract, Employee)
   - medium: supporting entities (Product, Address, Vendor, Project)
   - low: system/configuration entities
6. DATA_QUALITY: Based on sample data quality:
   - production: real business data with meaningful values
   - test: test data with placeholder/sample values
   - archive: historical or backup data

Respond with JSON only:
{{
  "analysis": [
    {{
      "object_name": "[schema].[table]",
      "entity_type": "Customer",
      "business_context": "Customer master data with contact information and account details",
      "column_analysis": {{
        "measures": ["total_orders", "credit_limit", "balance"],
        "entity_keys": ["customer_id", "account_number"],
        "name_columns": ["customer_name", "company_name", "contact_name"],
        "time_columns": ["created_date", "last_contact_date"],
        "category_columns": ["status", "customer_type", "region"]
      }},
      "bi_role": "dimension",
      "business_priority": "high",
      "data_quality": "production"
    }}
  ]
}}"""
    
    def _get_detailed_columns(self, table: TableInfo) -> List[Dict]:
        """Get enhanced column information"""
        columns = []
        for col in table.columns[:20]:  # First 20 columns
            col_info = {
                'name': col.get('name', ''),
                'type': col.get('data_type', ''),
                'nullable': col.get('is_nullable', True),
                'is_identity': col.get('is_identity', False),
                'is_primary_key': col.get('is_primary_key', False)
            }
            columns.append(col_info)
        return columns
    
    def _get_enhanced_sample(self, table: TableInfo) -> Dict:
        """Get enhanced sample data insights"""
        if not table.sample_data:
            return {'status': 'no_sample_data', 'insights': []}
        
        insights = []
        first_row = table.sample_data[0]
        
        # Analyze sample values for business insights
        for key, value in list(first_row.items())[:8]:
            if key.startswith('__'):
                continue
            
            insight = self._analyze_column_value(key, value)
            if insight:
                insights.append(insight)
        
        return {
            'status': 'has_sample_data',
            'row_count': len(table.sample_data),
            'insights': insights
        }
    
    def _analyze_column_value(self, column_name: str, value: Any) -> str:
        """Analyze individual column value for business insights"""
        col_lower = column_name.lower()
        
        if isinstance(value, (int, float)) and abs(value) > 0:
            if any(word in col_lower for word in ['amount', 'total', 'price', 'cost', 'revenue']):
                return f"{column_name}: monetary value ({value}) - likely financial measure"
            elif any(word in col_lower for word in ['count', 'quantity', 'qty', 'number']):
                return f"{column_name}: numeric value ({value}) - likely quantity measure"
            else:
                return f"{column_name}: numeric value ({value}) - potential measure or identifier"
        
        elif isinstance(value, str) and len(value) > 2:
            if any(word in col_lower for word in ['name', 'title', 'description', 'label']):
                return f"{column_name}: descriptive text - likely display name"
            elif any(word in col_lower for word in ['status', 'type', 'category', 'state']):
                return f"{column_name}: categorical text - likely classification"
            elif any(word in col_lower for word in ['id', 'key', 'code', 'number']):
                return f"{column_name}: identifier text - likely entity key"
            else:
                return f"{column_name}: text value - potential name or category"
        
        elif 'date' in col_lower or 'time' in col_lower:
            return f"{column_name}: temporal data - time dimension"
        
        return ""
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get enhanced LLM response"""
        try:
            system_msg = """You are an expert business intelligence analyst specializing in cross-industry database analysis.
Analyze database objects by understanding the business purpose from table names, column patterns, data types, and sample values.
Use the enhanced entity taxonomy to classify business data accurately across different industries.
Focus on identifying what real business information each object contains.
Respond with valid JSON only."""
            
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"      ⚠️ LLM request failed: {e}")
            return ""
    
    def _apply_enhanced_analysis(self, response: str, batch: List[TableInfo]) -> List[TableInfo]:
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
        
        # Apply enhanced analysis to each table
        for table in batch:
            analysis = analysis_map.get(table.full_name, {})
            
            if analysis:
                # Core entity classification
                table.entity_type = analysis.get('entity_type', 'Other')
                table.business_role = analysis.get('business_context', 'Unknown business purpose')
                table.confidence = 0.9  # High confidence from LLM
                
                # Enhanced column analysis
                col_analysis = analysis.get('column_analysis', {})
                table.measures = col_analysis.get('measures', [])
                table.entity_keys = col_analysis.get('entity_keys', [])
                table.name_columns = col_analysis.get('name_columns', [])
                table.time_columns = col_analysis.get('time_columns', [])
                
                # Business intelligence properties
                table.bi_role = analysis.get('bi_role', 'dimension')
                table.business_priority = analysis.get('business_priority', 'medium')
                table.data_type = 'operational' if table.bi_role == 'fact' else 'reference'
                table.grain = self._determine_grain(table.entity_type)
                
                # Enhanced properties
                if hasattr(table, '__dict__'):
                    table.category_columns = col_analysis.get('category_columns', [])
                    table.data_quality = analysis.get('data_quality', 'production')
                
                print(f"      ✅ {table.name}: {table.entity_type} ({table.bi_role}, {table.business_priority})")
            else:
                # Fallback classification
                table.entity_type = self._classify_by_name(table.name)
                print(f"      ⚠️ Fallback classification for {table.full_name}: {table.entity_type}")
        
        return batch
    
    def _determine_grain(self, entity_type: str) -> str:
        """Determine table grain from entity type"""
        grain_map = {
            'Customer': 'customer',
            'Payment': 'transaction',
            'Order': 'order',
            'Product': 'product',
            'Campaign': 'campaign',
            'Address': 'location',
            'Employee': 'employee',
            'Invoice': 'invoice',
            'Journal': 'journal_entry',
            'Vendor': 'vendor',
            'Project': 'project',
            'Ticket': 'ticket',
            'Asset': 'asset',
            'Inventory': 'inventory_item',
            'Shipment': 'shipment',
            'Subscription': 'subscription',
            'Contract': 'contract',
            'Event': 'event'
        }
        return grain_map.get(entity_type, entity_type.lower())
    
    def _classify_by_name(self, table_name: str) -> str:
        """Fallback classification by table name patterns"""
        name_lower = table_name.lower()
        
        # Check against taxonomy
        for entity_type, keywords in self.entity_taxonomy.items():
            if any(keyword in name_lower for keyword in keywords):
                return entity_type
        
        # Additional pattern matching
        if any(word in name_lower for word in ['συμβόλαια', 'contract']):
            return 'Contract'
        elif any(word in name_lower for word in ['business', 'company']):
            return 'Customer'
        elif any(word in name_lower for word in ['target', 'group']):
            return 'Campaign'
        
        return 'Other'

class EnhancedDomainAnalyzer:
    """Enhanced domain analysis for cross-industry business domains"""
    
    def determine_enhanced_domain(self, tables: List[TableInfo], rdl_info: Dict = None) -> Optional[BusinessDomain]:
        """Determine business domain using enhanced entity analysis"""
        if not tables:
            return None
        
        # Count entity types across all entities
        entity_counts = {}
        high_priority_count = 0
        fact_table_count = 0
        
        for table in tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            if getattr(table, 'business_priority', 'medium') == 'high':
                high_priority_count += 1
            
            if getattr(table, 'bi_role', 'dimension') == 'fact':
                fact_table_count += 1
        
        # Determine dominant business patterns
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        primary_entity = sorted_entities[0][0] if sorted_entities else 'Business'
        
        # Build enhanced capabilities
        capabilities = {
            'data_exploration': True,
            'basic_reporting': len(tables) > 0,
            'cross_entity_analysis': len(entity_counts) > 3
        }
        
        # Entity-specific capabilities
        for entity_type, count in entity_counts.items():
            if count >= 1:
                capability_name = f"{entity_type.lower()}_analysis"
                capabilities[capability_name] = True
        
        # Special capability combinations
        if entity_counts.get('Customer', 0) >= 1 and entity_counts.get('Payment', 0) >= 1:
            capabilities['customer_revenue_analysis'] = True
            capabilities['financial_reporting'] = True
        
        if entity_counts.get('Employee', 0) >= 1:
            capabilities['hr_analytics'] = True
        
        if entity_counts.get('Project', 0) >= 1:
            capabilities['project_management'] = True
        
        if entity_counts.get('Contract', 0) >= 1:
            capabilities['contract_management'] = True
        
        # Determine domain type with RDL context
        domain_confidence = 0.8
        if rdl_info and rdl_info.get('business_priority_signals'):
            domain_confidence = 0.95
        
        # Create appropriate domain based on entity mix
        if entity_counts.get('Contract', 0) >= 1 and rdl_info:
            domain_type = "Contract & Business Analytics"
            sample_questions = [
                "Show approved contracts by value",
                "What is our total contract value?",
                "List contracts by customer",
                "Show contract signing trends",
                "Who are our top contract customers?"
            ]
        elif entity_counts.get('Customer', 0) >= 1 and entity_counts.get('Payment', 0) >= 1:
            domain_type = "Customer & Financial Analytics"
            sample_questions = [
                "Who are our top paying customers?",
                "What is our total revenue?",
                "Show customer payment trends",
                "List high-value customers",
                "What's the average transaction amount?"
            ]
        elif entity_counts.get('Employee', 0) >= 1:
            domain_type = "HR & Employee Analytics"
            sample_questions = [
                "Show employee headcount",
                "List departments and staff",
                "Show employee performance data",
                "What are our salary trends?"
            ]
        elif entity_counts.get('Project', 0) >= 1:
            domain_type = "Project Management Analytics"
            sample_questions = [
                "Show active projects",
                "List project timelines",
                "Show project resource allocation",
                "What are our project completion rates?"
            ]
        else:
            domain_type = f"{primary_entity} Analytics"
            sample_questions = [
                f"Show {primary_entity.lower()} data",
                f"List all {primary_entity.lower()} records",
                f"What {primary_entity.lower()} information is available?",
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
        """Determine industry based on entity patterns"""
        # RDL context hints
        if rdl_info:
            report_title = rdl_info.get('report_title', '').lower()
            if 'contract' in report_title or 'συμβόλαια' in report_title:
                return "Contract Management"
        
        # Entity pattern analysis
        if entity_counts.get('Patient', 0) > 0:
            return "Healthcare"
        elif entity_counts.get('Student', 0) > 0:
            return "Education"
        elif entity_counts.get('Employee', 0) > 2:
            return "Human Resources"
        elif entity_counts.get('Project', 0) > 0:
            return "Project Management"
        elif entity_counts.get('Contract', 0) > 0:
            return "Legal/Contracts"
        elif entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0:
            return "Business/Commerce"
        else:
            return "Business Intelligence"

class SemanticAnalyzer:
    """Enhanced semantic analyzer with cross-industry support and RDL integration"""
    
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
        self.entity_analyzer = EnhancedEntityAnalyzer(self.llm)
        self.domain_analyzer = EnhancedDomainAnalyzer()
        
        # Enhanced cache manager
        self.cache_manager = CacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        
        print("✅ Enhanced Cross-Industry Semantic Analyzer initialized")
        print("   🌐 Cross-industry entity taxonomy")
        print("   📋 RDL business context integration")
        print("   🧠 Enhanced LLM analysis")
    
    async def analyze_tables(self, tables: List[TableInfo], discovery=None) -> bool:
        """Enhanced analysis with cross-industry entities and RDL integration"""
        print("🧠 ENHANCED CROSS-INDUSTRY SEMANTIC ANALYSIS")
        print("Architecture: Generalized entities + RDL business context")
        print("=" * 55)
        
        try:
            start_time = time.time()
            
            # Get RDL business context if available
            rdl_info = {}
            if discovery and hasattr(discovery, 'get_rdl_info'):
                rdl_info = discovery.get_rdl_info()
                if rdl_info:
                    print(f"   📋 Integrating RDL context: {rdl_info.get('report_title', 'Unknown report')}")
            
            # Enhanced entity analysis with RDL context
            self.tables = await self.entity_analyzer.analyze_tables_with_enhanced_taxonomy(tables, rdl_info)
            
            # Enhanced domain analysis
            self.domain = self.domain_analyzer.determine_enhanced_domain(self.tables, rdl_info)
            
            # Extract relationships
            self.relationships = self._extract_relationships()
            
            # Save enhanced cache
            self.cache_manager.save_analysis(self.tables, self.domain, self.relationships)
            
            # Show enhanced summary
            self._show_enhanced_summary(time.time() - start_time, rdl_info)
            
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
    
    def _show_enhanced_summary(self, elapsed_time: float, rdl_info: Dict):
        """Show enhanced analysis summary with cross-industry insights"""
        # Count entities across taxonomy
        entity_counts = {}
        for table in self.tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Count business attributes
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        high_priority = len([t for t in self.tables if getattr(t, 'business_priority', '') == 'high'])
        tables_with_names = len([t for t in self.tables if getattr(t, 'name_columns', [])])
        tables_with_measures = len([t for t in self.tables if getattr(t, 'measures', [])])
        
        print(f"\n✅ ENHANCED CROSS-INDUSTRY ANALYSIS COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Total objects: {len(self.tables)}")
        print(f"   🌐 Entity types found: {len(entity_counts)}")
        
        # Show top entity types
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        for entity_type, count in sorted_entities[:8]:
            priority_emoji = "🔥" if entity_type in ['Customer', 'Payment', 'Contract', 'Order'] else "📋"
            print(f"   {priority_emoji} {entity_type}: {count} objects")
        
        print(f"   📈 Fact tables: {fact_tables}")
        print(f"   🎯 High priority: {high_priority}")
        print(f"   📝 With names: {tables_with_names}")
        print(f"   📊 With measures: {tables_with_measures}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        # Show domain insights
        if self.domain:
            print(f"   🏢 Domain: {self.domain.domain_type}")
            print(f"   🏭 Industry: {self.domain.industry}")
            print(f"   💡 Capabilities: {len(self.domain.capabilities)} features")
        
        # Show RDL insights
        if rdl_info:
            print(f"   📋 RDL integration: {rdl_info.get('report_title', 'Unknown')}")
            if rdl_info.get('business_priority_signals'):
                print(f"   🎯 Business signals: {', '.join(rdl_info['business_priority_signals'])}")
    
    # Public API
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        # Count different entity types
        entity_counts = {}
        for table in self.tables:
            entity_type = table.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Count business attributes
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
            'analysis_method': 'enhanced_cross_industry_llm'
        }