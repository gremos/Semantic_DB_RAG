#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Semantic Analysis - Better BI Classification
Following README: Enhanced entity recognition and capability assessment
Simple, Readable, Maintainable
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

class EnhancedTableClassifier:
    """Enhanced table classification with better entity recognition"""
    
    def __init__(self, llm_analyzer):
        self.llm_analyzer = llm_analyzer
    
    async def classify_tables(self, tables: List[TableInfo]) -> int:
        """Enhanced classification with better business intelligence"""
        print(f"🏷️ Enhanced classification of {len(tables)} tables...")
        
        # First pass: Rule-based classification for speed
        self._rule_based_classification(tables)
        
        # Second pass: LLM classification for complex cases
        batch_size = 5
        classified = 0
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(batch)
            
            # Get LLM response
            response = await self.llm_analyzer.analyze(
                "You are a BI expert. Classify database tables for business intelligence. Focus on identifying customer tables, payment/transaction tables, and their relationships.",
                prompt
            )
            
            # Apply enhanced classifications
            classified += self._apply_enhanced_classifications(response, batch)
            await asyncio.sleep(0.3)  # Rate limiting
        
        # Third pass: Post-processing and validation
        self._post_process_classifications(tables)
        
        print(f"   ✅ Enhanced classification completed: {classified} tables")
        return classified
    
    def _rule_based_classification(self, tables: List[TableInfo]):
        """Fast rule-based classification"""
        for table in tables:
            table_name = table.name.lower()
            
            # Analyze columns for quick patterns
            column_analysis = self._analyze_columns_enhanced(table)
            
            # Customer entity detection
            if any(word in table_name for word in ['customer', 'client', 'account', 'buyer']):
                table.entity_type = 'Customer'
                table.data_type = 'reference'
                table.bi_role = 'dimension'
                table.grain = 'customer'
                table.entity_keys = column_analysis['entity_keys']
                table.confidence = 0.9
            
            # Payment/Transaction entity detection  
            elif any(word in table_name for word in ['payment', 'transaction', 'invoice', 'billing', 'revenue']):
                table.entity_type = 'Payment'
                table.data_type = 'operational'
                table.bi_role = 'fact'
                table.grain = 'transaction'
                table.measures = column_analysis['measures']
                table.entity_keys = column_analysis['entity_keys']
                table.time_columns = column_analysis['time_columns']
                table.confidence = 0.85
            
            # Order/Sales entity detection
            elif any(word in table_name for word in ['order', 'sale', 'purchase', 'contract']):
                table.entity_type = 'Order'
                table.data_type = 'operational'
                table.bi_role = 'fact'
                table.grain = 'order'
                table.measures = column_analysis['measures']
                table.entity_keys = column_analysis['entity_keys']
                table.time_columns = column_analysis['time_columns']
                table.confidence = 0.8
            
            # Product entity detection
            elif any(word in table_name for word in ['product', 'item', 'service', 'offering']):
                table.entity_type = 'Product'
                table.data_type = 'reference'
                table.bi_role = 'dimension'
                table.grain = 'product'
                table.entity_keys = column_analysis['entity_keys']
                table.confidence = 0.8
            
            # Campaign/Marketing entity detection
            elif any(word in table_name for word in ['campaign', 'marketing', 'promotion']):
                table.entity_type = 'Campaign'
                table.data_type = 'operational'
                table.bi_role = 'fact' if column_analysis['measures'] else 'dimension'
                table.grain = 'campaign'
                table.measures = column_analysis['measures']
                table.entity_keys = column_analysis['entity_keys']
                table.confidence = 0.7
            
            # Default classification
            else:
                table.entity_type = 'Unknown'
                table.data_type = 'reference'
                table.bi_role = 'dimension'
                table.grain = 'unknown'
                table.confidence = 0.3
            
            # Store enhanced column analysis
            table.measures = column_analysis['measures']
            table.entity_keys = column_analysis['entity_keys']
            table.time_columns = column_analysis['time_columns']
            table.filter_columns = column_analysis['filter_columns']
    
    def _analyze_columns_enhanced(self, table: TableInfo) -> Dict[str, List[str]]:
        """Enhanced column analysis for better BI patterns"""
        analysis = {
            'measures': [], 
            'entity_keys': [], 
            'time_columns': [], 
            'filter_columns': [],
            'name_columns': []
        }
        
        for col in table.columns:
            col_name = col.get('name', '').lower()
            col_type = col.get('data_type', '').lower()
            
            # Identify measure columns (more comprehensive)
            if any(t in col_type for t in ['decimal', 'money', 'float', 'numeric']):
                if any(w in col_name for w in [
                    'amount', 'value', 'price', 'total', 'cost', 'revenue', 
                    'sum', 'balance', 'payment', 'charge', 'fee'
                ]):
                    analysis['measures'].append(col.get('name'))
            elif 'int' in col_type:
                if any(w in col_name for w in [
                    'quantity', 'count', 'number', 'qty', 'volume'
                ]):
                    analysis['measures'].append(col.get('name'))
            
            # Identify entity key columns (enhanced)
            if col_name.endswith('id') or col_name.endswith('key'):
                analysis['entity_keys'].append(col.get('name'))
            elif any(w in col_name for w in ['customerid', 'clientid', 'accountid', 'userid']):
                analysis['entity_keys'].append(col.get('name'))
            
            # Identify time columns (comprehensive)
            if any(t in col_type for t in ['date', 'time', 'timestamp']):
                analysis['time_columns'].append(col.get('name'))
            elif any(w in col_name for w in [
                'date', 'time', 'created', 'modified', 'updated', 'start', 'end'
            ]):
                analysis['time_columns'].append(col.get('name'))
            
            # Identify name/title columns
            if any(w in col_name for w in ['name', 'title', 'label', 'description']):
                analysis['name_columns'].append(col.get('name'))
            
            # Identify filter columns (enhanced)
            if any(w in col_name for w in [
                'status', 'state', 'type', 'category', 'class', 'region', 
                'priority', 'level', 'grade', 'active', 'enabled'
            ]):
                analysis['filter_columns'].append(col.get('name'))
        
        return analysis
    
    def _create_enhanced_prompt(self, tables: List[TableInfo]) -> str:
        """Create enhanced classification prompt"""
        table_summaries = []
        for table in tables:
            # Enhanced table analysis
            column_analysis = self._analyze_columns_enhanced(table)
            
            # Build comprehensive table summary
            table_summary = {
                'table_name': table.full_name,
                'estimated_rows': table.row_count,
                'columns': self._get_column_summary(table),
                'measures': column_analysis['measures'],
                'entity_keys': column_analysis['entity_keys'],
                'time_columns': column_analysis['time_columns'],
                'name_columns': column_analysis['name_columns'],
                'sample': self._get_enhanced_sample_preview(table)
            }
            table_summaries.append(table_summary)
        
        return f"""
Analyze these tables for BUSINESS INTELLIGENCE with focus on customer and payment data:

CLASSIFICATION REQUIREMENTS:
1. ENTITY_TYPE: Customer, Payment, Order, Product, Campaign, or specific business entity
2. DATA_TYPE: operational (transactions), reference (master data), planning (targets)
3. BI_ROLE: fact (contains measures), dimension (contains attributes), bridge (relationships)
4. GRAIN: what each row represents (customer, transaction, order, product, etc.)
5. MEASURES: numeric columns for aggregation (amounts, quantities, counts)
6. ENTITY_KEYS: foreign keys for joining (CustomerID, ProductID, etc.)
7. TIME_COLUMNS: date/time columns for filtering
8. NAME_COLUMNS: text columns with names/titles

BUSINESS CONTEXT:
- Customer tables: contain customer information, names, demographics
- Payment tables: contain transaction amounts, payment data, revenue
- Order tables: contain order information, line items, sales data
- Look for relationships: CustomerID in payment tables links to Customer table

TABLES TO ANALYZE:
{json.dumps(table_summaries, indent=2)}

Respond with JSON only:
{{
  "classifications": [
    {{
      "table_name": "[schema].[table]",
      "entity_type": "Customer|Payment|Order|Product|Campaign|Other",
      "data_type": "operational|reference|planning",
      "bi_role": "fact|dimension|bridge",
      "grain": "customer|transaction|order|product|campaign|other",
      "measures": ["amount_column", "quantity_column"],
      "entity_keys": ["customer_id", "product_id"],
      "time_columns": ["date_column"],
      "name_columns": ["name_column", "title_column"],
      "filter_columns": ["status", "type"],
      "confidence": 0.9,
      "business_priority": "high|medium|low"
    }}
  ]
}}
"""
    
    def _get_column_summary(self, table: TableInfo) -> List[Dict[str, str]]:
        """Get enhanced column summary"""
        column_summary = []
        for col in table.columns[:12]:  # Limit to first 12 columns
            column_summary.append({
                'name': col.get('name', ''),
                'type': col.get('data_type', ''),
                'nullable': col.get('is_nullable', True)
            })
        return column_summary
    
    def _get_enhanced_sample_preview(self, table: TableInfo) -> str:
        """Get enhanced sample data preview"""
        if not table.sample_data:
            return "No sample data"
        
        first_row = table.sample_data[0]
        preview = []
        
        # Show relevant columns first
        priority_columns = []
        other_columns = []
        
        for key, value in first_row.items():
            if key.startswith('__'):
                continue
                
            key_lower = key.lower()
            if any(word in key_lower for word in ['name', 'customer', 'amount', 'total', 'date']):
                priority_columns.append(f"{key}={value}")
            else:
                other_columns.append(f"{key}={value}")
        
        # Combine with priority columns first
        preview = priority_columns[:3] + other_columns[:2]
        return " | ".join(preview)
    
    def _apply_enhanced_classifications(self, response: str, batch: List[TableInfo]) -> int:
        """Apply enhanced classifications to tables"""
        data = parse_json_response(response)
        if not data or 'classifications' not in data:
            return 0
        
        count = 0
        for classification in data['classifications']:
            table_name = classification.get('table_name', '')
            
            for table in batch:
                if table.full_name == table_name:
                    # Apply enhanced classifications
                    table.entity_type = classification.get('entity_type', table.entity_type)
                    table.data_type = classification.get('data_type', table.data_type)
                    table.bi_role = classification.get('bi_role', table.bi_role)
                    table.grain = classification.get('grain', table.grain)
                    
                    # Update measures and keys from LLM (merge with existing)
                    llm_measures = classification.get('measures', [])
                    existing_measures = getattr(table, 'measures', [])
                    table.measures = list(set(llm_measures + existing_measures))
                    
                    llm_entity_keys = classification.get('entity_keys', [])
                    existing_keys = getattr(table, 'entity_keys', [])
                    table.entity_keys = list(set(llm_entity_keys + existing_keys))
                    
                    llm_time_cols = classification.get('time_columns', [])
                    existing_time = getattr(table, 'time_columns', [])
                    table.time_columns = list(set(llm_time_cols + existing_time))
                    
                    llm_name_cols = classification.get('name_columns', [])
                    table.name_columns = llm_name_cols
                    
                    # Update confidence and priority
                    table.confidence = max(table.confidence, float(classification.get('confidence', 0.0)))
                    table.business_priority = classification.get('business_priority', 'medium')
                    
                    count += 1
                    break
        
        return count
    
    def _post_process_classifications(self, tables: List[TableInfo]):
        """Post-process classifications for consistency"""
        customer_tables = [t for t in tables if t.entity_type == 'Customer']
        payment_tables = [t for t in tables if t.entity_type == 'Payment']
        
        print(f"   📊 Post-processing: {len(customer_tables)} customer tables, {len(payment_tables)} payment tables")
        
        # Enhance customer tables
        for table in customer_tables:
            if not hasattr(table, 'name_columns'):
                table.name_columns = []
            
            # Look for name columns if not already identified
            if not table.name_columns:
                for col in table.columns:
                    col_name = col.get('name', '').lower()
                    if any(word in col_name for word in ['name', 'title', 'label']):
                        table.name_columns.append(col.get('name'))
        
        # Enhance payment/transaction tables
        for table in payment_tables:
            # Ensure they have measures
            if not table.measures:
                for col in table.columns:
                    col_name = col.get('name', '').lower()
                    col_type = col.get('data_type', '').lower()
                    if any(t in col_type for t in ['decimal', 'money', 'numeric']):
                        if any(w in col_name for w in ['amount', 'total', 'value', 'price']):
                            table.measures.append(col.get('name'))
                            break
            
            # Boost confidence for payment tables with measures
            if table.measures:
                table.confidence = min(1.0, table.confidence + 0.1)


class EnhancedSemanticAnalyzer:
    """Enhanced semantic analyzer with better BI classification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_analyzer = LLMAnalyzer(config)
        self.table_classifier = EnhancedTableClassifier(self.llm_analyzer)
        self.domain_analyzer = EnhancedDomainAnalyzer()
        self.cache_manager = CacheManager(config)
        
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """Enhanced analysis with better BI awareness"""
        print("🧠 Enhanced Semantic Analysis")
        print("Better customer/payment entity recognition and capability assessment")
        
        try:
            # Copy input tables
            self.tables = [table for table in tables]
            
            # Enhanced table classification
            await self.table_classifier.classify_tables(self.tables)
            
            # Enhance relationships
            self.relationships = self._enhance_relationships()
            
            # Enhanced domain analysis
            self.domain = self.domain_analyzer.determine_domain(self.tables)
            
            # Save cache
            self.cache_manager.save_cache(self.tables, self.domain, self.relationships)
            
            # Show enhanced summary
            self._show_enhanced_summary()
            return True
            
        except Exception as e:
            print(f"❌ Enhanced analysis failed: {e}")
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
        """Enhanced relationship extraction"""
        relationships = []
        
        for table in self.tables:
            for rel_info in table.relationships:
                if '->' in rel_info:
                    try:
                        parts = rel_info.split('->', 1)
                        from_col = parts[0].strip()
                        to_ref = parts[1].strip()
                        
                        # Enhanced relationship classification
                        rel_type = 'foreign_key'
                        confidence = 0.95
                        
                        # Check for fact-dimension patterns
                        if table.bi_role == 'fact' and 'customer' in from_col.lower():
                            rel_type = 'fact_dimension'
                            confidence = 0.98
                        
                        relationships.append(Relationship(
                            from_table=table.full_name,
                            to_table=to_ref.split('.')[0] if '.' in to_ref else to_ref,
                            relationship_type=rel_type,
                            confidence=confidence,
                            description=f"Enhanced: {from_col} -> {to_ref}"
                        ))
                    except Exception:
                        continue
        
        return relationships
    
    def _show_enhanced_summary(self):
        """Show enhanced analysis summary"""
        customer_tables = len([t for t in self.tables if t.entity_type == 'Customer'])
        payment_tables = len([t for t in self.tables if t.entity_type == 'Payment'])
        fact_tables = len([t for t in self.tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        with_measures = len([t for t in self.tables if getattr(t, 'measures', [])])
        with_names = len([t for t in self.tables if getattr(t, 'name_columns', [])])
        
        print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   👥 Customer tables: {customer_tables}")
        print(f"   💳 Payment tables: {payment_tables}")
        print(f"   📊 Fact tables: {fact_tables}")
        print(f"   ⚡ Operational tables: {operational_tables}")
        print(f"   📈 Tables with measures: {with_measures}")
        print(f"   📝 Tables with name columns: {with_names}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        if self.domain:
            print(f"   🎯 Domain: {self.domain.domain_type}")
        
        # Show top customer and payment tables
        customer_tables_list = [t for t in self.tables if t.entity_type == 'Customer']
        payment_tables_list = [t for t in self.tables if t.entity_type == 'Payment']
        
        if customer_tables_list:
            print(f"   👥 Top customer tables: {', '.join([t.name for t in customer_tables_list[:3]])}")
        
        if payment_tables_list:
            print(f"   💳 Top payment tables: {', '.join([t.name for t in payment_tables_list[:3]])}")
    
    # Public interface - Enhanced API
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
        operational_tables = len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational'])
        
        return {
            'total_tables': len(self.tables),
            'customer_tables': customer_tables,
            'payment_tables': payment_tables,
            'fact_tables': fact_tables,
            'operational_tables': operational_tables,
            'total_relationships': len(self.relationships),
            'domain_type': self.domain.domain_type if self.domain else None
        }


class EnhancedDomainAnalyzer:
    """Enhanced business domain analysis"""
    
    def determine_domain(self, tables: List[TableInfo]) -> Optional[BusinessDomain]:
        """Enhanced domain analysis with customer/payment focus"""
        customer_tables = len([t for t in tables if t.entity_type == 'Customer'])
        payment_tables = len([t for t in tables if t.entity_type == 'Payment'])
        fact_tables = len([t for t in tables if getattr(t, 'bi_role', '') == 'fact'])
        operational_tables = len([t for t in tables if getattr(t, 'data_type', '') == 'operational'])
        
        # Enhanced domain classification
        if customer_tables >= 1 and payment_tables >= 1:
            domain_type = "Customer Analytics System"
            confidence = 0.95
            capabilities = {
                'customer_analysis': True,
                'payment_analysis': True,
                'customer_segmentation': customer_tables >= 2,
                'revenue_analysis': payment_tables >= 1,
                'operational_reporting': operational_tables > 0
            }
            
            sample_questions = [
                "Who are our top 10 paying customers?",
                "What is our total revenue this quarter?", 
                "Which customers have the highest lifetime value?",
                "Show me customer payment trends by month",
                "List our most active customers by transaction count"
            ]
        
        elif fact_tables >= 2 and operational_tables >= 3:
            domain_type = "Operational BI System"
            confidence = 0.8
            capabilities = {
                'operational_reporting': True,
                'trend_analysis': True,
                'performance_metrics': True,
                'customer_analysis': customer_tables > 0,
                'financial_analysis': payment_tables > 0
            }
            
            sample_questions = [
                "What are our key performance metrics?",
                "Show me operational trends over time",
                "Which processes are most efficient?"
            ]
        
        else:
            domain_type = "Business Information System"
            confidence = 0.6
            capabilities = {
                'basic_reporting': True,
                'data_exploration': True,
                'customer_analysis': customer_tables > 0,
                'financial_analysis': payment_tables > 0
            }
            
            sample_questions = [
                "Show me a summary of available data",
                "What information do we have about customers?",
                "List the main business entities"
            ]
        
        return BusinessDomain(
            domain_type=domain_type,
            industry="Business Intelligence",
            confidence=confidence,
            sample_questions=sample_questions,
            capabilities=capabilities
        )


# Keep existing classes unchanged
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
            'version': '2.1-enhanced'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Enhanced analysis cache saved")
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
            'filter_columns': getattr(table, 'filter_columns', []),
            'name_columns': getattr(table, 'name_columns', []),
            'business_priority': getattr(table, 'business_priority', 'medium')
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
        
        # Enhanced BI-aware properties
        table.data_type = data.get('data_type', 'reference')
        table.bi_role = data.get('bi_role', 'dimension')
        table.grain = data.get('grain', 'unknown')
        table.measures = data.get('measures', [])
        table.entity_keys = data.get('entity_keys', [])
        table.time_columns = data.get('time_columns', [])
        table.filter_columns = data.get('filter_columns', [])
        table.name_columns = data.get('name_columns', [])
        table.business_priority = data.get('business_priority', 'medium')
        
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


# Create alias for backward compatibility
SemanticAnalyzer = EnhancedSemanticAnalyzer