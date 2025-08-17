#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Semantic Analysis
Following README: Classification, relationships, templates
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

class SemanticAnalyzer:
    """Simple semantic analyzer using LLM + sample data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            request_timeout=60,
            temperature=1.0  # Use default temperature
        )
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """Main analysis method"""
        print("🧠 Starting semantic analysis...")
        
        # Check cache first
        if self.load_from_cache():
            print(f"✅ Loaded from cache: {len(self.tables)} tables")
            return True
        
        try:
            self.tables = tables.copy()
            
            # Step 1: Classify tables using LLM with sample data
            await self.classify_tables()
            
            # Step 2: Enhance relationships
            self.enhance_relationships()
            
            # Step 3: Determine business domain
            await self.determine_domain()
            
            # Step 4: Save results
            self.save_to_cache()
            
            self.show_summary()
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
    
    async def classify_tables(self):
        """Classify tables using LLM with actual sample data"""
        print(f"🏷️ Classifying {len(self.tables)} tables...")
        
        # Process in batches to avoid token limits
        batch_size = 8
        classified = 0
        
        for i in range(0, len(self.tables), batch_size):
            batch = self.tables[i:i+batch_size]
            
            # Create classification prompt with sample data
            prompt = self.create_classification_prompt(batch)
            
            # Get LLM response
            response = await self.llm_analyze(prompt)
            
            # Apply classifications
            classified += self.apply_classifications(response, batch)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        print(f"   ✅ Classified {classified} tables")
    
    def create_classification_prompt(self, tables: List[TableInfo]) -> str:
        """Create classification prompt with actual sample data"""
        
        table_summaries = []
        for table in tables:
            # Get sample data preview
            sample_preview = ""
            if table.sample_data:
                first_row = table.sample_data[0]
                sample_items = []
                for key, value in list(first_row.items())[:4]:
                    if key != "__edge":  # Skip our metadata
                        sample_items.append(f"{key}: {value}")
                sample_preview = ", ".join(sample_items)
            
            table_summaries.append({
                'table_name': table.full_name,
                'columns': [f"{col['name']} ({col['data_type']})" for col in table.columns[:6]],
                'sample_data': sample_preview,
                'row_count': table.row_count
            })
        
        return f"""
Analyze these database tables and classify each one based on the actual sample data and column names:

TABLES:
{json.dumps(table_summaries, indent=2)}

For each table, determine:
1. Entity type based on the actual sample data and column names
2. Business role (Core or Supporting)
3. Confidence level (0.0 to 1.0)

Entity types:
- Customer: People, clients, accounts, users
- Payment: Financial transactions, billing, invoices
- Order: Sales, purchases, bookings, contracts
- Product: Items, inventory, catalog
- Company: Business entities, vendors
- Financial: Accounting, revenue data
- Reference: Lookup tables, codes, categories
- System: Technical/operational tables
- Unknown: Cannot determine

Business roles:
- Core: Central business entities (customers, orders, payments)
- Supporting: Reference data, lookup tables, configuration

Look at the actual sample data values and column names to decide.

Respond with JSON only:
{{
  "classifications": [
    {{
      "table_name": "[schema].[table]",
      "entity_type": "Customer",
      "business_role": "Core",
      "confidence": 0.9,
      "reasoning": "Sample data shows customer names and contact info"
    }}
  ]
}}
"""
    
    async def llm_analyze(self, prompt: str) -> str:
        """Send analysis request to LLM"""
        try:
            messages = [
                SystemMessage(content="You are a database analyst. Analyze tables using sample data and respond with valid JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
            return ""
    
    def apply_classifications(self, response: str, batch: List[TableInfo]) -> int:
        """Apply LLM classifications to tables"""
        
        data = parse_json_response(response)
        if not data or 'classifications' not in data:
            return 0
        
        count = 0
        for classification in data['classifications']:
            table_name = classification.get('table_name', '')
            entity_type = classification.get('entity_type', 'Unknown')
            business_role = classification.get('business_role', 'Supporting')
            confidence = float(classification.get('confidence', 0.0))
            
            # Find and update table
            for table in batch:
                if table.full_name == table_name:
                    table.entity_type = entity_type
                    table.business_role = business_role
                    table.confidence = confidence
                    count += 1
                    break
        
        return count
    
    def enhance_relationships(self):
        """Enhance relationships based on semantic classification"""
        print("🔗 Enhancing relationships...")
        
        # Copy existing relationships
        existing_relationships = []
        for table in self.tables:
            for rel_info in table.relationships:
                if '->' in rel_info:
                    parts = rel_info.split('->', 1)
                    from_col = parts[0].strip()
                    to_ref = parts[1].strip()
                    
                    existing_relationships.append(Relationship(
                        from_table=table.full_name,
                        to_table=to_ref.split('.')[0] if '.' in to_ref else to_ref,
                        relationship_type='foreign_key',
                        confidence=0.95,
                        description=f"FK: {from_col} -> {to_ref}"
                    ))
        
        # Add pattern-based relationships using semantic types
        table_by_entity = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                if table.entity_type not in table_by_entity:
                    table_by_entity[table.entity_type] = []
                table_by_entity[table.entity_type].append(table)
        
        # Find cross-entity relationships
        for table in self.tables:
            for col in table.columns:
                col_name = col.get('name', '').lower()
                
                # Look for foreign key patterns
                if col_name.endswith('id') and col_name != 'id':
                    entity_name = col_name[:-2]
                    
                    # Map common entity patterns
                    entity_mapping = {
                        'customer': 'Customer',
                        'client': 'Customer', 
                        'user': 'Customer',
                        'payment': 'Payment',
                        'order': 'Order',
                        'product': 'Product',
                        'company': 'Company'
                    }
                    
                    target_entity = entity_mapping.get(entity_name)
                    if target_entity and target_entity in table_by_entity:
                        for target_table in table_by_entity[target_entity]:
                            if target_table.full_name != table.full_name:
                                existing_relationships.append(Relationship(
                                    from_table=table.full_name,
                                    to_table=target_table.full_name,
                                    relationship_type='semantic_reference',
                                    confidence=0.8,
                                    description=f"Semantic link via {col_name}"
                                ))
        
        self.relationships = existing_relationships
        print(f"   ✅ Enhanced to {len(self.relationships)} relationships")
    
    async def determine_domain(self):
        """Determine business domain using entity distribution"""
        print("🏢 Determining business domain...")
        
        # Count entity types
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        # Create analysis prompt
        prompt = f"""
Based on this entity distribution, determine the business domain:

ENTITY DISTRIBUTION:
{json.dumps(entity_counts, indent=2)}

Determine:
1. What type of business system this represents
2. Key business capabilities
3. Sample questions users might ask

Common domains:
- E-Commerce: Customer, Product, Order, Payment entities
- CRM: Customer, Company, Contact entities  
- Financial: Payment, Financial, Account entities
- ERP: Product, Order, Company, Financial entities

Respond with JSON only:
{{
  "domain_type": "E-Commerce",
  "confidence": 0.8,
  "capabilities": ["customer_analysis", "payment_analysis", "order_analysis"],
  "sample_questions": [
    "How many customers do we have?",
    "What is our total revenue this year?"
  ]
}}
"""
        
        response = await self.llm_analyze(prompt)
        result = parse_json_response(response)
        
        if result:
            capabilities_dict = {cap: True for cap in result.get('capabilities', [])}
            
            self.domain = BusinessDomain(
                domain_type=result.get('domain_type', 'Business'),
                industry='Business',
                confidence=result.get('confidence', 0.5),
                sample_questions=result.get('sample_questions', []),
                capabilities=capabilities_dict
            )
        
        print(f"   ✅ Domain: {self.domain.domain_type if self.domain else 'Unknown'}")
    
    def show_summary(self):
        """Show analysis summary"""
        classified = sum(1 for t in self.tables if t.entity_type != 'Unknown')
        core_tables = sum(1 for t in self.tables if t.business_role == 'Core')
        
        print(f"\n📊 SEMANTIC ANALYSIS SUMMARY:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   🧠 Classified: {classified}")
        print(f"   ⭐ Core business tables: {core_tables}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        # Show entity distribution
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   🏢 Entity types:")
            for entity_type, count in sorted(entity_counts.items()):
                print(f"      • {entity_type}: {count}")
        
        if self.domain:
            print(f"   🎯 Domain: {self.domain.domain_type} (confidence: {self.domain.confidence:.2f})")
    
    def save_to_cache(self):
        """Save analysis to cache"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        data = {
            'tables': [self.table_to_dict(t) for t in self.tables],
            'relationships': [self.relationship_to_dict(r) for r in self.relationships],
            'domain': self.domain_to_dict() if self.domain else None,
            'analyzed': datetime.now().isoformat(),
            'version': '2.0-simple'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def load_from_cache(self) -> bool:
        """Load analysis from cache if fresh"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        if not cache_file.exists():
            return False
        
        try:
            # Check cache age
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.config.semantic_cache_hours * 3600):
                return False
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load tables
            self.tables = [self.dict_to_table(t) for t in data.get('tables', [])]
            
            # Load relationships
            self.relationships = [self.dict_to_relationship(r) for r in data.get('relationships', [])]
            
            # Load domain
            if data.get('domain'):
                self.domain = self.dict_to_domain(data['domain'])
            
            return True
            
        except Exception:
            return False
    
    def table_to_dict(self, table: TableInfo) -> Dict:
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
            'confidence': table.confidence
        }
    
    def dict_to_table(self, data: Dict) -> TableInfo:
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
        table.entity_type = data.get('entity_type', 'Unknown')
        table.business_role = data.get('business_role', 'Unknown')
        table.confidence = data.get('confidence', 0.0)
        return table
    
    def relationship_to_dict(self, rel: Relationship) -> Dict:
        """Convert Relationship to dictionary"""
        return {
            'from_table': rel.from_table,
            'to_table': rel.to_table,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'description': rel.description
        }
    
    def dict_to_relationship(self, data: Dict) -> Relationship:
        """Convert dictionary to Relationship"""
        return Relationship(
            from_table=data['from_table'],
            to_table=data['to_table'],
            relationship_type=data['relationship_type'],
            confidence=data['confidence'],
            description=data.get('description', '')
        )
    
    def domain_to_dict(self) -> Dict:
        """Convert BusinessDomain to dictionary"""
        return {
            'domain_type': self.domain.domain_type,
            'industry': self.domain.industry,
            'confidence': self.domain.confidence,
            'sample_questions': self.domain.sample_questions,
            'capabilities': self.domain.capabilities
        }
    
    def dict_to_domain(self, data: Dict) -> BusinessDomain:
        """Convert dictionary to BusinessDomain"""
        return BusinessDomain(
            domain_type=data['domain_type'],
            industry=data['industry'],
            confidence=data['confidence'],
            sample_questions=data['sample_questions'],
            capabilities=data['capabilities']
        )
    
    # Public interface methods
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain