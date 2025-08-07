#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Semantic Analysis - Let LLM Decide Everything
Uses actual sample data instead of hardcoded patterns
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

class LLMClient:
    """Simple LLM client"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,
            request_timeout=60
        )
    
    async def analyze(self, prompt: str) -> str:
        """Analyze with LLM"""
        try:
            messages = [
                SystemMessage(content="You are a business analyst. Analyze database structures using sample data. Respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
            return ""

class SemanticAnalyzer:
    """Simple semantic analyzer using LLM + sample data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """Main analysis - let LLM decide everything"""
        
        if self.load_from_cache():
            return True
        
        if not tables:
            print("❌ No tables to analyze")
            return False
        
        print(f"🧠 Starting LLM-driven analysis of {len(tables)} tables...")
        
        self.tables = tables.copy()
        
        try:
            # Step 1: Let LLM classify tables using sample data
            print("   🤖 Step 1: LLM classifying tables using sample data...")
            await self.classify_with_llm()
            
            # Step 2: Find relationships
            print("   🔗 Step 2: Finding relationships...")
            self.find_relationships()
            
            # Step 3: Determine domain
            print("   🏢 Step 3: Determining business domain...")
            await self.determine_domain()
            
            # Step 4: Save results
            print("   💾 Step 4: Saving results...")
            self.save_to_cache()
            
            # Show summary
            self.show_results()
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
    
    async def classify_with_llm(self):
        """Let LLM classify tables using actual sample data"""
        
        # Process tables in batches
        batch_size = 10
        classified_count = 0
        
        for i in range(0, len(self.tables), batch_size):
            batch = self.tables[i:i+batch_size]
            
            # Create prompt with actual sample data
            prompt = self.create_classification_prompt(batch)
            
            # Get LLM response
            response = await self.llm.analyze(prompt)
            
            # Apply classifications
            classified_count += self.apply_classifications(response, batch)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        print(f"      ✅ Classified {classified_count} tables using LLM + sample data")
    
    def create_classification_prompt(self, tables_batch: List[TableInfo]) -> str:
        """Create prompt with actual sample data"""
        
        table_data = []
        for table in tables_batch:
            # Get actual sample data
            sample_preview = ""
            if table.sample_data:
                first_row = table.sample_data[0]
                sample_items = []
                for key, value in list(first_row.items())[:6]:
                    sample_items.append(f"{key}: {value}")
                sample_preview = ", ".join(sample_items)
            
            table_data.append({
                'table_name': table.full_name,
                'columns': [f"{col['name']} ({col['data_type']})" for col in table.columns[:10]],
                'sample_data': sample_preview,
                'row_count': table.row_count
            })
        
        return f"""
Look at these database tables with ACTUAL SAMPLE DATA and classify each one:

TABLES WITH REAL DATA:
{json.dumps(table_data, indent=2)}

For each table, look at the ACTUAL sample data and determine:
1. What type of business entity this represents
2. How confident you are (0.0 to 1.0)

Entity types:
- Customer: People, clients, accounts with names/contacts
- Payment: Financial transactions, payments, billing
- Order: Sales, purchases, bookings
- Product: Items, inventory, catalog
- User: System users, employees
- Company: Business entities, vendors
- Contact: Address/contact information
- Financial: Accounting, revenue data
- Reference: Lookup tables, codes
- System: Technical/operational tables
- Unknown: Cannot determine from data

Look at the ACTUAL sample data values to decide.

JSON format:
{{
  "classifications": [
    {{
      "table_name": "[schema].[table]",
      "entity_type": "Customer",
      "confidence": 0.9,
      "reasoning": "Sample data shows customer names, emails, addresses"
    }}
  ]
}}
"""
    
    def apply_classifications(self, response: str, batch: List[TableInfo]) -> int:
        """Apply LLM classifications to tables"""
        
        try:
            data = self.parse_json(response)
            if not data or 'classifications' not in data:
                return 0
            
            count = 0
            for classification in data['classifications']:
                table_name = classification.get('table_name', '')
                entity_type = classification.get('entity_type', 'Unknown')
                confidence = float(classification.get('confidence', 0.0))
                
                # Find matching table
                for table in batch:
                    if table.full_name == table_name:
                        table.entity_type = entity_type
                        table.confidence = confidence
                        table.business_role = 'Core' if confidence > 0.8 else 'Supporting'
                        count += 1
                        break
            
            return count
            
        except Exception as e:
            print(f"      ⚠️ Failed to parse classifications: {e}")
            return 0
    
    def find_relationships(self):
        """Find simple relationships between tables"""
        
        for table in self.tables:
            column_names = [col.get('name', '').lower() for col in table.columns]
            
            for col_name in column_names:
                if col_name.endswith('_id') or col_name.endswith('id'):
                    entity_name = col_name.replace('_id', '').replace('id', '')
                    
                    # Find related table
                    for other_table in self.tables:
                        if other_table.full_name == table.full_name:
                            continue
                        
                        if (entity_name in other_table.name.lower() or 
                            entity_name in other_table.entity_type.lower()):
                            
                            self.relationships.append(Relationship(
                                from_table=table.full_name,
                                to_table=other_table.full_name,
                                relationship_type='reference',
                                confidence=0.7,
                                description=f"Reference via {col_name}"
                            ))
        
        print(f"      ✅ Found {len(self.relationships)} relationships")
    
    async def determine_domain(self):
        """Let LLM determine business domain"""
        
        # Count entity types
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        prompt = f"""
Based on this entity distribution, determine the business domain:

ENTITY DISTRIBUTION:
{json.dumps(entity_counts, indent=2)}

What type of business system is this? Consider the entities present.

JSON format:
{{
  "domain_type": "E-Commerce",
  "confidence": 0.8,
  "sample_questions": [
    "How many customers do we have?",
    "What is our total revenue?"
  ]
}}
"""
        
        response = await self.llm.analyze(prompt)
        result = self.parse_json(response)
        
        if result:
            self.domain = BusinessDomain(
                domain_type=result.get('domain_type', 'Business'),
                industry='Business',
                confidence=result.get('confidence', 0.5),
                sample_questions=result.get('sample_questions', []),
                capabilities=self.determine_capabilities(entity_counts)
            )
        
        print(f"      ✅ Identified domain: {self.domain.domain_type if self.domain else 'Unknown'}")
    
    def determine_capabilities(self, entity_counts: Dict[str, int]) -> Dict[str, bool]:
        """Determine system capabilities"""
        return {
            'customer_analysis': entity_counts.get('Customer', 0) > 0,
            'payment_analysis': entity_counts.get('Payment', 0) > 0,
            'order_analysis': entity_counts.get('Order', 0) > 0,
            'product_analysis': entity_counts.get('Product', 0) > 0,
            'financial_reporting': entity_counts.get('Payment', 0) > 0 or entity_counts.get('Financial', 0) > 0,
            'cross_entity_analysis': len([c for c in entity_counts.values() if c > 0]) >= 2
        }
    
    def show_results(self):
        """Show analysis results"""
        
        classified = sum(1 for t in self.tables if t.entity_type != 'Unknown')
        
        print(f"\n📊 SEMANTIC ANALYSIS RESULTS:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   🧠 Classified: {classified}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        
        # Show entity distribution
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   🏢 Business Entities:")
            for entity_type, count in sorted(entity_counts.items()):
                print(f"      • {entity_type}: {count} tables")
        
        if self.domain:
            print(f"   🏢 Domain: {self.domain.domain_type}")
    
    def parse_json(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            import re
            cleaned = response.strip()
            
            # Remove markdown
            if '```json' in cleaned:
                match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            elif '```' in cleaned:
                match = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            return json.loads(cleaned)
        except:
            return {}
    
    def save_to_cache(self):
        """Save results to cache"""
        
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        tables_data = []
        for table in self.tables:
            tables_data.append({
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
            })
        
        relationships_data = []
        for rel in self.relationships:
            relationships_data.append({
                'from_table': rel.from_table,
                'to_table': rel.to_table,
                'relationship_type': rel.relationship_type,
                'confidence': rel.confidence,
                'description': rel.description
            })
        
        domain_data = None
        if self.domain:
            domain_data = {
                'domain_type': self.domain.domain_type,
                'industry': self.domain.industry,
                'confidence': self.domain.confidence,
                'sample_questions': self.domain.sample_questions,
                'capabilities': self.domain.capabilities
            }
        
        data = {
            'tables': tables_data,
            'relationships': relationships_data,
            'domain': domain_data,
            'created': datetime.now().isoformat(),
            'version': '2.0-simple'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def load_from_cache(self) -> bool:
        """Load from cache"""
        
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        if not cache_file.exists():
            return False
        
        try:
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.config.semantic_cache_hours * 3600):
                return False
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load tables
            if 'tables' in data:
                self.tables = []
                for table_data in data['tables']:
                    table = TableInfo(
                        name=table_data['name'],
                        schema=table_data['schema'],
                        full_name=table_data['full_name'],
                        object_type=table_data['object_type'],
                        row_count=table_data['row_count'],
                        columns=table_data['columns'],
                        sample_data=table_data['sample_data'],
                        relationships=table_data.get('relationships', [])
                    )
                    table.entity_type = table_data.get('entity_type', 'Unknown')
                    table.business_role = table_data.get('business_role', 'Unknown')
                    table.confidence = table_data.get('confidence', 0.0)
                    self.tables.append(table)
            
            # Load relationships
            if 'relationships' in data:
                self.relationships = []
                for rel_data in data['relationships']:
                    self.relationships.append(Relationship(
                        from_table=rel_data['from_table'],
                        to_table=rel_data['to_table'],
                        relationship_type=rel_data['relationship_type'],
                        confidence=rel_data['confidence'],
                        description=rel_data.get('description', '')
                    ))
            
            # Load domain
            if 'domain' in data and data['domain']:
                domain_data = data['domain']
                self.domain = BusinessDomain(
                    domain_type=domain_data['domain_type'],
                    industry=domain_data['industry'],
                    confidence=domain_data['confidence'],
                    sample_questions=domain_data['sample_questions'],
                    capabilities=domain_data['capabilities']
                )
            
            print(f"✅ Loaded semantic analysis from cache")
            print(f"   📊 Tables: {len(self.tables)}")
            print(f"   🔗 Relationships: {len(self.relationships)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
            return False
    
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain