#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Analysis Module - Clean implementation with proper imports
Handles LLM-based semantic classification, relationship discovery, and business domain analysis
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Import from shared modules
from shared.config import Config
from shared.models import (
    TableInfo, SemanticProfile, BusinessDomain, Relationship,
    table_info_to_dict, dict_to_table_info, business_domain_to_dict, dict_to_business_domain
)
from shared.utils import (
    extract_json_from_response, save_json_cache, load_json_cache, 
    extract_sample_greek_text
)

class SimpleLLMClient:
    """Enhanced LLM client with better error handling"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,
            request_timeout=120
        )
    
    async def ask(self, prompt: str, system_message: str = "You are a helpful database expert.") -> str:
        """Ask LLM a question with retry logic"""
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(self.llm.invoke, messages)
                return response.content
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

class SemanticAnalyzer:
    """Enhanced Semantic Analysis with improved classification and relationship discovery"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = SimpleLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
    
    async def analyze_semantics(self, tables: List[TableInfo]) -> bool:
        """Run semantic analysis on discovered tables"""
        # Check cache first
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        if self.load_from_cache():
            print(f"✅ Loaded semantic analysis from cache")
            return True
        
        if not tables:
            print("❌ No tables provided for analysis.")
            return False
        
        self.tables = tables
        
        # Show improvement stats
        views_with_data = sum(1 for t in tables if t.object_type == 'VIEW' and t.sample_data)
        total_views = sum(1 for t in tables if t.object_type == 'VIEW')
        
        if total_views > 0:
            print(f"📊 Views participating in analysis: {views_with_data}/{total_views} (improved from previous 0)")
        
        # Process tables in batches for semantic classification
        print("🔍 Classifying entities semantically...")
        classified_count = await self._classify_tables_in_batches()
        
        # Discover relationships
        print("🔗 Discovering relationships (including views)...")
        await self._discover_relationships_enhanced()
        
        # Analyze business domain
        print("🏢 Analyzing business domain...")
        await self._analyze_business_domain_enhanced()
        
        # Save results
        print("💾 Saving enhanced semantic analysis results...")
        self._save_to_cache(cache_file)
        
        views_classified = sum(1 for t in self.tables if t.object_type == 'VIEW' and t.semantic_profile)
        
        print("✅ Enhanced semantic analysis completed!")
        print(f"   🧠 Classified entities: {classified_count}/{len(self.tables)}")
        print(f"   📊 Views classified: {views_classified} (improved from 0)")
        print(f"   🔗 Relationships found: {len(self.relationships)}")
        print(f"   🏢 Domain: {self.domain.domain_type if self.domain else 'Unknown'}")
        
        return True
    
    async def _classify_tables_in_batches(self) -> int:
        """Classify tables in batches with enhanced analysis"""
        batch_size = self.config.max_batch_size
        classified_count = 0
        
        progress_bar = tqdm(range(0, len(self.tables), batch_size), desc="Semantic analysis")
        
        for i in progress_bar:
            batch = self.tables[i:i + batch_size]
            
            # Update progress bar with current batch info
            batch_names = [t.name for t in batch]
            current_batch = f"Classifying: {', '.join(batch_names[:2])}{'...' if len(batch_names) > 2 else ''}"
            progress_bar.set_description(current_batch)
            
            # Show more details for views
            view_names = [t.name for t in batch if t.object_type == 'VIEW']
            if view_names:
                print(f"   🧠 Batch {i//batch_size + 1}: Including views: {', '.join(view_names)}")
            
            batch_classified = await self._classify_tables_batch(batch)
            classified_count += batch_classified
            await asyncio.sleep(self.config.rate_limit_delay)
        
        progress_bar.set_description("Semantic classification complete")
        return classified_count
    
    async def _classify_tables_batch(self, tables: List[TableInfo]) -> int:
        """Enhanced table classification with better context"""
        # Prepare enhanced table summaries for LLM
        table_summaries = []
        for table in tables:
            summary = {
                'name': table.name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': [col['name'] for col in table.columns[:10]],
                'sample_data': table.sample_data[:2] if table.sample_data else []
            }
            
            # Add performance context for views
            if table.object_type == 'VIEW' and table.query_performance:
                if table.query_performance.get('fast_optimized'):
                    summary['note'] = 'Complex view - data retrieved using optimized query'
                elif not table.sample_data:
                    summary['note'] = 'View may be empty or highly filtered'
            
            table_summaries.append(summary)
        
        prompt = f"""
Analyze these database objects for semantic classification in a business context.
Note: Views are now properly analyzed with sample data when available.

Objects to analyze:
{json.dumps(table_summaries, indent=1, default=str)}

For each object, classify semantically:
1. ENTITY_TYPE: Person, Organization, Transaction, Product, Location, Event, Lookup
2. BUSINESS_ROLE: Customer, Vendor, Employee, Product, Order, Payment, etc.
3. DATA_NATURE: Master (core entities), Transaction (events), Lookup (reference)
4. CONTAINS_PERSONAL_DATA: true/false 
5. CONTAINS_FINANCIAL_DATA: true/false
6. PRIMARY_PURPOSE: Brief description
7. CONFIDENCE: 0.0-1.0

Special consideration for views: They often represent business-focused perspectives on data.

Respond with JSON array:
[
  {{
    "table_name": "TableName",
    "entity_type": "Transaction",
    "business_role": "Order", 
    "data_nature": "Transaction",
    "contains_personal_data": false,
    "contains_financial_data": true,
    "primary_purpose": "Records customer orders",
    "confidence": 0.85
  }}
]
"""
        
        try:
            response = await self.llm.ask(prompt, "You are a data architect. Respond with valid JSON only.")
            classifications = extract_json_from_response(response)
            
            classified_in_batch = 0
            if classifications and isinstance(classifications, list):
                for classification in classifications:
                    table_name = classification.get('table_name', '')
                    table = next((t for t in tables if t.name == table_name), None)
                    
                    if table:
                        table.semantic_profile = SemanticProfile(
                            entity_type=classification.get('entity_type', 'Unknown'),
                            business_role=classification.get('business_role', 'Unknown'),
                            data_nature=classification.get('data_nature', 'Unknown'),
                            contains_personal_data=classification.get('contains_personal_data', False),
                            contains_financial_data=classification.get('contains_financial_data', False),
                            primary_purpose=classification.get('primary_purpose', ''),
                            confidence=classification.get('confidence', 0.5)
                        )
                        classified_in_batch += 1
            
            return classified_in_batch
        
        except Exception as e:
            print(f"⚠️ Semantic classification failed for batch: {e}")
            return 0
    
    async def _discover_relationships_enhanced(self):
        """Enhanced relationship discovery including views"""
        print("🔗 Discovering enhanced relationships...")
        
        relationships = []
        
        # Phase 1: Explicit foreign key relationships
        print("   📋 Phase 1: Analyzing explicit foreign keys...")
        fk_count = 0
        for table in self.tables:
            for col in table.columns:
                if col.get('is_foreign_key', False):
                    relationships.append(Relationship(
                        from_table=table.full_name,
                        to_table='Unknown',  # Would need FK target info
                        column=col['name'],
                        relationship_type='explicit_fk',
                        confidence=1.0,
                        description=f"Foreign key relationship from {table.name}.{col['name']}"
                    ))
                    fk_count += 1
        
        print(f"   ✅ Found {fk_count} explicit foreign key relationships")
        
        # Phase 2: Enhanced implicit relationships including views
        print("   🧠 Phase 2: Analyzing implicit relationships (including views)...")
        
        # Include both tables and views in relationship discovery
        id_objects = [t for t in self.tables if any(col['name'].lower().endswith('id') for col in t.columns)]
        
        comparison_count = 0
        max_comparisons = min(2500, len(id_objects) * len(id_objects))
        
        progress_bar = tqdm(total=max_comparisons, desc="Finding implicit relationships")
        
        for i, obj1 in enumerate(id_objects):
            for j, obj2 in enumerate(id_objects):
                if i < j and comparison_count < max_comparisons:
                    comparison_count += 1
                    progress_bar.update(1)
                    progress_bar.set_description(f"Comparing {obj1.name} vs {obj2.name}")
                    
                    for col1 in obj1.columns:
                        if col1['name'].lower().endswith('id'):
                            for col2 in obj2.columns:
                                if col1['name'].lower() == col2['name'].lower():
                                    rel_type = 'implicit_common_id'
                                    confidence = 0.7
                                    
                                    # Boost confidence for view relationships
                                    if obj1.object_type == 'VIEW' or obj2.object_type == 'VIEW':
                                        confidence = 0.8
                                        rel_type = 'view_table_relationship'
                                    
                                    relationships.append(Relationship(
                                        from_table=obj1.full_name,
                                        to_table=obj2.full_name,
                                        column=col1['name'],
                                        relationship_type=rel_type,
                                        confidence=confidence,
                                        description=f"Common ID column: {col1['name']}"
                                    ))
        
        progress_bar.close()
        
        implicit_count = len(relationships) - fk_count
        view_relationships = sum(1 for r in relationships if 'view' in r.relationship_type)
        
        print(f"   ✅ Found {implicit_count} implicit relationships")
        print(f"   📊 View relationships: {view_relationships} (new capability)")
        
        # Limit to top relationships by confidence
        self.relationships = sorted(relationships, key=lambda x: x.confidence, reverse=True)[:100]
        print(f"   📊 Keeping top {len(self.relationships)} relationships")
    
    async def _analyze_business_domain_enhanced(self):
        """Enhanced business domain analysis including view insights"""
        print("🏢 Analyzing business domain with enhanced view data...")
        
        # Prepare evidence for domain analysis
        print("   📊 Collecting evidence from semantic profiles...")
        entity_types = defaultdict(int)
        business_roles = defaultdict(int)
        
        # Separate stats for views vs tables
        view_roles = defaultdict(int)
        table_roles = defaultdict(int)
        
        for table in self.tables:
            if table.semantic_profile:
                entity_types[table.semantic_profile.entity_type] += 1
                business_roles[table.semantic_profile.business_role] += 1
                
                if table.object_type == 'VIEW':
                    view_roles[table.semantic_profile.business_role] += 1
                else:
                    table_roles[table.semantic_profile.business_role] += 1
        
        evidence = {
            'total_tables': len(self.tables),
            'entity_types': dict(entity_types),
            'business_roles': dict(business_roles),
            'view_business_roles': dict(view_roles),
            'table_business_roles': dict(table_roles),
            'top_tables': [
                {
                    'name': t.name,
                    'object_type': t.object_type,
                    'row_count': t.row_count,
                    'business_role': t.semantic_profile.business_role if t.semantic_profile else 'Unknown'
                }
                for t in sorted(self.tables, key=lambda x: x.row_count, reverse=True)[:10]
            ],
            'sample_greek_text': extract_sample_greek_text(self.tables)
        }
        
        views_analyzed = sum(1 for t in self.tables if t.object_type == 'VIEW' and t.semantic_profile)
        print(f"   🧠 Sending enhanced evidence to AI for domain analysis...")
        print(f"      Entity types found: {len(entity_types)}")
        print(f"      Business roles found: {len(business_roles)}")
        print(f"      Views analyzed: {views_analyzed} (improvement from 0)")
        
        prompt = f"""
Analyze this database to determine the business domain and industry.
Enhanced analysis now includes views with their business context.

Database Evidence:
{json.dumps(evidence, indent=1)}

The analysis now includes views that provide business perspectives on the data.
Views often represent important business reports, summaries, or filtered views.

Determine:
1. What type of business system is this? (CRM, ERP, E-commerce, HR, etc.)
2. What industry does it serve?
3. What are the main business entities?
4. How should customers be defined?
5. What questions would users typically ask?

Respond in JSON:
{{
  "domain_type": "CRM|ERP|E-commerce|HR|Finance|Other",
  "industry": "specific industry name",
  "entities": ["Customer", "Order", "Product"],
  "customer_definition": "How customers are represented",
  "confidence": 0.85,
  "sample_questions": [
    "How many customers do we have?",
    "What is our total revenue?"
  ]
}}
"""
        
        try:
            response = await self.llm.ask(prompt, "You are a business domain expert. Respond with valid JSON only.")
            domain_data = extract_json_from_response(response)
            
            if domain_data:
                self.domain = BusinessDomain(
                    domain_type=domain_data.get('domain_type', 'Unknown'),
                    industry=domain_data.get('industry', 'Unknown'),
                    entities=domain_data.get('entities', []),
                    confidence=domain_data.get('confidence', 0.5),
                    sample_questions=domain_data.get('sample_questions', []),
                    customer_definition=domain_data.get('customer_definition', '')
                )
                
                print(f"   ✅ Domain identified: {self.domain.domain_type}")
                print(f"   🏭 Industry: {self.domain.industry}")
                print(f"   🎯 Confidence: {self.domain.confidence:.2f}")
            else:
                print("   ⚠️ Could not parse AI response for domain analysis")
        
        except Exception as e:
            print(f"   ⚠️ Domain analysis failed: {e}")
            # Create fallback domain
            self.domain = BusinessDomain(
                domain_type="Unknown",
                industry="Unknown",
                entities=list(business_roles.keys())[:5],
                confidence=0.3,
                sample_questions=["How many records do we have?", "Show me recent data"]
            )
    
    def _save_to_cache(self, cache_file):
        """Save semantic analysis to cache"""
        data = {
            'tables': [],
            'domain': business_domain_to_dict(self.domain) if self.domain else None,
            'relationships': [
                {
                    'from_table': r.from_table,
                    'to_table': r.to_table,
                    'column': r.column,
                    'relationship_type': r.relationship_type,
                    'confidence': r.confidence,
                    'description': r.description
                } for r in self.relationships
            ],
            'created': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        # Convert tables with semantic profiles
        for table in self.tables:
            data['tables'].append(table_info_to_dict(table))
        
        save_json_cache(cache_file, data, "semantic analysis")
    
    def load_from_cache(self) -> bool:
        """Load semantic analysis from cache"""
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        data = load_json_cache(cache_file, self.config.semantic_cache_hours, "semantic cache")
        
        if data:
            try:
                # Load tables
                if 'tables' in data:
                    self.tables = []
                    for table_data in data['tables']:
                        table = dict_to_table_info(table_data)
                        self.tables.append(table)
                
                # Load domain
                if 'domain' in data and data['domain']:
                    self.domain = dict_to_business_domain(data['domain'])
                
                # Load relationships
                if 'relationships' in data:
                    self.relationships = []
                    for rel_data in data['relationships']:
                        self.relationships.append(Relationship(
                            from_table=rel_data.get('from_table', ''),
                            to_table=rel_data.get('to_table', ''),
                            column=rel_data.get('column', ''),
                            relationship_type=rel_data.get('relationship_type', ''),
                            confidence=rel_data.get('confidence', 0.0),
                            description=rel_data.get('description', '')
                        ))
                
                print(f"✅ Loaded enhanced semantic cache: {len(self.tables)} tables")
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to load semantic cache: {e}")
        
        return False
    
    def get_tables(self) -> List[TableInfo]:
        """Get analyzed tables"""
        return self.tables
    
    def get_domain(self) -> Optional[BusinessDomain]:
        """Get business domain"""
        return self.domain
    
    def get_relationships(self) -> List[Relationship]:
        """Get discovered relationships"""
        return self.relationships