#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Semantic Analysis - Readable and Maintainable
Focus on core functionality: entity classification and relationship discovery
"""

import asyncio
import json
import pyodbc
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Import from simplified modules
from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, dict_to_table, table_to_dict
from shared.utils import (
    extract_json_from_response, classify_entity_type_simple, 
    find_table_relationships_simple, create_llm_classification_prompt,
    save_cache, load_cache
)

class SimpleLLMClient:
    """Simple LLM client for entity classification"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,
            request_timeout=60
        )
    
    async def ask(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        try:
            messages = [
                SystemMessage(content="You are an expert database analyst. Respond with valid JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM request failed: {e}")
            return ""

class SimpleSemanticAnalyzer:
    """Simplified semantic analyzer focusing on core functionality"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = SimpleLLMClient(config)
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """Main analysis method - simplified and focused"""
        
        # Check cache first
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        cached_data = load_cache(cache_file, self.config.semantic_cache_hours)
        
        if cached_data:
            print("✅ Loading from semantic cache...")
            self._load_from_cache(cached_data)
            return True
        
        if not tables:
            print("❌ No tables to analyze")
            return False
        
        print(f"🧠 Starting semantic analysis of {len(tables)} tables...")
        
        try:
            # Step 1: Simple pattern-based classification
            print("   📊 Step 1: Pattern-based entity classification...")
            self._classify_entities_simple(tables)
            
            # Step 2: LLM-enhanced classification for unclear cases
            print("   🤖 Step 2: LLM enhancement for unclear entities...")
            await self._enhance_with_llm(tables)
            
            # Step 3: Find relationships
            print("   🔗 Step 3: Discovering relationships...")
            self._find_relationships(tables)
            
            # Step 4: Analyze business domain
            print("   🏢 Step 4: Analyzing business domain...")
            self._analyze_domain(tables)
            
            # Step 5: Save to cache
            print("   💾 Step 5: Saving analysis to cache...")
            self._save_to_cache(cache_file)
            
            # Show results
            self._show_results()
            
            print("✅ Semantic analysis completed!")
            return True
            
        except Exception as e:
            print(f"❌ Semantic analysis failed: {e}")
            return False
    
    def _classify_entities_simple(self, tables: List[TableInfo]):
        """Step 1: Simple pattern-based classification"""
        
        classified_count = 0
        
        for table in tables:
            entity_type, confidence = classify_entity_type_simple(
                table.name, table.columns, table.sample_data
            )
            
            if entity_type != 'Unknown':
                table.entity_type = entity_type
                table.confidence = confidence
                table.business_role = 'Core' if confidence > 0.8 else 'Supporting'
                classified_count += 1
        
        print(f"      ✅ Classified {classified_count} tables using patterns")
    
    async def _enhance_with_llm(self, tables: List[TableInfo]):
        """Step 2: Use LLM to classify unclear cases"""
        
        # Find tables that need LLM analysis (low confidence or unknown)
        unclear_tables = [t for t in tables if t.confidence < 0.6 or t.entity_type == 'Unknown']
        
        if not unclear_tables:
            print("      ✅ No unclear entities found")
            return
        
        print(f"      🤖 Analyzing {len(unclear_tables)} unclear entities with LLM...")
        
        # Process in batches
        batch_size = 10
        enhanced_count = 0
        
        for i in range(0, len(unclear_tables), batch_size):
            batch = unclear_tables[i:i+batch_size]
            
            # Create prompt
            prompt = create_llm_classification_prompt(batch)
            
            # Get LLM response
            response = await self.llm_client.ask(prompt)
            
            # Parse response
            classifications = extract_json_from_response(response)
            
            if classifications and 'classifications' in classifications:
                for classification in classifications['classifications']:
                    table_name = classification.get('table_name', '')
                    entity_type = classification.get('entity_type', 'Unknown')
                    confidence = float(classification.get('confidence', 0.0))
                    business_role = classification.get('business_role', 'Supporting')
                    
                    # Find matching table
                    for table in batch:
                        if table.full_name == table_name:
                            if confidence > table.confidence:  # Only update if better
                                table.entity_type = entity_type
                                table.confidence = confidence
                                table.business_role = business_role
                                enhanced_count += 1
                            break
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        print(f"      ✅ Enhanced {enhanced_count} entities using LLM")
    
    def _find_relationships(self, tables: List[TableInfo]):
        """Step 3: Find relationships between tables"""
        
        # Simple relationship discovery
        relationships = find_table_relationships_simple(tables)
        
        self.relationships = []
        for from_table, to_table, rel_type, confidence in relationships:
            self.relationships.append(Relationship(
                from_table=from_table,
                to_table=to_table,
                relationship_type=rel_type,
                confidence=confidence
            ))
        
        # Add relationships to tables
        for table in tables:
            table.connected_tables = []
            for rel in self.relationships:
                if rel.from_table == table.full_name:
                    table.connected_tables.append(rel.to_table)
                elif rel.to_table == table.full_name:
                    table.connected_tables.append(rel.from_table)
        
        print(f"      ✅ Found {len(self.relationships)} relationships")
    
    def _analyze_domain(self, tables: List[TableInfo]):
        """Step 4: Analyze business domain"""
        
        # Count entity types
        entity_counts = {}
        for table in tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        # Determine domain type
        if entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0:
            domain_type = "CRM/Financial"
        elif entity_counts.get('Order', 0) > 0 and entity_counts.get('Product', 0) > 0:
            domain_type = "E-Commerce"
        elif entity_counts.get('User', 0) > 0:
            domain_type = "User Management"
        else:
            domain_type = "Business Operations"
        
        # Generate sample questions
        sample_questions = []
        
        if 'Customer' in entity_counts:
            sample_questions.extend([
                "How many customers do we have?",
                "Show customer information",
                "List customer details"
            ])
        
        if 'Payment' in entity_counts:
            sample_questions.extend([
                "What is our total revenue?",
                "Show payment information",
                "Calculate total payments"
            ])
            
        if 'Customer' in entity_counts and 'Payment' in entity_counts:
            sample_questions.extend([
                "How many customers have made payments?",
                "Show paid customers",
                "Count total paid customers on 2025"
            ])
        
        if 'Order' in entity_counts:
            sample_questions.extend([
                "How many orders do we have?",
                "Show order information",
                "List recent orders"
            ])
        
        # Determine capabilities
        capabilities = {
            'customer_queries': 'Customer' in entity_counts,
            'payment_queries': 'Payment' in entity_counts,
            'order_queries': 'Order' in entity_counts,
            'paid_customer_analysis': 'Customer' in entity_counts and 'Payment' in entity_counts,
            'relationship_queries': len(self.relationships) > 0
        }
        
        # Calculate confidence
        classified_tables = sum(1 for t in tables if t.entity_type != 'Unknown')
        confidence = classified_tables / len(tables) if tables else 0.0
        
        self.domain = BusinessDomain(
            domain_type=domain_type,
            industry="Business",
            confidence=confidence,
            sample_questions=sample_questions,
            capabilities=capabilities
        )
        
        print(f"      ✅ Identified domain: {domain_type} (confidence: {confidence:.2f})")
    
    def _show_results(self):
        """Show analysis results"""
        
        classified_tables = [t for t in self.tables if t.entity_type != 'Unknown']
        
        print(f"\n📊 SEMANTIC ANALYSIS RESULTS:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   🧠 Classified: {len(classified_tables)}")
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
        
        # Show capabilities
        if self.domain and self.domain.capabilities:
            enabled_caps = [cap for cap, enabled in self.domain.capabilities.items() if enabled]
            if enabled_caps:
                print(f"   🎯 Query Capabilities:")
                for cap in enabled_caps:
                    print(f"      ✅ {cap.replace('_', ' ').title()}")
    
    def _save_to_cache(self, cache_file: Path):
        """Save analysis results to cache"""
        
        data = {
            'tables': [table_to_dict(table) for table in self.tables],
            'relationships': [
                {
                    'from_table': rel.from_table,
                    'to_table': rel.to_table,
                    'relationship_type': rel.relationship_type,
                    'confidence': rel.confidence,
                    'description': rel.description
                } for rel in self.relationships
            ],
            'domain': {
                'domain_type': self.domain.domain_type,
                'industry': self.domain.industry,
                'confidence': self.domain.confidence,
                'sample_questions': self.domain.sample_questions,
                'capabilities': self.domain.capabilities
            } if self.domain else None,
            'created': datetime.now().isoformat(),
            'version': 'simplified-v1.0'
        }
        
        save_cache(cache_file, data, "semantic analysis")
    
    def _load_from_cache(self, data: Dict):
        """Load analysis results from cache"""
        
        # Load tables
        if 'tables' in data:
            self.tables = [dict_to_table(table_data) for table_data in data['tables']]
        
        # Load relationships
        if 'relationships' in data:
            self.relationships = []
            for rel_data in data['relationships']:
                self.relationships.append(Relationship(
                    from_table=rel_data.get('from_table', ''),
                    to_table=rel_data.get('to_table', ''),
                    relationship_type=rel_data.get('relationship_type', ''),
                    confidence=rel_data.get('confidence', 0.0),
                    description=rel_data.get('description', '')
                ))
        
        # Load domain
        if 'domain' in data and data['domain']:
            domain_data = data['domain']
            self.domain = BusinessDomain(
                domain_type=domain_data.get('domain_type', ''),
                industry=domain_data.get('industry', ''),
                confidence=domain_data.get('confidence', 0.0),
                sample_questions=domain_data.get('sample_questions', []),
                capabilities=domain_data.get('capabilities', {})
            )
        
        print(f"   📊 Loaded: {len(self.tables)} tables, {len(self.relationships)} relationships")
    
    # Getter methods
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        return self.domain
    
    def get_business_analysis(self) -> Dict[str, Any]:
        """Get business analysis for compatibility"""
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        return {
            'entity_analysis': {
                'entity_counts': entity_counts,
                'high_confidence_entities': {
                    entity_type: [
                        {'table_name': t.full_name, 'confidence': t.confidence}
                        for t in self.tables 
                        if t.entity_type == entity_type and t.confidence > 0.7
                    ]
                    for entity_type in entity_counts.keys()
                }
            },
            'validation_results': {
                'overall_status': 'Ready' if len(self.relationships) > 0 else 'Limited',
                'entity_counts': entity_counts,
                'customer_payment_links': sum(1 for rel in self.relationships 
                                            if 'customer' in rel.from_table.lower() and 'payment' in rel.to_table.lower())
            }
        }

# For backward compatibility
EnhancedSemanticAnalyzer = SimpleSemanticAnalyzer
SemanticAnalyzer = SimpleSemanticAnalyzer