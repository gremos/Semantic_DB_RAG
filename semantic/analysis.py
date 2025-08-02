#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Semantic Analysis - Simple and Maintainable
Implements business domain analysis and entity classification
"""

import asyncio
import json
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

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship

class LLMClient:
    """Simple LLM client for semantic analysis"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,
            request_timeout=60
        )
    
    async def analyze_entities(self, prompt: str) -> str:
        """Analyze entities using LLM"""
        try:
            messages = [
                SystemMessage(content="You are an expert database analyst. Analyze database structures and classify business entities. Respond with valid JSON only."),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM analysis failed: {e}")
            return ""

class SemanticAnalyzer:
    """Enhanced semantic analyzer with business intelligence"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.domain: Optional[BusinessDomain] = None
        self.business_templates: Dict[str, Any] = {}
    
    async def analyze_tables(self, tables: List[TableInfo]) -> bool:
        """Main analysis method"""
        
        # Check cache first
        if self._load_from_cache():
            return True
        
        if not tables:
            print("❌ No tables to analyze")
            return False
        
        print(f"🧠 Starting enhanced semantic analysis of {len(tables)} tables...")
        
        # Store tables
        self.tables = tables.copy()
        
        try:
            # Step 1: Pattern-based classification
            print("   📊 Step 1: Pattern-based entity classification...")
            self._classify_entities_by_patterns()
            
            # Step 2: LLM-enhanced classification
            print("   🤖 Step 2: LLM-enhanced classification...")
            await self._enhance_with_llm()
            
            # Step 3: Discover relationships
            print("   🔗 Step 3: Discovering entity relationships...")
            self._discover_entity_relationships()
            
            # Step 4: Analyze business domain
            print("   🏢 Step 4: Analyzing business domain...")
            self._analyze_business_domain()
            
            # Step 5: Create business templates
            print("   📋 Step 5: Creating business query templates...")
            self._create_business_templates()
            
            # Step 6: Save results
            print("   💾 Step 6: Saving analysis results...")
            self._save_to_cache()
            
            # Show results
            self._show_analysis_results()
            
            print("✅ Enhanced semantic analysis completed!")
            return True
            
        except Exception as e:
            print(f"❌ Semantic analysis failed: {e}")
            return False
    
    def _classify_entities_by_patterns(self):
        """Step 1: Pattern-based entity classification"""
        
        classified_count = 0
        
        for table in self.tables:
            entity_type, confidence = self._get_entity_type_by_pattern(table)
            
            if entity_type != 'Unknown':
                table.entity_type = entity_type
                table.confidence = confidence
                table.business_role = 'Core' if confidence > 0.8 else 'Supporting'
                classified_count += 1
        
        print(f"      ✅ Classified {classified_count} tables using patterns")
    
    def _get_entity_type_by_pattern(self, table: TableInfo) -> tuple:
        """Classify entity type based on patterns"""
        
        name_lower = table.name.lower()
        column_names = [col.get('name', '').lower() for col in table.columns]
        
        # High confidence patterns - exact matches
        high_confidence_patterns = {
            'Customer': ['customer', 'client', 'account'],
            'Payment': ['payment', 'transaction', 'billing', 'invoice'],
            'Order': ['order', 'sale', 'purchase'],
            'Product': ['product', 'item', 'inventory'],
            'User': ['user', 'person', 'employee'],
            'Company': ['company', 'business', 'organization', 'vendor']
        }
        
        for entity_type, patterns in high_confidence_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return entity_type, 0.9
        
        # Medium confidence - column analysis
        column_patterns = {
            'Customer': ['customer_id', 'customer_name', 'email', 'phone'],
            'Payment': ['amount', 'payment_date', 'payment_method', 'total'],
            'Order': ['order_id', 'order_date', 'quantity', 'customer_id'],
            'Product': ['product_id', 'product_name', 'price', 'category'],
            'User': ['username', 'password', 'email', 'first_name', 'last_name'],
            'Contact': ['email', 'phone', 'address', 'contact_name']
        }
        
        for entity_type, patterns in column_patterns.items():
            matches = sum(1 for pattern in patterns if any(pattern in col for col in column_names))
            if matches >= 2:  # At least 2 matching columns
                confidence = min(0.7, 0.5 + (matches * 0.1))
                return entity_type, confidence
        
        # Low confidence - general patterns
        if any(col in column_names for col in ['id', 'name', 'created_date']):
            return 'Entity', 0.3
        
        return 'Unknown', 0.0
    
    async def _enhance_with_llm(self):
        """Step 2: Use LLM to enhance classification"""
        
        # Find tables that need LLM analysis
        unclear_tables = [t for t in self.tables if t.confidence < 0.6 or t.entity_type == 'Unknown']
        
        if not unclear_tables:
            print("      ✅ No unclear entities found")
            return
        
        print(f"      🤖 Analyzing {len(unclear_tables)} unclear entities with LLM...")
        
        # Process in small batches
        batch_size = 5
        enhanced_count = 0
        
        for i in range(0, len(unclear_tables), batch_size):
            batch = unclear_tables[i:i+batch_size]
            
            # Create LLM prompt
            prompt = self._create_classification_prompt(batch)
            
            # Get LLM response
            response = await self.llm_client.analyze_entities(prompt)
            
            # Parse and apply results
            enhanced_count += self._apply_llm_classifications(response, batch)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        print(f"      ✅ Enhanced {enhanced_count} entities using LLM")
    
    def _create_classification_prompt(self, tables_batch: List[TableInfo]) -> str:
        """Create LLM prompt for entity classification"""
        
        table_descriptions = []
        for table in tables_batch:
            columns_info = []
            for col in table.columns[:8]:  # Limit columns
                col_info = f"{col.get('name', '')} ({col.get('data_type', '')})"
                columns_info.append(col_info)
            
            sample_info = ""
            if table.sample_data:
                sample_info = str(table.sample_data[0])[:200]
            
            table_descriptions.append({
                'table_name': table.full_name,
                'columns': columns_info,
                'sample_data': sample_info,
                'row_count': table.row_count
            })
        
        prompt = f"""
Analyze these database tables and classify each as a business entity type.

TABLES TO CLASSIFY:
{json.dumps(table_descriptions, indent=2)}

For each table, determine:
1. Entity Type: Customer, Order, Product, Payment, User, Company, Contact, Financial, Reference, System, or Unknown
2. Confidence: 0.0 to 1.0 (how certain you are)
3. Business Role: Core, Supporting, Reference, or System

Consider:
- Table names and naming patterns
- Column names and data types
- Sample data content
- Table relationships

Respond with JSON only:
{{
  "classifications": [
    {{
      "table_name": "full_table_name",
      "entity_type": "Customer",
      "confidence": 0.8,
      "business_role": "Core",
      "reasoning": "brief explanation"
    }}
  ]
}}
"""
        return prompt
    
    def _apply_llm_classifications(self, response: str, batch: List[TableInfo]) -> int:
        """Apply LLM classification results"""
        
        try:
            # Parse JSON response
            data = self._extract_json_from_response(response)
            if not data or 'classifications' not in data:
                return 0
            
            enhanced_count = 0
            
            for classification in data['classifications']:
                table_name = classification.get('table_name', '')
                entity_type = classification.get('entity_type', 'Unknown')
                confidence = float(classification.get('confidence', 0.0))
                business_role = classification.get('business_role', 'Supporting')
                
                # Find matching table
                for table in batch:
                    if table.full_name == table_name:
                        # Only update if LLM is more confident
                        if confidence > table.confidence:
                            table.entity_type = entity_type
                            table.confidence = confidence
                            table.business_role = business_role
                            enhanced_count += 1
                        break
            
            return enhanced_count
            
        except Exception as e:
            print(f"      ⚠️ Failed to parse LLM response: {e}")
            return 0
    
    def _discover_entity_relationships(self):
        """Step 3: Discover relationships between entities"""
        
        # Simple relationship discovery based on entity types and column patterns
        for table in self.tables:
            if table.entity_type == 'Unknown':
                continue
            
            column_names = [col.get('name', '').lower() for col in table.columns]
            
            # Look for foreign key patterns
            for col_name in column_names:
                if col_name.endswith('_id') or col_name.endswith('id'):
                    entity_name = col_name.replace('_id', '').replace('id', '')
                    
                    # Find related entity
                    for other_table in self.tables:
                        if other_table.full_name == table.full_name:
                            continue
                        
                        if (entity_name in other_table.entity_type.lower() or 
                            entity_name in other_table.name.lower()):
                            
                            self.relationships.append(Relationship(
                                from_table=table.full_name,
                                to_table=other_table.full_name,
                                relationship_type='entity_reference',
                                confidence=0.7,
                                description=f"Entity relationship via {col_name}"
                            ))
        
        print(f"      ✅ Discovered {len(self.relationships)} entity relationships")
    
    def _analyze_business_domain(self):
        """Step 4: Analyze business domain and industry"""
        
        # Count entity types
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        # Determine domain type
        domain_type = self._determine_domain_type(entity_counts)
        
        # Generate sample questions
        sample_questions = self._generate_sample_questions(entity_counts)
        
        # Determine capabilities
        capabilities = self._determine_capabilities(entity_counts)
        
        # Calculate confidence
        classified_tables = sum(1 for t in self.tables if t.entity_type != 'Unknown')
        confidence = classified_tables / len(self.tables) if self.tables else 0.0
        
        self.domain = BusinessDomain(
            domain_type=domain_type,
            industry="Business",
            confidence=confidence,
            sample_questions=sample_questions,
            capabilities=capabilities
        )
        
        print(f"      ✅ Identified domain: {domain_type} (confidence: {confidence:.2f})")
    
    def _determine_domain_type(self, entity_counts: Dict[str, int]) -> str:
        """Determine business domain type"""
        
        # Domain scoring
        domain_scores = {
            'E-Commerce': 0,
            'CRM/Sales': 0,
            'Financial Services': 0,
            'User Management': 0,
            'Business Operations': 0
        }
        
        # E-Commerce indicators
        if entity_counts.get('Product', 0) > 0 and entity_counts.get('Order', 0) > 0:
            domain_scores['E-Commerce'] += 3
        if entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0:
            domain_scores['E-Commerce'] += 2
        
        # CRM/Sales indicators
        if entity_counts.get('Customer', 0) > 0:
            domain_scores['CRM/Sales'] += 2
        if entity_counts.get('Contact', 0) > 0:
            domain_scores['CRM/Sales'] += 1
        
        # Financial indicators
        if entity_counts.get('Payment', 0) > 0:
            domain_scores['Financial Services'] += 2
        if entity_counts.get('Financial', 0) > 0:
            domain_scores['Financial Services'] += 1
        
        # User Management indicators
        if entity_counts.get('User', 0) > 0:
            domain_scores['User Management'] += 2
        
        # Default to Business Operations
        domain_scores['Business Operations'] += 1
        
        # Return highest scoring domain
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    
    def _generate_sample_questions(self, entity_counts: Dict[str, int]) -> List[str]:
        """Generate sample questions based on entities"""
        
        questions = []
        
        if entity_counts.get('Customer', 0) > 0:
            questions.extend([
                "How many customers do we have?",
                "Show customer information",
                "List all customers"
            ])
        
        if entity_counts.get('Payment', 0) > 0:
            questions.extend([
                "What is our total revenue?",
                "Show payment information",
                "Calculate total payments for 2025"
            ])
        
        if entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0:
            questions.extend([
                "How many customers have made payments?",
                "Show paid customers",
                "Count total paid customers for 2025"
            ])
        
        if entity_counts.get('Order', 0) > 0:
            questions.extend([
                "How many orders do we have?",
                "Show order information",
                "List recent orders"
            ])
        
        if entity_counts.get('Product', 0) > 0:
            questions.extend([
                "How many products do we have?",
                "Show product information",
                "List all products"
            ])
        
        # Generic questions
        questions.extend([
            "Show system overview",
            "List all entities",
            "What data is available?"
        ])
        
        return questions[:15]  # Limit to 15 questions
    
    def _determine_capabilities(self, entity_counts: Dict[str, int]) -> Dict[str, bool]:
        """Determine system capabilities"""
        
        return {
            'customer_analysis': entity_counts.get('Customer', 0) > 0,
            'payment_analysis': entity_counts.get('Payment', 0) > 0,
            'order_analysis': entity_counts.get('Order', 0) > 0,
            'product_analysis': entity_counts.get('Product', 0) > 0,
            'user_management': entity_counts.get('User', 0) > 0,
            'financial_reporting': entity_counts.get('Payment', 0) > 0 or entity_counts.get('Financial', 0) > 0,
            'customer_payment_analysis': entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0,
            'cross_entity_analysis': len([c for c in entity_counts.values() if c > 0]) >= 2,
            'relationship_queries': len(self.relationships) > 0
        }
    
    def _create_business_templates(self):
        """Step 5: Create business query templates"""
        
        templates = {}
        
        # Customer templates
        customer_tables = [t for t in self.tables if t.entity_type == 'Customer']
        if customer_tables:
            templates['customer_count'] = {
                'description': 'Count total customers',
                'tables': [t.full_name for t in customer_tables],
                'sql_pattern': 'SELECT COUNT(*) as total_customers FROM {customer_table}'
            }
        
        # Payment templates
        payment_tables = [t for t in self.tables if t.entity_type == 'Payment']
        if payment_tables:
            templates['total_revenue'] = {
                'description': 'Calculate total revenue',
                'tables': [t.full_name for t in payment_tables],
                'sql_pattern': 'SELECT SUM(amount) as total_revenue FROM {payment_table}'
            }
        
        # Customer-Payment templates
        if customer_tables and payment_tables:
            templates['paid_customers'] = {
                'description': 'Count customers who have made payments',
                'tables': [t.full_name for t in customer_tables + payment_tables],
                'sql_pattern': 'SELECT COUNT(DISTINCT c.id) FROM {customer_table} c JOIN {payment_table} p ON c.id = p.customer_id'
            }
        
        self.business_templates = templates
        print(f"      ✅ Created {len(templates)} business query templates")
    
    def _show_analysis_results(self):
        """Show analysis results"""
        
        classified_tables = [t for t in self.tables if t.entity_type != 'Unknown']
        
        print(f"\n📊 SEMANTIC ANALYSIS RESULTS:")
        print(f"   📋 Total tables: {len(self.tables)}")
        print(f"   🧠 Classified: {len(classified_tables)}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        print(f"   📝 Business templates: {len(self.business_templates)}")
        
        # Show entity distribution
        entity_counts = {}
        for table in self.tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   🏢 Business Entities:")
            for entity_type, count in sorted(entity_counts.items()):
                print(f"      • {entity_type}: {count} tables")
        
        # Show domain info
        if self.domain:
            print(f"   🏢 Domain: {self.domain.domain_type} (confidence: {self.domain.confidence:.2f})")
            
            enabled_caps = [cap for cap, enabled in self.domain.capabilities.items() if enabled]
            if enabled_caps:
                print(f"   🎯 Capabilities: {len(enabled_caps)} query types enabled")
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        try:
            import re
            
            # Clean response
            cleaned = response.strip()
            
            # Remove markdown code blocks
            if '```json' in cleaned:
                match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            elif '```' in cleaned:
                match = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            return json.loads(cleaned)
            
        except Exception:
            return None
    
    def _save_to_cache(self):
        """Save analysis results to cache"""
        
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        # Convert to dictionary format
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
            'business_templates': self.business_templates,
            'created': datetime.now().isoformat(),
            'version': '2.0-enhanced'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load from cache if available"""
        
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        
        if not cache_file.exists():
            return False
        
        try:
            import time
            
            # Check cache age
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
            
            # Load business templates
            self.business_templates = data.get('business_templates', {})
            
            print(f"✅ Loaded semantic analysis from cache")
            print(f"   📊 Tables: {len(self.tables)}")
            print(f"   🔗 Relationships: {len(self.relationships)}")
            print(f"   📝 Templates: {len(self.business_templates)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
            return False
    
    def load_from_cache(self) -> bool:
        """Public method to load from cache"""
        return self._load_from_cache()
    
    def get_tables(self) -> List[TableInfo]:
        """Get analyzed tables"""
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        """Get discovered relationships"""
        return self.relationships
    
    def get_domain(self) -> Optional[BusinessDomain]:
        """Get business domain"""
        return self.domain
    
    def get_business_templates(self) -> Dict[str, Any]:
        """Get business query templates"""
        return self.business_templates