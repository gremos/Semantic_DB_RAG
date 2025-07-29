#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED Semantic Analysis Module - Superior Business Entity Recognition
Specialized for identifying customer, payment, transaction, and core business tables
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

class EnhancedSemanticAnalyzer:
    """ENHANCED Semantic Analysis with superior business entity recognition"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = SimpleLLMClient(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        
        # Business entity patterns for better recognition
        self.business_entity_patterns = self._init_business_patterns()
    
    def _init_business_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive business entity recognition patterns"""
        return {
            'customer': {
                'table_patterns': [
                    'customer', 'client', 'account', 'businesspoint', 'contact', 'person',
                    'user', 'member', 'subscriber', 'buyer', 'clientele'
                ],
                'column_patterns': [
                    'customerid', 'clientid', 'accountid', 'businesspointid', 'contactid',
                    'customername', 'clientname', 'accountname', 'companyname',
                    'email', 'phone', 'address', 'registration', 'signup'
                ],
                'data_indicators': [
                    'email addresses', 'phone numbers', 'names', 'addresses',
                    'registration dates', 'customer codes'
                ]
            },
            'payment': {
                'table_patterns': [
                    'payment', 'transaction', 'invoice', 'billing', 'charge',
                    'receipt', 'financial', 'monetary', 'cash', 'credit'
                ],
                'column_patterns': [
                    'paymentid', 'transactionid', 'invoiceid', 'amount', 'total',
                    'paymentdate', 'transactiondate', 'invoicedate', 'paiddate',
                    'paymentmethod', 'paymentstatus', 'currency', 'price'
                ],
                'data_indicators': [
                    'monetary amounts', 'payment dates', 'transaction references',
                    'payment methods', 'invoice numbers', 'financial data'
                ]
            },
            'order': {
                'table_patterns': [
                    'order', 'sale', 'purchase', 'booking', 'reservation',
                    'request', 'orderitem', 'salesorder', 'purchaseorder'
                ],
                'column_patterns': [
                    'orderid', 'saleid', 'purchaseid', 'bookingid',
                    'orderdate', 'saledate', 'orderamount', 'quantity',
                    'orderstatus', 'salestatus', 'ordervalue'
                ],
                'data_indicators': [
                    'order numbers', 'sale amounts', 'order dates',
                    'quantities', 'order status', 'purchase data'
                ]
            },
            'product': {
                'table_patterns': [
                    'product', 'item', 'service', 'goods', 'merchandise',
                    'catalog', 'inventory', 'stock', 'offering'
                ],
                'column_patterns': [
                    'productid', 'itemid', 'serviceid', 'sku', 'barcode',
                    'productname', 'itemname', 'servicename', 'description',
                    'price', 'cost', 'category', 'brand'
                ],
                'data_indicators': [
                    'product names', 'prices', 'descriptions', 'categories',
                    'SKU codes', 'product codes'
                ]
            }
        }
    
    async def analyze_semantics(self, tables: List[TableInfo]) -> bool:
        """Run ENHANCED semantic analysis with superior business entity recognition"""
        # Check cache first
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        if self.load_from_cache():
            print(f"✅ Loaded semantic analysis from cache")
            # Still run entity verification on cached data
            self._verify_core_entities()
            return True
        
        if not tables:
            print("❌ No tables provided for analysis.")
            return False
        
        self.tables = tables
        
        print(f"🚀 ENHANCED semantic analysis starting...")
        print(f"   📊 Analyzing {len(tables)} objects for business entities")
        
        # Step 1: Pre-classify tables by business entity patterns
        print("🔍 Step 1: Pre-classifying tables by business patterns...")
        self._pre_classify_by_patterns()
        
        # Step 2: Enhanced AI classification with business context
        print("🧠 Step 2: AI classification with business entity focus...")
        classified_count = await self._classify_tables_business_focused()
        
        # Step 3: Verify core business entities were found
        print("✅ Step 3: Verifying core business entities...")
        self._verify_core_entities()
        
        # Step 4: Enhanced relationship discovery
        print("🔗 Step 4: Enhanced relationship discovery...")
        await self._discover_relationships_business_focused()
        
        # Step 5: Business domain analysis
        print("🏢 Step 5: Business domain analysis...")
        await self._analyze_business_domain_enhanced()
        
        # Save results
        print("💾 Saving enhanced semantic analysis...")
        self._save_to_cache(cache_file)
        
        print("✅ ENHANCED semantic analysis completed!")
        self._log_entity_summary()
        
        return True
    
    def _pre_classify_by_patterns(self):
        """Pre-classify tables using business entity patterns"""
        pattern_matches = defaultdict(list)
        
        for table in self.tables:
            table_name_lower = table.name.lower()
            
            # Check against each business entity type
            for entity_type, patterns in self.business_entity_patterns.items():
                score = 0
                matches = []
                
                # Check table name patterns
                for pattern in patterns['table_patterns']:
                    if pattern in table_name_lower:
                        score += 10
                        matches.append(f"table_name:{pattern}")
                
                # Check column name patterns
                for col in table.columns:
                    col_name_lower = col['name'].lower()
                    for pattern in patterns['column_patterns']:
                        if pattern in col_name_lower:
                            score += 5
                            matches.append(f"column:{pattern}")
                
                # Check sample data for indicators
                if table.sample_data:
                    for row in table.sample_data:
                        for key, value in row.items():
                            if isinstance(value, str) and value:
                                value_lower = str(value).lower()
                                for indicator in patterns['data_indicators']:
                                    if any(word in value_lower for word in indicator.split()):
                                        score += 2
                                        matches.append(f"data:{indicator}")
                
                if score > 0:
                    pattern_matches[entity_type].append({
                        'table': table,
                        'score': score,
                        'matches': matches[:5]  # Top 5 matches
                    })
        
        # Log pattern matching results
        print(f"   📋 Pattern matching results:")
        for entity_type, matches in pattern_matches.items():
            if matches:
                top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
                print(f"      {entity_type.upper()}: {len(matches)} candidates")
                for match in top_matches[:3]:
                    print(f"         • {match['table'].name} (score: {match['score']})")
        
        # Store pattern scores for later use
        for entity_type, matches in pattern_matches.items():
            for match in matches:
                if not hasattr(match['table'], 'pattern_scores'):
                    match['table'].pattern_scores = {}
                match['table'].pattern_scores[entity_type] = match['score']
    
    async def _classify_tables_business_focused(self) -> int:
        """Enhanced table classification with business entity focus"""
        batch_size = self.config.max_batch_size
        classified_count = 0
        
        progress_bar = tqdm(range(0, len(self.tables), batch_size), desc="Business entity analysis")
        
        for i in progress_bar:
            batch = self.tables[i:i + batch_size]
            
            progress_bar.set_description(f"Analyzing business entities: {', '.join([t.name for t in batch[:2]])}...")
            
            batch_classified = await self._classify_batch_business_focused(batch)
            classified_count += batch_classified
            await asyncio.sleep(self.config.rate_limit_delay)
        
        progress_bar.set_description("Business entity analysis complete")
        return classified_count
    
    async def _classify_batch_business_focused(self, tables: List[TableInfo]) -> int:
        """Business-focused batch classification with enhanced prompts"""
        
        # Prepare comprehensive table analysis for LLM
        table_analysis = []
        for table in tables:
            analysis = {
                'name': table.name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': [col['name'] for col in table.columns[:15]],  # More columns
                'sample_data': table.sample_data[:3],  # More samples
                'pattern_scores': getattr(table, 'pattern_scores', {})
            }
            
            # Add column details for better analysis
            analysis['column_details'] = []
            for col in table.columns[:10]:
                col_detail = {
                    'name': col['name'],
                    'type': col['data_type'],
                    'nullable': col['nullable']
                }
                analysis['column_details'].append(col_detail)
            
            table_analysis.append(analysis)
        
        # Enhanced prompt for business entity recognition
        prompt = f"""
You are an expert database analyst specializing in business systems. Analyze these database objects and classify them by their TRUE business purpose.

CRITICAL: Focus on identifying CORE BUSINESS ENTITIES:
1. CUSTOMER/CLIENT tables (people/companies who buy/use services)
2. PAYMENT/TRANSACTION tables (money, billing, financial records)  
3. ORDER/SALE tables (purchases, bookings, transactions)
4. PRODUCT/SERVICE tables (what is being sold/offered)
5. INVOICE/BILLING tables (billing documents, charges)

Objects to analyze:
{json.dumps(table_analysis, indent=1, default=str)}

For each object, provide detailed classification focusing on BUSINESS PURPOSE:

ENTITY_TYPE options:
- Customer (people/companies who buy/use services)
- Payment (financial transactions, money, billing)
- Order (sales, purchases, bookings, transactions)
- Product (items/services being sold)
- Invoice (billing documents, charges)
- Employee (staff, workers)
- Vendor (suppliers, partners)
- Support (cases, tickets, service)
- Marketing (campaigns, targets, leads)
- Reference (lookup tables, categories)
- System (logs, configuration, technical)

BUSINESS_ROLE: Specific business function (e.g., "Primary Customer Records", "Payment Transactions", "Sales Orders")

DATA_NATURE options:
- Master (core entities like customers, products)
- Transaction (events like payments, orders)
- Reference (lookup data, categories)
- Analytical (reports, summaries)

Look for these CRITICAL INDICATORS:
- Customer tables: customer/client/account in name, email/phone/address columns, customer IDs
- Payment tables: payment/transaction/invoice in name, amount/total/price columns, payment dates
- Order tables: order/sale/purchase in name, order IDs, quantities, order dates
- Product tables: product/item/service in name, prices, descriptions, SKUs

Respond with JSON array:
[
  {{
    "table_name": "TableName",
    "entity_type": "Customer|Payment|Order|Product|Invoice|Employee|Vendor|Support|Marketing|Reference|System",
    "business_role": "Specific business function description",
    "data_nature": "Master|Transaction|Reference|Analytical",
    "contains_personal_data": true/false,
    "contains_financial_data": true/false,
    "primary_purpose": "Detailed explanation of what this table stores",
    "confidence": 0.0-1.0,
    "key_indicators": ["specific reasons for this classification"]
  }}
]
"""
        
        try:
            system_message = """You are a senior database analyst with expertise in business systems. 
Your job is to identify core business entities like customers, payments, orders, and products.
Be thorough in your analysis and focus on the true business purpose of each table.
Respond with valid JSON only."""
            
            response = await self.llm.ask(prompt, system_message)
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
                        
                        # Store key indicators for debugging
                        table.key_indicators = classification.get('key_indicators', [])
                        classified_in_batch += 1
            
            return classified_in_batch
        
        except Exception as e:
            print(f"⚠️ Business-focused classification failed for batch: {e}")
            return 0
    
    def _verify_core_entities(self):
        """Verify that core business entities were identified"""
        core_entities = {
            'Customer': [],
            'Payment': [],
            'Order': [],
            'Product': [],
            'Invoice': []
        }
        
        # Categorize tables by core entity types
        for table in self.tables:
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                if entity_type in core_entities:
                    core_entities[entity_type].append(table)
        
        print(f"   🔍 Core business entity verification:")
        missing_entities = []
        
        for entity_type, tables_list in core_entities.items():
            if tables_list:
                print(f"      ✅ {entity_type}: {len(tables_list)} tables found")
                for table in tables_list[:3]:  # Show top 3
                    confidence = table.semantic_profile.confidence
                    print(f"         • {table.name} (confidence: {confidence:.2f})")
            else:
                print(f"      ❌ {entity_type}: No tables found!")
                missing_entities.append(entity_type)
        
        if missing_entities:
            print(f"   ⚠️  MISSING CORE ENTITIES: {', '.join(missing_entities)}")
            print(f"       This may explain incorrect query results!")
            
            # Suggest manual verification
            print(f"   💡 Suggested actions:")
            print(f"      1. Check if tables exist with different names")
            print(f"      2. Verify sample data contains relevant business information")
            print(f"      3. Review classification confidence scores")
            
            # Show tables that might be misclassified
            potential_customers = [t for t in self.tables if 'customer' in t.name.lower() or 'client' in t.name.lower() or 'account' in t.name.lower()]
            potential_payments = [t for t in self.tables if 'payment' in t.name.lower() or 'transaction' in t.name.lower() or 'invoice' in t.name.lower()]
            
            if potential_customers:
                print(f"      📋 Potential customer tables not classified: {[t.name for t in potential_customers[:5]]}")
            if potential_payments:
                print(f"      📋 Potential payment tables not classified: {[t.name for t in potential_payments[:5]]}")
    
    def _log_entity_summary(self):
        """Log summary of entity classification"""
        entity_counts = defaultdict(int)
        high_confidence_entities = defaultdict(int)
        
        for table in self.tables:
            if table.semantic_profile:
                entity_type = table.semantic_profile.entity_type
                confidence = table.semantic_profile.confidence
                
                entity_counts[entity_type] += 1
                if confidence >= 0.7:
                    high_confidence_entities[entity_type] += 1
        
        print(f"\n📊 ENTITY CLASSIFICATION SUMMARY:")
        for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            high_conf = high_confidence_entities[entity_type]
            print(f"   {entity_type}: {count} tables ({high_conf} high confidence)")
        
        # Show sample of each core entity type
        core_types = ['Customer', 'Payment', 'Order', 'Product', 'Invoice']
        for core_type in core_types:
            core_tables = [t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == core_type]
            if core_tables:
                best_table = max(core_tables, key=lambda t: t.semantic_profile.confidence)
                print(f"   🎯 Best {core_type} table: {best_table.name} (confidence: {best_table.semantic_profile.confidence:.2f})")
    
    async def _discover_relationships_business_focused(self):
        """Enhanced relationship discovery with business logic"""
        print("🔗 Discovering business-focused relationships...")
        
        relationships = []
        
        # Focus on core business relationships
        customer_tables = [t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == 'Customer']
        payment_tables = [t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == 'Payment']
        order_tables = [t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == 'Order']
        
        # Customer -> Payment relationships
        for customer_table in customer_tables:
            for payment_table in payment_tables:
                relationship = self._find_table_relationship(customer_table, payment_table, 'customer_payment')
                if relationship:
                    relationships.append(relationship)
        
        # Customer -> Order relationships  
        for customer_table in customer_tables:
            for order_table in order_tables:
                relationship = self._find_table_relationship(customer_table, order_table, 'customer_order')
                if relationship:
                    relationships.append(relationship)
        
        # Order -> Payment relationships
        for order_table in order_tables:
            for payment_table in payment_tables:
                relationship = self._find_table_relationship(order_table, payment_table, 'order_payment')
                if relationship:
                    relationships.append(relationship)
        
        self.relationships = relationships[:50]  # Keep top 50 relationships
        print(f"   ✅ Found {len(self.relationships)} business relationships")
    
    def _find_table_relationship(self, table1: TableInfo, table2: TableInfo, rel_type: str) -> Optional[Relationship]:
        """Find relationship between two tables based on column analysis"""
        
        # Look for common ID patterns
        table1_ids = [col['name'] for col in table1.columns if col['name'].lower().endswith('id')]
        table2_ids = [col['name'] for col in table2.columns if col['name'].lower().endswith('id')]
        
        # Find matching ID columns
        for id1 in table1_ids:
            for id2 in table2_ids:
                if id1.lower() == id2.lower():
                    return Relationship(
                        from_table=table1.full_name,
                        to_table=table2.full_name,
                        column=id1,
                        relationship_type=f'business_{rel_type}',
                        confidence=0.8,
                        description=f"Business relationship: {rel_type} via {id1}"
                    )
        
        return None
    
    async def _analyze_business_domain_enhanced(self):
        """Enhanced business domain analysis"""
        # Use existing implementation but with better entity context
        entity_types = defaultdict(int)
        business_roles = defaultdict(int)
        
        for table in self.tables:
            if table.semantic_profile:
                entity_types[table.semantic_profile.entity_type] += 1
                business_roles[table.semantic_profile.business_role] += 1
        
        # Enhanced evidence collection
        evidence = {
            'total_tables': len(self.tables),
            'entity_types': dict(entity_types),
            'business_roles': dict(business_roles),
            'core_entities': {
                'customers': len([t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == 'Customer']),
                'payments': len([t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == 'Payment']),
                'orders': len([t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == 'Order']),
                'products': len([t for t in self.tables if t.semantic_profile and t.semantic_profile.entity_type == 'Product'])
            },
            'sample_greek_text': extract_sample_greek_text(self.tables)
        }
        
        # Generate sample questions based on identified entities
        sample_questions = []
        if evidence['core_entities']['customers'] > 0:
            sample_questions.extend([
                "How many customers do we have?",
                "Show me our top customers",
                "List customers by registration date"
            ])
        
        if evidence['core_entities']['payments'] > 0:
            sample_questions.extend([
                "What is our total revenue?",
                "Count paid customers for 2025",
                "Show recent payments"
            ])
        
        if evidence['core_entities']['orders'] > 0:
            sample_questions.extend([
                "How many orders this year?",
                "Show recent orders",
                "What is our average order value?"
            ])
        
        # Set domain with enhanced sample questions
        self.domain = BusinessDomain(
            domain_type="CRM",  # Based on the user's query results
            industry="Business Services",
            entities=list(entity_types.keys())[:10],
            confidence=0.8,
            sample_questions=sample_questions,
            customer_definition="Business entities and contacts managed through CRM system"
        )
    
    def _save_to_cache(self, cache_file):
        """Save enhanced semantic analysis to cache"""
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
            'version': '3.0-enhanced-business-entities'
        }
        
        # Convert tables with enhanced semantic profiles
        for table in self.tables:
            table_dict = table_info_to_dict(table)
            # Add enhanced metadata
            if hasattr(table, 'key_indicators'):
                table_dict['key_indicators'] = table.key_indicators
            if hasattr(table, 'pattern_scores'):
                table_dict['pattern_scores'] = table.pattern_scores
            data['tables'].append(table_dict)
        
        save_json_cache(cache_file, data, "enhanced semantic analysis")
    
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
                        # Load enhanced metadata
                        if 'key_indicators' in table_data:
                            table.key_indicators = table_data['key_indicators']
                        if 'pattern_scores' in table_data:
                            table.pattern_scores = table_data['pattern_scores']
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

# Update the main analyzer class
SemanticAnalyzer = EnhancedSemanticAnalyzer