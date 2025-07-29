#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED Semantic Analysis - True Business Entity Recognition
Focuses on identifying REAL customer, payment, and business relationships
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

class BusinessEntityAnalyzer:
    """Enhanced business entity analyzer that focuses on REAL business relationships"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,
            request_timeout=120
        )
        
        # Business entity templates for better recognition
        self.business_patterns = self._init_business_entity_templates()
    
    def _init_business_entity_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive business entity recognition templates"""
        return {
            'Customer': {
                'description': 'People or companies who buy/use services',
                'name_patterns': [
                    'customer', 'client', 'account', 'businesspoint', 'contact', 
                    'person', 'user', 'member', 'subscriber', 'buyer'
                ],
                'column_patterns': [
                    'customerid', 'clientid', 'accountid', 'businesspointid',
                    'customername', 'clientname', 'accountname', 'companyname',
                    'email', 'phone', 'address', 'registrationdate'
                ],
                'data_indicators': [
                    'email addresses', 'phone numbers', 'person names', 
                    'company names', 'addresses', 'registration dates'
                ],
                'business_questions': [
                    'Who are our customers?', 'How many customers do we have?',
                    'Customer contact information', 'Customer demographics'
                ]
            },
            'Payment': {
                'description': 'Financial transactions, money, billing records',
                'name_patterns': [
                    'payment', 'transaction', 'invoice', 'billing', 'financial',
                    'monetary', 'charge', 'receipt', 'revenue', 'cash'
                ],
                'column_patterns': [
                    'paymentid', 'transactionid', 'invoiceid', 'amount', 'total',
                    'paymentdate', 'transactiondate', 'paymentmethod', 'currency',
                    'price', 'cost', 'fee', 'value', 'sum'
                ],
                'data_indicators': [
                    'monetary amounts', 'payment dates', 'transaction IDs',
                    'payment methods', 'currency codes', 'financial values'
                ],
                'business_questions': [
                    'How much revenue did we generate?', 'Who has paid?',
                    'Payment history', 'Transaction records'
                ]
            },
            'Order': {
                'description': 'Sales, purchases, bookings, transactions',
                'name_patterns': [
                    'order', 'sale', 'purchase', 'booking', 'reservation',
                    'request', 'orderitem', 'salesorder'
                ],
                'column_patterns': [
                    'orderid', 'saleid', 'purchaseid', 'bookingid',
                    'orderdate', 'saledate', 'orderamount', 'quantity',
                    'orderstatus', 'salestatus'
                ],
                'data_indicators': [
                    'order numbers', 'sale amounts', 'order dates',
                    'quantities', 'order status', 'purchase records'
                ],
                'business_questions': [
                    'How many orders do we have?', 'Order history',
                    'Sales by period', 'Order status tracking'
                ]
            }
        }
    
    async def analyze_business_entities(self, tables: List[TableInfo]) -> Dict[str, Any]:
        """
        Enhanced business entity analysis with focus on REAL relationships
        """
        print("🧠 Enhanced Business Entity Analysis Starting...")
        
        # Step 1: Pre-classify using advanced pattern matching
        print("🔍 Step 1: Advanced pattern-based pre-classification...")
        pattern_results = self._advanced_pattern_classification(tables)
        
        # Step 2: Deep LLM analysis with business focus
        print("🧠 Step 2: Deep LLM business entity analysis...")
        llm_results = await self._deep_llm_business_analysis(tables, pattern_results)
        
        # Step 3: Cross-validation and relationship discovery
        print("🔗 Step 3: Business relationship discovery...")
        relationships = await self._discover_business_relationships(tables, llm_results)
        
        # Step 4: Validate business logic
        print("✅ Step 4: Business logic validation...")
        validated_results = await self._validate_business_logic(tables, llm_results, relationships)
        
        return {
            'entity_classifications': validated_results,
            'relationships': relationships,
            'validation_results': self._get_validation_summary(validated_results, relationships)
        }
    
    def _advanced_pattern_classification(self, tables: List[TableInfo]) -> Dict[str, Dict]:
        """Advanced pattern-based classification with scoring"""
        results = {}
        
        for table in tables:
            table_scores = {}
            table_evidence = {}
            
            for entity_type, patterns in self.business_patterns.items():
                score = 0
                evidence = []
                
                # Table name analysis
                table_name_lower = table.name.lower()
                for pattern in patterns['name_patterns']:
                    if pattern in table_name_lower:
                        score += 20
                        evidence.append(f"table_name_match:{pattern}")
                
                # Column analysis
                for col in table.columns:
                    col_name_lower = col['name'].lower()
                    for pattern in patterns['column_patterns']:
                        if pattern in col_name_lower:
                            score += 10
                            evidence.append(f"column_match:{col['name']}:{pattern}")
                
                # Sample data analysis
                if table.sample_data:
                    for row in table.sample_data[:3]:
                        for key, value in row.items():
                            if value and isinstance(value, str):
                                # Check for business-specific data patterns
                                if entity_type == 'Customer':
                                    if '@' in str(value) and '.' in str(value):  # Email pattern
                                        score += 15
                                        evidence.append(f"email_pattern:{key}")
                                    elif any(char.isdigit() for char in str(value)) and len(str(value)) > 8:  # Phone pattern
                                        score += 10
                                        evidence.append(f"phone_pattern:{key}")
                                
                                elif entity_type == 'Payment':
                                    if isinstance(value, (int, float)) and value > 0:  # Money amount
                                        score += 12
                                        evidence.append(f"monetary_value:{key}")
                
                if score > 0:
                    table_scores[entity_type] = score
                    table_evidence[entity_type] = evidence[:5]  # Top 5 pieces of evidence
            
            results[table.name] = {
                'scores': table_scores,
                'evidence': table_evidence,
                'top_entity': max(table_scores, key=table_scores.get) if table_scores else None,
                'confidence': max(table_scores.values()) / 100 if table_scores else 0
            }
        
        return results
    
    async def _deep_llm_business_analysis(self, tables: List[TableInfo], pattern_results: Dict) -> Dict[str, Dict]:
        """Deep LLM analysis focused on business entity identification"""
        
        batch_size = 8  # Smaller batches for better analysis
        results = {}
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i + batch_size]
            print(f"   Analyzing batch {i//batch_size + 1}/{(len(tables) + batch_size - 1)//batch_size}")
            
            batch_analysis = await self._analyze_batch_with_business_focus(batch, pattern_results)
            results.update(batch_analysis)
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        return results
    
    async def _analyze_batch_with_business_focus(self, tables: List[TableInfo], pattern_results: Dict) -> Dict:
        """Analyze a batch of tables with enhanced business focus"""
        
        # Prepare comprehensive table analysis for LLM
        table_summaries = []
        for table in tables:
            pattern_info = pattern_results.get(table.name, {})
            
            summary = {
                'table_name': table.name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': [
                    {
                        'name': col['name'],
                        'type': col['data_type'],
                        'nullable': col['nullable']
                    } for col in table.columns[:15]
                ],
                'sample_data': table.sample_data[:3],
                'pattern_analysis': {
                    'top_entity_guess': pattern_info.get('top_entity'),
                    'confidence': pattern_info.get('confidence', 0),
                    'evidence': pattern_info.get('evidence', {})
                }
            }
            table_summaries.append(summary)
        
        # Enhanced business-focused prompt
        prompt = f"""
You are a senior business analyst specializing in enterprise database systems. Your task is to identify the TRUE business purpose of each database object.

CRITICAL BUSINESS ENTITY TYPES:
1. **Customer**: People/companies who buy/use services (PRIMARY business entity)
2. **Payment**: Financial transactions, billing, revenue records (CRITICAL for business)
3. **Order**: Sales, purchases, transactions, bookings
4. **Product**: Items/services being sold
5. **Invoice**: Billing documents, charges
6. **Employee**: Staff, workers
7. **Vendor**: Suppliers, partners
8. **Support**: Customer service, help desk
9. **Marketing**: Campaigns, leads, advertising
10. **Reference**: Lookup tables, categories, configurations
11. **System**: Technical logs, configurations

BUSINESS ANALYSIS RULES:
- CUSTOMER tables contain people/company information (names, contacts, addresses)
- PAYMENT tables contain financial amounts, transaction dates, payment methods
- Tables with monetary values (amount, total, price) are likely PAYMENT or ORDER related
- Look for ID columns that link entities (CustomerID, PaymentID, OrderID)
- Consider the business process: Customers → Orders → Payments → Invoices

DATABASE OBJECTS TO ANALYZE:
{json.dumps(table_summaries, indent=1, default=str)}

For each table, provide detailed business analysis:

RESPONSE FORMAT (JSON ARRAY):
[
  {{
    "table_name": "TableName",
    "entity_type": "Customer|Payment|Order|Product|Invoice|Employee|Vendor|Support|Marketing|Reference|System",
    "business_purpose": "Detailed explanation of what this table stores in business terms",
    "business_role": "Primary Customer Master|Payment Transactions|Order Records|Product Catalog|etc.",
    "key_business_indicators": [
      "Specific evidence for this classification"
    ],
    "customer_link": "How this table relates to customers (if applicable)",
    "payment_link": "How this table relates to payments (if applicable)", 
    "confidence": 0.0-1.0,
    "critical_columns": [
      "Names of most important business columns"
    ],
    "sample_business_questions": [
      "What business questions could this table answer?"
    ]
  }}
]

FOCUS ON BUSINESS RELATIONSHIPS:
- Identify which tables contain customer information
- Identify which tables contain payment/financial information  
- Identify how customers connect to payments (via orders, invoices, etc.)
- Look for tables that bridge customers and payments

Respond with ONLY valid JSON array:
"""
        
        try:
            system_message = """You are a business database expert. Focus on identifying real business entities and their relationships. 
Pay special attention to customer and payment tables as these are critical for business queries.
Provide detailed, accurate business analysis. Respond with valid JSON only."""
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            classifications = extract_json_from_response(response.content)
            
            results = {}
            if classifications and isinstance(classifications, list):
                for classification in classifications:
                    table_name = classification.get('table_name', '')
                    if table_name:
                        results[table_name] = classification
            
            return results
        
        except Exception as e:
            print(f"⚠️ LLM analysis failed for batch: {e}")
            return {}
    
    async def _discover_business_relationships(self, tables: List[TableInfo], llm_results: Dict) -> List[Dict]:
        """Discover business relationships with focus on customer-payment links"""
        
        print("   🔍 Discovering customer-payment-order relationships...")
        
        # Categorize tables by business entity
        entity_tables = defaultdict(list)
        for table in tables:
            table_analysis = llm_results.get(table.name, {})
            entity_type = table_analysis.get('entity_type', 'Unknown')
            entity_tables[entity_type].append(table)
        
        relationships = []
        
        # Focus on critical business relationships
        customer_tables = entity_tables.get('Customer', [])
        payment_tables = entity_tables.get('Payment', [])
        order_tables = entity_tables.get('Order', [])
        
        print(f"   📊 Found: {len(customer_tables)} Customer, {len(payment_tables)} Payment, {len(order_tables)} Order tables")
        
        # Customer -> Payment relationships (CRITICAL)
        for customer_table in customer_tables:
            for payment_table in payment_tables:
                relationship = await self._analyze_table_relationship(
                    customer_table, payment_table, 'customer_payment', llm_results
                )
                if relationship:
                    relationships.append(relationship)
        
        # Customer -> Order relationships
        for customer_table in customer_tables:
            for order_table in order_tables:
                relationship = await self._analyze_table_relationship(
                    customer_table, order_table, 'customer_order', llm_results
                )
                if relationship:
                    relationships.append(relationship)
        
        # Order -> Payment relationships
        for order_table in order_tables:
            for payment_table in payment_tables:
                relationship = await self._analyze_table_relationship(
                    order_table, payment_table, 'order_payment', llm_results
                )
                if relationship:
                    relationships.append(relationship)
        
        return relationships[:50]  # Keep top 50 relationships
    
    async def _analyze_table_relationship(self, table1: TableInfo, table2: TableInfo, 
                                        relationship_type: str, llm_results: Dict) -> Optional[Dict]:
        """Analyze relationship between two tables using column analysis and LLM"""
        
        # Quick column-based analysis first
        table1_ids = [col['name'] for col in table1.columns if col['name'].lower().endswith('id')]
        table2_ids = [col['name'] for col in table2.columns if col['name'].lower().endswith('id')]
        
        # Look for matching ID columns
        for id1 in table1_ids:
            for id2 in table2_ids:
                if id1.lower() == id2.lower():
                    return {
                        'from_table': table1.full_name,
                        'to_table': table2.full_name,
                        'link_column': id1,
                        'relationship_type': relationship_type,
                        'confidence': 0.8,
                        'discovery_method': 'column_matching',
                        'description': f"Tables linked via {id1} column"
                    }
        
        # If no direct column match, use LLM for deeper analysis
        return await self._llm_relationship_analysis(table1, table2, relationship_type, llm_results)
    
    async def _llm_relationship_analysis(self, table1: TableInfo, table2: TableInfo, 
                                       relationship_type: str, llm_results: Dict) -> Optional[Dict]:
        """Use LLM to analyze potential relationships between tables"""
        
        table1_analysis = llm_results.get(table1.name, {})
        table2_analysis = llm_results.get(table2.name, {})
        
        prompt = f"""
Analyze if these two business tables are related and how:

TABLE 1: {table1.name}
- Business Purpose: {table1_analysis.get('business_purpose', 'Unknown')}
- Entity Type: {table1_analysis.get('entity_type', 'Unknown')}
- Key Columns: {[col['name'] for col in table1.columns[:10]]}
- Sample Data: {table1.sample_data[:2] if table1.sample_data else 'None'}

TABLE 2: {table2.name}  
- Business Purpose: {table2_analysis.get('business_purpose', 'Unknown')}
- Entity Type: {table2_analysis.get('entity_type', 'Unknown')}
- Key Columns: {[col['name'] for col in table2.columns[:10]]}
- Sample Data: {table2.sample_data[:2] if table2.sample_data else 'None'}

RELATIONSHIP TYPE TO ANALYZE: {relationship_type}

Respond with JSON:
{{
  "has_relationship": true/false,
  "link_method": "Description of how they're linked",
  "link_columns": ["column1", "column2"],
  "confidence": 0.0-1.0,
  "business_description": "How this relationship works in business terms"
}}

Respond with ONLY JSON:
"""
        
        try:
            system_message = "You are a database relationship expert. Analyze table relationships based on business logic and data patterns."
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            analysis = extract_json_from_response(response.content)
            
            if analysis and analysis.get('has_relationship', False):
                return {
                    'from_table': table1.full_name,
                    'to_table': table2.full_name,
                    'link_column': ', '.join(analysis.get('link_columns', [])),
                    'relationship_type': relationship_type,
                    'confidence': analysis.get('confidence', 0.5),
                    'discovery_method': 'llm_analysis',
                    'description': analysis.get('business_description', '')
                }
            
        except Exception as e:
            print(f"⚠️ LLM relationship analysis failed: {e}")
        
        return None
    
    async def _validate_business_logic(self, tables: List[TableInfo], 
                                     llm_results: Dict, relationships: List[Dict]) -> Dict:
        """Validate that business logic makes sense"""
        
        print("   ✅ Validating business entity classifications...")
        
        # Count entities by type
        entity_counts = defaultdict(int)
        high_confidence_entities = defaultdict(list)
        
        for table in tables:
            table_analysis = llm_results.get(table.name, {})
            entity_type = table_analysis.get('entity_type', 'Unknown')
            confidence = table_analysis.get('confidence', 0)
            
            entity_counts[entity_type] += 1
            if confidence >= 0.7:
                high_confidence_entities[entity_type].append({
                    'name': table.name,
                    'confidence': confidence,
                    'purpose': table_analysis.get('business_purpose', '')
                })
        
        # Validate core business entities exist
        core_entities = ['Customer', 'Payment', 'Order']
        missing_entities = []
        weak_entities = []
        
        for entity in core_entities:
            if entity_counts[entity] == 0:
                missing_entities.append(entity)
            elif len(high_confidence_entities[entity]) == 0:
                weak_entities.append(entity)
        
        # Check for customer-payment relationships
        customer_payment_links = [
            r for r in relationships 
            if r['relationship_type'] in ['customer_payment', 'customer_order', 'order_payment']
        ]
        
        validation_summary = {
            'entity_counts': dict(entity_counts),
            'high_confidence_entities': dict(high_confidence_entities),
            'missing_critical_entities': missing_entities,
            'weak_entities': weak_entities,
            'customer_payment_links': len(customer_payment_links),
            'total_relationships': len(relationships),
            'validation_status': 'GOOD' if not missing_entities and customer_payment_links else 'NEEDS_REVIEW'
        }
        
        # Update table semantic profiles
        for table in tables:
            table_analysis = llm_results.get(table.name, {})
            if table_analysis:
                table.semantic_profile = SemanticProfile(
                    entity_type=table_analysis.get('entity_type', 'Unknown'),
                    business_role=table_analysis.get('business_role', 'Unknown'),
                    data_nature='Master' if table_analysis.get('entity_type') in ['Customer', 'Product'] else 'Transaction',
                    contains_personal_data=table_analysis.get('entity_type') == 'Customer',
                    contains_financial_data=table_analysis.get('entity_type') in ['Payment', 'Order', 'Invoice'],
                    primary_purpose=table_analysis.get('business_purpose', ''),
                    confidence=table_analysis.get('confidence', 0.5)
                )
                
                # Store additional metadata
                table.business_indicators = table_analysis.get('key_business_indicators', [])
                table.sample_questions = table_analysis.get('sample_business_questions', [])
        
        return validation_summary
    
    def _get_validation_summary(self, validation_results: Dict, relationships: List[Dict]) -> Dict:
        """Get human-readable validation summary"""
        
        status = validation_results['validation_status']
        missing = validation_results['missing_critical_entities']
        weak = validation_results['weak_entities']
        
        summary = {
            'overall_status': status,
            'issues': [],
            'recommendations': []
        }
        
        if missing:
            summary['issues'].append(f"Missing critical entities: {', '.join(missing)}")
            summary['recommendations'].append("Review table classifications - some business entities may not be identified")
        
        if weak:
            summary['issues'].append(f"Low confidence entities: {', '.join(weak)}")
            summary['recommendations'].append("Review entity classifications for better confidence")
        
        if validation_results['customer_payment_links'] == 0:
            summary['issues'].append("No customer-payment relationships found")
            summary['recommendations'].append("Critical: Customer-payment links needed for business queries")
        
        if not summary['issues']:
            summary['recommendations'].append("Business entity analysis looks good!")
        
        return summary


class EnhancedSemanticAnalyzer:
    """Enhanced semantic analyzer using the improved business entity analyzer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.business_analyzer = BusinessEntityAnalyzer(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.business_analysis: Dict[str, Any] = {}
    
    async def analyze_semantics(self, tables: List[TableInfo]) -> bool:
        """Run enhanced semantic analysis with improved business entity recognition"""
        
        # Check cache first
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        if self.load_from_cache():
            print(f"✅ Loaded enhanced semantic analysis from cache")
            return True
        
        if not tables:
            print("❌ No tables provided for analysis.")
            return False
        
        self.tables = tables
        
        print(f"🚀 ENHANCED semantic analysis starting...")
        print(f"   📊 Analyzing {len(tables)} objects with improved business focus")
        
        try:
            # Run enhanced business entity analysis
            self.business_analysis = await self.business_analyzer.analyze_business_entities(tables)
            
            # Convert relationships to proper format
            self.relationships = []
            for rel_data in self.business_analysis['relationships']:
                self.relationships.append(Relationship(
                    from_table=rel_data['from_table'],
                    to_table=rel_data['to_table'],
                    column=rel_data['link_column'],
                    relationship_type=rel_data['relationship_type'],
                    confidence=rel_data['confidence'],
                    description=rel_data['description']
                ))
            
            # Generate business domain analysis
            await self._analyze_business_domain_enhanced()
            
            # Save results
            print("💾 Saving enhanced semantic analysis...")
            self._save_to_cache(cache_file)
            
            # Show validation results
            self._show_validation_results()
            
            print("✅ ENHANCED semantic analysis completed!")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced semantic analysis failed: {e}")
            return False
    
    async def _analyze_business_domain_enhanced(self):
        """Enhanced business domain analysis"""
        
        validation_results = self.business_analysis.get('validation_results', {})
        entity_counts = validation_results.get('entity_counts', {})
        
        # Determine domain type based on entity distribution
        if entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0:
            domain_type = "CRM/Sales"
        elif entity_counts.get('Patient', 0) > 0:
            domain_type = "Healthcare"
        elif entity_counts.get('Student', 0) > 0:
            domain_type = "Education"
        else:
            domain_type = "Business"
        
        # Generate sample questions based on identified entities
        sample_questions = []
        
        if entity_counts.get('Customer', 0) > 0:
            sample_questions.extend([
                "How many customers do we have?",
                "Who are our top customers?",
                "Show customer contact information"
            ])
        
        if entity_counts.get('Payment', 0) > 0:
            sample_questions.extend([
                "What is our total revenue?",
                "How many customers have paid?",
                "Show payment history",
                "Count paid customers for 2025"
            ])
        
        if entity_counts.get('Order', 0) > 0:
            sample_questions.extend([
                "How many orders do we have?",
                "Show recent orders",
                "What is our average order value?"
            ])
        
        self.domain = BusinessDomain(
            domain_type=domain_type,
            industry="Business Services",
            entities=list(entity_counts.keys()),
            confidence=0.85,
            sample_questions=sample_questions,
            customer_definition="Business entities identified through enhanced analysis"
        )
    
    def _show_validation_results(self):
        """Show validation results to user"""
        
        validation = self.business_analysis.get('validation_results', {})
        
        print(f"\n📊 BUSINESS ENTITY VALIDATION:")
        print(f"   Status: {validation.get('overall_status', 'Unknown')}")
        
        # Show entity counts
        entity_counts = validation.get('entity_counts', {})
        for entity_type, count in entity_counts.items():
            if count > 0:
                print(f"   • {entity_type}: {count} tables")
        
        # Show high confidence entities
        high_conf = validation.get('high_confidence_entities', {})
        for entity_type in ['Customer', 'Payment', 'Order']:
            tables_list = high_conf.get(entity_type, [])
            if tables_list:
                best_table = max(tables_list, key=lambda x: x['confidence'])
                print(f"   🎯 Best {entity_type} table: {best_table['name']} (confidence: {best_table['confidence']:.2f})")
        
        # Show issues and recommendations
        summary = validation.get('validation_summary', {})
        if summary.get('issues'):
            print(f"\n⚠️  Issues found:")
            for issue in summary['issues']:
                print(f"   • {issue}")
        
        if summary.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        # Show relationship summary
        customer_payment_links = validation.get('customer_payment_links', 0)
        total_relationships = validation.get('total_relationships', 0)
        
        print(f"\n🔗 Relationships discovered:")
        print(f"   • Total relationships: {total_relationships}")
        print(f"   • Customer-payment links: {customer_payment_links}")
        
        if customer_payment_links == 0:
            print(f"   ❌ CRITICAL: No customer-payment relationships found!")
            print(f"      This will cause 'paid customer' queries to fail")
    
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
            'business_analysis': self.business_analysis,
            'created': datetime.now().isoformat(),
            'version': '4.0-enhanced-business-entities'
        }
        
        # Convert tables with enhanced semantic profiles
        for table in self.tables:
            table_dict = table_info_to_dict(table)
            # Add enhanced metadata
            if hasattr(table, 'business_indicators'):
                table_dict['business_indicators'] = table.business_indicators
            if hasattr(table, 'sample_questions'):
                table_dict['sample_questions'] = table.sample_questions
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
                        if 'business_indicators' in table_data:
                            table.business_indicators = table_data['business_indicators']
                        if 'sample_questions' in table_data:
                            table.sample_questions = table_data['sample_questions']
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
                
                # Load business analysis
                if 'business_analysis' in data:
                    self.business_analysis = data['business_analysis']
                
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
    
    def get_business_analysis(self) -> Dict[str, Any]:
        """Get detailed business analysis results"""
        return self.business_analysis


# Export classes for import
SemanticAnalyzer = EnhancedSemanticAnalyzer

# Make SimpleLLMClient available at module level
__all__ = ['EnhancedSemanticAnalyzer', 'SemanticAnalyzer', 'SimpleLLMClient', 'BusinessEntityAnalyzer']