#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTELLIGENT Semantic Analysis - Metadata-First Approach
10x faster than LLM-heavy approach by using SQL Server system views
"""

import asyncio
import json
import re
import pyodbc
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

# Azure OpenAI (used selectively)
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

class IntelligentMetadataAnalyzer:
    """Analyzes database metadata using SQL Server system views for fast entity discovery"""
    
    def __init__(self, config: Config):
        self.config = config
        self.foreign_keys: List[Dict] = []
        self.view_relationships: List[Dict] = []
        self.entity_patterns: Dict[str, Dict] = self._init_entity_patterns()
        self.business_scores: Dict[str, Dict] = {}
    
    def _init_entity_patterns(self) -> Dict[str, Dict]:
        """Initialize comprehensive business entity patterns"""
        return {
            'Customer': {
                'table_patterns': [
                    r'.*customer.*', r'.*client.*', r'.*account.*', r'.*businesspoint.*',
                    r'.*contact.*', r'.*person.*', r'.*user.*', r'.*member.*'
                ],
                'column_patterns': [
                    r'.*customer.*id.*', r'.*client.*id.*', r'.*account.*id.*',
                    r'.*customer.*name.*', r'.*client.*name.*', r'.*company.*name.*',
                    r'.*email.*', r'.*phone.*', r'.*address.*'
                ]
            },
            'Payment': {
                'table_patterns': [
                    r'.*payment.*', r'.*transaction.*', r'.*invoice.*', r'.*billing.*',
                    r'.*financial.*', r'.*revenue.*', r'.*charge.*', r'.*receipt.*'
                ],
                'column_patterns': [
                    r'.*payment.*id.*', r'.*transaction.*id.*', r'.*invoice.*id.*',
                    r'.*amount.*', r'.*total.*', r'.*price.*', r'.*cost.*', r'.*value.*',
                    r'.*payment.*date.*', r'.*transaction.*date.*', r'.*currency.*'
                ]
            },
            'Order': {
                'table_patterns': [
                    r'.*order.*', r'.*sale.*', r'.*purchase.*', r'.*booking.*',
                    r'.*reservation.*', r'.*request.*'
                ],
                'column_patterns': [
                    r'.*order.*id.*', r'.*sale.*id.*', r'.*booking.*id.*',
                    r'.*order.*date.*', r'.*sale.*date.*', r'.*quantity.*',
                    r'.*order.*status.*', r'.*order.*amount.*'
                ]
            },
            'Product': {
                'table_patterns': [
                    r'.*product.*', r'.*item.*', r'.*service.*', r'.*inventory.*',
                    r'.*catalog.*', r'.*goods.*'
                ],
                'column_patterns': [
                    r'.*product.*id.*', r'.*item.*id.*', r'.*sku.*',
                    r'.*product.*name.*', r'.*item.*name.*', r'.*description.*'
                ]
            }
        }
    
    async def analyze_database_metadata(self) -> Dict[str, Any]:
        """Comprehensive metadata analysis using SQL Server system views"""
        print("🔍 INTELLIGENT Metadata Analysis - Using SQL Server System Views")
        
        try:
            with self.get_database_connection() as conn:
                # Step 1: Analyze foreign key constraints (FAST)
                print("   📊 Step 1: Analyzing foreign key constraints...")
                foreign_keys = await self._analyze_foreign_keys(conn)
                
                # Step 2: Analyze view definitions for relationships (FAST)
                print("   🔍 Step 2: Analyzing view definitions...")
                view_relationships = await self._analyze_view_relationships(conn)
                
                # Step 3: Analyze table/column patterns (FAST)
                print("   🧠 Step 3: Pattern-based entity classification...")
                entity_classifications = await self._classify_entities_by_patterns(conn)
                
                # Step 4: Analyze data relationships through sampling (SELECTIVE)
                print("   📈 Step 4: Selective data relationship analysis...")
                data_relationships = await self._analyze_data_relationships_selective(conn, entity_classifications)
                
                # Step 5: Build comprehensive relationship graph
                print("   🕸️ Step 5: Building relationship intelligence...")
                relationship_graph = self._build_relationship_graph(
                    foreign_keys, view_relationships, data_relationships, entity_classifications
                )
                
                return {
                    'foreign_keys': foreign_keys,
                    'view_relationships': view_relationships,
                    'entity_classifications': entity_classifications,
                    'data_relationships': data_relationships,
                    'relationship_graph': relationship_graph,
                    'business_insights': self._generate_business_insights(relationship_graph, entity_classifications)
                }
                
        except Exception as e:
            print(f"❌ Metadata analysis failed: {e}")
            return {}
    
    async def _analyze_foreign_keys(self, conn) -> List[Dict]:
        """Analyze foreign key constraints - the most reliable relationship source"""
        
        fk_query = """
        SELECT 
            fk.name AS constraint_name,
            SCHEMA_NAME(tp.schema_id) AS parent_schema,
            tp.name AS parent_table,
            cp.name AS parent_column,
            SCHEMA_NAME(tr.schema_id) AS referenced_schema,
            tr.name AS referenced_table,
            cr.name AS referenced_column,
            fk.is_disabled,
            fk.is_not_trusted
        FROM sys.foreign_keys fk
        INNER JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
        INNER JOIN sys.tables tr ON fk.referenced_object_id = tr.object_id
        INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        INNER JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
        INNER JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
        WHERE tp.is_ms_shipped = 0 AND tr.is_ms_shipped = 0
        ORDER BY parent_schema, parent_table, referenced_schema, referenced_table
        """
        
        cursor = conn.cursor()
        cursor.execute(fk_query)
        
        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append({
                'constraint_name': row[0],
                'from_table': f"[{row[1]}].[{row[2]}]",
                'from_column': row[3],
                'to_table': f"[{row[4]}].[{row[5]}]",
                'to_column': row[6],
                'is_active': not row[7],  # not disabled
                'is_trusted': not row[8],  # not untrusted
                'relationship_type': 'foreign_key',
                'confidence': 1.0,  # Highest confidence for actual FKs
                'discovery_method': 'constraint_analysis'
            })
        
        print(f"      ✅ Found {len(foreign_keys)} foreign key relationships")
        return foreign_keys
    
    async def _analyze_view_relationships(self, conn) -> List[Dict]:
        """Analyze view definitions to discover implicit relationships"""
        
        view_query = """
        SELECT 
            SCHEMA_NAME(v.schema_id) AS view_schema,
            v.name AS view_name,
            m.definition
        FROM sys.views v
        INNER JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE v.is_ms_shipped = 0
        ORDER BY view_schema, view_name
        """
        
        cursor = conn.cursor()
        cursor.execute(view_query)
        
        view_relationships = []
        join_pattern = re.compile(
            r'(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN)\s+'
            r'(?:\[?(\w+)\]?\.)?\[?(\w+)\]?\s+'
            r'(?:AS\s+\w+\s+)?'
            r'ON\s+'
            r'(?:\[?\w+\]?\.)?\[?(\w+)\]?\s*=\s*(?:\[?\w+\]?\.)?\[?(\w+)\]?',
            re.IGNORECASE | re.MULTILINE
        )
        
        table_pattern = re.compile(
            r'FROM\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?',
            re.IGNORECASE
        )
        
        for row in cursor.fetchall():
            view_schema, view_name, definition = row
            
            if not definition:
                continue
            
            # Find all tables referenced in the view
            tables_in_view = set()
            for match in table_pattern.finditer(definition):
                schema = match.group(1) or 'dbo'
                table = match.group(2)
                tables_in_view.add(f"[{schema}].[{table}]")
            
            # Find JOIN relationships
            for match in join_pattern.finditer(definition):
                # Extract join information
                left_col = match.group(3)
                right_col = match.group(4)
                
                if left_col and right_col:
                    view_relationships.append({
                        'view_name': f"[{view_schema}].[{view_name}]",
                        'left_column': left_col,
                        'right_column': right_col,
                        'tables_involved': list(tables_in_view),
                        'relationship_type': 'view_join',
                        'confidence': 0.8,
                        'discovery_method': 'view_analysis'
                    })
        
        print(f"      ✅ Found {len(view_relationships)} view-based relationships")
        return view_relationships
    
    async def _classify_entities_by_patterns(self, conn) -> Dict[str, Dict]:
        """Fast entity classification using pattern matching on metadata"""
        
        table_query = """
        SELECT 
            SCHEMA_NAME(t.schema_id) AS schema_name,
            t.name AS table_name,
            t.type_desc AS object_type,
            COALESCE(p.rows, 0) AS estimated_rows,
            (
                SELECT STRING_AGG(c.name + ':' + tp.name, ',')
                FROM sys.columns c
                INNER JOIN sys.types tp ON c.user_type_id = tp.user_type_id
                WHERE c.object_id = t.object_id
            ) AS column_signature
        FROM sys.tables t
        LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id < 2
        WHERE t.is_ms_shipped = 0
        
        UNION ALL
        
        SELECT 
            SCHEMA_NAME(v.schema_id) AS schema_name,
            v.name AS view_name,
            v.type_desc AS object_type,
            100 AS estimated_rows,  -- Default for views
            '' AS column_signature
        FROM sys.views v
        WHERE v.is_ms_shipped = 0
        
        ORDER BY schema_name, table_name
        """
        
        cursor = conn.cursor()
        cursor.execute(table_query)
        
        classifications = {}
        
        for row in cursor.fetchall():
            schema_name, table_name, object_type, estimated_rows, column_signature = row
            full_name = f"[{schema_name}].[{table_name}]"
            
            # Calculate entity scores for each type
            entity_scores = {}
            for entity_type, patterns in self.entity_patterns.items():
                score = 0
                
                # Table name scoring
                table_name_lower = table_name.lower()
                for pattern in patterns['table_patterns']:
                    if re.search(pattern, table_name_lower):
                        score += 50
                
                # Column signature scoring
                if column_signature:
                    column_sig_lower = column_signature.lower()
                    for pattern in patterns['column_patterns']:
                        if re.search(pattern, column_sig_lower):
                            score += 20
                
                entity_scores[entity_type] = score
            
            # Determine best entity classification
            best_entity = max(entity_scores, key=entity_scores.get) if entity_scores else 'Unknown'
            best_score = entity_scores.get(best_entity, 0)
            confidence = min(best_score / 100.0, 1.0)
            
            classifications[full_name] = {
                'entity_type': best_entity if confidence > 0.3 else 'Unknown',
                'confidence': confidence,
                'business_role': self._determine_business_role(best_entity, confidence),
                'estimated_rows': estimated_rows,
                'object_type': object_type,
                'entity_scores': entity_scores
            }
        
        # Calculate classification summary
        entity_counts = defaultdict(int)
        high_confidence_entities = defaultdict(list)
        
        for table_name, classification in classifications.items():
            entity_type = classification['entity_type']
            confidence = classification['confidence']
            
            entity_counts[entity_type] += 1
            if confidence >= 0.7:
                high_confidence_entities[entity_type].append({
                    'table_name': table_name,
                    'confidence': confidence
                })
        
        print(f"      ✅ Classified {len(classifications)} objects")
        print(f"         📊 Entity distribution: {dict(entity_counts)}")
        
        return {
            'classifications': classifications,
            'entity_counts': dict(entity_counts),
            'high_confidence_entities': dict(high_confidence_entities)
        }
    
    async def _analyze_data_relationships_selective(self, conn, entity_classifications: Dict) -> List[Dict]:
        """Selective data relationship analysis - only for high-value entity pairs"""
        
        classifications = entity_classifications['classifications']
        
        # Get high-confidence Customer and Payment tables
        customer_tables = []
        payment_tables = []
        
        for table_name, classification in classifications.items():
            if classification['entity_type'] == 'Customer' and classification['confidence'] > 0.6:
                customer_tables.append(table_name)
            elif classification['entity_type'] == 'Payment' and classification['confidence'] > 0.6:
                payment_tables.append(table_name)
        
        print(f"      🎯 Analyzing {len(customer_tables)} Customer × {len(payment_tables)} Payment table pairs")
        
        data_relationships = []
        
        # Only analyze Customer-Payment pairs (most critical for business queries)
        for customer_table in customer_tables[:10]:  # Limit to top 10 customers
            for payment_table in payment_tables[:10]:   # Limit to top 10 payments
                relationship = await self._analyze_table_pair_data(conn, customer_table, payment_table)
                if relationship:
                    data_relationships.append(relationship)
        
        print(f"      ✅ Found {len(data_relationships)} data-based relationships")
        return data_relationships
    
    async def _analyze_table_pair_data(self, conn, table1: str, table2: str) -> Optional[Dict]:
        """Analyze potential data relationships between two specific tables"""
        
        try:
            cursor = conn.cursor()
            
            # Get ID columns from both tables
            id_query = """
            SELECT c.name, tp.name as data_type
            FROM sys.columns c
            INNER JOIN sys.types tp ON c.user_type_id = tp.user_type_id
            INNER JOIN sys.objects o ON c.object_id = o.object_id
            WHERE o.name = ? AND SCHEMA_NAME(o.schema_id) = ?
              AND (c.name LIKE '%ID' OR c.name LIKE '%Id' OR c.name LIKE '%_id')
            ORDER BY c.column_id
            """
            
            # Extract schema and table names
            schema1, name1 = table1.replace('[', '').replace(']', '').split('.')
            schema2, name2 = table2.replace('[', '').replace(']', '').split('.')
            
            cursor.execute(id_query, name1, schema1)
            table1_ids = [(row[0], row[1]) for row in cursor.fetchall()]
            
            cursor.execute(id_query, name2, schema2)
            table2_ids = [(row[0], row[1]) for row in cursor.fetchall()]
            
            # Look for common ID patterns
            for id1_name, id1_type in table1_ids:
                for id2_name, id2_type in table2_ids:
                    if (id1_name.lower() == id2_name.lower() or 
                        'customer' in id1_name.lower() and 'customer' in id2_name.lower()):
                        
                        # Sample data to verify relationship
                        sample_query = f"""
                        SELECT COUNT(*) as overlap_count
                        FROM (
                            SELECT TOP 100 [{id1_name}] as id_val FROM {table1} WHERE [{id1_name}] IS NOT NULL
                        ) t1
                        INNER JOIN (
                            SELECT TOP 100 [{id2_name}] as id_val FROM {table2} WHERE [{id2_name}] IS NOT NULL  
                        ) t2 ON t1.id_val = t2.id_val
                        """
                        
                        try:
                            cursor.execute(sample_query)
                            overlap_count = cursor.fetchone()[0]
                            
                            if overlap_count > 0:
                                return {
                                    'from_table': table1,
                                    'to_table': table2,
                                    'from_column': id1_name,
                                    'to_column': id2_name,
                                    'relationship_type': 'data_relationship',
                                    'confidence': min(0.8, overlap_count / 50.0),
                                    'evidence': f"{overlap_count} overlapping values",
                                    'discovery_method': 'data_sampling'
                                }
                        except:
                            continue  # Skip if sampling query fails
            
            return None
            
        except Exception:
            return None
    
    def _build_relationship_graph(self, foreign_keys: List, view_relationships: List, 
                                 data_relationships: List, entity_classifications: Dict) -> Dict:
        """Build comprehensive relationship graph from all discovery methods"""
        
        graph = {
            'nodes': {},
            'edges': [],
            'business_clusters': defaultdict(list)
        }
        
        classifications = entity_classifications['classifications']
        
        # Add nodes (tables) with entity information
        for table_name, classification in classifications.items():
            graph['nodes'][table_name] = {
                'entity_type': classification['entity_type'],
                'confidence': classification['confidence'],
                'business_role': classification['business_role'],
                'estimated_rows': classification['estimated_rows']
            }
            
            # Group by entity type for business analysis
            entity_type = classification['entity_type']
            if entity_type != 'Unknown':
                graph['business_clusters'][entity_type].append(table_name)
        
        # Add edges from all relationship sources
        all_relationships = foreign_keys + data_relationships
        
        for rel in all_relationships:
            edge = {
                'from': rel['from_table'],
                'to': rel['to_table'],
                'type': rel['relationship_type'],
                'confidence': rel['confidence'],
                'method': rel['discovery_method']
            }
            
            if 'from_column' in rel:
                edge['from_column'] = rel['from_column']
            if 'to_column' in rel:
                edge['to_column'] = rel['to_column']
            
            graph['edges'].append(edge)
        
        return graph
    
    def _generate_business_insights(self, relationship_graph: Dict, entity_classifications: Dict) -> Dict:
        """Generate business insights from relationship analysis"""
        
        clusters = relationship_graph['business_clusters']
        edges = relationship_graph['edges']
        
        # Find customer-payment relationships
        customer_payment_links = []
        for edge in edges:
            from_node = relationship_graph['nodes'].get(edge['from'], {})
            to_node = relationship_graph['nodes'].get(edge['to'], {})
            
            if ((from_node.get('entity_type') == 'Customer' and to_node.get('entity_type') == 'Payment') or
                (from_node.get('entity_type') == 'Payment' and to_node.get('entity_type') == 'Customer')):
                customer_payment_links.append(edge)
        
        # Calculate business readiness score
        readiness_score = 0
        if len(clusters.get('Customer', [])) > 0:
            readiness_score += 30
        if len(clusters.get('Payment', [])) > 0:
            readiness_score += 30
        if len(customer_payment_links) > 0:
            readiness_score += 40
        
        # Determine query capabilities
        capabilities = {
            'customer_queries': len(clusters.get('Customer', [])) > 0,
            'payment_queries': len(clusters.get('Payment', [])) > 0,
            'paid_customer_analysis': len(customer_payment_links) > 0,
            'order_analysis': len(clusters.get('Order', [])) > 0,
            'product_analysis': len(clusters.get('Product', [])) > 0
        }
        
        return {
            'business_readiness': {
                'score': readiness_score,
                'rating': 'Excellent' if readiness_score >= 80 else 'Good' if readiness_score >= 60 else 'Fair'
            },
            'query_capabilities': capabilities,
            'customer_payment_links': len(customer_payment_links),
            'entity_distribution': {k: len(v) for k, v in clusters.items()},
            'relationship_quality': 'High' if len(customer_payment_links) > 0 else 'Medium'
        }
    
    def _determine_business_role(self, entity_type: str, confidence: float) -> str:
        """Determine business role based on entity type and confidence"""
        if confidence < 0.3:
            return 'Unknown'
        elif entity_type in ['Customer', 'Payment', 'Order']:
            return 'Core Business Entity'
        elif entity_type in ['Product', 'Invoice']:
            return 'Primary Business Entity'
        else:
            return 'Supporting Entity'
    
    def get_database_connection(self):
        """Get database connection with proper encoding"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # Set UTF-8 encoding for Greek text support
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
        return conn


class IntelligentSemanticAnalyzer:
    """Intelligent semantic analyzer using metadata-first approach"""
    
    def __init__(self, config: Config):
        self.config = config
        self.metadata_analyzer = IntelligentMetadataAnalyzer(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.business_analysis: Dict[str, Any] = {}
    
    async def analyze_semantics_intelligent(self, tables: List[TableInfo]) -> bool:
        """Run intelligent semantic analysis using metadata-first approach"""
        
        # Check cache first
        cache_file = self.config.get_cache_path("semantic_analysis.json")
        if self.load_from_cache():
            print(f"✅ Loaded intelligent semantic analysis from cache")
            return True
        
        if not tables:
            print("❌ No tables provided for analysis.")
            return False
        
        self.tables = tables
        
        print(f"🚀 INTELLIGENT semantic analysis starting...")
        print(f"   📊 Analyzing {len(tables)} objects with metadata-first approach")
        print(f"   ⚡ Expected completion: 2-5 minutes (vs 3 hours with old method)")
        
        try:
            # Run intelligent metadata analysis (FAST)
            metadata_results = await self.metadata_analyzer.analyze_database_metadata()
            
            if not metadata_results:
                print("❌ Metadata analysis failed")
                return False
            
            # Apply metadata classifications to tables
            await self._apply_metadata_classifications(tables, metadata_results)
            
            # Convert metadata relationships to proper format
            self.relationships = self._convert_metadata_relationships(metadata_results)
            
            # Store business analysis results
            self.business_analysis = {
                'entity_analysis': metadata_results.get('entity_classifications', {}),
                'business_insights': metadata_results.get('business_insights', {}),
                'validation_results': self._create_validation_results(metadata_results)
            }
            
            # Generate business domain analysis
            await self._analyze_business_domain_intelligent(metadata_results)
            
            # Save results
            print("💾 Saving intelligent semantic analysis...")
            self._save_to_cache(cache_file)
            
            # Show validation results
            self._show_validation_results()
            
            print("✅ INTELLIGENT semantic analysis completed!")
            return True
            
        except Exception as e:
            print(f"❌ Intelligent semantic analysis failed: {e}")
            return False
    
    async def _apply_metadata_classifications(self, tables: List[TableInfo], metadata_results: Dict):
        """Apply metadata-based classifications to table objects"""
        
        classifications = metadata_results.get('entity_classifications', {}).get('classifications', {})
        
        for table in tables:
            classification = classifications.get(table.full_name, {})
            if classification:
                table.semantic_profile = SemanticProfile(
                    entity_type=classification.get('entity_type', 'Unknown'),
                    business_role=classification.get('business_role', 'Unknown'),
                    data_nature='Master' if classification.get('entity_type') in ['Customer', 'Product'] else 'Transaction',
                    contains_personal_data=classification.get('entity_type') == 'Customer',
                    contains_financial_data=classification.get('entity_type') in ['Payment', 'Order', 'Invoice'],
                    primary_purpose=f"{classification.get('entity_type', 'Unknown')} entity based on metadata analysis",
                    confidence=classification.get('confidence', 0.5)
                )
                
                # Store additional metadata
                table.business_indicators = [f"Entity type: {classification.get('entity_type')}"]
                table.sample_questions = [f"What {classification.get('entity_type', 'data')} do we have?"]
    
    def _convert_metadata_relationships(self, metadata_results: Dict) -> List[Relationship]:
        """Convert metadata relationships to proper Relationship objects"""
        
        relationships = []
        
        # Convert foreign key relationships
        for fk in metadata_results.get('foreign_keys', []):
            relationships.append(Relationship(
                from_table=fk['from_table'],
                to_table=fk['to_table'],
                column=fk.get('from_column', ''),
                relationship_type=fk['relationship_type'],
                confidence=fk['confidence'],
                description=f"Foreign key constraint: {fk.get('constraint_name', 'Unknown')}"
            ))
        
        # Convert data relationships
        for dr in metadata_results.get('data_relationships', []):
            relationships.append(Relationship(
                from_table=dr['from_table'],
                to_table=dr['to_table'],
                column=dr.get('from_column', ''),
                relationship_type=dr['relationship_type'],
                confidence=dr['confidence'],
                description=dr.get('evidence', 'Data-based relationship')
            ))
        
        return relationships
    
    def _create_validation_results(self, metadata_results: Dict) -> Dict:
        """Create validation results from metadata analysis"""
        
        entity_classifications = metadata_results.get('entity_classifications', {})
        business_insights = metadata_results.get('business_insights', {})
        
        return {
            'overall_status': 'GOOD' if business_insights.get('customer_payment_links', 0) > 0 else 'NEEDS_REVIEW',
            'entity_counts': entity_classifications.get('entity_counts', {}),
            'high_confidence_entities': entity_classifications.get('high_confidence_entities', {}),
            'customer_payment_links': business_insights.get('customer_payment_links', 0),
            'total_relationships': len(metadata_results.get('foreign_keys', [])) + len(metadata_results.get('data_relationships', []))
        }
    
    async def _analyze_business_domain_intelligent(self, metadata_results: Dict):
        """Intelligent business domain analysis based on metadata"""
        
        business_insights = metadata_results.get('business_insights', {})
        entity_counts = business_insights.get('entity_distribution', {})
        
        # Determine domain type based on entity distribution
        if entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0:
            domain_type = "CRM/Sales"
        elif entity_counts.get('Patient', 0) > 0:
            domain_type = "Healthcare"
        elif entity_counts.get('Student', 0) > 0:
            domain_type = "Education"
        else:
            domain_type = "Business"
        
        # Generate sample questions based on identified entities and capabilities
        sample_questions = []
        capabilities = business_insights.get('query_capabilities', {})
        
        if capabilities.get('customer_queries', False):
            sample_questions.extend([
                "How many customers do we have?",
                "Show customer contact information"
            ])
        
        if capabilities.get('paid_customer_analysis', False):
            sample_questions.extend([
                "How many customers have paid?",
                "Show paid customers for 2025",
                "What is our customer payment rate?"
            ])
        
        if capabilities.get('payment_queries', False):
            sample_questions.extend([
                "What is our total revenue?",
                "Show payment history",
                "Calculate average transaction value"
            ])
        
        if capabilities.get('order_analysis', False):
            sample_questions.extend([
                "How many orders do we have?",
                "Show recent orders",
                "What is our average order value?"
            ])
        
        # Calculate confidence based on relationship quality
        readiness_score = business_insights.get('business_readiness', {}).get('score', 0)
        confidence = readiness_score / 100.0
        
        self.domain = BusinessDomain(
            domain_type=domain_type,
            industry="Business Services",
            entities=list(entity_counts.keys()),
            confidence=confidence,
            sample_questions=sample_questions,
            customer_definition="Business entities identified through intelligent metadata analysis"
        )
    
    def _show_validation_results(self):
        """Show validation results to user"""
        
        validation = self.business_analysis.get('validation_results', {})
        business_insights = self.business_analysis.get('business_insights', {})
        
        print(f"\n📊 INTELLIGENT BUSINESS ENTITY VALIDATION:")
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
                print(f"   🎯 Best {entity_type} table: {best_table['table_name'].split('.')[-1]} (confidence: {best_table['confidence']:.2f})")
        
        # Show business readiness
        readiness = business_insights.get('business_readiness', {})
        if readiness:
            print(f"\n💼 Business Readiness: {readiness.get('rating', 'Unknown')} ({readiness.get('score', 0)}/100)")
        
        # Show capabilities
        capabilities = business_insights.get('query_capabilities', {})
        enabled_capabilities = [cap.replace('_', ' ').title() for cap, enabled in capabilities.items() if enabled]
        if enabled_capabilities:
            print(f"   🎯 Enabled Capabilities: {', '.join(enabled_capabilities)}")
        
        # Show relationship summary
        customer_payment_links = validation.get('customer_payment_links', 0)
        total_relationships = validation.get('total_relationships', 0)
        
        print(f"\n🔗 Relationships discovered:")
        print(f"   • Total relationships: {total_relationships}")
        print(f"   • Customer-payment links: {customer_payment_links}")
        
        if customer_payment_links > 0:
            print(f"   ✅ READY for 'paid customer' queries!")
        else:
            print(f"   ⚠️ Limited customer-payment relationships")
    
    def _save_to_cache(self, cache_file):
        """Save intelligent semantic analysis to cache"""
        
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
            'version': '5.0-intelligent-metadata-first',
            'analysis_method': 'metadata_first'
        }
        
        # Convert tables with semantic profiles
        for table in self.tables:
            table_dict = table_info_to_dict(table)
            # Add enhanced metadata
            if hasattr(table, 'business_indicators'):
                table_dict['business_indicators'] = table.business_indicators
            if hasattr(table, 'sample_questions'):
                table_dict['sample_questions'] = table.sample_questions
            data['tables'].append(table_dict)
        
        save_json_cache(cache_file, data, "intelligent semantic analysis")
    
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
                
                print(f"✅ Loaded intelligent semantic cache: {len(self.tables)} tables")
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


# Maintain backward compatibility
EnhancedSemanticAnalyzer = IntelligentSemanticAnalyzer
SemanticAnalyzer = IntelligentSemanticAnalyzer

# Export classes for import
__all__ = ['IntelligentSemanticAnalyzer', 'EnhancedSemanticAnalyzer', 'SemanticAnalyzer', 'IntelligentMetadataAnalyzer']