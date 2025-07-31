#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED Semantic Analysis - Comprehensive Structure Discovery
Combines views analysis, foreign keys, and LLM entity scanning for complete understanding
"""

import asyncio
import json
import re
import pyodbc
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

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

@dataclass
class ViewJoinAnalysis:
    """Detailed view join analysis result"""
    view_name: str
    base_tables: List[str]
    join_conditions: List[Dict[str, Any]]
    where_conditions: List[str]
    aggregations: List[str]
    calculated_fields: List[str]
    business_logic_complexity: str
    relationship_strength: float

@dataclass
class ForeignKeyRelationship:
    """Enhanced foreign key relationship with business context"""
    constraint_name: str
    parent_table: str
    parent_column: str
    referenced_table: str
    referenced_column: str
    is_enforced: bool
    cascade_rules: Dict[str, str]
    business_relationship_type: str
    cardinality: str

@dataclass
class EntityDiscovery:
    """LLM-discovered entity information"""
    table_name: str
    entity_type: str
    confidence: float
    business_role: str
    key_attributes: List[str]
    relationships_suggested: List[str]
    business_rules: List[str]
    data_patterns: Dict[str, Any]

class EnhancedViewAnalyzer:
    """Advanced SQL view analyzer with comprehensive JOIN detection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.join_patterns = self._compile_join_patterns()
        self.function_patterns = self._compile_function_patterns()
        
    def _compile_join_patterns(self) -> Dict[str, re.Pattern]:
        """Compile comprehensive JOIN detection patterns"""
        return {
            'inner_join': re.compile(
                r'(?:INNER\s+)?JOIN\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?(?:\s+(?:AS\s+)?(\w+))?\s+'
                r'ON\s+((?:[^)]+|\([^)]*\))*?)(?=\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            'left_join': re.compile(
                r'LEFT\s+(?:OUTER\s+)?JOIN\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?(?:\s+(?:AS\s+)?(\w+))?\s+'
                r'ON\s+((?:[^)]+|\([^)]*\))*?)(?=\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            'right_join': re.compile(
                r'RIGHT\s+(?:OUTER\s+)?JOIN\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?(?:\s+(?:AS\s+)?(\w+))?\s+'
                r'ON\s+((?:[^)]+|\([^)]*\))*?)(?=\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            'full_join': re.compile(
                r'FULL\s+(?:OUTER\s+)?JOIN\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?(?:\s+(?:AS\s+)?(\w+))?\s+'
                r'ON\s+((?:[^)]+|\([^)]*\))*?)(?=\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            'cross_join': re.compile(
                r'CROSS\s+JOIN\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?(?:\s+(?:AS\s+)?(\w+))?',
                re.IGNORECASE
            ),
            'from_tables': re.compile(
                r'FROM\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?(?:\s+(?:AS\s+)?(\w+))?',
                re.IGNORECASE
            )
        }
    
    def _compile_function_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for SQL functions and aggregations"""
        return {
            'aggregations': re.compile(
                r'\b(COUNT|SUM|AVG|MIN|MAX|STDEV|VAR)\s*\([^)]+\)',
                re.IGNORECASE
            ),
            'window_functions': re.compile(
                r'\b(ROW_NUMBER|RANK|DENSE_RANK|NTILE|LAG|LEAD)\s*\([^)]*\)\s+OVER\s*\(',
                re.IGNORECASE
            ),
            'case_statements': re.compile(
                r'\bCASE\s+(?:WHEN|[^)]+)+\s+END\b',
                re.IGNORECASE | re.DOTALL
            ),
            'subqueries': re.compile(
                r'\(\s*SELECT\s+[^)]+\)',
                re.IGNORECASE | re.DOTALL
            )
        }
    
    async def analyze_all_views(self, conn) -> List[ViewJoinAnalysis]:
        """Analyze all views with comprehensive JOIN detection"""
        
        view_query = """
        SELECT 
            SCHEMA_NAME(v.schema_id) AS view_schema,
            v.name AS view_name,
            m.definition,
            v.create_date,
            v.modify_date,
            CASE 
                WHEN m.definition LIKE '%JOIN%' THEN 'Complex'
                WHEN m.definition LIKE '%UNION%' THEN 'Union'
                WHEN m.definition LIKE '%SELECT%SELECT%' THEN 'Nested'
                ELSE 'Simple'
            END AS complexity_hint
        FROM sys.views v
        INNER JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE v.is_ms_shipped = 0
        ORDER BY complexity_hint DESC, view_schema, view_name
        """
        
        cursor = conn.cursor()
        cursor.execute(view_query)
        
        views_analysis = []
        
        print(f"   🔍 Analyzing view definitions for comprehensive JOIN patterns...")
        view_rows = cursor.fetchall()
        
        for row in tqdm(view_rows, desc="Analyzing views", unit="view"):
            view_schema, view_name, definition, create_date, modify_date, complexity_hint = row
            
            if not definition:
                continue
            
            analysis = await self._analyze_single_view(
                f"[{view_schema}].[{view_name}]", 
                definition, 
                complexity_hint
            )
            
            if analysis:
                views_analysis.append(analysis)
        
        print(f"   ✅ Analyzed {len(views_analysis)} views with JOIN patterns")
        return views_analysis
    
    async def _analyze_single_view(self, view_name: str, definition: str, complexity_hint: str) -> Optional[ViewJoinAnalysis]:
        """Analyze a single view with detailed JOIN pattern extraction"""
        
        try:
            # Clean and normalize the SQL definition
            cleaned_definition = self._clean_sql_definition(definition)
            
            # Extract base tables
            base_tables = self._extract_base_tables(cleaned_definition)
            
            # Extract JOIN conditions with detailed analysis
            join_conditions = self._extract_join_conditions(cleaned_definition)
            
            # Extract WHERE conditions
            where_conditions = self._extract_where_conditions(cleaned_definition)
            
            # Extract aggregations and calculated fields
            aggregations = self._extract_aggregations(cleaned_definition)
            calculated_fields = self._extract_calculated_fields(cleaned_definition)
            
            # Determine business logic complexity
            business_complexity = self._assess_business_complexity(
                cleaned_definition, join_conditions, aggregations, calculated_fields
            )
            
            # Calculate relationship strength based on JOIN patterns
            relationship_strength = self._calculate_relationship_strength(join_conditions, base_tables)
            
            return ViewJoinAnalysis(
                view_name=view_name,
                base_tables=base_tables,
                join_conditions=join_conditions,
                where_conditions=where_conditions,
                aggregations=aggregations,
                calculated_fields=calculated_fields,
                business_logic_complexity=business_complexity,
                relationship_strength=relationship_strength
            )
            
        except Exception as e:
            print(f"   ⚠️ Failed to analyze view {view_name}: {e}")
            return None
    
    def _clean_sql_definition(self, definition: str) -> str:
        """Clean and normalize SQL definition for parsing"""
        # Remove comments
        definition = re.sub(r'--.*$', '', definition, flags=re.MULTILINE)
        definition = re.sub(r'/\*.*?\*/', '', definition, flags=re.DOTALL)
        
        # Normalize whitespace
        definition = re.sub(r'\s+', ' ', definition)
        
        # Remove CREATE VIEW portion
        definition = re.sub(r'^.*?AS\s+', '', definition, flags=re.IGNORECASE)
        
        return definition.strip()
    
    def _extract_base_tables(self, definition: str) -> List[str]:
        """Extract all base tables referenced in the view"""
        tables = set()
        
        # Extract from FROM clause
        from_matches = self.join_patterns['from_tables'].findall(definition)
        for match in from_matches:
            schema, table, alias = match
            if table:
                full_name = f"[{schema or 'dbo'}].[{table}]"
                tables.add(full_name)
        
        # Extract from all JOIN types
        for join_type, pattern in self.join_patterns.items():
            if join_type != 'from_tables':
                matches = pattern.findall(definition)
                for match in matches:
                    if len(match) >= 2:
                        schema, table = match[0], match[1]
                        if table:
                            full_name = f"[{schema or 'dbo'}].[{table}]"
                            tables.add(full_name)
        
        return list(tables)
    
    def _extract_join_conditions(self, definition: str) -> List[Dict[str, Any]]:
        """Extract detailed JOIN conditions with relationship analysis"""
        join_conditions = []
        
        for join_type, pattern in self.join_patterns.items():
            if join_type == 'from_tables':
                continue
                
            matches = pattern.findall(definition)
            for match in matches:
                if len(match) >= 4:
                    schema, table, alias, condition = match[0], match[1], match[2], match[3]
                    
                    # Parse the join condition
                    condition_analysis = self._parse_join_condition(condition)
                    
                    join_info = {
                        'join_type': join_type.replace('_', ' ').title(),
                        'table': f"[{schema or 'dbo'}].[{table}]",
                        'alias': alias or table,
                        'condition': condition.strip(),
                        'left_columns': condition_analysis['left_columns'],
                        'right_columns': condition_analysis['right_columns'],
                        'operators': condition_analysis['operators'],
                        'is_equality_join': condition_analysis['is_equality_join'],
                        'complexity': condition_analysis['complexity']
                    }
                    
                    join_conditions.append(join_info)
        
        return join_conditions
    
    def _parse_join_condition(self, condition: str) -> Dict[str, Any]:
        """Parse JOIN condition to extract column relationships"""
        
        # Extract column references (table.column or [table].[column])
        column_pattern = re.compile(r'(?:\[?(\w+)\]?\.)?\[?(\w+)\]?')
        columns = column_pattern.findall(condition)
        
        # Extract operators
        operator_pattern = re.compile(r'(\=|\<\>|\!\=|\<|\>|\<=|\>=|LIKE|IN|EXISTS)', re.IGNORECASE)
        operators = operator_pattern.findall(condition)
        
        # Determine if it's an equality join
        is_equality_join = '=' in condition and '<>' not in condition and '!=' not in condition
        
        # Assess complexity
        complexity = 'Simple'
        if len(operators) > 1:
            complexity = 'Medium'
        if any(op.upper() in ['LIKE', 'IN', 'EXISTS'] for op in operators):
            complexity = 'Complex'
        if '(' in condition or 'CASE' in condition.upper():
            complexity = 'Complex'
        
        return {
            'left_columns': [f"{col[0] or 'unknown'}.{col[1]}" for col in columns[:len(columns)//2]],
            'right_columns': [f"{col[0] or 'unknown'}.{col[1]}" for col in columns[len(columns)//2:]],
            'operators': operators,
            'is_equality_join': is_equality_join,
            'complexity': complexity
        }
    
    def _extract_where_conditions(self, definition: str) -> List[str]:
        """Extract WHERE clause conditions"""
        where_pattern = re.compile(r'WHERE\s+(.*?)(?=\s+GROUP\s+BY|\s+HAVING|\s+ORDER\s+BY|$)', re.IGNORECASE | re.DOTALL)
        matches = where_pattern.findall(definition)
        
        conditions = []
        for match in matches:
            # Split by AND/OR to get individual conditions
            condition_parts = re.split(r'\s+(?:AND|OR)\s+', match, flags=re.IGNORECASE)
            conditions.extend([part.strip() for part in condition_parts if part.strip()])
        
        return conditions
    
    def _extract_aggregations(self, definition: str) -> List[str]:
        """Extract aggregation functions"""
        aggregations = []
        
        agg_matches = self.function_patterns['aggregations'].findall(definition)
        aggregations.extend(agg_matches)
        
        window_matches = self.function_patterns['window_functions'].findall(definition)
        aggregations.extend([f"{match} (Window Function)" for match in window_matches])
        
        return aggregations
    
    def _extract_calculated_fields(self, definition: str) -> List[str]:
        """Extract calculated fields and CASE statements"""
        calculated = []
        
        case_matches = self.function_patterns['case_statements'].findall(definition)
        calculated.extend([f"CASE Statement: {match[:50]}..." if len(match) > 50 else f"CASE Statement: {match}" for match in case_matches])
        
        # Extract calculated fields (columns with mathematical operations)
        calc_pattern = re.compile(r'(\w+\s*[\+\-\*\/]\s*\w+)', re.IGNORECASE)
        calc_matches = calc_pattern.findall(definition)
        calculated.extend([f"Calculation: {match}" for match in calc_matches])
        
        return calculated
    
    def _assess_business_complexity(self, definition: str, join_conditions: List[Dict], 
                                  aggregations: List[str], calculated_fields: List[str]) -> str:
        """Assess the business logic complexity of the view"""
        
        complexity_score = 0
        
        # JOIN complexity
        complexity_score += len(join_conditions) * 2
        complex_joins = sum(1 for join in join_conditions if join['complexity'] == 'Complex')
        complexity_score += complex_joins * 3
        
        # Aggregation complexity
        complexity_score += len(aggregations) * 2
        
        # Calculated fields complexity
        complexity_score += len(calculated_fields) * 1
        
        # Subquery complexity
        subquery_matches = self.function_patterns['subqueries'].findall(definition)
        complexity_score += len(subquery_matches) * 4
        
        # UNION complexity
        if 'UNION' in definition.upper():
            complexity_score += 5
        
        # CTE complexity
        if 'WITH' in definition.upper() and 'AS' in definition.upper():
            complexity_score += 3
        
        if complexity_score <= 5:
            return 'Simple'
        elif complexity_score <= 15:
            return 'Medium'
        elif complexity_score <= 30:
            return 'Complex'
        else:
            return 'Very Complex'
    
    def _calculate_relationship_strength(self, join_conditions: List[Dict], base_tables: List[str]) -> float:
        """Calculate relationship strength based on JOIN patterns"""
        if not join_conditions or len(base_tables) < 2:
            return 0.0
        
        strength = 0.0
        
        # Base strength from number of joins
        strength += min(len(join_conditions) / len(base_tables), 1.0) * 0.4
        
        # Strength from equality joins (stronger relationships)
        equality_joins = sum(1 for join in join_conditions if join['is_equality_join'])
        strength += (equality_joins / len(join_conditions)) * 0.3
        
        # Strength from join types (INNER joins indicate stronger relationships)
        inner_joins = sum(1 for join in join_conditions if 'Inner' in join['join_type'])
        strength += (inner_joins / len(join_conditions)) * 0.3
        
        return min(strength, 1.0)

class EnhancedForeignKeyAnalyzer:
    """Enhanced foreign key analyzer with business context and cascade rules"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def analyze_all_foreign_keys(self, conn) -> List[ForeignKeyRelationship]:
        """Comprehensive foreign key analysis with business context - FIXED SQL query"""
        
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
            fk.is_not_trusted,
            fk.delete_referential_action_desc,
            fk.update_referential_action_desc,
            -- FIXED: Get data types correctly from sys.types
            pt.name AS parent_column_type,
            rt.name AS referenced_column_type,
            cp.is_nullable AS parent_nullable,
            -- Business context hints
            CASE 
                WHEN tp.name LIKE '%customer%' OR tr.name LIKE '%customer%' THEN 'Customer Relationship'
                WHEN tp.name LIKE '%order%' OR tr.name LIKE '%order%' THEN 'Order Relationship'
                WHEN tp.name LIKE '%product%' OR tr.name LIKE '%product%' THEN 'Product Relationship'
                WHEN tp.name LIKE '%payment%' OR tr.name LIKE '%payment%' THEN 'Payment Relationship'
                ELSE 'General Relationship'
            END AS business_context,
            -- Cardinality estimation
            CASE 
                WHEN cp.is_nullable = 1 THEN 'One-to-Zero-or-Many'
                ELSE 'One-to-Many'
            END AS estimated_cardinality
        FROM sys.foreign_keys fk
        INNER JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
        INNER JOIN sys.tables tr ON fk.referenced_object_id = tr.object_id
        INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        INNER JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
        INNER JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
        -- FIXED: Join with sys.types to get actual data type names
        INNER JOIN sys.types pt ON cp.user_type_id = pt.user_type_id
        INNER JOIN sys.types rt ON cr.user_type_id = rt.user_type_id
        WHERE tp.is_ms_shipped = 0 AND tr.is_ms_shipped = 0
        ORDER BY business_context, parent_schema, parent_table, referenced_schema, referenced_table
        """
        
        cursor = conn.cursor()
        cursor.execute(fk_query)
        
        foreign_keys = []
        
        print(f"   📊 Analyzing foreign key constraints with business context...")
        
        for row in cursor.fetchall():
            (constraint_name, parent_schema, parent_table, parent_column,
            referenced_schema, referenced_table, referenced_column,
            is_disabled, is_not_trusted, delete_action, update_action,
            parent_column_type, referenced_column_type, parent_nullable,
            business_context, estimated_cardinality) = row
            
            fk_relationship = ForeignKeyRelationship(
                constraint_name=constraint_name,
                parent_table=f"[{parent_schema}].[{parent_table}]",
                parent_column=parent_column,
                referenced_table=f"[{referenced_schema}].[{referenced_table}]",
                referenced_column=referenced_column,
                is_enforced=not is_disabled,
                cascade_rules={
                    'on_delete': delete_action,
                    'on_update': update_action
                },
                business_relationship_type=business_context,
                cardinality=estimated_cardinality
            )
            
            foreign_keys.append(fk_relationship)
        
        print(f"   ✅ Found {len(foreign_keys)} foreign key relationships with business context")
        
        # Analyze relationship patterns
        self._analyze_relationship_patterns(foreign_keys)
        
        return foreign_keys
    
    def _analyze_relationship_patterns(self, foreign_keys: List[ForeignKeyRelationship]):
        """Analyze patterns in foreign key relationships"""
        
        # Group by business relationship type
        business_groups = defaultdict(list)
        for fk in foreign_keys:
            business_groups[fk.business_relationship_type].append(fk)
        
        print(f"   📈 Foreign Key Relationship Patterns:")
        for business_type, fks in business_groups.items():
            print(f"      • {business_type}: {len(fks)} relationships")
        
        # Analyze cascade rules
        cascade_patterns = defaultdict(int)
        for fk in foreign_keys:
            delete_rule = fk.cascade_rules.get('on_delete', 'NO ACTION')
            cascade_patterns[delete_rule] += 1
        
        print(f"   🔗 Cascade Rule Patterns:")
        for rule, count in cascade_patterns.items():
            print(f"      • {rule}: {count} relationships")

class LLMEntityScanner:
    """Advanced LLM-based entity scanning with structured analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.1,  # Low temperature for consistent analysis
            request_timeout=180
        )
    
    async def scan_entities_comprehensive(self, tables: List[TableInfo]) -> List[EntityDiscovery]:
        """Comprehensive LLM-based entity scanning with business intelligence"""
        
        print(f"   🧠 LLM Entity Scanning: Analyzing {len(tables)} objects for business entities...")
        
        # Group tables for batch analysis
        table_batches = self._create_table_batches(tables, batch_size=10)
        
        all_discoveries = []
        
        for i, batch in enumerate(table_batches):
            print(f"   📋 Processing LLM batch {i+1}/{len(table_batches)} ({len(batch)} tables)")
            
            batch_discoveries = await self._analyze_table_batch(batch)
            all_discoveries.extend(batch_discoveries)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        print(f"   ✅ LLM Entity Scanning completed: {len(all_discoveries)} entities discovered")
        
        return all_discoveries
    
    def _create_table_batches(self, tables: List[TableInfo], batch_size: int = 10) -> List[List[TableInfo]]:
        """Create batches of tables for LLM analysis"""
        batches = []
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i + batch_size]
            batches.append(batch)
        return batches
    
    async def _analyze_table_batch(self, tables: List[TableInfo]) -> List[EntityDiscovery]:
        """Analyze a batch of tables with LLM"""
        
        # Prepare table context for LLM
        table_contexts = []
        for table in tables:
            context = {
                'table_name': table.full_name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': [
                    {
                        'name': col['name'],
                        'type': col['data_type'],
                        'nullable': col.get('nullable', True)
                    } for col in table.columns[:15]  # Limit columns for context
                ],
                'sample_data': table.sample_data[:3] if table.sample_data else []
            }
            table_contexts.append(context)
        
        # Create comprehensive analysis prompt
        prompt = self._create_entity_analysis_prompt(table_contexts)
        
        try:
            system_message = """You are an expert database architect and business analyst specializing in entity recognition and business domain modeling. 

Analyze database tables to identify business entities, relationships, and domain patterns. Focus on:
1. Business entity classification (Customer, Order, Product, Payment, etc.)
2. Business relationships and dependencies
3. Data patterns and business rules
4. Domain-specific insights

Respond with a well-structured JSON analysis."""
            
            response = await self.llm.ainvoke([
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ])
            
            # Parse LLM response
            discoveries = self._parse_llm_entity_response(response.content, tables)
            return discoveries
            
        except Exception as e:
            print(f"   ⚠️ LLM entity analysis failed for batch: {e}")
            return []
    
    def _create_entity_analysis_prompt(self, table_contexts: List[Dict]) -> str:
        """Create comprehensive entity analysis prompt for LLM"""
        
        prompt = f"""
Analyze the following database tables to identify business entities and relationships:

DATABASE TABLES FOR ANALYSIS:
{json.dumps(table_contexts, indent=2)}

ANALYSIS REQUIREMENTS:
1. For each table, identify:
   - Primary business entity type (Customer, Order, Product, Payment, Employee, etc.)
   - Confidence level (0.0 to 1.0)
   - Business role (Core, Supporting, Reference, Lookup)
   - Key business attributes that define this entity
   - Suggested relationships to other entities
   - Business rules or constraints evident from the data

2. Look for patterns that indicate:
   - Master data vs transactional data
   - Lookup tables vs operational tables
   - Bridge/junction tables for many-to-many relationships
   - Audit/history tables

3. Consider column naming conventions:
   - ID columns indicate entity keys
   - Foreign key patterns (CustomerID, OrderID, etc.)
   - Date columns indicate temporal aspects
   - Amount/quantity columns indicate measurements

4. Analyze sample data for:
   - Data patterns and formats
   - Business context clues
   - Relationship indicators

RESPONSE FORMAT (JSON):
{{
  "entity_discoveries": [
    {{
      "table_name": "[schema].[table]",
      "entity_type": "Customer|Order|Product|Payment|Employee|Other",
      "confidence": 0.0-1.0,
      "business_role": "Core|Supporting|Reference|Lookup|Bridge|Audit",
      "key_attributes": ["column1", "column2"],
      "relationships_suggested": ["RelationshipToEntity1", "RelationshipToEntity2"],
      "business_rules": ["BusinessRule1", "BusinessRule2"],
      "data_patterns": {{
        "primary_key_pattern": "description",
        "temporal_aspects": "description",
        "data_volume_implications": "description"
      }}
    }}
  ],
  "domain_insights": {{
    "primary_business_domain": "CRM|ERP|Financial|Healthcare|Education|Other",
    "confidence": 0.0-1.0,
    "key_business_processes": ["Process1", "Process2"],
    "data_maturity": "Basic|Intermediate|Advanced"
  }}
}}

Provide comprehensive analysis focusing on business value and entity relationships.
"""
        return prompt
    
    def _parse_llm_entity_response(self, response: str, tables: List[TableInfo]) -> List[EntityDiscovery]:
        """Parse LLM response into EntityDiscovery objects"""
        
        try:
            # Extract JSON from response
            json_data = extract_json_from_response(response)
            
            if not json_data or 'entity_discoveries' not in json_data:
                print(f"   ⚠️ Invalid LLM response format")
                return []
            
            discoveries = []
            
            for discovery_data in json_data['entity_discoveries']:
                try:
                    discovery = EntityDiscovery(
                        table_name=discovery_data.get('table_name', ''),
                        entity_type=discovery_data.get('entity_type', 'Unknown'),
                        confidence=float(discovery_data.get('confidence', 0.0)),
                        business_role=discovery_data.get('business_role', 'Unknown'),
                        key_attributes=discovery_data.get('key_attributes', []),
                        relationships_suggested=discovery_data.get('relationships_suggested', []),
                        business_rules=discovery_data.get('business_rules', []),
                        data_patterns=discovery_data.get('data_patterns', {})
                    )
                    discoveries.append(discovery)
                    
                except Exception as e:
                    print(f"   ⚠️ Failed to parse entity discovery: {e}")
                    continue
            
            return discoveries
            
        except Exception as e:
            print(f"   ⚠️ Failed to parse LLM entity response: {e}")
            return []

class ComprehensiveStructureAnalyzer:
    """Comprehensive analyzer that combines all analysis methods"""
    
    def __init__(self, config: Config):
        self.config = config
        self.view_analyzer = EnhancedViewAnalyzer(config)
        self.fk_analyzer = EnhancedForeignKeyAnalyzer(config)
        self.llm_scanner = LLMEntityScanner(config)
        
        # Results storage
        self.view_analyses: List[ViewJoinAnalysis] = []
        self.foreign_key_relationships: List[ForeignKeyRelationship] = []
        self.entity_discoveries: List[EntityDiscovery] = []
        self.comprehensive_graph: nx.DiGraph = nx.DiGraph()
        
    async def analyze_complete_structure(self, tables: List[TableInfo]) -> Dict[str, Any]:
        """Run comprehensive structure analysis combining all methods"""
        
        print(f"🚀 COMPREHENSIVE Structure Analysis: Combining all discovery methods")
        print(f"   📊 Analyzing {len(tables)} database objects")
        
        results = {}
        
        try:
            with self._get_connection() as conn:
                # 1. Enhanced View Analysis
                print(f"\n🔍 Step 1: Enhanced View JOIN Analysis")
                self.view_analyses = await self.view_analyzer.analyze_all_views(conn)
                results['view_analyses'] = self._serialize_view_analyses()
                
                # 2. Enhanced Foreign Key Analysis
                print(f"\n📊 Step 2: Enhanced Foreign Key Analysis")
                self.foreign_key_relationships = await self.fk_analyzer.analyze_all_foreign_keys(conn)
                results['foreign_key_relationships'] = self._serialize_foreign_keys()
                
            # 3. LLM Entity Scanning
            print(f"\n🧠 Step 3: LLM Entity Scanning")
            self.entity_discoveries = await self.llm_scanner.scan_entities_comprehensive(tables)
            results['entity_discoveries'] = self._serialize_entity_discoveries()
            
            # 4. Build Comprehensive Relationship Graph
            print(f"\n🕸️ Step 4: Building Comprehensive Relationship Graph")
            comprehensive_graph = self._build_comprehensive_graph(tables)
            results['comprehensive_graph'] = self._serialize_graph(comprehensive_graph)
            
            # 5. Generate Business Intelligence Summary
            print(f"\n💼 Step 5: Generating Business Intelligence Summary")
            business_intelligence = self._generate_business_intelligence()
            results['business_intelligence'] = business_intelligence
            
            # 6. Create Entity-Relationship Matrix
            print(f"\n📋 Step 6: Creating Entity-Relationship Matrix")
            er_matrix = self._create_entity_relationship_matrix()
            results['entity_relationship_matrix'] = er_matrix
            
            return results
            
        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")
            return {}
    
    def _build_comprehensive_graph(self, tables: List[TableInfo]) -> nx.DiGraph:
        """Build comprehensive relationship graph from all analysis sources"""
        
        graph = nx.DiGraph()
        
        # Add all tables as nodes with entity information
        for table in tables:
            entity_info = self._get_entity_info_for_table(table)
            
            graph.add_node(table.full_name, **{
                'table_name': table.name,
                'schema': table.schema,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'entity_type': entity_info.get('entity_type', 'Unknown'),
                'business_role': entity_info.get('business_role', 'Unknown'),
                'confidence': entity_info.get('confidence', 0.0)
            })
        
        # Add foreign key relationships
        for fk in self.foreign_key_relationships:
            if graph.has_node(fk.parent_table) and graph.has_node(fk.referenced_table):
                graph.add_edge(fk.parent_table, fk.referenced_table, **{
                    'relationship_type': 'foreign_key',
                    'constraint_name': fk.constraint_name,
                    'columns': f"{fk.parent_column} -> {fk.referenced_column}",
                    'business_type': fk.business_relationship_type,
                    'cardinality': fk.cardinality,
                    'confidence': 1.0,
                    'source': 'constraint_analysis'
                })
        
        # Add view-based relationships
        for view_analysis in self.view_analyses:
            view_node = view_analysis.view_name
            
            if graph.has_node(view_node):
                # Connect view to all its base tables
                for base_table in view_analysis.base_tables:
                    if graph.has_node(base_table):
                        graph.add_edge(view_node, base_table, **{
                            'relationship_type': 'view_dependency',
                            'strength': view_analysis.relationship_strength,
                            'complexity': view_analysis.business_logic_complexity,
                            'confidence': view_analysis.relationship_strength,
                            'source': 'view_analysis'
                        })
                
                # Add relationships implied by JOIN conditions
                for join in view_analysis.join_conditions:
                    join_table = join['table']
                    if graph.has_node(join_table):
                        for base_table in view_analysis.base_tables:
                            if base_table != join_table and graph.has_node(base_table):
                                graph.add_edge(base_table, join_table, **{
                                    'relationship_type': 'view_join_implied',
                                    'join_type': join['join_type'],
                                    'is_equality_join': join['is_equality_join'],
                                    'confidence': 0.7 if join['is_equality_join'] else 0.5,
                                    'source': 'view_join_analysis'
                                })
        
        # Add LLM-suggested relationships
        for discovery in self.entity_discoveries:
            source_table = discovery.table_name
            if graph.has_node(source_table):
                for suggested_rel in discovery.relationships_suggested:
                    # Try to find matching table for relationship
                    for table in tables:
                        if (suggested_rel.lower() in table.name.lower() or 
                            table.name.lower() in suggested_rel.lower()):
                            if graph.has_node(table.full_name):
                                graph.add_edge(source_table, table.full_name, **{
                                    'relationship_type': 'llm_suggested',
                                    'suggestion': suggested_rel,
                                    'confidence': discovery.confidence * 0.8,  # Slightly lower confidence
                                    'source': 'llm_analysis'
                                })
        
        self.comprehensive_graph = graph
        return graph
    
    def _get_entity_info_for_table(self, table: TableInfo) -> Dict[str, Any]:
        """Get entity information for a table from all analysis sources"""
        
        # Check LLM discoveries first (highest detail)
        for discovery in self.entity_discoveries:
            if discovery.table_name == table.full_name:
                return {
                    'entity_type': discovery.entity_type,
                    'business_role': discovery.business_role,
                    'confidence': discovery.confidence
                }
        
        # Check existing semantic profile
        if table.semantic_profile:
            return {
                'entity_type': table.semantic_profile.entity_type,
                'business_role': table.semantic_profile.business_role,
                'confidence': table.semantic_profile.confidence
            }
        
        # Default unknown
        return {
            'entity_type': 'Unknown',
            'business_role': 'Unknown',
            'confidence': 0.0
        }
    
    def _generate_business_intelligence(self) -> Dict[str, Any]:
        """Generate comprehensive business intelligence summary"""
        
        # Analyze entity distribution from all sources
        entity_distribution = defaultdict(int)
        high_confidence_entities = defaultdict(list)
        
        for discovery in self.entity_discoveries:
            entity_distribution[discovery.entity_type] += 1
            if discovery.confidence >= 0.7:
                high_confidence_entities[discovery.entity_type].append({
                    'table_name': discovery.table_name,
                    'confidence': discovery.confidence
                })
        
        # Analyze relationship quality
        total_relationships = len(self.foreign_key_relationships)
        view_relationships = len(self.view_analyses)
        llm_relationships = sum(len(d.relationships_suggested) for d in self.entity_discoveries)
        
        # Calculate business readiness score
        readiness_score = 0
        
        # Core entities present
        core_entities = ['Customer', 'Order', 'Product', 'Payment']
        present_core = sum(1 for entity in core_entities if entity_distribution[entity] > 0)
        readiness_score += (present_core / len(core_entities)) * 40
        
        # Relationship completeness
        if total_relationships > 0:
            readiness_score += 25
        if view_relationships > 0:
            readiness_score += 15
        if llm_relationships > 0:
            readiness_score += 10
        
        # High confidence classifications
        total_high_conf = sum(len(entities) for entities in high_confidence_entities.values())
        total_entities = sum(entity_distribution.values())
        if total_entities > 0:
            confidence_ratio = total_high_conf / total_entities
            readiness_score += confidence_ratio * 10
        
        # Determine capabilities
        capabilities = {
            'customer_queries': entity_distribution.get('Customer', 0) > 0,
            'order_analysis': entity_distribution.get('Order', 0) > 0,
            'product_analysis': entity_distribution.get('Product', 0) > 0,
            'payment_analysis': entity_distribution.get('Payment', 0) > 0,
            'relationship_analysis': total_relationships > 0,
            'view_based_analysis': view_relationships > 0,
            'advanced_querying': readiness_score > 60
        }
        
        return {
            'entity_distribution': dict(entity_distribution),
            'high_confidence_entities': dict(high_confidence_entities),
            'relationship_summary': {
                'foreign_key_relationships': total_relationships,
                'view_relationships': view_relationships,
                'llm_suggested_relationships': llm_relationships,
                'total_discovered_relationships': total_relationships + view_relationships + llm_relationships
            },
            'business_readiness': {
                'score': int(readiness_score),
                'rating': 'Excellent' if readiness_score >= 80 else 'Good' if readiness_score >= 60 else 'Fair' if readiness_score >= 40 else 'Limited'
            },
            'query_capabilities': capabilities,
            'analysis_quality': {
                'constraint_coverage': 'High' if total_relationships > 5 else 'Medium' if total_relationships > 0 else 'Low',
                'view_analysis_depth': 'High' if view_relationships > 10 else 'Medium' if view_relationships > 3 else 'Low',
                'entity_classification_confidence': 'High' if total_high_conf > total_entities * 0.7 else 'Medium' if total_high_conf > total_entities * 0.4 else 'Low'
            }
        }
    
    def _create_entity_relationship_matrix(self) -> Dict[str, Any]:
        """Create entity-relationship matrix for visualization"""
        
        # Get all unique entity types
        entity_types = set()
        for discovery in self.entity_discoveries:
            entity_types.add(discovery.entity_type)
        
        entity_types = sorted(list(entity_types))
        
        # Create relationship matrix
        matrix = {}
        for source_entity in entity_types:
            matrix[source_entity] = {}
            for target_entity in entity_types:
                matrix[source_entity][target_entity] = {
                    'relationship_count': 0,
                    'relationship_types': [],
                    'confidence_avg': 0.0
                }
        
        # Populate matrix from foreign keys
        for fk in self.foreign_key_relationships:
            source_entity = self._get_entity_type_for_table(fk.parent_table)
            target_entity = self._get_entity_type_for_table(fk.referenced_table)
            
            if source_entity in matrix and target_entity in matrix:
                matrix[source_entity][target_entity]['relationship_count'] += 1
                matrix[source_entity][target_entity]['relationship_types'].append('Foreign Key')
                matrix[source_entity][target_entity]['confidence_avg'] = 1.0
        
        # Populate matrix from view relationships
        for view_analysis in self.view_analyses:
            view_entity = self._get_entity_type_for_table(view_analysis.view_name)
            for base_table in view_analysis.base_tables:
                base_entity = self._get_entity_type_for_table(base_table)
                
                if view_entity in matrix and base_entity in matrix:
                    matrix[view_entity][base_entity]['relationship_count'] += 1
                    matrix[view_entity][base_entity]['relationship_types'].append('View Dependency')
                    current_conf = matrix[view_entity][base_entity]['confidence_avg']
                    matrix[view_entity][base_entity]['confidence_avg'] = max(current_conf, view_analysis.relationship_strength)
        
        return {
            'entity_types': entity_types,
            'relationship_matrix': matrix,
            'matrix_summary': {
                'total_entity_types': len(entity_types),
                'total_relationship_pairs': sum(1 for source in matrix.values() for target in source.values() if target['relationship_count'] > 0),
                'strongest_relationships': self._find_strongest_relationships(matrix)
            }
        }
    
    def _get_entity_type_for_table(self, table_name: str) -> str:
        """Get entity type for a table name"""
        for discovery in self.entity_discoveries:
            if discovery.table_name == table_name:
                return discovery.entity_type
        return 'Unknown'
    
    def _find_strongest_relationships(self, matrix: Dict) -> List[Dict]:
        """Find the strongest relationships in the matrix"""
        strong_relationships = []
        
        for source_entity, targets in matrix.items():
            for target_entity, relationship_info in targets.items():
                if relationship_info['relationship_count'] > 0:
                    strength = relationship_info['confidence_avg'] * relationship_info['relationship_count']
                    strong_relationships.append({
                        'source': source_entity,
                        'target': target_entity,
                        'strength': strength,
                        'count': relationship_info['relationship_count'],
                        'types': relationship_info['relationship_types']
                    })
        
        # Sort by strength and return top 10
        strong_relationships.sort(key=lambda x: x['strength'], reverse=True)
        return strong_relationships[:10]
    
    def _serialize_view_analyses(self) -> List[Dict]:
        """Serialize view analyses for JSON storage"""
        return [
            {
                'view_name': va.view_name,
                'base_tables': va.base_tables,
                'join_conditions': va.join_conditions,
                'where_conditions': va.where_conditions,
                'aggregations': va.aggregations,
                'calculated_fields': va.calculated_fields,
                'business_logic_complexity': va.business_logic_complexity,
                'relationship_strength': va.relationship_strength
            }
            for va in self.view_analyses
        ]
    
    def _serialize_foreign_keys(self) -> List[Dict]:
        """Serialize foreign key relationships for JSON storage"""
        return [
            {
                'constraint_name': fk.constraint_name,
                'parent_table': fk.parent_table,
                'parent_column': fk.parent_column,
                'referenced_table': fk.referenced_table,
                'referenced_column': fk.referenced_column,
                'is_enforced': fk.is_enforced,
                'cascade_rules': fk.cascade_rules,
                'business_relationship_type': fk.business_relationship_type,
                'cardinality': fk.cardinality
            }
            for fk in self.foreign_key_relationships
        ]
    
    def _serialize_entity_discoveries(self) -> List[Dict]:
        """Serialize entity discoveries for JSON storage"""
        return [
            {
                'table_name': ed.table_name,
                'entity_type': ed.entity_type,
                'confidence': ed.confidence,
                'business_role': ed.business_role,
                'key_attributes': ed.key_attributes,
                'relationships_suggested': ed.relationships_suggested,
                'business_rules': ed.business_rules,
                'data_patterns': ed.data_patterns
            }
            for ed in self.entity_discoveries
        ]
    
    def _serialize_graph(self, graph: nx.DiGraph) -> Dict:
        """Serialize NetworkX graph for JSON storage"""
        return {
            'nodes': [
                {'id': node, **data}
                for node, data in graph.nodes(data=True)
            ],
            'edges': [
                {'source': source, 'target': target, **data}
                for source, target, data in graph.edges(data=True)
            ],
            'graph_metrics': {
                'total_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_connected': nx.is_weakly_connected(graph)
            }
        }
    
    def _get_connection(self):
        """Get database connection with proper encoding"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # Set UTF-8 encoding for Greek text support
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
        return conn

# Enhanced main analyzer class that uses comprehensive structure analysis
class EnhancedSemanticAnalyzer:
    """Enhanced semantic analyzer using comprehensive structure discovery"""
    
    def __init__(self, config: Config):
        self.config = config
        self.comprehensive_analyzer = ComprehensiveStructureAnalyzer(config)
        self.tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []
        self.comprehensive_results: Dict[str, Any] = {}
    
    async def analyze_semantics_comprehensive(self, tables: List[TableInfo]) -> bool:
        """Run comprehensive semantic analysis using all discovery methods"""
        
        # Check cache first
        cache_file = self.config.get_cache_path("comprehensive_semantic_analysis.json")
        if self.load_from_cache():
            print(f"✅ Loaded comprehensive semantic analysis from cache")
            return True
        
        if not tables:
            print("❌ No tables provided for analysis.")
            return False
        
        self.tables = tables
        
        print(f"🚀 COMPREHENSIVE semantic analysis starting...")
        print(f"   📊 Analyzing {len(tables)} objects with all discovery methods")
        print(f"   🔍 Views + Foreign Keys + LLM Entity Scanning")
        
        try:
            # Run comprehensive structure analysis
            self.comprehensive_results = await self.comprehensive_analyzer.analyze_complete_structure(tables)
            
            if not self.comprehensive_results:
                print("❌ Comprehensive analysis failed")
                return False
            
            # Apply comprehensive results to tables
            await self._apply_comprehensive_results(tables)
            
            # Convert results to relationships
            self.relationships = self._extract_comprehensive_relationships()
            
            # Generate business domain analysis
            await self._analyze_business_domain_comprehensive()
            
            # Save results
            print("💾 Saving comprehensive semantic analysis...")
            self._save_to_cache(cache_file)
            
            # Show comprehensive results
            self._show_comprehensive_results()
            
            print("✅ COMPREHENSIVE semantic analysis completed!")
            return True
            
        except Exception as e:
            print(f"❌ Comprehensive semantic analysis failed: {e}")
            return False
    
    async def _apply_comprehensive_results(self, tables: List[TableInfo]):
        """Apply comprehensive analysis results to table objects"""
        
        entity_discoveries = self.comprehensive_results.get('entity_discoveries', [])
        
        # Create lookup for entity discoveries
        entity_lookup = {ed['table_name']: ed for ed in entity_discoveries}
        
        for table in tables:
            entity_data = entity_lookup.get(table.full_name)
            if entity_data:
                table.semantic_profile = SemanticProfile(
                    entity_type=entity_data.get('entity_type', 'Unknown'),
                    business_role=entity_data.get('business_role', 'Unknown'),
                    data_nature='Master' if entity_data.get('business_role') in ['Core', 'Reference'] else 'Transaction',
                    contains_personal_data=entity_data.get('entity_type') in ['Customer', 'Employee', 'Person'],
                    contains_financial_data=entity_data.get('entity_type') in ['Payment', 'Order', 'Invoice'],
                    primary_purpose=f"{entity_data.get('entity_type', 'Unknown')} entity from comprehensive analysis",
                    confidence=entity_data.get('confidence', 0.5)
                )
                
                # Store additional comprehensive data
                table.business_indicators = entity_data.get('business_rules', [])
                table.sample_questions = [f"Show {entity_data.get('entity_type', 'data')} information"]
                
                # Add relationship context
                table.relationship_context = {
                    'key_attributes': entity_data.get('key_attributes', []),
                    'suggested_relationships': entity_data.get('relationships_suggested', []),
                    'data_patterns': entity_data.get('data_patterns', {})
                }
    
    def _extract_comprehensive_relationships(self) -> List[Relationship]:
        """Extract relationships from comprehensive analysis results"""
        
        relationships = []
        
        # Extract foreign key relationships
        fk_relationships = self.comprehensive_results.get('foreign_key_relationships', [])
        for fk in fk_relationships:
            relationships.append(Relationship(
                from_table=fk['parent_table'],
                to_table=fk['referenced_table'],
                column=fk['parent_column'],
                relationship_type='foreign_key',
                confidence=1.0,
                description=f"Foreign key constraint: {fk['constraint_name']}",
                business_significance=0.9,
                discovery_method='constraint_analysis',
                from_entity_type=self._get_entity_type_for_table_name(fk['parent_table']),
                to_entity_type=self._get_entity_type_for_table_name(fk['referenced_table']),
                business_description=fk['business_relationship_type'],
                validation_status='validated'
            ))
        
        # Extract view-based relationships
        view_analyses = self.comprehensive_results.get('view_analyses', [])
        for view in view_analyses:
            for base_table in view['base_tables']:
                relationships.append(Relationship(
                    from_table=view['view_name'],
                    to_table=base_table,
                    column='',
                    relationship_type='view_dependency',
                    confidence=view['relationship_strength'],
                    description=f"View dependency with {view['business_logic_complexity']} complexity",
                    business_significance=view['relationship_strength'],
                    discovery_method='view_analysis',
                    from_entity_type='View',
                    to_entity_type=self._get_entity_type_for_table_name(base_table),
                    business_description=f"View aggregation/transformation",
                    validation_status='inferred'
                ))
        
        return relationships
    
    def _get_entity_type_for_table_name(self, table_name: str) -> str:
        """Get entity type for a table name from comprehensive results"""
        entity_discoveries = self.comprehensive_results.get('entity_discoveries', [])
        for ed in entity_discoveries:
            if ed['table_name'] == table_name:
                return ed['entity_type']
        return 'Unknown'
    
    async def _analyze_business_domain_comprehensive(self):
        """Analyze business domain using comprehensive results"""
        
        business_intelligence = self.comprehensive_results.get('business_intelligence', {})
        
        if not business_intelligence:
            return
        
        entity_distribution = business_intelligence.get('entity_distribution', {})
        capabilities = business_intelligence.get('query_capabilities', {})
        readiness = business_intelligence.get('business_readiness', {})
        
        # Determine domain type
        if entity_distribution.get('Customer', 0) > 0 and entity_distribution.get('Order', 0) > 0:
            domain_type = "E-Commerce/Sales"
        elif entity_distribution.get('Customer', 0) > 0 and entity_distribution.get('Payment', 0) > 0:
            domain_type = "CRM/Financial"
        elif entity_distribution.get('Patient', 0) > 0:
            domain_type = "Healthcare"
        elif entity_distribution.get('Student', 0) > 0:
            domain_type = "Education"
        else:
            domain_type = "Business Operations"
        
        # Generate comprehensive sample questions
        sample_questions = []
        
        if capabilities.get('customer_queries', False):
            sample_questions.extend([
                "How many customers do we have?",
                "Show customer demographics and segmentation",
                "Which customers are most active?"
            ])
        
        if capabilities.get('order_analysis', False):
            sample_questions.extend([
                "What is our total order volume?",
                "Show order trends over time",
                "Which products are most popular?"
            ])
        
        if capabilities.get('payment_analysis', False):
            sample_questions.extend([
                "What is our total revenue?",
                "Show payment methods breakdown",
                "Calculate average transaction value"
            ])
        
        if capabilities.get('relationship_analysis', False):
            sample_questions.extend([
                "Show customer purchase behavior",
                "Analyze cross-selling opportunities",
                "Identify high-value customer segments"
            ])
        
        self.domain = BusinessDomain(
            domain_type=domain_type,
            industry="Multi-Domain Business",
            entities=list(entity_distribution.keys()),
            confidence=readiness.get('score', 0) / 100.0,
            sample_questions=sample_questions,
            customer_definition="Comprehensive analysis of business entities and relationships",
            core_business_processes=["Data Analysis", "Relationship Discovery", "Business Intelligence"],
            business_readiness_score=readiness.get('score', 0),
            query_capabilities=capabilities,
            relationship_quality=business_intelligence.get('analysis_quality', {}).get('constraint_coverage', 'Unknown')
        )
    
    def _show_comprehensive_results(self):
        """Show comprehensive analysis results"""
        
        business_intelligence = self.comprehensive_results.get('business_intelligence', {})
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
        
        # Entity distribution
        entity_distribution = business_intelligence.get('entity_distribution', {})
        print(f"   🏢 Business Entities Discovered:")
        for entity_type, count in entity_distribution.items():
            print(f"      • {entity_type}: {count} tables")
        
        # Relationship summary
        rel_summary = business_intelligence.get('relationship_summary', {})
        print(f"   🔗 Relationship Discovery Summary:")
        print(f"      • Foreign Key Relationships: {rel_summary.get('foreign_key_relationships', 0)}")
        print(f"      • View-based Relationships: {rel_summary.get('view_relationships', 0)}")
        print(f"      • LLM Suggested Relationships: {rel_summary.get('llm_suggested_relationships', 0)}")
        print(f"      • Total Discovered: {rel_summary.get('total_discovered_relationships', 0)}")
        
        # Business readiness
        readiness = business_intelligence.get('business_readiness', {})
        print(f"   💼 Business Readiness: {readiness.get('rating', 'Unknown')} ({readiness.get('score', 0)}/100)")
        
        # Capabilities
        capabilities = business_intelligence.get('query_capabilities', {})
        enabled_caps = [cap.replace('_', ' ').title() for cap, enabled in capabilities.items() if enabled]
        if enabled_caps:
            print(f"   🎯 Enabled Capabilities: {', '.join(enabled_caps)}")
        
        # Analysis quality
        quality = business_intelligence.get('analysis_quality', {})
        print(f"   📈 Analysis Quality:")
        for aspect, rating in quality.items():
            print(f"      • {aspect.replace('_', ' ').title()}: {rating}")
    
    def _save_to_cache(self, cache_file):
        """Save comprehensive analysis to cache"""
        
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
                    'description': r.description,
                    'business_significance': r.business_significance,
                    'discovery_method': r.discovery_method,
                    'from_entity_type': r.from_entity_type,
                    'to_entity_type': r.to_entity_type,
                    'business_description': r.business_description,
                    'validation_status': r.validation_status
                } for r in self.relationships
            ],
            'comprehensive_results': self.comprehensive_results,
            'created': datetime.now().isoformat(),
            'version': '6.0-comprehensive-structure-analysis',
            'analysis_method': 'comprehensive_multi_source'
        }
        
        # Convert tables with comprehensive profiles
        for table in self.tables:
            table_dict = table_info_to_dict(table)
            # Add comprehensive metadata
            if hasattr(table, 'business_indicators'):
                table_dict['business_indicators'] = table.business_indicators
            if hasattr(table, 'sample_questions'):
                table_dict['sample_questions'] = table.sample_questions
            if hasattr(table, 'relationship_context'):
                table_dict['relationship_context'] = table.relationship_context
            data['tables'].append(table_dict)
        
        save_json_cache(cache_file, data, "comprehensive semantic analysis")
    
    def load_from_cache(self) -> bool:
        """Load comprehensive analysis from cache"""
        cache_file = self.config.get_cache_path("comprehensive_semantic_analysis.json")
        data = load_json_cache(cache_file, self.config.semantic_cache_hours, "comprehensive semantic cache")
        
        if data:
            try:
                # Load tables
                if 'tables' in data:
                    self.tables = []
                    for table_data in data['tables']:
                        table = dict_to_table_info(table_data)
                        # Load comprehensive metadata
                        if 'business_indicators' in table_data:
                            table.business_indicators = table_data['business_indicators']
                        if 'sample_questions' in table_data:
                            table.sample_questions = table_data['sample_questions']
                        if 'relationship_context' in table_data:
                            table.relationship_context = table_data['relationship_context']
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
                            description=rel_data.get('description', ''),
                            business_significance=rel_data.get('business_significance', 0.0),
                            discovery_method=rel_data.get('discovery_method', 'unknown'),
                            from_entity_type=rel_data.get('from_entity_type', 'Unknown'),
                            to_entity_type=rel_data.get('to_entity_type', 'Unknown'),
                            business_description=rel_data.get('business_description', ''),
                            validation_status=rel_data.get('validation_status', 'unvalidated')
                        ))
                
                # Load comprehensive results
                if 'comprehensive_results' in data:
                    self.comprehensive_results = data['comprehensive_results']
                
                print(f"✅ Loaded comprehensive semantic cache: {len(self.tables)} tables, {len(self.relationships)} relationships")
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to load comprehensive semantic cache: {e}")
        
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
        """Get comprehensive business analysis results"""
        return self.comprehensive_results

# For backward compatibility
IntelligentSemanticAnalyzer = EnhancedSemanticAnalyzer
SemanticAnalyzer = EnhancedSemanticAnalyzer

# Export all classes
__all__ = [
    'EnhancedSemanticAnalyzer', 'IntelligentSemanticAnalyzer', 'SemanticAnalyzer',
    'ComprehensiveStructureAnalyzer', 'EnhancedViewAnalyzer', 'EnhancedForeignKeyAnalyzer', 'LLMEntityScanner',
    'ViewJoinAnalysis', 'ForeignKeyRelationship', 'EntityDiscovery'
]