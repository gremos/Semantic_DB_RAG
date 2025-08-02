#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Utility Functions - Simple and Maintainable
Supports all features from README including 4-stage pipeline, view analysis, and business intelligence
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# ==========================================
# JSON and Response Processing
# ==========================================

def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from LLM response with robust parsing"""
    if not response:
        return None
    
    try:
        # Clean the response
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
        
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^[^{[]*', '', cleaned)  # Remove text before JSON
        cleaned = re.sub(r'[^}\]]*$', '', cleaned)  # Remove text after JSON
        
        # Try to parse JSON
        return json.loads(cleaned)
        
    except Exception as e:
        print(f"   ⚠️ Failed to parse JSON response: {e}")
        print(f"   📝 Response preview: {response[:200]}...")
        return None

# ==========================================
# Entity Classification
# ==========================================

def classify_entity_type_by_patterns(table_name: str, columns: List[Dict], sample_data: List[Dict]) -> Tuple[str, float]:
    """
    Classify entity type using comprehensive pattern matching
    Returns: (entity_type, confidence)
    """
    name_lower = table_name.lower()
    column_names = [col.get('name', '').lower() for col in columns]
    
    # High confidence exact matches
    high_confidence_patterns = {
        'Customer': {
            'name_patterns': ['customer', 'client', 'account', 'buyer'],
            'column_patterns': ['customer_id', 'customer_name', 'email', 'phone', 'address'],
            'confidence': 0.9
        },
        'Payment': {
            'name_patterns': ['payment', 'transaction', 'billing', 'invoice', 'charge'],
            'column_patterns': ['amount', 'payment_date', 'payment_method', 'total', 'charge_amount'],
            'confidence': 0.9
        },
        'Order': {
            'name_patterns': ['order', 'sale', 'purchase', 'booking'],
            'column_patterns': ['order_id', 'order_date', 'quantity', 'customer_id', 'order_total'],
            'confidence': 0.9
        },
        'Product': {
            'name_patterns': ['product', 'item', 'inventory', 'catalog'],
            'column_patterns': ['product_id', 'product_name', 'price', 'category', 'sku'],
            'confidence': 0.9
        },
        'User': {
            'name_patterns': ['user', 'person', 'employee', 'staff', 'member'],
            'column_patterns': ['username', 'password', 'email', 'first_name', 'last_name', 'user_id'],
            'confidence': 0.8
        },
        'Company': {
            'name_patterns': ['company', 'business', 'organization', 'vendor', 'supplier'],
            'column_patterns': ['company_name', 'business_name', 'tax_id', 'company_id'],
            'confidence': 0.8
        }
    }
    
    # Check high confidence patterns
    for entity_type, patterns in high_confidence_patterns.items():
        # Check name patterns
        name_score = sum(1 for pattern in patterns['name_patterns'] if pattern in name_lower)
        
        # Check column patterns
        column_score = sum(1 for pattern in patterns['column_patterns'] 
                          if any(pattern in col for col in column_names))
        
        # Calculate confidence
        if name_score > 0:
            confidence = patterns['confidence']
            if column_score >= 2:  # Bonus for matching columns
                confidence = min(0.95, confidence + 0.05)
            return entity_type, confidence
        elif column_score >= 3:  # Strong column evidence
            return entity_type, patterns['confidence'] - 0.1
    
    # Medium confidence patterns
    medium_confidence_patterns = {
        'Contact': {
            'column_patterns': ['email', 'phone', 'address', 'contact_name', 'mobile'],
            'confidence': 0.6
        },
        'Financial': {
            'column_patterns': ['amount', 'total', 'price', 'cost', 'revenue', 'balance'],
            'confidence': 0.6
        },
        'Reference': {
            'name_patterns': ['lookup', 'reference', 'code', 'type', 'status', 'category'],
            'confidence': 0.5
        },
        'Log': {
            'name_patterns': ['log', 'audit', 'history', 'event', 'activity'],
            'confidence': 0.4
        }
    }
    
    for entity_type, patterns in medium_confidence_patterns.items():
        name_matches = sum(1 for pattern in patterns.get('name_patterns', []) if pattern in name_lower)
        column_matches = sum(1 for pattern in patterns.get('column_patterns', []) 
                           if any(pattern in col for col in column_names))
        
        if name_matches > 0 or column_matches >= 2:
            return entity_type, patterns['confidence']
    
    # Low confidence - general entity indicators
    if any(col in column_names for col in ['id', 'name', 'created_date', 'updated_date']):
        return 'Entity', 0.3
    
    return 'Unknown', 0.0

def classify_entity_type_simple(table_name: str, columns: List[Dict], sample_data: List[Dict]) -> Tuple[str, float]:
    """Simplified version for backward compatibility"""
    return classify_entity_type_by_patterns(table_name, columns, sample_data)

# ==========================================
# Relationship Discovery
# ==========================================

def find_table_relationships_comprehensive(tables: List['TableInfo']) -> List[Tuple[str, str, str, float]]:
    """
    Comprehensive relationship discovery using multiple methods
    Returns: List of (from_table, to_table, relationship_type, confidence)
    """
    relationships = []
    
    # Create lookup structures
    table_lookup = {table.name.lower(): table for table in tables}
    column_index = {}
    
    for table in tables:
        column_index[table.full_name] = [col.get('name', '').lower() for col in table.columns]
    
    # Method 1: Foreign key pattern matching
    for table in tables:
        table_cols = column_index[table.full_name]
        
        for col_name in table_cols:
            if col_name.endswith('_id') and col_name != 'id':
                entity_name = col_name[:-3]  # Remove '_id'
                
                # Look for matching table
                for other_table in tables:
                    if table.full_name == other_table.full_name:
                        continue
                    
                    other_name = other_table.name.lower()
                    if (entity_name in other_name or 
                        other_name in entity_name or
                        similar_names(entity_name, other_name)):
                        
                        relationships.append((
                            table.full_name,
                            other_table.full_name,
                            'foreign_key_pattern',
                            0.8
                        ))
                        break
            
            elif col_name.endswith('id') and col_name != 'id':
                entity_name = col_name[:-2]  # Remove 'id'
                
                # Look for matching table
                for other_table in tables:
                    if table.full_name == other_table.full_name:
                        continue
                    
                    other_name = other_table.name.lower()
                    if entity_name in other_name or other_name in entity_name:
                        relationships.append((
                            table.full_name,
                            other_table.full_name,
                            'id_pattern',
                            0.7
                        ))
                        break
    
    # Method 2: Entity type relationships
    entity_relationships = {
        'Customer': ['Payment', 'Order', 'Contact'],
        'Order': ['Customer', 'Product', 'Payment'],
        'Payment': ['Customer', 'Order'],
        'Product': ['Order', 'Company'],
        'User': ['Company', 'Contact']
    }
    
    for table in tables:
        if not hasattr(table, 'entity_type') or table.entity_type == 'Unknown':
            continue
        
        related_entities = entity_relationships.get(table.entity_type, [])
        
        for other_table in tables:
            if (hasattr(other_table, 'entity_type') and 
                other_table.entity_type in related_entities and
                table.full_name != other_table.full_name):
                
                relationships.append((
                    table.full_name,
                    other_table.full_name,
                    'entity_relationship',
                    0.6
                ))
    
    # Method 3: Sample data correlation (simplified)
    relationships.extend(_find_data_correlations(tables))
    
    # Remove duplicates and return
    return list(set(relationships))

def find_table_relationships_simple(tables: List['TableInfo']) -> List[Tuple[str, str, str, float]]:
    """Simplified version for backward compatibility"""
    return find_table_relationships_comprehensive(tables)

def _find_data_correlations(tables: List['TableInfo']) -> List[Tuple[str, str, str, float]]:
    """Find relationships based on sample data correlations"""
    relationships = []
    
    # Simple implementation - look for ID values that appear in multiple tables
    id_values = {}
    
    for table in tables:
        if not table.sample_data:
            continue
        
        # Find ID-like columns
        id_columns = [col['name'] for col in table.columns 
                     if col['name'].lower().endswith('id')]
        
        for id_col in id_columns:
            values = set()
            for row in table.sample_data:
                if id_col in row and row[id_col] is not None:
                    values.add(str(row[id_col]))
            
            if values:
                id_values[(table.full_name, id_col)] = values
    
    # Look for overlapping values
    id_items = list(id_values.items())
    for i, ((table1, col1), values1) in enumerate(id_items):
        for ((table2, col2), values2) in id_items[i+1:]:
            if table1 != table2:
                overlap = values1.intersection(values2)
                if overlap and len(overlap) >= 2:  # At least 2 common values
                    confidence = min(0.7, len(overlap) / max(len(values1), len(values2)))
                    relationships.append((
                        table1, table2, 'data_correlation', confidence
                    ))
    
    return relationships

def similar_names(name1: str, name2: str, threshold: float = 0.6) -> bool:
    """Check if two names are similar using simple string matching"""
    if not name1 or not name2:
        return False
    
    # Simple similarity check
    longer = max(len(name1), len(name2))
    common = sum(1 for a, b in zip(name1, name2) if a == b)
    
    return (common / longer) >= threshold

# ==========================================
# 4-Stage Pipeline Support
# ==========================================

def find_related_tables_fuzzy(question: str, tables: List['TableInfo']) -> List['TableInfo']:
    """
    Enhanced fuzzy table matching for 4-stage pipeline
    Returns tables sorted by relevance score
    """
    question_lower = question.lower()
    scored_tables = []
    
    # Enhanced keyword mapping
    business_keywords = {
        # Customer-related
        'customer': ['Customer', 'Contact', 'User', 'Person', 'Client'],
        'client': ['Customer', 'Contact', 'User', 'Person'],
        'user': ['User', 'Customer', 'Person', 'Contact'],
        
        # Payment-related
        'payment': ['Payment', 'Financial', 'Transaction', 'Billing'],
        'paid': ['Payment', 'Financial', 'Transaction'],
        'money': ['Payment', 'Financial'],
        'revenue': ['Payment', 'Financial', 'Order'],
        'total': ['Payment', 'Financial', 'Order'],
        
        # Order-related
        'order': ['Order', 'Sale', 'Purchase', 'Product'],
        'sale': ['Order', 'Sale', 'Product', 'Customer'],
        'purchase': ['Order', 'Purchase', 'Product'],
        
        # Product-related
        'product': ['Product', 'Item', 'Inventory', 'Order'],
        'item': ['Product', 'Item', 'Order'],
        
        # General
        'count': ['Customer', 'Order', 'Product', 'Payment', 'User'],
        'list': ['Customer', 'Order', 'Product', 'User'],
        'show': ['Customer', 'Order', 'Product', 'Payment', 'User']
    }
    
    for table in tables:
        score = 0
        
        # Direct name matching (highest priority)
        table_name_lower = table.name.lower()
        for word in question_lower.split():
            if word in table_name_lower:
                score += 15
        
        # Entity type matching
        entity_type = getattr(table, 'entity_type', 'Unknown')
        if entity_type != 'Unknown':
            for keyword, entity_types in business_keywords.items():
                if keyword in question_lower and entity_type in entity_types:
                    score += 10
        
        # Column name matching
        column_names = [col.get('name', '').lower() for col in table.columns]
        for word in question_lower.split():
            for col_name in column_names:
                if word in col_name:
                    score += 5
        
        # Confidence boost
        confidence = getattr(table, 'confidence', 0.0)
        if confidence > 0.8:
            score += 8
        elif confidence > 0.6:
            score += 5
        elif confidence > 0.4:
            score += 3
        
        # Data availability boost
        if table.row_count > 1000:
            score += 6
        elif table.row_count > 100:
            score += 4
        elif table.row_count > 0:
            score += 2
        
        # Sample data quality
        if len(table.sample_data) >= 3:
            score += 3
        elif len(table.sample_data) > 0:
            score += 1
        
        # Business role priority
        business_role = getattr(table, 'business_role', 'Unknown')
        if business_role == 'Core':
            score += 5
        elif business_role == 'Supporting':
            score += 3
        
        if score > 0:
            scored_tables.append((table, score))
    
    # Sort by score and return top matches
    scored_tables.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 8 tables (increased from 5 for better coverage)
    return [table for table, score in scored_tables[:8]]

def generate_business_sql_prompt(question: str, tables: List['TableInfo'], intent: Dict[str, Any] = None) -> str:
    """Generate enhanced SQL prompt for business questions"""
    
    # Prepare table schemas with business context
    table_schemas = []
    for table in tables:
        entity_type = getattr(table, 'entity_type', 'Unknown')
        confidence = getattr(table, 'confidence', 0.0)
        
        # Select most relevant columns (prioritize business columns)
        business_columns = []
        other_columns = []
        
        for col in table.columns[:15]:  # Limit to 15 columns
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            col_info = f"{col_name} ({col_type})"
            
            # Prioritize business-relevant columns
            if any(keyword in col_name.lower() for keyword in 
                  ['id', 'name', 'date', 'amount', 'total', 'count', 'email', 'phone']):
                business_columns.append(col_info)
            else:
                other_columns.append(col_info)
        
        # Combine with business columns first
        all_columns = business_columns + other_columns[:10]  # Limit total columns
        
        table_schema = {
            'table': table.full_name,
            'entity_type': entity_type,
            'confidence': confidence,
            'columns': all_columns,
            'row_count': table.row_count,
            'sample_data': table.sample_data[0] if table.sample_data else None
        }
        table_schemas.append(table_schema)
    
    # Enhanced prompt with business context
    prompt = f"""
Generate SQL Server T-SQL query to answer this business question: "{question}"

AVAILABLE TABLES WITH BUSINESS CONTEXT:
{json.dumps(table_schemas, indent=2, default=str)}

QUERY REQUIREMENTS:
1. Use proper SQL Server syntax with square brackets for table/column names
2. Use appropriate JOINs if multiple tables are needed
3. Add meaningful WHERE clauses for business logic
4. Use TOP 100 to limit results unless asking for counts/totals/sums
5. Use meaningful column aliases for results
6. Consider date filters (2025, current year, etc.) if mentioned
7. Format numbers appropriately for business reporting

BUSINESS INTELLIGENCE GUIDELINES:
- For customer queries: Focus on active customers with data
- For payment queries: Consider date ranges and positive amounts
- For count queries: Use COUNT(DISTINCT ...) when appropriate
- For revenue queries: SUM amounts and group appropriately
- Join tables using ID relationships when multiple entities needed

Generate ONLY the SQL query, no explanations or comments:
"""
    
    return prompt

def generate_simple_sql_prompt(question: str, tables: List['TableInfo']) -> str:
    """Simplified version for backward compatibility"""
    return generate_business_sql_prompt(question, tables)

# ==========================================
# LLM Prompt Generation
# ==========================================

def create_llm_classification_prompt(tables_batch: List['TableInfo']) -> str:
    """Create comprehensive LLM prompt for entity classification"""
    
    table_descriptions = []
    for table in tables_batch:
        # Prepare column information
        columns_info = []
        for col in table.columns[:10]:  # Limit columns
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            is_pk = col.get('is_primary_key', False)
            pk_marker = " [PK]" if is_pk else ""
            columns_info.append(f"{col_name} ({col_type}){pk_marker}")
        
        # Prepare sample data
        sample_info = ""
        if table.sample_data:
            sample_row = table.sample_data[0]
            sample_items = []
            for key, value in list(sample_row.items())[:5]:  # Limit sample columns
                if value is not None:
                    sample_items.append(f"{key}: {str(value)[:50]}")
            sample_info = ", ".join(sample_items)
        
        table_descriptions.append({
            'table_name': table.full_name,
            'object_type': table.object_type,
            'row_count': table.row_count,
            'columns': columns_info,
            'sample_data': sample_info,
            'existing_classification': getattr(table, 'entity_type', 'Unknown'),
            'existing_confidence': getattr(table, 'confidence', 0.0)
        })
    
    prompt = f"""
Analyze these database tables and classify each as a business entity type.

TABLES TO CLASSIFY:
{json.dumps(table_descriptions, indent=2)}

CLASSIFICATION GUIDELINES:
- Customer: Tables storing customer/client information (names, contacts, accounts)
- Payment: Tables with financial transactions, payments, billing information
- Order: Tables with sales, purchases, bookings, order information
- Product: Tables with products, items, inventory, catalog information
- User: Tables with system users, employees, authentication information
- Company: Tables with business entities, vendors, suppliers, organizations
- Contact: Tables focused on contact information, addresses, communications
- Financial: Tables with financial data, accounting, revenue information
- Reference: Lookup tables, codes, categories, status values
- System: Technical tables for system operation, logs, configuration
- Unknown: Cannot determine business purpose from available information

For each table, determine:
1. Entity Type: Choose from the types above
2. Confidence: 0.0 to 1.0 (how certain you are about the classification)
3. Business Role: Core (primary business entity), Supporting (secondary), Reference (lookup), System (technical)
4. Reasoning: Brief explanation of your classification

Consider:
- Table and column names (most important indicator)
- Data types and primary keys
- Sample data content and patterns
- Relationships suggested by foreign key patterns

Respond with JSON only:
{{
  "classifications": [
    {{
      "table_name": "full_table_name",
      "entity_type": "Customer",
      "confidence": 0.9,
      "business_role": "Core",
      "reasoning": "Table contains customer names, emails, and contact information"
    }}
  ]
}}
"""
    return prompt

def create_domain_analysis_prompt(tables: List['TableInfo']) -> str:
    """Create prompt for business domain analysis"""
    
    # Analyze entity distribution
    entity_counts = {}
    for table in tables:
        entity_type = getattr(table, 'entity_type', 'Unknown')
        if entity_type != 'Unknown':
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Sample table information
    table_summaries = []
    for table in tables[:20]:  # Limit to first 20 tables
        entity_type = getattr(table, 'entity_type', 'Unknown')
        confidence = getattr(table, 'confidence', 0.0)
        
        table_summaries.append({
            'name': table.name,
            'entity_type': entity_type,
            'confidence': confidence,
            'row_count': table.row_count
        })
    
    prompt = f"""
Analyze this database structure and determine the business domain and industry.

ENTITY DISTRIBUTION:
{json.dumps(entity_counts, indent=2)}

SAMPLE TABLES:
{json.dumps(table_summaries, indent=2)}

Determine:
1. Primary business domain (E-Commerce, CRM/Sales, Financial Services, Healthcare, Manufacturing, etc.)
2. Industry category (Retail, Technology, Healthcare, Finance, etc.)
3. Confidence level (0.0 to 1.0)
4. Key business capabilities this system supports
5. Sample questions users might ask

Respond with JSON only:
{{
  "domain_type": "E-Commerce",
  "industry": "Retail Technology",
  "confidence": 0.85,
  "capabilities": [
    "customer_management",
    "order_processing", 
    "payment_tracking",
    "inventory_management"
  ],
  "sample_questions": [
    "How many customers do we have?",
    "What is our monthly revenue?",
    "Show top-selling products"
  ]
}}
"""
    return prompt

# ==========================================
# SQL Processing
# ==========================================

def clean_sql_query(response: str) -> Optional[str]:
    """Clean and validate SQL query from LLM response"""
    if not response:
        return None
    
    # Remove markdown formatting
    cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Remove common prefixes
    cleaned = re.sub(r'^[^S]*(?=SELECT)', '', cleaned, flags=re.IGNORECASE)
    
    # Extract SQL query
    lines = cleaned.split('\n')
    sql_lines = []
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith('SELECT') or sql_lines:
            sql_lines.append(line)
            if line.endswith(';'):
                break
    
    if sql_lines:
        sql = '\n'.join(sql_lines).rstrip(';')
        return sql if sql.strip() else None
    
    # Fallback - look for SELECT statement anywhere
    select_match = re.search(r'(SELECT.*?)(?:;|\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return None

def validate_sql_syntax(sql_query: str) -> Tuple[bool, Optional[str]]:
    """Basic SQL syntax validation"""
    if not sql_query:
        return False, "Empty query"
    
    sql_upper = sql_query.upper()
    
    # Check for required SELECT
    if not sql_upper.startswith('SELECT'):
        return False, "Query must start with SELECT"
    
    # Check for dangerous operations
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Dangerous operation detected: {keyword}"
    
    # Basic syntax checks
    if sql_upper.count('(') != sql_upper.count(')'):
        return False, "Unmatched parentheses"
    
    return True, None

# ==========================================
# Data Processing
# ==========================================

def safe_database_value(value) -> Any:
    """Convert database value to safe, serializable format"""
    if value is None:
        return None
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif hasattr(value, '__str__'):
        str_value = str(value)
        return str_value[:500] if len(str_value) > 500 else str_value  # Truncate very long values
    else:
        return str(value)[:500]

def should_exclude_table(table_name: str, schema_name: str = None) -> bool:
    """Determine if table should be excluded from analysis"""
    
    name_lower = table_name.lower()
    full_name_lower = f"{schema_name}.{table_name}".lower() if schema_name else name_lower
    
    # System and technical exclusions
    exclusion_patterns = [
        # SQL Server system objects
        'msreplication', 'trace_xe', 'syscommittab', 'sysdiagrams',
        'dtproperties', '__msnpeer', '__msdbm', 'mspeer_',
        
        # Common technical/broken objects
        'corrupted', 'broken', 'damaged', 'invalid', 'temp_',
        'backup_', 'archive_', 'deleted_', 'old_',
        
        # Replication and sync objects
        'conflict_', 'reseed_', 'msmerge_', 'snapshot_',
        
        # Very long problematic names
        'businesspointidentificationwiththirdpartydata',
        'timingview'
    ]
    
    for pattern in exclusion_patterns:
        if pattern in name_lower:
            return True
    
    # Schema-based exclusions
    if schema_name:
        excluded_schemas = ['sys', 'information_schema', 'msdb', 'master', 'tempdb']
        if schema_name.lower() in excluded_schemas:
            return True
    
    return False

# ==========================================
# Cache Management
# ==========================================

def save_cache(filepath: Path, data: Dict, description: str = "data") -> bool:
    """Save data to cache file with error handling"""
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        cache_data = {
            'data': data,
            'created': datetime.now().isoformat(),
            'version': '2.0-enhanced'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False, default=str)
        
        file_size = filepath.stat().st_size / 1024  # Size in KB
        print(f"   💾 Saved {description} to cache ({file_size:.1f}KB)")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Failed to save {description} to cache: {e}")
        return False

def load_cache(filepath: Path, max_age_hours: int = 24) -> Optional[Dict]:
    """Load data from cache file with age validation"""
    if not filepath.exists():
        return None
    
    try:
        # Check file age
        file_age = time.time() - filepath.stat().st_mtime
        age_hours = file_age / 3600
        
        if age_hours > max_age_hours:
            print(f"   ⏰ Cache expired ({age_hours:.1f}h old, max {max_age_hours}h)")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Return the actual data (handle both old and new format)
        if 'data' in cache_data:
            print(f"   📁 Loaded from cache ({age_hours:.1f}h old)")
            return cache_data['data']
        else:
            # Old format - return as is
            print(f"   📁 Loaded from cache (legacy format)")
            return cache_data
            
    except Exception as e:
        print(f"   ⚠️ Failed to load cache: {e}")
        return None

# ==========================================
# Business Intelligence
# ==========================================

def generate_sample_questions(entity_counts: Dict[str, int], domain_type: str) -> List[str]:
    """Generate relevant sample questions based on available entities"""
    
    questions = []
    
    # Customer questions
    if entity_counts.get('Customer', 0) > 0:
        questions.extend([
            "How many customers do we have?",
            "Show customer information",
            "List all customers with contact details",
            "Who are our top customers?"
        ])
    
    # Payment questions
    if entity_counts.get('Payment', 0) > 0:
        questions.extend([
            "What is our total revenue for 2025?",
            "Show payment information",
            "Calculate monthly revenue",
            "What are our payment trends?"
        ])
    
    # Combined customer-payment questions
    if entity_counts.get('Customer', 0) > 0 and entity_counts.get('Payment', 0) > 0:
        questions.extend([
            "How many customers have made payments?",
            "Show paid customers for 2025",
            "Count total paid customers",
            "Which customers haven't paid recently?"
        ])
    
    # Order questions
    if entity_counts.get('Order', 0) > 0:
        questions.extend([
            "How many orders do we have?",
            "Show recent orders",
            "What is our average order value?",
            "List orders by customer"
        ])
    
    # Product questions
    if entity_counts.get('Product', 0) > 0:
        questions.extend([
            "How many products do we sell?",
            "Show product catalog",
            "What are our best-selling products?",
            "List products by category"
        ])
    
    # Domain-specific questions
    if domain_type == 'E-Commerce':
        questions.extend([
            "Show monthly sales growth",
            "Customer lifetime value analysis",
            "Product performance by region"
        ])
    elif domain_type == 'CRM/Sales':
        questions.extend([
            "Lead conversion rates",
            "Sales pipeline analysis",
            "Customer acquisition trends"
        ])
    elif domain_type == 'Financial Services':
        questions.extend([
            "Transaction volume analysis",
            "Account balance summaries",
            "Risk assessment metrics"
        ])
    
    # Generic business questions
    questions.extend([
        "Show me a system overview",
        "What data is available?",
        "List all business entities"
    ])
    
    # Remove duplicates and limit
    unique_questions = list(dict.fromkeys(questions))  # Preserves order
    return unique_questions[:20]  # Limit to 20 questions

def determine_business_capabilities(entity_counts: Dict[str, int], relationships: List) -> Dict[str, bool]:
    """Determine system capabilities based on available entities and relationships"""
    
    capabilities = {}
    
    # Basic entity capabilities
    capabilities['customer_analysis'] = entity_counts.get('Customer', 0) > 0
    capabilities['payment_analysis'] = entity_counts.get('Payment', 0) > 0
    capabilities['order_analysis'] = entity_counts.get('Order', 0) > 0
    capabilities['product_analysis'] = entity_counts.get('Product', 0) > 0
    capabilities['user_management'] = entity_counts.get('User', 0) > 0
    capabilities['company_analysis'] = entity_counts.get('Company', 0) > 0
    
    # Advanced capabilities based on combinations
    capabilities['financial_reporting'] = (
        entity_counts.get('Payment', 0) > 0 or 
        entity_counts.get('Financial', 0) > 0
    )
    
    capabilities['customer_payment_analysis'] = (
        entity_counts.get('Customer', 0) > 0 and 
        entity_counts.get('Payment', 0) > 0
    )
    
    capabilities['sales_analysis'] = (
        entity_counts.get('Order', 0) > 0 and 
        entity_counts.get('Customer', 0) > 0
    )
    
    capabilities['inventory_management'] = (
        entity_counts.get('Product', 0) > 0 and 
        entity_counts.get('Order', 0) > 0
    )
    
    # Relationship-based capabilities
    capabilities['cross_entity_analysis'] = len([c for c in entity_counts.values() if c > 0]) >= 2
    capabilities['relationship_queries'] = len(relationships) > 0
    capabilities['complex_joins'] = len(relationships) >= 3
    
    return capabilities

# For backward compatibility
clean_sql_response = clean_sql_query