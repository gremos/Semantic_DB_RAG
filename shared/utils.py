#!/usr/bin/env python3  
# -*- coding: utf-8 -*-
"""
Simplified utility functions - Focus on core functionality
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from LLM response"""
    try:
        # Clean the response
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if '```json' in cleaned:
            cleaned = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
            if cleaned:
                cleaned = cleaned.group(1)
        elif '```' in cleaned:
            cleaned = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
            if cleaned:
                cleaned = cleaned.group(1)
        
        # Try to parse JSON
        return json.loads(cleaned)
        
    except Exception as e:
        print(f"   ⚠️ Failed to parse JSON: {e}")
        return None

def classify_entity_type_simple(table_name: str, columns: List[Dict], sample_data: List[Dict]) -> tuple:
    """
    Simple entity classification based on name patterns and column analysis
    Returns: (entity_type, confidence)
    """
    name_lower = table_name.lower()
    
    # High confidence patterns
    if any(word in name_lower for word in ['customer', 'client', 'account']):
        return 'Customer', 0.9
    
    if any(word in name_lower for word in ['order', 'sale', 'purchase']):
        return 'Order', 0.9
        
    if any(word in name_lower for word in ['product', 'item', 'inventory']):
        return 'Product', 0.9
        
    if any(word in name_lower for word in ['payment', 'invoice', 'billing', 'transaction']):
        return 'Payment', 0.9
        
    if any(word in name_lower for word in ['user', 'person', 'employee', 'contact']):
        return 'Person', 0.8
        
    if any(word in name_lower for word in ['company', 'business', 'organization', 'vendor']):
        return 'Organization', 0.8
        
    # Column-based analysis
    column_names = [col.get('name', '').lower() for col in columns]
    
    # Look for ID patterns that suggest entity type
    if any('customer' in col for col in column_names):
        return 'Customer', 0.7
    if any('order' in col for col in column_names):
        return 'Order', 0.7
    if any('product' in col for col in column_names):
        return 'Product', 0.7
    if any('payment' in col for col in column_names):
        return 'Payment', 0.7
        
    # Check for common entity indicators
    if any(col in column_names for col in ['email', 'phone', 'address']):
        return 'Contact', 0.6
        
    if any(col in column_names for col in ['amount', 'total', 'price', 'cost']):
        return 'Financial', 0.6
        
    return 'Unknown', 0.0

def find_table_relationships_simple(tables: List['TableInfo']) -> List[tuple]:
    """
    Simple relationship discovery based on column name matching
    Returns: List of (from_table, to_table, relationship_type, confidence)
    """
    relationships = []
    
    # Create column index for faster lookups
    table_columns = {}
    for table in tables:
        table_columns[table.full_name] = [col.get('name', '').lower() for col in table.columns]
    
    # Look for foreign key patterns
    for table in tables:
        table_cols = table_columns[table.full_name]
        
        for col_name in table_cols:
            if col_name.endswith('id') and col_name != 'id':
                # Look for matching table
                entity_name = col_name[:-2]  # Remove 'id'
                
                for other_table in tables:
                    if table.full_name == other_table.full_name:
                        continue
                        
                    other_name = other_table.name.lower()
                    if entity_name in other_name or other_name in entity_name:
                        relationships.append((
                            table.full_name,
                            other_table.full_name,
                            'foreign_key',
                            0.8
                        ))
    
    return relationships

def create_llm_classification_prompt(tables_batch: List['TableInfo']) -> str:
    """Create focused LLM prompt for entity classification"""
    
    table_info = []
    for table in tables_batch:
        columns_str = ', '.join([col.get('name', '') for col in table.columns[:10]])
        sample_str = str(table.sample_data[0]) if table.sample_data else "No sample data"
        
        table_info.append({
            'name': table.name,
            'full_name': table.full_name,
            'columns': columns_str,
            'sample': sample_str,
            'row_count': table.row_count
        })
    
    prompt = f"""
Analyze these database tables and classify each one as a business entity type.

TABLES TO CLASSIFY:
{json.dumps(table_info, indent=2)}

For each table, determine:
1. Entity Type: Customer, Order, Product, Payment, User, Company, Contact, Financial, Lookup, or Unknown
2. Confidence: 0.0 to 1.0 (how sure you are)
3. Business Role: Core, Supporting, Reference, or System

Look for patterns in:
- Table names (customers, orders, products, payments)
- Column names (customer_id, order_date, product_name, amount)
- Sample data content and structure

Respond with JSON only:
{{
  "classifications": [
    {{
      "table_name": "full_table_name",
      "entity_type": "Customer",
      "confidence": 0.9,
      "business_role": "Core"
    }}
  ]
}}
"""
    return prompt

def find_related_tables_fuzzy(question: str, tables: List['TableInfo']) -> List['TableInfo']:
    """
    Find tables related to question using fuzzy matching
    Much simpler than the complex relationship-aware approach
    """
    question_lower = question.lower()
    scored_tables = []
    
    # Keywords to table type mapping
    keywords = {
        'customer': ['Customer', 'Contact', 'User', 'Person'],
        'client': ['Customer', 'Contact', 'User', 'Person'],
        'payment': ['Payment', 'Financial', 'Order'],
        'paid': ['Payment', 'Financial', 'Order'],
        'order': ['Order', 'Product', 'Customer'],
        'sale': ['Order', 'Product', 'Customer'],
        'product': ['Product', 'Order'],
        'revenue': ['Payment', 'Financial', 'Order'],
        'total': ['Payment', 'Financial', 'Order'],
        'count': ['Customer', 'Order', 'Product', 'Payment']
    }
    
    for table in tables:
        score = 0
        
        # Direct name matching
        for keyword, entity_types in keywords.items():
            if keyword in question_lower:
                if table.entity_type in entity_types:
                    score += 10
                elif keyword in table.name.lower():
                    score += 8
        
        # Confidence boost
        if table.confidence > 0.7:
            score += 5
        elif table.confidence > 0.5:
            score += 3
        
        # Row count consideration (prefer tables with data)
        if table.row_count > 1000:
            score += 3
        elif table.row_count > 100:
            score += 2
        elif table.row_count > 0:
            score += 1
        
        if score > 0:
            scored_tables.append((table, score))
    
    # Sort by score and return top matches
    scored_tables.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 5 tables
    return [table for table, score in scored_tables[:5]]

def generate_simple_sql_prompt(question: str, tables: List['TableInfo']) -> str:
    """Generate simple SQL prompt for LLM"""
    
    table_schemas = []
    for table in tables:
        columns = []
        for col in table.columns[:10]:  # Limit columns
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            columns.append(f"{col_name} ({col_type})")
        
        table_schemas.append({
            'table': table.full_name,
            'entity_type': table.entity_type,
            'columns': columns,
            'sample_row_count': table.row_count
        })
    
    prompt = f"""
Generate SQL Server T-SQL query to answer this question: "{question}"

Available tables:
{json.dumps(table_schemas, indent=2)}

Rules:
1. Use proper SQL Server syntax with square brackets for names
2. Use appropriate JOINs if multiple tables are needed
3. Add reasonable WHERE clauses to filter data
4. Use TOP 100 to limit results unless asking for totals/counts
5. Return ONLY the SQL query, no explanations

SQL Query:
"""
    return prompt

def save_cache(filepath: Path, data: Dict, description: str = "data"):
    """Save data to cache file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"   💾 Saved {description} to cache")
    except Exception as e:
        print(f"   ⚠️ Failed to save {description}: {e}")

def load_cache(filepath: Path, max_age_hours: int = 24) -> Optional[Dict]:
    """Load data from cache file"""
    if not filepath.exists():
        return None
        
    try:
        # Check age
        age_hours = (datetime.now().timestamp() - filepath.stat().st_mtime) / 3600
        if age_hours > max_age_hours:
            print(f"   ⏰ Cache expired ({age_hours:.1f}h old)")
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"   ⚠️ Failed to load cache: {e}")
        return None

def clean_sql_query(response: str) -> Optional[str]:
    """Clean SQL response from LLM"""
    # Remove markdown
    cleaned = re.sub(r'```sql\s*', '', response)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Extract SQL (look for SELECT statement)
    lines = cleaned.split('\n')
    sql_lines = []
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith('SELECT') or sql_lines:
            sql_lines.append(line)
            if line.endswith(';'):
                break
    
    if sql_lines:
        return '\n'.join(sql_lines).rstrip(';')
    
    # Fallback
    if 'SELECT' in response.upper():
        return response.strip().rstrip(';')
        
    return None

# Alias for backward compatibility
def clean_sql_response(response: str) -> Optional[str]:
    """Backward compatibility alias"""
    return clean_sql_query(response)

def safe_database_value(value) -> Any:
    """Convert database value to safe format"""
    if value is None:
        return None
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, (str, int, float, bool)):
        return value
    else:
        return str(value)[:200]  # Truncate long values

def should_exclude_table(table_name: str, schema_name: str) -> bool:
    """
    MINIMAL exclusion logic - Only exclude truly problematic objects
    """
    full_name = f"{schema_name}.{table_name}"
    name_lower = table_name.lower()
    
    # Only exclude clearly problematic objects
    critical_exclusions = [
        'msreplication', 'trace_xe', 'syscommittab', 'sysdiagrams',
        'dtproperties', '__msnpeer', '__msdbm', 'mspeer_',
        'corrupted', 'broken', 'damaged', 'invalid',
        'conflict_', 'reseed_', 'msmerge_'
    ]
    
    for exclusion in critical_exclusions:
        if exclusion in name_lower:
            return True
    
    # Don't exclude common business objects
    return False