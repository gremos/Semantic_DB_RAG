#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Utility Functions
"""

import json
import re
from typing import Dict, Any
from datetime import datetime

def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response"""
    try:
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
        
        # Remove extra text
        cleaned = re.sub(r'^[^{[]*', '', cleaned)
        cleaned = re.sub(r'[^}\]]*$', '', cleaned)
        
        return json.loads(cleaned)
        
    except Exception:
        return {}

def clean_sql_query(response: str) -> str:
    """Clean SQL query from LLM response"""
    # Remove markdown
    cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Extract SELECT statement
    lines = cleaned.strip().split('\n')
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
    select_match = re.search(r'(SELECT.*?)(?:;|\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return cleaned.strip()

def safe_database_value(value) -> Any:
    """Convert database value to safe format"""
    if value is None:
        return None
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, (str, int, float, bool)):
        return value
    else:
        return str(value)[:200]

def should_exclude_table(table_name: str, schema_name: str = None) -> bool:
    """Check if table should be excluded"""
    name_lower = table_name.lower()
    
    exclusions = [
        'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
        'corrupted', 'broken', 'temp_', 'backup_', 'deleted_'
    ]
    
    for pattern in exclusions:
        if pattern in name_lower:
            return True
    
    if schema_name and schema_name.lower() in ['sys', 'information_schema']:
        return True
    
    return False