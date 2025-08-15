#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Utility Functions - DRY, SOLID, YAGNI
Single purpose functions with clear responsibilities
"""

import json
import re
from typing import Dict, Any, Union
from datetime import datetime

def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response with robust error handling
    Single responsibility: JSON parsing only
    """
    if not response or not response.strip():
        return {}
    
    try:
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
        
        # Remove text before first { or [
        cleaned = re.sub(r'^[^{[]*', '', cleaned)
        # Remove text after last } or ]
        cleaned = re.sub(r'[^}\]]*$', '', cleaned)
        
        return json.loads(cleaned)
        
    except (json.JSONDecodeError, AttributeError):
        return {}

def clean_sql_query(response: str) -> str:
    """
    Extract and clean SQL query from LLM response
    Single responsibility: SQL cleaning only
    """
    if not response or not response.strip():
        return ""
    
    # Remove markdown
    cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Extract SQL statement by lines
    lines = cleaned.strip().split('\n')
    sql_lines = []
    found_select = False
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith(('SELECT', 'WITH')) or found_select:
            found_select = True
            sql_lines.append(line)
            if line.endswith(';'):
                break
    
    if sql_lines:
        return '\n'.join(sql_lines).rstrip(';').strip()
    
    # Fallback: regex extraction
    select_match = re.search(r'(SELECT.*?)(?:;|\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return cleaned.strip()

def safe_database_value(value: Any) -> Union[str, int, float, bool, None]:
    """
    Convert database value to safe JSON-serializable format
    Single responsibility: Data type conversion
    """
    if value is None:
        return None
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, bytes):
        try:
            return value.decode('utf-8')[:200]
        except UnicodeDecodeError:
            return "[Binary Data]"
    else:
        return str(value)[:200]  # Limit string length for safety

def should_exclude_table(table_name: str, schema_name: str = None) -> bool:
    """
    Check if table should be excluded from discovery
    Single responsibility: Table filtering logic
    """
    if not table_name:
        return True
    
    name_lower = table_name.lower()
    
    # System and temporary table patterns
    exclusion_patterns = [
        'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
        'corrupted', 'broken', 'temp_', 'backup_', 'deleted_', 'test_'
    ]
    
    for pattern in exclusion_patterns:
        if pattern in name_lower:
            return True
    
    # Exclude system schemas
    if schema_name and schema_name.lower() in ['sys', 'information_schema']:
        return True
    
    return False

def format_number(value: Union[int, float]) -> str:
    """
    Format numbers with thousands separators
    Single responsibility: Number formatting
    """
    if isinstance(value, (int, float)) and value >= 1000:
        return f"{value:,}"
    return str(value)

def truncate_text(text: str, max_length: int = 40) -> str:
    """
    Truncate text with ellipsis if too long
    Single responsibility: Text truncation
    """
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def normalize_table_name(schema: str, table: str) -> str:
    """
    Create normalized table name with proper formatting
    Single responsibility: Table name formatting
    """
    return f"[{schema}].[{table}]"

def extract_keywords(text: str, min_length: int = 3) -> list:
    """
    Extract meaningful keywords from text
    Single responsibility: Keyword extraction
    """
    if not text:
        return []
    
    # Split and filter words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out short words and common stop words
    stop_words = {'the', 'and', 'are', 'for', 'with', 'this', 'that', 'from', 'what', 'how'}
    keywords = [word for word in words 
                if len(word) >= min_length and word not in stop_words]
    
    return list(set(keywords))  # Remove duplicates

def validate_sql_safety(sql: str) -> bool:
    """
    Basic validation for SQL safety (read-only operations)
    Single responsibility: SQL safety validation
    """
    if not sql:
        return False
    
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT or WITH
    if not sql_upper.startswith(('SELECT', 'WITH')):
        return False
    
    # Check for dangerous keywords
    dangerous_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
        'TRUNCATE', 'MERGE', 'EXEC', 'EXECUTE', 'SP_', 'XP_'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False
    
    return True

def calculate_confidence_score(factors: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """
    Calculate weighted confidence score from multiple factors
    Single responsibility: Confidence calculation
    """
    if not factors:
        return 0.0
    
    # Default weights if not provided
    if weights is None:
        weights = {factor: 1.0 for factor in factors.keys()}
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for factor, value in factors.items():
        weight = weights.get(factor, 1.0)
        weighted_sum += value * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    score = weighted_sum / total_weight
    return max(0.0, min(1.0, score))  # Clamp between 0 and 1