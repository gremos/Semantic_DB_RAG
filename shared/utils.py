#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions - Simple, Clean, Maintainable
Following DRY, SOLID, YAGNI principles
Essential utilities only - no unnecessary complexity
"""

import json
import re
from typing import Dict, Any, Union, List
from datetime import datetime

def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with robust cleaning"""
    if not response or not response.strip():
        return {}
    
    try:
        cleaned = _clean_json_response(response)
        return json.loads(cleaned) if cleaned else {}
    except (json.JSONDecodeError, AttributeError):
        return {}

def _clean_json_response(response: str) -> str:
    """Clean response text for JSON parsing"""
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
    
    # Extract JSON content
    cleaned = re.sub(r'^[^{[]*', '', cleaned)  # Remove text before first { or [
    cleaned = re.sub(r'[^}\]]*$', '', cleaned)  # Remove text after last } or ]
    
    return cleaned.strip()

def clean_sql_query(response: str) -> str:
    """Extract and clean SQL query from LLM response"""
    if not response or not response.strip():
        return ""
    
    # Remove markdown formatting
    cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Extract SQL by lines
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

def validate_sql_safety(sql: str) -> bool:
    """Validate SQL for read-only safety"""
    if not sql:
        return False
    
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT or WITH
    if not sql_upper.startswith(('SELECT', 'WITH')):
        return False
    
    # Check for dangerous operations
    dangerous_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
        'TRUNCATE', 'MERGE', 'EXEC', 'EXECUTE', 'SP_', 'XP_'
    ]
    
    return not any(keyword in sql_upper for keyword in dangerous_keywords)

def safe_database_value(value: Any) -> Union[str, int, float, bool, None]:
    """Convert database value to safe JSON-serializable format"""
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
        return str(value)[:200]  # Limit length for safety

def format_number(value: Union[int, float]) -> str:
    """Format numbers with thousands separators"""
    if isinstance(value, (int, float)) and abs(value) >= 1000:
        return f"{value:,}"
    return str(value)

def truncate_text(text: str, max_length: int = 40) -> str:
    """Truncate text with ellipsis if needed"""
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful keywords from text"""
    if not text:
        return []
    
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter words
    stop_words = {'the', 'and', 'are', 'for', 'with', 'this', 'that', 'from', 'what', 'how'}
    keywords = [word for word in words 
               if len(word) >= min_length and word not in stop_words]
    
    return list(set(keywords))  # Remove duplicates

def normalize_table_name(schema: str, table: str) -> str:
    """Create normalized table name with proper formatting"""
    return f"[{schema}].[{table}]"

def should_exclude_table(table_name: str, schema_name: str = None, 
                        exclusion_patterns: List[str] = None) -> bool:
    """Check if table should be excluded from discovery"""
    if not table_name:
        return True
    
    name_lower = table_name.lower()
    
    # Use provided patterns or defaults
    if exclusion_patterns is None:
        exclusion_patterns = [
            'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
            'corrupted', 'broken', 'temp_', 'backup_', 'test_'
        ]
    
    # Check patterns
    for pattern in exclusion_patterns:
        if pattern.lower() in name_lower:
            return True
    
    # Check system schemas
    if schema_name and schema_name.lower() in ['sys', 'information_schema']:
        return True
    
    return False

def calculate_confidence_score(factors: Dict[str, float], 
                             weights: Dict[str, float] = None) -> float:
    """Calculate weighted confidence score from multiple factors"""
    if not factors:
        return 0.0
    
    # Default equal weights
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

def classify_confidence(score: float) -> str:
    """Classify confidence score into categories"""
    if score >= 0.9:
        return "Very High"
    elif score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Medium"
    elif score >= 0.3:
        return "Low"
    else:
        return "Very Low"

def safe_execute(func, *args, default=None, **kwargs):
    """Safely execute function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ Error in {func.__name__}: {e}")
        return default

def log_error(error: Exception, context: str = ""):
    """Log error with context"""
    error_msg = f"Error: {type(error).__name__}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    print(f"❌ {error_msg}")

# Performance monitoring decorator
def timer(func):
    """Simple timing decorator for performance monitoring"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} took {end - start:.3f}s")
        return result
    
    return wrapper