#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions - Enhanced and Smart
Simple, readable, maintainable - Essential utilities with intelligence
"""

import json
import re
import time
from typing import Dict, Any, Union, List, Optional
from datetime import datetime
from functools import wraps

def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with intelligent cleaning"""
    if not response or not response.strip():
        return {}
    
    try:
        cleaned = _clean_json_response(response)
        if not cleaned:
            return {}
            
        parsed = json.loads(cleaned)
        
        # Validate parsed data structure
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            # Sometimes LLM returns array, take first dict
            return parsed[0]
        else:
            return {}
            
    except (json.JSONDecodeError, AttributeError, KeyError):
        return {}

def _clean_json_response(response: str) -> str:
    """Intelligently clean response text for JSON parsing"""
    cleaned = response.strip()
    
    # Remove common LLM prefixes/suffixes
    prefixes_to_remove = [
        "here's the json:", "here is the json:", "json:", "```json", "```",
        "the response is:", "result:", "analysis:"
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove markdown code blocks
    code_block_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'`(.*?)`'
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            break
    
    # Extract JSON content between first { and last }
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
    
    # Extract JSON content between first [ and last ] (for arrays)
    if not cleaned.startswith('{'):
        first_bracket = cleaned.find('[')
        last_bracket = cleaned.rfind(']')
        
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            cleaned = cleaned[first_bracket:last_bracket + 1]
    
    return cleaned.strip()

def clean_sql_query(response: str) -> str:
    """Extract and clean SQL query from LLM response with smart detection"""
    if not response or not response.strip():
        return ""
    
    cleaned = response.strip()
    
    # Remove markdown formatting
    sql_patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'`([^`]*SELECT[^`]*)`'
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            break
    
    # Extract SQL statements intelligently
    lines = cleaned.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        line = line.strip()
        
        # Start of SQL
        if line.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
            in_sql = True
            sql_lines.append(line)
        elif in_sql:
            # Continue SQL until we hit a clear end
            if line.endswith(';'):
                sql_lines.append(line.rstrip(';'))
                break
            elif line and not line.startswith(('--', '//', '#')):
                sql_lines.append(line)
            elif not line:
                # Empty line might end SQL
                if sql_lines and any(keyword in ' '.join(sql_lines).upper() 
                                   for keyword in ['FROM', 'WHERE', 'SELECT']):
                    break
    
    if sql_lines:
        return '\n'.join(sql_lines).strip()
    
    # Fallback: regex extraction
    sql_patterns = [
        r'(SELECT\s+.*?)(?:;|\n\s*\n|$)',
        r'(WITH\s+.*?)(?:;|\n\s*\n|$)',
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return cleaned

def validate_sql_safety(sql: str) -> bool:
    """Enhanced SQL safety validation"""
    if not sql or len(sql.strip()) < 5:
        return False
    
    sql_upper = sql.upper().strip()
    
    # Must start with safe operations
    safe_starts = ['SELECT', 'WITH']
    if not any(sql_upper.startswith(start) for start in safe_starts):
        return False
    
    # Check for dangerous operations (comprehensive list)
    dangerous_keywords = [
        # Data modification
        'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'REPLACE',
        # Schema modification  
        'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
        # System operations
        'EXEC', 'EXECUTE', 'SP_', 'XP_', 'DBCC',
        # Security risks
        'GRANT', 'REVOKE', 'DENY',
        # Bulk operations
        'BULK', 'OPENROWSET', 'OPENDATASOURCE'
    ]
    
    # Check each dangerous keyword
    for keyword in dangerous_keywords:
        if f' {keyword} ' in f' {sql_upper} ':
            return False
        if f' {keyword}(' in f' {sql_upper} ':
            return False
    
    # Additional safety checks
    if '--' in sql and 'xp_' in sql.lower():
        return False  # Potential command injection
    
    return True

def safe_database_value(value: Any) -> Union[str, int, float, bool, None]:
    """Convert database value to safe JSON-serializable format with intelligence"""
    if value is None:
        return None
    elif isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        # Handle special float values
        if isinstance(value, float):
            if value != value:  # NaN
                return None
            if value == float('inf') or value == float('-inf'):
                return None
        return value
    elif isinstance(value, str):
        return value
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, bytes):
        try:
            # Try UTF-8 decode first
            decoded = value.decode('utf-8')
            return decoded[:500] if len(decoded) > 500 else decoded
        except UnicodeDecodeError:
            try:
                # Try latin-1 as fallback
                decoded = value.decode('latin-1', errors='replace')
                return f"[Binary: {decoded[:50]}...]" if len(decoded) > 50 else f"[Binary: {decoded}]"
            except:
                return "[Binary Data]"
    else:
        # Convert other types to string with length limit
        try:
            str_value = str(value)
            return str_value[:500] if len(str_value) > 500 else str_value
        except:
            return "[Unprintable Data]"

def format_number(value: Union[int, float], precision: int = 2) -> str:
    """Smart number formatting with locale awareness"""
    if not isinstance(value, (int, float)):
        return str(value)
    
    if isinstance(value, float) and (value != value or value == float('inf')):
        return "N/A"
    
    # Handle very large numbers
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1000:
        if isinstance(value, int):
            return f"{value:,}"
        else:
            return f"{value:,.{precision}f}"
    else:
        if isinstance(value, int):
            return str(value)
        else:
            return f"{value:.{precision}f}".rstrip('0').rstrip('.')

def truncate_text(text: str, max_length: int = 40, smart_truncate: bool = True) -> str:
    """Smart text truncation with word boundary awareness"""
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    if smart_truncate:
        # Try to truncate at word boundary
        words = text[:max_length - 3].split()
        if len(words) > 1:
            return ' '.join(words[:-1]) + "..."
    
    return text[:max_length - 3] + "..."

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """Extract meaningful keywords with business intelligence"""
    if not text:
        return []
    
    # Normalize text
    text_lower = text.lower()
    
    # Extract words using improved pattern
    word_pattern = r'\b[a-zA-Z][a-zA-Z0-9_]*\b'
    words = re.findall(word_pattern, text_lower)
    
    # Enhanced stop words for business context
    stop_words = {
        'the', 'and', 'are', 'for', 'with', 'this', 'that', 'from', 'what', 'how',
        'but', 'not', 'you', 'all', 'can', 'had', 'has', 'have', 'her', 'was',
        'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'old',
        'see', 'him', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put',
        'say', 'she', 'too', 'may'
    }
    
    # Business-relevant words get priority
    business_priority_words = {
        'customer', 'payment', 'order', 'product', 'revenue', 'sales', 'amount',
        'total', 'count', 'sum', 'average', 'name', 'date', 'time', 'id'
    }
    
    # Filter and score words
    keyword_scores = {}
    for word in words:
        if (len(word) >= min_length and 
            word not in stop_words and 
            not word.isdigit()):
            
            # Score based on relevance
            score = 1
            if word in business_priority_words:
                score += 2
            if word.endswith('_id') or word.endswith('id'):
                score += 1
            if any(term in word for term in ['amount', 'total', 'count', 'sum']):
                score += 1
            
            keyword_scores[word] = keyword_scores.get(word, 0) + score
    
    # Return top keywords sorted by score
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:max_keywords]]

def normalize_table_name(schema: str, table: str) -> str:
    """Create properly formatted table name"""
    # Handle already bracketed names
    if schema.startswith('[') and schema.endswith(']'):
        clean_schema = schema[1:-1]
    else:
        clean_schema = schema
    
    if table.startswith('[') and table.endswith(']'):
        clean_table = table[1:-1]
    else:
        clean_table = table
    
    return f"[{clean_schema}].[{clean_table}]"

def should_exclude_table(table_name: str, schema_name: str = None, 
                        exclusion_patterns: List[str] = None) -> bool:
    """Smart table exclusion with pattern matching"""
    if not table_name:
        return True
    
    name_lower = table_name.lower()
    
    # Default exclusion patterns (comprehensive)
    if exclusion_patterns is None:
        exclusion_patterns = [
            # System tables
            'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
            # Backup/temp patterns
            'backup', 'temp_', 'tmp_', 'staging_', 'etl_', 'import_', 'export_',
            # Test patterns
            'test_', 'testing_', 'demo_', 'sample_',
            # Archive patterns
            'old_', 'archive_', 'historical_', 'deleted_',
            # System/maintenance
            'log_', 'audit_', 'trace_', 'sync_'
        ]
    
    # Check patterns
    for pattern in exclusion_patterns:
        if pattern.lower() in name_lower:
            return True
    
    # Check for date suffixes (likely archive tables)
    date_patterns = [
        r'\d{8}$',      # YYYYMMDD
        r'\d{6}$',      # YYYYMM
        r'\d{4}$',      # YYYY
        r'_\d{4}_\d{2}_\d{2}$',  # _YYYY_MM_DD
        r'_backup$',
        r'_copy$'
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, name_lower):
            return True
    
    # Check system schemas
    if schema_name:
        system_schemas = ['sys', 'information_schema', 'db_owner', 'db_accessadmin']
        if schema_name.lower() in system_schemas:
            return True
    
    return False

def calculate_confidence_score(factors: Dict[str, float], 
                             weights: Dict[str, float] = None,
                             boost_factors: List[str] = None) -> float:
    """Calculate intelligent weighted confidence score"""
    if not factors:
        return 0.0
    
    # Smart default weights based on importance
    if weights is None:
        weights = {}
        for factor in factors.keys():
            if 'entity' in factor or 'type' in factor:
                weights[factor] = 2.0  # Entity matching is most important
            elif 'purpose' in factor or 'business' in factor:
                weights[factor] = 1.5  # Business purpose is important
            elif 'quality' in factor:
                weights[factor] = 1.2  # Quality matters
            else:
                weights[factor] = 1.0  # Default weight
    
    # Apply boost factors for special cases
    if boost_factors:
        for factor in boost_factors:
            if factor in factors:
                factors[factor] *= 1.2  # 20% boost
    
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
    """Classify confidence score with business context"""
    if score >= 0.95:
        return "Excellent"
    elif score >= 0.85:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.50:
        return "Medium"
    elif score >= 0.30:
        return "Low"
    else:
        return "Very Low"

def detect_duplicate_similarity(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """Detect if two table names are likely duplicates"""
    if not name1 or not name2:
        return False
    
    name1_clean = name1.lower().strip()
    name2_clean = name2.lower().strip()
    
    # Exact match
    if name1_clean == name2_clean:
        return True
    
    # One contains the other
    if name1_clean in name2_clean or name2_clean in name1_clean:
        return True
    
    # Remove common suffixes and compare
    suffixes = [
        r'\d{8}$', r'\d{6}$', r'\d{4}$',  # Date patterns
        r'_copy$', r'_backup$', r'_old$', r'_new$',
        r'_temp$', r'_tmp$', r'_archive$'
    ]
    
    clean1 = name1_clean
    clean2 = name2_clean
    
    for suffix in suffixes:
        clean1 = re.sub(suffix, '', clean1)
        clean2 = re.sub(suffix, '', clean2)
    
    if clean1 == clean2:
        return True
    
    # Calculate similarity ratio
    similarity = _calculate_string_similarity(clean1, clean2)
    return similarity >= threshold

def _calculate_string_similarity(str1: str, str2: str) -> float:
    """Calculate string similarity using simple algorithm"""
    if not str1 or not str2:
        return 0.0
    
    if str1 == str2:
        return 1.0
    
    # Simple character-based similarity
    longer = str1 if len(str1) > len(str2) else str2
    shorter = str2 if len(str1) > len(str2) else str1
    
    if len(longer) == 0:
        return 1.0
    
    matches = sum(1 for i, char in enumerate(shorter) 
                  if i < len(longer) and char == longer[i])
    
    return matches / len(longer)

def safe_execute(func, *args, default=None, log_errors: bool = True, **kwargs):
    """Safely execute function with comprehensive error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            print(f"⚠️ Error in {func.__name__}: {e}")
        return default

def timing_decorator(func):
    """Enhanced timing decorator with memory tracking"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > 1.0:
                print(f"⏱️ {func.__name__} took {execution_time:.2f}s")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper

def batch_processor(items: List[Any], batch_size: int = 5, delay: float = 0.1):
    """Process items in batches with intelligent delays"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
        
        # Add delay between batches (except for last batch)
        if i + batch_size < len(items) and delay > 0:
            time.sleep(delay)

def format_table_summary(table_info) -> str:
    """Format table information for display"""
    if not table_info:
        return "No table information"
    
    name = getattr(table_info, 'name', 'Unknown')
    entity_type = getattr(table_info, 'entity_type', 'Unknown')
    row_count = getattr(table_info, 'row_count', 0)
    
    parts = [f"{name} ({entity_type})"]
    
    if row_count > 0:
        parts.append(f"{row_count:,} rows")
    
    quality = getattr(table_info, 'table_quality', 'unknown')
    if quality != 'production':
        parts.append(f"[{quality}]")
    
    return " - ".join(parts)

def log_performance(operation: str, duration: float, details: Dict[str, Any] = None):
    """Log performance metrics in a structured way"""
    log_parts = [f"⏱️ {operation}: {duration:.2f}s"]
    
    if details:
        detail_parts = []
        for key, value in details.items():
            if isinstance(value, (int, float)):
                detail_parts.append(f"{key}={value}")
            else:
                detail_parts.append(f"{key}={str(value)[:20]}")
        
        if detail_parts:
            log_parts.append(f"({', '.join(detail_parts)})")
    
    print(" ".join(log_parts))