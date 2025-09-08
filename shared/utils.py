#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions - Fixed Safety Validation and Enhanced SQL Processing
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
Fixed: Safety validation leading space issue, CTE support, normalization
"""

import json
import re
import time
from typing import Dict, Any, Union, List, Optional
from datetime import datetime
from functools import wraps

# SQLGlot for SQL safety validation (Architecture requirement)
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with enhanced cleaning"""
    if not response or not response.strip():
        return {}
    
    try:
        cleaned = _clean_json_response(response)
        if not cleaned:
            return {}
            
        parsed = json.loads(cleaned)
        
        # Enhanced validation for expected structures
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list) and len(parsed) > 0:
            # Handle array responses - take first valid dict
            for item in parsed:
                if isinstance(item, dict):
                    return item
            return {}
        else:
            return {}
            
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
        return {}

def _clean_json_response(response: str) -> str:
    """Enhanced JSON cleaning with better pattern matching"""
    cleaned = response.strip()
    
    # Remove common LLM prefixes/suffixes
    prefixes_to_remove = [
        "here's the json:", "here is the json:", "json:", "```json", "```",
        "the response is:", "result:", "analysis:", "here's the analysis:",
        "the json response:", "json response:", "response:"
    ]
    
    for prefix in prefixes_to_remove:
        pattern = prefix.replace(":", r"\s*:")
        if re.match(rf"^\s*{pattern}", cleaned, re.IGNORECASE):
            cleaned = re.sub(rf"^\s*{pattern}\s*", "", cleaned, flags=re.IGNORECASE)
            break
    
    # Enhanced markdown code block removal
    code_block_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'`([^`]*\{[^`]*\}[^`]*)`'
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            break
    
    # Extract JSON content between braces
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace:last_brace + 1]
    
    return cleaned.strip()

def clean_sql_query(response: str) -> str:
    """FIXED: Enhanced SQL query extraction and normalization"""
    if not response or not response.strip():
        return ""
    
    cleaned = response.strip()
    
    # Enhanced SQL extraction patterns
    sql_patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```tsql\s*(.*?)\s*```',
        r'```\s*(SELECT\s+.*?)\s*```',
        r'```\s*(WITH\s+.*?)\s*```',
        r'`([^`]*(?:SELECT|WITH)[^`]*)`'
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            break
    
    # Extract SQL statements
    lines = cleaned.split('\n')
    sql_lines = []
    in_sql = False
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
    
    for line in lines:
        line = line.strip()
        
        # Start of SQL statement
        if any(line.upper().startswith(keyword) for keyword in sql_keywords):
            in_sql = True
            sql_lines.append(line)
        elif in_sql:
            # Continue SQL statement
            if line.endswith(';'):
                sql_lines.append(line.rstrip(';'))
                break
            elif line and not line.startswith(('--', '//', '#', '/*')):
                sql_lines.append(line)
            elif not line and sql_lines:
                # Empty line might end SQL
                current_sql = '\n'.join(sql_lines).upper()
                if 'FROM' in current_sql and ('SELECT' in current_sql or 'WITH' in current_sql):
                    break
    
    if sql_lines:
        result = '\n'.join(sql_lines).strip()
        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result)
        return result
    
    # Fallback regex patterns
    fallback_patterns = [
        r'(SELECT\s+(?:TOP\s*\(\s*\d+\s*\)\s+)?.*?FROM\s+.*?)(?:;|\n\s*\n|$)',
        r'(WITH\s+.*?SELECT\s+.*?FROM\s+.*?)(?:;|\n\s*\n|$)',
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            return re.sub(r'\s+', ' ', result)
    
    return cleaned

def validate_sql_safety(sql: str) -> bool:
    """FIXED: SQL safety validation with corrected leading space issue and CTE support"""
    if not sql or len(sql.strip()) < 5:
        return False
    
    # FIXED: Don't prepend leading space that breaks startswith
    sql_stripped = sql.strip()
    sql_normalized = sql_stripped.upper()
    
    # FIXED: Check without leading space
    if not sql_normalized.startswith(('SELECT', 'WITH')):
        return False
    
    # Dangerous operations
    dangerous_patterns = [
        r'\b(?:INSERT|UPDATE|DELETE|MERGE|REPLACE)\b',
        r'\b(?:DROP|CREATE|ALTER|TRUNCATE)\b',
        r'\b(?:EXEC|EXECUTE|SP_|XP_|DBCC)\b',
        r'\b(?:GRANT|REVOKE|DENY)\b',
        r'\b(?:BULK|OPENROWSET|OPENDATASOURCE)\b',
        r'\bXP_CMDSHELL\b',
        r'\b(?:BACKUP|RESTORE)\b'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_normalized, re.IGNORECASE):
            return False
    
    # Check for multiple statements
    sql_clean = sql.strip().rstrip(';')
    if ';' in sql_clean:
        return False
    
    # FIXED: sqlglot validation with CTE support
    if HAS_SQLGLOT:
        try:
            parsed = sqlglot.parse_one(sql, dialect="tsql")
            if not parsed:
                return False
            
            # FIXED: Accept both SELECT and WITH (CTE)
            if isinstance(parsed, sqlglot.expressions.Select):
                return True
            elif isinstance(parsed, sqlglot.expressions.With):
                # CTE - check if it contains a SELECT
                for node in parsed.walk():
                    if isinstance(node, sqlglot.expressions.Select):
                        return True
                return False
            else:
                return False
                
        except Exception:
            return False
    
    return True

def safe_database_value(value: Any) -> Union[str, int, float, bool, None]:
    """Enhanced database value conversion with better type handling"""
    if value is None:
        return None
    elif isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        # Enhanced handling of special numeric values
        if isinstance(value, float):
            if value != value:  # NaN check
                return None
            if value == float('inf') or value == float('-inf'):
                return None
            # Handle very large/small numbers
            if abs(value) > 1e15:
                return str(value)
        return value
    elif isinstance(value, str):
        # Enhanced string handling with length limits
        if len(value) > 1000:
            return value[:1000] + "..."
        return value
    elif isinstance(value, datetime):
        try:
            return value.isoformat()
        except:
            return str(value)
    elif isinstance(value, bytes):
        try:
            decoded = value.decode('utf-8')
            if len(decoded) > 200:
                return decoded[:200] + "..."
            return decoded
        except UnicodeDecodeError:
            try:
                decoded = value.decode('latin-1', errors='replace')
                return f"[Binary: {decoded[:100]}...]" if len(decoded) > 100 else f"[Binary: {decoded}]"
            except:
                return "[Binary Data]"
    else:
        try:
            str_value = str(value)
            return str_value[:500] if len(str_value) > 500 else str_value
        except:
            return "[Unprintable Data]"

def should_exclude_table(table_name: str, schema_name: str = None, 
                        exclusion_patterns: List[str] = None) -> bool:
    """Enhanced table exclusion with SQL Server specific patterns"""
    if not table_name:
        return True
    
    name_lower = table_name.lower()
    
    # Enhanced SQL Server exclusion patterns
    if exclusion_patterns is None:
        exclusion_patterns = [
            # SQL Server system tables
            'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
            'mspeer_', 'msdistribution_', 'msmerge_', 'mssubscription_',
            # Backup/temp patterns
            'backup', 'temp_', 'tmp_', 'staging_', 'etl_', 'import_', 'export_',
            # Test patterns  
            'test_', 'testing_', 'demo_', 'sample_', 'example_',
            # Archive patterns
            'old_', 'archive_', 'historical_', 'deleted_', 'obsolete_',
            # System/maintenance
            'log_', 'audit_', 'trace_', 'sync_', 'queue_',
            # SQL Server specific
            'aspnet_', 'elmah_', '__mig', '__ef'
        ]
    
    # Check exclusion patterns
    for pattern in exclusion_patterns:
        if pattern.lower() in name_lower:
            return True
    
    # Enhanced date suffix detection
    date_patterns = [
        r'\d{8}$',                    # YYYYMMDD
        r'\d{6}$',                    # YYYYMM  
        r'\d{4}$',                    # YYYY
        r'_\d{4}_\d{2}_\d{2}$',      # _YYYY_MM_DD
        r'_\d{4}\d{2}\d{2}$',        # _YYYYMMDD
        r'_backup$', r'_bak$',        # Backup suffixes
        r'_copy$', r'_old$', r'_new$'
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, name_lower):
            return True
    
    # System schema detection
    if schema_name:
        system_schemas = [
            'sys', 'information_schema', 'db_owner', 'db_accessadmin', 
            'db_backupoperator', 'db_datareader', 'db_datawriter', 
            'db_ddladmin', 'db_denydatareader', 'db_denydatawriter',
            'guest', 'INFORMATION_SCHEMA'
        ]
        if schema_name.lower() in [s.lower() for s in system_schemas]:
            return True
    
    return False

def normalize_table_name(schema: str, table: str) -> str:
    """Enhanced table name normalization for SQL Server"""
    clean_schema = schema.strip('[]')
    clean_table = table.strip('[]')
    
    # Validate names (basic SQL Server identifier rules)
    identifier_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    
    if not re.match(identifier_pattern, clean_schema):
        clean_schema = f'"{clean_schema}"'
    if not re.match(identifier_pattern, clean_table):
        clean_table = f'"{clean_table}"'
    
    return f"[{clean_schema}].[{clean_table}]"

def format_number(value: Union[int, float], precision: int = 2) -> str:
    """Enhanced number formatting with better scaling"""
    if not isinstance(value, (int, float)):
        return str(value)
    
    if isinstance(value, float) and (value != value or value == float('inf')):
        return "N/A"
    
    abs_value = abs(value)
    
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}T"
    elif abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs_value >= 1000:
        if isinstance(value, int):
            return f"{value:,}"
        else:
            return f"{value:,.{precision}f}"
    else:
        if isinstance(value, int) or value.is_integer():
            return str(int(value))
        else:
            formatted = f"{value:.{precision}f}"
            return formatted.rstrip('0').rstrip('.')

def truncate_text(text: str, max_length: int = 40, smart_truncate: bool = True) -> str:
    """Enhanced text truncation with Unicode support"""
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    if smart_truncate:
        # Try to truncate at word boundary
        words = text[:max_length - 3].split()
        if len(words) > 1:
            truncated = ' '.join(words[:-1])
            # Check if we saved meaningful space
            if len(truncated) >= max_length * 0.7:
                return truncated + "..."
    
    # Unicode-aware truncation
    try:
        truncated = text.encode('utf-8')[:max_length - 3].decode('utf-8', errors='ignore')
        return truncated + "..."
    except:
        return text[:max_length - 3] + "..."

def classify_confidence(score: float) -> str:
    """Enhanced confidence classification"""
    if score >= 0.95:
        return "Excellent"
    elif score >= 0.90:
        return "Very High"
    elif score >= 0.80:
        return "High"
    elif score >= 0.65:
        return "Good"
    elif score >= 0.50:
        return "Medium"
    elif score >= 0.35:
        return "Low"
    else:
        return "Very Low"

def detect_data_quality(sample_data: List[Dict[str, Any]]) -> str:
    """Enhanced data quality detection from sample data"""
    if not sample_data:
        return "unknown"
    
    test_indicators = 0
    production_indicators = 0
    
    for row in sample_data[:3]:
        for key, value in row.items():
            if key.startswith('__'):
                continue
                
            if isinstance(value, str):
                value_lower = value.lower()
                
                # Test data indicators
                if any(test_word in value_lower for test_word in 
                      ['test', 'lorem', 'ipsum', 'example', 'sample', 'dummy', 'fake']):
                    test_indicators += 1
                elif any(test_pattern in value_lower for test_pattern in 
                        ['@example.com', 'test@', '123-45-6789']):
                    test_indicators += 1
                
                # Production data indicators
                elif len(value) > 10 and not any(char.isdigit() for char in value):
                    production_indicators += 1
                elif '@' in value and '.' in value and 'test' not in value_lower:
                    production_indicators += 1
            
            elif isinstance(value, (int, float)) and value > 0:
                if isinstance(value, float) and 0.01 < value < 1000000:
                    production_indicators += 1
                elif isinstance(value, int) and 1 < value < 10000000:
                    production_indicators += 1
    
    if test_indicators > production_indicators:
        return "test"
    elif production_indicators > 0:
        return "production"
    else:
        return "unknown"

def extract_business_keywords(text: str, entity_type: str = None) -> List[str]:
    """Enhanced keyword extraction with business context"""
    if not text:
        return []
    
    text_lower = text.lower()
    
    # Business keyword patterns by entity type
    business_patterns = {
        'Customer': ['customer', 'client', 'account', 'contact', 'user', 'member'],
        'Payment': ['payment', 'transaction', 'invoice', 'billing', 'revenue', 'amount'],
        'Contract': ['contract', 'agreement', 'deal', 'terms'],
        'Order': ['order', 'purchase', 'sale', 'quote', 'request'],
        'Employee': ['employee', 'staff', 'worker', 'personnel', 'user'],
        'Product': ['product', 'item', 'catalog', 'inventory', 'sku']
    }
    
    # General business keywords
    general_keywords = [
        'name', 'title', 'description', 'amount', 'total', 'count', 'date',
        'created', 'modified', 'status', 'type', 'category', 'active', 'id'
    ]
    
    # Extract using pattern matching
    word_pattern = r'\b[a-zA-Z][a-zA-Z0-9_]*\b'
    words = re.findall(word_pattern, text_lower)
    
    # Stop words
    stop_words = {
        'the', 'and', 'are', 'for', 'with', 'this', 'that', 'from', 'what', 'how',
        'but', 'not', 'you', 'all', 'can', 'had', 'has', 'have', 'her', 'was',
        'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'old',
        'see', 'him', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put'
    }
    
    # Score and filter keywords
    keyword_scores = {}
    for word in words:
        if len(word) >= 3 and word not in stop_words and not word.isdigit():
            score = 1
            
            # Entity-specific scoring
            if entity_type and entity_type in business_patterns:
                if word in business_patterns[entity_type]:
                    score += 3
            
            # General business relevance
            if word in general_keywords:
                score += 2
            
            # Pattern-based scoring
            if word.endswith('_id') or word.endswith('id'):
                score += 2
            if any(term in word for term in ['amount', 'total', 'count', 'sum']):
                score += 2
            
            keyword_scores[word] = keyword_scores.get(word, 0) + score
    
    # Return top scored keywords
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:10]]

def log_performance_metric(operation: str, duration: float, 
                          details: Dict[str, Any] = None, 
                          success: bool = True):
    """Enhanced performance logging"""
    status_emoji = "✅" if success else "❌"
    
    log_parts = [f"{status_emoji} {operation}: {duration:.2f}s"]
    
    if details:
        detail_parts = []
        for key, value in details.items():
            if isinstance(value, (int, float)):
                if key.endswith('_count') or key.endswith('_size'):
                    detail_parts.append(f"{key}={value:,}")
                else:
                    detail_parts.append(f"{key}={value}")
            else:
                detail_parts.append(f"{key}={str(value)[:30]}")
        
        if detail_parts:
            log_parts.append(f"({', '.join(detail_parts)})")
    
    print(" ".join(log_parts))

def validate_sql_server_identifier(identifier: str) -> bool:
    """Validate SQL Server identifier according to naming rules"""
    if not identifier:
        return False
    
    reserved_words = {
        'select', 'from', 'where', 'insert', 'update', 'delete', 'create', 
        'drop', 'alter', 'table', 'index', 'view', 'procedure', 'function',
        'database', 'schema', 'user', 'login', 'role', 'grant', 'revoke'
    }
    
    if identifier.lower() in reserved_words:
        return False
    
    # Basic pattern check
    pattern = r'^[a-zA-Z_@#][a-zA-Z0-9_@#$]*$'
    return bool(re.match(pattern, identifier)) and len(identifier) <= 128

def timing_decorator(include_args: bool = False):
    """Enhanced timing decorator"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                execution_time = time.time() - start_time
                
                details = {}
                if include_args and args:
                    details['args_count'] = len(args)
                if include_args and kwargs:
                    details['kwargs_count'] = len(kwargs)
                
                log_performance_metric(func.__name__, execution_time, details, True)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                log_performance_metric(func.__name__, execution_time, {'error': str(e)[:50]}, False)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                details = {}
                if include_args and args:
                    details['args_count'] = len(args)
                if include_args and kwargs:
                    details['kwargs_count'] = len(kwargs)
                
                log_performance_metric(func.__name__, execution_time, details, True)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                log_performance_metric(func.__name__, execution_time, {'error': str(e)[:50]}, False)
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def safe_execute_with_retry(func, max_retries: int = 3, delay: float = 1.0, 
                           *args, **kwargs):
    """Enhanced safe execution with retry logic"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                print(f"   ⚠️ Attempt {attempt + 1} failed, retrying: {e}")
            else:
                print(f"   ❌ All {max_retries} attempts failed: {e}")
    
    raise last_exception