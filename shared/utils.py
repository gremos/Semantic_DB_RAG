#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED Shared utility functions for 500+ object analysis
Less aggressive filtering to maximize business data discovery
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from LLM response with improved parsing"""
    try:
        # First try direct parsing
        parsed = json.loads(response.strip())
        return decode_unicode_in_dict(parsed)
    except:
        # Try extracting from code blocks and patterns
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'\[.*?\]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    clean_match = match if 'json' not in pattern else match
                    parsed = json.loads(clean_match)
                    return decode_unicode_in_dict(parsed)
                except:
                    continue
        return None

def decode_unicode_in_dict(obj):
    """Recursively decode Unicode escape sequences in dictionary/list structures"""
    if isinstance(obj, dict):
        return {key: decode_unicode_in_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [decode_unicode_in_dict(item) for item in obj]
    elif isinstance(obj, str):
        try:
            # Handle Unicode properly - simplified approach
            return obj
        except UnicodeError:
            return obj
    else:
        return obj

def safe_database_value(value) -> Any:
    """Convert database value to JSON-safe format with proper UTF-8 handling"""
    if value is None:
        return None
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, bytes):
        try:
            decoded = value.decode('utf-8')
            return decoded[:500] if len(decoded) > 500 else decoded
        except UnicodeDecodeError:
            return f"<binary_{len(value)}_bytes>"
    elif isinstance(value, str):
        return value[:500] if len(value) > 500 else value
    elif isinstance(value, (int, float, bool)):
        return value
    else:
        try:
            str_value = str(value)
            return str_value[:500] if len(str_value) > 500 else str_value
        except UnicodeError:
            return str(value).encode('utf-8', errors='replace').decode('utf-8')[:500]

def save_json_cache(filepath: Path, data: Dict, description: str = "data"):
    """Save data to JSON cache file with proper UTF-8 encoding"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        print(f"   💾 {description} saved to: {filepath}")
    except Exception as e:
        print(f"⚠️ Failed to save {description}: {e}")

def load_json_cache(filepath: Path, max_age_hours: int = 24, description: str = "data") -> Optional[Dict]:
    """Load data from JSON cache file with age checking"""
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure Unicode characters are properly handled
        data = decode_unicode_in_dict(data)
        
        # Check age if created timestamp exists
        if 'created' in data:
            try:
                created = datetime.fromisoformat(data['created'])
                age_hours = (datetime.now() - created).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    print(f"⏰ {description} cache expired ({age_hours:.1f}h old)")
                    return None
            except:
                pass  # Continue with data if timestamp parsing fails
        
        return data
        
    except Exception as e:
        print(f"⚠️ Failed to load {description}: {e}")
        return None

def should_exclude_table(table_name: str, schema_name: str) -> bool:
    """
    MINIMAL exclusion logic - Only exclude truly problematic objects
    
    CHANGED FROM AGGRESSIVE TO MINIMAL FILTERING:
    - Previous version excluded many potentially useful business objects
    - New version only excludes clearly broken/system objects
    - Goal: Analyze as many of the 500+ objects as possible
    """
    full_name = f"{schema_name}.{table_name}"
    name_lower = table_name.lower()
    
    # ONLY exclude clearly problematic objects (minimal list)
    critical_exclusions = [
        # System-only objects
        'msreplication', 'trace_xe', 'syscommittab', 'sysdiagrams',
        'dtproperties', '__msnpeer', '__msdbm', 'mspeer_',
        
        # Clearly corrupted/broken objects
        'corrupted', 'broken', 'damaged', 'invalid', 'error_log',
        
        # SQL Server internal replication
        'conflict_', 'reseed_', 'msmerge_',
        
        # Only backup tables with clear date patterns (not just backup word)
        '_bck_202', '_backup_202', 'backup_202'  # Only dated backups
    ]
    
    # Check critical exclusions only
    for exclusion in critical_exclusions:
        if exclusion in name_lower:
            return True
    
    # DON'T exclude common business objects that were previously filtered
    # These should be KEPT and analyzed:
    keep_patterns = [
        'customer', 'client', 'user', 'account', 'person', 'business',
        'order', 'sales', 'product', 'item', 'service',
        'payment', 'invoice', 'transaction', 'financial',
        'address', 'contact', 'phone', 'email',
        'company', 'organization', 'vendor', 'supplier',
        'campaign', 'marketing', 'advertisement',
        'task', 'assignment', 'project', 'comment',
        'log', 'audit', 'history'  # Even these can be valuable
    ]
    
    # If it matches business patterns, definitely keep it
    for pattern in keep_patterns:
        if pattern in name_lower:
            return False  # Force keep business objects
    
    # Schema-based filtering (only exclude clearly system schemas)
    system_schemas = ['sys', 'information_schema', 'guest', 'INFORMATION_SCHEMA']
    if schema_name in system_schemas:
        return True
    
    # DEFAULT: Keep the object (analyze it)
    # This is a major change from the previous aggressive filtering
    return False

def should_exclude_table_original_aggressive(table_name: str, schema_name: str) -> bool:
    """
    ORIGINAL AGGRESSIVE exclusion logic - REPLACED by minimal version above
    This version excluded too many potentially useful objects
    Kept here for reference only - DO NOT USE
    """
    exclude_patterns = [
        # System and timing tables
        'YPSTimingView', '*Timing*', '*Log*', '*Audit*', 'sys*',
        # Backup and temporary tables  
        '*_backup', '*_temp', '*_staging', '*Renamed', '*History',
        '*_BCKP_*', '*Bck_*', '*_BCK_*', '*backup*', '*bckp*',
        # Archive and snapshot tables
        '*_archive*', '*_snapshot*', '*_copy*', '*_old*',
        # ETL and staging tables
        '*_staging*', '*_load*', '*_import*', '*_export*',
        # Test and development tables
        '*_test*', '*_dev*', '*_tmp*', '*temp*',
        'BusinessPointIdentificationWithThirdPartyData'
    ]
    
    full_name = f"{schema_name}.{table_name}"
    table_lower = table_name.lower()
    
    # Check exclusion patterns
    for pattern in exclude_patterns:
        if '*' in pattern:
            pattern_clean = pattern.replace('*', '').lower()
            if pattern_clean in table_lower:
                return True
        elif pattern.lower() == table_lower:
            return True
    
    # Additional backup table detection by date patterns
    backup_date_patterns = [
        r'_\d{8}$',      # _20250120
        r'_\d{6}$',      # _202501
        r'Bck_\d{8}$',   # Bck_20190523
        r'BCKP_\d{8}$'   # BCKP_20250129
    ]
    
    for pattern in backup_date_patterns:
        if re.search(pattern, table_name):
            return True
    
    return False

def classify_object_priority(table_name: str, schema_name: str, object_type: str, estimated_rows: int) -> int:
    """
    Classify object priority for processing order (higher = more important)
    Helps ensure business-critical objects are processed first in large datasets
    """
    name_lower = table_name.lower()
    priority = 0
    
    # Base priority by object type
    if object_type in ['BASE TABLE', 'TABLE']:
        priority += 1000  # Tables get highest base priority
    elif object_type == 'VIEW':
        priority += 800   # Views get high priority (many contain business logic)
    
    # Business domain priority (VERY HIGH - these are the most valuable)
    high_value_business = [
        'customer', 'client', 'businesspoint', 'account', 'user',
        'order', 'sales', 'purchase', 'transaction', 'payment',
        'product', 'item', 'service', 'inventory',
        'invoice', 'billing', 'financial', 'revenue',
        'person', 'employee', 'staff', 'contact'
    ]
    
    for term in high_value_business:
        if term in name_lower:
            priority += 500  # Major business priority boost
            break
    
    # Medium value business objects
    medium_value_business = [
        'address', 'phone', 'email', 'communication',
        'company', 'organization', 'vendor', 'supplier',
        'campaign', 'marketing', 'advertisement', 'promotion',
        'task', 'assignment', 'project', 'workflow',
        'category', 'type', 'status', 'configuration'
    ]
    
    for term in medium_value_business:
        if term in name_lower:
            priority += 200
            break
    
    # Data volume consideration (objects with data are more valuable)
    if estimated_rows > 10000:
        priority += 100
    elif estimated_rows > 1000:
        priority += 50
    elif estimated_rows > 100:
        priority += 20
    elif estimated_rows > 0:
        priority += 10
    
    # Schema priority (business schemas first)
    if schema_name.lower() == 'dbo':
        priority += 100
    elif schema_name.lower() in ['business', 'sales', 'finance', 'hr', 'crm']:
        priority += 150
    
    # Penalize only clearly problematic objects
    problematic_patterns = [
        'temp', 'tmp', 'test', 'debug', 'error',
        'backup_20', 'bck_20', '_old_', '_archive'
    ]
    
    for pattern in problematic_patterns:
        if pattern in name_lower:
            priority -= 200
            break
    
    return max(priority, 0)  # Never go below 0

def clean_sql_response(response: str) -> Optional[str]:
    """Clean SQL response from LLM"""
    sql_query = response.strip()
    
    # Remove code block markers
    sql_query = re.sub(r'```sql\s*', '', sql_query)
    sql_query = re.sub(r'```\s*', '', sql_query)
    
    # Remove any explanatory text before or after SQL
    lines = sql_query.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith('SELECT') or line.upper().startswith('WITH'):
            in_sql = True
        
        if in_sql and line:
            sql_lines.append(line)
        
        # Stop at end of SQL statement
        if in_sql and line.endswith(';'):
            break
    
    if sql_lines:
        cleaned_sql = '\n'.join(sql_lines)
        # Remove trailing semicolon if present
        cleaned_sql = cleaned_sql.rstrip(';')
        return cleaned_sql
    
    # Fallback: if no clear SQL structure found, return original
    if sql_query.upper().strip().startswith('SELECT'):
        return sql_query.rstrip(';')
    
    return None

def extract_sample_greek_text(tables: List['TableInfo']) -> List[str]:
    """Extract sample Greek text for domain analysis context"""
    greek_samples = []
    
    for table in tables[:15]:  # Check more tables for better Greek text detection
        if table.sample_data:
            for row in table.sample_data[:3]:  # Check more samples
                for key, value in row.items():
                    if isinstance(value, str) and any(ord(char) > 127 for char in value):
                        greek_samples.append(f"{table.name}.{key}: {value[:50]}")
                        if len(greek_samples) >= 10:  # Collect more samples
                            return greek_samples
    
    return greek_samples

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def estimate_processing_time(object_count: int, workers: int, timeout_per_object: int = 20) -> str:
    """Estimate total processing time for large datasets"""
    # Conservative estimate: each object takes average of timeout/2 seconds
    avg_time_per_object = timeout_per_object / 2
    
    # With parallel processing
    total_time_seconds = (object_count * avg_time_per_object) / workers
    
    # Add overhead for batching and setup
    overhead = object_count * 0.1  # 0.1 seconds overhead per object
    total_time_seconds += overhead
    
    return format_duration(total_time_seconds)

def get_performance_recommendations(object_count: int) -> List[str]:
    """Get performance recommendations based on dataset size"""
    recommendations = []
    
    if object_count > 500:
        recommendations.extend([
            f"🚀 Large dataset detected ({object_count} objects)",
            "💡 Consider running discovery during off-peak hours",
            "⚡ Enable aggressive parallelism for faster processing",
            "📊 Monitor progress - large datasets take 15-30 minutes",
            "💾 Results will be cached for faster subsequent runs"
        ])
    elif object_count > 200:
        recommendations.extend([
            f"📊 Medium dataset ({object_count} objects)",
            "⚡ Default parallelism should work well",
            "⏱️ Expected completion: 5-15 minutes"
        ])
    else:
        recommendations.extend([
            f"✅ Small dataset ({object_count} objects)",
            "🏃 Quick processing expected: 2-5 minutes"
        ])
    
    return recommendations

def log_filtering_statistics(total_found: int, excluded_count: int, processed_count: int):
    """Log statistics about object filtering for transparency"""
    kept_count = total_found - excluded_count
    exclusion_rate = (excluded_count / total_found) * 100 if total_found > 0 else 0
    
    print(f"\n📊 Object Filtering Statistics:")
    print(f"   • Total objects found: {total_found}")
    print(f"   • Objects excluded: {excluded_count} ({exclusion_rate:.1f}%)")
    print(f"   • Objects kept for analysis: {kept_count}")
    print(f"   • Objects successfully processed: {processed_count}")
    
    if exclusion_rate > 30:
        print(f"   ⚠️ High exclusion rate - consider reviewing filtering rules")
    elif exclusion_rate < 5:
        print(f"   ✅ Minimal exclusion - analyzing most business objects")
    else:
        print(f"   👍 Balanced filtering - keeping business-relevant objects")