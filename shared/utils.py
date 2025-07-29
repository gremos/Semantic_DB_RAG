#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utility functions
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
    """Enhanced exclusion logic for better filtering"""
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
    
    for table in tables[:10]:  # Check first 10 tables
        if table.sample_data:
            for row in table.sample_data[:2]:
                for key, value in row.items():
                    if isinstance(value, str) and any(ord(char) > 127 for char in value):
                        greek_samples.append(f"{table.name}.{key}: {value[:50]}")
                        if len(greek_samples) >= 5:
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