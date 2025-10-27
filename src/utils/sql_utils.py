"""
SQL utilities using sqlglot for parsing, normalization, and validation.
"""

import sqlglot
from typing import Optional, List, Dict, Any


def parse_sql(sql: str, dialect: str = 'mssql') -> Optional[sqlglot.Expression]:
    """Parse SQL string into AST."""
    try:
        return sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return None


def normalize_sql(sql: str, dialect: str = 'mssql') -> str:
    """Normalize SQL formatting."""
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        return parsed.sql(dialect=dialect, pretty=True)
    except Exception:
        return sql


def validate_readonly(sql: str) -> bool:
    """Validate that SQL is read-only (no DML/DDL)."""
    sql_upper = sql.upper()
    forbidden_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 
        'DROP', 'ALTER', 'CREATE', 'GRANT', 'REVOKE'
    ]
    return not any(keyword in sql_upper for keyword in forbidden_keywords)


def lint_sql(sql: str, dialect: str = 'mssql') -> List[str]:
    """
    Lint SQL for common issues.
    Returns list of warning messages.
    """
    warnings = []
    
    # Parse check
    parsed = parse_sql(sql, dialect)
    if not parsed:
        warnings.append("Failed to parse SQL")
        return warnings
    
    # Read-only check
    if not validate_readonly(sql):
        warnings.append("SQL contains write operations")
    
    return warnings
