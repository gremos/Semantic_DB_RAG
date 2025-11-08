"""
Grounding Verification for QuadRails
Ensures every SQL token maps to Discovery/Semantic Model objects
"""

import logging
from typing import Dict, Any, List, Set
import sqlglot

logger = logging.getLogger(__name__)


def verify_grounding(
    sql: str,
    semantic_model: Dict[str, Any],
    dialect: str = "mssql"
) -> Dict[str, Any]:
    """
    Verify that SQL is grounded in semantic model
    
    Returns:
        {
            'grounded': bool,
            'violations': List[str],
            'warnings': List[str],
            'coverage': float (0.0-1.0)
        }
    """
    result = {
        'grounded': True,
        'violations': [],
        'warnings': [],
        'coverage': 1.0
    }
    
    try:
        # Build semantic model lookups
        valid_tables = _build_table_lookup(semantic_model)
        valid_columns = _build_column_lookup(semantic_model)
        valid_joins = _build_join_lookup(semantic_model)
        
        # Parse SQL
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Extract tables, columns, and joins from SQL
        used_tables = set()
        used_columns = set()
        used_joins = set()
        
        # Check tables
        for table in parsed.find_all(sqlglot.exp.Table):
            table_name = table.name
            used_tables.add(table_name)
            
            if table_name not in valid_tables:
                result['violations'].append(f"Table '{table_name}' not in semantic model")
                result['grounded'] = False
        
        # Check columns
        for column in parsed.find_all(sqlglot.exp.Column):
            col_name = column.name
            table_name = column.table if hasattr(column, 'table') and column.table else None
            
            if table_name:
                full_col = f"{table_name}.{col_name}"
                used_columns.add(full_col)
                
                if full_col not in valid_columns:
                    result['violations'].append(f"Column '{full_col}' not in semantic model")
                    result['grounded'] = False
            else:
                # Column without table qualifier - check if it exists in any used table
                found = False
                for t in used_tables:
                    if f"{t}.{col_name}" in valid_columns:
                        found = True
                        break
                
                if not found:
                    result['warnings'].append(f"Column '{col_name}' ambiguous or not found")
        
        # Check joins
        for join in parsed.find_all(sqlglot.exp.Join):
            # Extract join condition
            on_clause = join.args.get('on')
            if on_clause:
                # Parse join columns
                for condition in on_clause.find_all(sqlglot.exp.EQ):
                    left = str(condition.left)
                    right = str(condition.right)
                    
                    join_pair = f"{left} = {right}"
                    used_joins.add(join_pair)
                    
                    # Check if this join is defined in semantic model
                    if not _is_valid_join(left, right, valid_joins):
                        result['warnings'].append(f"Join '{join_pair}' not explicitly defined in relationships")
        
        # Calculate coverage
        if used_tables or used_columns:
            total_used = len(used_tables) + len(used_columns)
            violations_count = len(result['violations'])
            result['coverage'] = max(0.0, 1.0 - (violations_count / total_used))
        
    except Exception as e:
        logger.error(f"Grounding verification failed: {e}")
        result['grounded'] = False
        result['violations'].append(f"Verification error: {str(e)}")
    
    return result


def _build_table_lookup(semantic_model: Dict[str, Any]) -> Set[str]:
    """Build set of valid table names"""
    tables = set()
    
    for entity in semantic_model.get('entities', []):
        source = entity.get('source', '')
        # Extract table name from schema.table
        table_name = source.split('.')[-1] if '.' in source else source
        tables.add(table_name)
    
    for dim in semantic_model.get('dimensions', []):
        source = dim.get('source', '')
        if not source.startswith('VIRTUAL_'):  # Skip virtual dimensions
            table_name = source.split('.')[-1] if '.' in source else source
            tables.add(table_name)
    
    for fact in semantic_model.get('facts', []):
        source = fact.get('source', '')
        table_name = source.split('.')[-1] if '.' in source else source
        tables.add(table_name)
    
    return tables


def _build_column_lookup(semantic_model: Dict[str, Any]) -> Set[str]:
    """Build set of valid table.column pairs"""
    columns = set()
    
    for entity in semantic_model.get('entities', []):
        source = entity.get('source', '')
        table_name = source.split('.')[-1] if '.' in source else source
        
        for col in entity.get('columns', []):
            columns.add(f"{table_name}.{col['name']}")
    
    for dim in semantic_model.get('dimensions', []):
        source = dim.get('source', '')
        if source.startswith('VIRTUAL_'):
            # Virtual dimension - use dimension name as table alias
            table_name = dim['name']
        else:
            table_name = source.split('.')[-1] if '.' in source else source
        
        for attr in dim.get('attributes', []):
            columns.add(f"{table_name}.{attr['name']}")
    
    for fact in semantic_model.get('facts', []):
        source = fact.get('source', '')
        table_name = source.split('.')[-1] if '.' in source else source
        
        # Add grain columns
        for grain_col in fact.get('grain', []):
            columns.add(f"{table_name}.{grain_col}")
        
        # Add foreign key columns
        for fk in fact.get('foreign_keys', []):
            columns.add(f"{table_name}.{fk['column']}")
    
    return columns


def _build_join_lookup(semantic_model: Dict[str, Any]) -> Set[str]:
    """Build set of valid join pairs"""
    joins = set()
    
    for rel in semantic_model.get('relationships', []):
        from_col = rel.get('from', '')
        to_col = rel.get('to', '')
        
        # Store bidirectional
        joins.add(f"{from_col}={to_col}")
        joins.add(f"{to_col}={from_col}")
    
    return joins


def _is_valid_join(left: str, right: str, valid_joins: Set[str]) -> bool:
    """Check if a join is valid"""
    join_str = f"{left}={right}"
    reverse_join = f"{right}={left}"
    
    return join_str in valid_joins or reverse_join in valid_joins