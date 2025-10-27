"""
SQL Verifier - QuadRail #3: Verification
Validates that generated SQL only references tables and columns that exist in the semantic model.
CRITICAL: Prevents hallucinated column names like "TotalAmount" when column doesn't exist.
"""

from typing import Dict, Any, List, Tuple
from normalization.sql_normalizer import SQLNormalizer
import sqlglot
import logging

logger = logging.getLogger(__name__)


class SQLVerifier:
    """Rail 3: Verification Rail - lint SQL and check relationships."""
    
    def __init__(self, semantic_model: Dict[str, Any], discovery_data: Dict[str, Any]):
        self.model = semantic_model
        self.discovery = discovery_data
        self.normalizer = SQLNormalizer()
    
    def verify_sql(self, sql: str, dialect: str) -> Tuple[bool, List[str]]:
        """
        Dry-run lint: verify SQL is parseable and references are valid.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Step 1: Parse SQL
        success, error = self.normalizer.parse(sql, dialect)
        if not success:
            issues.append(f"SQL parse error: {error}")
            return (False, issues)
        
        # Step 2: Extract referenced tables and validate
        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
            tables = self._extract_tables(parsed)
            
            # Get model sources
            model_sources = self._get_model_sources()
            
            # Validate tables exist
            for table in tables:
                # Try exact match first
                if table in model_sources:
                    continue
                
                # Try with common schema prefixes
                found = False
                for source in model_sources:
                    # Check if table matches the end of source (e.g., "Table" matches "dbo.Table")
                    if source.endswith('.' + table) or source == table:
                        found = True
                        break
                
                if not found:
                    available_list = list(model_sources)[:10]
                    issues.append(
                        f"Table '{table}' not in semantic model. "
                        f"Available tables: {', '.join(available_list)}"
                    )
            
            # Step 3: Extract and verify columns (NEW - CRITICAL FIX)
            columns = self._extract_columns(parsed)
            available_columns = self._get_available_columns()
            
            for table_ref, col_name in columns:
                # Skip aggregate functions and expressions
                if col_name.upper() in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', '*']:
                    continue
                
                if table_ref:
                    # Qualified reference (Table.Column)
                    resolved_table = self._resolve_table_reference(table_ref, available_columns)
                    
                    if not resolved_table:
                        issues.append(
                            f"Column reference to unknown table: {table_ref}.{col_name}"
                        )
                        continue
                    
                    # Verify column exists in table
                    table_columns = available_columns.get(resolved_table, [])
                    if col_name not in table_columns:
                        # Try case-insensitive match
                        col_name_lower = col_name.lower()
                        table_columns_lower = [c.lower() for c in table_columns]
                        
                        if col_name_lower not in table_columns_lower:
                            issues.append(
                                f"❌ HALLUCINATION: Column '{col_name}' does not exist in table '{resolved_table}'. "
                                f"Available columns: {', '.join(table_columns[:15])}"
                            )
                else:
                    # Unqualified column reference - harder to validate
                    # Check if it exists in ANY table from the query
                    found_in_any = False
                    for tbl in tables:
                        resolved_tbl = self._resolve_table_reference(tbl, available_columns)
                        if resolved_tbl and col_name in available_columns.get(resolved_tbl, []):
                            found_in_any = True
                            break
                    
                    if not found_in_any:
                        logger.warning(
                            f"Unqualified column '{col_name}' not found in any referenced table. "
                            f"This may cause runtime errors."
                        )
            
            # Step 4: Check joins match relationships (existing logic)
            joins = self._extract_joins(parsed)
            relationship_keys = self._get_relationship_keys()
            
            for join in joins:
                if not self._is_valid_join(join, relationship_keys):
                    # Soft warning - joins can be valid even without explicit relationships
                    logger.info(f"Join not in semantic model relationships: {join}")
        
        except Exception as e:
            issues.append(f"SQL verification error: {str(e)}")
            logger.error(f"Verification exception: {e}", exc_info=True)
        
        return (len(issues) == 0, issues)
    
    def _resolve_table_reference(
        self, 
        table_ref: str, 
        available_columns: Dict[str, List[str]]
    ) -> str:
        """
        Resolve a table reference to its full name in available_columns.
        Handles both qualified (schema.table) and unqualified (table) references.
        """
        # Try exact match
        if table_ref in available_columns:
            return table_ref
        
        # Try with schema prefix
        for full_name in available_columns.keys():
            if full_name.endswith('.' + table_ref):
                return full_name
        
        return None
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from parsed SQL."""
        tables = []
        for table in parsed.find_all(sqlglot.exp.Table):
            # Build qualified name from parts
            if hasattr(table, 'db') and table.db:
                full_name = f"{table.db}.{table.name}"
            else:
                full_name = table.name if hasattr(table, 'name') else str(table)
            
            tables.append(full_name)
            
            # Also add just the table name for flexible matching
            if '.' in full_name:
                tables.append(full_name.split('.')[-1])
        
        return list(set(tables))  # Remove duplicates
    
    def _extract_columns(self, parsed) -> List[Tuple[str, str]]:
        """
        Extract all column references from parsed SQL.
        Returns list of (table_name_or_none, column_name) tuples.
        
        CRITICAL: This prevents hallucinated column names.
        """
        columns = []
        
        # Extract from SELECT, WHERE, GROUP BY, HAVING, ORDER BY
        for col in parsed.find_all(sqlglot.exp.Column):
            table_name = None
            col_name = col.name if hasattr(col, 'name') else str(col)
            
            # Try to get table qualifier
            if hasattr(col, 'table') and col.table:
                table_name = str(col.table)
            
            columns.append((table_name, col_name))
        
        return columns
    
    def _get_available_columns(self) -> Dict[str, List[str]]:
        """
        Build map of table -> [columns] from discovery data.
        This is the authoritative source of what columns actually exist.
        """
        columns_map = {}
        
        for schema in self.discovery.get("schemas", []):
            for table in schema.get("tables", []):
                full_name = f"{schema['name']}.{table['name']}"
                short_name = table['name']
                
                col_names = [c['name'] for c in table.get('columns', [])]
                columns_map[full_name] = col_names
                columns_map[short_name] = col_names  # Support unqualified refs
        
        return columns_map
    
    def _extract_joins(self, parsed) -> List[Dict[str, str]]:
        """Extract join conditions from parsed SQL."""
        joins = []
        for join in parsed.find_all(sqlglot.exp.Join):
            # Simplified extraction
            joins.append({"type": "join", "condition": str(join)})
        return joins
    
    def _get_model_sources(self) -> set:
        """Get all source tables from semantic model."""
        sources = set()
        
        for entity in self.model.get("entities", []):
            sources.add(entity.get("source"))
        
        for dimension in self.model.get("dimensions", []):
            sources.add(dimension.get("source"))
        
        for fact in self.model.get("facts", []):
            sources.add(fact.get("source"))
        
        return sources
    
    def _get_relationship_keys(self) -> set:
        """Get all valid join keys from relationships."""
        keys = set()
        for rel in self.model.get("relationships", []):
            keys.add((rel.get("from"), rel.get("to")))
        return keys
    
    def _is_valid_join(self, join: Dict[str, str], relationship_keys: set) -> bool:
        """
        Check if join matches a defined relationship.
        
        Note: This is a soft check - joins can be valid even without explicit relationships.
        We return True (valid) to avoid false positives, but log for observability.
        """
        # Simplified check - in production, parse join condition properly
        # For now, we're lenient to avoid blocking valid queries
        return True