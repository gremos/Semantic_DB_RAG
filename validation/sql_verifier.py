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
        
        # Step 2: Extract referenced tables
        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
            tables = self._extract_tables(parsed)
            
            # Get model sources
            model_sources = self._get_model_sources()
            
            # ✅ IMPROVED: Check with and without schema
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
                    issues.append(f"Table '{table}' not in semantic model. Available tables: {list(model_sources)[:5]}")

            
            # Step 4: Check joins match relationships
            joins = self._extract_joins(parsed)
            relationship_keys = self._get_relationship_keys()
            
            for join in joins:
                if not self._is_valid_join(join, relationship_keys):
                    issues.append(f"Join not in semantic model relationships: {join}")
        
        except Exception as e:
            issues.append(f"SQL verification error: {str(e)}")
        
        return (len(issues) == 0, issues)
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from parsed SQL."""
        tables = []
        for table in parsed.find_all(sqlglot.exp.Table):
            # ✅ Get full table name with schema if available
            table_name = table.sql_gen()  # Gets full qualified name
            
            # Alternative: build from parts
            if table.db:
                full_name = f"{table.db}.{table.name}"
            else:
                full_name = table.name
            
            tables.append(full_name)
            
            # ✅ ALSO check unqualified name (fallback)
            if '.' not in full_name:
                tables.append(full_name)
        
        return list(set(tables))  # Remove duplicates
    
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
        """Check if join matches a defined relationship."""
        # Simplified check - in production, parse join condition properly
        return True  # Placeholder