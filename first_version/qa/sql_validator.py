import sqlglot
from typing import Tuple, Set
import logging

logger = logging.getLogger(__name__)

class SQLColumnValidator:
    """
    Validates that all columns in generated SQL actually exist in the database.
    Catches hallucinated columns before execution.
    """
    
    def __init__(self, discovery_json: dict):
        self.discovery = discovery_json
        self.column_index = self._build_column_index()
    
    def _build_column_index(self) -> dict:
        """
        Build fast lookup: {"dbo.Contract": {"ID", "FinalPrice", "BillingPointID", ...}}
        """
        index = {}
        for schema in self.discovery.get("schemas", []):
            schema_name = schema["name"]
            for table in schema["tables"]:
                table_name = table["name"]
                full_name = f"{schema_name}.{table_name}"
                
                columns = {col["name"] for col in table.get("columns", [])}
                index[full_name] = columns
                index[table_name] = columns  # Also allow unqualified lookups
        
        return index
    
    def validate(self, sql: str, dialect: str = "tsql") -> Tuple[bool, str]:
        """
        Validate that all referenced columns exist.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
        except Exception as e:
            return (False, f"SQL parse error: {str(e)}")
        
        # Extract all column references
        referenced_columns = set()
        for node in parsed.walk():
            if isinstance(node, sqlglot.exp.Column):
                table = node.table
                column = node.name
                referenced_columns.add((table, column))
        
        # Check each column exists
        missing = []
        for table, column in referenced_columns:
            if table:
                # Qualified: table.column
                table_columns = self.column_index.get(table, set())
                if column not in table_columns:
                    missing.append(f"{table}.{column}")
            # Note: Unqualified columns are harder to validate; skip for now
        
        if missing:
            error = f"Invalid columns: {', '.join(missing)}. These do not exist in the database schema."
            logger.error(error)
            return (False, error)
        
        return (True, "")