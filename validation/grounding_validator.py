from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class GroundingValidator:
    """Rail 1: Grounding Rail - ensure all references exist in discovery."""
    
    def __init__(self, discovery_data: Dict[str, Any]):
        self.discovery = discovery_data
        self._build_lookup()
    
    def _build_lookup(self):
        """Build fast lookup structures."""
        self.tables = set()
        self.columns = {}  # table -> [columns]
        
        for schema in self.discovery.get("schemas", []):
            schema_name = schema["name"]
            for table in schema.get("tables", []):
                table_name = table["name"]
                full_name = f"{schema_name}.{table_name}"
                self.tables.add(full_name)
                
                self.columns[full_name] = set(
                    col["name"] for col in table.get("columns", [])
                )
    
    def validate_semantic_model(self, model: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate semantic model references actual discovery objects.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check entities
        for entity in model.get("entities", []):
            source = entity.get("source")
            if source and source not in self.tables:
                errors.append(f"Entity '{entity['name']}' references unknown source: {source}")
        
        # Check dimensions
        for dimension in model.get("dimensions", []):
            source = dimension.get("source")
            if source and source not in self.tables:
                errors.append(f"Dimension '{dimension['name']}' references unknown source: {source}")
        
        # Check facts
        for fact in model.get("facts", []):
            source = fact.get("source")
            if source and source not in self.tables:
                errors.append(f"Fact '{fact['name']}' references unknown source: {source}")
            
            # Check measures depend on actual columns
            if source and source in self.columns:
                table_columns = self.columns[source]
                for measure in fact.get("measures", []):
                    for dep_col in measure.get("depends_on", []):
                        if dep_col not in table_columns:
                            errors.append(
                                f"Measure '{measure['name']}' depends on unknown column: {dep_col}"
                            )
        
        return (len(errors) == 0, errors)
    
    def validate_sql_references(self, sql: str, referenced_tables: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate SQL references known tables.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        for table in referenced_tables:
            if table not in self.tables:
                errors.append(f"SQL references unknown table: {table}")
        
        return (len(errors) == 0, errors)