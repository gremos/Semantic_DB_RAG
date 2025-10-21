from typing import List, Dict, Any
from connectors.base import DatabaseConnector
from normalization.sql_normalizer import SQLNormalizer

class AssetParser:
    """Parse views and stored procedures."""
    
    def __init__(self, connector: DatabaseConnector, normalizer: SQLNormalizer):
        self.connector = connector
        self.normalizer = normalizer
    
    def parse_views(self, schemas: List[str]) -> List[Dict[str, Any]]:
        """Extract and normalize view definitions."""
        assets = []
        dialect = self.connector.get_dialect()
        
        for schema in schemas:
            tables = self.connector.get_tables(schema, [])
            
            for table_info in tables:
                if table_info["type"] == "view":
                    view_name = table_info["name"]
                    definition = self.connector.get_view_definition(schema, view_name)
                    
                    if definition:
                        success, normalized, error = self.normalizer.normalize(definition, dialect)
                        
                        assets.append({
                            "kind": "view",
                            "name": f"{schema}.{view_name}",
                            "sql_normalized": normalized if success else definition,
                            "normalization_error": error if not success else None
                        })
        
        return assets
    
    def parse_stored_procedures(self, schemas: List[str]) -> List[Dict[str, Any]]:
        """Extract and normalize stored procedure definitions."""
        assets = []
        dialect = self.connector.get_dialect()
        
        for schema in schemas:
            procedures = self.connector.get_stored_procedures(schema)
            
            for proc in procedures:
                success, normalized, error = self.normalizer.normalize(
                    proc["definition"], 
                    dialect
                )
                
                assets.append({
                    "kind": "stored_procedure",
                    "name": proc["name"],
                    "sql_normalized": normalized if success else proc["definition"],
                    "normalization_error": error if not success else None
                })
        
        return assets