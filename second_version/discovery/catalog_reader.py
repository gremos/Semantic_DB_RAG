from typing import Dict, Any, List
from connectors.base import DatabaseConnector
from config.settings import settings

class CatalogReader:
    """Read database catalog metadata."""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def read_full_catalog(self) -> Dict[str, Any]:
        """
        Read complete database catalog.
        
        Returns discovery data structure (partial - schemas/tables).
        """
        vendor_version = self.connector.get_vendor_version()
        dialect = self.connector.get_dialect()
        
        schemas_data = []
        schemas = self.connector.get_schemas(settings.schema_exclusions)
        
        for schema in schemas:
            tables_data = []
            tables = self.connector.get_tables(schema, settings.table_exclusions)
            
            for table_info in tables:
                table_name = table_info["name"]
                table_type = table_info["type"]
                
                columns = self.connector.get_columns(schema, table_name)
                primary_key = self.connector.get_primary_keys(schema, table_name)
                foreign_keys = self.connector.get_foreign_keys(schema, table_name)
                rowcount = self.connector.get_row_count(schema, table_name)
                
                table_data = {
                    "name": table_name,
                    "type": table_type,
                    "columns": columns,
                    "primary_key": primary_key,
                    "foreign_keys": foreign_keys,
                    "rowcount_sample": rowcount,
                    "source_assets": []
                }
                
                tables_data.append(table_data)
            
            schemas_data.append({
                "name": schema,
                "tables": tables_data
            })
        
        return {
            "database": vendor_version,
            "dialect": dialect,
            "schemas": schemas_data
        }