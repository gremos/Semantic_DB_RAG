from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DiscoveryCompressor:
    """Compress discovery data for incremental LLM processing."""
    
    @staticmethod
    def compress(discovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress discovery to minimal representation.
        
        Returns: Compressed discovery suitable for incremental processing.
        """
        compressed = {
            "database": discovery_data.get("database"),
            "dialect": discovery_data.get("dialect"),
            "tables": {},
            "column_samples": discovery_data.get("column_samples", {})
        }
        
        # Compress each table
        for schema in discovery_data.get("schemas", []):
            schema_name = schema["name"]
            
            for table in schema.get("tables", []):
                table_name = table["name"]
                full_name = f"{schema_name}.{table_name}"
                
                compressed["tables"][full_name] = {
                    "name": table_name,
                    "schema": schema_name,
                    "type": table.get("type"),
                    "columns": DiscoveryCompressor._compress_columns(table.get("columns", [])),
                    "pk": table.get("primary_key", []),
                    "fks": DiscoveryCompressor._compress_fks(table.get("foreign_keys", [])),
                    "rows": table.get("rowcount_sample", 0)
                }
        
        logger.info(f"Compressed {len(compressed['tables'])} tables")
        return compressed
    
    @staticmethod
    def _compress_columns(columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress column list to essential info only."""
        return [
            {
                "name": col["name"],
                "type": col["type"],
                "nullable": col.get("nullable", True)
            }
            for col in columns
        ]
    
    @staticmethod
    def _compress_fks(fks: List[Dict[str, str]]) -> List[str]:
        """Compress foreign keys to compact notation."""
        return [
            f"{fk['column']}→{fk['ref_table']}.{fk['ref_column']}"
            for fk in fks
        ]
    
    @staticmethod
    def get_table_for_classification(
        compressed: Dict[str, Any], 
        table_name: str
    ) -> Dict[str, Any]:
        """Extract minimal info for table classification."""
        table = compressed["tables"].get(table_name)
        if not table:
            return None
        
        return {
            "name": table["name"],
            "schema": table["schema"],
            "full_name": table_name,
            "column_names": [c["name"] for c in table["columns"]],
            "column_types": [c["type"] for c in table["columns"]],
            "has_pk": len(table["pk"]) > 0,
            "has_fks": len(table["fks"]) > 0,
            "row_count": table["rows"],
            "type": table["type"]
        }
    
    @staticmethod
    def get_columns_for_table(
        compressed: Dict[str, Any],
        table_name: str
    ) -> List[Dict[str, Any]]:
        """Get full column info for a table."""
        table = compressed["tables"].get(table_name)
        if not table:
            return []
        return table["columns"]
    
    @staticmethod
    def get_fks_for_table(
        compressed: Dict[str, Any],
        table_name: str
    ) -> List[str]:
        """Get foreign keys for a table."""
        table = compressed["tables"].get(table_name)
        if not table:
            return []
        return table["fks"]