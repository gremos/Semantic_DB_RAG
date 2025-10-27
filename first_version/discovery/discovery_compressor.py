from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DiscoveryCompressor:
    """Compress discovery data while PRESERVING critical column metadata."""
    
    @staticmethod
    def compress(
        discovery_data: Dict[str, Any],
        column_samples: Dict[str, Any]  # NEW: Enhanced samples with classifications
    ) -> Dict[str, Any]:
        """
        Compress discovery to minimal representation while preserving semantic intelligence.
        
        Returns: Compressed discovery suitable for incremental processing.
        """
        compressed = {
            "database": discovery_data.get("database"),
            "dialect": discovery_data.get("dialect"),
            "tables": {},
            "column_samples": {},  # Will be enhanced
            "column_classifications": {},  # NEW: Semantic role mappings
            "nl_mappings": {}  # NEW: Natural language → column mappings
        }
        
        # Build NL mappings index for fast lookup
        nl_mappings = {}
        
        # Compress each table
        for schema in discovery_data.get("schemas", []):
            schema_name = schema["name"]
            
            for table in schema.get("tables", []):
                table_name = table["name"]
                full_name = f"{schema_name}.{table_name}"
                
                # Compress columns with ENHANCED metadata
                columns_compressed = []
                for col in table.get("columns", []):
                    col_name = col["name"]
                    full_col = f"{full_name}.{col_name}"
                    
                    # Get classification if available
                    sample_data = column_samples.get(full_col, {})
                    classification = sample_data.get("classification", {})
                    
                    col_compressed = {
                        "name": col_name,
                        "type": col["type"],
                        "nullable": col.get("nullable", True),
                        "semantic_role": classification.get("semantic_role"),
                        "priority": classification.get("priority"),
                        "is_pk": classification.get("is_pk", False),
                        "is_fk": classification.get("is_fk", False)
                    }
                    
                    columns_compressed.append(col_compressed)
                    
                    # Store classification separately for easy lookup
                    if classification:
                        compressed["column_classifications"][full_col] = classification
                        
                        # Build NL mappings
                        for alias in classification.get("nl_aliases", []):
                            if alias not in nl_mappings:
                                nl_mappings[alias] = []
                            nl_mappings[alias].append(full_col)
                    
                    # Store sample values
                    if sample_data.get("values"):
                        compressed["column_samples"][full_col] = {
                            "values": sample_data["values"],
                            "distinct_count": sample_data.get("distinct_count", 0)
                        }
                
                compressed["tables"][full_name] = {
                    "name": table_name,
                    "schema": schema_name,
                    "type": table.get("type"),
                    "columns": columns_compressed,
                    "pk": table.get("primary_key", []),
                    "fks": DiscoveryCompressor._compress_fks(table.get("foreign_keys", [])),
                    "rows": table.get("rowcount_sample", 0)
                }
        
        # Store NL mappings for Q&A phase
        compressed["nl_mappings"] = nl_mappings
        
        logger.info(f"Compressed {len(compressed['tables'])} tables")
        logger.info(f"Created {len(nl_mappings)} natural language mappings")
        logger.info(f"Classified {len(compressed['column_classifications'])} columns")
        
        return compressed
    
    @staticmethod
    def _compress_fks(fks: List[Dict[str, str]]) -> List[str]:
        """Compress foreign keys to compact notation."""
        return [
            f"{fk['column']}→{fk['ref_table']}.{fk['ref_column']}"
            for fk in fks
        ]
    
    # Rest of the methods remain the same...
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
            "column_roles": [c.get("semantic_role") for c in table["columns"]],  # NEW
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
    
    @staticmethod
    def get_nl_mapping(
        compressed: Dict[str, Any],
        natural_language_phrase: str
    ) -> List[str]:
        """
        NEW: Get columns that match a natural language phrase.
        
        Returns list of full column names (schema.table.column)
        """
        phrase_lower = natural_language_phrase.lower()
        nl_mappings = compressed.get("nl_mappings", {})
        
        # Direct match
        if phrase_lower in nl_mappings:
            return nl_mappings[phrase_lower]
        
        # Fuzzy match (contains)
        matches = []
        for alias, columns in nl_mappings.items():
            if phrase_lower in alias or alias in phrase_lower:
                matches.extend(columns)
        
        return list(set(matches))  # Remove duplicates