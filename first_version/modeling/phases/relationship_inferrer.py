from typing import Dict, Any, Tuple, List, Optional
from llm.azure_client import AzureLLMClient
from utils.json_extractor import JSONExtractor
import logging

logger = logging.getLogger(__name__)

class RelationshipInferrer:
    """Phase 4: Infer relationships from foreign keys."""
    
    SYSTEM_PROMPT = """You are a data modeling expert. Describe database relationships.

Given a foreign key relationship, provide:
- from: Source table/entity name
- to: Target table/entity name  
- cardinality: "many-to-one", "one-to-one", "one-to-many"
- business_meaning: One sentence describing the relationship

Return ONLY JSON: {"from": "...", "to": "...", "cardinality": "...", "business_meaning": "..."}"""
    
    def __init__(self, llm_client: AzureLLMClient):
        self.llm = llm_client
        self.json_extractor = JSONExtractor()
    
    def infer_relationship(
        self,
        from_table: str,
        fk_column: str,
        to_table: str,
        to_column: str
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Infer relationship semantics from FK.
        
        Returns:
            (success, relationship_info, error)
        """
        try:
            user_prompt = f"""Foreign key relationship:
From table: {from_table}
FK column: {fk_column}
References table: {to_table}
References column: {to_column}

Describe this relationship in business terms.
Return JSON only."""
            
            response = self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
            
            result, method = self.json_extractor.extract(response, log_failures=False)
            
            if not result:
                logger.warning(f"Failed to infer relationship {from_table}→{to_table}, using default")
                result = self._default_relationship(from_table, fk_column, to_table)
            
            return (True, result, "")
        
        except Exception as e:
            logger.error(f"Error inferring relationship {from_table}→{to_table}: {e}")
            result = self._default_relationship(from_table, fk_column, to_table)
            return (True, result, "")
    
    def _default_relationship(
        self,
        from_table: str,
        fk_column: str,
        to_table: str
    ) -> Dict[str, Any]:
        """Default relationship if LLM fails."""
        return {
            "from": from_table,
            "to": to_table,
            "cardinality": "many-to-one",
            "type": "foreign_key",
            "business_meaning": f"Each {from_table} record references one {to_table} record via {fk_column}"
        }
    
    def infer_relationships_heuristic(
            self,
            compressed_discovery: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
            """
            NEW: Infer relationships from column naming patterns when no FKs exist.
            
            Patterns:
            - CustomerID in table A → Customer.ID
            - ProductCode in table B → Product.Code
            - BusinessPointID → BusinessPoint.ID
            """
            inferred = []
            tables = compressed_discovery.get("tables", {})
            
            logger.info("  Using heuristic relationship inference (no FKs found)")
            
            for from_table, from_data in tables.items():
                for col in from_data.get("columns", []):
                    col_name = col["name"]
                    col_lower = col_name.lower()
                    
                    # Skip if it's a PK
                    if col_name in from_data.get("pk", []):
                        continue
                    
                    # Pattern 1: ColumnName ends with "ID" → look for table "Column"
                    if col_name.endswith("ID") and len(col_name) > 2:
                        target_table_name = col_name[:-2]  # Remove "ID"
                        target_full = self._find_table_by_name(tables, target_table_name)
                        
                        if target_full:
                            inferred.append({
                                "from": from_table,
                                "to": target_full,
                                "cardinality": "many-to-one",
                                "type": "inferred_from_naming",
                                "business_meaning": f"Each {from_table} references one {target_full} via {col_name}",
                                "inferred_via": f"{col_name} → {target_full}.ID (naming pattern)"
                            })
                            logger.debug(f"    Inferred: {from_table}.{col_name} → {target_full}")
                    
                    # Pattern 2: ColumnName ends with "Code" → look for table "Column"  
                    elif col_name.endswith("Code") and len(col_name) > 4:
                        target_table_name = col_name[:-4]
                        target_full = self._find_table_by_name(tables, target_table_name)
                        
                        if target_full:
                            inferred.append({
                                "from": from_table,
                                "to": target_full,
                                "cardinality": "many-to-one",
                                "type": "inferred_from_naming",
                                "business_meaning": f"Each {from_table} references one {target_full} via {col_name}",
                                "inferred_via": f"{col_name} → {target_full}.Code (naming pattern)"
                            })
                            logger.debug(f"    Inferred: {from_table}.{col_name} → {target_full}")
            
            logger.info(f"  Heuristically inferred {len(inferred)} relationships")
            return inferred
        
    def _find_table_by_name(
            self, 
            tables: Dict[str, Any], 
            partial_name: str
        ) -> Optional[str]:
            """Find a table that matches the partial name (case-insensitive)."""
            partial_lower = partial_name.lower()
            
            for full_table_name in tables.keys():
                # Extract table name without schema
                table_name = full_table_name.split('.')[-1]
                
                if table_name.lower() == partial_lower:
                    return full_table_name
            
            return None