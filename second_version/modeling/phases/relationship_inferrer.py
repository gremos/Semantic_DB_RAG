from typing import Dict, Any, Tuple
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
            "business_meaning": f"Each {from_table} record references one {to_table} record via {fk_column}"
        }