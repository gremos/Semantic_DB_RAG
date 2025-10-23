from typing import Dict, Any, Tuple
from llm.azure_client import AzureLLMClient
from utils.json_extractor import JSONExtractor
import logging

logger = logging.getLogger(__name__)

class TableClassifier:
    """Phase 1: Classify each table as FACT, DIMENSION, or ENTITY."""
    
    SYSTEM_PROMPT = """You are a data modeling expert. Classify database tables.

FACT tables:
- Transactional data (orders, sales, events)
- Has numeric measures (amounts, quantities, counts)
- Many rows (thousands+)
- Has foreign keys to dimension/entity tables

DIMENSION tables:
- Lookup/reference data (products, categories, dates)
- Descriptive attributes
- Moderate rows (hundreds to thousands)
- Often referenced by fact tables

ENTITY tables:
- Master data (customers, suppliers, employees)
- Business objects
- Can have many rows
- Core business concepts

Return ONLY JSON: {"classification": "fact|dimension|entity", "confidence": "high|medium|low", "reasoning": "one sentence"}"""
    
    def __init__(self, llm_client: AzureLLMClient):
        self.llm = llm_client
        self.json_extractor = JSONExtractor()
    
    def classify_table(self, table_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        """
        Classify a single table.
        
        Returns:
            (success, classification_result, error)
        """
        try:
            # FIX: Handle None values in row_count
            row_count = table_info.get('row_count', 0) or 0
            
            user_prompt = f"""Table: {table_info['full_name']}
Type: {table_info['type']}
Columns ({len(table_info['column_names'])}): {', '.join(table_info['column_names'][:20])}{'...' if len(table_info['column_names']) > 20 else ''}
Column types: {', '.join(table_info['column_types'][:20])}
Row count: {row_count:,}
Has primary key: {table_info['has_pk']}
Has foreign keys: {table_info['has_fks']}

Classify this table. Return JSON only."""
            
            response = self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
            
            result, method = self.json_extractor.extract(response, log_failures=False)
            
            if not result:
                logger.warning(f"Failed to classify {table_info['full_name']}")
                # Default classification based on heuristics
                result = self._heuristic_classification(table_info)
            
            return (True, result, "")
        
        except Exception as e:
            logger.error(f"Error classifying {table_info['full_name']}: {e}")
            result = self._heuristic_classification(table_info)
            return (True, result, "")
    
    def _heuristic_classification(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic classification if LLM fails."""
        col_names_lower = [c.lower() for c in table_info['column_names']]
        
        # Check for fact indicators
        measure_keywords = ['amount', 'price', 'quantity', 'cost', 'revenue', 'total']
        has_measures = any(any(kw in col for kw in measure_keywords) for col in col_names_lower)
        
        # Check for dimension indicators
        dim_keywords = ['name', 'description', 'category', 'type']
        has_dim_attrs = any(any(kw in col for kw in dim_keywords) for col in col_names_lower)
        
        if has_measures and table_info['has_fks']:
            return {
                "classification": "fact",
                "confidence": "medium",
                "reasoning": "Has measure-like columns and foreign keys (heuristic)"
            }
        elif has_dim_attrs and not table_info['has_fks']:
            return {
                "classification": "dimension",
                "confidence": "medium",
                "reasoning": "Has descriptive attributes, no FKs (heuristic)"
            }
        else:
            return {
                "classification": "entity",
                "confidence": "low",
                "reasoning": "Default classification (heuristic)"
            }