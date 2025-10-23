from typing import Dict, Any, List, Tuple
from llm.azure_client import AzureLLMClient
from utils.json_extractor import JSONExtractor
import logging

logger = logging.getLogger(__name__)

class StatusColumnAnalyzer:
    """Phase 3: Analyze status indicator columns."""
    
    SYSTEM_PROMPT = """You are a data modeling expert. Analyze status/flag columns.

For status indicator columns, explain:
1. What does NULL mean?
2. What does a populated value mean?
3. How to filter for 'active' records?

Return ONLY JSON: {
  "semantic_role": "status_indicator",
  "null_means": "description",
  "value_means": "description", 
  "active_filter": "SQL condition",
  "description": "full explanation"
}"""
    
    STATUS_KEYWORDS = ['cancelled', 'deleted', 'active', 'enabled', 'disabled', 
                      'status', 'state', 'flag', 'is', 'has']
    
    def __init__(self, llm_client: AzureLLMClient):
        self.llm = llm_client
        self.json_extractor = JSONExtractor()
    
    def identify_status_columns(self, columns: List[Dict[str, Any]]) -> List[str]:
        """Identify columns that might be status indicators."""
        status_cols = []
        
        for col in columns:
            col_name_lower = col['name'].lower()
            if any(kw in col_name_lower for kw in self.STATUS_KEYWORDS):
                status_cols.append(col['name'])
        
        return status_cols
    
    def analyze_status_column(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
        nullable: bool,
        sample_values: List = None
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Analyze a single status column.
        
        Returns:
            (success, analysis_result, error)
        """
        try:
            user_prompt = f"""Table: {table_name}
Column: {column_name}
Type: {column_type}
Nullable: {nullable}
Sample values: {sample_values if sample_values else 'Not available'}

Analyze this status indicator column. Explain what NULL vs populated values mean.
Return JSON only."""
            
            response = self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
            
            result, method = self.json_extractor.extract(response, log_failures=False)
            
            if not result:
                logger.warning(f"Failed to analyze {table_name}.{column_name}, using heuristic")
                result = self._heuristic_analysis(column_name, column_type, nullable, sample_values)
            
            return (True, result, "")
        
        except Exception as e:
            logger.error(f"Error analyzing {table_name}.{column_name}: {e}")
            result = self._heuristic_analysis(column_name, column_type, nullable, sample_values)
            return (True, result, "")
    
    def _heuristic_analysis(
        self,
        column_name: str,
        column_type: str,
        nullable: bool,
        sample_values: List
    ) -> Dict[str, Any]:
        """Fallback heuristic analysis."""
        col_lower = column_name.lower()
        
        if 'cancelled' in col_lower:
            if 'date' in column_type.lower() or 'time' in column_type.lower():
                return {
                    "semantic_role": "status_indicator",
                    "null_means": "Record is active (not cancelled)",
                    "value_means": "Record is cancelled on this date",
                    "active_filter": f"{column_name} IS NULL",
                    "description": f"{column_name}: NULL means active, populated means cancelled"
                }
            else:
                return {
                    "semantic_role": "status_indicator",
                    "null_means": "Record is active",
                    "value_means": "Record is cancelled",
                    "active_filter": f"{column_name} IS NULL OR {column_name} = 0",
                    "description": f"{column_name}: Indicates cancellation status"
                }
        
        elif 'deleted' in col_lower:
            return {
                "semantic_role": "status_indicator",
                "null_means": "Record is active (not deleted)",
                "value_means": "Record is deleted",
                "active_filter": f"{column_name} IS NULL",
                "description": f"{column_name}: NULL means active, populated means deleted"
            }
        
        elif 'active' in col_lower or 'enabled' in col_lower:
            return {
                "semantic_role": "status_indicator",
                "null_means": "Record is inactive",
                "value_means": "Record is active when TRUE/1",
                "active_filter": f"{column_name} = 1",
                "description": f"{column_name}: 1/TRUE means active, 0/FALSE means inactive"
            }
        
        else:
            return {
                "semantic_role": "status_indicator",
                "null_means": "Unknown",
                "value_means": "Unknown",
                "active_filter": "",
                "description": f"{column_name}: Status indicator (semantics unclear)"
            }