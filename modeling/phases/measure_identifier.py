from typing import Dict, Any, List, Tuple
from llm.azure_client import AzureLLMClient
from utils.json_extractor import JSONExtractor
import logging

logger = logging.getLogger(__name__)

class MeasureIdentifier:
    """Phase 2: Identify measures for fact tables."""
    
    SYSTEM_PROMPT = """You are a data modeling expert. Identify measures (aggregations) for fact tables.

For each numeric column, suggest:
- name: Business-friendly measure name (e.g., "TotalRevenue", "AveragePrice")
- expression: SQL aggregate (e.g., "SUM(Amount)", "AVG(Price)")
- datatype: decimal, integer, float
- format: currency, number, percentage, duration
- depends_on: Array of column names used

Return ONLY JSON: {"measures": [{"name": "...", "expression": "...", "datatype": "...", "format": "...", "depends_on": [...]}]}"""
    
    def __init__(self, llm_client: AzureLLMClient):
        self.llm = llm_client
        self.json_extractor = JSONExtractor()
    
    def identify_measures(
        self, 
        table_name: str,
        columns: List[Dict[str, Any]]
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Identify measures for a fact table.
        
        Returns:
            (success, measures_list, error)
        """
        # Filter to numeric columns
        numeric_types = ['int', 'integer', 'bigint', 'smallint', 'tinyint', 
                        'decimal', 'numeric', 'float', 'real', 'money', 'smallmoney']
        
        numeric_columns = [
            col for col in columns 
            if any(nt in col['type'].lower() for nt in numeric_types)
        ]
        
        if not numeric_columns:
            logger.info(f"No numeric columns in {table_name}")
            return (True, [], "")
        
        try:
            col_info = [f"{col['name']} ({col['type']})" for col in numeric_columns]
            
            user_prompt = f"""Fact table: {table_name}
Numeric columns: {', '.join(col_info)}

Suggest measures for this fact table. Consider:
- Sum for amounts, quantities, totals
- Count for record counts
- Average for rates, averages
- Min/Max for ranges

Return JSON only."""
            
            response = self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
            
            result, method = self.json_extractor.extract(response, log_failures=False)
            
            if not result or "measures" not in result:
                logger.warning(f"Failed to identify measures for {table_name}, using defaults")
                measures = self._default_measures(numeric_columns)  # FIX: Assign to variable
                return (True, measures, "")  # FIX: Return the list, not dict
            
            return (True, result.get("measures", []), "")
        
        except Exception as e:
            logger.error(f"Error identifying measures for {table_name}: {e}")
            measures = self._default_measures(numeric_columns)  # FIX: Assign to variable
            return (True, measures, "")  # FIX: Return the list, not dict
    
    def _default_measures(self, numeric_columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate default measures if LLM fails."""
        measures = []
        
        for col in numeric_columns[:5]:  # Limit to first 5
            col_name = col['name']
            col_type = col['type'].lower()
            
            # Determine aggregation
            if 'amount' in col_name.lower() or 'price' in col_name.lower() or 'cost' in col_name.lower():
                agg = 'SUM'
                format_type = 'currency'
            elif 'quantity' in col_name.lower() or 'count' in col_name.lower():
                agg = 'SUM'
                format_type = 'number'
            else:
                agg = 'SUM'
                format_type = 'number'
            
            measures.append({
                "name": f"Total{col_name}",
                "expression": f"{agg}({col_name})",
                "datatype": col_type,
                "format": format_type,
                "depends_on": [col_name]
            })
        
        return measures