"""
Use LLM to select the best table/column when multiple candidates exist.
This prevents refusals due to ambiguity.
"""

from typing import List, Dict, Optional
import json


class TableDisambiguator:
    """Intelligently select the best table/measure for a question using LLM + evidence."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def disambiguate_customer_source(
        self, 
        candidates: List[str], 
        question: str,
        semantic_model: dict,
        view_usage: dict
    ) -> Dict:
        """
        Given multiple possible 'customer' tables, pick the best one.
        
        Returns:
        {
            "selected": "dbo.BusinessPoint",
            "column": "BrandName",
            "confidence": 0.92,
            "reasoning": "Most frequently used in views (12 references) and contains business-friendly names"
        }
        """
        
        # Build evidence for each candidate
        evidence = []
        for candidate in candidates:
            table_name = candidate.split(".")[0] if "." in candidate else candidate
            
            evidence.append({
                "table_column": candidate,
                "view_usage_count": view_usage.get(table_name, {}).get("total_references", 0),
                "is_in_semantic_model": self._is_in_semantic_model(candidate, semantic_model),
                "related_measures": self._find_related_measures(table_name, semantic_model)
            })
        
        prompt = f"""
You are a semantic model expert. The user asked: "{question}"

Multiple possible customer sources found:
{json.dumps(evidence, indent=2)}

Select the BEST customer source based on:
1. Highest view_usage_count (most authoritative)
2. Present in semantic_model (validated entity)
3. Has related sales measures (can answer revenue questions)
4. Business-friendly column names (BrandName > CustomerName > Name)

Output JSON:
{{
  "selected": "dbo.BusinessPoint.BrandName",
  "confidence": 0.92,
  "reasoning": "Explanation here"
}}

Output valid JSON only.
"""
        
        response = self.llm.invoke(prompt)
        
        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # Fallback: pick highest usage
            best = max(evidence, key=lambda e: e["view_usage_count"])
            return {
                "selected": best["table_column"],
                "confidence": 0.6,
                "reasoning": "Selected based on highest view usage (fallback)"
            }
    
    def disambiguate_measure(
        self, 
        candidates: List[str], 
        question: str,
        semantic_model: dict,
        view_usage: dict
    ) -> Dict:
        """
        Given multiple possible 'sales' measures, pick the best one.
        """
        
        evidence = []
        for candidate in candidates:
            # Parse table.column
            parts = candidate.split(".")
            table_name = parts[0] if len(parts) > 1 else candidate
            
            evidence.append({
                "measure": candidate,
                "table": table_name,
                "view_usage_count": view_usage.get(table_name, {}).get("total_references", 0),
                "is_in_facts": self._is_in_facts(table_name, semantic_model),
                "typical_aggregation": self._guess_aggregation(candidate)
            })
        
        prompt = f"""
You are a semantic model expert. The user asked: "{question}"

Multiple possible sales measures found:
{json.dumps(evidence, indent=2)}

Select the BEST measure based on:
1. Present in facts (validated measure)
2. Highest view_usage_count (most authoritative)
3. Appropriate for "total sales" (likely SUM of a Price/Amount field)

Output JSON:
{{
  "selected": "dbo.ContractProduct.TotalPrice",
  "aggregation": "SUM",
  "confidence": 0.88,
  "reasoning": "Explanation here"
}}

Output valid JSON only.
"""
        
        response = self.llm.invoke(prompt)
        
        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # Fallback: pick highest usage
            best = max(evidence, key=lambda e: e["view_usage_count"])
            return {
                "selected": best["measure"],
                "aggregation": "SUM",
                "confidence": 0.5,
                "reasoning": "Selected based on highest view usage (fallback)"
            }
    
    def _is_in_semantic_model(self, table_col: str, semantic_model: dict) -> bool:
        """Check if table.column is in entities or dimensions."""
        for entity in semantic_model.get("entities", []):
            if table_col.startswith(entity["source"]):
                return True
        for dim in semantic_model.get("dimensions", []):
            if table_col.startswith(dim["source"]):
                return True
        return False
    
    def _is_in_facts(self, table: str, semantic_model: dict) -> bool:
        """Check if table is a fact."""
        for fact in semantic_model.get("facts", []):
            if table in fact["source"]:
                return True
        return False
    
    def _find_related_measures(self, table: str, semantic_model: dict) -> List[str]:
        """Find measures from facts related to this table."""
        measures = []
        for fact in semantic_model.get("facts", []):
            # Check if this table is referenced via FK
            for fk in fact.get("foreign_keys", []):
                if table in fk.get("references", ""):
                    measures.extend([m["name"] for m in fact.get("measures", [])])
        return measures
    
    def _guess_aggregation(self, measure: str) -> str:
        """Guess appropriate aggregation based on column name."""
        lower = measure.lower()
        if "count" in lower or "quantity" in lower or "qty" in lower:
            return "SUM"
        if "price" in lower or "amount" in lower or "revenue" in lower or "total" in lower:
            return "SUM"
        if "rate" in lower or "percent" in lower:
            return "AVG"
        return "SUM"  # Default