"""
Export semantic model as enhanced JSON with metadata and documentation.
"""

from typing import Dict, Any, List
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class JSONExporter:
    """Export semantic model as documented JSON."""
    
    @staticmethod
    def export(semantic_model: Dict[str, Any], output_file: str = "semantic_model_export.json") -> bool:
        """
        Export semantic model as enhanced JSON.
        
        Args:
            semantic_model: Semantic model JSON
            output_file: Output filename
        
        Returns:
            Success boolean
        """
        try:
            logger.info(f"Exporting to JSON: {output_file}")
            
            # Add metadata
            enhanced_model = {
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "format": "semantic_model_json",
                    "compatible_with": ["Power BI", "Tableau", "Looker", "Custom BI Tools"]
                },
                "model": semantic_model,
                "usage_guide": {
                    "description": "This semantic model can be used to generate SQL queries for BI tools",
                    "tables": {
                        "facts": [f["name"] for f in semantic_model.get("facts", [])],
                        "dimensions": [d["name"] for d in semantic_model.get("dimensions", [])],
                        "entities": [e["name"] for e in semantic_model.get("entities", [])]
                    },
                    "measures_available": JSONExporter._list_measures(semantic_model),
                    "metrics_available": [m["name"] for m in semantic_model.get("metrics", [])]
                }
            }
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_model, f, indent=2)
            
            logger.info(f"✅ JSON exported: {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _list_measures(semantic_model: Dict[str, Any]) -> Dict[str, List[str]]:
        """List all measures by fact table."""
        measures_by_fact = {}
        
        for fact in semantic_model.get("facts", []):
            fact_name = fact["name"]
            measures = [m["name"] for m in fact.get("measures", [])]
            if measures:
                measures_by_fact[fact_name] = measures
        
        return measures_by_fact