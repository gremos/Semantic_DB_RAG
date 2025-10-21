from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class EscalationHandler:
    """Rail 4: Escalation Rail - generate refusals with clarifications."""
    
    @staticmethod
    def create_refusal(
        reason: str, 
        missing_objects: List[str] = None,
        clarifying_questions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create refusal JSON when semantics are unclear.
        
        Returns: Refusal structure matching answer schema.
        """
        refusal = {
            "status": "refuse",
            "refusal": {
                "reason": reason
            }
        }
        
        if missing_objects:
            refusal["refusal"]["missing_objects"] = missing_objects[:10]  # Limit
        
        if clarifying_questions:
            refusal["refusal"]["clarifying_questions"] = clarifying_questions[:3]  # Max 3
        
        return refusal
    
    @staticmethod
    def suggest_clarifications(question: str, model: Dict[str, Any]) -> List[str]:
        """
        Generate clarifying questions based on ambiguous query.
        
        Returns: List of 1-3 concrete questions.
        """
        suggestions = []
        
        # Check if metrics are defined
        metrics = model.get("metrics", [])
        if "revenue" in question.lower() and not any("revenue" in m["name"].lower() for m in metrics):
            suggestions.append("Should 'Revenue' be calculated as gross or net? Please define.")
        
        # Check for time dimensions
        if "recent" in question.lower() or "last" in question.lower():
            if not any("date" in d["name"].lower() for d in model.get("dimensions", [])):
                suggestions.append("What time period should be used? (e.g., last 30 days, last quarter)")
        
        # Check for customer segmentation
        if "customer" in question.lower() and "segment" in question.lower():
            if not any("segment" in e["name"].lower() for e in model.get("entities", [])):
                suggestions.append("How should customers be segmented? (e.g., by industry, size, region)")
        
        return suggestions[:3]