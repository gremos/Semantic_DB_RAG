"""
Export semantic model as human-readable Markdown documentation.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MarkdownExporter:
    """Export semantic model as Markdown documentation."""
    
    @staticmethod
    def export(semantic_model: Dict[str, Any], output_file: str = "SEMANTIC_MODEL.md") -> bool:
        """
        Export semantic model as Markdown documentation.
        
        Args:
            semantic_model: Semantic model JSON
            output_file: Output filename
        
        Returns:
            Success boolean
        """
        try:
            logger.info(f"Exporting to Markdown: {output_file}")
            
            lines = []
            
            # Header
            lines.extend([
                "# Semantic Model Documentation",
                "",
                f"**Dialect:** {semantic_model.get('audit', {}).get('dialect', 'N/A')}",
                "",
                "## Table of Contents",
                "",
                "- [Facts](#facts)",
                "- [Dimensions](#dimensions)",
                "- [Entities](#entities)",
                "- [Relationships](#relationships)",
                "- [Metrics](#metrics)",
                "",
                "---",
                ""
            ])
            
            # Facts
            lines.extend(MarkdownExporter._document_facts(semantic_model))
            
            # Dimensions
            lines.extend(MarkdownExporter._document_dimensions(semantic_model))
            
            # Entities
            lines.extend(MarkdownExporter._document_entities(semantic_model))
            
            # Relationships
            lines.extend(MarkdownExporter._document_relationships(semantic_model))
            
            # Metrics
            lines.extend(MarkdownExporter._document_metrics(semantic_model))
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"✅ Markdown documentation exported: {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export Markdown: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _document_facts(semantic_model: Dict[str, Any]) -> List[str]:
        """Document fact tables."""
        lines = [
            "## Facts",
            "",
            "Fact tables contain transactional/event data with measures.",
            ""
        ]
        
        for fact in semantic_model.get("facts", []):
            lines.extend([
                f"### {fact['name']}",
                "",
                f"**Source:** `{fact['source']}`",
                f"**Grain:** {', '.join(fact.get('grain', []))}",
                ""
            ])
            
            # Columns
            if fact.get("columns"):
                lines.append("**Columns:**")
                lines.append("")
                lines.append("| Column | Type | Role | Description |")
                lines.append("|--------|------|------|-------------|")
                
                for col in fact.get("columns", []):
                    lines.append(
                        f"| {col['name']} | {col.get('type', 'N/A')} | "
                        f"{col.get('semantic_role', 'N/A')} | "
                        f"{col.get('description', 'N/A')} |"
                    )
                
                lines.append("")
            
            # Measures
            if fact.get("measures"):
                lines.append("**Measures:**")
                lines.append("")
                
                for measure in fact.get("measures", []):
                    lines.extend([
                        f"- **{measure['name']}**",
                        f"  - Expression: `{measure.get('expression', 'N/A')}`",
                        f"  - Format: {measure.get('format', 'N/A')}",
                        f"  - Depends on: {', '.join(measure.get('depends_on', []))}",
                    ])
                    
                    if measure.get("filters_applied"):
                        lines.append(f"  - Filters: {', '.join(measure['filters_applied'])}")
                    
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return lines
    
    @staticmethod
    def _document_dimensions(semantic_model: Dict[str, Any]) -> List[str]:
        """Document dimension tables."""
        lines = [
            "## Dimensions",
            "",
            "Dimension tables provide descriptive attributes for analysis.",
            ""
        ]
        
        for dimension in semantic_model.get("dimensions", []):
            lines.extend([
                f"### {dimension['name']}",
                "",
                f"**Source:** `{dimension['source']}`",
                f"**Keys:** {', '.join(dimension.get('keys', []))}",
                f"**Attributes:** {', '.join(dimension.get('attributes', []))}",
                "",
                "---",
                ""
            ])
        
        return lines
    
    @staticmethod
    def _document_entities(semantic_model: Dict[str, Any]) -> List[str]:
        """Document entity tables."""
        lines = [
            "## Entities",
            "",
            "Entity tables represent core business objects.",
            ""
        ]
        
        for entity in semantic_model.get("entities", []):
            lines.extend([
                f"### {entity['name']}",
                "",
                f"**Source:** `{entity['source']}`",
                f"**Primary Key:** {', '.join(entity.get('primary_key', []))}",
                f"**Description:** {entity.get('description', 'N/A')}",
                "",
                "---",
                ""
            ])
        
        return lines
    
    @staticmethod
    def _document_relationships(semantic_model: Dict[str, Any]) -> List[str]:
        """Document relationships."""
        lines = [
            "## Relationships",
            "",
            "| From | To | Cardinality | Type |",
            "|------|----|-----------|----|"
        ]
        
        for rel in semantic_model.get("relationships", []):
            lines.append(
                f"| {rel.get('from', 'N/A')} | {rel.get('to', 'N/A')} | "
                f"{rel.get('cardinality', 'N/A')} | {rel.get('type', 'N/A')} |"
            )
        
        lines.extend(["", "---", ""])
        
        return lines
    
    @staticmethod
    def _document_metrics(semantic_model: Dict[str, Any]) -> List[str]:
        """Document metrics."""
        lines = [
            "## Metrics",
            "",
            "Business metrics derived from measures.",
            ""
        ]
        
        for metric in semantic_model.get("metrics", []):
            lines.extend([
                f"### {metric['name']}",
                "",
                f"**Purpose:** {metric.get('purpose', 'N/A')}",
                f"**Logic:** {metric.get('logic', 'N/A')}",
                f"**Inputs:** {', '.join(metric.get('inputs', []))}",
                f"**Constraints:** {', '.join(metric.get('constraints', []))}",
                "",
                f"_{metric.get('explain', '')}_",
                "",
                "---",
                ""
            ])
        
        return lines