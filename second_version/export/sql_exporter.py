"""
Export semantic model as SQL DDL (CREATE VIEW statements).
These views can be used directly in Power BI, Tableau, or any SQL tool.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SQLExporter:
    """Export semantic model as SQL views."""
    
    @staticmethod
    def export(semantic_model: Dict[str, Any], output_file: str = "semantic_views.sql") -> bool:
        """
        Export semantic model as SQL view definitions.
        
        Args:
            semantic_model: Semantic model JSON
            output_file: Output filename (.sql extension)
        
        Returns:
            Success boolean
        """
        try:
            logger.info(f"Exporting to SQL views: {output_file}")
            
            dialect = semantic_model.get("audit", {}).get("dialect", "tsql")
            sql_statements = []
            
            # Header
            sql_statements.append("-- Semantic Model SQL Views")
            sql_statements.append(f"-- Generated from semantic model")
            sql_statements.append(f"-- Dialect: {dialect}")
            sql_statements.append("")
            
            # Create schema for views
            sql_statements.append("-- Create semantic schema")
            sql_statements.append("IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'semantic')")
            sql_statements.append("    EXEC('CREATE SCHEMA semantic');")
            sql_statements.append("GO")
            sql_statements.append("")
            
            # Create views for facts (with measures)
            for fact in semantic_model.get("facts", []):
                sql_statements.extend(SQLExporter._create_fact_view(fact, dialect))
                sql_statements.append("")
            
            # Create views for dimensions
            for dimension in semantic_model.get("dimensions", []):
                sql_statements.extend(SQLExporter._create_dimension_view(dimension, dialect))
                sql_statements.append("")
            
            # Create aggregate views (one per metric)
            for metric in semantic_model.get("metrics", []):
                sql_statements.extend(SQLExporter._create_metric_view(metric, semantic_model, dialect))
                sql_statements.append("")
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sql_statements))
            
            logger.info(f"✅ SQL views exported: {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export SQL views: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _create_fact_view(fact: Dict[str, Any], dialect: str) -> List[str]:
        """Create view for fact table with computed measures."""
        view_name = f"semantic.{fact['name']}"
        source = fact["source"]
        
        sql = [
            f"-- Fact: {fact['name']}",
            f"-- Source: {source}",
            f"CREATE OR ALTER VIEW {view_name} AS",
            "SELECT"
        ]
        
        # Add all columns
        columns = fact.get("columns", [])
        col_lines = []
        
        for col in columns:
            comment = f"-- {col.get('description', '')}" if col.get('description') else ""
            col_lines.append(f"    {col['name']} {comment}")
        
        # Add computed measures as columns
        for measure in fact.get("measures", []):
            # Don't add aggregates to row-level view, just document them
            col_lines.append(f"    -- Measure: {measure['name']} = {measure['expression']}")
        
        sql.append(",\n".join(col_lines))
        sql.append(f"FROM {source}")
        
        # Add WHERE clause for status filters (if any)
        filters = []
        for col in columns:
            if col.get("semantic_role") == "status_indicator":
                desc = col.get("description", "")
                if "NULL" in desc and "active" in desc.lower():
                    filters.append(f"{col['name']} IS NULL")
        
        if filters:
            sql.append("WHERE")
            sql.append(f"    {' AND '.join(filters)}")
            sql.append("    -- Filter for active records only")
        
        sql.append("GO")
        
        return sql
    
    @staticmethod
    def _create_dimension_view(dimension: Dict[str, Any], dialect: str) -> List[str]:
        """Create view for dimension table."""
        view_name = f"semantic.{dimension['name']}"
        source = dimension["source"]
        
        sql = [
            f"-- Dimension: {dimension['name']}",
            f"-- Source: {source}",
            f"CREATE OR ALTER VIEW {view_name} AS",
            "SELECT"
        ]
        
        # Add all columns
        columns = dimension.get("columns", [])
        col_lines = []
        
        for col in columns:
            comment = f"-- {col.get('description', '')}" if col.get('description') else ""
            col_lines.append(f"    {col['name']} {comment}")
        
        sql.append(",\n".join(col_lines))
        sql.append(f"FROM {source}")
        sql.append("GO")
        
        return sql
    
    @staticmethod
    def _create_metric_view(metric: Dict[str, Any], semantic_model: Dict[str, Any], dialect: str) -> List[str]:
        """Create aggregated view for metric."""
        view_name = f"semantic.Metric_{metric['name'].replace(' ', '_')}"
        
        sql = [
            f"-- Metric: {metric['name']}",
            f"-- Purpose: {metric.get('purpose', '')}",
            f"-- Logic: {metric.get('logic', '')}",
            f"CREATE OR ALTER VIEW {view_name} AS"
        ]
        
        # This is simplified - real implementation would parse metric logic
        sql.append("-- TODO: Implement metric aggregation")
        sql.append(f"-- Inputs: {', '.join(metric.get('inputs', []))}")
        sql.append(f"-- Constraints: {', '.join(metric.get('constraints', []))}")
        sql.append("SELECT 1 AS PlaceholderMetric")
        sql.append("GO")
        
        return sql