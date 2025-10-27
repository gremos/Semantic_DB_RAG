"""
Extract concrete relationships from views, stored procedures, and RDLs.
These relationships are MORE authoritative than inferred FK patterns.
"""

import re
import sqlglot
from typing import List, Dict, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class SemanticRelationship:
    """A relationship derived from actual SQL usage, not just schema."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    source_asset: str  # e.g., "view:dbo.vCustomerSales"
    join_type: str  # INNER, LEFT, RIGHT
    confidence: float  # 0.0-1.0, based on source authority
    usage_count: int = 1  # how many views/SPs use this join


class SemanticRelationshipExtractor:
    """Extract relationships from views, stored procedures, and RDLs."""
    
    def __init__(self, dialect: str = "mssql"):
        self.dialect = dialect
        self.relationships: Dict[str, SemanticRelationship] = {}
    
    def extract_from_discovery(self, discovery_json: dict) -> List[SemanticRelationship]:
        """Main entry point: extract from all named assets."""
        
        # Process views and stored procedures
        for asset in discovery_json.get("named_assets", []):
            if asset["kind"] in ["view", "stored_procedure"]:
                self._extract_from_sql(
                    sql=asset["sql_normalized"],
                    source_name=asset["name"],
                    source_kind=asset["kind"]
                )
        
        # Process RDL datasets
        for asset in discovery_json.get("named_assets", []):
            if asset["kind"] == "rdl":
                self._extract_from_rdl(
                    rdl_path=asset["path"],
                    datasets=asset.get("datasets", [])
                )
        
        # Deduplicate and rank by confidence
        return self._deduplicate_and_rank()
    
    def _extract_from_sql(self, sql: str, source_name: str, source_kind: str):
        """Parse JOIN clauses from normalized SQL."""
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Find all JOIN expressions
            for join in parsed.find_all(sqlglot.exp.Join):
                left_table = self._extract_table_name(join.this)
                right_table = self._extract_table_name(join.expression)
                
                # Extract ON condition
                if join.args.get("on"):
                    on_clause = join.args["on"]
                    left_col, right_col = self._extract_join_columns(on_clause)
                    
                    if left_col and right_col:
                        join_type = join.side or "INNER"  # Default to INNER
                        
                        # Confidence based on source authority
                        confidence = 0.9 if source_kind == "view" else 0.7
                        
                        rel_key = f"{left_table}.{left_col}->{right_table}.{right_col}"
                        
                        if rel_key in self.relationships:
                            # Increment usage count
                            self.relationships[rel_key].usage_count += 1
                            self.relationships[rel_key].confidence = min(
                                1.0, 
                                self.relationships[rel_key].confidence + 0.05
                            )
                        else:
                            self.relationships[rel_key] = SemanticRelationship(
                                from_table=left_table,
                                from_column=left_col,
                                to_table=right_table,
                                to_column=right_col,
                                source_asset=f"{source_kind}:{source_name}",
                                join_type=join_type,
                                confidence=confidence
                            )
        
        except Exception as e:
            print(f"Failed to parse SQL from {source_name}: {e}")
    
    def _extract_table_name(self, node) -> Optional[str]:
        """Extract fully qualified table name from AST node."""
        if isinstance(node, sqlglot.exp.Table):
            schema = node.args.get("db", {}).get("this") if node.args.get("db") else None
            table = node.args.get("this", {}).get("this") if node.args.get("this") else None
            
            if schema and table:
                return f"{schema}.{table}"
            return table
        return None
    
    def _extract_join_columns(self, on_clause) -> tuple:
        """Extract left and right columns from ON condition."""
        # Look for equality: table1.col1 = table2.col2
        if isinstance(on_clause, sqlglot.exp.EQ):
            left = on_clause.left
            right = on_clause.right
            
            left_col = self._column_to_string(left)
            right_col = self._column_to_string(right)
            
            return left_col, right_col
        
        return None, None
    
    def _column_to_string(self, col_expr) -> Optional[str]:
        """Convert column expression to string."""
        if isinstance(col_expr, sqlglot.exp.Column):
            return col_expr.sql(dialect=self.dialect)
        return None
    
    def _extract_from_rdl(self, rdl_path: str, datasets: List[str]):
        """Extract relationships from RDL dataset queries."""
        try:
            tree = ET.parse(rdl_path)
            root = tree.getroot()
            
            # RDL namespace handling
            ns = {"rdl": "http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition"}
            
            for dataset in root.findall(".//rdl:DataSet", ns):
                query_elem = dataset.find(".//rdl:CommandText", ns)
                if query_elem is not None and query_elem.text:
                    self._extract_from_sql(
                        sql=query_elem.text,
                        source_name=f"rdl:{rdl_path}",
                        source_kind="rdl"
                    )
        
        except Exception as e:
            print(f"Failed to parse RDL {rdl_path}: {e}")
    
    def _deduplicate_and_rank(self) -> List[SemanticRelationship]:
        """Return relationships sorted by confidence and usage."""
        rels = list(self.relationships.values())
        
        # Sort by: confidence DESC, usage_count DESC
        rels.sort(key=lambda r: (r.confidence, r.usage_count), reverse=True)
        
        return rels


def enhance_discovery_with_semantic_relationships(discovery_json: dict) -> dict:
    """
    Add a new section to discovery JSON with extracted semantic relationships.
    This runs AFTER Phase 1 discovery but BEFORE Phase 2 modeling.
    """
    extractor = SemanticRelationshipExtractor(dialect=discovery_json.get("dialect", "mssql"))
    
    semantic_rels = extractor.extract_from_discovery(discovery_json)
    
    # Add to discovery JSON
    discovery_json["semantic_relationships"] = [
        {
            "from": f"{r.from_table}.{r.from_column}",
            "to": f"{r.to_table}.{r.to_column}",
            "join_type": r.join_type,
            "confidence": r.confidence,
            "usage_count": r.usage_count,
            "source_asset": r.source_asset
        }
        for r in semantic_rels
    ]
    
    return discovery_json