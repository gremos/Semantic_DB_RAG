"""
Extracts relationships from SQL views, stored procedures, and RDLs
to supplement FK-based relationships with inferred ones.
"""

import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import sqlglot
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferredRelationship:
    """Represents a relationship inferred from SQL or RDL"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: float  # 0.0 to 1.0
    source_type: str  # 'view_join', 'sp_join', 'rdl_lookup', 'naming_pattern'
    source_name: str  # Name of the view/SP/RDL


class RelationshipExtractor:
    """Extracts relationships from various SQL sources"""
    
    def __init__(self, dialect: str = "mssql"):
        self.dialect = dialect
        
    def extract_from_view(self, view_name: str, view_sql: str) -> List[InferredRelationship]:
        """
        Parse view SQL to extract JOIN relationships.
        
        Example:
        SELECT * FROM dbo.Customer c
        INNER JOIN dbo.Orders o ON c.CustomerID = o.CustomerID
        
        Returns: InferredRelationship(from_table='dbo.Orders', from_column='CustomerID',
                                       to_table='dbo.Customer', to_column='CustomerID',
                                       confidence=0.9, source_type='view_join', source_name=view_name)
        """
        relationships = []
        
        try:
            parsed = sqlglot.parse_one(view_sql, read=self.dialect)
            
            # Extract all JOIN conditions
            for join in parsed.find_all(sqlglot.exp.Join):
                on_clause = join.args.get("on")
                if not on_clause:
                    continue
                    
                # Parse equality conditions: table1.col = table2.col
                if isinstance(on_clause, sqlglot.exp.EQ):
                    left = on_clause.left
                    right = on_clause.right
                    
                    # Extract table.column from both sides
                    left_table, left_col = self._extract_table_column(left, parsed)
                    right_table, right_col = self._extract_table_column(right, parsed)
                    
                    if all([left_table, left_col, right_table, right_col]):
                        # Higher confidence for explicit JOINs
                        relationships.append(InferredRelationship(
                            from_table=left_table,
                            from_column=left_col,
                            to_table=right_table,
                            to_column=right_col,
                            confidence=0.9,
                            source_type='view_join',
                            source_name=view_name
                        ))
                        
        except Exception as e:
            logger.warning(f"Failed to parse view {view_name}: {e}")
            
        return relationships
    
    def extract_from_stored_procedure(self, sp_name: str, sp_sql: str) -> List[InferredRelationship]:
        """Extract relationships from stored procedure JOINs"""
        # Similar logic to extract_from_view
        return self.extract_from_view(sp_name, sp_sql)
    
    def extract_from_rdl(self, rdl_path: str, rdl_datasets: List[Dict]) -> List[InferredRelationship]:
        """
        Extract relationships from RDL dataset definitions.
        
        RDL datasets often have CommandText with JOINs or lookup relationships.
        """
        relationships = []
        
        for dataset in rdl_datasets:
            command_text = dataset.get('command_text', '')
            if command_text:
                # Parse SQL from RDL dataset
                dataset_name = dataset.get('name', 'unknown')
                relationships.extend(self.extract_from_view(
                    f"RDL:{rdl_path}:{dataset_name}",
                    command_text
                ))
        
        return relationships
    
    def _extract_table_column(self, node, parsed) -> tuple[Optional[str], Optional[str]]:
        """
        Extract table and column from a sqlglot node.
        
        Handles: table.column, alias.column
        """
        if not isinstance(node, sqlglot.exp.Column):
            return None, None
            
        column_name = node.name
        table_ref = node.table
        
        if not table_ref:
            return None, None
        
        # Resolve alias to actual table name
        table_name = self._resolve_alias(table_ref, parsed)
        
        return table_name, column_name
    
    def _resolve_alias(self, alias: str, parsed) -> Optional[str]:
        """Resolve table alias to actual table name from FROM/JOIN clauses"""
        # Find all FROM and JOIN clauses
        for table in parsed.find_all(sqlglot.exp.Table):
            table_alias = table.alias
            if table_alias and table_alias.lower() == alias.lower():
                return table.name
        
        # If no alias found, assume it's the actual table name
        return alias
    
    def deduplicate_and_merge(
        self,
        fk_relationships: List[Dict],
        inferred_relationships: List[InferredRelationship]
    ) -> List[Dict]:
        """
        Merge FK-based relationships with inferred ones.
        
        Priority:
        1. FK relationships (confidence = 1.0)
        2. View JOINs (confidence = 0.9)
        3. SP JOINs (confidence = 0.8)
        4. RDL lookups (confidence = 0.7)
        5. Naming patterns (confidence = 0.6)
        """
        merged = []
        seen = set()
        
        # Add FK relationships first (highest confidence)
        for fk in fk_relationships:
            key = (fk['from'], fk['to'])
            if key not in seen:
                fk['confidence'] = 1.0
                fk['source_type'] = 'foreign_key'
                merged.append(fk)
                seen.add(key)
        
        # Add inferred relationships if not already present
        for inferred in sorted(inferred_relationships, key=lambda x: -x.confidence):
            from_key = f"{inferred.from_table}.{inferred.from_column}"
            to_key = f"{inferred.to_table}.{inferred.to_column}"
            key = (from_key, to_key)
            
            if key not in seen:
                merged.append({
                    'from': from_key,
                    'to': to_key,
                    'cardinality': 'many-to-one',  # Default assumption
                    'type': inferred.source_type,
                    'confidence': inferred.confidence,
                    'source': inferred.source_name,
                    'business_meaning': f"Inferred from {inferred.source_type} in {inferred.source_name}"
                })
                seen.add(key)
        
        return merged


class UsageFrequencyAnalyzer:
    """Analyze how frequently tables are used in views/SPs/RDLs"""
    
    def __init__(self):
        self.table_usage: Dict[str, int] = {}
        
    def analyze_discovery(self, discovery_json: Dict) -> Dict[str, int]:
        """
        Count how many views/SPs/RDLs reference each table.
        
        Returns: {table_name: usage_count}
        """
        table_usage = {}
        
        # Analyze named assets
        for asset in discovery_json.get('named_assets', []):
            sql = asset.get('sql_normalized', '')
            if not sql:
                continue
                
            # Parse SQL and find all referenced tables
            try:
                parsed = sqlglot.parse_one(sql, read=discovery_json.get('dialect', 'mssql'))
                for table in parsed.find_all(sqlglot.exp.Table):
                    table_name = self._normalize_table_name(table.name, table.db)
                    table_usage[table_name] = table_usage.get(table_name, 0) + 1
                    
            except Exception as e:
                logger.warning(f"Failed to parse {asset.get('name')}: {e}")
        
        self.table_usage = table_usage
        return table_usage
    
    def _normalize_table_name(self, table: str, schema: Optional[str] = None) -> str:
        """Normalize to schema.table format"""
        if schema:
            return f"{schema}.{table}"
        return table
    
    def get_usage_score(self, table_name: str) -> float:
        """
        Get normalized usage score (0.0 to 1.0) for a table.
        
        Higher score = more frequently used in views/SPs/RDLs
        """
        if not self.table_usage:
            return 0.5  # Neutral score if no usage data
        
        max_usage = max(self.table_usage.values()) if self.table_usage else 1
        usage = self.table_usage.get(table_name, 0)
        
        return min(usage / max_usage, 1.0)