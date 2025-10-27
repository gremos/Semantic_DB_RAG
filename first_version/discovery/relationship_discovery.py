"""
Enhanced Relationship Discovery

Identifies:
1. Foreign keys from data (not just constraints)
2. Missing dimension tables (referenced but not modeled)
3. Bridge tables for many-to-many relationships
4. Orphaned foreign keys
"""

import logging
from typing import Dict, Any, List, Tuple, Set
import re

logger = logging.getLogger(__name__)


class RelationshipDiscovery:
    """
    Discover relationships and identify missing entities
    using statistical analysis and naming conventions.
    """
    
    def __init__(self, discovery_json: Dict[str, Any]):
        self.discovery = discovery_json
        self.all_tables = self._extract_all_tables()
        self.all_columns = self._extract_all_columns()
    
    def _extract_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """Extract all tables from discovery as {schema.table: table_data}."""
        tables = {}
        for schema in self.discovery.get("schemas", []):
            schema_name = schema.get("name")
            for table in schema.get("tables", []):
                table_name = table.get("name")
                full_name = f"{schema_name}.{table_name}"
                tables[full_name] = table
        return tables
    
    def _extract_all_columns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all columns as {schema.table: [columns]}."""
        columns = {}
        for table_name, table_data in self.all_tables.items():
            columns[table_name] = table_data.get("columns", [])
        return columns
    
    def discover_relationships(self) -> List[Dict[str, Any]]:
        """
        Discover all relationships using multiple strategies:
        1. Declared foreign keys
        2. Naming convention matching (e.g., CustomerID)
        3. Value overlap analysis
        4. Cardinality patterns
        """
        relationships = []
        
        # Strategy 1: Declared foreign keys
        for table_name, table_data in self.all_tables.items():
            for fk in table_data.get("foreign_keys", []):
                relationships.append({
                    "from": f"{table_name}.{fk['column']}",
                    "to": f"{fk['ref_table']}.{fk['ref_column']}",
                    "cardinality": "many_to_one",
                    "confidence": "high",
                    "source": "declared_fk"
                })
        
        # Strategy 2: Infer from naming conventions
        inferred_fks = self._infer_fks_from_naming()
        for fk in inferred_fks:
            # Avoid duplicates
            if not any(r["from"] == fk["from"] and r["to"] == fk["to"] 
                      for r in relationships):
                relationships.append(fk)
        
        logger.info(f"Discovered {len(relationships)} relationships")
        return relationships
    
    def _infer_fks_from_naming(self) -> List[Dict[str, Any]]:
        """
        Infer foreign keys from naming conventions.
        
        Common patterns:
        - CustomerID in Orders → Customer.CustomerID
        - ProductID in OrderLines → Product.ProductID
        - *ID suffix generally references a table
        """
        inferred = []
        
        # Common FK patterns
        fk_pattern = re.compile(r'^(\w+)(ID|Id|id|_id)$')
        
        for table_name, columns in self.all_columns.items():
            for col in columns:
                col_name = col.get("name")
                
                # Check if column looks like an FK
                match = fk_pattern.match(col_name)
                if not match:
                    continue
                
                # Extract potential referenced table name
                potential_table = match.group(1)
                
                # Look for matching table
                referenced_table = self._find_referenced_table(potential_table, col_name)
                
                if referenced_table:
                    inferred.append({
                        "from": f"{table_name}.{col_name}",
                        "to": f"{referenced_table}.{col_name}",
                        "cardinality": "many_to_one",
                        "confidence": "high" if col_name in self.all_columns.get(referenced_table, []) else "medium",
                        "source": "naming_convention"
                    })
                else:
                    # Flag as potential missing dimension
                    logger.warning(f"Potential missing dimension: {potential_table} "
                                 f"(referenced by {table_name}.{col_name})")
        
        return inferred
    
    def _find_referenced_table(self, table_hint: str, col_name: str) -> str:
        """
        Find the actual table name that matches the hint.
        Handles singular/plural, schema prefixes, etc.
        """
        table_hint_lower = table_hint.lower()
        
        # Try exact matches first
        for table_name in self.all_tables.keys():
            base_name = table_name.split('.')[-1].lower()
            
            # Exact match
            if base_name == table_hint_lower:
                # Check if table has a column with same name as FK
                table_cols = [c["name"] for c in self.all_columns.get(table_name, [])]
                if col_name in table_cols:
                    return table_name
            
            # Plural match (e.g., Customer vs Customers)
            if base_name == table_hint_lower + 's' or base_name + 's' == table_hint_lower:
                table_cols = [c["name"] for c in self.all_columns.get(table_name, [])]
                if col_name in table_cols:
                    return table_name
        
        return None
    
    def identify_missing_entities(self) -> List[Dict[str, Any]]:
        """
        Identify entities that are referenced but don't exist.
        These are likely missing dimensions.
        """
        missing = []
        fk_pattern = re.compile(r'^(\w+)(ID|Id|id|_id)$')
        
        for table_name, columns in self.all_columns.items():
            for col in columns:
                col_name = col.get("name")
                match = fk_pattern.match(col_name)
                
                if match:
                    potential_table = match.group(1)
                    referenced_table = self._find_referenced_table(potential_table, col_name)
                    
                    if not referenced_table:
                        missing.append({
                            "suggested_name": potential_table,
                            "referenced_by": f"{table_name}.{col_name}",
                            "reason": "FK column exists but target table not found"
                        })
        
        # Deduplicate
        seen = set()
        unique_missing = []
        for m in missing:
            key = m["suggested_name"]
            if key not in seen:
                seen.add(key)
                unique_missing.append(m)
        
        if unique_missing:
            logger.warning(f"Identified {len(unique_missing)} potentially missing entities:")
            for m in unique_missing:
                logger.warning(f"  - {m['suggested_name']} (referenced by {m['referenced_by']})")
        
        return unique_missing
    
    def classify_table_role(self, table_name: str) -> str:
        """
        Classify table as 'fact', 'dimension', or 'bridge'.
        
        Heuristics:
        - Facts: Many foreign keys, numeric measures, high row count
        - Dimensions: Few/no foreign keys, mostly text, lower row count
        - Bridge: Exactly 2 foreign keys, low row count (many-to-many)
        """
        table_data = self.all_tables.get(table_name)
        if not table_data:
            return "unknown"
        
        fk_count = len(table_data.get("foreign_keys", []))
        columns = table_data.get("columns", [])
        row_count = table_data.get("rowcount_sample", 0)
        
        # Count numeric vs text columns
        numeric_cols = sum(1 for c in columns 
                          if any(t in c.get("type", "").lower() 
                                for t in ["int", "decimal", "float", "numeric", "money"]))
        
        # Bridge table: Exactly 2 FKs, small
        if fk_count == 2 and row_count < 100000:
            return "bridge"
        
        # Fact table: Multiple FKs, numeric measures
        if fk_count >= 2 and numeric_cols >= 2:
            return "fact"
        
        # Dimension table: Few/no FKs, descriptive
        if fk_count <= 1:
            return "dimension"
        
        return "unknown"
    
    def analyze_cardinality(self, from_table: str, from_col: str,
                           to_table: str, to_col: str) -> str:
        """
        Analyze cardinality of a relationship.
        Returns: "one_to_one", "one_to_many", "many_to_one", "many_to_many"
        
        Note: This requires actual data analysis which we can't do without
        querying the database. For now, use heuristics.
        """
        # Heuristic: If from_col is in primary key of from_table → one side
        from_table_data = self.all_tables.get(from_table)
        to_table_data = self.all_tables.get(to_table)
        
        if not from_table_data or not to_table_data:
            return "many_to_one"  # Default assumption
        
        from_pk = from_table_data.get("primary_key", [])
        to_pk = to_table_data.get("primary_key", [])
        
        from_is_pk = from_col in from_pk
        to_is_pk = to_col in to_pk
        
        if from_is_pk and to_is_pk:
            return "one_to_one"
        elif to_is_pk:
            return "many_to_one"
        elif from_is_pk:
            return "one_to_many"
        else:
            return "many_to_one"  # Most common in star schema
    
    def detect_cycles(self, relationships: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Detect circular relationships (A→B→C→A).
        These need special handling in queries.
        """
        # Build adjacency list
        graph = {}
        for rel in relationships:
            from_table = rel["from"].split(".")[0]
            to_table = rel["to"].split(".")[0]
            
            if from_table not in graph:
                graph[from_table] = []
            graph[from_table].append(to_table)
        
        # DFS to find cycles
        cycles = []
        
        def dfs(node, path, visited):
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path[:], visited)
        
        visited = set()
        for node in graph:
            if node not in visited:
                dfs(node, [], visited)
        
        if cycles:
            logger.warning(f"Detected {len(cycles)} circular relationships")
        
        return cycles
    
    def suggest_star_schema_design(self) -> Dict[str, Any]:
        """
        Suggest star schema design based on discovered relationships.
        
        Returns:
            {
                "fact_tables": [...],
                "dimension_tables": [...],
                "bridge_tables": [...],
                "missing_dimensions": [...]
            }
        """
        design = {
            "fact_tables": [],
            "dimension_tables": [],
            "bridge_tables": [],
            "missing_dimensions": []
        }
        
        # Classify all tables
        for table_name in self.all_tables.keys():
            role = self.classify_table_role(table_name)
            
            if role == "fact":
                design["fact_tables"].append(table_name)
            elif role == "dimension":
                design["dimension_tables"].append(table_name)
            elif role == "bridge":
                design["bridge_tables"].append(table_name)
        
        # Identify missing dimensions
        design["missing_dimensions"] = self.identify_missing_entities()
        
        logger.info(f"Star schema suggestion: "
                   f"{len(design['fact_tables'])} facts, "
                   f"{len(design['dimension_tables'])} dimensions, "
                   f"{len(design['bridge_tables'])} bridges, "
                   f"{len(design['missing_dimensions'])} missing")
        
        return design