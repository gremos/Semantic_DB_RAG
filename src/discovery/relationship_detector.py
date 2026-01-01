"""
Relationship Detection Module

Detects relationships between database tables using multiple strategies:
1. Explicit foreign keys (schema-defined) - HIGHEST CONFIDENCE
2. Curated asset JOINs (views/SPs/RDLs) - HIGH CONFIDENCE  
3. Implicit value overlap with name matching - MEDIUM/LOW CONFIDENCE

Priority:
- Explicit > Curated > Implicit
- Relationships are deduplicated and ranked by confidence
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
from difflib import SequenceMatcher
import sqlglot
from sqlglot import parse_one, exp

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def detect_relationships(
    discovery_data: Dict[str, Any],
    overlap_threshold: float = 0.98,  # ✅ STRICTER: 98% minimum
    name_similarity_threshold: float = 0.7,  # ✅ STRICTER: 70% name match
    sample_size: int = 1000,
    min_cardinality_ratio: float = 3.0  # ✅ STRICTER: 3x difference for many-to-one
) -> List[Dict[str, Any]]:
    """
    Detect relationships between tables using multiple strategies
    
    Priority order:
    1. Explicit foreign keys (schema-defined) - confidence='very_high'
    2. Curated asset JOINs (views/SPs/RDLs) - confidence='high'
    3. Implicit value overlap with name matching - confidence='medium'/'low'
    
    Args:
        discovery_data: Discovery JSON from Phase 1
        overlap_threshold: Minimum overlap rate for implicit relationships (default: 0.95)
        name_similarity_threshold: Minimum name similarity for implicit rels (default: 0.6)
        sample_size: Sample size for overlap detection
        min_cardinality_ratio: Minimum ratio for many-to-one detection (default: 2.0)
        
    Returns:
        List of deduplicated relationship definitions with confidence scores
    """
    relationships = []
    
    logger.info("=" * 80)
    logger.info("RELATIONSHIP DETECTION")
    logger.info("=" * 80)
    
    # Strategy 1: Explicit foreign keys from schema
    logger.info("Strategy 1: Extracting explicit foreign keys from schema...")
    explicit_rels = _extract_explicit_foreign_keys(discovery_data)
    relationships.extend(explicit_rels)
    logger.info(f"  ✓ Found {len(explicit_rels)} explicit foreign key relationships")
    
    # Strategy 2: Extract JOINs from curated assets (NEW - HIGH CONFIDENCE)
    logger.info("Strategy 2: Extracting relationships from curated assets (views/SPs/RDLs)...")
    curated_rels = _extract_relationships_from_curated_assets(discovery_data)
    relationships.extend(curated_rels)
    logger.info(f"  ✓ Found {len(curated_rels)} relationships from curated assets")
    
    # Strategy 3: Implicit relationships via value overlap (LOWEST CONFIDENCE)
    logger.info("Strategy 3: Detecting implicit relationships via value overlap...")
    implicit_rels = _detect_implicit_relationships(
        discovery_data,
        overlap_threshold=overlap_threshold,
        name_similarity_threshold=name_similarity_threshold,
        sample_size=sample_size,
        min_cardinality_ratio=min_cardinality_ratio
    )
    relationships.extend(implicit_rels)
    logger.info(f"  ✓ Found {len(implicit_rels)} implicit relationships")
    
    # Deduplicate and rank by confidence
    logger.info("Deduplicating and ranking relationships...")
    relationships = _deduplicate_and_rank_relationships(relationships)
    
    # Log summary by confidence
    confidence_counts = defaultdict(int)
    for rel in relationships:
        confidence_counts[rel["confidence"]] += 1
    
    logger.info("=" * 80)
    logger.info(f"RELATIONSHIP DETECTION COMPLETE")
    logger.info(f"  Total relationships: {len(relationships)}")
    logger.info(f"  By confidence:")
    for conf in ["very_high", "high", "medium", "low"]:
        if conf in confidence_counts:
            logger.info(f"    {conf}: {confidence_counts[conf]}")
    logger.info("=" * 80)
    
    return relationships


# ============================================================================
# STRATEGY 1: EXPLICIT FOREIGN KEYS
# ============================================================================

def _extract_explicit_foreign_keys(
    discovery_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extract explicit foreign key relationships from schema metadata
    
    These are defined in the database schema and are the most reliable.
    Confidence: 'very_high'
    """
    relationships = []
    
    for schema in discovery_data.get("schemas", []):
        schema_name = schema.get("name")
        
        for table in schema.get("tables", []):
            table_name = table.get("name")
            full_table_name = f"{schema_name}.{table_name}"
            
            for fk in table.get("foreign_keys", []):
                from_col = fk.get("column")
                ref_table = fk.get("ref_table")
                ref_col = fk.get("ref_column")
                
                if not all([from_col, ref_table, ref_col]):
                    continue
                
                relationships.append({
                    "from": f"{full_table_name}.{from_col}",
                    "to": f"{ref_table}.{ref_col}",
                    "cardinality": "many_to_one",  # FK is typically many-to-one
                    "confidence": "very_high",
                    "method": "explicit_foreign_key",
                    "overlap_rate": None,
                    "source_asset": None
                })
    
    return relationships


# ============================================================================
# STRATEGY 2: CURATED ASSET JOINS (Views/SPs/RDLs)
# ============================================================================

def _extract_relationships_from_curated_assets(
    discovery_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extract relationships from JOIN clauses in views, stored procedures, and RDL queries
    
    This provides HIGH CONFIDENCE relationships because they are explicitly written
    by domain experts and are known to work in production.
    
    Returns:
        List of relationship dicts with confidence='high'
    """
    relationships = []
    named_assets = discovery_data.get("named_assets", [])
    
    for asset in named_assets:
        asset_kind = asset.get("kind", "")
        asset_name = asset.get("name", "")
        sql_normalized = asset.get("sql_normalized", "")
        
        if not sql_normalized:
            continue
        
        # Skip if not a query-based asset
        if asset_kind not in ["view", "stored_procedure", "storedprocedure", "rdl"]:
            continue
        
        try:
            # Parse SQL to extract JOIN conditions
            parsed = parse_one(sql_normalized, read="tsql")
            
            # Find all JOIN nodes
            for join_node in parsed.find_all(exp.Join):
                join_on = join_node.args.get("on")
                
                if not join_on:
                    continue
                
                # Extract equality conditions from ON clause
                for condition in _extract_equality_conditions(join_on):
                    left_table, left_col = _parse_column_reference(condition["left"])
                    right_table, right_col = _parse_column_reference(condition["right"])
                    
                    if not all([left_table, left_col, right_table, right_col]):
                        continue
                    
                    # Determine cardinality hint from JOIN type
                    join_type = join_node.args.get("kind", "").upper() if join_node.args.get("kind") else "INNER"
                    cardinality_hint = _infer_cardinality_from_join_type(join_type)
                    
                    relationships.append({
                        "from": f"{left_table}.{left_col}",
                        "to": f"{right_table}.{right_col}",
                        "cardinality": cardinality_hint,
                        "confidence": "high",
                        "method": "curated_asset_join",
                        "source_asset": asset_name,
                        "source_kind": asset_kind,
                        "overlap_rate": None
                    })
                    
        except Exception as e:
            logger.debug(f"  Failed to parse SQL for {asset_name}: {e}")
            continue
    
    return relationships


def _extract_equality_conditions(on_clause) -> List[Dict[str, Any]]:
    """
    Extract equality conditions from JOIN ON clause
    
    Example: ON a.id = b.id AND a.type = 'X'
    Returns: [{"left": "a.id", "right": "b.id", "operator": "="}]
    
    Handles:
    - Simple equality: a.id = b.id
    - Compound conditions: a.id = b.id AND a.type = b.type
    - Nested conditions: (a.id = b.id) OR (a.alt_id = b.id)
    """
    conditions = []
    
    # Handle compound conditions (AND/OR)
    if isinstance(on_clause, (exp.And, exp.Or)):
        for child in on_clause.args.values():
            if isinstance(child, list):
                for item in child:
                    conditions.extend(_extract_equality_conditions(item))
            else:
                conditions.extend(_extract_equality_conditions(child))
    
    # Handle equality conditions
    elif isinstance(on_clause, exp.EQ):
        left = on_clause.args.get("this")
        right = on_clause.args.get("expression")
        
        # Only extract column = column (not column = literal)
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            conditions.append({
                "left": left.sql(),
                "right": right.sql(),
                "operator": "="
            })
    
    return conditions


def _parse_column_reference(col_ref: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse column reference into (table, column)

    Examples:
        "schema.table.column" -> ("schema.table", "column")
        "table.column" -> ("table", "column")
        "t1.column" -> ("t1", "column")
        "column" -> (None, "column")

    Returns:
        (table_name, column_name) or (None, column_name) if table can't be determined
    """
    # Remove quotes if present
    col_ref = col_ref.replace('"', '').replace('[', '').replace(']', '')

    parts = col_ref.split(".")

    if len(parts) >= 3:
        # schema.table.column
        return (".".join(parts[:-1]), parts[-1])
    elif len(parts) == 2:
        # table.column or alias.column
        return (parts[0], parts[1])
    else:
        # Just column name - can't determine table
        return (None, col_ref)


def _parse_join_condition(on_clause) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse JOIN ON clause to extract first equality condition tables/columns.

    Used by RDL parser to extract relationship info from JOIN conditions.

    Args:
        on_clause: sqlglot expression representing ON clause

    Returns:
        (left_table, left_col, right_table, right_col) or (None, None, None, None) if not parseable
    """
    conditions = _extract_equality_conditions(on_clause)

    if not conditions:
        return (None, None, None, None)

    # Use first equality condition
    cond = conditions[0]
    left_table, left_col = _parse_column_reference(cond["left"])
    right_table, right_col = _parse_column_reference(cond["right"])

    return (left_table, left_col, right_table, right_col)


def _infer_cardinality_from_join_type(join_type: str) -> str:
    """
    Infer cardinality hint from JOIN type
    
    This is a heuristic - not always accurate but better than nothing
    
    Examples:
        LEFT JOIN: many-to-one (many left rows -> one right row)
        RIGHT JOIN: one-to-many
        INNER JOIN: many-to-one (most common pattern)
    """
    join_type_upper = join_type.upper() if join_type else ""
    
    # LEFT/LEFT OUTER JOIN often implies many-to-one
    if "LEFT" in join_type_upper:
        return "many_to_one"
    
    # RIGHT/RIGHT OUTER JOIN implies one-to-many
    elif "RIGHT" in join_type_upper:
        return "one_to_many"
    
    # INNER JOIN - assume many-to-one (most common pattern)
    else:
        return "many_to_one"


# ============================================================================
# STRATEGY 3: IMPLICIT VALUE OVERLAP
# ============================================================================

def _detect_implicit_relationships(
    discovery_data: Dict[str, Any],
    overlap_threshold: float = 0.95,
    name_similarity_threshold: float = 0.6,
    sample_size: int = 1000,
    min_cardinality_ratio: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Detect implicit relationships via value overlap
    
    ENHANCED with:
    - Column name similarity requirement (reduces false positives)
    - Cardinality detection (many-to-one vs one-to-one)
    - Confidence scoring based on overlap rate AND name match
    
    Args:
        overlap_threshold: Minimum overlap rate (0.95 = 95%)
        name_similarity_threshold: Minimum column name similarity (0.6 = 60%)
        sample_size: Max rows to sample per table
        min_cardinality_ratio: Min ratio for many-to-one (default: 2.0)
        
    Returns:
        List of implicit relationships with confidence scores
    """
    relationships = []
    
    # Build index of all columns by table
    column_index = _build_column_index(discovery_data)
    
    # Find candidate column pairs (potential FKs)
    candidate_pairs = _find_candidate_column_pairs(
        column_index,
        name_similarity_threshold
    )
    
    logger.info(f"  Found {len(candidate_pairs)} candidate column pairs (after name filtering)")
    
    # Calculate overlap for each candidate pair
    overlaps = _calculate_value_overlaps(
        discovery_data,
        candidate_pairs,
        sample_size
    )
    
    logger.info(f"  Calculated overlap for {len(overlaps)} pairs")
    
    # Build relationships from overlaps
    for (col1_full, col2_full), stats in overlaps.items():
        overlap_rate = stats["overlap_rate"]
        name_similarity = stats["name_similarity"]
        cardinality = stats["cardinality"]
        sample_size1 = stats.get("sample_size1", 0)
        sample_size2 = stats.get("sample_size2", 0)
        
        # Filter by overlap threshold
        if overlap_rate < overlap_threshold:
            continue
        
        # ✅ CRITICAL FIX: Reject suspicious one-to-one relationships
        # These are likely false positives (different tables shouldn't have identical ID sets)
        # ✅ CRITICAL FIX: Reject suspicious one-to-one relationships
        # These are likely false positives (different tables shouldn't have identical ID sets)
        if cardinality == "one_to_one":
            # Extract table names
            table1 = ".".join(col1_full.split(".")[:-1])
            table2 = ".".join(col2_full.split(".")[:-1])
            
            # Extract column names
            col1_name = col1_full.split(".")[-1].lower()
            col2_name = col2_full.split(".")[-1].lower()
            
            # ✅ AGGRESSIVE: Reject ALL one-to-one unless:
            # 1. Tables have parent/child naming pattern (e.g., "Order" and "OrderDetail")
            # 2. Column names are highly similar (>0.8)
            # 3. One column is explicitly named after the other table (e.g., "CustomerID" in Orders)
            
            tables_related = (
                table1.lower() in table2.lower() or 
                table2.lower() in table1.lower()
            )
            
            # Check if column name matches table name (FK pattern)
            table1_name = table1.split('.')[-1].lower()
            table2_name = table2.split('.')[-1].lower()
            fk_pattern_match = (
                table1_name in col1_name or 
                table2_name in col2_name or
                table1_name in col2_name or
                table2_name in col1_name
            )
            
            high_name_similarity = name_similarity >= 0.8
            perfect_overlap = overlap_rate >= 0.995
            
            # ✅ Reject unless ALL criteria met
            if perfect_overlap and not (tables_related and fk_pattern_match and high_name_similarity):
                logger.debug(
                    f"  ❌ Rejecting 1:1 (unrelated tables): {col1_full} <-> {col2_full} "
                    f"(overlap={overlap_rate:.3f}, name_sim={name_similarity:.2f})"
                )
                continue
            
            # ✅ Additional check: Reject if generic ID columns from unrelated tables
            if col1_name == 'id' and col2_name == 'id' and not tables_related:
                logger.debug(f"  ❌ Rejecting generic ID columns from unrelated tables: {table1} <-> {table2}")
                continue
        
        # ✅ REQUIRED: Enforce many-to-one cardinality for implicit relationships
        # One-to-many is acceptable only if strong name pattern (e.g., ParentID -> Parent.ID)
        if cardinality == "one_to_many":
            # Check if column names suggest parent-child relationship
            col1_name = col1_full.split(".")[-1].lower()
            col2_name = col2_full.split(".")[-1].lower()
            table2_name = col2_full.split(".")[-2].lower()
            
            # Pattern: CustomerID -> Customer.ID
            has_name_pattern = (
                col1_name.replace("id", "").replace("_", "") == table2_name.replace("_", "") or
                col2_name == "id" and table2_name in col1_name
            )
            
            if not has_name_pattern:
                logger.debug(
                    f"  ⚠️ Skipping one-to-many without name pattern: {col1_full} -> {col2_full}"
                )
                continue
        
        # Determine confidence based on overlap rate AND name match AND cardinality
        if overlap_rate >= 0.98 and name_similarity >= 0.8 and cardinality == "many_to_one":
            confidence = "high"
        elif overlap_rate >= 0.95 and name_similarity >= 0.7:
            confidence = "medium"
        elif overlap_rate >= 0.90 and name_similarity >= 0.6:
            confidence = "low"
        else:
            # Too low confidence - skip
            continue
        
        relationships.append({
            "from": col1_full,
            "to": col2_full,
            "cardinality": cardinality,
            "confidence": confidence,
            "method": "value_overlap",
            "overlap_rate": overlap_rate,
            "name_similarity": name_similarity,
            "source_asset": None
        })
    
    return relationships


def _build_column_index(discovery_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build an index of all columns grouped by data type
    
    Returns:
        Dict mapping data_type -> list of column info
        Example: {"int": [{"full_name": "dbo.Orders.CustomerID", ...}]}
    """
    column_index = defaultdict(list)
    
    for schema in discovery_data.get("schemas", []):
        schema_name = schema.get("name")
        
        for table in schema.get("tables", []):
            table_name = table.get("name")
            full_table_name = f"{schema_name}.{table_name}"
            
            for col in table.get("columns", []):
                col_name = col.get("name")
                col_type = col.get("type", "").lower()
                
                # Normalize type (strip length/precision)
                normalized_type = col_type.split("(")[0].strip()
                
                column_index[normalized_type].append({
                    "full_name": f"{full_table_name}.{col_name}",
                    "table": full_table_name,
                    "column": col_name,
                    "type": normalized_type,
                    "stats": col.get("stats", {})
                })
    
    return column_index


def _find_candidate_column_pairs(
    column_index: Dict[str, List[Dict[str, Any]]],
    name_similarity_threshold: float
) -> List[Tuple[str, str, float]]:
    """
    Find candidate column pairs that might be related
    
    Filters:
    1. Same data type
    2. Different tables
    3. Column name similarity >= threshold
    4. At least one column ends with 'ID' or 'Key' (FK pattern)
    
    Returns:
        List of (col1_full_name, col2_full_name, name_similarity)
    """
    candidates = []
    
    for data_type, columns in column_index.items():
        # Skip non-FK types
        if data_type not in ["int", "bigint", "smallint", "varchar", "nvarchar", "uniqueidentifier", "guid"]:
            continue
        
        # Compare all pairs within same type
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Skip same table
                if col1["table"] == col2["table"]:
                    continue
                
                # Calculate name similarity
                name1 = col1["column"].lower()
                name2 = col2["column"].lower()
                similarity = SequenceMatcher(None, name1, name2).ratio()
                
                # Check if at least one looks like an ID/Key column
                is_id_pattern = (
                    name1.endswith("id") or name1.endswith("key") or
                    name2.endswith("id") or name2.endswith("key") or
                    name1 == "id" or name2 == "id"
                )
                
                # Also check for suffix matching (e.g., "CustomerID" matches "ID")
                suffix_match = (
                    (name1.endswith("id") and name2 == "id") or
                    (name2.endswith("id") and name1 == "id")
                )
                
                # Accept if:
                # - High name similarity (>=threshold) AND has ID pattern, OR
                # - Suffix match with lower threshold
                if (similarity >= name_similarity_threshold and is_id_pattern) or \
                   (suffix_match and similarity >= 0.4):
                    candidates.append((col1["full_name"], col2["full_name"], similarity))
    
    return candidates


def _calculate_value_overlaps(
    discovery_data: Dict[str, Any],
    candidate_pairs: List[Tuple[str, str, float]],
    sample_size: int
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Calculate value overlap rate for candidate column pairs
    
    Uses sample values from discovery data to estimate overlap.
    Also determines cardinality (one-to-one vs many-to-one).
    
    Returns:
        Dict mapping (col1, col2) -> {overlap_rate, name_similarity, cardinality}
    """
    overlaps = {}
    
    # Build value sets from discovery data
    value_cache = _build_value_cache(discovery_data)
    
    for col1_full, col2_full, name_similarity in candidate_pairs:
        # Get sample values
        values1 = value_cache.get(col1_full, set())
        values2 = value_cache.get(col2_full, set())
        
        if not values1 or not values2:
            continue
        
        # Calculate overlap
        intersection = values1 & values2
        overlap_rate = len(intersection) / max(len(values1), len(values2))
        
        # Determine cardinality (heuristic based on distinct counts)
        # ✅ IMPROVED: Stricter cardinality detection using min_cardinality_ratio
        distinct1 = len(values1)
        distinct2 = len(values2)
        
        # Use the min_cardinality_ratio parameter (default 2.0)
        # Require at least 2x difference for directional relationships
        ratio_1_to_2 = distinct1 / max(distinct2, 1)
        ratio_2_to_1 = distinct2 / max(distinct1, 1)
        
        if ratio_1_to_2 >= min_cardinality_ratio:
            # distinct1 >> distinct2: many col1 values -> few col2 values
            cardinality = "many_to_one"
        elif ratio_2_to_1 >= min_cardinality_ratio:
            # distinct2 >> distinct1: few col1 values -> many col2 values
            cardinality = "one_to_many"
        else:
            # Similar counts - likely one-to-one OR suspicious
            cardinality = "one_to_one"
        
        overlaps[(col1_full, col2_full)] = {
            "overlap_rate": overlap_rate,
            "name_similarity": name_similarity,
            "cardinality": cardinality,
            "sample_size1": distinct1,
            "sample_size2": distinct2
        }
    
    return overlaps


def _build_value_cache(discovery_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build cache of sample values for each column
    
    Returns:
        Dict mapping "schema.table.column" -> set of sample values
    """
    value_cache = {}
    
    for schema in discovery_data.get("schemas", []):
        schema_name = schema.get("name")
        
        for table in schema.get("tables", []):
            table_name = table.get("name")
            full_table_name = f"{schema_name}.{table_name}"
            
            for col in table.get("columns", []):
                col_name = col.get("name")
                full_col_name = f"{full_table_name}.{col_name}"
                
                # Get sample values from stats
                stats = col.get("stats", {})
                sample_values = stats.get("sample_values", [])
                
                if sample_values:
                    # Convert to set of strings for comparison
                    value_cache[full_col_name] = set(str(v) for v in sample_values if v is not None)
    
    return value_cache


# ============================================================================
# DEDUPLICATION AND RANKING
# ============================================================================

def _deduplicate_and_rank_relationships(
    relationships: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Deduplicate relationships and rank by confidence/method
    
    Priority (highest to lowest):
    1. Explicit foreign keys (confidence='very_high')
    2. Curated asset joins (confidence='high')
    3. Value overlap with strong name match (confidence='high'/'medium')
    4. Value overlap with weak name match (confidence='low')
    
    When duplicates exist, keep the highest confidence relationship.
    Also normalizes bidirectional relationships (A->B and B->A become single A->B).
    """
    # Group by normalized (from, to) pair
    grouped = defaultdict(list)
    
    for rel in relationships:
        from_col = rel["from"]
        to_col = rel["to"]
        
        # Normalize: always sort lexicographically to catch bidirectional duplicates
        key = tuple(sorted([from_col, to_col]))
        grouped[key].append(rel)
    
    # Keep highest confidence relationship for each pair
    confidence_rank = {
        "very_high": 4,
        "high": 3,
        "medium": 2,
        "low": 1
    }
    
    method_rank = {
        "explicit_foreign_key": 4,
        "curated_asset_join": 3,
        "value_overlap": 2
    }
    
    deduplicated = []
    for (col1, col2), rels in grouped.items():
        # Sort by confidence, then method, then overlap_rate
        def sort_key(r):
            return (
                confidence_rank.get(r["confidence"], 0),
                method_rank.get(r["method"], 0),
                r.get("overlap_rate", 0) or 0
            )
        
        best_rel = max(rels, key=sort_key)
        
        # Ensure consistent direction (from -> to based on cardinality)
        if best_rel.get("cardinality") == "one_to_many":
            # Swap direction for one-to-many
            best_rel["from"], best_rel["to"] = best_rel["to"], best_rel["from"]
            best_rel["cardinality"] = "many_to_one"
        
        deduplicated.append(best_rel)
    
    # Sort by confidence for readability
    deduplicated.sort(
        key=lambda r: confidence_rank.get(r["confidence"], 0),
        reverse=True
    )
    
    return deduplicated


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_relationships(
    relationships: List[Dict[str, Any]],
    discovery_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate relationships against discovery data
    
    Checks:
    - Both columns exist in schema
    - Data types are compatible
    - Referenced tables exist
    
    Returns:
        Validation report with errors and warnings
    """
    report = {
        "valid_count": 0,
        "invalid_count": 0,
        "errors": [],
        "warnings": []
    }
    
    # Build column existence index
    existing_columns = set()
    for schema in discovery_data.get("schemas", []):
        schema_name = schema.get("name")
        for table in schema.get("tables", []):
            table_name = table.get("name")
            for col in table.get("columns", []):
                col_name = col.get("name")
                existing_columns.add(f"{schema_name}.{table_name}.{col_name}")
    
    for rel in relationships:
        from_col = rel["from"]
        to_col = rel["to"]
        
        # Check existence
        if from_col not in existing_columns:
            report["errors"].append(f"Column not found: {from_col}")
            report["invalid_count"] += 1
        elif to_col not in existing_columns:
            report["errors"].append(f"Column not found: {to_col}")
            report["invalid_count"] += 1
        else:
            report["valid_count"] += 1
        
        # Check for self-referential relationships
        from_table = ".".join(from_col.split(".")[:-1])
        to_table = ".".join(to_col.split(".")[:-1])
        
        if from_table == to_table:
            report["warnings"].append(f"Self-referential relationship: {from_col} -> {to_col}")
    
    return report


def format_relationships_summary(relationships: List[Dict[str, Any]]) -> str:
    """
    Format relationships as human-readable summary
    
    Returns:
        Formatted string for logging/display
    """
    lines = ["\n" + "=" * 80]
    lines.append("RELATIONSHIP SUMMARY")
    lines.append("=" * 80)
    
    # Group by confidence
    by_confidence = defaultdict(list)
    for rel in relationships:
        by_confidence[rel["confidence"]].append(rel)
    
    for confidence in ["very_high", "high", "medium", "low"]:
        if confidence not in by_confidence:
            continue
        
        rels = by_confidence[confidence]
        lines.append(f"\n{confidence.upper()} Confidence ({len(rels)} relationships):")
        
        for rel in rels[:10]:  # Show first 10
            from_col = rel["from"]
            to_col = rel["to"]
            method = rel["method"]
            cardinality = rel["cardinality"]
            
            line = f"  {from_col} -> {to_col} ({cardinality})"
            
            if method == "curated_asset_join":
                line += f" [from {rel.get('source_kind', 'asset')}]"
            elif method == "value_overlap":
                overlap = rel.get("overlap_rate", 0)
                line += f" [overlap={overlap:.1%}]"
            
            lines.append(line)
        
        if len(rels) > 10:
            lines.append(f"  ... and {len(rels) - 10} more")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def build_relationship_graph(
    relationships: List[Dict[str, Any]],
    min_confidence: str = "medium"
) -> Dict[str, List[str]]:
    """
    Build adjacency list graph from relationships
    
    Args:
        relationships: List of relationship definitions
        min_confidence: Minimum confidence level to include
        
    Returns:
        Dict mapping table -> list of related tables
    """
    confidence_order = ["low", "medium", "high", "very_high"]
    min_idx = confidence_order.index(min_confidence) if min_confidence in confidence_order else 0
    
    graph = defaultdict(set)
    
    for rel in relationships:
        # Filter by confidence
        conf_idx = confidence_order.index(rel["confidence"]) if rel["confidence"] in confidence_order else 0
        if conf_idx < min_idx:
            continue
        
        from_table = ".".join(rel["from"].split(".")[:-1])
        to_table = ".".join(rel["to"].split(".")[:-1])
        
        graph[from_table].add(to_table)
        graph[to_table].add(from_table)  # Bidirectional
    
    # Convert sets to sorted lists
    return {table: sorted(list(related)) for table, related in graph.items()}