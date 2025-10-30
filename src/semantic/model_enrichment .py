"""
Semantic Enrichment Module

Advanced features for semantic model building:
- Duplicate table detection
- Measure inference from column names and types
- Semantic type detection
- Relationship confidence scoring
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ============================================================================
# SEMANTIC TYPE PATTERNS
# ============================================================================

SEMANTIC_TYPE_PATTERNS = {
    # Identifiers
    "id": [
        r".*id$", r".*_id$", r"^id$", r".*key$", r".*_key$",
        r".*number$", r".*_number$", r".*code$", r".*_code$"
    ],
    
    # Person/Organization
    "person_or_org_name": [
        r".*name$", r".*_name$", r"customer.*", r".*customer$",
        r".*company$", r".*organization$", r".*org$"
    ],
    
    # Contact info
    "email": [r".*email$", r".*_email$", r".*mail$"],
    "phone": [r".*phone$", r".*_phone$", r".*tel$", r".*mobile$"],
    "address": [r".*address$", r".*_address$", r".*street$", r".*city$"],
    
    # Temporal
    "date": [r".*date$", r".*_date$", r".*_dt$"],
    "datetime": [r".*datetime$", r".*timestamp$", r".*_ts$"],
    "year": [r".*year$", r"^year$", r".*_year$"],
    "month": [r".*month$", r"^month$", r".*_month$"],
    "quarter": [r".*quarter$", r"^quarter$", r".*_qtr$"],
    
    # Geographic
    "country": [r".*country$", r".*_country$"],
    "state": [r".*state$", r".*_state$", r".*province$"],
    "city": [r".*city$", r".*_city$"],
    "zip_code": [r".*zip$", r".*postal.*", r".*zipcode$"],
    
    # Financial
    "currency": [
        r".*amount$", r".*price$", r".*cost$", r".*revenue$",
        r".*total$", r".*subtotal$", r".*discount$", r".*tax$"
    ],
    
    # Measurement
    "quantity": [r".*quantity$", r".*qty$", r".*count$", r".*units$"],
    "percentage": [r".*percent.*", r".*rate$", r".*_pct$"],
    "weight": [r".*weight$", r".*_weight$"],
    "distance": [r".*distance$", r".*length$", r".*height$", r".*width$"],
    
    # Category/Classification
    "category": [r".*category$", r".*_category$", r".*type$", r".*_type$"],
    "status": [r".*status$", r".*_status$", r".*state$"],
    "flag": [r".*flag$", r".*_flag$", r"is_.*", r"has_.*"],
}


def detect_semantic_type(column_name: str, column_type: str) -> Optional[str]:
    """
    Detect semantic type from column name and SQL type
    
    Args:
        column_name: Column name
        column_type: SQL type (e.g., 'int', 'varchar(50)', 'datetime')
        
    Returns:
        Semantic type string or None
    """
    col_lower = column_name.lower()
    type_lower = column_type.lower()
    
    # Check patterns
    for semantic_type, patterns in SEMANTIC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, col_lower):
                # Validate against SQL type
                if _is_type_compatible(semantic_type, type_lower):
                    return semantic_type
    
    return None


def _is_type_compatible(semantic_type: str, sql_type: str) -> bool:
    """Check if semantic type is compatible with SQL type"""
    
    # ID types should be numeric or string
    if semantic_type == "id":
        return any(t in sql_type for t in ["int", "bigint", "numeric", "varchar", "char", "uuid"])
    
    # Dates
    if semantic_type in ["date", "datetime", "year", "month", "quarter"]:
        return any(t in sql_type for t in ["date", "time", "timestamp"])
    
    # Currency
    if semantic_type in ["currency", "quantity", "percentage"]:
        return any(t in sql_type for t in ["decimal", "numeric", "float", "money", "int"])
    
    # Strings
    if semantic_type in ["person_or_org_name", "email", "phone", "address", "category", "status"]:
        return any(t in sql_type for t in ["varchar", "char", "text", "string"])
    
    # Flags
    if semantic_type == "flag":
        return any(t in sql_type for t in ["bit", "boolean", "bool", "tinyint"])
    
    return True  # Default to compatible


# ============================================================================
# MEASURE INFERENCE
# ============================================================================

MEASURE_KEYWORDS = {
    "SUM": ["amount", "total", "subtotal", "revenue", "cost", "price", "quantity", "qty", "count"],
    "AVG": ["average", "avg", "rate", "score", "rating"],
    "COUNT": ["count", "number", "qty"],
    "MIN": ["min", "minimum", "lowest"],
    "MAX": ["max", "maximum", "highest"],
}


def infer_measures_from_columns(
    columns: List[Dict[str, Any]], 
    table_name: str
) -> List[Dict[str, Any]]:
    """
    Infer potential measures from numeric columns
    
    Args:
        columns: List of column dictionaries
        table_name: Table name for context
        
    Returns:
        List of measure definitions
    """
    measures = []
    
    for col in columns:
        col_name = col.get("name", "")
        col_type = col.get("type", "").lower()
        
        # Check if numeric
        if not any(t in col_type for t in ["int", "float", "decimal", "numeric", "money"]):
            continue
        
        # Infer aggregation
        aggregation = _infer_aggregation(col_name)
        
        if aggregation:
            measure = {
                "name": col_name,
                "expression": f"{aggregation}({col_name})",
                "description": f"{aggregation} of {col_name}"
            }
            
            # Detect unit
            col_lower = col_name.lower()
            
            if any(kw in col_lower for kw in ["amount", "price", "cost", "revenue", "total", "discount", "tax"]):
                measure["unit"] = "currency"
                measure["currency"] = "USD"  # Default
                measure["format_hint"] = "currency(2)"
            
            elif any(kw in col_lower for kw in ["percent", "rate", "pct"]):
                measure["unit"] = "percentage"
                measure["format_hint"] = "percentage(2)"
            
            elif any(kw in col_lower for kw in ["quantity", "qty", "count", "units"]):
                measure["unit"] = "count"
                measure["format_hint"] = "number(0)"
            
            measures.append(measure)
    
    return measures


def _infer_aggregation(column_name: str) -> Optional[str]:
    """Infer aggregation function from column name"""
    col_lower = column_name.lower()
    
    # Check keywords
    for agg, keywords in MEASURE_KEYWORDS.items():
        if any(kw in col_lower for kw in keywords):
            return agg
    
    # Default to SUM for numeric columns
    return "SUM"


# ============================================================================
# DUPLICATE DETECTION
# ============================================================================

def detect_duplicate_tables(
    tables: List[Dict[str, Any]], 
    threshold: float = 0.8
) -> List[Tuple[str, str, float]]:
    """
    Detect duplicate tables based on column similarity
    
    Args:
        tables: List of table dictionaries with columns
        threshold: Similarity threshold (0-1)
        
    Returns:
        List of (table1, table2, similarity_score) tuples
    """
    duplicates = []
    
    for i, table1 in enumerate(tables):
        for table2 in tables[i+1:]:
            similarity = _calculate_table_similarity(table1, table2)
            
            if similarity >= threshold:
                duplicates.append((
                    table1.get("full_name", table1.get("name")),
                    table2.get("full_name", table2.get("name")),
                    similarity
                ))
    
    return duplicates


def _calculate_table_similarity(table1: Dict[str, Any], table2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two tables based on columns
    
    Considers:
    - Column name overlap
    - Column type match
    - Primary key overlap
    """
    cols1 = {c["name"].lower(): c.get("type", "").lower() for c in table1.get("columns", [])}
    cols2 = {c["name"].lower(): c.get("type", "").lower() for c in table2.get("columns", [])}
    
    if not cols1 or not cols2:
        return 0.0
    
    # Name overlap
    common_names = set(cols1.keys()) & set(cols2.keys())
    name_similarity = len(common_names) / max(len(cols1), len(cols2))
    
    # Type match for common columns
    type_matches = sum(
        1 for name in common_names 
        if cols1[name] == cols2[name]
    )
    type_similarity = type_matches / len(common_names) if common_names else 0
    
    # Combined score
    return 0.7 * name_similarity + 0.3 * type_similarity


def rank_duplicate_tables(
    duplicates: List[Tuple[str, str, float]],
    table_types: Dict[str, str]
) -> Dict[str, str]:
    """
    Rank duplicate tables by quality
    
    Args:
        duplicates: List of (table1, table2, similarity) tuples
        table_types: Dict mapping table name to type (view, table, etc.)
        
    Returns:
        Dict mapping duplicate table to its canonical version
    """
    type_priority = {
        "view": 1,
        "stored_procedure": 2,
        "rdl_dataset": 3,
        "table": 4
    }
    
    duplicate_map = {}
    
    for table1, table2, similarity in duplicates:
        type1 = table_types.get(table1, "table")
        type2 = table_types.get(table2, "table")
        
        priority1 = type_priority.get(type1, 4)
        priority2 = type_priority.get(type2, 4)
        
        # Lower priority number = higher quality
        if priority1 < priority2:
            duplicate_map[table2] = table1
        else:
            duplicate_map[table1] = table2
    
    return duplicate_map


# ============================================================================
# RELATIONSHIP CONFIDENCE SCORING
# ============================================================================

def calculate_relationship_confidence(
    relationship: Dict[str, Any],
    discovery_data: Dict[str, Any]
) -> str:
    """
    Calculate confidence score for a relationship
    
    Returns: "high", "medium", or "low"
    """
    score = 0.0
    
    # Method-based scoring
    method = relationship.get("method", "")
    
    if method == "explicit_fk":
        score += 0.5  # Database-defined = high confidence
    elif method in ["rdl_join_analysis", "view_join_analysis"]:
        score += 0.3  # Curated = medium-high confidence
    elif method == "value_overlap":
        # Use overlap rate
        overlap = relationship.get("overlap_rate", 0)
        score += 0.3 * overlap
    elif method == "name_pattern":
        score += 0.1  # Name matching = low confidence
    
    # Cardinality scoring (many-to-one is most common and reliable)
    cardinality = relationship.get("cardinality", "")
    if cardinality == "many_to_one":
        score += 0.2
    elif cardinality == "one_to_one":
        score += 0.15
    
    # Verification scoring
    if "verification" in relationship:
        verification = relationship["verification"]
        if "overlap_rate" in verification:
            score += 0.2 * verification["overlap_rate"]
    
    # Convert to category
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"


# ============================================================================
# DISPLAY CONFIGURATION INFERENCE
# ============================================================================

def infer_display_config(
    table_name: str,
    columns: List[Dict[str, Any]],
    primary_key: List[str]
) -> Dict[str, Any]:
    """
    Infer display configuration for a table
    
    Returns:
        Display config dict with label/search/sort settings
    """
    config = {
        "display_name": _format_display_name(table_name),
        "default_label_column": None,
        "default_search_columns": [],
        "default_sort": None
    }
    
    # Find label column
    label_candidates = []
    for col in columns:
        col_name = col["name"]
        col_lower = col_name.lower()
        
        # Name columns are good labels
        if "name" in col_lower and col_name not in primary_key:
            label_candidates.append((col_name, 10))
        # Description columns
        elif "description" in col_lower or "desc" in col_lower:
            label_candidates.append((col_name, 8))
        # Title columns
        elif "title" in col_lower:
            label_candidates.append((col_name, 9))
    
    if label_candidates:
        label_candidates.sort(key=lambda x: x[1], reverse=True)
        config["default_label_column"] = label_candidates[0][0]
        config["default_search_columns"].append(label_candidates[0][0])
    
    # Add other searchable columns
    for col in columns:
        col_name = col["name"]
        col_type = col.get("type", "").lower()
        
        # String columns are searchable
        if any(t in col_type for t in ["varchar", "char", "text", "string"]):
            if col_name not in config["default_search_columns"]:
                config["default_search_columns"].append(col_name)
                
                # Limit to 5 search columns
                if len(config["default_search_columns"]) >= 5:
                    break
    
    # Add PK to search if not many columns
    if len(config["default_search_columns"]) < 3 and primary_key:
        for pk in primary_key:
            if pk not in config["default_search_columns"]:
                config["default_search_columns"].append(pk)
    
    # Default sort by label column or PK
    if config["default_label_column"]:
        config["default_sort"] = {
            "column": config["default_label_column"],
            "direction": "asc"
        }
    elif primary_key:
        config["default_sort"] = {
            "column": primary_key[0],
            "direction": "asc"
        }
    
    return config


def _format_display_name(table_name: str) -> str:
    """
    Format table name into display-friendly name
    
    Examples:
        tbl_customer -> Customer
        DimDate -> Date
        FactSales -> Sales
    """
    # Remove common prefixes
    name = re.sub(r"^(tbl_|dim|fact|vw_|v_)", "", table_name, flags=re.IGNORECASE)
    
    # Split on underscores or camelCase
    if "_" in name:
        words = name.split("_")
    else:
        # Split camelCase
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
    
    # Capitalize each word
    formatted = " ".join(word.capitalize() for word in words if word)
    
    return formatted or table_name


# ============================================================================
# COLUMN ROLE DETECTION
# ============================================================================

def detect_column_roles(
    columns: List[Dict[str, Any]],
    primary_key: List[str],
    foreign_keys: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Detect roles for columns
    
    Roles:
    - primary_key: Part of PK
    - foreign_key: FK reference
    - label: Display name column
    - attribute: Descriptive attribute
    - measure: Numeric measure
    """
    fk_columns = {fk["column"] for fk in foreign_keys}
    
    enriched = []
    
    for col in columns:
        col_name = col["name"]
        col_type = col.get("type", "").lower()
        
        # Determine role
        if col_name in primary_key:
            role = "primary_key"
        elif col_name in fk_columns:
            role = "foreign_key"
        elif "name" in col_name.lower():
            role = "label"
        elif any(t in col_type for t in ["int", "float", "decimal", "numeric", "money"]):
            role = "measure"
        else:
            role = "attribute"
        
        col_enriched = col.copy()
        col_enriched["role"] = role
        enriched.append(col_enriched)
    
    return enriched