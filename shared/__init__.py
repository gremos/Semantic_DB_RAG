"""
Shared Components for Semantic Database RAG System
Enhanced models, utilities, and configuration for version 2.0
"""

from .config import Config
from .models import (
    TableInfo, BusinessDomain, Relationship, QueryResult, SystemStatus,
    DatabaseObject, AnalysisStats, BusinessTemplate,
    table_to_dict, dict_to_table, domain_to_dict, dict_to_domain,
    relationship_to_dict, dict_to_relationship
)
from .utils import (
    extract_json_from_response, classify_entity_type_by_patterns, 
    find_table_relationships_comprehensive, create_llm_classification_prompt,
    find_related_tables_fuzzy, generate_business_sql_prompt,
    clean_sql_query, safe_database_value, save_cache, load_cache,
    should_exclude_table, generate_sample_questions, determine_business_capabilities
)

__version__ = "2.0.0-enhanced"
__all__ = [
    # Core classes
    "Config",
    "TableInfo", "BusinessDomain", "Relationship", "QueryResult", "SystemStatus",
    "DatabaseObject", "AnalysisStats", "BusinessTemplate",
    
    # Model conversion utilities
    "table_to_dict", "dict_to_table", "domain_to_dict", "dict_to_domain",
    "relationship_to_dict", "dict_to_relationship",
    
    # Entity classification and analysis
    "extract_json_from_response", "classify_entity_type_by_patterns",
    "find_table_relationships_comprehensive", "create_llm_classification_prompt",
    
    # 4-stage pipeline support
    "find_related_tables_fuzzy", "generate_business_sql_prompt",
    
    # Data processing and validation
    "clean_sql_query", "safe_database_value", "save_cache", "load_cache",
    "should_exclude_table",
    
    # Business intelligence
    "generate_sample_questions", "determine_business_capabilities"
]