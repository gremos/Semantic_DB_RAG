#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BI-Aware Data Models - Simple, Readable, Maintainable
Following README: Enhanced for Business Intelligence with clean methods
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple

@dataclass
class DatabaseObject:
    """Simple database object for discovery"""
    schema: str
    name: str
    object_type: str
    estimated_rows: int = 0
    
    @property
    def full_name(self) -> str:
        return f"[{self.schema}].[{self.name}]"

@dataclass
class TableInfo:
    """BI-Enhanced table information with capability assessment"""
    name: str
    schema: str
    full_name: str
    object_type: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    relationships: List[str] = field(default_factory=list)
    
    # Standard semantic properties
    entity_type: str = "Unknown"
    business_role: str = "Unknown" 
    confidence: float = 0.0
    
    # BI-Aware properties for capability contracts
    data_type: str = "reference"  # operational, planning, reference
    bi_role: str = "dimension"    # fact, dimension, bridge
    grain: str = "unknown"        # customer, transaction, order, event
    
    # BI Capabilities for contract validation
    measures: List[str] = field(default_factory=list)      # Numeric columns for aggregation
    entity_keys: List[str] = field(default_factory=list)   # Keys for grouping/filtering
    time_columns: List[str] = field(default_factory=list)  # Date/time columns
    filter_columns: List[str] = field(default_factory=list) # Status, type, region columns
    
    def has_capability(self, capability_type: str) -> bool:
        """Check if table has specific BI capability"""
        if capability_type == "measures":
            return len(self.measures) > 0
        elif capability_type == "time":
            return len(self.time_columns) > 0
        elif capability_type == "entities":
            return len(self.entity_keys) > 0
        elif capability_type == "operational":
            return self.data_type == "operational"
        return False
    
    def get_capability_score(self) -> float:
        """Calculate overall BI capability score"""
        capabilities = [
            1.0 if self.has_capability("measures") else 0.0,
            1.0 if self.has_capability("time") else 0.0,
            1.0 if self.has_capability("entities") else 0.0,
            1.0 if self.has_capability("operational") else 0.5,
            1.0 if self.row_count > 0 else 0.0
        ]
        return sum(capabilities) / len(capabilities)

@dataclass
class BusinessDomain:
    """BI-Enhanced business domain classification"""
    domain_type: str
    industry: str
    confidence: float
    sample_questions: List[str]
    capabilities: Dict[str, bool] = field(default_factory=dict)
    
    # BI-specific domain properties
    bi_maturity: str = "Basic"  # Basic, Intermediate, Advanced
    analytical_patterns: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0

@dataclass
class Relationship:
    """Enhanced database relationship with BI context"""
    from_table: str
    to_table: str
    relationship_type: str  # foreign_key, fact_dimension, semantic_reference
    confidence: float
    description: str = ""
    
    # BI-specific relationship properties
    cardinality: str = "unknown"  # one_to_one, one_to_many, many_to_many
    join_strength: str = "weak"   # weak, moderate, strong
    bi_pattern: str = "unknown"   # star_schema, snowflake, bridge

@dataclass 
class QueryResult:
    """Enhanced query execution result with BI metrics"""
    question: str
    sql_query: str
    results: List[Dict[str, Any]]
    error: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    # BI-aware result properties
    result_type: str = "data"  # data, ner, error
    capability_score: float = 0.0
    evidence_reasoning: List[str] = field(default_factory=list)
    
    def is_successful(self) -> bool:
        return self.error is None
    
    def has_results(self) -> bool:
        return len(self.results) > 0
    
    def is_single_value(self) -> bool:
        return (len(self.results) == 1 and 
                len(self.results[0]) == 1)
    
    def get_single_value(self) -> Any:
        if self.is_single_value():
            return list(self.results[0].values())[0]
        return None
    
    def is_ner(self) -> bool:
        """Check if result is a Non-Executable Analysis Report"""
        return self.result_type == "ner"

@dataclass
class CapabilityContract:
    """BI capability contract for query validation"""
    grain: Optional[str] = None
    measures: List[str] = field(default_factory=list)
    time_column: Optional[str] = None
    entity_keys: List[str] = field(default_factory=list)
    join_paths: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    quality_checks: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if contract meets minimal BI requirements"""
        # Basic completeness: has grain and either measures or entity keys
        has_grain = bool(self.grain and self.grain != 'unknown')
        has_capabilities = bool(self.measures or self.entity_keys)
        has_data = self.quality_checks.get('row_count', 0) > 0
        
        return has_grain and has_capabilities and has_data
    
    def get_completeness_score(self) -> float:
        """Get contract completeness as percentage"""
        checks = [
            bool(self.grain and self.grain != 'unknown'),
            bool(self.measures or self.entity_keys),
            bool(self.time_column),
            bool(self.entity_keys),
            bool(self.quality_checks.get('row_count', 0) > 0)
        ]
        return sum(checks) / len(checks)
    
    def get_missing_capabilities(self) -> List[str]:
        """Get list of missing capabilities"""
        missing = []
        
        if not self.grain or self.grain == 'unknown':
            missing.append("Row grain identification")
        
        if not self.measures and not self.entity_keys:
            missing.append("Numeric measures or entity keys")
        elif not self.measures:
            missing.append("Numeric measures for aggregation")
        
        if not self.time_column:
            missing.append("Time/date column for filtering")
        
        if not self.entity_keys:
            missing.append("Entity keys for grouping")
        
        quality = self.quality_checks
        if quality.get('row_count', 0) == 0:
            missing.append("Data availability (zero rows)")
        
        return missing

@dataclass
class AnalyticalTask:
    """Normalized analytical task from natural language"""
    task_type: str  # aggregation, ranking, trend, distribution, cohort, funnel
    metrics: List[str] = field(default_factory=list)
    entity: Optional[str] = None
    time_window: Optional[str] = None
    bucketing: Optional[str] = None
    grouping: List[str] = field(default_factory=list)
    top_limit: Optional[int] = None
    filters: List[str] = field(default_factory=list)
    
    def requires_aggregation(self) -> bool:
        """Check if task requires aggregation functions"""
        return self.task_type in ['aggregation', 'ranking', 'trend']
    
    def requires_grouping(self) -> bool:
        """Check if task requires GROUP BY"""
        return len(self.grouping) > 0 or self.task_type in ['trend', 'distribution']
    
    def get_complexity_score(self) -> float:
        """Get task complexity score (0-1)"""
        complexity = 0.0
        
        # Base complexity by type
        type_complexity = {
            'aggregation': 0.3,
            'ranking': 0.4,
            'trend': 0.6,
            'distribution': 0.7,
            'cohort': 0.8,
            'funnel': 0.9
        }
        complexity += type_complexity.get(self.task_type, 0.5)
        
        # Additional complexity factors
        if self.grouping:
            complexity += 0.1
        if self.time_window:
            complexity += 0.1
        if len(self.filters) > 1:
            complexity += 0.1
        
        return min(1.0, complexity)

@dataclass
class EvidenceScore:
    """Enhanced Evidence-based scoring for table selection"""
    role_match: float = 0.0
    join_evidence: float = 0.0
    lexical_match: float = 0.0
    graph_proximity: float = 0.0  # Now used for table quality
    operational_tag: float = 0.0
    row_count: float = 0.0
    freshness: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Calculate weighted total evidence score - Updated weights"""
        weights = {
            'role_match': 3.0,        # BI role importance
            'lexical_match': 4.0,     # Name matching (high priority)
            'graph_proximity': 5.0,   # NEW: Table quality (highest priority)
            'operational_tag': 2.0,   # Operational data preference
            'join_evidence': 2.0,     # Relationship connectivity
            'row_count': 1.0,         # Data volume (low priority)
            'freshness': 1.0          # Recency (low priority)
        }
        
        total = (
            self.role_match * weights['role_match'] +
            self.lexical_match * weights['lexical_match'] +
            self.graph_proximity * weights['graph_proximity'] +
            self.operational_tag * weights['operational_tag'] +
            self.join_evidence * weights['join_evidence'] +
            self.row_count * weights['row_count'] +
            self.freshness * weights['freshness']
        )
        
        max_possible = sum(weights.values())
        return total / max_possible
    
    def get_explanation(self) -> List[str]:
        """Get human-readable explanation of evidence"""
        explanations = []
        
        if self.role_match > 0.7:
            explanations.append("Strong BI role match (fact/operational table)")
        if self.lexical_match > 0.7:
            explanations.append("Strong semantic match to query intent")
        if self.graph_proximity > 0.8:
            explanations.append("High-quality main table (not temp/dated)")
        elif self.graph_proximity < 0.6:
            explanations.append("⚠️ Lower quality table (temp/dated/bridge)")
        if self.operational_tag > 0.8:
            explanations.append("Contains operational (non-planning) data")
        if self.row_count > 0.5:
            explanations.append("Sufficient data volume available")
        
        return explanations

@dataclass
class NonExecutableAnalysisReport:
    """NER - Report when capability contracts fail"""
    question: str
    normalized_task: Dict[str, Any]  # AnalyticalTask as dict
    missing_capabilities: List[str]
    top_candidate_tables: List[Tuple[str, float]]  # (table_name, evidence_score)
    fix_paths: List[str]
    suggested_queries: List[str] = field(default_factory=list)
    
    def to_query_result(self) -> QueryResult:
        """Convert NER to QueryResult format"""
        return QueryResult(
            question=self.question,
            sql_query="-- NER: No executable SQL generated",
            results=[],
            error=f"Capability check failed: {', '.join(self.missing_capabilities[:3])}",
            result_type="ner",
            evidence_reasoning=self.fix_paths
        )

# Type aliases for better code readability
TableList = List[TableInfo]
RelationshipList = List[Relationship]
QueryCapabilityResult = Union[TableList, NonExecutableAnalysisReport]

# Utility functions for BI-aware operations
def create_fact_table(name: str, schema: str, measures: List[str], 
                     entity_keys: List[str], time_columns: List[str],
                     sample_data: List[Dict[str, Any]]) -> TableInfo:
    """Create a properly configured fact table"""
    table = TableInfo(
        name=name,
        schema=schema,
        full_name=f"[{schema}].[{name}]",
        object_type="BASE TABLE",
        row_count=len(sample_data),
        columns=[],  # Would be populated from discovery
        sample_data=sample_data,
        data_type="operational",
        bi_role="fact",
        grain="transaction",
        measures=measures,
        entity_keys=entity_keys,
        time_columns=time_columns
    )
    return table

def create_dimension_table(name: str, schema: str, entity_type: str,
                          entity_keys: List[str], 
                          sample_data: List[Dict[str, Any]]) -> TableInfo:
    """Create a properly configured dimension table"""
    table = TableInfo(
        name=name,
        schema=schema,
        full_name=f"[{schema}].[{name}]",
        object_type="BASE TABLE",
        row_count=len(sample_data),
        columns=[],  # Would be populated from discovery
        sample_data=sample_data,
        entity_type=entity_type,
        data_type="reference",
        bi_role="dimension",
        grain=entity_type.lower(),
        entity_keys=entity_keys
    )
    return table

def calculate_bi_readiness(tables: TableList) -> Dict[str, Any]:
    """Calculate overall BI readiness of discovered schema"""
    if not tables:
        return {'readiness_score': 0.0, 'issues': ['No tables available']}
    
    fact_tables = [t for t in tables if getattr(t, 'bi_role', '') == 'fact']
    operational_tables = [t for t in tables if getattr(t, 'data_type', '') == 'operational']
    tables_with_measures = [t for t in tables if getattr(t, 'measures', [])]
    tables_with_time = [t for t in tables if getattr(t, 'time_columns', [])]
    
    # Calculate readiness factors
    factors = {
        'has_fact_tables': len(fact_tables) > 0,
        'has_operational_data': len(operational_tables) > 0,
        'has_measures': len(tables_with_measures) > 0,
        'has_time_dimensions': len(tables_with_time) > 0,
        'sufficient_volume': sum(t.row_count for t in tables) > 1000
    }
    
    readiness_score = sum(factors.values()) / len(factors)
    
    # Identify issues
    issues = []
    if not factors['has_fact_tables']:
        issues.append("No fact tables identified for aggregation")
    if not factors['has_operational_data']:
        issues.append("No operational data found (only planning/reference)")
    if not factors['has_measures']:
        issues.append("No numeric measures available for analysis")
    if not factors['has_time_dimensions']:
        issues.append("No time columns for temporal analysis")
    
    return {
        'readiness_score': readiness_score,
        'factors': factors,
        'issues': issues,
        'fact_table_count': len(fact_tables),
        'operational_table_count': len(operational_tables),
        'total_data_volume': sum(t.row_count for t in tables)
    }