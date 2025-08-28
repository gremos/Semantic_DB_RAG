#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models - Simplified & Essential
Following DRY, SOLID, YAGNI principles
Keep only what's needed
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union

@dataclass
class DatabaseObject:
    """Basic database object"""
    schema: str
    name: str
    object_type: str
    estimated_rows: int = 0
    
    @property
    def full_name(self) -> str:
        return f"[{self.schema}].[{self.name}]"

@dataclass
class TableInfo:
    """Essential table information for BI"""
    name: str
    schema: str
    full_name: str
    object_type: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    relationships: List[str] = field(default_factory=list)
    
    # LLM Analysis Results
    entity_type: str = "Unknown"
    business_role: str = "Unknown"
    confidence: float = 0.0
    
    # BI Properties (set by LLM analysis)
    bi_role: str = "dimension"          # fact or dimension
    measures: List[str] = field(default_factory=list)      # Amount columns
    entity_keys: List[str] = field(default_factory=list)   # ID columns  
    name_columns: List[str] = field(default_factory=list)  # Name columns
    time_columns: List[str] = field(default_factory=list)  # Date columns
    business_priority: str = "medium"   # high, medium, low
    
    # Derived properties
    data_type: str = "reference"        # operational or reference
    grain: str = "unknown"              # customer, transaction, etc.
    
    def has_names(self) -> bool:
        """Check if table has name columns"""
        return len(self.name_columns) > 0
    
    def has_amounts(self) -> bool:
        """Check if table has amount columns"""
        return len(self.measures) > 0
    
    def has_data(self) -> bool:
        """Check if table has data"""
        return self.row_count > 0
    
    def is_customer_related(self) -> bool:
        """Check if customer-related table"""
        return (self.entity_type == 'Customer' or 
                'customer' in self.name.lower())
    
    def is_payment_related(self) -> bool:
        """Check if payment-related table"""
        return (self.entity_type == 'Payment' or
                any(word in self.name.lower() for word in ['payment', 'transaction', 'invoice']))

@dataclass
class BusinessDomain:
    """Business domain classification"""
    domain_type: str
    industry: str
    confidence: float
    sample_questions: List[str]
    capabilities: Dict[str, bool] = field(default_factory=dict)

@dataclass
class Relationship:
    """Table relationship"""
    from_table: str
    to_table: str
    relationship_type: str
    confidence: float
    description: str = ""

@dataclass
class QueryResult:
    """Query execution result"""
    question: str
    sql_query: str
    results: List[Dict[str, Any]]
    error: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    result_type: str = "data"
    
    def is_successful(self) -> bool:
        """Check if query was successful"""
        return self.error is None
    
    def has_results(self) -> bool:
        """Check if query returned results"""
        return len(self.results) > 0
    
    def is_single_value(self) -> bool:
        """Check if result is a single value"""
        return (len(self.results) == 1 and 
                len(self.results[0]) == 1)
    
    def get_single_value(self) -> Any:
        """Get single value result"""
        if self.is_single_value():
            return list(self.results[0].values())[0]
        return None

# Compatibility classes (simplified versions)
@dataclass
class AnalyticalTask:
    """Simplified analytical task for compatibility"""
    task_type: str = "list"
    metrics: List[str] = field(default_factory=list)
    entity: Optional[str] = None
    crm_entities: List[str] = field(default_factory=list)
    time_window: Optional[str] = None
    grouping: List[str] = field(default_factory=list)
    top_limit: Optional[int] = None
    filters: List[str] = field(default_factory=list)

@dataclass
class CapabilityContract:
    """Simplified capability contract for compatibility"""
    grain: Optional[str] = None
    measures: List[str] = field(default_factory=list)
    time_column: Optional[str] = None
    entity_keys: List[str] = field(default_factory=list)
    name_columns: List[str] = field(default_factory=list)
    quality_checks: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Basic completeness check"""
        return bool(self.grain and (self.measures or self.entity_keys))

@dataclass  
class EvidenceScore:
    """Simplified evidence score for compatibility"""
    role_match: float = 0.0
    join_evidence: float = 0.0
    lexical_match: float = 0.0
    graph_proximity: float = 0.0
    operational_tag: float = 0.0
    row_count: float = 0.0
    freshness: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Simple total score calculation"""
        return (self.lexical_match * 0.4 + 
                self.role_match * 0.3 + 
                self.graph_proximity * 0.2 + 
                self.operational_tag * 0.1)

# Type aliases for clarity
TableList = List[TableInfo]
RelationshipList = List[Relationship]