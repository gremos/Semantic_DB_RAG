#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models - Enhanced with LLM Analysis Results
Simple, clean, maintainable
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

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
    """Enhanced table information with LLM analysis results"""
    # Basic table info
    name: str
    schema: str
    full_name: str
    object_type: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    relationships: List[str] = field(default_factory=list)
    
    # LLM Analysis Results
    entity_type: str = "Unknown"              # Customer, Payment, Order, etc.
    business_role: str = "Unknown"            # Master data, transactions, etc.
    business_purpose: str = "Unknown"         # Specific business purpose
    confidence: float = 0.0                   # Analysis confidence
    
    # Table Quality Assessment
    table_quality: str = "production"         # production, test_copy, backup, archive
    business_priority: str = "medium"         # high, medium, low
    
    # BI Properties (from LLM analysis)
    bi_role: str = "dimension"               # fact (transactions) or dimension (reference)
    measures: List[str] = field(default_factory=list)      # Numeric columns for aggregation
    name_columns: List[str] = field(default_factory=list)  # Display name columns
    key_columns: List[str] = field(default_factory=list)   # ID and key columns
    time_columns: List[str] = field(default_factory=list)  # Date/time columns
    
    # Legacy properties for compatibility
    entity_keys: List[str] = field(default_factory=list)   # Same as key_columns
    data_type: str = "reference"             # operational or reference
    grain: str = "unknown"                   # customer, transaction, etc.
    
    def __post_init__(self):
        """Sync legacy properties with new ones"""
        if not self.entity_keys and self.key_columns:
            self.entity_keys = self.key_columns
        elif self.entity_keys and not self.key_columns:
            self.key_columns = self.entity_keys
    
    # Utility methods
    def has_names(self) -> bool:
        """Check if table has display name columns"""
        return len(self.name_columns) > 0
    
    def has_amounts(self) -> bool:
        """Check if table has amount/measure columns"""
        return len(self.measures) > 0
    
    def has_data(self) -> bool:
        """Check if table has data"""
        return self.row_count > 0
    
    def has_dates(self) -> bool:
        """Check if table has time columns"""
        return len(self.time_columns) > 0
    
    def is_production_quality(self) -> bool:
        """Check if table is production quality"""
        return self.table_quality == 'production'
    
    def is_high_priority(self) -> bool:
        """Check if table is high business priority"""
        return self.business_priority == 'high'
    
    def is_fact_table(self) -> bool:
        """Check if table is a fact table (contains measures)"""
        return self.bi_role == 'fact' or len(self.measures) > 0
    
    def is_dimension_table(self) -> bool:
        """Check if table is a dimension table (reference data)"""
        return self.bi_role == 'dimension'
    
    def is_customer_related(self) -> bool:
        """Check if customer-related table"""
        return (self.entity_type == 'Customer' or 
                'customer' in self.name.lower() or
                'client' in self.name.lower())
    
    def is_payment_related(self) -> bool:
        """Check if payment-related table"""
        return (self.entity_type == 'Payment' or
                any(word in self.name.lower() 
                    for word in ['payment', 'transaction', 'invoice', 'billing']))
    
    def is_order_related(self) -> bool:
        """Check if order-related table"""
        return (self.entity_type == 'Order' or
                any(word in self.name.lower()
                    for word in ['order', 'purchase', 'sale']))
    
    def get_display_columns(self, max_cols: int = 3) -> List[str]:
        """Get best columns for display"""
        display_cols = []
        
        # Prefer name columns first
        display_cols.extend(self.name_columns[:max_cols])
        
        # Then key columns if we need more
        remaining = max_cols - len(display_cols)
        if remaining > 0:
            display_cols.extend(self.key_columns[:remaining])
        
        # Then first few regular columns
        remaining = max_cols - len(display_cols)
        if remaining > 0:
            for col in self.columns[:remaining + 5]:  # Check more columns
                col_name = col.get('name', '')
                if (col_name and 
                    col_name not in display_cols and 
                    len(display_cols) < max_cols):
                    display_cols.append(col_name)
        
        return display_cols
    
    def get_sort_column(self) -> Optional[str]:
        """Get best column for sorting results"""
        # Prefer measure columns for sorting
        if self.measures:
            return self.measures[0]
        
        # Then time columns
        if self.time_columns:
            return self.time_columns[0]
        
        # Then key columns
        if self.key_columns:
            return self.key_columns[0]
        
        # Finally, any column
        if self.columns:
            return self.columns[0].get('name')
        
        return None
    
    def get_summary(self) -> str:
        """Get human-readable table summary"""
        parts = [f"{self.entity_type}"]
        
        if self.business_purpose != "Unknown":
            parts.append(f"({self.business_purpose})")
        
        if self.row_count > 0:
            parts.append(f"[{self.row_count:,} rows]")
        
        return " ".join(parts)

@dataclass
class BusinessDomain:
    """Business domain classification"""
    domain_type: str
    industry: str
    confidence: float
    sample_questions: List[str]
    capabilities: Dict[str, bool] = field(default_factory=dict)
    
    def has_customer_analytics(self) -> bool:
        """Check if domain supports customer analytics"""
        return self.capabilities.get('customer_analysis', False)
    
    def has_payment_analytics(self) -> bool:
        """Check if domain supports payment analytics"""
        return self.capabilities.get('payment_analysis', False)
    
    def has_order_analytics(self) -> bool:
        """Check if domain supports order analytics"""
        return self.capabilities.get('order_analysis', False)

@dataclass
class Relationship:
    """Table relationship"""
    from_table: str
    to_table: str
    relationship_type: str
    confidence: float
    description: str = ""
    
    def is_foreign_key(self) -> bool:
        """Check if relationship is a foreign key"""
        return self.relationship_type == 'foreign_key'
    
    def is_high_confidence(self) -> bool:
        """Check if relationship has high confidence"""
        return self.confidence >= 0.8

@dataclass
class QueryResult:
    """Query execution result with enhanced information"""
    question: str
    sql_query: str
    results: List[Dict[str, Any]]
    error: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    result_type: str = "data"
    
    # Analysis metadata
    intent_confidence: float = 0.0
    table_selection_method: str = "scoring"
    sql_generation_method: str = "llm"
    
    def is_successful(self) -> bool:
        """Check if query was successful"""
        return self.error is None
    
    def has_results(self) -> bool:
        """Check if query returned results"""
        return len(self.results) > 0
    
    def is_single_value(self) -> bool:
        """Check if result is a single value (aggregate)"""
        return (len(self.results) == 1 and 
                len(self.results[0]) == 1)
    
    def get_single_value(self) -> Any:
        """Get single value result"""
        if self.is_single_value():
            return list(self.results[0].values())[0]
        return None
    
    def get_result_summary(self) -> str:
        """Get human-readable result summary"""
        if not self.is_successful():
            return f"Failed: {self.error}"
        
        if not self.has_results():
            return "No results found"
        
        if self.is_single_value():
            value = self.get_single_value()
            if isinstance(value, (int, float)) and abs(value) >= 1000:
                return f"Result: {value:,}"
            else:
                return f"Result: {value}"
        
        return f"{len(self.results)} rows returned"

# Compatibility classes (simplified for legacy code)
@dataclass
class AnalyticalTask:
    """Analytical task definition"""
    task_type: str = "list"
    metrics: List[str] = field(default_factory=list)
    entity: Optional[str] = None
    crm_entities: List[str] = field(default_factory=list)
    time_window: Optional[str] = None
    grouping: List[str] = field(default_factory=list)
    top_limit: Optional[int] = None
    filters: List[str] = field(default_factory=list)
    
    def is_ranking_task(self) -> bool:
        """Check if task is about ranking/top results"""
        return self.task_type == 'ranking' or self.top_limit is not None
    
    def is_aggregation_task(self) -> bool:
        """Check if task requires aggregation"""
        return self.task_type in ['aggregation', 'sum', 'total']

@dataclass
class CapabilityContract:
    """Table capability assessment"""
    grain: Optional[str] = None
    measures: List[str] = field(default_factory=list)
    time_column: Optional[str] = None
    entity_keys: List[str] = field(default_factory=list)
    name_columns: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    data_freshness: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if capability is complete"""
        return (self.grain is not None and 
                (len(self.measures) > 0 or len(self.entity_keys) > 0))
    
    def supports_aggregation(self) -> bool:
        """Check if supports aggregation queries"""
        return len(self.measures) > 0
    
    def supports_grouping(self) -> bool:
        """Check if supports grouping queries"""
        return len(self.entity_keys) > 0 or len(self.name_columns) > 0

@dataclass  
class EvidenceScore:
    """Evidence scoring for table selection"""
    entity_match: float = 0.0
    purpose_match: float = 0.0
    quality_score: float = 0.0
    data_availability: float = 0.0
    column_relevance: float = 0.0
    business_priority: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Calculate total weighted score"""
        weights = {
            'entity_match': 0.3,
            'purpose_match': 0.2,
            'quality_score': 0.2,
            'data_availability': 0.1,
            'column_relevance': 0.1,
            'business_priority': 0.1
        }
        
        return (self.entity_match * weights['entity_match'] +
                self.purpose_match * weights['purpose_match'] +
                self.quality_score * weights['quality_score'] +
                self.data_availability * weights['data_availability'] +
                self.column_relevance * weights['column_relevance'] +
                self.business_priority * weights['business_priority'])
    
    def is_high_score(self) -> bool:
        """Check if score is high enough"""
        return self.total_score >= 0.7

# Type aliases for clarity
TableList = List[TableInfo]
RelationshipList = List[Relationship]
ResultList = List[Dict[str, Any]]