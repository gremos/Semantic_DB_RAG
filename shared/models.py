#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models - Enhanced for Better Revenue Table Selection
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
Enhanced: Revenue readiness scoring, better table classification
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
    """Enhanced table information with revenue readiness scoring"""
    # Basic table info
    name: str
    schema: str
    full_name: str
    object_type: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    relationships: List[str] = field(default_factory=list)
    
    # Enhanced LLM Analysis Results
    entity_type: str = "Unknown"              # Customer, Payment, CustomerRevenue, etc.
    business_role: str = "Unknown"            # Revenue Analytics, Customer Master Data, etc.
    business_purpose: str = "Unknown"         # Specific business purpose
    confidence: float = 0.0                   # Analysis confidence
    
    # Enhanced Revenue Analytics Support
    revenue_readiness: float = 0.0            # 0.0-1.0 score for revenue queries
    bi_role: str = "dimension"               # fact, dimension, lookup
    business_priority: str = "medium"         # high, medium, low
    
    # Enhanced Column Classification
    measures: List[str] = field(default_factory=list)      # Amount/revenue columns
    name_columns: List[str] = field(default_factory=list)  # Display name columns
    entity_keys: List[str] = field(default_factory=list)   # ID/key columns
    time_columns: List[str] = field(default_factory=list)  # Date/time columns
    
    # Legacy compatibility
    key_columns: List[str] = field(default_factory=list)   # Same as entity_keys
    data_type: str = "reference"             # operational or reference
    grain: str = "unknown"                   # customer, transaction, etc.
    table_quality: str = "production"        # production, test, archive
    
    def __post_init__(self):
        """Sync legacy properties and calculate derived scores"""
        # Sync legacy properties
        if not self.entity_keys and self.key_columns:
            self.entity_keys = self.key_columns
        elif self.entity_keys and not self.key_columns:
            self.key_columns = self.entity_keys
        
        # Auto-calculate revenue readiness if not set
        if self.revenue_readiness == 0.0:
            self.revenue_readiness = self._calculate_revenue_readiness()
    
    def _calculate_revenue_readiness(self) -> float:
        """Calculate revenue readiness score based on table characteristics"""
        score = 0.0
        
        # Entity type scoring (40% weight)
        entity_scores = {
            'CustomerRevenue': 1.0,    # Perfect for revenue queries
            'Payment': 0.9,            # Excellent for revenue queries
            'Order': 0.8,              # Good for revenue queries
            'Customer': 0.6,           # Supports revenue with joins
            'Contract': 0.7,           # Good for contract revenue
            'PaymentMethod': 0.1,      # Poor - lookup table
            'Category': 0.1,           # Poor - lookup table
            'System': 0.0,             # Not suitable
            'Other': 0.3               # Unknown potential
        }
        score += entity_scores.get(self.entity_type, 0.3) * 0.4
        
        # BI role scoring (25% weight)
        if self.bi_role == 'fact':
            score += 0.25
        elif self.bi_role == 'dimension':
            score += 0.15
        elif self.bi_role == 'lookup':
            score += 0.05
        
        # Measures availability (20% weight)
        if len(self.measures) >= 2:
            score += 0.20
        elif len(self.measures) == 1:
            score += 0.15
        elif self._has_revenue_columns():
            score += 0.10
        
        # Data volume (10% weight)
        if self.row_count > 1000:
            score += 0.10
        elif self.row_count > 100:
            score += 0.07
        elif self.row_count > 10:
            score += 0.03
        
        # Customer linkage (5% weight)
        if len(self.entity_keys) > 0 or self._has_customer_keys():
            score += 0.05
        
        return min(score, 1.0)
    
    def _has_revenue_columns(self) -> bool:
        """Check if table has revenue-like columns"""
        revenue_keywords = ['amount', 'total', 'revenue', 'value', 'price', 'cost']
        for col in self.columns:
            col_name = col.get('name', '').lower()
            col_type = col.get('data_type', '').lower()
            if (any(keyword in col_name for keyword in revenue_keywords) and
                col_type in ['decimal', 'money', 'float', 'numeric']):
                return True
        return False
    
    def _has_customer_keys(self) -> bool:
        """Check if table has customer-related keys"""
        customer_keywords = ['customer', 'client', 'account', 'business_point']
        for col in self.columns:
            col_name = col.get('name', '').lower()
            if (any(keyword in col_name for keyword in customer_keywords) and
                col_name.endswith('id')):
                return True
        return False
    
    # Enhanced Utility Methods
    def is_revenue_ready(self) -> bool:
        """Check if table is ready for revenue queries"""
        return self.revenue_readiness >= 0.7
    
    def is_fact_table(self) -> bool:
        """Check if table is a fact table"""
        return self.bi_role == 'fact' or len(self.measures) > 0
    
    def is_lookup_table(self) -> bool:
        """Check if table is a lookup/reference table"""
        return (self.bi_role == 'lookup' or 
                self.entity_type in ['PaymentMethod', 'Category'] or
                self.row_count < 50)
    
    def is_customer_related(self) -> bool:
        """Check if customer-related table"""
        return (self.entity_type in ['Customer', 'CustomerRevenue'] or 
                'customer' in self.name.lower() or
                'client' in self.name.lower())
    
    def is_payment_related(self) -> bool:
        """Check if payment-related table"""
        return (self.entity_type in ['Payment', 'CustomerRevenue'] or
                any(word in self.name.lower() 
                    for word in ['payment', 'transaction', 'invoice', 'billing']))
    
    def supports_revenue_queries(self) -> bool:
        """Check if table supports revenue queries"""
        return (self.is_revenue_ready() and 
                not self.is_lookup_table() and
                (self.is_customer_related() or self.is_payment_related()))
    
    def get_revenue_score(self) -> float:
        """Get comprehensive revenue query score"""
        if self.is_lookup_table():
            return 0.1  # Strongly discourage lookup tables
        
        base_score = self.revenue_readiness
        
        # Boost for customer + payment combination
        if self.entity_type == 'CustomerRevenue':
            base_score *= 1.2
        elif self.entity_type == 'Payment' and self._has_customer_keys():
            base_score *= 1.1
        
        # Penalty for small tables (likely lookup)
        if self.row_count < 100:
            base_score *= 0.7
        
        # Boost for high business priority
        if self.business_priority == 'high':
            base_score *= 1.1
        
        return min(base_score, 1.0)
    
    def get_display_columns(self, max_cols: int = 3) -> List[str]:
        """Get best columns for display"""
        display_cols = []
        
        # Prefer name columns first
        display_cols.extend(self.name_columns[:max_cols])
        
        # Then key columns if we need more
        remaining = max_cols - len(display_cols)
        if remaining > 0:
            display_cols.extend(self.entity_keys[:remaining])
        
        # Then first few regular columns
        remaining = max_cols - len(display_cols)
        if remaining > 0:
            for col in self.columns[:remaining + 5]:
                col_name = col.get('name', '')
                if (col_name and 
                    col_name not in display_cols and 
                    len(display_cols) < max_cols):
                    display_cols.append(col_name)
        
        return display_cols
    
    def get_best_measure_column(self) -> Optional[str]:
        """Get best measure column for revenue queries"""
        if self.measures:
            return self.measures[0]
        
        # Look for revenue-like columns
        revenue_keywords = ['amount', 'total', 'revenue', 'value', 'price']
        for col in self.columns:
            col_name = col.get('name', '').lower()
            col_type = col.get('data_type', '').lower()
            if (any(keyword in col_name for keyword in revenue_keywords) and
                col_type in ['decimal', 'money', 'float', 'numeric']):
                return col.get('name')
        
        return None
    
    def get_best_customer_key(self) -> Optional[str]:
        """Get best customer key column"""
        customer_keywords = ['customer', 'client', 'account', 'business_point']
        
        for col in self.columns:
            col_name = col.get('name', '').lower()
            if (any(keyword in col_name for keyword in customer_keywords) and
                col_name.endswith('id')):
                return col.get('name')
        
        # Fallback to any ID column
        if self.entity_keys:
            return self.entity_keys[0]
        
        return None
    
    def get_summary(self) -> str:
        """Get human-readable table summary"""
        parts = [f"{self.entity_type}"]
        
        if self.business_role != "Unknown":
            parts.append(f"({self.business_role})")
        
        if self.is_revenue_ready():
            parts.append("[Revenue Ready]")
        elif self.is_lookup_table():
            parts.append("[Lookup]")
        
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
        return self.capabilities.get('customer_analytics', False)
    
    def has_revenue_analytics(self) -> bool:
        """Check if domain supports revenue analytics"""
        return self.capabilities.get('revenue_analytics', False)
    
    def has_customer_revenue_analysis(self) -> bool:
        """Check if domain supports customer revenue analysis"""
        return self.capabilities.get('customer_revenue_analysis', False)

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
    sql_generation_method: str = "template"
    
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
    revenue_readiness: float = 0.0  # Enhanced: Revenue readiness scoring
    
    @property
    def total_score(self) -> float:
        """Calculate total weighted score with revenue emphasis"""
        weights = {
            'entity_match': 0.25,
            'revenue_readiness': 0.25,     # Enhanced: High weight for revenue
            'purpose_match': 0.15,
            'quality_score': 0.15,
            'data_availability': 0.10,
            'column_relevance': 0.05,
            'business_priority': 0.05
        }
        
        return (self.entity_match * weights['entity_match'] +
                self.revenue_readiness * weights['revenue_readiness'] +
                self.purpose_match * weights['purpose_match'] +
                self.quality_score * weights['quality_score'] +
                self.data_availability * weights['data_availability'] +
                self.column_relevance * weights['column_relevance'] +
                self.business_priority * weights['business_priority'])
    
    def is_high_score(self) -> bool:
        """Check if score is high enough"""
        return self.total_score >= 0.7
    
    def is_revenue_suitable(self) -> bool:
        """Check if suitable for revenue queries"""
        return self.revenue_readiness >= 0.7 and self.total_score >= 0.6

# Type aliases for clarity
TableList = List[TableInfo]
RelationshipList = List[Relationship]
ResultList = List[Dict[str, Any]]