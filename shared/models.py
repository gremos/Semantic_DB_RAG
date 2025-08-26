#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced BI-Aware Data Models - Enhanced with Additional Properties
Following README: Enhanced for Business Intelligence with better entity support
Simple, Readable, Maintainable
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
    """Enhanced table information with comprehensive BI capabilities"""
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
    
    # Enhanced BI Capabilities for contract validation
    measures: List[str] = field(default_factory=list)        # Numeric columns for aggregation
    entity_keys: List[str] = field(default_factory=list)     # Keys for grouping/filtering
    time_columns: List[str] = field(default_factory=list)    # Date/time columns
    filter_columns: List[str] = field(default_factory=list)  # Status, type, region columns
    name_columns: List[str] = field(default_factory=list)    # Name/title columns for display
    
    # Business priority for table selection
    business_priority: str = "medium"  # high, medium, low
    
    def get_first_3_samples(self) -> List[Dict[str, Any]]:
        """Get first 3 sample rows"""
        return [row for row in self.sample_data 
                if row.get('__sample_position__', '').startswith('first_')]
    
    def get_last_3_samples(self) -> List[Dict[str, Any]]:
        """Get last 3 sample rows"""
        return [row for row in self.sample_data 
                if row.get('__sample_position__', '').startswith('last_')]
    
    def get_clean_samples(self) -> List[Dict[str, Any]]:
        """Get samples without position markers"""
        clean_samples = []
        for row in self.sample_data:
            clean_row = {k: v for k, v in row.items() 
                        if not k.startswith('__')}
            clean_samples.append(clean_row)
        return clean_samples
    
    def has_capability(self, capability_type: str) -> bool:
        """Check if table has specific BI capability"""
        if capability_type == "measures":
            return len(self.measures) > 0
        elif capability_type == "time":
            return len(self.time_columns) > 0
        elif capability_type == "entities":
            return len(self.entity_keys) > 0
        elif capability_type == "names":
            return len(self.name_columns) > 0
        elif capability_type == "operational":
            return self.data_type == "operational"
        elif capability_type == "samples":
            return len(self.sample_data) > 0
        elif capability_type == "high_priority":
            return self.business_priority == "high"
        return False
    
    def get_capability_score(self) -> float:
        """Calculate overall BI capability score"""
        capabilities = [
            1.0 if self.has_capability("measures") else 0.0,
            1.0 if self.has_capability("time") else 0.0,
            1.0 if self.has_capability("entities") else 0.0,
            1.0 if self.has_capability("names") else 0.0,
            1.0 if self.has_capability("operational") else 0.5,
            1.0 if self.row_count > 0 else 0.0,
            1.0 if self.has_capability("samples") else 0.0,
            1.0 if self.has_capability("high_priority") else 0.5
        ]
        return sum(capabilities) / len(capabilities)
    
    def get_sample_summary(self) -> Dict[str, Any]:
        """Get summary of sample data"""
        first_3 = self.get_first_3_samples()
        last_3 = self.get_last_3_samples()
        
        return {
            'total_samples': len(self.sample_data),
            'first_3_count': len(first_3),
            'last_3_count': len(last_3),
            'sampling_method': 'first_3_plus_last_3',
            'has_position_markers': any('__sample_position__' in row for row in self.sample_data)
        }
    
    def is_customer_table(self) -> bool:
        """Check if this is a customer-related table"""
        return (self.entity_type == 'Customer' or 
                'customer' in self.name.lower() or
                'client' in self.name.lower())
    
    def is_payment_table(self) -> bool:
        """Check if this is a payment/transaction table"""
        return (self.entity_type == 'Payment' or
                any(word in self.name.lower() for word in ['payment', 'transaction', 'invoice', 'billing']))
    
    def is_fact_table(self) -> bool:
        """Check if this is a fact table with measures"""
        return self.bi_role == 'fact' and len(self.measures) > 0
    
    def get_display_columns(self) -> List[str]:
        """Get columns suitable for display (names, titles, etc.)"""
        display_cols = self.name_columns.copy()
        
        # Add other potential display columns
        for col in self.columns:
            col_name = col.get('name', '').lower()
            if (col_name not in [c.lower() for c in display_cols] and
                any(word in col_name for word in ['description', 'title', 'label'])):
                display_cols.append(col.get('name', ''))
        
        return display_cols[:3]  # Limit to top 3
    
    def get_amount_columns(self) -> List[str]:
        """Get columns that represent monetary amounts"""
        amount_cols = []
        for measure in self.measures:
            if any(word in measure.lower() for word in ['amount', 'total', 'price', 'revenue', 'cost']):
                amount_cols.append(measure)
        return amount_cols

@dataclass
class ViewInfo:
    """Enhanced view information with definition storage"""
    schema: str
    name: str
    full_name: str
    definition: str
    create_date: Optional[str] = None
    modify_date: Optional[str] = None
    referenced_objects: List[str] = field(default_factory=list)
    parsed_joins: List[Dict[str, Any]] = field(default_factory=list)
    parsing_success: bool = False
    query_type: str = "unknown"
    
    def get_definition_summary(self) -> Dict[str, Any]:
        """Get summary of view definition"""
        return {
            'name': self.full_name,
            'query_type': self.query_type,
            'definition_length': len(self.definition) if self.definition else 0,
            'referenced_tables': len(self.referenced_objects),
            'join_count': len(self.parsed_joins),
            'parsing_successful': self.parsing_success
        }

@dataclass
class StoredProcedureInfo:
    """Enhanced stored procedure information with definition storage"""
    schema: str
    name: str
    full_name: str
    definition: str
    create_date: Optional[str] = None
    modify_date: Optional[str] = None
    type_desc: str = "SQL_STORED_PROCEDURE"
    referenced_objects: List[str] = field(default_factory=list)
    select_statements: List[str] = field(default_factory=list)
    parsing_success: bool = False
    procedure_type: str = "unknown"
    
    def get_definition_summary(self) -> Dict[str, Any]:
        """Get summary of procedure definition"""
        return {
            'name': self.full_name,
            'procedure_type': self.procedure_type,
            'definition_length': len(self.definition) if self.definition else 0,
            'referenced_tables': len(self.referenced_objects),
            'select_statements_count': len(self.select_statements),
            'parsing_successful': self.parsing_success
        }

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
    name_columns: List[str] = field(default_factory=list)  # Added for customer names
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
    
    def is_customer_query_ready(self) -> bool:
        """Check if ready for customer-related queries"""
        return (self.is_complete() and 
                (len(self.name_columns) > 0 or len(self.entity_keys) > 0))
    
    def is_payment_query_ready(self) -> bool:
        """Check if ready for payment/revenue queries"""
        return (self.is_complete() and 
                len(self.measures) > 0 and
                len(self.entity_keys) > 0)
    
    def get_completeness_score(self) -> float:
        """Get contract completeness as percentage"""
        checks = [
            bool(self.grain and self.grain != 'unknown'),
            bool(self.measures or self.entity_keys),
            bool(self.time_column),
            bool(self.entity_keys),
            bool(self.name_columns),  # Added name columns check
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
        
        if not self.name_columns:
            missing.append("Name/title columns for display")
        
        quality = self.quality_checks
        if quality.get('row_count', 0) == 0:
            missing.append("Data availability (zero rows)")
        
        return missing

@dataclass
class AnalyticalTask:
    """Enhanced analytical task from natural language"""
    task_type: str  # aggregation, ranking, trend, distribution, cohort, funnel
    metrics: List[str] = field(default_factory=list)
    entity: Optional[str] = None
    crm_entities: List[str] = field(default_factory=list)  # Added for enhanced entity support
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
    
    def requires_customer_data(self) -> bool:
        """Check if task requires customer information"""
        return ('customer' in self.crm_entities or 
                'customer' in str(self.entity).lower() if self.entity else False)
    
    def requires_payment_data(self) -> bool:
        """Check if task requires payment/revenue information"""
        return ('payment' in self.crm_entities or
                any(metric in ['revenue', 'amount', 'payment'] for metric in self.metrics))
    
    def requires_names(self) -> bool:
        """Check if task requires name/title columns"""
        return 'name' in self.grouping or any('name' in g for g in self.grouping)
    
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
        if len(self.crm_entities) > 1:
            complexity += 0.1
        
        return min(1.0, complexity)

@dataclass
class EvidenceScore:
    """Enhanced Evidence-based scoring for table selection"""
    role_match: float = 0.0
    join_evidence: float = 0.0
    lexical_match: float = 0.0
    graph_proximity: float = 0.0  # Table quality score
    operational_tag: float = 0.0
    row_count: float = 0.0
    freshness: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Calculate weighted total evidence score - Enhanced weights"""
        weights = {
            'role_match': 2.5,        # BI role importance
            'lexical_match': 4.5,     # Entity matching (highest priority)  
            'graph_proximity': 4.0,   # Table quality (very high priority)
            'operational_tag': 2.0,   # Operational data preference
            'join_evidence': 1.5,     # Relationship connectivity
            'row_count': 1.0,         # Data volume (low priority)
            'freshness': 0.5          # Recency (lowest priority)
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
        
        if self.lexical_match > 0.8:
            explanations.append("Excellent entity match to query intent")
        elif self.lexical_match > 0.6:
            explanations.append("Good entity match to query intent")
        elif self.lexical_match < 0.3:
            explanations.append("⚠️ Weak entity match to query")
            
        if self.role_match > 0.7:
            explanations.append("Strong BI role match (fact/operational table)")
        elif self.role_match < 0.4:
            explanations.append("⚠️ BI role mismatch")
            
        if self.graph_proximity > 0.8:
            explanations.append("High-quality business table")
        elif self.graph_proximity < 0.5:
            explanations.append("⚠️ Lower quality table (temp/test/dated)")
            
        if self.operational_tag > 0.8:
            explanations.append("Contains operational (transactional) data")
            
        if self.row_count > 0.5:
            explanations.append("Sufficient data volume")
        elif self.row_count < 0.1:
            explanations.append("⚠️ Limited data volume")
        
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

# Enhanced discovery result container
@dataclass
class DiscoveryResult:
    """Container for complete discovery results"""
    tables: List[TableInfo] = field(default_factory=list)
    views: Dict[str, ViewInfo] = field(default_factory=dict)
    stored_procedures: Dict[str, StoredProcedureInfo] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    discovery_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get discovery summary statistics"""
        customer_tables = len([t for t in self.tables if t.is_customer_table()])
        payment_tables = len([t for t in self.tables if t.is_payment_table()])
        
        return {
            'total_tables': len(self.tables),
            'total_views': len(self.views),
            'total_procedures': len(self.stored_procedures),
            'total_relationships': len(self.relationships),
            'customer_tables': customer_tables,
            'payment_tables': payment_tables,
            'fact_tables': len([t for t in self.tables if t.is_fact_table()]),
            'operational_tables': len([t for t in self.tables if getattr(t, 'data_type', '') == 'operational']),
            'tables_with_samples': len([t for t in self.tables if t.sample_data]),
            'tables_with_names': len([t for t in self.tables if t.name_columns]),
            'sampling_method': 'first_3_plus_last_3'
        }

# Type aliases for better code readability
TableList = List[TableInfo]
RelationshipList = List[Relationship]
QueryCapabilityResult = Union[TableList, NonExecutableAnalysisReport]

# Enhanced utility functions for customer/payment analysis
def find_customer_tables(tables: TableList) -> TableList:
    """Find all customer-related tables"""
    return [t for t in tables if t.is_customer_table()]

def find_payment_tables(tables: TableList) -> TableList:
    """Find all payment/transaction tables"""
    return [t for t in tables if t.is_payment_table()]

def find_tables_with_names(tables: TableList) -> TableList:
    """Find tables that have name/title columns"""
    return [t for t in tables if t.name_columns]

def find_tables_with_amounts(tables: TableList) -> TableList:
    """Find tables with monetary amount columns"""
    return [t for t in tables if t.get_amount_columns()]

def suggest_customer_payment_join(customer_tables: TableList, payment_tables: TableList) -> List[Tuple[TableInfo, TableInfo, str]]:
    """Suggest possible joins between customer and payment tables"""
    joins = []
    
    for customer_table in customer_tables:
        for payment_table in payment_tables:
            # Look for common entity keys
            customer_keys = set(key.lower() for key in customer_table.entity_keys)
            payment_keys = set(key.lower() for key in payment_table.entity_keys)
            
            common_keys = customer_keys.intersection(payment_keys)
            if common_keys:
                common_key = list(common_keys)[0]
                joins.append((customer_table, payment_table, f"Common key: {common_key}"))
            elif any('customer' in key.lower() for key in payment_table.entity_keys):
                joins.append((customer_table, payment_table, "Customer foreign key detected"))
    
    return joins

def calculate_enhanced_bi_readiness(tables: TableList) -> Dict[str, Any]:
    """Calculate enhanced BI readiness focusing on customer/payment capabilities"""
    if not tables:
        return {'readiness_score': 0.0, 'issues': ['No tables available']}
    
    customer_tables = find_customer_tables(tables)
    payment_tables = find_payment_tables(tables)
    tables_with_names = find_tables_with_names(tables)
    tables_with_amounts = find_tables_with_amounts(tables)
    fact_tables = [t for t in tables if t.is_fact_table()]
    
    # Enhanced readiness factors
    factors = {
        'has_customer_data': len(customer_tables) > 0,
        'has_payment_data': len(payment_tables) > 0,
        'has_customer_names': len(tables_with_names) > 0,
        'has_amount_measures': len(tables_with_amounts) > 0,
        'has_fact_tables': len(fact_tables) > 0,
        'sufficient_volume': sum(t.row_count for t in tables) > 1000,
        'customer_payment_linkable': len(suggest_customer_payment_join(customer_tables, payment_tables)) > 0
    }
    
    readiness_score = sum(factors.values()) / len(factors)
    
    # Identify issues
    issues = []
    if not factors['has_customer_data']:
        issues.append("No customer tables identified")
    if not factors['has_payment_data']:
        issues.append("No payment/transaction tables found")
    if not factors['has_customer_names']:
        issues.append("No customer name columns available")
    if not factors['has_amount_measures']:
        issues.append("No monetary amount measures found")
    if not factors['customer_payment_linkable']:
        issues.append("Cannot link customers to payments")
    
    return {
        'readiness_score': readiness_score,
        'factors': factors,
        'issues': issues,
        'customer_table_count': len(customer_tables),
        'payment_table_count': len(payment_tables),
        'total_data_volume': sum(t.row_count for t in tables),
        'customer_payment_joins': len(suggest_customer_payment_join(customer_tables, payment_tables)),
        'top_customer_tables': [t.name for t in customer_tables[:3]],
        'top_payment_tables': [t.name for t in payment_tables[:3]]
    }

# Enhanced table creation helpers
def create_enhanced_customer_table(name: str, schema: str, 
                                 name_columns: List[str], entity_keys: List[str],
                                 sample_data: List[Dict[str, Any]]) -> TableInfo:
    """Create a properly configured customer table"""
    table = TableInfo(
        name=name,
        schema=schema,
        full_name=f"[{schema}].[{name}]",
        object_type="BASE TABLE",
        row_count=len(sample_data),
        columns=[],  # Would be populated from discovery
        sample_data=sample_data,
        entity_type="Customer",
        data_type="reference",
        bi_role="dimension",
        grain="customer",
        entity_keys=entity_keys,
        name_columns=name_columns,
        business_priority="high"
    )
    return table

def create_enhanced_payment_table(name: str, schema: str, 
                                measures: List[str], entity_keys: List[str], 
                                time_columns: List[str],
                                sample_data: List[Dict[str, Any]]) -> TableInfo:
    """Create a properly configured payment table"""
    table = TableInfo(
        name=name,
        schema=schema,
        full_name=f"[{schema}].[{name}]",
        object_type="BASE TABLE",
        row_count=len(sample_data),
        columns=[],  # Would be populated from discovery
        sample_data=sample_data,
        entity_type="Payment",
        data_type="operational",
        bi_role="fact",
        grain="transaction",
        measures=measures,
        entity_keys=entity_keys,
        time_columns=time_columns,
        business_priority="high"
    )
    return table