#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Models - Simple and Maintainable
Supports all features from README while keeping models clean
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class TableInfo:
    """Enhanced table information with semantic properties"""
    
    # Basic table properties
    name: str
    schema: str
    full_name: str
    object_type: str  # 'BASE TABLE', 'TABLE', 'VIEW'
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    
    # Semantic properties (added by semantic analysis)
    entity_type: str = "Unknown"  # Customer, Payment, Order, Product, etc.
    business_role: str = "Unknown"  # Core, Supporting, Reference, System
    confidence: float = 0.0  # Confidence in classification (0.0-1.0)
    
    # Relationship information
    relationships: List[str] = field(default_factory=list)  # Foreign keys
    connected_tables: List[str] = field(default_factory=list)  # Related tables
    
    # Query performance metadata
    query_performance: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize computed properties"""
        if self.query_performance is None:
            self.query_performance = {}
    
    def is_classified(self) -> bool:
        """Check if table has been semantically classified"""
        return self.entity_type != "Unknown" and self.confidence > 0.0
    
    def is_high_confidence(self) -> bool:
        """Check if classification has high confidence"""
        return self.confidence >= 0.8
    
    def get_column_names(self) -> List[str]:
        """Get list of column names"""
        return [col.get('name', '') for col in self.columns]
    
    def has_data(self) -> bool:
        """Check if table has sample data"""
        return len(self.sample_data) > 0
    
    def get_primary_keys(self) -> List[str]:
        """Get primary key columns"""
        return [col['name'] for col in self.columns if col.get('is_primary_key', False)]

@dataclass
class BusinessDomain:
    """Business domain information with capabilities"""
    
    domain_type: str  # E-Commerce, CRM/Sales, Financial Services, etc.
    industry: str  # Business, Healthcare, Retail, etc.
    confidence: float  # Confidence in domain identification (0.0-1.0)
    sample_questions: List[str]  # Sample questions users can ask
    capabilities: Dict[str, bool] = field(default_factory=dict)  # Available query types
    
    def get_enabled_capabilities(self) -> List[str]:
        """Get list of enabled capabilities"""
        return [cap for cap, enabled in self.capabilities.items() if enabled]
    
    def supports_capability(self, capability: str) -> bool:
        """Check if domain supports a specific capability"""
        return self.capabilities.get(capability, False)

@dataclass
class Relationship:
    """Database relationship with confidence scoring"""
    
    from_table: str
    to_table: str
    relationship_type: str  # foreign_key, pattern_match, view_join, entity_reference
    confidence: float  # Confidence in relationship (0.0-1.0)
    description: str = ""
    
    def is_high_confidence(self) -> bool:
        """Check if relationship has high confidence"""
        return self.confidence >= 0.8
    
    def is_foreign_key(self) -> bool:
        """Check if relationship is a foreign key"""
        return self.relationship_type == 'foreign_key'

@dataclass
class QueryResult:
    """Query execution result with metadata"""
    
    question: str
    sql_query: str
    results: List[Dict[str, Any]]
    error: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    # 4-stage pipeline metadata
    pipeline_stages: Optional[Dict[str, Any]] = None
    business_validation: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed properties"""
        if self.pipeline_stages is None:
            self.pipeline_stages = {}
    
    def is_successful(self) -> bool:
        """Check if query was successful"""
        return self.error is None
    
    def has_results(self) -> bool:
        """Check if query returned results"""
        return len(self.results) > 0
    
    def is_single_value(self) -> bool:
        """Check if result is a single value (count, sum, etc.)"""
        return (len(self.results) == 1 and 
                len(self.results[0]) == 1)
    
    def get_single_value(self) -> Any:
        """Get single value result"""
        if self.is_single_value():
            return list(self.results[0].values())[0]
        return None
    
    def get_result_summary(self) -> str:
        """Get human-readable result summary"""
        if self.error:
            return f"Error: {self.error}"
        elif self.is_single_value():
            value = self.get_single_value()
            column_name = list(self.results[0].keys())[0]
            if isinstance(value, (int, float)):
                return f"{column_name}: {value:,}"
            else:
                return f"{column_name}: {value}"
        elif self.has_results():
            return f"{len(self.results)} rows returned"
        else:
            return "No results"

@dataclass
class DatabaseObject:
    """Database object for discovery phase"""
    
    schema: str
    name: str
    object_type: str  # 'BASE TABLE', 'VIEW'
    estimated_rows: int = 0
    priority_score: int = 0
    
    def __post_init__(self):
        """Calculate priority score for processing order"""
        self.priority_score = self._calculate_priority()
    
    def _calculate_priority(self) -> int:
        """Calculate priority score based on business relevance"""
        score = 0
        name_lower = self.name.lower()
        
        # Base score by object type
        if self.object_type in ['BASE TABLE', 'TABLE']:
            score += 100
        elif self.object_type == 'VIEW':
            score += 80
        
        # Data volume consideration
        if self.estimated_rows > 0:
            score += min(50, self.estimated_rows // 10000)
        
        # Business entity recognition
        high_value_keywords = [
            'customer', 'client', 'account', 'user',
            'payment', 'transaction', 'invoice', 'billing',
            'order', 'sale', 'purchase', 'product', 'item'
        ]
        
        for keyword in high_value_keywords:
            if keyword in name_lower:
                score += 50
                break
        
        return max(score, 0)
    
    @property
    def full_name(self) -> str:
        """Get full table name"""
        return f"[{self.schema}].[{self.name}]"

@dataclass
class AnalysisStats:
    """Statistics tracking for analysis operations"""
    
    start_time: datetime = field(default_factory=datetime.now)
    total_objects_found: int = 0
    objects_processed: int = 0
    objects_excluded: int = 0
    successful_analyses: int = 0
    analysis_errors: int = 0
    
    def reset(self):
        """Reset statistics"""
        self.start_time = datetime.now()
        self.total_objects_found = 0
        self.objects_processed = 0
        self.objects_excluded = 0
        self.successful_analyses = 0
        self.analysis_errors = 0
    
    def get_success_rate(self) -> float:
        """Get analysis success rate"""
        if self.objects_processed == 0:
            return 0.0
        return (self.successful_analyses / self.objects_processed) * 100
    
    def get_duration(self) -> float:
        """Get analysis duration in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_objects_found': self.total_objects_found,
            'objects_processed': self.objects_processed,
            'objects_excluded': self.objects_excluded,
            'successful_analyses': self.successful_analyses,
            'analysis_errors': self.analysis_errors,
            'success_rate': self.get_success_rate(),
            'duration_seconds': self.get_duration()
        }

@dataclass
class SystemStatus:
    """Overall system status"""
    
    discovery_completed: bool = False
    semantic_analysis_completed: bool = False
    tables_found: int = 0
    entities_classified: int = 0
    relationships_found: int = 0
    domain_identified: bool = False
    ready_for_queries: bool = False
    
    def update_from_discovery(self, tables: List[TableInfo], relationships: List[Relationship]):
        """Update status from discovery results"""
        self.discovery_completed = True
        self.tables_found = len(tables)
        self.relationships_found = len(relationships)
        self._check_ready_status()
    
    def update_from_semantic_analysis(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Update status from semantic analysis results"""
        self.semantic_analysis_completed = True
        self.entities_classified = sum(1 for t in tables if t.is_classified())
        self.domain_identified = domain is not None
        self._check_ready_status()
    
    def _check_ready_status(self):
        """Check if system is ready for queries"""
        self.ready_for_queries = (
            self.discovery_completed and 
            self.tables_found > 0 and
            (self.semantic_analysis_completed or self.entities_classified > 0)
        )
    
    def get_summary(self) -> str:
        """Get status summary"""
        status_items = []
        
        if self.discovery_completed:
            status_items.append(f"📊 {self.tables_found} tables discovered")
        else:
            status_items.append("📊 Discovery pending")
        
        if self.semantic_analysis_completed:
            status_items.append(f"🧠 {self.entities_classified} entities classified")
        else:
            status_items.append("🧠 Semantic analysis pending")
        
        if self.relationships_found > 0:
            status_items.append(f"🔗 {self.relationships_found} relationships")
        
        if self.ready_for_queries:
            status_items.append("✅ Ready for queries")
        else:
            status_items.append("⏳ Not ready for queries")
        
        return " | ".join(status_items)

# Business template for common query patterns
@dataclass
class BusinessTemplate:
    """Template for common business queries"""
    
    name: str
    description: str
    required_entities: List[str]
    sql_pattern: str
    sample_questions: List[str]
    confidence_threshold: float = 0.7
    
    def can_execute(self, available_entities: Dict[str, float]) -> bool:
        """Check if template can be executed with available entities"""
        for entity in self.required_entities:
            if entity not in available_entities:
                return False
            if available_entities[entity] < self.confidence_threshold:
                return False
        return True

# Utility functions for model conversion
def table_to_dict(table: TableInfo) -> Dict[str, Any]:
    """Convert TableInfo to dictionary for serialization"""
    return {
        'name': table.name,
        'schema': table.schema,
        'full_name': table.full_name,
        'object_type': table.object_type,
        'row_count': table.row_count,
        'columns': table.columns,
        'sample_data': table.sample_data,
        'entity_type': table.entity_type,
        'business_role': table.business_role,
        'confidence': table.confidence,
        'relationships': table.relationships,
        'connected_tables': table.connected_tables,
        'query_performance': table.query_performance
    }

def dict_to_table(data: Dict[str, Any]) -> TableInfo:
    """Convert dictionary to TableInfo"""
    table = TableInfo(
        name=data.get('name', ''),
        schema=data.get('schema', ''),
        full_name=data.get('full_name', ''),
        object_type=data.get('object_type', ''),
        row_count=data.get('row_count', 0),
        columns=data.get('columns', []),
        sample_data=data.get('sample_data', []),
        entity_type=data.get('entity_type', 'Unknown'),
        business_role=data.get('business_role', 'Unknown'),
        confidence=data.get('confidence', 0.0),
        relationships=data.get('relationships', []),
        connected_tables=data.get('connected_tables', []),
        query_performance=data.get('query_performance', {})
    )
    return table

def domain_to_dict(domain: BusinessDomain) -> Dict[str, Any]:
    """Convert BusinessDomain to dictionary"""
    return {
        'domain_type': domain.domain_type,
        'industry': domain.industry,
        'confidence': domain.confidence,
        'sample_questions': domain.sample_questions,
        'capabilities': domain.capabilities
    }

def dict_to_domain(data: Dict[str, Any]) -> BusinessDomain:
    """Convert dictionary to BusinessDomain"""
    return BusinessDomain(
        domain_type=data.get('domain_type', ''),
        industry=data.get('industry', ''),
        confidence=data.get('confidence', 0.0),
        sample_questions=data.get('sample_questions', []),
        capabilities=data.get('capabilities', {})
    )

def relationship_to_dict(relationship: Relationship) -> Dict[str, Any]:
    """Convert Relationship to dictionary"""
    return {
        'from_table': relationship.from_table,
        'to_table': relationship.to_table,
        'relationship_type': relationship.relationship_type,
        'confidence': relationship.confidence,
        'description': relationship.description
    }

def dict_to_relationship(data: Dict[str, Any]) -> Relationship:
    """Convert dictionary to Relationship"""
    return Relationship(
        from_table=data.get('from_table', ''),
        to_table=data.get('to_table', ''),
        relationship_type=data.get('relationship_type', ''),
        confidence=data.get('confidence', 0.0),
        description=data.get('description', '')
    )

# For backward compatibility
table_info_to_dict = table_to_dict
dict_to_table_info = dict_to_table
business_domain_to_dict = domain_to_dict
dict_to_business_domain = dict_to_domain