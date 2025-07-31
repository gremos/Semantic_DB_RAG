#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified shared data models - Readable and Maintainable
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class TableInfo:
    """Simplified table information"""
    name: str
    schema: str
    full_name: str
    object_type: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    
    # Semantic information
    entity_type: str = "Unknown"
    business_role: str = "Unknown"
    confidence: float = 0.0
    
    # Relationships
    relationships: List[str] = field(default_factory=list)
    connected_tables: List[str] = field(default_factory=list)
    
    # For compatibility with existing code
    semantic_profile: Optional['SemanticProfile'] = None
    query_performance: Optional[Dict[str, Any]] = None

@dataclass
class SemanticProfile:
    """Semantic profile for database entities"""
    entity_type: str
    business_role: str
    data_nature: str = "Unknown"
    contains_personal_data: bool = False
    contains_financial_data: bool = False
    primary_purpose: str = ""
    confidence: float = 0.0

@dataclass
class BusinessDomain:
    """Business domain information"""
    domain_type: str
    industry: str
    confidence: float
    sample_questions: List[str]
    capabilities: Dict[str, bool] = field(default_factory=dict)

@dataclass
class Relationship:
    """Database relationship"""
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

@dataclass 
class SystemStatus:
    """System status tracking"""
    tables_found: int = 0
    entities_classified: int = 0
    relationships_found: int = 0
    ready_for_queries: bool = False

class DatabaseObject:
    """Database object information for discovery"""
    
    def __init__(self, schema: str, name: str, object_type: str, estimated_rows: int = 0):
        self.schema = schema
        self.name = name
        self.object_type = object_type
        self.estimated_rows = estimated_rows
        self.priority_score = self._calculate_priority_score()
    
    def _calculate_priority_score(self) -> int:
        """Calculate priority score for processing order"""
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
        high_value_patterns = [
            'customer', 'client', 'account', 'user',
            'payment', 'transaction', 'invoice', 'billing',
            'order', 'sale', 'purchase', 'product', 'item'
        ]
        
        for pattern in high_value_patterns:
            if pattern in name_lower:
                score += 50
                break
        
        return max(score, 0)

class AnalysisStats:
    """Statistics tracking for analysis operations"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_objects_found = 0
        self.objects_processed = 0
        self.objects_excluded = 0
        self.successful_analyses = 0
        self.analysis_errors = 0
        self.start_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_objects_found': self.total_objects_found,
            'objects_processed': self.objects_processed,
            'objects_excluded': self.objects_excluded,
            'successful_analyses': self.successful_analyses,
            'analysis_errors': self.analysis_errors,
            'analysis_duration': (datetime.now() - self.start_time).total_seconds()
        }

# Utility functions
def table_to_dict(table: TableInfo) -> Dict[str, Any]:
    """Convert TableInfo to dictionary"""
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
        'connected_tables': table.connected_tables
    }

def dict_to_table(data: Dict[str, Any]) -> TableInfo:
    """Convert dictionary to TableInfo"""
    return TableInfo(
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
        connected_tables=data.get('connected_tables', [])
    )

# For backward compatibility with existing code
def table_info_to_dict(table: TableInfo) -> Dict[str, Any]:
    """Backward compatibility function"""
    return table_to_dict(table)

def dict_to_table_info(data: Dict[str, Any]) -> TableInfo:
    """Backward compatibility function"""
    return dict_to_table(data)

def business_domain_to_dict(domain: BusinessDomain) -> Dict[str, Any]:
    """Convert BusinessDomain to dictionary"""
    return {
        'domain_type': domain.domain_type,
        'industry': domain.industry,
        'confidence': domain.confidence,
        'sample_questions': domain.sample_questions,
        'capabilities': domain.capabilities
    }

def dict_to_business_domain(data: Dict[str, Any]) -> BusinessDomain:
    """Convert dictionary to BusinessDomain"""
    return BusinessDomain(
        domain_type=data.get('domain_type', ''),
        industry=data.get('industry', ''),
        confidence=data.get('confidence', 0.0),
        sample_questions=data.get('sample_questions', []),
        capabilities=data.get('capabilities', {})
    )