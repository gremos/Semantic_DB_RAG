#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data models and structures
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, NamedTuple
from datetime import datetime

@dataclass
class TableInfo:
    """Enhanced table information structure"""
    name: str
    schema: str
    full_name: str
    object_type: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    semantic_profile: Optional['SemanticProfile'] = None
    relationships: List[Dict[str, Any]] = None
    query_performance: Optional[Dict[str, Any]] = None

@dataclass
class SemanticProfile:
    """Semantic profile for database entities"""
    entity_type: str
    business_role: str
    data_nature: str
    contains_personal_data: bool
    contains_financial_data: bool
    primary_purpose: str
    confidence: float

@dataclass
class BusinessDomain:
    """Business domain information"""
    domain_type: str
    industry: str
    entities: List[str]
    confidence: float
    sample_questions: List[str]
    customer_definition: str = ""

@dataclass
class Relationship:
    """Database relationship information"""
    from_table: str
    to_table: str
    column: str
    relationship_type: str
    confidence: float
    description: str = ""

class DatabaseObject(NamedTuple):
    """Database object information for discovery"""
    schema: str
    name: str
    object_type: str
    estimated_rows: int

@dataclass
class SystemStatus:
    """System status tracking"""
    discovery_completed: bool = False
    analysis_completed: bool = False
    tables_discovered: int = 0
    relationships_found: int = 0
    domain_identified: bool = False
    cache_available: bool = False

@dataclass
class QueryResult:
    """Query execution result"""
    question: str
    relevant_tables: List[str]
    sql_query: str
    results: List[Dict[str, Any]]
    results_count: int
    execution_error: Optional[str] = None
    execution_time: float = 0.0

class AnalysisStats:
    """Statistics tracking for analysis operations"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_objects_found = 0
        self.objects_processed = 0
        self.objects_excluded = 0
        self.backup_tables_excluded = 0
        self.successful_analyses = 0
        self.analysis_errors = 0
        self.sample_data_errors = 0
        self.fast_query_successes = 0
        self.view_estimation_improved = True
        self.start_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_objects_found': self.total_objects_found,
            'objects_processed': self.objects_processed,
            'objects_excluded': self.objects_excluded,
            'backup_tables_excluded': self.backup_tables_excluded,
            'successful_analyses': self.successful_analyses,
            'analysis_errors': self.analysis_errors,
            'sample_data_errors': self.sample_data_errors,
            'fast_query_successes': self.fast_query_successes,
            'view_estimation_improved': self.view_estimation_improved,
            'analysis_duration': (datetime.now() - self.start_time).total_seconds()
        }

# Utility functions for model conversion
def table_info_to_dict(table: TableInfo) -> Dict[str, Any]:
    """Convert TableInfo to dictionary for JSON serialization"""
    result = asdict(table)
    if table.semantic_profile:
        result['semantic_profile'] = asdict(table.semantic_profile)
    return result

def dict_to_table_info(data: Dict[str, Any]) -> TableInfo:
    """Convert dictionary back to TableInfo"""
    if 'semantic_profile' in data and data['semantic_profile']:
        data['semantic_profile'] = SemanticProfile(**data['semantic_profile'])
    return TableInfo(**data)

def business_domain_to_dict(domain: BusinessDomain) -> Dict[str, Any]:
    """Convert BusinessDomain to dictionary"""
    return asdict(domain)

def dict_to_business_domain(data: Dict[str, Any]) -> BusinessDomain:
    """Convert dictionary back to BusinessDomain"""
    return BusinessDomain(**data)