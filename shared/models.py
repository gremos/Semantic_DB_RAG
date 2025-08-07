#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Data Models
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

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
    """Table information"""
    name: str
    schema: str
    full_name: str
    object_type: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    relationships: List[str] = field(default_factory=list)
    
    # Semantic properties (added by analysis)
    entity_type: str = "Unknown"
    business_role: str = "Unknown"
    confidence: float = 0.0
    
    def get_column_names(self) -> List[str]:
        return [col.get('name', '') for col in self.columns]
    
    def has_data(self) -> bool:
        return len(self.sample_data) > 0

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
    
    def is_successful(self) -> bool:
        return self.error is None
    
    def has_results(self) -> bool:
        return len(self.results) > 0
    
    def is_single_value(self) -> bool:
        return len(self.results) == 1 and len(self.results[0]) == 1
    
    def get_single_value(self) -> Any:
        if self.is_single_value():
            return list(self.results[0].values())[0]
        return None