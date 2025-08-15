#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Data Models - Proper type handling and validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

@dataclass
class DatabaseObject:
    """Simple database object for discovery with type safety"""
    schema: str
    name: str
    object_type: str
    estimated_rows: int = 0
    
    def __post_init__(self):
        """Validate and convert types after initialization"""
        try:
            self.schema = str(self.schema) if self.schema else 'dbo'
            self.name = str(self.name) if self.name else 'unknown'
            self.object_type = str(self.object_type) if self.object_type else 'TABLE'
            self.estimated_rows = int(self.estimated_rows) if self.estimated_rows else 0
        except (ValueError, TypeError):
            self.schema = 'dbo'
            self.name = 'unknown'
            self.object_type = 'TABLE'
            self.estimated_rows = 0
    
    @property
    def full_name(self) -> str:
        """Get full table name with proper formatting"""
        try:
            return f"[{self.schema}].[{self.name}]"
        except:
            return "[dbo].[unknown]"

@dataclass
class TableInfo:
    """Table information with type safety"""
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
    
    def __post_init__(self):
        """Validate and convert types after initialization"""
        try:
            self.name = str(self.name) if self.name else 'unknown'
            self.schema = str(self.schema) if self.schema else 'dbo'
            self.full_name = str(self.full_name) if self.full_name else f"[{self.schema}].[{self.name}]"
            self.object_type = str(self.object_type) if self.object_type else 'TABLE'
            self.row_count = int(self.row_count) if self.row_count else 0
            self.entity_type = str(self.entity_type) if self.entity_type else 'Unknown'
            self.business_role = str(self.business_role) if self.business_role else 'Unknown'
            self.confidence = float(self.confidence) if self.confidence else 0.0
            
            # Ensure lists are actually lists
            if not isinstance(self.columns, list):
                self.columns = []
            if not isinstance(self.sample_data, list):
                self.sample_data = []
            if not isinstance(self.relationships, list):
                self.relationships = []
                
        except (ValueError, TypeError) as e:
            print(f"Warning: Error in TableInfo validation: {e}")
            # Set safe defaults
            self.name = 'unknown'
            self.schema = 'dbo'
            self.full_name = '[dbo].[unknown]'
            self.object_type = 'TABLE'
            self.row_count = 0
            self.columns = []
            self.sample_data = []
            self.relationships = []
            self.entity_type = 'Unknown'
            self.business_role = 'Unknown'
            self.confidence = 0.0
    
    def get_column_names(self) -> List[str]:
        """Get list of column names safely"""
        try:
            names = []
            for col in self.columns:
                if isinstance(col, dict) and 'name' in col:
                    names.append(str(col['name']))
            return names
        except:
            return []
    
    def has_data(self) -> bool:
        """Check if table has sample data"""
        try:
            return len(self.sample_data) > 0
        except:
            return False

@dataclass
class BusinessDomain:
    """Business domain information with type safety"""
    domain_type: str
    industry: str
    confidence: float
    sample_questions: List[str]
    capabilities: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and convert types after initialization"""
        try:
            self.domain_type = str(self.domain_type) if self.domain_type else 'Business'
            self.industry = str(self.industry) if self.industry else 'Business'
            self.confidence = float(self.confidence) if self.confidence else 0.0
            
            if not isinstance(self.sample_questions, list):
                self.sample_questions = []
            else:
                # Ensure all questions are strings
                self.sample_questions = [str(q) for q in self.sample_questions if q]
            
            if not isinstance(self.capabilities, dict):
                self.capabilities = {}
                
        except (ValueError, TypeError):
            self.domain_type = 'Business'
            self.industry = 'Business'
            self.confidence = 0.0
            self.sample_questions = []
            self.capabilities = {}

@dataclass
class Relationship:
    """Database relationship with type safety"""
    from_table: str
    to_table: str
    relationship_type: str
    confidence: float
    description: str = ""
    
    def __post_init__(self):
        """Validate and convert types after initialization"""
        try:
            self.from_table = str(self.from_table) if self.from_table else 'unknown'
            self.to_table = str(self.to_table) if self.to_table else 'unknown'
            self.relationship_type = str(self.relationship_type) if self.relationship_type else 'unknown'
            self.confidence = float(self.confidence) if self.confidence else 0.0
            self.description = str(self.description) if self.description else ''
        except (ValueError, TypeError):
            self.from_table = 'unknown'
            self.to_table = 'unknown'
            self.relationship_type = 'unknown'
            self.confidence = 0.0
            self.description = ''

@dataclass
class QueryResult:
    """Query execution result with type safety"""
    question: str
    sql_query: str
    results: List[Dict[str, Any]]
    error: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    def __post_init__(self):
        """Validate and convert types after initialization"""
        try:
            self.question = str(self.question) if self.question else ''
            self.sql_query = str(self.sql_query) if self.sql_query else ''
            self.error = str(self.error) if self.error else None
            self.execution_time = float(self.execution_time) if self.execution_time else 0.0
            
            if not isinstance(self.results, list):
                self.results = []
            if not isinstance(self.tables_used, list):
                self.tables_used = []
            else:
                # Ensure all table names are strings
                self.tables_used = [str(t) for t in self.tables_used if t]
                
        except (ValueError, TypeError):
            self.question = ''
            self.sql_query = ''
            self.results = []
            self.error = 'Error in query result validation'
            self.tables_used = []
            self.execution_time = 0.0
    
    def is_successful(self) -> bool:
        """Check if query was successful"""
        return self.error is None
    
    def has_results(self) -> bool:
        """Check if query returned results"""
        try:
            return len(self.results) > 0
        except:
            return False
    
    def is_single_value(self) -> bool:
        """Check if result is a single value"""
        try:
            return (len(self.results) == 1 and 
                    isinstance(self.results[0], dict) and 
                    len(self.results[0]) == 1)
        except:
            return False
    
    def get_single_value(self) -> Any:
        """Get single value result"""
        try:
            if self.is_single_value():
                return list(self.results[0].values())[0]
        except:
            pass
        return None