#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced shared data models and structures with comprehensive relationship discovery support
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class TableInfo:
    """Enhanced table information structure with relationship context"""
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
    
    # Enhanced relationship and business context
    business_indicators: List[str] = field(default_factory=list)
    sample_questions: List[str] = field(default_factory=list)
    relationship_context: Dict[str, Any] = field(default_factory=dict)
    business_significance: str = "Supporting"  # Core, Supporting, Reference
    constraint_info: List[Dict[str, Any]] = field(default_factory=list)

    # Enhanced comprehensive analysis fields
    comprehensive_entity_type: Optional[str] = None
    entity_confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    discovered_relationships: List[str] = field(default_factory=list)
    view_dependencies: List[str] = field(default_factory=list)
    foreign_key_references: List[str] = field(default_factory=list)
    llm_insights: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticProfile:
    """Enhanced semantic profile for database entities with relationship awareness"""
    entity_type: str
    business_role: str
    data_nature: str
    contains_personal_data: bool
    contains_financial_data: bool
    primary_purpose: str
    confidence: float
    
    # Enhanced relationship and business intelligence
    relationship_strength: str = "Unknown"  # Strong, Moderate, Weak
    central_entity: bool = False
    connected_entities: List[str] = field(default_factory=list)
    business_process_role: str = "Supporting"  # Core, Supporting, Reference
    data_update_frequency: str = "Unknown"  # High, Medium, Low

@dataclass 
class BusinessDomain:
    """Enhanced business domain information with comprehensive analysis"""
    domain_type: str
    industry: str
    entities: List[str]
    confidence: float
    sample_questions: List[str]
    customer_definition: str = ""
    
    # Enhanced business intelligence
    core_business_processes: List[str] = field(default_factory=list)
    business_readiness_score: int = 0
    query_capabilities: Dict[str, bool] = field(default_factory=dict)
    relationship_quality: str = "Unknown"  # Excellent, Good, Fair, Poor

@dataclass
class Relationship:
    """Enhanced database relationship information with business context"""
    from_table: str
    to_table: str
    column: str
    relationship_type: str
    confidence: float
    description: str = ""
    
    # Enhanced relationship intelligence
    business_significance: float = 0.0
    discovery_method: str = "unknown"  # constraint, view_analysis, data_pattern, llm_inference
    from_entity_type: str = "Unknown"
    to_entity_type: str = "Unknown"
    business_description: str = ""
    validation_status: str = "unvalidated"  # validated, unvalidated, disputed

@dataclass
class DatabaseConstraint:
    """Database constraint information from system tables"""
    constraint_name: str
    constraint_type: str  # FK, PK, UNIQUE, CHECK
    parent_table: str
    parent_column: str
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None
    is_enforced: bool = True
    discovery_confidence: float = 1.0

@dataclass
class ViewDefinition:
    """View definition with parsed relationship information"""
    view_name: str
    definition: str
    referenced_tables: List[str]
    join_conditions: List[Dict[str, str]]
    where_conditions: List[str]
    business_purpose: str = "Unknown"
    relationship_insights: List[str] = field(default_factory=list)

@dataclass
class DataRelationship:
    """Data-driven relationship discovery result"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str
    confidence_score: float
    evidence: Dict[str, Any]
    
    # Enhanced validation
    data_quality_score: float = 0.0
    sample_overlap_ratio: float = 0.0
    business_relevance: str = "Unknown"  # High, Medium, Low

@dataclass
class ComprehensiveAnalysisResult:
    """Results from comprehensive multi-source analysis"""
    view_analyses: List[Dict[str, Any]]
    foreign_key_relationships: List[Dict[str, Any]]
    entity_discoveries: List[Dict[str, Any]]
    comprehensive_graph: Dict[str, Any]
    business_intelligence: Dict[str, Any]
    entity_relationship_matrix: Dict[str, Any]
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_version: str = "6.0-comprehensive"
    total_processing_time: float = 0.0

@dataclass 
class RelationshipValidation:
    """Validation result for discovered relationships"""
    relationship_id: str
    source_methods: List[str]  # ['foreign_key', 'view_analysis', 'llm_suggested']
    confidence_scores: Dict[str, float]
    overall_confidence: float
    validation_status: str  # 'validated', 'conflicting', 'single_source'
    business_significance: float
    recommended_for_queries: bool

class DatabaseObject:
    """Enhanced database object information for discovery with relationship context"""
    
    def __init__(self, schema: str, name: str, object_type: str, estimated_rows: int = 0):
        self.schema = schema
        self.name = name
        self.object_type = object_type
        self.estimated_rows = estimated_rows
        self.priority_score = self._calculate_priority_score()
        self.business_relevance = self._determine_business_relevance()
        self.relationship_potential = self._assess_relationship_potential()
    
    def _calculate_priority_score(self) -> int:
        """Calculate enhanced priority score for processing order"""
        score = 0
        name_lower = self.name.lower()
        
        # Base score by object type
        if self.object_type == 'BASE TABLE':
            score += 100
        elif self.object_type == 'TABLE':
            score += 100
        elif self.object_type == 'VIEW':
            score += 80  # Views often contain business logic
        
        # Data volume consideration
        if self.estimated_rows > 0:
            score += min(50, self.estimated_rows // 10000)  # Max 50 points for data volume
        
        # Enhanced business entity recognition
        high_value_patterns = [
            'customer', 'client', 'businesspoint', 'account', 'user',
            'payment', 'transaction', 'invoice', 'billing', 'financial',
            'order', 'sale', 'purchase', 'product', 'item',
            'person', 'employee', 'contact', 'address'
        ]
        
        medium_value_patterns = [
            'task', 'assignment', 'work', 'activity', 'workflow',
            'category', 'type', 'status', 'configuration', 'setting',
            'log', 'audit', 'history', 'note', 'comment'
        ]
        
        # High value business objects
        for pattern in high_value_patterns:
            if pattern in name_lower:
                score += 50
                break
        else:
            # Medium value business objects
            for pattern in medium_value_patterns:
                if pattern in name_lower:
                    score += 25
                    break
        
        # Relationship potential (objects that likely connect to others)
        relationship_indicators = ['id', 'key', 'ref', 'link']
        if any(indicator in name_lower for indicator in relationship_indicators):
            score += 20
        
        # Penalize clearly problematic objects
        problematic_patterns = [
            'temp', 'tmp', 'backup', 'bck', 'test', 'debug',
            'error', 'log_archive', 'deprecated', 'old_'
        ]
        
        for pattern in problematic_patterns:
            if pattern in name_lower:
                score -= 30
                break
        
        return max(score, 0)
    
    def _determine_business_relevance(self) -> str:
        """Determine business relevance of the object"""
        name_lower = self.name.lower()
        
        # Core business entities
        core_patterns = ['customer', 'payment', 'order', 'product', 'invoice', 'account']
        if any(pattern in name_lower for pattern in core_patterns):
            return "CORE"
        
        # Supporting business entities  
        supporting_patterns = ['user', 'employee', 'contact', 'address', 'task', 'workflow']
        if any(pattern in name_lower for pattern in supporting_patterns):
            return "SUPPORTING"
        
        # Reference/lookup entities
        reference_patterns = ['type', 'category', 'status', 'config', 'setting', 'lookup']
        if any(pattern in name_lower for pattern in reference_patterns):
            return "REFERENCE"
        
        # System/technical entities
        system_patterns = ['log', 'audit', 'trace', 'temp', 'backup', 'system']
        if any(pattern in name_lower for pattern in system_patterns):
            return "SYSTEM"
        
        return "UNKNOWN"
    
    def _assess_relationship_potential(self) -> str:
        """Assess the potential for this object to have relationships"""
        name_lower = self.name.lower()
        
        # High relationship potential (central entities)
        if self.business_relevance == "CORE":
            return "HIGH"
        
        # Medium relationship potential (connecting entities)
        connecting_patterns = ['assignment', 'link', 'bridge', 'junction', 'mapping']
        if any(pattern in name_lower for pattern in connecting_patterns):
            return "HIGH"
        
        # Standard relationship potential
        if self.business_relevance in ["SUPPORTING", "REFERENCE"]:
            return "MEDIUM"
        
        # Low relationship potential (isolated entities)
        if self.business_relevance == "SYSTEM":
            return "LOW"
        
        return "MEDIUM"

@dataclass
class SystemStatus:
    """Enhanced system status tracking with relationship discovery"""
    discovery_completed: bool = False
    analysis_completed: bool = False
    relationship_discovery_completed: bool = False
    tables_discovered: int = 0
    relationships_found: int = 0
    constraints_found: int = 0
    views_analyzed: int = 0
    domain_identified: bool = False
    cache_available: bool = False
    business_readiness_score: int = 0

@dataclass
class QueryResult:
    """Enhanced query execution result with business context"""
    question: str
    relevant_tables: List[str]
    sql_query: str
    results: List[Dict[str, Any]]
    results_count: int
    execution_error: Optional[str] = None
    execution_time: float = 0.0
    
    # Enhanced query intelligence
    query_complexity: str = "Simple"  # Simple, Medium, Complex
    business_significance: str = "Unknown"  # High, Medium, Low
    entities_involved: List[str] = field(default_factory=list)
    relationships_used: List[str] = field(default_factory=list)
    business_interpretation: str = ""

class AnalysisStats:
    """Enhanced statistics tracking for comprehensive analysis operations"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Discovery stats
        self.total_objects_found = 0
        self.objects_processed = 0
        self.objects_excluded = 0
        self.successful_analyses = 0
        self.analysis_errors = 0
        
        # Relationship discovery stats
        self.constraints_discovered = 0
        self.views_analyzed = 0
        self.data_relationships_found = 0
        self.relationships_validated = 0
        
        # Entity classification stats
        self.entities_classified = 0
        self.high_confidence_classifications = 0
        self.business_entities_found = 0
        
        # Quality metrics
        self.relationship_discovery_quality = "Unknown"
        self.entity_classification_quality = "Unknown"
        self.business_readiness_assessment = "Unknown"
        
        self.start_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            # Discovery metrics
            'total_objects_found': self.total_objects_found,
            'objects_processed': self.objects_processed,
            'objects_excluded': self.objects_excluded,
            'successful_analyses': self.successful_analyses,
            'analysis_errors': self.analysis_errors,
            
            # Relationship discovery metrics
            'constraints_discovered': self.constraints_discovered,
            'views_analyzed': self.views_analyzed,
            'data_relationships_found': self.data_relationships_found,
            'relationships_validated': self.relationships_validated,
            
            # Entity classification metrics
            'entities_classified': self.entities_classified,
            'high_confidence_classifications': self.high_confidence_classifications,
            'business_entities_found': self.business_entities_found,
            
            # Quality assessments
            'relationship_discovery_quality': self.relationship_discovery_quality,
            'entity_classification_quality': self.entity_classification_quality,
            'business_readiness_assessment': self.business_readiness_assessment,
            
            # Performance metrics
            'analysis_duration': (datetime.now() - self.start_time).total_seconds(),
            'objects_per_second': self.objects_processed / max((datetime.now() - self.start_time).total_seconds(), 1)
        }
    
    def calculate_success_rate(self) -> float:
        """Calculate overall analysis success rate"""
        if self.objects_processed == 0:
            return 0.0
        return self.successful_analyses / self.objects_processed
    
    def calculate_relationship_coverage(self) -> float:
        """Calculate relationship discovery coverage"""
        total_possible = self.objects_processed * (self.objects_processed - 1) / 2  # n(n-1)/2
        if total_possible == 0:
            return 0.0
        return min(1.0, (self.constraints_discovered + self.data_relationships_found) / total_possible)

# Enhanced utility functions for model conversion with relationship support

def table_info_to_dict(table: TableInfo) -> Dict[str, Any]:
    """Convert enhanced TableInfo to dictionary for JSON serialization"""
    result = asdict(table)
    if table.semantic_profile:
        result['semantic_profile'] = asdict(table.semantic_profile)
    return result

def dict_to_table_info(data: Dict[str, Any]) -> TableInfo:
    """Convert dictionary back to enhanced TableInfo"""
    # Handle semantic profile
    if 'semantic_profile' in data and data['semantic_profile']:
        data['semantic_profile'] = SemanticProfile(**data['semantic_profile'])
    
    # Handle default values for new fields
    defaults = {
        'business_indicators': [],
        'sample_questions': [],
        'relationship_context': {},
        'business_significance': 'Supporting',
        'constraint_info': []
    }
    
    for key, default_value in defaults.items():
        if key not in data:
            data[key] = default_value
    
    return TableInfo(**data)

def business_domain_to_dict(domain: BusinessDomain) -> Dict[str, Any]:
    """Convert enhanced BusinessDomain to dictionary"""
    return asdict(domain)

def dict_to_business_domain(data: Dict[str, Any]) -> BusinessDomain:
    """Convert dictionary back to enhanced BusinessDomain"""
    # Handle default values for new fields
    defaults = {
        'core_business_processes': [],
        'business_readiness_score': 0,
        'query_capabilities': {},
        'relationship_quality': 'Unknown'
    }
    
    for key, default_value in defaults.items():
        if key not in data:
            data[key] = default_value
    
    return BusinessDomain(**data)

def relationship_to_dict(relationship: Relationship) -> Dict[str, Any]:
    """Convert enhanced Relationship to dictionary"""
    return asdict(relationship)

def dict_to_relationship(data: Dict[str, Any]) -> Relationship:
    """Convert dictionary back to enhanced Relationship"""
    # Handle default values for new fields
    defaults = {
        'business_significance': 0.0,
        'discovery_method': 'unknown',
        'from_entity_type': 'Unknown',
        'to_entity_type': 'Unknown',
        'business_description': '',
        'validation_status': 'unvalidated'
    }
    
    for key, default_value in defaults.items():
        if key not in data:
            data[key] = default_value
    
    return Relationship(**data)

# Validation helpers

def validate_table_info(table: TableInfo) -> List[str]:
    """Validate TableInfo object and return list of issues"""
    issues = []
    
    if not table.name:
        issues.append("Table name is required")
    
    if not table.schema:
        issues.append("Schema is required")
    
    if not table.full_name:
        issues.append("Full name is required")
    
    if not table.columns:
        issues.append("At least one column is required")
    
    if table.semantic_profile and table.semantic_profile.confidence < 0 or table.semantic_profile.confidence > 1:
        issues.append("Semantic profile confidence must be between 0 and 1")
    
    return issues

def validate_relationship(relationship: Relationship) -> List[str]:
    """Validate Relationship object and return list of issues"""
    issues = []
    
    if not relationship.from_table:
        issues.append("From table is required")
    
    if not relationship.to_table:
        issues.append("To table is required")
    
    if relationship.from_table == relationship.to_table:
        issues.append("Self-relationships not currently supported")
    
    if relationship.confidence < 0 or relationship.confidence > 1:
        issues.append("Confidence must be between 0 and 1")
    
    if relationship.business_significance < 0 or relationship.business_significance > 1:
        issues.append("Business significance must be between 0 and 1")
    
    return issues

# Factory functions for creating objects with defaults

def create_table_info(name: str, schema: str, object_type: str, **kwargs) -> TableInfo:
    """Factory function to create TableInfo with sensible defaults"""
    defaults = {
        'full_name': f"[{schema}].[{name}]",
        'row_count': 0,
        'columns': [],
        'sample_data': [],
        'relationships': [],
        'business_indicators': [],
        'sample_questions': [],
        'relationship_context': {},
        'business_significance': 'Supporting',
        'constraint_info': []
    }
    
    # Merge provided kwargs with defaults
    for key, value in kwargs.items():
        defaults[key] = value
    
    return TableInfo(
        name=name,
        schema=schema,
        object_type=object_type,
        **defaults
    )

def create_relationship(from_table: str, to_table: str, relationship_type: str, **kwargs) -> Relationship:
    """Factory function to create Relationship with sensible defaults"""
    defaults = {
        'column': '',
        'confidence': 0.5,
        'description': '',
        'business_significance': 0.5,
        'discovery_method': 'unknown',
        'from_entity_type': 'Unknown',
        'to_entity_type': 'Unknown',
        'business_description': '',
        'validation_status': 'unvalidated'
    }
    
    # Merge provided kwargs with defaults
    for key, value in kwargs.items():
        defaults[key] = value
    
    return Relationship(
        from_table=from_table,
        to_table=to_table,
        relationship_type=relationship_type,
        **defaults
    )