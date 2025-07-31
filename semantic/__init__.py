"""
Enhanced semantic analysis module for comprehensive structure discovery
Updated for comprehensive multi-source analysis approach
"""

from .analysis import (
    EnhancedSemanticAnalyzer, 
    ComprehensiveStructureAnalyzer,
    EnhancedViewAnalyzer,
    EnhancedForeignKeyAnalyzer,
    LLMEntityScanner,
    ViewJoinAnalysis,
    ForeignKeyRelationship,
    EntityDiscovery
)

# Maintain backward compatibility
IntelligentSemanticAnalyzer = EnhancedSemanticAnalyzer
SemanticAnalyzer = EnhancedSemanticAnalyzer

__all__ = [
    "EnhancedSemanticAnalyzer", 
    "IntelligentSemanticAnalyzer", 
    "SemanticAnalyzer",
    "ComprehensiveStructureAnalyzer",
    "EnhancedViewAnalyzer",
    "EnhancedForeignKeyAnalyzer", 
    "LLMEntityScanner",
    "ViewJoinAnalysis",
    "ForeignKeyRelationship",
    "EntityDiscovery"
]