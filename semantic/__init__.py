"""
Enhanced semantic analysis module for AI-powered classification and relationship discovery
Updated for intelligent metadata-first approach
"""

from .analysis import IntelligentSemanticAnalyzer, IntelligentMetadataAnalyzer

# Maintain backward compatibility
EnhancedSemanticAnalyzer = IntelligentSemanticAnalyzer
SemanticAnalyzer = IntelligentSemanticAnalyzer

__all__ = ["IntelligentSemanticAnalyzer", "EnhancedSemanticAnalyzer", "SemanticAnalyzer", "IntelligentMetadataAnalyzer"]
