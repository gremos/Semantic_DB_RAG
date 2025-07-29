"""
Enhanced semantic analysis module for AI-powered classification and relationship discovery
"""

from .analysis import EnhancedSemanticAnalyzer, SimpleLLMClient, BusinessEntityAnalyzer

# Maintain backward compatibility
SemanticAnalyzer = EnhancedSemanticAnalyzer

__all__ = ["EnhancedSemanticAnalyzer", "SemanticAnalyzer", "SimpleLLMClient", "BusinessEntityAnalyzer"]