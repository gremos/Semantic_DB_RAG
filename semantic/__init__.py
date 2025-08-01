"""
Simplified semantic analysis module - matching actual implementation
"""

from .analysis import (
    SimpleSemanticAnalyzer,
    SimpleLLMClient
)

# Maintain backward compatibility with aliases
EnhancedSemanticAnalyzer = SimpleSemanticAnalyzer
IntelligentSemanticAnalyzer = SimpleSemanticAnalyzer  
SemanticAnalyzer = SimpleSemanticAnalyzer

__all__ = [
    "SimpleSemanticAnalyzer",
    "SimpleLLMClient", 
    "EnhancedSemanticAnalyzer",
    "IntelligentSemanticAnalyzer",
    "SemanticAnalyzer"
]