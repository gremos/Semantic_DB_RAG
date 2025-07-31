"""
Enhanced interactive query interface module for natural language processing
Fixed method names and improved performance
"""

from .query_interface import IntelligentQueryInterface, IntelligentLLMClient

# Maintain backward compatibility
EnhancedQueryInterface = IntelligentQueryInterface
QueryInterface = IntelligentQueryInterface
InteractiveLLMClient = IntelligentLLMClient

__all__ = ["IntelligentQueryInterface", "EnhancedQueryInterface", "QueryInterface", 
           "IntelligentLLMClient", "InteractiveLLMClient"]