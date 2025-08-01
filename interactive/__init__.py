"""
Simplified interactive query interface module - matching actual implementation
"""

from .query_interface import SimpleQueryInterface, SimpleLLMClient

# Maintain backward compatibility with aliases
IntelligentQueryInterface = SimpleQueryInterface
EnhancedQueryInterface = SimpleQueryInterface
QueryInterface = SimpleQueryInterface
IntelligentLLMClient = SimpleLLMClient
InteractiveLLMClient = SimpleLLMClient

__all__ = ["SimpleQueryInterface", "SimpleLLMClient", "IntelligentQueryInterface", 
           "EnhancedQueryInterface", "QueryInterface", "IntelligentLLMClient", "InteractiveLLMClient"]