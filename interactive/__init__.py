"""
Enhanced interactive query interface module for natural language processing
"""

from .query_interface import EnhancedQueryInterface, InteractiveLLMClient

# Maintain backward compatibility
QueryInterface = EnhancedQueryInterface

__all__ = ["EnhancedQueryInterface", "QueryInterface", "InteractiveLLMClient"]