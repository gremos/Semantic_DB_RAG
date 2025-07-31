"""
Enhanced interactive query interface module for natural language processing
"""

from .query_interface import IntelligentQueryInterface as EnhancedQueryInterface, IntelligentLLMClient as InteractiveLLMClient

QueryInterface = EnhancedQueryInterface

__all__ = ["EnhancedQueryInterface", "QueryInterface", "InteractiveLLMClient"]