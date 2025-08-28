#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration - Simplified & Clean
Following DRY, SOLID, YAGNI principles
Essential settings only
"""

import os
from pathlib import Path
from typing import List

class Config:
    """Simple, focused configuration"""
    
    def __init__(self):
        # Essential LLM settings
        self.azure_endpoint = self._get_required('AZURE_ENDPOINT')
        self.api_key = self._get_required('AZURE_OPENAI_API_KEY')
        self.deployment_name = self._get_str('DEPLOYMENT_NAME', 'gpt-4o-mini')
        self.api_version = self._get_str('MODEL_VERSION', '2024-12-01-preview')
        
        # Essential database settings  
        self.connection_string = self._get_required('DATABASE_CONNECTION_STRING')
        
        # Simple cache settings
        self.discovery_cache_hours = self._get_int('DISCOVERY_CACHE_HOURS', 24)
        self.semantic_cache_hours = self._get_int('SEMANTIC_CACHE_HOURS', 24)
        
        # Basic exclusions
        self.table_exclusions = self._get_list('TABLE_EXCLUSIONS')
        self.schema_exclusions = self._get_list('SCHEMA_EXCLUSIONS', ['sys', 'information_schema'])
        
        # UTF-8 support
        self.utf8_encoding = self._get_bool('UTF8_ENCODING', True)
        
        # Cache directory
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Validate essentials
        self._validate()
    
    def _get_required(self, key: str) -> str:
        """Get required environment variable"""
        value = os.getenv(key, '').strip().strip('"\'')
        if not value:
            raise ValueError(f"Required environment variable {key} is missing")
        return value
    
    def _get_str(self, key: str, default: str = '') -> str:
        """Get string from environment with default"""
        return os.getenv(key, default).strip().strip('"\'')
    
    def _get_int(self, key: str, default: int) -> int:
        """Get integer from environment with default"""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean from environment with default"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_list(self, key: str, default: List[str] = None) -> List[str]:
        """Get comma-separated list from environment"""
        if default is None:
            default = []
        
        value = os.getenv(key, '')
        if not value:
            return default
        
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _validate(self):
        """Validate essential configuration"""
        # Already validated in _get_required calls
        pass
    
    def get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_directory / filename
    
    def get_database_connection_string(self) -> str:
        """Get database connection string"""
        return self.connection_string
    
    def get_exclusion_patterns(self) -> List[str]:
        """Get table exclusion patterns"""
        default_patterns = [
            'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
            'backup', 'corrupted', 'broken', 'temp_', 'test_'
        ]
        return default_patterns + self.table_exclusions
    
    def get_health_check(self) -> dict:
        """Get configuration health status"""
        return {
            'llm_configured': bool(self.azure_endpoint and self.api_key),
            'database_configured': bool(self.connection_string),
            'cache_directory_exists': self.cache_directory.exists(),
            'utf8_support': self.utf8_encoding
        }