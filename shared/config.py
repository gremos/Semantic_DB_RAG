#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Configuration Management
Following README: Simple, Readable, Maintainable
"""

import os
from pathlib import Path

class Config:
    """Simple configuration with clean defaults"""
    
    def __init__(self):
        # Azure OpenAI (Required)
        self.azure_endpoint = self._get_str('AZURE_ENDPOINT')
        self.api_key = self._get_str('AZURE_OPENAI_API_KEY')
        self.deployment_name = self._get_str('DEPLOYMENT_NAME', 'gpt-4.1-mini')
        self.api_version = self._get_str('MODEL_VERSION', '2024-12-01-preview')
        
        # Database (Required) 
        self.connection_string = self._get_str('DATABASE_CONNECTION_STRING')
        
        # Feature flags (README requirements)
        self.enable_4_stage_pipeline = self._get_bool('ENABLE_4_STAGE_PIPELINE', True)
        self.enable_view_analysis = self._get_bool('ENABLE_VIEW_ANALYSIS', True)
        self.enable_sql_validation = self._get_bool('SQL_SYNTAX_VALIDATION', True)
        
        # Performance settings
        self.discovery_cache_hours = self._get_int('DISCOVERY_CACHE_HOURS', 24)
        self.semantic_cache_hours = self._get_int('SEMANTIC_CACHE_HOURS', 24)
        self.query_timeout_seconds = self._get_int('QUERY_TIMEOUT_SECONDS', 30)
        self.max_retry_attempts = self._get_int('MAX_RETRY_ATTEMPTS', 2)
        self.row_limit_default = self._get_int('ROW_LIMIT_DEFAULT', 100)
        
        # Cache directory
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Validate required settings
        self._validate()
    
    def _get_str(self, key: str, default: str = '') -> str:
        """Get string environment variable"""
        return os.getenv(key, default).strip().strip('"\'')
    
    def _get_int(self, key: str, default: int) -> int:
        """Get integer environment variable"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _validate(self):
        """Validate required configuration"""
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required")
        if not self.connection_string:
            raise ValueError("DATABASE_CONNECTION_STRING is required")
        if not self.azure_endpoint:
            raise ValueError("AZURE_ENDPOINT is required")
    
    def get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_directory / filename
    
    def get_database_connection_string(self) -> str:
        """Get database connection string"""
        return self.connection_string