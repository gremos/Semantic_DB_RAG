#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management - Simple, Readable, Maintainable  
Following README: DRY, SOLID, YAGNI principles
"""

import os
from pathlib import Path
from typing import List

class Config:
    """Simple configuration with clean validation"""
    
    def __init__(self):
        # Core settings (Required)
        self.azure_endpoint = self._get_str('AZURE_ENDPOINT')
        self.api_key = self._get_str('AZURE_OPENAI_API_KEY')
        self.deployment_name = self._get_str('DEPLOYMENT_NAME', 'gpt-4.1-mini')
        self.api_version = self._get_str('MODEL_VERSION', '2024-12-01-preview')
        self.connection_string = self._get_str('DATABASE_CONNECTION_STRING')
        
        # Pipeline features
        self.enable_4_stage_pipeline = self._get_bool('ENABLE_4_STAGE_PIPELINE', True)
        self.enable_view_analysis = self._get_bool('ENABLE_VIEW_ANALYSIS', True)
        self.enable_sql_validation = self._get_bool('SQL_SYNTAX_VALIDATION', True)
        
        # Performance settings
        self.discovery_cache_hours = self._get_int('DISCOVERY_CACHE_HOURS', 24)
        self.semantic_cache_hours = self._get_int('SEMANTIC_CACHE_HOURS', 24)
        self.query_timeout_seconds = self._get_int('QUERY_TIMEOUT_SECONDS', 30)
        self.max_parallel_workers = self._get_int('MAX_PARALLEL_WORKERS', 8)
        self.use_fast_queries = self._get_bool('USE_FAST_QUERIES', True)
        
        # Exclusions
        self.table_exclusions = self._get_list('TABLE_EXCLUSIONS')
        self.schema_exclusions = self._get_list('SCHEMA_EXCLUSIONS', ['sys', 'information_schema'])
        
        # International support
        self.utf8_encoding = self._get_bool('UTF8_ENCODING', True)
        
        # Cache directory
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Validate
        self._validate()
    
    def _get_str(self, key: str, default: str = '') -> str:
        """Get string from environment"""
        return os.getenv(key, default).strip().strip('"\'')
    
    def _get_int(self, key: str, default: int) -> int:
        """Get integer from environment"""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean from environment"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def _get_list(self, key: str, default: List[str] = None) -> List[str]:
        """Get list from comma-separated environment variable"""
        if default is None:
            default = []
        
        value = os.getenv(key, '')
        if not value:
            return default
        
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _validate(self):
        """Validate configuration"""
        errors = []
        
        if not self.api_key:
            errors.append("AZURE_OPENAI_API_KEY is required")
        if not self.azure_endpoint:
            errors.append("AZURE_ENDPOINT is required")
        if not self.connection_string:
            errors.append("DATABASE_CONNECTION_STRING is required")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
    
    def get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_directory / filename
    
    def get_database_connection_string(self) -> str:
        """Get database connection string"""
        return self.connection_string
    
    def get_exclusion_patterns(self) -> List[str]:
        """Get exclusion patterns"""
        default_patterns = [
            'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
            'backup', 'corrupted', 'broken', 'temp_', 'test_'
        ]
        return default_patterns + self.table_exclusions
    
    def is_4_stage_pipeline_enabled(self) -> bool:
        """Check if 4-stage pipeline is enabled"""
        return self.enable_4_stage_pipeline
    
    def is_view_analysis_enabled(self) -> bool:
        """Check if view analysis is enabled"""
        return self.enable_view_analysis
    
    def is_sql_validation_enabled(self) -> bool:
        """Check if SQL validation is enabled"""
        return self.enable_sql_validation
    
    def get_health_check(self) -> dict:
        """Get configuration health check"""
        return {
            'llm_configured': bool(self.azure_endpoint and self.api_key),
            'database_configured': bool(self.connection_string),
            '4_stage_pipeline': self.enable_4_stage_pipeline,
            'view_analysis': self.enable_view_analysis,
            'sql_validation': self.enable_sql_validation,
            'cache_directory_exists': self.cache_directory.exists(),
            'utf8_support': self.utf8_encoding
        }