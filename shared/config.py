#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management - Simple, Readable, Maintainable  
Following README: DRY, SOLID, YAGNI principles
"""

import os
from pathlib import Path
from typing import Optional, List

class Config:
    """Configuration with clean validation and environment support"""
    
    def __init__(self):
        # Core Azure OpenAI settings (Required)
        self.azure_endpoint = self._get_str('AZURE_ENDPOINT')
        self.api_key = self._get_str('AZURE_OPENAI_API_KEY')
        self.deployment_name = self._get_str('DEPLOYMENT_NAME', 'gpt-4.1-mini')
        self.api_version = self._get_str('MODEL_VERSION', '2024-12-01-preview')
        
        # Database settings (Required)
        self.connection_string = self._get_str('DATABASE_CONNECTION_STRING')
        
        # 4-Stage Pipeline features (README requirements)
        self.enable_4_stage_pipeline = self._get_bool('ENABLE_4_STAGE_PIPELINE', True)
        self.enable_view_analysis = self._get_bool('ENABLE_VIEW_ANALYSIS', True)
        self.enable_sproc_analysis = self._get_bool('ENABLE_SPROC_ANALYSIS', True)
        self.enable_sql_validation = self._get_bool('SQL_SYNTAX_VALIDATION', True)
        self.enable_result_validation = self._get_bool('ENABLE_RESULT_VALIDATION', True)
        
        # Performance and cache settings
        self.discovery_cache_hours = self._get_int('DISCOVERY_CACHE_HOURS', 24)
        self.semantic_cache_hours = self._get_int('SEMANTIC_CACHE_HOURS', 24)
        self.query_timeout_seconds = self._get_int('QUERY_TIMEOUT_SECONDS', 30)
        self.max_retry_attempts = self._get_int('MAX_RETRY_ATTEMPTS', 2)
        self.row_limit_default = self._get_int('ROW_LIMIT_DEFAULT', 100)
        
        # Advanced features
        self.use_fast_queries = self._get_bool('USE_FAST_QUERIES', True)
        self.max_parallel_workers = self._get_int('MAX_PARALLEL_WORKERS', 8)
        
        # Filtering and exclusions (merged from env as requested)
        self.table_exclusions = self._get_list('TABLE_EXCLUSIONS')
        self.schema_exclusions = self._get_list('SCHEMA_EXCLUSIONS', ['sys', 'information_schema'])
        
        # International support (README requirement)
        self.utf8_encoding = self._get_bool('UTF8_ENCODING', True)
        self.preserve_international_chars = self._get_bool('PRESERVE_INTERNATIONAL_CHARS', True)
        
        # Pipeline-specific settings
        self.table_selection_confidence = self._get_float('TABLE_SELECTION_CONFIDENCE', 0.7)
        self.relationship_validation = self._get_bool('RELATIONSHIP_VALIDATION', True)
        self.pipeline_timeout_seconds = self._get_int('PIPELINE_TIMEOUT_SECONDS', 60)
        
        # Cache and data directory
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Validate configuration
        self._validate()
    
    def _get_str(self, key: str, default: str = '') -> str:
        """Get string environment variable with clean handling"""
        return os.getenv(key, default).strip().strip('"\'')
    
    def _get_int(self, key: str, default: int) -> int:
        """Get integer environment variable with fallback"""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    def _get_float(self, key: str, default: float) -> float:
        """Get float environment variable with fallback"""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def _get_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        """Get list from comma-separated environment variable"""
        if default is None:
            default = []
        
        value = os.getenv(key, '')
        if not value:
            return default
        
        # Split by comma and clean each item
        items = [item.strip() for item in value.split(',')]
        return [item for item in items if item]  # Remove empty items
    
    def _validate(self):
        """Validate required configuration with helpful error messages"""
        errors = []
        
        if not self.api_key:
            errors.append("AZURE_OPENAI_API_KEY is required")
        
        if not self.azure_endpoint:
            errors.append("AZURE_ENDPOINT is required")
        
        if not self.connection_string:
            errors.append("DATABASE_CONNECTION_STRING is required")
        
        # Validate ranges
        if self.query_timeout_seconds <= 0:
            errors.append("QUERY_TIMEOUT_SECONDS must be positive")
        
        if self.table_selection_confidence < 0 or self.table_selection_confidence > 1:
            errors.append("TABLE_SELECTION_CONFIDENCE must be between 0 and 1")
        
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
        """Get merged exclusion patterns for database objects"""
        # Default patterns from README
        default_patterns = [
            'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
            'bck', 'backup', 'dev', 'temp_', 'test_', 'corrupted', 'broken',
            'timingview', 'viewbatchactionanalysis', 'viewfixofferfailure'
        ]
        
        # Merge with environment patterns
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
    
    def get_performance_settings(self) -> dict:
        """Get performance-related settings"""
        return {
            'query_timeout': self.query_timeout_seconds,
            'max_workers': self.max_parallel_workers,
            'use_fast_queries': self.use_fast_queries,
            'row_limit': self.row_limit_default
        }
    
    def get_pipeline_settings(self) -> dict:
        """Get 4-stage pipeline settings"""
        return {
            'enabled': self.enable_4_stage_pipeline,
            'table_confidence': self.table_selection_confidence,
            'relationship_validation': self.relationship_validation,
            'timeout': self.pipeline_timeout_seconds,
            'max_retries': self.max_retry_attempts
        }
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return (f"Config(endpoint='{self.azure_endpoint}', "
                f"deployment='{self.deployment_name}', "
                f"4stage={self.enable_4_stage_pipeline}, "
                f"cache_hours={self.discovery_cache_hours})")
    
    def validate_llm_settings(self) -> bool:
        """Validate LLM configuration is complete"""
        return bool(self.azure_endpoint and self.api_key and self.deployment_name)
    
    def validate_database_settings(self) -> bool:
        """Validate database configuration is complete"""
        return bool(self.connection_string)
    
    def get_health_check(self) -> dict:
        """Get configuration health check info"""
        return {
            'llm_configured': self.validate_llm_settings(),
            'database_configured': self.validate_database_settings(),
            '4_stage_pipeline': self.enable_4_stage_pipeline,
            'view_analysis': self.enable_view_analysis,
            'sql_validation': self.enable_sql_validation,
            'cache_directory_exists': self.cache_directory.exists(),
            'utf8_support': self.utf8_encoding
        }