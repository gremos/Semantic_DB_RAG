#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration - Simple and Clean
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
"""

import os
from pathlib import Path
from typing import List, Dict, Any

class Config:
    """Simple, focused configuration with smart defaults"""
    
    def __init__(self):
        # Essential LLM settings
        self.azure_endpoint = self._get_required('AZURE_ENDPOINT')
        self.api_key = self._get_required('AZURE_OPENAI_API_KEY')
        self.deployment_name = self._get_str('DEPLOYMENT_NAME', 'gpt-4o-mini')
        self.api_version = self._get_str('MODEL_VERSION', '2024-12-01-preview')
        
        # Essential database settings  
        self.connection_string = self._get_required('DATABASE_CONNECTION_STRING')
        
        # Cache settings (reasonable defaults)
        self.discovery_cache_hours = self._get_int('DISCOVERY_CACHE_HOURS', 24)
        self.semantic_cache_hours = self._get_int('SEMANTIC_CACHE_HOURS', 24)
        
        # Table exclusions
        self.table_exclusions = self._get_list('TABLE_EXCLUSIONS')
        self.schema_exclusions = self._get_list('SCHEMA_EXCLUSIONS', ['sys', 'information_schema'])
        
        # Performance settings
        self.query_timeout = self._get_int('QUERY_TIMEOUT_SECONDS', 30)
        self.max_sample_rows = self._get_int('MAX_SAMPLE_ROWS', 6)
        self.max_tables_per_batch = self._get_int('MAX_TABLES_PER_BATCH', 3)
        
        # International support
        self.utf8_encoding = self._get_bool('UTF8_ENCODING', True)
        
        # Cache directory
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Validate essential settings
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
        # LLM validation
        if not self.azure_endpoint.startswith('https://'):
            raise ValueError("AZURE_ENDPOINT must start with https://")
        
        if len(self.api_key) < 20:
            raise ValueError("AZURE_OPENAI_API_KEY appears to be invalid")
        
        # Database validation  
        if 'Driver=' not in self.connection_string:
            raise ValueError("DATABASE_CONNECTION_STRING must include Driver= parameter")
    
    def get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        cache_file = self.cache_directory / filename
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        return cache_file
    
    def get_database_connection_string(self) -> str:
        """Get database connection string with timeout"""
        conn_str = self.connection_string
        
        # Add timeout if not present
        if 'Connection Timeout' not in conn_str and 'Connect Timeout' not in conn_str:
            conn_str += f";Connection Timeout={self.query_timeout}"
        
        return conn_str
    
    def get_exclusion_patterns(self) -> List[str]:
        """Get smart table exclusion patterns"""
        # Smart defaults for common system/temp tables
        smart_defaults = [
            'msreplication', 'syscommittab', 'sysdiagrams', 'dtproperties',
            'backup', 'corrupted', 'broken', 'temp_', 'test_', 'old_',
            'archive_', 'staging_', 'etl_', 'import_', 'export_'
        ]
        
        return smart_defaults + self.table_exclusions
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get system health status"""
        health = {
            'llm_configured': bool(self.azure_endpoint and self.api_key),
            'database_configured': bool(self.connection_string),
            'cache_directory_exists': self.cache_directory.exists(),
            'cache_writable': self._check_cache_writable()
        }
        
        # Overall health score
        critical_checks = [
            health['llm_configured'],
            health['database_configured'], 
            health['cache_directory_exists']
        ]
        health['overall_health'] = all(critical_checks)
        
        return health
    
    def _check_cache_writable(self) -> bool:
        """Check if cache directory is writable"""
        try:
            test_file = self.cache_directory / '.test_write'
            test_file.write_text('test')
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings"""
        return {
            'query_timeout': self.query_timeout,
            'max_sample_rows': self.max_sample_rows,
            'max_tables_per_batch': self.max_tables_per_batch,
            'cache_hours_discovery': self.discovery_cache_hours,
            'cache_hours_semantic': self.semantic_cache_hours
        }
    
    def validate_llm_config(self) -> bool:
        """Validate LLM configuration"""
        try:
            from langchain_openai import AzureChatOpenAI
            
            # Test LLM initialization  
            llm = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                azure_deployment=self.deployment_name,
                api_version=self.api_version,
                request_timeout=10
            )
            return True
        except Exception as e:
            print(f"LLM configuration error: {e}")
            return False
    
    def validate_database_config(self) -> bool:
        """Validate database configuration"""
        try:
            import pyodbc
            
            # Test database connection
            with pyodbc.connect(self.get_database_connection_string()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return True
        except Exception as e:
            print(f"Database configuration error: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return f"""Config:
  LLM: {self.deployment_name} @ {self.azure_endpoint}
  Cache: {self.cache_directory}
  Performance: {self.max_tables_per_batch} tables/batch, {self.query_timeout}s timeout"""