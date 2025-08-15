#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Configuration Management - Proper type handling
"""

import os
from pathlib import Path

class Config:
    """Simple configuration management with proper error handling"""
    
    def __init__(self):
        try:
            # Required settings with safe defaults
            self.azure_endpoint = self.get_env_str('AZURE_ENDPOINT', 'https://gyp-weu-02-res01-prdenv-cog01.openai.azure.com/')
            self.deployment_name = self.get_env_str('DEPLOYMENT_NAME', 'gpt-4.1-mini')
            self.api_version = self.get_env_str('MODEL_VERSION', '2024-12-01-preview')
            self.api_key = self.get_env_str('AZURE_OPENAI_API_KEY')
            self.connection_string = self.get_env_str('DATABASE_CONNECTION_STRING')
            
            # Cache settings
            self.cache_directory = Path('data')
            self.cache_directory.mkdir(exist_ok=True)
            self.discovery_cache_hours = self.get_env_int('DISCOVERY_CACHE_HOURS', 24)
            self.semantic_cache_hours = self.get_env_int('SEMANTIC_CACHE_HOURS', 24)
            
            # Performance settings
            self.max_results = self.get_env_int('MAX_RESULTS', 100)
            
            # Validate required settings
            if not self.api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required")
            if not self.connection_string:
                raise ValueError("DATABASE_CONNECTION_STRING is required")
            
            print("✅ Configuration loaded successfully")
            
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            raise
    
    def get_env_str(self, key: str, default: str = None) -> str:
        """Safely get string environment variable"""
        try:
            value = os.getenv(key, default)
            if value is None:
                return ""
            return str(value).strip().strip('"\'')
        except Exception:
            return default or ""
    
    def get_env_int(self, key: str, default: int = 0) -> int:
        """Safely get integer environment variable"""
        try:
            value = os.getenv(key)
            if value is None:
                return default
            return int(str(value).strip())
        except (ValueError, TypeError):
            return default
    
    def get_env_bool(self, key: str, default: bool = False) -> bool:
        """Safely get boolean environment variable"""
        try:
            value = os.getenv(key)
            if value is None:
                return default
            return str(value).lower().strip() in ['true', '1', 'yes', 'on']
        except Exception:
            return default
    
    def get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        try:
            return self.cache_directory / str(filename)
        except Exception:
            return Path('data') / 'cache.json'
    
    def get_database_connection_string(self) -> str:
        """Get database connection string"""
        return self.connection_string