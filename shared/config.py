#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared configuration management
"""

import os
from pathlib import Path

class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        self.load_environment()
        self.validate_config()
    
    def load_environment(self):
        """Load and set configuration from environment variables"""
        # Azure OpenAI Configuration
        self.azure_endpoint = os.getenv('AZURE_ENDPOINT', 'https://gyp-weu-02-res01-prdenv-cog01.openai.azure.com/')
        self.deployment_name = os.getenv('DEPLOYMENT_NAME', 'gpt-4.1-mini')
        self.api_version = os.getenv('MODEL_VERSION', '2024-12-01-preview')
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        
        # Database Configuration
        self.connection_string = os.getenv('DATABASE_CONNECTION_STRING')
        
        # Cache Configuration
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Discovery Configuration
        self.discovery_cache_hours = 24
        self.semantic_cache_hours = 48
        self.connection_timeout = 15
        self.command_timeout = 10
        
        # Query Configuration
        self.max_results = 100
        self.max_batch_size = 5
        self.rate_limit_delay = 0.5
    
    def validate_config(self):
        """Validate required configuration"""
        missing = []
        
        if not self.api_key:
            missing.append('AZURE_OPENAI_API_KEY')
        
        if not self.connection_string:
            missing.append('DATABASE_CONNECTION_STRING')
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    def get_cache_path(self, filename: str) -> Path:
        """Get full path for cache file"""
        return self.cache_directory / filename
    
    def get_database_connection_string(self) -> str:
        """Get enhanced database connection string"""
        connection_string = f"{self.connection_string};Connection Timeout={self.connection_timeout};Command Timeout={self.command_timeout}"
        
        if "charset" not in connection_string.lower() and "encoding" not in connection_string.lower():
            connection_string += ";charset=UTF8"
        
        return connection_string