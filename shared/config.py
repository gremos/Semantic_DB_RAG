#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Configuration Management
"""

import os
from pathlib import Path

class Config:
    """Simple configuration management"""
    
    def __init__(self):
        # Required settings
        self.azure_endpoint = os.getenv('AZURE_ENDPOINT', 'https://gyp-weu-02-res01-prdenv-cog01.openai.azure.com/')
        self.deployment_name = os.getenv('DEPLOYMENT_NAME', 'gpt-4.1-mini')
        self.api_version = os.getenv('MODEL_VERSION', '2024-12-01-preview')
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.connection_string = os.getenv('DATABASE_CONNECTION_STRING')
        
        # Cache settings
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        self.discovery_cache_hours = int(os.getenv('DISCOVERY_CACHE_HOURS', '24'))
        self.semantic_cache_hours = int(os.getenv('SEMANTIC_CACHE_HOURS', '24'))
        
        # Performance settings
        self.max_results = int(os.getenv('MAX_RESULTS', '100'))
        
        # Validate required settings
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required")
        if not self.connection_string:
            raise ValueError("DATABASE_CONNECTION_STRING is required")
        
        print("✅ Configuration loaded successfully")
    
    def get_cache_path(self, filename: str) -> Path:
        return self.cache_directory / filename
    
    def get_database_connection_string(self) -> str:
        return self.connection_string