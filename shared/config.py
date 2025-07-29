#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared configuration management - Enhanced with all required attributes
"""

import os
from pathlib import Path

class Config:
    """Centralized configuration management with all required attributes"""
    
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
        self.server = os.getenv('DB_SERVER', 'localhost')
        self.database = os.getenv('DB_DATABASE', 'master')
        self.username = os.getenv('DB_USERNAME', '')
        self.password = os.getenv('DB_PASSWORD', '')
        
        # Cache Configuration
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Discovery Configuration - NOW INCLUDED
        self.discovery_cache_hours = int(os.getenv('DISCOVERY_CACHE_HOURS', '24'))
        self.semantic_cache_hours = int(os.getenv('SEMANTIC_CACHE_HOURS', '48'))
        self.max_discovery_objects = int(os.getenv('MAX_DISCOVERY_OBJECTS', '50'))
        self.max_parallel_workers = int(os.getenv('MAX_PARALLEL_WORKERS', '8'))
        self.query_timeout_seconds = int(os.getenv('QUERY_TIMEOUT_SECONDS', '30'))
        
        # Connection timeouts
        self.connection_timeout = int(os.getenv('CONNECTION_TIMEOUT', '15'))
        self.command_timeout = int(os.getenv('COMMAND_TIMEOUT', '10'))
        
        # Query Configuration
        self.max_results = int(os.getenv('MAX_RESULTS', '100'))
        self.max_batch_size = int(os.getenv('MAX_BATCH_SIZE', '5'))
        self.rate_limit_delay = float(os.getenv('RATE_LIMIT_DELAY', '0.5'))
        
        # Advanced options
        self.use_fast_queries = os.getenv('USE_FAST_QUERIES', 'true').lower() == 'true'
        self.exclude_backup_tables = os.getenv('EXCLUDE_BACKUP_TABLES', 'true').lower() == 'true'
    
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
        """Get enhanced database connection string with proper Unicode support"""
        if self.connection_string:
            # Use provided connection string with enhancements
            connection_string = f"{self.connection_string};Connection Timeout={self.connection_timeout};Command Timeout={self.command_timeout}"
        else:
            # Build connection string from components
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"MARS_Connection=yes;"
                f"Connection Timeout={self.connection_timeout};"
                f"Query Timeout={self.query_timeout_seconds};"
            )
        
        # Ensure UTF-8 support for Greek text
        if "charset" not in connection_string.lower() and "encoding" not in connection_string.lower():
            connection_string += ";charset=UTF8"
        
        return connection_string