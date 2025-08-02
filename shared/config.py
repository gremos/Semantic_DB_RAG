#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Configuration Management - Clean and Maintainable
Supports all features from README while keeping code simple
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Simple configuration management for the Semantic Database RAG System"""
    
    def __init__(self):
        self._load_environment()
        self._validate_required_config()
        self._set_defaults()
    
    def _load_environment(self):
        """Load configuration from environment variables"""
        
        # Azure OpenAI Configuration (Required)
        self.azure_endpoint = os.getenv('AZURE_ENDPOINT', 'https://gyp-weu-02-res01-prdenv-cog01.openai.azure.com/')
        self.deployment_name = os.getenv('DEPLOYMENT_NAME', 'gpt-4.1-mini')
        self.api_version = os.getenv('MODEL_VERSION', '2024-12-01-preview')
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        
        # Database Configuration (Required)
        self.connection_string = os.getenv('DATABASE_CONNECTION_STRING')
        
        # Alternative database connection components
        self.server = os.getenv('DB_SERVER', 'localhost')
        self.database = os.getenv('DB_DATABASE', 'master')
        self.username = os.getenv('DB_USERNAME', '')
        self.password = os.getenv('DB_PASSWORD', '')
        
        # Cache Configuration
        self.cache_directory = Path('data')
        self.cache_directory.mkdir(exist_ok=True)
        
        # Discovery Configuration
        self.discovery_cache_hours = int(os.getenv('DISCOVERY_CACHE_HOURS', '24'))
        self.semantic_cache_hours = int(os.getenv('SEMANTIC_CACHE_HOURS', '24'))
        
        # Performance Configuration
        self.max_parallel_workers = int(os.getenv('MAX_PARALLEL_WORKERS', '8'))
        self.query_timeout_seconds = int(os.getenv('QUERY_TIMEOUT_SECONDS', '30'))
        self.max_results = int(os.getenv('MAX_RESULTS', '100'))
        
        # Feature Flags
        self.enable_4_stage_pipeline = os.getenv('ENABLE_4_STAGE_PIPELINE', 'true').lower() == 'true'
        self.enable_view_analysis = os.getenv('ENABLE_VIEW_ANALYSIS', 'true').lower() == 'true'
        self.enable_result_validation = os.getenv('ENABLE_RESULT_VALIDATION', 'true').lower() == 'true'
        self.use_fast_queries = os.getenv('USE_FAST_QUERIES', 'true').lower() == 'true'
        
        # Sample Data Configuration
        self.samples_per_object = int(os.getenv('SAMPLES_PER_OBJECT', '5'))
    
    def _validate_required_config(self):
        """Validate required configuration"""
        
        missing = []
        
        if not self.api_key:
            missing.append('AZURE_OPENAI_API_KEY')
        
        if not self.connection_string and not (self.server and self.database):
            missing.append('DATABASE_CONNECTION_STRING or DB_SERVER/DB_DATABASE')
        
        if missing:
            print("❌ Configuration Error!")
            print("Missing required environment variables:")
            for var in missing:
                print(f"   • {var}")
            print("\n💡 Please check your .env file and ensure these variables are set.")
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    
    def _set_defaults(self):
        """Set intelligent defaults"""
        
        # Batch processing defaults
        self.max_batch_size = max(self.max_parallel_workers * 2, 16)
        
        # Connection settings
        self.connection_timeout = 15
        self.command_timeout = 30
        
        print(f"✅ Configuration loaded successfully")
        print(f"   🔧 Workers: {self.max_parallel_workers}, Batch size: {self.max_batch_size}")
        print(f"   ⚡ Features: 4-stage pipeline: {self.enable_4_stage_pipeline}")
        print(f"   📊 Samples per object: {self.samples_per_object}")
    
    def get_cache_path(self, filename: str) -> Path:
        """Get full path for cache file"""
        return self.cache_directory / filename
    
    def get_database_connection_string(self) -> str:
        """Get optimized database connection string"""
        
        if self.connection_string:
            # Use provided connection string with optimizations
            connection_string = self.connection_string.rstrip(';')
            
            # Add performance optimizations if not present
            optimizations = [
                "MARS_Connection=yes",
                "MultipleActiveResultSets=true",
                "Pooling=true",
                f"Connection Timeout={self.connection_timeout}",
                f"Command Timeout={self.command_timeout}"
            ]
            
            for opt in optimizations:
                key = opt.split('=')[0]
                if key not in connection_string:
                    connection_string += f";{opt}"
        
        else:
            # Build connection string from components
            if self.username and self.password:
                # SQL Server authentication
                connection_string = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={self.server};"
                    f"DATABASE={self.database};"
                    f"UID={self.username};"
                    f"PWD={self.password};"
                )
            else:
                # Windows authentication
                connection_string = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={self.server};"
                    f"DATABASE={self.database};"
                    f"Trusted_Connection=yes;"
                )
            
            # Add performance optimizations
            connection_string += (
                f"MARS_Connection=yes;"
                f"MultipleActiveResultSets=true;"
                f"Connection Timeout={self.connection_timeout};"
                f"Command Timeout={self.command_timeout};"
                f"Pooling=true;"
            )
        
        return connection_string
    
    def get_feature_summary(self) -> Dict[str, bool]:
        """Get summary of enabled features"""
        return {
            '4-Stage Automated Pipeline': self.enable_4_stage_pipeline,
            'View Definition Mining': self.enable_view_analysis,  
            'Result Validation': self.enable_result_validation,
            'Fast Query Optimization': self.use_fast_queries,
            'Enhanced Discovery': True,
            'Semantic Analysis': True,
            'Business Intelligence': True,
            'International Character Support': True
        }
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            'max_parallel_workers': self.max_parallel_workers,
            'max_batch_size': self.max_batch_size,
            'query_timeout_seconds': self.query_timeout_seconds,
            'max_results': self.max_results,
            'samples_per_object': self.samples_per_object,
            'discovery_cache_hours': self.discovery_cache_hours,
            'semantic_cache_hours': self.semantic_cache_hours
        }