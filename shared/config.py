#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBUST Configuration Management - Conservative defaults for stability
Optimized for 678+ objects with adaptive performance and error recovery
"""

import os
from pathlib import Path
from typing import List

class Config:
    """ROBUST configuration management with conservative defaults for large datasets"""
    
    def __init__(self):
        self.load_environment()
        self.validate_config()
        self.apply_stability_optimizations()
    
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
        
        # ROBUST Discovery Configuration (conservative defaults)
        self.discovery_cache_hours = int(os.getenv('DISCOVERY_CACHE_HOURS', '12'))
        self.semantic_cache_hours = int(os.getenv('SEMANTIC_CACHE_HOURS', '24'))
        
        # UNLIMITED object processing
        self.max_discovery_objects = None  # Process ALL objects
        
        # CONSERVATIVE Parallel Processing (reduced for stability)
        self.max_parallel_workers = int(os.getenv('MAX_PARALLEL_WORKERS', '12'))  # Reduced from 16 to 12
        self.query_timeout_seconds = int(os.getenv('QUERY_TIMEOUT_SECONDS', '25'))  # Increased for stability
        
        # STABLE Connection Settings
        self.connection_timeout = int(os.getenv('CONNECTION_TIMEOUT', '15'))  # Increased
        self.command_timeout = int(os.getenv('COMMAND_TIMEOUT', '30'))  # Increased
        
        # ROBUST Query Configuration
        self.max_results = int(os.getenv('MAX_RESULTS', '100'))
        self.max_batch_size = int(os.getenv('MAX_BATCH_SIZE', '12'))  # Conservative batch size
        self.rate_limit_delay = float(os.getenv('RATE_LIMIT_DELAY', '0.5'))  # Increased delay
        
        # PERFORMANCE vs STABILITY Options
        self.use_fast_queries = os.getenv('USE_FAST_QUERIES', 'true').lower() == 'true'
        self.exclude_backup_tables = os.getenv('EXCLUDE_BACKUP_TABLES', 'false').lower() == 'true'
        
        # ROBUST Processing Features
        self.enable_adaptive_performance = os.getenv('ENABLE_ADAPTIVE_PERFORMANCE', 'true').lower() == 'true'
        self.enable_retry_logic = os.getenv('ENABLE_RETRY_LOGIC', 'true').lower() == 'true'
        self.max_retry_attempts = int(os.getenv('MAX_RETRY_ATTEMPTS', '1'))
        
        # Sample Data Configuration
        self.samples_per_object = int(os.getenv('SAMPLES_PER_OBJECT', '5'))
        self.max_sample_rows = int(os.getenv('MAX_SAMPLE_ROWS', '5'))
        
        # Advanced Stability Settings
        self.enable_connection_pooling = os.getenv('ENABLE_CONNECTION_POOLING', 'true').lower() == 'true'
        self.batch_processing_delay = float(os.getenv('BATCH_PROCESSING_DELAY', '0.5'))  # Pause between batches
        self.resource_management = os.getenv('RESOURCE_MANAGEMENT', 'conservative').lower()
    
    def apply_stability_optimizations(self):
        """Apply stability optimizations for large datasets"""
        print("🛡️ ROBUST MODE: Applying stability optimizations for 678+ objects")
        
        # Adaptive worker calculation based on system capability
        if self.enable_adaptive_performance:
            # Conservative approach: never exceed system limits
            original_workers = self.max_parallel_workers
            
            if original_workers > 16:
                self.max_parallel_workers = 12  # Cap for stability
                print(f"   🔧 Reduced workers: {original_workers} → {self.max_parallel_workers} (stability)")
            elif original_workers > 12:
                self.max_parallel_workers = 10  # Conservative reduction
                print(f"   🔧 Reduced workers: {original_workers} → {self.max_parallel_workers} (stability)")
            
            # Adaptive batch size based on workers
            self.adaptive_batch_size = max(self.max_parallel_workers * 2, 16)
            
            # Adaptive timeout based on system load
            if self.resource_management == 'conservative':
                self.query_timeout_seconds = min(self.query_timeout_seconds + 10, 45)
                print(f"   ⏱️ Conservative timeout: {self.query_timeout_seconds}s")
        
        print(f"   ⚙️ Final configuration: {self.max_parallel_workers} workers, {self.max_batch_size} batch size")
        print(f"   🔄 Retry logic: {'Enabled' if self.enable_retry_logic else 'Disabled'}")
        print(f"   📊 Resource management: {self.resource_management}")
    
    def validate_config(self):
        """Validate required configuration with helpful error messages"""
        missing = []
        
        if not self.api_key:
            missing.append('AZURE_OPENAI_API_KEY')
        
        if not self.connection_string:
            missing.append('DATABASE_CONNECTION_STRING')
        
        if missing:
            print("❌ Configuration Error!")
            print("Missing required environment variables:")
            for var in missing:
                print(f"   • {var}")
            print("\n💡 Please check your .env file and ensure these variables are set:")
            print("   AZURE_OPENAI_API_KEY=your_key_here")
            print("   DATABASE_CONNECTION_STRING=your_connection_string_here")
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    def get_cache_path(self, filename: str) -> Path:
        """Get full path for cache file"""
        return self.cache_directory / filename
    
    def get_database_connection_string(self) -> str:
        """Get ROBUST database connection string with stability optimizations"""
        if self.connection_string:
            # Use provided connection string with STABILITY enhancements
            connection_string = f"{self.connection_string};"
            
            # Add stability-focused options
            stability_options = [
                f"Connection Timeout={self.connection_timeout}",
                f"Command Timeout={self.command_timeout}",
                "MARS_Connection=yes",
                "MultipleActiveResultSets=true",
                "Pooling=true",
                "Max Pool Size=50",      # Reduced from 100 for stability
                "Min Pool Size=2",       # Reduced from 5
                "Connection Lifetime=600",  # Increased to 10 minutes
                "Enlist=false",
                "Application Name=SemanticRAG_Robust"  # For monitoring
            ]
            
            # Add options that aren't already present
            for option in stability_options:
                key = option.split('=')[0]
                if key not in connection_string:
                    connection_string += f"{option};"
        else:
            # Build ROBUST connection string from components
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"MARS_Connection=yes;"
                f"MultipleActiveResultSets=true;"
                f"Connection Timeout={self.connection_timeout};"
                f"Command Timeout={self.command_timeout};"
                f"Query Timeout={self.query_timeout_seconds};"
                f"Pooling=true;"
                f"Max Pool Size=50;"
                f"Min Pool Size=2;"
                f"Connection Lifetime=600;"
                f"Enlist=false;"
                f"Application Name=SemanticRAG_Robust;"
            )
        
        # Ensure UTF-8 support for Greek text
        if "charset" not in connection_string.lower() and "encoding" not in connection_string.lower():
            connection_string += "charset=UTF8;"
        
        return connection_string.rstrip(';')
    
    def get_performance_summary(self) -> str:
        """Get summary of robust performance configuration"""
        return f"""
🛡️ ROBUST Configuration Summary:
   • Discovery scope: {'ALL objects (unlimited)' if self.max_discovery_objects is None else f'{self.max_discovery_objects} objects'}
   • Parallel workers: {self.max_parallel_workers} (adaptive/conservative)
   • Batch size: {self.max_batch_size}
   • Timeout per object: {self.query_timeout_seconds}s
   • Samples per object: {self.samples_per_object}
   • Rate limit delay: {self.rate_limit_delay}s
   • FAST queries: {'Enabled' if self.use_fast_queries else 'Disabled'}
   • Adaptive performance: {'Enabled' if self.enable_adaptive_performance else 'Disabled'}
   • Retry logic: {'Enabled' if self.enable_retry_logic else 'Disabled'}
   • Resource management: {self.resource_management}
   • Connection pooling: {'Enabled' if self.enable_connection_pooling else 'Disabled'}
        """.strip()
    
    def get_optimal_batch_size(self, total_objects: int) -> int:
        """Get optimal batch size based on dataset size and stability"""
        base_batch = self.max_batch_size
        
        if total_objects > 500:
            # Large datasets: conservative batching for stability
            return max(base_batch, 16)
        elif total_objects > 200:
            # Medium datasets: balanced approach
            return max(base_batch, 12)
        else:
            # Small datasets: can use smaller batches
            return max(base_batch, 8)
    
    def get_optimal_workers(self, total_objects: int) -> int:
        """Get optimal worker count based on dataset size and stability"""
        base_workers = self.max_parallel_workers
        
        if total_objects > 500:
            # Large datasets: cap workers for stability
            return min(base_workers, 12)
        elif total_objects > 200:
            # Medium datasets: moderate workers
            return min(base_workers, 10)
        else:
            # Small datasets: can use more workers
            return min(base_workers, 8)
    
    def get_stability_recommendations(self, object_count: int) -> List[str]:
        """Get stability recommendations based on dataset size"""
        recommendations = []
        
        if object_count > 500:
            recommendations.extend([
                f"🛡️ Large dataset detected ({object_count} objects)",
                "⚡ Using conservative parallel processing for stability",
                "🔄 Retry logic enabled for failed objects",
                "⏱️ Extended timeouts to prevent failures",
                "📊 Adaptive batching based on system performance",
                "💾 Results will be cached for faster subsequent runs"
            ])
        elif object_count > 200:
            recommendations.extend([
                f"📊 Medium dataset ({object_count} objects)",
                "⚖️ Balanced performance and stability settings",
                "🔄 Retry logic available if needed",
                "⏱️ Expected completion: 10-20 minutes"
            ])
        else:
            recommendations.extend([
                f"✅ Small dataset ({object_count} objects)",
                "🏃 Standard processing expected: 5-10 minutes",
                "⚡ Can use higher parallelism safely"
            ])
        
        return recommendations