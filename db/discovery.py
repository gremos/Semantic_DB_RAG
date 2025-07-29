#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed and Optimized Database Discovery Module
Resolves the method name error and adds massive performance improvements
"""

import pyodbc
import asyncio
import concurrent.futures
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

class Config:
    """Enhanced configuration class"""
    def __init__(self):
        # Azure OpenAI Configuration
        self.model_version = os.getenv('MODEL_VERSION', '2024-12-01-preview')
        self.azure_endpoint = os.getenv('AZURE_ENDPOINT', 'https://gyp-weu-02-res01-prdenv-cog01.openai.azure.com/')
        self.deployment_name = os.getenv('DEPLOYMENT_NAME', 'gpt-4.1-mini')
        
        # Database Configuration
        self.server = os.getenv('DB_SERVER', 'localhost')
        self.database = os.getenv('DB_DATABASE', 'master')
        self.username = os.getenv('DB_USERNAME', '')
        self.password = os.getenv('DB_PASSWORD', '')
        
        # Discovery Settings
        self.discovery_cache_hours = int(os.getenv('DISCOVERY_CACHE_HOURS', '24'))
        self.max_discovery_objects = int(os.getenv('MAX_DISCOVERY_OBJECTS', '50'))
        self.max_parallel_workers = int(os.getenv('MAX_PARALLEL_WORKERS', '8'))
        self.query_timeout_seconds = int(os.getenv('QUERY_TIMEOUT_SECONDS', '30'))
        
    def get_database_connection_string(self):
        """Get optimized database connection string"""
        return (f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"MARS_Connection=yes;"
                f"Connection Timeout=20;"
                f"Query Timeout={self.query_timeout_seconds};")
    
    def get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / filename

class TableInfo:
    """Table information class"""
    def __init__(self, name: str, schema: str, full_name: str, object_type: str, 
                 row_count: int, columns: List[Dict], sample_data: List[Dict], 
                 relationships: List = None, query_performance: Dict = None):
        self.name = name
        self.schema = schema
        self.full_name = full_name
        self.object_type = object_type
        self.row_count = row_count
        self.columns = columns
        self.sample_data = sample_data
        self.relationships = relationships or []
        self.query_performance = query_performance or {}

class DatabaseObject:
    """Database object info"""
    def __init__(self, schema: str, name: str, object_type: str, estimated_rows: int = 0):
        self.schema = schema
        self.name = name
        self.object_type = object_type
        self.estimated_rows = estimated_rows
        self.priority = self._calculate_priority()
    
    def _calculate_priority(self) -> int:
        """Calculate priority score for processing order"""
        score = 0
        name_lower = self.name.lower()
        
        # Tables get higher priority than views
        if self.object_type == 'BASE TABLE':
            score += 100
        elif self.object_type == 'TABLE':
            score += 100
        
        # Objects with data get priority
        if self.estimated_rows > 0:
            score += min(50, self.estimated_rows // 1000)
        
        # Business objects get priority
        business_keywords = [
            'customer', 'product', 'order', 'sales', 'user', 'account',
            'transaction', 'payment', 'invoice', 'contract', 'person',
            'company', 'address', 'contact', 'item', 'service'
        ]
        if any(word in name_lower for word in business_keywords):
            score += 30
        
        # Penalize problematic objects
        problem_keywords = [
            'temp', 'tmp', 'backup', 'bck', 'log', 'audit', 'trace',
            'error', 'debug', 'test', 'staging', 'stage'
        ]
        if any(word in name_lower for word in problem_keywords):
            score -= 50
        
        return score

class AnalysisStats:
    """Statistics tracking"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_objects_found = 0
        self.objects_processed = 0
        self.successful_analyses = 0
        self.analysis_errors = 0
        self.objects_excluded = 0
        self.backup_tables_excluded = 0
        self.sample_data_errors = 0
        self.fast_query_successes = 0
    
    def to_dict(self):
        return self.__dict__

def safe_database_value(value):
    """Safely convert database value to JSON-serializable format"""
    if value is None:
        return None
    elif isinstance(value, (int, float, str, bool)):
        return value
    elif hasattr(value, 'isoformat'):  # datetime objects
        return value.isoformat()
    else:
        # Convert to string and truncate if too long
        str_value = str(value)
        return str_value[:500] if len(str_value) > 500 else str_value

def should_exclude_table(table_name: str, schema_name: str) -> bool:
    """Check if table should be excluded"""
    name_lower = table_name.lower()
    schema_lower = schema_name.lower()
    
    # System schemas
    if schema_lower in ['sys', 'information_schema', 'db_owner', 'db_accessadmin']:
        return True
    
    # System tables
    system_patterns = [
        'sysdiagram', 'dtproperties', '__refactorlog', 'aspnet_',
        'elmah_', 'webpages_', 'migrationhistory'
    ]
    if any(pattern in name_lower for pattern in system_patterns):
        return True
    
    # Backup and temp tables
    backup_patterns = [
        'backup', 'bckp', '_bck', 'bck_', 'temp_', 'tmp_', '_temp', '_tmp'
    ]
    if any(pattern in name_lower for pattern in backup_patterns):
        return True
    
    return False

class DatabaseDiscovery:
    """Fixed and optimized database discovery class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.stats = AnalysisStats()
    
    def get_database_connection(self):
        """Get optimized database connection with proper Greek text support"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # Critical: Proper Unicode handling for Greek text
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
        return conn
    
    # 🔧 FIXED: Method name matches expected interface
    async def discover_database(self, limit: Optional[int] = None) -> bool:
        """
        FIXED: Main discovery method with proper name and massive performance improvements
        
        Args:
            limit: Maximum number of objects to process (defaults to config setting)
        
        Returns:
            bool: True if discovery was successful
        """
        print("🚀 Starting FIXED and OPTIMIZED database discovery...")
        
        # Use config default if no limit provided
        if limit is None:
            limit = self.config.max_discovery_objects
        
        # Check cache first
        if self.load_from_cache():
            print(f"✅ Loaded {len(self.tables)} objects from cache")
            return True
        
        self.stats.reset()
        
        try:
            # Step 1: Fast object discovery with smart filtering
            print("📊 Discovering objects with smart filtering...")
            objects = await self._get_objects_smart_and_fast()
            
            if not objects:
                print("❌ No database objects found")
                return False
            
            # Step 2: Apply intelligent filtering and prioritization
            filtered_objects = self._apply_smart_filtering(objects, limit)
            
            # Step 3: Log discovery plan
            self._log_discovery_plan(objects, filtered_objects)
            
            # Step 4: Process objects with parallel execution and timeout control
            await self._process_objects_parallel_optimized(filtered_objects)
            
            # Step 5: Save results
            await self._save_discovery_results()
            
            # Step 6: Log completion stats
            self._log_completion_stats()
            
            return len(self.tables) > 0
            
        except Exception as e:
            print(f"❌ Discovery failed with error: {e}")
            return False
    
    async def _get_objects_smart_and_fast(self) -> List[DatabaseObject]:
        """Get database objects with smart, fast query that pre-filters problems"""
        
        # Ultra-optimized query that excludes problems upfront
        query = """
        -- Get tables (highest priority)
        SELECT TOP 200
            SCHEMA_NAME(t.schema_id) as schema_name,
            t.name as table_name,
            'BASE TABLE' as object_type,
            COALESCE(p.rows, 0) as estimated_rows
        FROM sys.tables t
        LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id < 2
        WHERE t.is_ms_shipped = 0
          AND SCHEMA_NAME(t.schema_id) NOT IN ('sys', 'information_schema')
          AND t.name NOT LIKE '%backup%'
          AND t.name NOT LIKE '%bck%'
          AND t.name NOT LIKE '%temp%'
          AND t.name NOT LIKE '%tmp%'
          AND t.name NOT LIKE '%log%'
          AND t.name NOT LIKE '%audit%'
          AND t.name NOT LIKE '%trace%'
        
        UNION ALL
        
        -- Get views (lower priority, more filtered)
        SELECT TOP 100
            SCHEMA_NAME(v.schema_id) as schema_name,
            v.name as view_name,
            'VIEW' as object_type,
            100 as estimated_rows
        FROM sys.views v
        WHERE v.is_ms_shipped = 0
          AND SCHEMA_NAME(v.schema_id) NOT IN ('sys', 'information_schema')
          AND v.name NOT LIKE '%backup%'
          AND v.name NOT LIKE '%temp%'
          AND v.name NOT LIKE '%bck%'
          AND v.name NOT LIKE '%To%'        -- Often problematic cross-db views
          AND v.name NOT LIKE '%External%'
          AND v.name NOT LIKE '%Linked%'
          AND v.name NOT LIKE '%XO%'        -- Based on your errors, these seem problematic
        
        ORDER BY object_type DESC, estimated_rows DESC
        """
        
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                objects = []
                for row in cursor.fetchall():
                    objects.append(DatabaseObject(
                        schema=row[0],
                        name=row[1],
                        object_type=row[2],
                        estimated_rows=row[3]
                    ))
                
                self.stats.total_objects_found = len(objects)
                print(f"   Found {len(objects)} objects after smart pre-filtering")
                return objects
                
        except Exception as e:
            print(f"❌ Failed to get database objects: {e}")
            return []
    
    def _apply_smart_filtering(self, objects: List[DatabaseObject], limit: int) -> List[DatabaseObject]:
        """Apply intelligent filtering and prioritization"""
        
        filtered = []
        excluded_count = 0
        
        for obj in objects:
            # Apply additional filtering
            if should_exclude_table(obj.name, obj.schema):
                excluded_count += 1
                continue
            
            filtered.append(obj)
        
        # Sort by priority (highest first)
        filtered.sort(key=lambda x: x.priority, reverse=True)
        
        # Apply limit
        if limit and limit < len(filtered):
            filtered = filtered[:limit]
        
        self.stats.objects_excluded = excluded_count
        return filtered
    
    def _log_discovery_plan(self, all_objects: List[DatabaseObject], filtered_objects: List[DatabaseObject]):
        """Log the discovery plan"""
        tables = sum(1 for obj in filtered_objects if obj.object_type in ['BASE TABLE', 'TABLE'])
        views = sum(1 for obj in filtered_objects if obj.object_type == 'VIEW')
        
        print(f"📊 Discovery Plan:")
        print(f"   • Total objects found: {len(all_objects)}")
        print(f"   • Objects to process: {len(filtered_objects)} (Tables: {tables}, Views: {views})")
        print(f"   • Excluded objects: {self.stats.objects_excluded}")
        print(f"   • Max parallel workers: {self.config.max_parallel_workers}")
        print(f"   • Timeout per object: {self.config.query_timeout_seconds}s")
        
        estimated_time = (len(filtered_objects) // self.config.max_parallel_workers) * 3
        print(f"   • Estimated completion: ~{estimated_time} minutes")
    
    async def _process_objects_parallel_optimized(self, objects: List[DatabaseObject]):
        """Process objects with parallel execution and aggressive optimization"""
        print(f"\n🔄 Processing {len(objects)} objects with {self.config.max_parallel_workers} workers...")
        
        # Progress bar
        pbar = tqdm(total=len(objects), desc="Analyzing objects", unit="obj")
        
        # Process in batches to avoid overwhelming database
        batch_size = self.config.max_parallel_workers * 2
        
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            
            # Process batch with true parallelism
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                # Submit all tasks
                future_to_object = {
                    executor.submit(self._analyze_object_with_timeout, obj): obj
                    for obj in batch
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_object, timeout=300):
                    obj = future_to_object[future]
                    
                    try:
                        result = future.result(timeout=self.config.query_timeout_seconds)
                        if result:
                            self.tables.append(result)
                            self.stats.successful_analyses += 1
                        else:
                            self.stats.analysis_errors += 1
                    except concurrent.futures.TimeoutError:
                        self.stats.analysis_errors += 1
                    except Exception:
                        self.stats.analysis_errors += 1
                    
                    self.stats.objects_processed += 1
                    pbar.update(1)
                    pbar.set_description(f"Success: {self.stats.successful_analyses}/{self.stats.objects_processed}")
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        pbar.close()
    
    def _analyze_object_with_timeout(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze single object with timeout and error handling"""
        try:
            # Use separate connection for thread safety
            with self.get_database_connection() as conn:
                conn.timeout = self.config.query_timeout_seconds
                cursor = conn.cursor()
                
                return self._analyze_object_fast_and_safe(cursor, obj)
                
        except Exception:
            # Silent fail for timeouts and expected errors
            return None
    
    def _analyze_object_fast_and_safe(self, cursor, obj: DatabaseObject) -> Optional[TableInfo]:
        """Fast and safe object analysis"""
        full_name = f"[{obj.schema}].[{obj.name}]"
        
        try:
            # Get column info with simple query
            columns = self._get_columns_simple(cursor, obj.schema, obj.name)
            if not columns:
                return None
            
            # Get sample data with aggressive optimization
            sample_data = self._get_sample_data_ultra_fast(cursor, obj, full_name)
            
            return TableInfo(
                name=obj.name,
                schema=obj.schema,
                full_name=full_name,
                object_type=obj.object_type,
                row_count=obj.estimated_rows,
                columns=columns,
                sample_data=sample_data,
                relationships=[],
                query_performance={'fast_optimized': True}
            )
            
        except Exception:
            return None
    
    def _get_columns_simple(self, cursor, schema: str, name: str) -> List[Dict[str, Any]]:
        """Get column info with simple, fast query"""
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        
        try:
            cursor.execute(query, schema, name)
            columns = []
            
            for row in cursor.fetchall():
                columns.append({
                    'name': row[0],
                    'data_type': row[1],
                    'nullable': row[2] == 'YES',
                    'default': row[3],
                    'is_primary_key': False,  # Skip expensive key lookups
                    'is_foreign_key': False
                })
            
            return columns
        except:
            return []
    
    def _get_sample_data_ultra_fast(self, cursor, obj: DatabaseObject, full_name: str) -> List[Dict[str, Any]]:
        """Get sample data with ultra-fast, simple queries"""
        
        # Different strategies for tables vs views
        if obj.object_type in ['BASE TABLE', 'TABLE']:
            queries = [
                f"SELECT TOP 3 * FROM {full_name}",
                f"SELECT TOP 1 * FROM {full_name}"
            ]
        else:
            queries = [
                f"SELECT TOP 1 * FROM {full_name}",
            ]
        
        for query in queries:
            try:
                cursor.execute(query)
                
                if not cursor.description:
                    continue
                
                col_names = [col[0] for col in cursor.description]
                sample_data = []
                
                for row in cursor.fetchall():
                    row_dict = {}
                    for i, value in enumerate(row):
                        if i < len(col_names):
                            row_dict[col_names[i]] = safe_database_value(value)
                    sample_data.append(row_dict)
                
                return sample_data
                
            except:
                continue
        
        return []
    
    async def _save_discovery_results(self):
        """Save discovery results to cache"""
        print(f"\n💾 Saving optimized results to cache...")
        
        cache_file = self.config.get_cache_path("database_structure_optimized.json")
        
        # Convert TableInfo objects to dictionaries
        tables_data = []
        for table in self.tables:
            tables_data.append({
                'name': table.name,
                'schema': table.schema,
                'full_name': table.full_name,
                'object_type': table.object_type,
                'row_count': table.row_count,
                'columns': table.columns,
                'sample_data': table.sample_data,
                'relationships': table.relationships,
                'query_performance': table.query_performance
            })
        
        data = {
            'tables': tables_data,
            'created': datetime.now().isoformat(),
            'version': '3.0-optimized',
            'analysis_stats': self.stats.to_dict(),
            'config_used': {
                'max_objects': self.config.max_discovery_objects,
                'max_workers': self.config.max_parallel_workers,
                'timeout_seconds': self.config.query_timeout_seconds
            }
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✅ Results saved to {cache_file}")
        except Exception as e:
            print(f"   ❌ Failed to save cache: {e}")
    
    def _log_completion_stats(self):
        """Log detailed completion statistics"""
        tables = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        views = sum(1 for t in self.tables if t.object_type == 'VIEW')
        
        print(f"\n✅ OPTIMIZED discovery completed!")
        print(f"   📊 Total processed: {self.stats.objects_processed} objects")
        print(f"   ✅ Successful: {self.stats.successful_analyses} (Tables: {tables}, Views: {views})")
        print(f"   ❌ Failed: {self.stats.analysis_errors}")
        print(f"   ⏭️ Excluded: {self.stats.objects_excluded}")
        
        if self.stats.objects_processed > 0:
            success_rate = (self.stats.successful_analyses / self.stats.objects_processed) * 100
            print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if self.tables:
            print(f"\n📋 Sample discovered objects:")
            for i, table in enumerate(self.tables[:5]):
                cols = len(table.columns)
                samples = len(table.sample_data)
                print(f"   {i+1}. {table.full_name} ({table.object_type}) - {cols} cols, {samples} samples")
    
    def load_from_cache(self) -> bool:
        """Load from cache if available and recent"""
        cache_file = self.config.get_cache_path("database_structure_optimized.json")
        
        if not cache_file.exists():
            return False
        
        try:
            # Check cache age
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.config.discovery_cache_hours * 3600):
                return False
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'tables' in data:
                self.tables = []
                for table_data in data['tables']:
                    table = TableInfo(
                        name=table_data['name'],
                        schema=table_data['schema'],
                        full_name=table_data['full_name'],
                        object_type=table_data['object_type'],
                        row_count=table_data['row_count'],
                        columns=table_data['columns'],
                        sample_data=table_data['sample_data'],
                        relationships=table_data.get('relationships', []),
                        query_performance=table_data.get('query_performance', {})
                    )
                    self.tables.append(table)
                return True
                
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
        
        return False
    
    def get_tables(self) -> List[TableInfo]:
        """Get discovered tables"""
        return self.tables
    
    def get_stats(self) -> AnalysisStats:
        """Get discovery statistics"""
        return self.stats


# Usage example
async def main():
    """Main function demonstrating the fixed discovery"""
    print("🔧 Fixed Database Discovery with Performance Optimization")
    print("=" * 60)
    
    # Initialize with configuration
    config = Config()
    discovery = DatabaseDiscovery(config)
    
    # Run the FIXED discovery method
    print(f"🚀 Starting discovery with limit={config.max_discovery_objects}")
    success = await discovery.discover_database(limit=50)  # Process top 50 objects
    
    if success:
        tables = discovery.get_tables()
        stats = discovery.get_stats()
        
        print(f"\n🎉 Discovery successful!")
        print(f"   📊 Discovered {len(tables)} objects")
        print(f"   ⚡ Success rate: {(stats.successful_analyses/max(stats.objects_processed,1)*100):.1f}%")
        
        # Show sample results
        if tables:
            print(f"\n📋 Sample discovered objects:")
            for i, table in enumerate(tables[:5]):
                cols = len(table.columns)
                samples = len(table.sample_data)
                print(f"   {i+1}. {table.full_name} - {cols} columns, {samples} samples")
        
        return True
    else:
        print("❌ Discovery failed")
        return False

if __name__ == "__main__":
    asyncio.run(main())