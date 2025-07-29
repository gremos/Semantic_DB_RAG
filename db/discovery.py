#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficient Database Discovery Module
High-performance database discovery with parallel processing and smart filtering
"""

import pyodbc
import asyncio
import concurrent.futures
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
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
    """Configuration class"""
    def __init__(self):
        self.server = os.getenv('DB_SERVER', 'localhost')
        self.database = os.getenv('DB_DATABASE', 'master')
        self.username = os.getenv('DB_USERNAME', '')
        self.password = os.getenv('DB_PASSWORD', '')
        self.discovery_cache_hours = 24
        
    def get_database_connection_string(self):
        return (f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"MARS_Connection=yes;"
                f"Connection Timeout=30;"
                f"Query Timeout=60;")
    
    def get_cache_path(self, filename: str) -> Path:
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / filename

class DatabaseObject:
    def __init__(self, schema: str, name: str, object_type: str, estimated_rows: int = 0):
        self.schema = schema
        self.name = name
        self.object_type = object_type
        self.estimated_rows = estimated_rows
        self.priority = self._calculate_priority()
    
    def _calculate_priority(self) -> int:
        """Calculate object priority for processing order"""
        priority = 0
        
        # Tables get higher priority than views
        if self.object_type == 'BASE TABLE':
            priority += 100
        
        # Objects with data get higher priority
        if self.estimated_rows > 0:
            priority += min(50, self.estimated_rows // 1000)
        
        # Penalize objects that are likely problematic
        name_lower = self.name.lower()
        if any(word in name_lower for word in ['temp', 'tmp', 'backup', 'bck', 'log', 'audit']):
            priority -= 50
        
        # Boost common business objects
        if any(word in name_lower for word in ['customer', 'product', 'order', 'sales', 'user']):
            priority += 30
        
        return priority

class EfficientDatabaseDiscovery:
    """High-performance database discovery with parallel processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables = []
        self.stats = {
            'total_objects': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'timeouts': 0
        }
    
    def get_database_connection(self):
        """Get optimized database connection"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # Optimize for Unicode and performance
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
        return conn
    
    async def discover_database_efficient(self, limit: Optional[int] = None, 
                                        max_workers: int = 8, 
                                        timeout_seconds: int = 30) -> bool:
        """
        Efficient database discovery with parallel processing
        
        Args:
            limit: Maximum number of objects to process
            max_workers: Number of parallel workers
            timeout_seconds: Timeout for individual object analysis
        """
        print("🚀 Starting efficient database discovery...")
        
        # Check cache first
        if self._load_from_cache():
            print(f"✅ Loaded {len(self.tables)} objects from cache")
            return True
        
        try:
            # Get database objects with smart filtering
            objects = await self._get_database_objects_smart()
            if not objects:
                print("❌ No database objects found")
                return False
            
            # Apply smart filtering and prioritization
            filtered_objects = self._filter_and_prioritize_objects(objects, limit)
            
            self._log_discovery_plan(objects, filtered_objects, max_workers, timeout_seconds)
            
            # Process objects in parallel with timeout control
            await self._process_objects_parallel(filtered_objects, max_workers, timeout_seconds)
            
            # Save results
            await self._save_results()
            
            self._log_completion_stats()
            return len(self.tables) > 0
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    async def _get_database_objects_smart(self) -> List[DatabaseObject]:
        """Get database objects with smart, fast queries"""
        print("📊 Discovering database objects with smart filtering...")
        
        # Simplified, fast query focusing on essential information
        query = """
        SELECT 
            SCHEMA_NAME(t.schema_id) as schema_name,
            t.name as table_name,
            'BASE TABLE' as object_type,
            COALESCE(p.rows, 0) as estimated_rows
        FROM sys.tables t
        LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id < 2
        WHERE t.is_ms_shipped = 0
          AND SCHEMA_NAME(t.schema_id) NOT IN ('sys', 'information_schema')
        
        UNION ALL
        
        SELECT 
            SCHEMA_NAME(v.schema_id) as schema_name,
            v.name as view_name,
            'VIEW' as object_type,
            100 as estimated_rows  -- Conservative estimate for views
        FROM sys.views v
        WHERE v.is_ms_shipped = 0
          AND SCHEMA_NAME(v.schema_id) NOT IN ('sys', 'information_schema')
          AND v.name NOT LIKE '%backup%'
          AND v.name NOT LIKE '%temp%'
          AND v.name NOT LIKE '%tmp%'
        
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
                
                self.stats['total_objects'] = len(objects)
                return objects
                
        except Exception as e:
            print(f"❌ Failed to get database objects: {e}")
            return []
    
    def _filter_and_prioritize_objects(self, objects: List[DatabaseObject], 
                                     limit: Optional[int]) -> List[DatabaseObject]:
        """Filter and prioritize objects for processing"""
        
        # Filter out obvious problem objects
        filtered = []
        for obj in objects:
            name_lower = obj.name.lower()
            
            # Skip obvious system/temp/backup objects
            if any(pattern in name_lower for pattern in [
                'sysdiagram', 'dtproperties', '__refactorlog', 'aspnet_',
                'backup', 'bckp', '_bck', 'temp_', 'tmp_', '_temp',
                'log_', '_log', 'audit_', '_audit', 'trace_'
            ]):
                self.stats['skipped'] += 1
                continue
            
            # Skip views that are likely to be problematic (contain external references)
            if obj.object_type == 'VIEW' and any(indicator in name_lower for indicator in [
                'to', 'from', 'external', 'linked', 'remote'
            ]):
                self.stats['skipped'] += 1
                continue
            
            filtered.append(obj)
        
        # Sort by priority (highest first)
        filtered.sort(key=lambda x: x.priority, reverse=True)
        
        # Apply limit
        if limit and limit < len(filtered):
            filtered = filtered[:limit]
        
        return filtered
    
    def _log_discovery_plan(self, all_objects: List[DatabaseObject], 
                          filtered_objects: List[DatabaseObject],
                          max_workers: int, timeout_seconds: int):
        """Log the discovery plan"""
        tables = sum(1 for obj in filtered_objects if obj.object_type == 'BASE TABLE')
        views = sum(1 for obj in filtered_objects if obj.object_type == 'VIEW')
        
        print(f"📊 Discovery Plan:")
        print(f"   • Total objects found: {len(all_objects)}")
        print(f"   • Objects to process: {len(filtered_objects)} (Tables: {tables}, Views: {views})")
        print(f"   • Skipped objects: {self.stats['skipped']}")
        print(f"   • Parallel workers: {max_workers}")
        print(f"   • Timeout per object: {timeout_seconds}s")
        print(f"   • Estimated completion: ~{len(filtered_objects) // max_workers // 2} minutes")
    
    async def _process_objects_parallel(self, objects: List[DatabaseObject], 
                                      max_workers: int, timeout_seconds: int):
        """Process objects in parallel with timeout control"""
        print(f"\n🔄 Processing {len(objects)} objects with {max_workers} workers...")
        
        # Create progress bar
        pbar = tqdm(total=len(objects), desc="Analyzing objects", unit="obj")
        
        # Process in batches to avoid overwhelming the database
        batch_size = max_workers * 2
        
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            
            # Process batch with ThreadPoolExecutor for true parallelism
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks in the batch
                future_to_object = {
                    executor.submit(self._analyze_object_with_timeout, obj, timeout_seconds): obj
                    for obj in batch
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_object):
                    obj = future_to_object[future]
                    try:
                        result = future.result()
                        if result:
                            self.tables.append(result)
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                    except concurrent.futures.TimeoutError:
                        self.stats['timeouts'] += 1
                    except Exception as e:
                        self.stats['failed'] += 1
                    
                    self.stats['processed'] += 1
                    pbar.update(1)
                    pbar.set_description(f"Processed: {self.stats['successful']}/{self.stats['processed']}")
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        pbar.close()
    
    def _analyze_object_with_timeout(self, obj: DatabaseObject, timeout_seconds: int) -> Optional[Dict]:
        """Analyze a single object with timeout control"""
        try:
            # Use a separate connection for each thread
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                
                # Set query timeout
                cursor.timeout = timeout_seconds
                
                return self._analyze_object_fast(cursor, obj)
                
        except pyodbc.Error as e:
            error_code = str(e)
            if 'timeout' in error_code.lower():
                # Don't log timeout errors, they're expected
                pass
            elif '42S02' in error_code:
                # Object not found - common for views
                pass
            else:
                print(f"      ❌ {obj.schema}.{obj.name}: {str(e)[:50]}...")
            return None
        except Exception as e:
            print(f"      ❌ {obj.schema}.{obj.name}: {str(e)[:50]}...")
            return None
    
    def _analyze_object_fast(self, cursor, obj: DatabaseObject) -> Optional[Dict]:
        """Fast analysis of a database object"""
        full_name = f"[{obj.schema}].[{obj.name}]"
        
        try:
            # Get basic column info (fast query)
            columns = self._get_columns_fast(cursor, obj.schema, obj.name)
            if not columns:
                return None
            
            # Get sample data (with aggressive timeout)
            sample_data = self._get_sample_data_fast(cursor, obj, full_name)
            
            return {
                'name': obj.name,
                'schema': obj.schema,
                'full_name': full_name,
                'object_type': obj.object_type,
                'row_count': obj.estimated_rows,
                'columns': columns,
                'sample_data': sample_data,
                'relationships': [],
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception:
            return None
    
    def _get_columns_fast(self, cursor, schema: str, name: str) -> List[Dict]:
        """Get column information with fast, simple query"""
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
                    'default': row[3]
                })
            
            return columns
        except:
            return []
    
    def _get_sample_data_fast(self, cursor, obj: DatabaseObject, full_name: str) -> List[Dict]:
        """Get sample data with very aggressive optimization"""
        # Different strategies for tables vs views
        if obj.object_type == 'BASE TABLE':
            queries = [
                f"SELECT TOP 3 * FROM {full_name} OPTION (FAST 1)",
                f"SELECT TOP 1 * FROM {full_name}"
            ]
        else:
            queries = [
                f"SELECT TOP 1 * FROM {full_name} OPTION (FAST 1)",
                f"SELECT TOP 1 * FROM {full_name}"
            ]
        
        for query in queries:
            try:
                cursor.execute(query)
                
                if cursor.description:
                    col_names = [col[0] for col in cursor.description]
                    sample_data = []
                    
                    for row in cursor.fetchall():
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(col_names):
                                # Simple value conversion
                                if value is None:
                                    row_dict[col_names[i]] = None
                                elif isinstance(value, (int, float, str, bool)):
                                    row_dict[col_names[i]] = value
                                else:
                                    row_dict[col_names[i]] = str(value)[:100]  # Truncate long values
                        sample_data.append(row_dict)
                    
                    return sample_data
            except:
                continue
        
        return []
    
    def _load_from_cache(self) -> bool:
        """Load from cache if available and recent"""
        cache_file = self.config.get_cache_path("database_structure_efficient.json")
        
        if not cache_file.exists():
            return False
        
        try:
            # Check if cache is recent (within cache hours)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.config.discovery_cache_hours * 3600):
                return False
            
            import json
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'tables' in data:
                self.tables = data['tables']
                return True
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
        
        return False
    
    async def _save_results(self):
        """Save results to cache"""
        cache_file = self.config.get_cache_path("database_structure_efficient.json")
        
        data = {
            'tables': self.tables,
            'created': datetime.now().isoformat(),
            'version': '3.0-efficient',
            'stats': self.stats
        }
        
        try:
            import json
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Results saved to {cache_file}")
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
    
    def _log_completion_stats(self):
        """Log completion statistics"""
        tables = sum(1 for t in self.tables if t.get('object_type') == 'BASE TABLE')
        views = sum(1 for t in self.tables if t.get('object_type') == 'VIEW')
        
        print(f"\n✅ Efficient discovery completed!")
        print(f"   📊 Processed: {self.stats['processed']} objects")
        print(f"   ✅ Successful: {self.stats['successful']} (Tables: {tables}, Views: {views})")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   ⏰ Timeouts: {self.stats['timeouts']}")
        print(f"   ⏭️ Skipped: {self.stats['skipped']}")
        
        success_rate = (self.stats['successful'] / max(self.stats['processed'], 1)) * 100
        print(f"   📈 Success rate: {success_rate:.1f}%")


# Usage example
async def main():
    """Main function to run efficient discovery"""
    config = Config()
    discovery = EfficientDatabaseDiscovery(config)
    
    # Run efficient discovery with customizable parameters
    success = await discovery.discover_database_efficient(
        limit=100,              # Process first 100 high-priority objects
        max_workers=12,         # Use 12 parallel workers
        timeout_seconds=15      # 15 second timeout per object
    )
    
    if success:
        print(f"\n🎉 Discovery successful! Found {len(discovery.tables)} objects")
        
        # Show sample of discovered objects
        if discovery.tables:
            print(f"\n📋 Sample discovered objects:")
            for i, table in enumerate(discovery.tables[:5]):
                obj_type = table.get('object_type', 'Unknown')
                col_count = len(table.get('columns', []))
                sample_count = len(table.get('sample_data', []))
                print(f"   {i+1}. {table['full_name']} ({obj_type}) - {col_count} cols, {sample_count} samples")
    else:
        print("❌ Discovery failed")

if __name__ == "__main__":
    asyncio.run(main())