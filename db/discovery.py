#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed and Optimized Database Discovery Module
Uses shared Config and models - Clean, maintainable implementation
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

# Import from shared modules
from shared.config import Config
from shared.models import TableInfo, DatabaseObject, AnalysisStats
from shared.utils import safe_database_value, should_exclude_table

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
    
    async def discover_database(self, limit: Optional[int] = None) -> bool:
        """
        Main discovery method with proper name and massive performance improvements
        
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
          AND v.name NOT LIKE '%XO%'        -- Based on errors, these seem problematic
        
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
        
        # Sort by priority (highest first) - using estimated_rows as simple priority
        filtered.sort(key=lambda x: x.estimated_rows, reverse=True)
        
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
        
        cache_file = self.config.get_cache_path("database_structure.json")
        
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
        cache_file = self.config.get_cache_path("database_structure.json")
        
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
    
    # Run the discovery method
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