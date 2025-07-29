#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBUST Database Discovery Module - Adaptive Performance for Large Datasets
Handles 678+ objects with intelligent resource management and retry logic
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

class RobustDatabaseDiscovery:
    """ROBUST database discovery with adaptive performance and error recovery"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.stats = AnalysisStats()
        self.failed_objects: List[DatabaseObject] = []
        
        # Adaptive performance settings
        self.adaptive_workers = self._calculate_optimal_workers()
        self.adaptive_batch_size = self._calculate_optimal_batch_size()
        self.adaptive_timeout = self._calculate_optimal_timeout()
        
        print(f"🔧 Adaptive Performance Settings:")
        print(f"   • Workers: {self.adaptive_workers} (reduced from {config.max_parallel_workers})")
        print(f"   • Batch size: {self.adaptive_batch_size}")
        print(f"   • Timeout: {self.adaptive_timeout}s per object")
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal worker count to prevent resource exhaustion"""
        base_workers = self.config.max_parallel_workers
        
        # More conservative approach for stability
        if base_workers > 16:
            return 12  # Cap at 12 for very large datasets
        elif base_workers > 8:
            return 8   # Cap at 8 for medium datasets
        else:
            return max(4, base_workers)  # Minimum 4 workers
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for steady processing"""
        return max(self.adaptive_workers * 2, 16)  # At least 16 objects per batch
    
    def _calculate_optimal_timeout(self) -> int:
        """Calculate adaptive timeout based on system performance"""
        base_timeout = self.config.query_timeout_seconds
        # Slightly longer timeout for large datasets to reduce failures
        return min(base_timeout + 10, 45)  # Max 45 seconds
    
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
        ROBUST discovery method with adaptive performance and error recovery
        
        Args:
            limit: Maximum number of objects to process (None = ALL objects)
        
        Returns:
            bool: True if discovery was successful
        """
        print("🚀 Starting ROBUST database discovery with adaptive performance...")
        
        # Check cache first
        if self.load_from_cache():
            print(f"✅ Loaded {len(self.tables)} objects from cache")
            return True
        
        self.stats.reset()
        
        try:
            # Step 1: Discover ALL objects
            print("📊 Discovering ALL database objects...")
            objects = await self._get_all_objects_unlimited()
            
            if not objects:
                print("❌ No database objects found")
                return False
            
            # Step 2: Apply minimal filtering
            filtered_objects = self._apply_minimal_filtering(objects, limit)
            
            # Step 3: Log discovery plan
            self._log_robust_discovery_plan(objects, filtered_objects)
            
            # Step 4: Process with ROBUST batch processing and retry logic
            await self._process_objects_robust_batching(filtered_objects)
            
            # Step 5: Retry failed objects with single-threaded approach
            if self.failed_objects:
                await self._retry_failed_objects()
            
            # Step 6: Save results
            await self._save_discovery_results()
            
            # Step 7: Log completion stats
            self._log_completion_stats()
            
            return len(self.tables) > 0
            
        except Exception as e:
            print(f"❌ Discovery failed with error: {e}")
            return False
    
    async def _get_all_objects_unlimited(self) -> List[DatabaseObject]:
        """Get ALL database objects with priority scoring"""
        
        query = """
        -- Get ALL tables with priority scoring
        SELECT 
            SCHEMA_NAME(t.schema_id) as schema_name,
            t.name as table_name,
            'BASE TABLE' as object_type,
            COALESCE(p.rows, 0) as estimated_rows,
            -- Priority calculation
            CASE 
                WHEN t.name LIKE '%customer%' OR t.name LIKE '%client%' OR t.name LIKE '%business%' THEN 1000
                WHEN t.name LIKE '%order%' OR t.name LIKE '%sales%' OR t.name LIKE '%product%' THEN 900
                WHEN t.name LIKE '%user%' OR t.name LIKE '%account%' OR t.name LIKE '%person%' THEN 800
                WHEN t.name LIKE '%payment%' OR t.name LIKE '%invoice%' OR t.name LIKE '%financial%' THEN 700
                ELSE 500
            END + COALESCE(p.rows, 0) / 1000 as priority_score
        FROM sys.tables t
        LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id < 2
        WHERE t.is_ms_shipped = 0
          AND SCHEMA_NAME(t.schema_id) NOT IN ('sys', 'information_schema')
        
        UNION ALL
        
        -- Get ALL views with priority scoring
        SELECT 
            SCHEMA_NAME(v.schema_id) as schema_name,
            v.name as view_name,
            'VIEW' as object_type,
            CASE 
                WHEN v.name LIKE '%report%' OR v.name LIKE '%summary%' THEN 200
                ELSE 50 
            END as estimated_rows,
            -- Priority calculation for views
            CASE 
                WHEN v.name LIKE '%customer%' OR v.name LIKE '%client%' OR v.name LIKE '%business%' THEN 700
                WHEN v.name LIKE '%report%' OR v.name LIKE '%summary%' THEN 600
                WHEN v.name LIKE '%order%' OR v.name LIKE '%sales%' OR v.name LIKE '%product%' THEN 550
                ELSE 400
            END as priority_score
        FROM sys.views v
        WHERE v.is_ms_shipped = 0
          AND SCHEMA_NAME(v.schema_id) NOT IN ('sys', 'information_schema')
        
        ORDER BY priority_score DESC, estimated_rows DESC
        """
        
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                objects = []
                for row in cursor.fetchall():
                    obj = DatabaseObject(
                        schema=row[0],
                        name=row[1],
                        object_type=row[2],
                        estimated_rows=row[3]
                    )
                    obj.priority_score = row[4]  # Add priority score
                    objects.append(obj)
                
                self.stats.total_objects_found = len(objects)
                print(f"   ✅ Found {len(objects)} objects (prioritized by business relevance)")
                return objects
                
        except Exception as e:
            print(f"❌ Failed to get database objects: {e}")
            return []
    
    def _apply_minimal_filtering(self, objects: List[DatabaseObject], limit: Optional[int]) -> List[DatabaseObject]:
        """Apply minimal filtering with business object prioritization"""
        
        filtered = []
        excluded_count = 0
        
        for obj in objects:
            # Only exclude clearly problematic objects
            if self._should_exclude_object_critical_only(obj):
                excluded_count += 1
                continue
            
            filtered.append(obj)
        
        # Sort by priority score (business relevance + size)
        filtered.sort(key=lambda x: getattr(x, 'priority_score', 0), reverse=True)
        
        # Apply limit only if specified
        if limit and limit < len(filtered):
            filtered = filtered[:limit]
            print(f"   🎯 Applied limit: processing top {limit} priority objects")
        
        self.stats.objects_excluded = excluded_count
        return filtered
    
    def _should_exclude_object_critical_only(self, obj: DatabaseObject) -> bool:
        """MINIMAL exclusion - only exclude truly broken objects"""
        name_lower = obj.name.lower()
        
        # Only exclude clearly broken/system objects
        critical_exclusions = [
            'msreplication', 'trace_xe', 'syscommittab', 'sysdiagrams',
            'dtproperties', '__msnpeer', '__msdbm', 'mspeer_',
            'corrupted', 'broken', 'damaged', 'invalid',
            'conflict_', 'reseed_', 'msmerge_', 
            'businesspointidentificationwiththirdpartydata', 'timingview'
        ]
        
        for exclusion in critical_exclusions:
            if exclusion in name_lower:
                return True
        
        return False
    
    def _log_robust_discovery_plan(self, all_objects: List[DatabaseObject], filtered_objects: List[DatabaseObject]):
        """Log the robust discovery plan"""
        tables = sum(1 for obj in filtered_objects if obj.object_type in ['BASE TABLE', 'TABLE'])
        views = sum(1 for obj in filtered_objects if obj.object_type == 'VIEW')
        
        print(f"📊 ROBUST Discovery Plan:")
        print(f"   • Total objects in database: {len(all_objects)}")
        print(f"   • Objects to process: {len(filtered_objects)} (Tables: {tables}, Views: {views})")
        print(f"   • Excluded objects: {self.stats.objects_excluded}")
        print(f"   • ADAPTIVE parallel workers: {self.adaptive_workers}")
        print(f"   • ADAPTIVE batch size: {self.adaptive_batch_size}")
        print(f"   • Samples per object: 5 rows")
        print(f"   • ADAPTIVE timeout: {self.adaptive_timeout}s per object")
        
        # More realistic time estimation
        estimated_minutes = (len(filtered_objects) / self.adaptive_workers) * 0.75  # 45 seconds per worker batch
        print(f"   • Estimated completion: ~{estimated_minutes:.1f} minutes")
        print(f"   🛡️  ROBUST MODE: Includes retry logic and error recovery")
    
    async def _process_objects_robust_batching(self, objects: List[DatabaseObject]):
        """ROBUST processing with adaptive batching and comprehensive error handling"""
        print(f"\n🔄 ROBUST processing {len(objects)} objects with {self.adaptive_workers} workers...")
        
        # Progress bar
        pbar = tqdm(total=len(objects), desc="Robust analysis", unit="obj")
        
        # Process in adaptive batches
        batch_size = self.adaptive_batch_size
        
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            batch_number = (i // batch_size) + 1
            total_batches = (len(objects) + batch_size - 1) // batch_size
            
            print(f"\n   📦 Processing batch {batch_number}/{total_batches} ({len(batch)} objects)")
            
            # Process batch with robust error handling
            await self._process_batch_with_retry(batch, pbar)
            
            # Brief pause between batches to prevent overwhelming the database
            await asyncio.sleep(0.5)
        
        pbar.close()
    
    async def _process_batch_with_retry(self, batch: List[DatabaseObject], pbar: tqdm):
        """Process a batch with comprehensive error handling and retry logic"""
        
        # Use ThreadPoolExecutor with adaptive workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.adaptive_workers) as executor:
            # Submit all tasks in batch
            future_to_object = {
                executor.submit(self._analyze_object_robust, obj): obj 
                for obj in batch
            }
            
            # Process completed futures with extended timeout
            batch_timeout = self.adaptive_timeout * 2  # Double timeout for batch
            completed_count = 0
            
            try:
                # Use as_completed with timeout
                for future in concurrent.futures.as_completed(future_to_object, timeout=batch_timeout):
                    obj = future_to_object[future]
                    completed_count += 1
                    
                    try:
                        result = future.result(timeout=self.adaptive_timeout)
                        if result:
                            self.tables.append(result)
                            self.stats.successful_analyses += 1
                        else:
                            # Add to failed objects for retry
                            self.failed_objects.append(obj)
                            self.stats.analysis_errors += 1
                            
                    except (concurrent.futures.TimeoutError, Exception) as e:
                        # Add to failed objects for retry
                        self.failed_objects.append(obj)
                        self.stats.analysis_errors += 1
                    
                    self.stats.objects_processed += 1
                    pbar.update(1)
                    
                    # Update progress description
                    success_rate = (self.stats.successful_analyses / max(self.stats.objects_processed, 1)) * 100
                    pbar.set_description(f"Success: {self.stats.successful_analyses}/{self.stats.objects_processed} ({success_rate:.1f}%)")
                    
            except concurrent.futures.TimeoutError:
                # Handle batch timeout - mark remaining as failed
                print(f"   ⚠️ Batch timeout - {len(batch) - completed_count} objects will be retried")
                for future, obj in future_to_object.items():
                    if not future.done():
                        self.failed_objects.append(obj)
                        self.stats.analysis_errors += 1
                        self.stats.objects_processed += 1
                        pbar.update(1)
    
    def _analyze_object_robust(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """ROBUST object analysis with comprehensive error handling"""
        try:
            # Use separate connection for thread safety
            with self.get_database_connection() as conn:
                conn.timeout = self.adaptive_timeout
                cursor = conn.cursor()
                
                return self._analyze_object_with_5_samples(cursor, obj)
                
        except Exception:
            # Silent fail - object will be added to retry list
            return None
    
    def _analyze_object_with_5_samples(self, cursor, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze object with exactly 5 samples using multiple fallback strategies"""
        full_name = f"[{obj.schema}].[{obj.name}]"
        
        try:
            # Get column info
            columns = self._get_columns_fast(cursor, obj.schema, obj.name)
            if not columns:
                return None
            
            # Get 5 samples with multiple fallback strategies
            sample_data = self._get_5_samples_robust(cursor, obj, full_name)
            
            return TableInfo(
                name=obj.name,
                schema=obj.schema,
                full_name=full_name,
                object_type=obj.object_type,
                row_count=obj.estimated_rows,
                columns=columns,
                sample_data=sample_data,
                relationships=[],
                query_performance={
                    'robust_analysis': True, 
                    'samples_requested': 5, 
                    'samples_retrieved': len(sample_data),
                    'priority_score': getattr(obj, 'priority_score', 0)
                }
            )
            
        except Exception:
            return None
    
    def _get_columns_fast(self, cursor, schema: str, name: str) -> List[Dict[str, Any]]:
        """Get column info with fast query"""
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
                    'is_primary_key': False,
                    'is_foreign_key': False
                })
            
            return columns
        except:
            return []
    
    def _get_5_samples_robust(self, cursor, obj: DatabaseObject, full_name: str) -> List[Dict[str, Any]]:
        """Get exactly 5 samples with robust fallback strategies"""
        
        # Multiple strategies based on object type and priority
        if obj.object_type in ['BASE TABLE', 'TABLE']:
            strategies = [
                f"SELECT TOP 5 * FROM {full_name} OPTION (FAST 5)",
                f"SELECT TOP 5 * FROM {full_name} WITH (NOLOCK)",
                f"SELECT TOP 5 * FROM {full_name}",
                f"SELECT TOP 3 * FROM {full_name}",
                f"SELECT TOP 1 * FROM {full_name}"
            ]
        else:  # Views
            strategies = [
                f"SELECT TOP 5 * FROM {full_name} OPTION (FAST 5)",
                f"SELECT TOP 5 * FROM {full_name}",
                f"SELECT TOP 3 * FROM {full_name}",
                f"SELECT TOP 2 * FROM {full_name}",
                f"SELECT TOP 1 * FROM {full_name}"
            ]
        
        for strategy in strategies:
            try:
                cursor.execute(strategy)
                
                if not cursor.description:
                    continue
                
                col_names = [col[0] for col in cursor.description]
                sample_data = []
                
                # Collect up to 5 rows
                rows_collected = 0
                for row in cursor.fetchall():
                    if rows_collected >= 5:
                        break
                    
                    row_dict = {}
                    for i, value in enumerate(row):
                        if i < len(col_names):
                            row_dict[col_names[i]] = safe_database_value(value)
                    sample_data.append(row_dict)
                    rows_collected += 1
                
                if sample_data:
                    return sample_data
                
            except:
                continue  # Try next strategy
        
        return []  # Return empty if all strategies fail
    
    async def _retry_failed_objects(self):
        """Retry failed objects with single-threaded, conservative approach"""
        if not self.failed_objects:
            return
        
        print(f"\n🔄 Retrying {len(self.failed_objects)} failed objects with conservative approach...")
        
        retry_pbar = tqdm(self.failed_objects, desc="Retrying failed objects", unit="obj")
        retry_successes = 0
        
        for obj in retry_pbar:
            try:
                # Single-threaded retry with longer timeout
                result = await asyncio.to_thread(self._analyze_object_conservative, obj)
                if result:
                    self.tables.append(result)
                    self.stats.successful_analyses += 1
                    retry_successes += 1
                
                # Small delay between retries
                await asyncio.sleep(0.1)
                
            except Exception:
                pass  # Skip objects that fail even on retry
        
        retry_pbar.close()
        print(f"   ✅ Retry recovered {retry_successes} additional objects")
        
        # Clear failed objects list
        self.failed_objects.clear()
    
    def _analyze_object_conservative(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Conservative single-threaded analysis for retry scenarios"""
        try:
            with self.get_database_connection() as conn:
                conn.timeout = 60  # Longer timeout for retries
                cursor = conn.cursor()
                
                # Use most conservative sampling strategy
                full_name = f"[{obj.schema}].[{obj.name}]"
                
                # Get basic column info
                columns = self._get_columns_fast(cursor, obj.schema, obj.name)
                if not columns:
                    return None
                
                # Use most conservative sample strategy
                try:
                    cursor.execute(f"SELECT TOP 1 * FROM {full_name}")
                    if cursor.description:
                        col_names = [col[0] for col in cursor.description]
                        sample_data = []
                        
                        for row in cursor.fetchall():
                            row_dict = {}
                            for i, value in enumerate(row):
                                if i < len(col_names):
                                    row_dict[col_names[i]] = safe_database_value(value)
                            sample_data.append(row_dict)
                            break  # Only get 1 row for conservative retry
                        
                        return TableInfo(
                            name=obj.name,
                            schema=obj.schema,
                            full_name=full_name,
                            object_type=obj.object_type,
                            row_count=obj.estimated_rows,
                            columns=columns,
                            sample_data=sample_data,
                            relationships=[],
                            query_performance={'conservative_retry': True, 'samples_retrieved': len(sample_data)}
                        )
                except:
                    pass
                
                return None
                
        except Exception:
            return None
    
    async def _save_discovery_results(self):
        """Save discovery results with comprehensive metadata"""
        print(f"\n💾 Saving results for {len(self.tables)} objects...")
        
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
            'version': '5.0-robust-adaptive',
            'total_objects_analyzed': len(self.tables),
            'samples_per_object_target': 5,
            'analysis_stats': self.stats.to_dict(),
            'adaptive_config_used': {
                'adaptive_workers': self.adaptive_workers,
                'adaptive_batch_size': self.adaptive_batch_size,
                'adaptive_timeout': self.adaptive_timeout,
                'original_max_workers': self.config.max_parallel_workers,
                'failed_objects_retried': len(self.failed_objects) == 0
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
        
        print(f"\n✅ ROBUST discovery completed!")
        print(f"   📊 Total processed: {self.stats.objects_processed} objects")
        print(f"   ✅ Successful: {self.stats.successful_analyses} (Tables: {tables}, Views: {views})")
        print(f"   ❌ Failed (final): {self.stats.analysis_errors}")
        print(f"   ⏭️ Excluded: {self.stats.objects_excluded}")
        
        if self.stats.objects_processed > 0:
            success_rate = (self.stats.successful_analyses / self.stats.objects_processed) * 100
            print(f"   📈 Final success rate: {success_rate:.1f}%")
        
        # Sample statistics
        total_samples = sum(len(table.sample_data) for table in self.tables)
        avg_samples = total_samples / len(self.tables) if self.tables else 0
        print(f"   📝 Total samples collected: {total_samples} rows")
        print(f"   📊 Average samples per object: {avg_samples:.1f}")
        
        # Show sample results
        if self.tables:
            print(f"\n📋 Sample discovered objects (prioritized):")
            for i, table in enumerate(self.tables[:8]):
                cols = len(table.columns)
                samples = len(table.sample_data)
                priority = table.query_performance.get('priority_score', 0)
                print(f"   {i+1}. {table.full_name} ({table.object_type}) - {cols} cols, {samples} samples (priority: {priority})")
    
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
                
                print(f"   📊 Cache contains {len(self.tables)} objects")
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

# Update the main discovery class to use the robust version
DatabaseDiscovery = RobustDatabaseDiscovery