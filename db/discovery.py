#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Discovery Module
Handles database connection, schema discovery, and sample data collection
"""

import pyodbc
import asyncio
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

from shared.config import Config
from shared.models import TableInfo, DatabaseObject, AnalysisStats
from shared.utils import (
    safe_database_value, save_json_cache, load_json_cache, 
    should_exclude_table
)
from shared.models import table_info_to_dict, dict_to_table_info

class DatabaseDiscovery:
    """Enhanced Database Discovery with improved view handling and FAST queries"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.stats = AnalysisStats()
    
    def get_database_connection(self):
        """Get database connection with optimized settings"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # Set connection to handle Unicode properly
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
        return conn
    
    async def discover_database(self, limit: Optional[int] = None) -> bool:
        """Enhanced database discovery with improved view handling and FAST option"""
        # Check cache first
        cache_file = self.config.get_cache_path("database_structure.json")
        if self.load_from_cache():
            print(f"✅ Loaded {len(self.tables)} objects from cache")
            return True
        
        self.stats.reset()
        
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                
                # Get database objects with improved view estimation
                print("📊 Discovering database objects with enhanced view estimation...")
                all_objects = await self._get_database_objects_improved(cursor)
                
                # Filter excluded objects
                filtered_objects = self._filter_objects(all_objects)
                
                # Apply limit
                objects_to_process = filtered_objects[:limit] if limit else filtered_objects
                
                self._log_discovery_stats(all_objects, filtered_objects, objects_to_process, limit)
                
                # Process each object
                await self._process_objects(cursor, objects_to_process)
                
                # Save results
                await self._save_discovery_results(cache_file)
                
                self._log_completion_stats()
                
                return len(self.tables) > 0
                
        except Exception as e:
            print(f"❌ Enhanced discovery failed: {e}")
            return False
    
    async def _get_database_objects_improved(self, cursor) -> List[DatabaseObject]:
        """Get database objects with proper view row estimation"""
        query = """
        WITH ViewRowCounts AS (
            -- Better view estimation using system metadata
            SELECT 
                s.name as schema_name,
                v.name as view_name,
                CASE 
                    WHEN v.is_replicated = 0 AND v.is_ms_shipped = 0 
                    THEN 1000  -- Default estimate for user views
                    ELSE 0 
                END as estimated_rows
            FROM sys.views v
            INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
            WHERE v.is_ms_shipped = 0  -- Exclude system views
        )
        SELECT 
            t.TABLE_SCHEMA as schema_name,
            t.TABLE_NAME as table_name,
            t.TABLE_TYPE as object_type,
            CASE 
                WHEN t.TABLE_TYPE = 'BASE TABLE' THEN ISNULL(p.rows, 0)
                WHEN t.TABLE_TYPE = 'VIEW' THEN ISNULL(vrc.estimated_rows, 500)
                ELSE 0
            END as estimated_rows
        FROM INFORMATION_SCHEMA.TABLES t
        LEFT JOIN sys.tables st ON t.TABLE_NAME = st.name 
                                 AND t.TABLE_SCHEMA = SCHEMA_NAME(st.schema_id)
                                 AND t.TABLE_TYPE = 'BASE TABLE'
        LEFT JOIN sys.partitions p ON st.object_id = p.object_id AND p.index_id < 2
        LEFT JOIN ViewRowCounts vrc ON t.TABLE_SCHEMA = vrc.schema_name 
                                    AND t.TABLE_NAME = vrc.view_name 
                                    AND t.TABLE_TYPE = 'VIEW'
        WHERE t.TABLE_TYPE IN ('BASE TABLE', 'VIEW')
          AND t.TABLE_SCHEMA NOT IN ('sys', 'information_schema')
        ORDER BY 
            t.TABLE_TYPE DESC,
            estimated_rows DESC
        """
        
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
        return objects
    
    def _filter_objects(self, all_objects: List[DatabaseObject]) -> List[DatabaseObject]:
        """Filter out excluded objects"""
        filtered_objects = []
        excluded_count = 0
        backup_excluded = 0
        
        for obj in all_objects:
            if should_exclude_table(obj.name, obj.schema):
                excluded_count += 1
                if any(pattern in obj.name.lower() for pattern in ['bckp', 'bck', 'backup']):
                    backup_excluded += 1
            else:
                filtered_objects.append(obj)
        
        self.stats.objects_excluded = excluded_count
        self.stats.backup_tables_excluded = backup_excluded
        return filtered_objects
    
    def _log_discovery_stats(self, all_objects: List[DatabaseObject], 
                           filtered_objects: List[DatabaseObject], 
                           objects_to_process: List[DatabaseObject], 
                           limit: Optional[int]):
        """Log discovery statistics"""
        print(f"📊 Enhanced Discovery Results:")
        print(f"   Total objects found: {len(all_objects)}")
        print(f"   ✅ Processing: {len(filtered_objects)} objects")
        print(f"   ⏭️  Excluded: {self.stats.objects_excluded} objects ({self.stats.backup_tables_excluded} backup tables)")
        
        # Show view vs table breakdown
        views = [obj for obj in filtered_objects if obj.object_type == 'VIEW']
        tables = [obj for obj in filtered_objects if obj.object_type == 'BASE TABLE']
        print(f"   📋 Tables: {len(tables)} | Views: {len(views)} (with better estimation)")
        
        if limit and limit < len(filtered_objects):
            print(f"   📋 Limited to first {limit} objects for this run")
        
        print(f"\n🔍 Starting enhanced analysis of {len(objects_to_process)} objects...")
        self.stats.objects_processed = len(objects_to_process)
    
    async def _process_objects(self, cursor, objects_to_process: List[DatabaseObject]):
        """Process each database object with enhanced analysis"""
        progress_bar = tqdm(objects_to_process, desc="Analyzing objects")
        processed_count = 0
        
        for obj in progress_bar:
            processed_count += 1
            current_object = f"{obj.schema}.{obj.name}"
            progress_bar.set_description(f"Analyzing {current_object}")
            
            # Show progress for smaller datasets
            if len(objects_to_process) <= 50:
                estimated_display = f"{obj.estimated_rows:,}" if obj.estimated_rows > 0 else "estimated"
                print(f"   📋 Processing: {current_object} ({obj.object_type}, {estimated_display} rows)")
            elif processed_count % 25 == 0:
                print(f"   📊 Progress: {processed_count}/{len(objects_to_process)} - Current: {current_object}")
            
            try:
                table_info = await self._analyze_object_enhanced(cursor, obj)
                if table_info:
                    self.tables.append(table_info)
                    self.stats.successful_analyses += 1
                    
                    # Track performance improvements
                    if table_info.query_performance and table_info.query_performance.get('fast_query_used'):
                        self.stats.fast_query_successes += 1
                    
                    if not table_info.sample_data and obj.object_type == 'BASE TABLE':
                        self.stats.sample_data_errors += 1
                else:
                    self.stats.analysis_errors += 1
            except Exception as e:
                self.stats.analysis_errors += 1
                print(f"      ❌ Critical error analyzing {current_object}: {str(e)[:60]}...")
            
            # Brief pause to avoid overwhelming database
            await asyncio.sleep(0.02)
        
        progress_bar.set_description("Analysis complete")
        progress_bar.close()
    
    async def _analyze_object_enhanced(self, cursor, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze database object with FAST queries and improved view handling"""
        full_name = f"[{obj.schema}].[{obj.name}]"
        
        try:
            # Get column information
            columns = self._get_column_info(cursor, obj.schema, obj.name)
            
            # Get sample data with FAST query optimization
            sample_result = await self._get_sample_data_optimized(cursor, obj, full_name)
            
            # For views, try to get more accurate row count if sample was successful
            actual_row_count = obj.estimated_rows
            if obj.object_type == 'VIEW' and sample_result['data']:
                actual_row_count = await self._estimate_view_rows_better(cursor, full_name)
            
            return TableInfo(
                name=obj.name,
                schema=obj.schema,
                full_name=full_name,
                object_type=obj.object_type,
                row_count=actual_row_count,
                columns=columns,
                sample_data=sample_result['data'],
                relationships=[],
                query_performance=sample_result['performance']
            )
            
        except Exception as e:
            self._log_object_error(obj, e)
            return None
    
    def _get_column_info(self, cursor, schema: str, name: str) -> List[Dict[str, Any]]:
        """Get detailed column information"""
        query = """
        SELECT 
            c.COLUMN_NAME,
            c.DATA_TYPE,
            c.IS_NULLABLE,
            c.COLUMN_DEFAULT,
            c.CHARACTER_MAXIMUM_LENGTH,
            CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as IS_PRIMARY_KEY,
            CASE WHEN fk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as IS_FOREIGN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS c
        LEFT JOIN (
            SELECT ku.COLUMN_NAME, ku.TABLE_NAME, ku.TABLE_SCHEMA
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku 
                ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ) pk ON c.TABLE_SCHEMA = pk.TABLE_SCHEMA 
            AND c.TABLE_NAME = pk.TABLE_NAME 
            AND c.COLUMN_NAME = pk.COLUMN_NAME
        LEFT JOIN (
            SELECT ku.COLUMN_NAME, ku.TABLE_NAME, ku.TABLE_SCHEMA
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku 
                ON rc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
        ) fk ON c.TABLE_SCHEMA = fk.TABLE_SCHEMA 
            AND c.TABLE_NAME = fk.TABLE_NAME 
            AND c.COLUMN_NAME = fk.COLUMN_NAME
        WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
        ORDER BY c.ORDINAL_POSITION
        """
        
        cursor.execute(query, schema, name)
        columns = []
        
        for row in cursor.fetchall():
            columns.append({
                'name': row[0],
                'data_type': row[1],
                'nullable': row[2] == 'YES',
                'default': row[3],
                'max_length': row[4],
                'is_primary_key': bool(row[5]),
                'is_foreign_key': bool(row[6])
            })
        
        return columns
    
    async def _get_sample_data_optimized(self, cursor, obj: DatabaseObject, full_name: str) -> Dict[str, Any]:
        """Get sample data using OPTION (FAST n) optimization"""
        
        # Define strategies based on object type with FAST queries
        if obj.object_type == 'VIEW':
            strategies = [
                {
                    'query': f"SELECT * FROM {full_name} OPTION (FAST 3)",
                    'description': 'Fast 3 rows from view',
                    'is_fast': True
                },
                {
                    'query': f"SELECT TOP 3 * FROM {full_name} OPTION (FAST 1)",
                    'description': 'Fast 1 row, limit to 3',
                    'is_fast': True
                },
                {
                    'query': f"SELECT TOP 1 * FROM {full_name}",
                    'description': 'Simple top 1 fallback',
                    'is_fast': False
                }
            ]
        else:
            strategies = [
                {
                    'query': f"SELECT * FROM {full_name} OPTION (FAST 10)",
                    'description': 'Fast 10 rows from table',
                    'is_fast': True
                },
                {
                    'query': f"SELECT TOP 10 * FROM {full_name} OPTION (FAST 5)",
                    'description': 'Fast 5 rows, limit to 10',
                    'is_fast': True
                },
                {
                    'query': f"SELECT TOP 5 * FROM {full_name}",
                    'description': 'Simple top 5 fallback',
                    'is_fast': False
                }
            ]
        
        sample_data = []
        performance_info = {
            'execution_time': 0,
            'query_used': '',
            'fast_query_used': False,
            'strategy_used': '',
            'success': False
        }
        
        # Try strategies in order
        for attempt, strategy in enumerate(strategies):
            try:
                start_time = time.time()
                cursor.execute(strategy['query'])
                execution_time = time.time() - start_time
                
                if cursor.description:
                    col_names = [col[0] for col in cursor.description]
                    
                    for row in cursor.fetchall():
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(col_names):
                                row_dict[col_names[i]] = safe_database_value(value)
                        sample_data.append(row_dict)
                
                # Success! Record performance and return
                performance_info.update({
                    'execution_time': execution_time,
                    'query_used': strategy['query'],
                    'fast_query_used': strategy['is_fast'],
                    'strategy_used': strategy['description'],
                    'success': True
                })
                
                # Log successful fast queries
                if strategy['is_fast'] and execution_time < 1.0:
                    print(f"      ⚡ FAST success: {obj.name} in {execution_time:.3f}s")
                
                break
                
            except Exception as e:
                last_error = str(e)
                performance_info['execution_time'] = time.time() - start_time
                
                # Only show error on final attempt
                if attempt == len(strategies) - 1:
                    if obj.object_type == 'VIEW':
                        print(f"      ⚠️ View {obj.name}: {last_error} (may be empty/filtered)")
                    else:
                        print(f"      ❌ Table {obj.name}: All queries failed - {last_error}")
                
                continue
        
        return {
            'data': sample_data,
            'performance': performance_info
        }
    
    async def _estimate_view_rows_better(self, cursor, full_name: str) -> int:
        """Better view row estimation using FAST queries"""
        try:
            cursor.execute(f"SELECT COUNT(*) FROM (SELECT TOP 1000 * FROM {full_name}) as sample OPTION (FAST 100)")
            result = cursor.fetchone()
            count = result[0] if result else 0
            
            # If we got 1000, there might be more
            if count == 1000:
                return 1000  # Conservative estimate
            else:
                return count
        except:
            return 500  # Default estimate
    
    def _log_object_error(self, obj: DatabaseObject, error: Exception):
        """Log object analysis error"""
        error_code = getattr(error, 'args', [''])[0] if hasattr(error, 'args') and error.args else ''
        
        if '08004' in str(error_code):
            print(f"      ❌ {obj.schema}.{obj.name}: Connection timeout")
        elif '42S02' in str(error_code):
            print(f"      ❌ {obj.schema}.{obj.name}: Object not found")
        elif '42000' in str(error_code):
            print(f"      ❌ {obj.schema}.{obj.name}: Insufficient permissions")
        else:
            print(f"      ❌ {obj.schema}.{obj.name}: {str(error)[:60]}...")
    
    async def _save_discovery_results(self, cache_file: Path):
        """Save discovery results to cache"""
        print(f"\n💾 Saving enhanced results to cache...")
        
        data = {
            'tables': [table_info_to_dict(t) for t in self.tables],
            'created': datetime.now().isoformat(),
            'version': '2.0',
            'analysis_stats': self.stats.to_dict()
        }
        
        save_json_cache(cache_file, data, "discovery results")
    
    def _log_completion_stats(self):
        """Log completion statistics"""
        table_count = sum(1 for t in self.tables if t.object_type == 'BASE TABLE')
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        views_with_data = sum(1 for t in self.tables if t.object_type == 'VIEW' and t.sample_data)
        
        print(f"✅ Enhanced discovery completed!")
        print(f"   📋 Successfully analyzed: {len(self.tables)} objects")
        print(f"      • Tables: {table_count}")
        print(f"      • Views: {view_count} (NEW: {views_with_data} with sample data)")
        print(f"   ⚡ Fast query successes: {self.stats.fast_query_successes}")
        print(f"   ⚠️  Analysis errors: {self.stats.analysis_errors}")
        print(f"   📊 Sample data errors: {self.stats.sample_data_errors}")
        print(f"   ⏭️  Excluded objects: {self.stats.objects_excluded} (including {self.stats.backup_tables_excluded} backup tables)")
        
        if views_with_data > 0:
            print(f"\n🎉 IMPROVEMENT: {views_with_data} views now have sample data (previously 0)!")
    
    def load_from_cache(self) -> bool:
        """Load discovery results from cache"""
        cache_file = self.config.get_cache_path("database_structure.json")
        data = load_json_cache(cache_file, self.config.discovery_cache_hours, "discovery cache")
        
        if data and 'tables' in data:
            self.tables = []
            for table_data in data['tables']:
                table = dict_to_table_info(table_data)
                self.tables.append(table)
            return True
        
        return False
    
    def get_tables(self) -> List[TableInfo]:
        """Get discovered tables"""
        return self.tables
    
    def get_stats(self) -> AnalysisStats:
        """Get discovery statistics"""
        return self.stats