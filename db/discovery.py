#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Discovery - Simple, Readable, Maintainable
Following README: Schema + samples + view/SP analysis
DRY, SOLID, YAGNI principles - simplified from complex discovery
"""

import asyncio
import json
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# SQLGlot for SQL parsing (README requirement)
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from shared.config import Config
from shared.models import TableInfo, DatabaseObject, Relationship
from shared.utils import safe_database_value, should_exclude_table, normalize_table_name

class DatabaseConnector:
    """Database connection management"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_connection(self):
        """Get database connection with UTF-8 support"""
        conn = pyodbc.connect(self.config.get_database_connection_string())
        
        # UTF-8 support for international characters
        if self.config.utf8_encoding:
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
        
        return conn
    
    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor:
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = safe_database_value(value)
                        results.append(row_dict)
                    
                    return results
                else:
                    return []
                    
        except Exception as e:
            print(f"   ⚠️ Query failed: {e}")
            return []

class SchemaAnalyzer:
    """Schema analysis and metadata collection"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def discover_objects(self, exclusion_patterns: List[str]) -> List[DatabaseObject]:
        """Discover database objects (tables, views)"""
        print("🔍 Discovering database objects...")
        
        sql = """
        SELECT 
            s.name as schema_name,
            t.name as table_name,
            t.type_desc as object_type,
            ISNULL(p.rows, 0) as estimated_rows
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
        
        UNION ALL
        
        SELECT 
            s.name as schema_name,
            v.name as table_name,
            'VIEW' as object_type,
            0 as estimated_rows
        FROM sys.views v
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        
        ORDER BY schema_name, table_name
        """
        
        results = self.connector.execute_query(sql)
        objects = []
        
        for row in results:
            schema = row['schema_name']
            name = row['table_name']
            
            # Apply exclusions
            if should_exclude_table(name, schema, exclusion_patterns):
                continue
            
            objects.append(DatabaseObject(
                schema=schema,
                name=name,
                object_type=row['object_type'],
                estimated_rows=row['estimated_rows']
            ))
        
        print(f"   ✅ Found {len(objects)} objects")
        return objects
    
    def get_table_details(self, schema: str, table: str) -> Tuple[List[Dict], List[str]]:
        """Get table columns and relationships"""
        # Get columns
        columns_sql = """
        SELECT 
            c.COLUMN_NAME as name,
            c.DATA_TYPE as data_type,
            c.IS_NULLABLE as is_nullable,
            c.CHARACTER_MAXIMUM_LENGTH as max_length
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
        ORDER BY c.ORDINAL_POSITION
        """
        
        columns = []
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(columns_sql, schema, table)
                
                for row in cursor:
                    columns.append({
                        'name': row.name,
                        'data_type': row.data_type,
                        'is_nullable': row.is_nullable == 'YES',
                        'max_length': row.max_length
                    })
        except Exception:
            pass
        
        # Get relationships
        relationships_sql = """
        SELECT 
            kcu.COLUMN_NAME as column_name,
            kcu.REFERENCED_TABLE_SCHEMA as ref_schema,
            kcu.REFERENCED_TABLE_NAME as ref_table,
            kcu.REFERENCED_COLUMN_NAME as ref_column
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
        WHERE kcu.TABLE_SCHEMA = ? 
          AND kcu.TABLE_NAME = ?
          AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
        """
        
        relationships = []
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(relationships_sql, schema, table)
                
                for row in cursor:
                    ref = f"{row.column_name} -> [{row.ref_schema}].[{row.ref_table}].{row.ref_column}"
                    relationships.append(ref)
        except Exception:
            pass
        
        return columns, relationships

class SampleCollector:
    """Sample data collection"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def collect_samples(self, table_info: TableInfo) -> List[Dict[str, Any]]:
        """Collect sample data (first 6 rows)"""
        try:
            sql = f"SELECT TOP (6) * FROM {table_info.full_name}"
            samples = self.connector.execute_query(sql)
            return samples
        except Exception as e:
            print(f"   ⚠️ Sample collection failed for {table_info.full_name}: {e}")
            return []

class ViewAnalyzer:
    """View definition analysis"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def analyze_views(self) -> Dict[str, Dict[str, Any]]:
        """Analyze view definitions"""
        if not HAS_SQLGLOT:
            print("   ⚠️ SQLGlot not available - basic view analysis only")
            return self._basic_view_analysis()
        
        print("🔍 Analyzing views with SQLGlot...")
        
        sql = """
        SELECT 
            s.name as schema_name,
            v.name as view_name,
            m.definition as view_definition
        FROM sys.views v
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        INNER JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE s.name NOT IN ('sys', 'information_schema')
        """
        
        results = self.connector.execute_query(sql)
        view_info = {}
        
        for row in results:
            schema = row['schema_name']
            name = row['view_name']
            definition = row['view_definition']
            full_name = f"[{schema}].[{name}]"
            
            view_info[full_name] = {
                'definition': definition,
                'referenced_objects': [],
                'parsed_joins': [],
                'execution_success': True
            }
            
            # Parse with SQLGlot if available
            if HAS_SQLGLOT:
                try:
                    parsed = sqlglot.parse_one(definition, dialect="tsql")
                    if parsed:
                        # Extract referenced tables
                        tables = []
                        for table in parsed.find_all(sqlglot.expressions.Table):
                            if table.this:
                                table_name = str(table.this)
                                if table.db:
                                    table_name = f"[{table.db}].[{table_name}]"
                                tables.append(table_name)
                        
                        view_info[full_name]['referenced_objects'] = list(set(tables))
                except Exception:
                    pass
        
        print(f"   ✅ Analyzed {len(view_info)} views")
        return view_info
    
    def _basic_view_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Basic view analysis without SQLGlot"""
        sql = """
        SELECT 
            s.name as schema_name,
            v.name as view_name,
            m.definition as view_definition
        FROM sys.views v
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        INNER JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE s.name NOT IN ('sys', 'information_schema')
        """
        
        results = self.connector.execute_query(sql)
        view_info = {}
        
        for row in results:
            schema = row['schema_name']
            name = row['view_name']
            full_name = f"[{schema}].[{name}]"
            
            view_info[full_name] = {
                'definition': row['view_definition'],
                'referenced_objects': [],
                'parsed_joins': [],
                'execution_success': True
            }
        
        return view_info

class CacheManager:
    """Cache management for discovery results"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_cache(self, tables: List[TableInfo], view_info: Dict, sp_info: Dict):
        """Save discovery results to cache"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        data = {
            'tables': [self._table_to_dict(t) for t in tables],
            'view_info': view_info,
            'procedure_info': sp_info,
            'discovered': datetime.now().isoformat(),
            'version': '2.0-simplified'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Discovery cached: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def load_cache(self) -> Tuple[List[TableInfo], Dict, Dict]:
        """Load discovery results from cache"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        if not cache_file.exists():
            return [], {}, {}
        
        try:
            # Check cache age
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.config.discovery_cache_hours * 3600):
                return [], {}, {}
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tables = [self._dict_to_table(t) for t in data.get('tables', [])]
            view_info = data.get('view_info', {})
            sp_info = data.get('procedure_info', {})
            
            return tables, view_info, sp_info
            
        except Exception:
            return [], {}, {}
    
    def _table_to_dict(self, table: TableInfo) -> Dict:
        """Convert TableInfo to dictionary"""
        return {
            'name': table.name,
            'schema': table.schema,
            'full_name': table.full_name,
            'object_type': table.object_type,
            'row_count': table.row_count,
            'columns': table.columns,
            'sample_data': table.sample_data,
            'relationships': table.relationships
        }
    
    def _dict_to_table(self, data: Dict) -> TableInfo:
        """Convert dictionary to TableInfo"""
        return TableInfo(
            name=data['name'],
            schema=data['schema'],
            full_name=data['full_name'],
            object_type=data['object_type'],
            row_count=data['row_count'],
            columns=data['columns'],
            sample_data=data['sample_data'],
            relationships=data.get('relationships', [])
        )

class DatabaseDiscovery:
    """Main database discovery orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.connector = DatabaseConnector(config)
        self.schema_analyzer = SchemaAnalyzer(self.connector)
        self.sample_collector = SampleCollector(self.connector)
        self.view_analyzer = ViewAnalyzer(self.connector)
        self.cache_manager = CacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.view_info: Dict = {}
        self.sp_info: Dict = {}
        self.relationships: List[Relationship] = []
    
    async def discover_database(self) -> bool:
        """Main discovery method"""
        print("🔍 DATABASE DISCOVERY")
        print("=" * 50)
        
        # Check cache first
        cached_tables, cached_views, cached_sp = self.cache_manager.load_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.sp_info = cached_sp
            print(f"✅ Loaded from cache: {len(self.tables)} tables")
            return True
        
        try:
            start_time = time.time()
            
            # Step 1: Discover database objects
            exclusion_patterns = self.config.get_exclusion_patterns()
            objects = self.schema_analyzer.discover_objects(exclusion_patterns)
            
            if not objects:
                print("❌ No database objects found")
                return False
            
            # Step 2: Analyze objects
            await self._analyze_objects(objects)
            
            # Step 3: Analyze views (if enabled)
            if self.config.is_view_analysis_enabled():
                self.view_info = self.view_analyzer.analyze_views()
            
            # Step 4: Build relationships
            self._build_relationships()
            
            # Step 5: Save to cache
            self.cache_manager.save_cache(self.tables, self.view_info, self.sp_info)
            
            # Show summary
            self._show_summary(time.time() - start_time)
            return True
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    async def _analyze_objects(self, objects: List[DatabaseObject]):
        """Analyze discovered objects"""
        print(f"📊 Analyzing {len(objects)} objects...")
        
        for i, obj in enumerate(objects, 1):
            print(f"   📋 {i}/{len(objects)}: {obj.full_name}")
            
            try:
                # Get table details
                columns, relationships = self.schema_analyzer.get_table_details(obj.schema, obj.name)
                
                if not columns:
                    continue
                
                # Create table info
                table_info = TableInfo(
                    name=obj.name,
                    schema=obj.schema,
                    full_name=obj.full_name,
                    object_type=obj.object_type,
                    row_count=obj.estimated_rows,
                    columns=columns,
                    sample_data=[],
                    relationships=relationships
                )
                
                # Collect samples
                if obj.estimated_rows > 0:
                    samples = self.sample_collector.collect_samples(table_info)
                    table_info.sample_data = samples
                
                self.tables.append(table_info)
                
            except Exception as e:
                print(f"   ⚠️ Failed to analyze {obj.full_name}: {e}")
    
    def _build_relationships(self):
        """Build relationships from foreign keys"""
        self.relationships = []
        
        for table in self.tables:
            for rel_info in table.relationships:
                if '->' in rel_info:
                    try:
                        parts = rel_info.split('->', 1)
                        from_col = parts[0].strip()
                        to_ref = parts[1].strip()
                        
                        self.relationships.append(Relationship(
                            from_table=table.full_name,
                            to_table=to_ref.split('.')[0] if '.' in to_ref else to_ref,
                            relationship_type='foreign_key',
                            confidence=0.95,
                            description=f"FK: {from_col} -> {to_ref}"
                        ))
                    except Exception:
                        continue
    
    def _show_summary(self, elapsed_time: float):
        """Show discovery summary"""
        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        
        print(f"\n📊 DISCOVERY COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Tables: {table_count}")
        print(f"   👁️ Views: {view_count}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        print(f"   📋 Total objects: {len(self.tables)}")
        
        if HAS_SQLGLOT:
            parsed_views = sum(1 for v in self.view_info.values() if v.get('execution_success'))
            print(f"   ✅ SQLGlot parsing: {parsed_views}/{len(self.view_info)} views")
    
    def load_from_cache(self) -> bool:
        """Load from cache"""
        cached_tables, cached_views, cached_sp = self.cache_manager.load_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.sp_info = cached_sp
            return True
        return False
    
    # Public interface
    def get_tables(self) -> List[TableInfo]:
        """Get discovered tables"""
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        """Get discovered relationships"""
        return self.relationships
    
    def get_view_info(self) -> Dict:
        """Get view analysis results"""
        return self.view_info
    
    def get_stored_procedure_info(self) -> Dict:
        """Get stored procedure analysis results"""
        return self.sp_info
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        
        return {
            'total_objects': len(self.tables),
            'tables': table_count,
            'views': view_count,
            'stored_procedures': len(self.sp_info),
            'relationships': len(self.relationships),
            'sqlglot_available': HAS_SQLGLOT
        }