#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Discovery - Simple, Readable, Maintainable
Following README: Schema + samples + view/SP analysis with SQLGlot
DRY, SOLID, YAGNI principles
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
    """Simple database connection - Single responsibility"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_connection(self):
        """Get database connection with UTF-8 support"""
        conn = pyodbc.connect(self.config.get_database_connection_string())
        
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
                    return [
                        {columns[i]: safe_database_value(row[i]) for i in range(len(columns))}
                        for row in cursor
                    ]
                return []
        except Exception as e:
            print(f"   ⚠️ Query failed: {e}")
            return []

class SchemaAnalyzer:
    """Schema analysis - Single responsibility"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def discover_objects(self, exclusion_patterns: List[str]) -> List[DatabaseObject]:
        """Discover database objects (tables, views)"""
        print("🔍 Discovering database objects...")
        
        sql = """
        SELECT s.name as schema_name, t.name as table_name, t.type_desc as object_type,
               ISNULL(p.rows, 0) as estimated_rows
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
        UNION ALL
        SELECT s.name as schema_name, v.name as table_name, 'VIEW' as object_type, 0 as estimated_rows
        FROM sys.views v
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        ORDER BY schema_name, table_name
        """
        
        results = self.connector.execute_query(sql)
        objects = []
        
        for row in results:
            schema = row['schema_name']
            name = row['table_name']
            
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
    
    def get_table_info(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Get complete table information"""
        columns = self._get_columns(obj.schema, obj.name)
        if not columns:
            return None
        
        relationships = self._get_relationships(obj.schema, obj.name)
        samples = self._get_samples(obj.full_name, columns)
        
        return TableInfo(
            name=obj.name,
            schema=obj.schema,
            full_name=obj.full_name,
            object_type=obj.object_type,
            row_count=obj.estimated_rows,
            columns=columns,
            sample_data=samples,
            relationships=relationships
        )
    
    def _get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get column information"""
        sql = """
        SELECT COLUMN_NAME as name, DATA_TYPE as data_type, IS_NULLABLE as is_nullable,
               CHARACTER_MAXIMUM_LENGTH as max_length, NUMERIC_PRECISION as precision,
               NUMERIC_SCALE as scale
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, schema, table)
                return [
                    {
                        'name': row.name,
                        'data_type': row.data_type,
                        'is_nullable': row.is_nullable == 'YES',
                        'max_length': row.max_length,
                        'precision': row.precision,
                        'scale': row.scale
                    }
                    for row in cursor
                ]
        except Exception:
            return []
    
    def _get_relationships(self, schema: str, table: str) -> List[str]:
        """Get foreign key relationships"""
        sql = """
        SELECT COLUMN_NAME as column_name, REFERENCED_TABLE_SCHEMA as ref_schema,
               REFERENCED_TABLE_NAME as ref_table, REFERENCED_COLUMN_NAME as ref_column
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, schema, table)
                return [
                    f"{row.column_name} -> [{row.ref_schema}].[{row.ref_table}].{row.ref_column}"
                    for row in cursor
                ]
        except Exception:
            return []
    
    def _get_samples(self, full_name: str, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get sample data (first 3 + last 3 as per README)"""
        if not columns:
            return []
        
        # Try to get primary key for ordering
        pk_col = next((col['name'] for col in columns 
                      if 'id' in col['name'].lower() and col['name'].lower().endswith('id')), None)
        
        if pk_col:
            sql = f"""
            SELECT * FROM (
                SELECT TOP 3 *, 'first' as sample_type FROM {full_name} ORDER BY [{pk_col}] ASC
            ) first_rows
            UNION ALL
            SELECT * FROM (
                SELECT TOP 3 *, 'last' as sample_type FROM {full_name} ORDER BY [{pk_col}] DESC
            ) last_rows
            """
        else:
            sql = f"SELECT TOP 6 * FROM {full_name}"
        
        return self.connector.execute_query(sql)

class ViewAnalyzer:
    """View analysis with SQLGlot - Single responsibility"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def analyze_views(self) -> Dict[str, Dict[str, Any]]:
        """Analyze views with SQLGlot parsing"""
        print("🔍 Analyzing views with SQLGlot...")
        
        sql = """
        SELECT s.name as schema_name, v.name as view_name, m.definition as view_definition
        FROM sys.views v
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        INNER JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE s.name NOT IN ('sys', 'information_schema')
        ORDER BY s.name, v.name
        """
        
        results = self.connector.execute_query(sql)
        view_info = {}
        
        for row in results:
            schema = row['schema_name']
            name = row['view_name']
            definition = row['view_definition']
            full_name = f"[{schema}].[{name}]"
            
            analysis = self._analyze_view_definition(definition)
            view_info[full_name] = analysis
        
        print(f"   ✅ Analyzed {len(view_info)} views")
        return view_info
    
    def _analyze_view_definition(self, definition: str) -> Dict[str, Any]:
        """Analyze single view definition"""
        result = {
            'definition': definition,
            'referenced_objects': [],
            'parsed_joins': [],
            'execution_success': False
        }
        
        if not HAS_SQLGLOT:
            return result
        
        try:
            parsed = sqlglot.parse_one(definition, dialect="tsql")
            if parsed:
                result['referenced_objects'] = self._extract_tables(parsed)
                result['parsed_joins'] = self._extract_joins(parsed)
                result['execution_success'] = True
        except Exception:
            pass
        
        return result
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table references"""
        tables = []
        for table in parsed.find_all(sqlglot.expressions.Table):
            if table.this:
                table_name = str(table.this)
                if table.db:
                    table_name = f"[{table.db}].[{table_name}]"
                tables.append(table_name)
        return list(set(tables))
    
    def _extract_joins(self, parsed) -> List[Dict[str, str]]:
        """Extract JOIN patterns"""
        joins = []
        for join in parsed.find_all(sqlglot.expressions.Join):
            joins.append({
                'type': join.side or 'INNER',
                'condition': str(join.on) if join.on else ''
            })
        return joins

class CacheManager:
    """Simple cache management - Single responsibility"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_cache(self, tables: List[TableInfo], view_info: Dict, sp_info: Dict):
        """Save discovery results"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        data = {
            'tables': [self._table_to_dict(t) for t in tables],
            'view_info': view_info,
            'procedure_info': sp_info,
            'discovered': datetime.now().isoformat(),
            'version': '2.0-simple'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Saved to cache: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def load_cache(self) -> Tuple[List[TableInfo], Dict, Dict]:
        """Load from cache if fresh"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        if not cache_file.exists():
            return [], {}, {}
        
        try:
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
        """Convert TableInfo to dict"""
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
        """Convert dict to TableInfo"""
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
    """Main discovery orchestrator - Clean interface"""
    
    def __init__(self, config: Config):
        self.config = config
        self.connector = DatabaseConnector(config)
        self.schema_analyzer = SchemaAnalyzer(self.connector)
        self.view_analyzer = ViewAnalyzer(self.connector)
        self.cache_manager = CacheManager(config)
        
        self.tables: List[TableInfo] = []
        self.view_info: Dict = {}
        self.sp_info: Dict = {}
        self.relationships: List[Relationship] = []
    
    async def discover_database(self) -> bool:
        """Main discovery following README structure"""
        print("🔍 DATABASE DISCOVERY")
        print("Following README: Schema + samples + views/SPs with SQLGlot")
        print("=" * 60)
        
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
            
            # Discover objects
            exclusion_patterns = self.config.get_exclusion_patterns()
            objects = self.schema_analyzer.discover_objects(exclusion_patterns)
            
            if not objects:
                print("❌ No database objects found")
                return False
            
            # Analyze tables
            await self._analyze_tables(objects)
            
            # Analyze views
            if self.config.is_view_analysis_enabled():
                self.view_info = self.view_analyzer.analyze_views()
            
            # Extract relationships
            self._extract_relationships()
            
            # Save cache
            self.cache_manager.save_cache(self.tables, self.view_info, self.sp_info)
            
            # Show summary
            self._show_summary(time.time() - start_time)
            return True
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    async def _analyze_tables(self, objects: List[DatabaseObject]):
        """Analyze tables in parallel or sequential"""
        print(f"📊 Analyzing {len(objects)} objects...")
        
        if self.config.use_fast_queries and len(objects) > 10:
            await self._analyze_parallel(objects)
        else:
            await self._analyze_sequential(objects)
    
    async def _analyze_parallel(self, objects: List[DatabaseObject]):
        """Parallel analysis"""
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            futures = [executor.submit(self.schema_analyzer.get_table_info, obj) for obj in objects]
            
            for future in as_completed(futures):
                try:
                    table_info = future.result()
                    if table_info:
                        self.tables.append(table_info)
                except Exception as e:
                    print(f"   ⚠️ Table analysis failed: {e}")
    
    async def _analyze_sequential(self, objects: List[DatabaseObject]):
        """Sequential analysis"""
        for i, obj in enumerate(objects, 1):
            print(f"   📋 {i}/{len(objects)}: {obj.full_name}")
            
            table_info = self.schema_analyzer.get_table_info(obj)
            if table_info:
                self.tables.append(table_info)
    
    def _extract_relationships(self):
        """Extract relationships from foreign keys"""
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
        """Load from cache (public interface)"""
        cached_tables, cached_views, cached_sp = self.cache_manager.load_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.sp_info = cached_sp
            return True
        return False
    
    # Public interface - Clean API
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_view_info(self) -> Dict:
        return self.view_info
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        
        return {
            'total_objects': len(self.tables),
            'tables': table_count,
            'views': view_count,
            'relationships': len(self.relationships),
            'sqlglot_available': HAS_SQLGLOT
        }