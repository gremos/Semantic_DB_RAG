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
    """Database connection management - Single responsibility (SOLID)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_connection(self):
        """Get database connection with UTF-8 support"""
        conn = pyodbc.connect(self.config.get_database_connection_string())
        
        # UTF-8 support for international characters (README requirement)
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
    """Schema analysis with SQLGlot integration - Single responsibility (SOLID)"""
    
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
    
    def get_table_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get column information for table"""
        sql = """
        SELECT 
            c.COLUMN_NAME as name,
            c.DATA_TYPE as data_type,
            c.IS_NULLABLE as is_nullable,
            c.CHARACTER_MAXIMUM_LENGTH as max_length,
            c.NUMERIC_PRECISION as precision,
            c.NUMERIC_SCALE as scale
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
        ORDER BY c.ORDINAL_POSITION
        """
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, schema, table)
                
                columns = []
                for row in cursor:
                    columns.append({
                        'name': row.name,
                        'data_type': row.data_type,
                        'is_nullable': row.is_nullable == 'YES',
                        'max_length': row.max_length,
                        'precision': row.precision,
                        'scale': row.scale
                    })
                
                return columns
        except Exception:
            return []
    
    def get_table_relationships(self, schema: str, table: str) -> List[str]:
        """Get foreign key relationships"""
        sql = """
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
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, schema, table)
                
                relationships = []
                for row in cursor:
                    ref = f"{row.column_name} -> [{row.ref_schema}].[{row.ref_table}].{row.ref_column}"
                    relationships.append(ref)
                
                return relationships
        except Exception:
            return []

class SampleCollector:
    """Sample data collection following README policy - Single responsibility (SOLID)"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def collect_samples(self, table_info: TableInfo) -> List[Dict[str, Any]]:
        """Collect first 3 and last 3 rows (README requirement)"""
        try:
            # Get primary key for ordering
            pk_column = self._get_primary_key(table_info.schema, table_info.name)
            
            if pk_column:
                return self._collect_ordered_samples(table_info.full_name, pk_column)
            else:
                return self._collect_arbitrary_samples(table_info.full_name)
                
        except Exception as e:
            print(f"   ⚠️ Sample collection failed for {table_info.full_name}: {e}")
            return []
    
    def _get_primary_key(self, schema: str, table: str) -> Optional[str]:
        """Get primary key column for ordering"""
        sql = """
        SELECT c.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
          ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
         AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
         AND tc.TABLE_NAME = kcu.TABLE_NAME
        WHERE tc.TABLE_SCHEMA = ? 
          AND tc.TABLE_NAME = ?
          AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ORDER BY kcu.ORDINAL_POSITION
        """
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, schema, table)
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception:
            return None
    
    def _collect_ordered_samples(self, full_name: str, pk_column: str) -> List[Dict[str, Any]]:
        """Collect samples with deterministic ordering (README policy)"""
        samples = []
        
        # First 3 rows (ORDER BY PK ASC)
        sql_first = f"SELECT TOP (3) * FROM {full_name} ORDER BY [{pk_column}] ASC"
        first_rows = self.connector.execute_query(sql_first)
        samples.extend(first_rows)
        
        # Last 3 rows (ORDER BY PK DESC, then reverse)
        sql_last = f"""
        SELECT * FROM (
          SELECT TOP (3) * FROM {full_name} ORDER BY [{pk_column}] DESC
        ) t ORDER BY [{pk_column}] ASC
        """
        last_rows = self.connector.execute_query(sql_last)
        
        # Avoid duplicates
        first_pks = {row.get(pk_column) for row in first_rows}
        for row in last_rows:
            if row.get(pk_column) not in first_pks:
                samples.append(row)
        
        # Add metadata about ordering
        for sample in samples:
            sample['__ordering'] = {
                'column': pk_column,
                'deterministic': True
            }
        
        return samples[:6]  # Limit to 6 total
    
    def _collect_arbitrary_samples(self, full_name: str) -> List[Dict[str, Any]]:
        """Collect arbitrary samples when no PK available"""
        sql = f"SELECT TOP (6) * FROM {full_name}"
        samples = self.connector.execute_query(sql)
        
        # Add metadata about ordering
        for sample in samples:
            sample['__ordering'] = {
                'column': None,
                'deterministic': False
            }
        
        return samples

class ViewAnalyzer:
    """View definition analysis with SQLGlot parsing - Single responsibility (SOLID)"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def analyze_views(self) -> Dict[str, Dict[str, Any]]:
        """Analyze views and extract business logic (README requirement)"""
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
        ORDER BY s.name, v.name
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
                'execution_success': False
            }
            
            # Parse with SQLGlot
            try:
                parsed = sqlglot.parse_one(definition, dialect="tsql")
                if parsed:
                    # Extract referenced tables
                    tables = self._extract_tables(parsed)
                    view_info[full_name]['referenced_objects'] = tables
                    
                    # Extract joins
                    joins = self._extract_joins(parsed)
                    view_info[full_name]['parsed_joins'] = joins
                    
                    view_info[full_name]['execution_success'] = True
            except Exception as e:
                print(f"   ⚠️ Failed to parse view {full_name}: {e}")
        
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
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract referenced tables from parsed SQL"""
        tables = []
        
        for table in parsed.find_all(sqlglot.expressions.Table):
            if table.this:
                table_name = str(table.this)
                if table.db:
                    table_name = f"[{table.db}].[{table_name}]"
                tables.append(table_name)
        
        return list(set(tables))
    
    def _extract_joins(self, parsed) -> List[Dict[str, str]]:
        """Extract JOIN patterns from parsed SQL"""
        joins = []
        
        for join in parsed.find_all(sqlglot.expressions.Join):
            join_info = {
                'type': join.side or 'INNER',
                'condition': str(join.on) if join.on else ''
            }
            joins.append(join_info)
        
        return joins

class StoredProcedureAnalyzer:
    """Stored procedure analysis - Single responsibility (SOLID)"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def analyze_procedures(self) -> Dict[str, Dict[str, Any]]:
        """Analyze stored procedures for SELECT statements (README requirement)"""
        print("🔍 Analyzing stored procedures...")
        
        sql = """
        SELECT 
            s.name as schema_name,
            p.name as procedure_name,
            m.definition as procedure_definition
        FROM sys.procedures p
        INNER JOIN sys.schemas s ON p.schema_id = s.schema_id
        INNER JOIN sys.sql_modules m ON p.object_id = m.object_id
        WHERE s.name NOT IN ('sys', 'information_schema')
          AND p.name NOT LIKE 'sp_%'
          AND p.name NOT LIKE 'xp_%'
        ORDER BY s.name, p.name
        """
        
        results = self.connector.execute_query(sql)
        procedure_info = {}
        
        for row in results:
            schema = row['schema_name']
            name = row['procedure_name']
            definition = row['procedure_definition']
            full_name = f"[{schema}].[{name}]"
            
            # Analyze procedure text
            analysis = self._analyze_procedure_text(definition)
            
            procedure_info[full_name] = {
                'has_dynamic_sql': analysis['has_dynamic_sql'],
                'select_statements': analysis['select_statements'],
                'parsed_joins': analysis['parsed_joins'],
                'referenced_objects': analysis['referenced_objects']
            }
        
        print(f"   ✅ Analyzed {len(procedure_info)} procedures")
        return procedure_info
    
    def _analyze_procedure_text(self, definition: str) -> Dict[str, Any]:
        """Analyze procedure text for SELECT statements (static only)"""
        analysis = {
            'has_dynamic_sql': False,
            'select_statements': [],
            'parsed_joins': [],
            'referenced_objects': []
        }
        
        # Check for dynamic SQL
        definition_lower = definition.lower()
        if any(pattern in definition_lower for pattern in ['exec(', 'execute(', 'sp_executesql']):
            analysis['has_dynamic_sql'] = True
            return analysis  # Skip dynamic SQL as per README
        
        # Extract SELECT statements using regex (simple approach)
        import re
        select_pattern = r'SELECT\s+.*?(?=;|\bFROM\s+\(|\bUNION\b|\bEXCEPT\b|\bINTERSECT\b|$)'
        matches = re.findall(select_pattern, definition, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clean_select = match.strip()
            if len(clean_select) > 20:  # Filter out trivial selects
                analysis['select_statements'].append(clean_select)
                
                # Try SQLGlot parsing if available
                if HAS_SQLGLOT:
                    try:
                        parsed = sqlglot.parse_one(clean_select, dialect="tsql")
                        if parsed:
                            # Extract joins and tables
                            joins = self._extract_procedure_joins(parsed)
                            tables = self._extract_procedure_tables(parsed)
                            
                            analysis['parsed_joins'].extend(joins)
                            analysis['referenced_objects'].extend(tables)
                    except Exception:
                        continue
        
        return analysis
    
    def _extract_procedure_joins(self, parsed) -> List[Dict[str, str]]:
        """Extract joins from procedure SELECT"""
        joins = []
        
        for join in parsed.find_all(sqlglot.expressions.Join):
            join_info = {
                'left': str(join.this) if join.this else '',
                'right': str(join.expression) if join.expression else '',
                'type': join.side or 'INNER'
            }
            joins.append(join_info)
        
        return joins
    
    def _extract_procedure_tables(self, parsed) -> List[str]:
        """Extract tables from procedure SELECT"""
        tables = []
        
        for table in parsed.find_all(sqlglot.expressions.Table):
            if table.this:
                table_name = str(table.this)
                if table.db:
                    table_name = f"[{table.db}].[{table_name}]"
                tables.append(table_name)
        
        return list(set(tables))

class CacheManager:
    """Cache management for discovery results - Single responsibility (SOLID)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_discovery_cache(self, tables: List[TableInfo], view_info: Dict, sp_info: Dict):
        """Save discovery results to cache"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        data = {
            'tables': [self._table_to_dict(t) for t in tables],
            'view_info': view_info,
            'procedure_info': sp_info,
            'discovered': datetime.now().isoformat(),
            'version': '2.0-simple-improved'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Saved to cache: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def load_discovery_cache(self) -> Tuple[List[TableInfo], Dict, Dict]:
        """Load discovery results from cache if fresh"""
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
    """Main database discovery orchestrator - Clean interface (README compliant)"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components (SOLID principle)
        self.connector = DatabaseConnector(config)
        self.schema_analyzer = SchemaAnalyzer(self.connector)
        self.sample_collector = SampleCollector(self.connector)
        self.view_analyzer = ViewAnalyzer(self.connector)
        self.sp_analyzer = StoredProcedureAnalyzer(self.connector)
        self.cache_manager = CacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.view_info: Dict = {}
        self.sp_info: Dict = {}
        self.relationships: List[Relationship] = []
    
    async def discover_database(self) -> bool:
        """Main discovery method following README structure"""
        print("🔍 ADVANCED DATABASE DISCOVERY")
        print("Following README: Schema + samples + views/SPs with SQLGlot")
        print("=" * 70)
        
        # Check cache first
        cached_tables, cached_views, cached_sp = self.cache_manager.load_discovery_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.sp_info = cached_sp
            print(f"✅ Loaded from cache: {len(self.tables)} tables, {len(cached_views)} views")
            return True
        
        try:
            start_time = time.time()
            
            # Step 1: Discover database objects
            exclusion_patterns = self.config.get_exclusion_patterns()
            objects = self.schema_analyzer.discover_objects(exclusion_patterns)
            
            if not objects:
                print("❌ No database objects found")
                return False
            
            # Step 2: Analyze tables and views
            await self._analyze_objects(objects)
            
            # Step 3: Analyze views (if enabled)
            if self.config.is_view_analysis_enabled():
                self.view_info = self.view_analyzer.analyze_views()
            
            # Step 4: Analyze stored procedures (if enabled)
            if self.config.enable_sproc_analysis:
                self.sp_info = self.sp_analyzer.analyze_procedures()
            
            # Step 5: Extract relationships
            self._extract_relationships()
            
            # Step 6: Save to cache
            self.cache_manager.save_discovery_cache(self.tables, self.view_info, self.sp_info)
            
            # Show summary
            self._show_discovery_summary(time.time() - start_time)
            return True
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    async def _analyze_objects(self, objects: List[DatabaseObject]):
        """Analyze discovered objects in parallel"""
        print(f"📊 Analyzing {len(objects)} objects...")
        
        if self.config.use_fast_queries and len(objects) > 10:
            await self._parallel_analysis(objects)
        else:
            await self._sequential_analysis(objects)
    
    async def _parallel_analysis(self, objects: List[DatabaseObject]):
        """Parallel analysis for better performance"""
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            futures = []
            
            for obj in objects:
                future = executor.submit(self._analyze_single_object, obj)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    table_info = future.result()
                    if table_info:
                        self.tables.append(table_info)
                except Exception as e:
                    print(f"   ⚠️ Object analysis failed: {e}")
    
    async def _sequential_analysis(self, objects: List[DatabaseObject]):
        """Sequential analysis for smaller datasets"""
        for i, obj in enumerate(objects, 1):
            print(f"   📋 {i}/{len(objects)}: {obj.full_name}")
            
            table_info = self._analyze_single_object(obj)
            if table_info:
                self.tables.append(table_info)
    
    def _analyze_single_object(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze single database object"""
        try:
            # Get columns
            columns = self.schema_analyzer.get_table_columns(obj.schema, obj.name)
            if not columns:
                return None
            
            # Get relationships
            relationships = self.schema_analyzer.get_table_relationships(obj.schema, obj.name)
            
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
            
            # Collect samples (README policy: first 3 + last 3)
            if obj.estimated_rows > 0:
                samples = self.sample_collector.collect_samples(table_info)
                table_info.sample_data = samples
            
            return table_info
            
        except Exception as e:
            print(f"   ⚠️ Failed to analyze {obj.full_name}: {e}")
            return None
    
    def _extract_relationships(self):
        """Extract relationships from foreign keys and views"""
        self.relationships = []
        
        # Extract FK relationships
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
        
        # Extract view-based relationships
        for view_name, view_data in self.view_info.items():
            for join in view_data.get('parsed_joins', []):
                if 'condition' in join:
                    self.relationships.append(Relationship(
                        from_table=view_name,
                        to_table='',  # Would need more parsing
                        relationship_type='view_join',
                        confidence=0.8,
                        description=f"View join: {join['condition']}"
                    ))
    
    def _show_discovery_summary(self, elapsed_time: float):
        """Show discovery summary"""
        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        
        print(f"\n📊 DISCOVERY COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Tables: {table_count}")
        print(f"   👁️ Views: {view_count} (definitions: {len(self.view_info)})")
        print(f"   ⚙️ Stored Procedures: {len(self.sp_info)}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        print(f"   📋 Total objects: {len(self.tables)}")
        
        # Show SQLGlot status
        if HAS_SQLGLOT:
            parsed_views = sum(1 for v in self.view_info.values() if v.get('execution_success'))
            print(f"   ✅ SQLGlot parsing: {parsed_views}/{len(self.view_info)} views parsed")
        else:
            print(f"   ⚠️ SQLGlot not available - basic analysis only")
        
        # Show sample statistics
        with_samples = sum(1 for t in self.tables if t.sample_data)
        print(f"   📈 Sample data: {with_samples}/{len(self.tables)} objects")
    
    def load_from_cache(self) -> bool:
        """Load from cache (public interface)"""
        cached_tables, cached_views, cached_sp = self.cache_manager.load_discovery_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.sp_info = cached_sp
            return True
        return False
    
    # Public interface methods - Clean API (YAGNI principle)
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
        with_samples = sum(1 for t in self.tables if t.sample_data)
        
        return {
            'total_objects': len(self.tables),
            'tables': table_count,
            'views': view_count,
            'stored_procedures': len(self.sp_info),
            'relationships': len(self.relationships),
            'objects_with_samples': with_samples,
            'sqlglot_available': HAS_SQLGLOT,
            'view_definitions_captured': len(self.view_info)
        }