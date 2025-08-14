#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Database Discovery with Comprehensive Test Mode Output
Implements view definition mining, relationship discovery, executable view patterns, and detailed test reporting

Features:
- Test mode: analyze only 5 tables, 5 views, 5 stored procedures
- Comprehensive test output with actual discovered content
- Fixed view definition extraction using multiple methods
- Stored procedure discovery and analysis
- Enhanced output saving with full object details
- Debug mode for troubleshooting
- Complete test results with sample data and business patterns
"""

import pyodbc
import asyncio
import json
import time
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Progress bar (safe fallback if tqdm isn't available)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    class tqdm:  # minimal no-op fallback
        def __init__(self, iterable=None, total=None, desc="", unit="", dynamic_ncols=True,
                     mininterval=0.2, bar_format=None):
            self.iterable = iterable if iterable is not None else []
            self.total = total if total is not None else (len(self.iterable) if hasattr(self.iterable, "__len__") else None)
        def __iter__(self):
            for x in self.iterable:
                yield x
        def set_postfix_str(self, s):  # noqa: D401
            pass
        def set_description(self, s):
            pass
        def write(self, s):
            print(s)
        def update(self, n=1):
            pass
        def close(self):
            pass

from shared.config import Config
from shared.models import TableInfo, Relationship, DatabaseObject


class DatabaseDiscovery:
    """Enhanced database discovery with comprehensive test mode output"""

    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.view_definitions: Dict[str, str] = {}  # Legacy - kept for compatibility
        self.view_info: Dict[str, Dict[str, Any]] = {}  # Enhanced view information with samples
        self.stored_procedure_info: Dict[str, Dict[str, Any]] = {}  # Stored procedure information
        self.test_results: Dict[str, Any] = {}  # Test run results
        
        # Configuration
        self.per_object_timeout_sec = getattr(self.config, "discovery_timeout_sec", 300)
        self.debug_mode = getattr(self.config, "debug_mode", False)
        self.test_mode = False  # Will be set by caller
        
        # Test mode output file
        self.test_output_file = None

    def get_database_connection(self):
        """Get database connection with Greek text support"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)

        # UTF-8 encoding for international characters
        try:
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
        except Exception:
            # Some drivers may not support these; ignore gracefully
            pass

        return conn

    async def discover_database(self, limit: Optional[int] = None, test_mode: bool = False) -> bool:
        """Main discovery method with optional test mode"""
        
        self.test_mode = test_mode
        
        if test_mode:
            print("🧪 Starting TEST MODE discovery (5 tables, 5 views, 5 stored procedures)...")
            self.test_output_file = self.config.get_cache_path(f"test_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.test_results = {
                "start_time": datetime.now().isoformat(),
                "mode": "test",
                "limit_per_type": 5,
                "discovery_steps": [],
                "errors": [],
                "discovered_objects": {
                    "tables": [],
                    "views": [],
                    "stored_procedures": []
                },
                "sample_data_examples": {},
                "business_patterns": {},
                "query_templates": [],
                "relationships_detail": [],
                "summary": {}
            }
        else:
            print("🚀 Starting full database discovery...")

        # Check cache first (skip in test mode)
        if not test_mode and self._load_from_cache():
            print(f"✅ Loaded {len(self.tables)} objects from cache")
            return True

        try:
            # Step 1: Get all database objects
            print("📊 Discovering database objects...")
            objects = self._get_all_objects(test_mode=test_mode)
            
            if test_mode:
                self.test_results["discovery_steps"].append({
                    "step": "object_discovery",
                    "objects_found": len(objects),
                    "types": {obj.object_type: sum(1 for o in objects if o.object_type == obj.object_type) for obj in objects}
                })

            if not objects:
                error_msg = "No database objects found"
                print(f"❌ {error_msg}")
                if test_mode:
                    self.test_results["errors"].append(error_msg)
                    self._save_test_results()
                return False

            # Apply limit if specified
            if limit and limit < len(objects):
                objects = objects[:limit]
                print(f"   🎯 Limited to top {limit} objects")

            print(f"   ✅ Found {len(objects)} objects to analyze")

            # Step 2: Analyze objects with progress bar
            print("🔄 Analyzing objects and collecting samples...")
            await self._analyze_objects(objects)
            
            if test_mode:
                self._update_test_results_with_tables()
                self.test_results["discovery_steps"].append({
                    "step": "object_analysis",
                    "objects_analyzed": len(self.tables),
                    "sample_data_collected": sum(len(t.sample_data) for t in self.tables)
                })

            # Step 3: Enhanced view analysis with query execution
            print("👁️ Extracting view definitions and executing for patterns...")
            await self._extract_view_definitions_and_samples()
            
            if test_mode:
                self._update_test_results_with_views()
                successful_views = sum(1 for v in self.view_info.values() if v['execution_success'])
                self.test_results["discovery_steps"].append({
                    "step": "view_analysis",
                    "views_processed": len(self.view_info),
                    "views_executed_successfully": successful_views,
                    "definitions_extracted": sum(1 for v in self.view_info.values() if v['definition'] and 'Definition not available' not in v['definition'])
                })

            # Step 4: Stored procedure analysis (new)
            print("⚙️ Analyzing stored procedures...")
            await self._analyze_stored_procedures()
            
            if test_mode:
                self._update_test_results_with_procedures()
                self.test_results["discovery_steps"].append({
                    "step": "stored_procedure_analysis",
                    "procedures_processed": len(self.stored_procedure_info),
                    "definitions_extracted": sum(1 for sp in self.stored_procedure_info.values() if sp['definition'])
                })

            # Step 5: Discover relationships
            print("🔗 Discovering relationships...")
            self._discover_relationships()
            
            if test_mode:
                self._update_test_results_with_relationships()
                self.test_results["discovery_steps"].append({
                    "step": "relationship_discovery",
                    "relationships_found": len(self.relationships),
                    "relationship_types": {rel.relationship_type: sum(1 for r in self.relationships if r.relationship_type == rel.relationship_type) for rel in self.relationships}
                })

            # Step 6: Save results
            print("💾 Saving results...")
            if test_mode:
                self._finalize_test_results()
                self._save_test_results()
            else:
                self._save_to_cache()

            # Show summary
            self._show_discovery_summary()

            return True

        except Exception as e:
            error_msg = f"Discovery failed: {e}"
            print(f"❌ {error_msg}")
            if test_mode:
                self.test_results["errors"].append(error_msg)
                self._save_test_results()
            return False

    def _update_test_results_with_tables(self):
        """Add detailed table information to test results"""
        if not self.test_mode:
            return
            
        tables_detail = []
        for table in self.tables:
            if table.object_type in ['BASE TABLE', 'TABLE']:
                table_detail = {
                    "name": table.name,
                    "schema": table.schema,
                    "full_name": table.full_name,
                    "row_count": table.row_count,
                    "column_count": len(table.columns),
                    "columns": [
                        {
                            "name": col['name'],
                            "data_type": col['data_type'],
                            "nullable": col.get('nullable', True),
                            "is_primary_key": col.get('is_primary_key', False)
                        }
                        for col in table.columns[:10]  # First 10 columns
                    ],
                    "sample_data_count": len(table.sample_data),
                    "sample_data": table.sample_data[:2],  # First 2 sample rows
                    "foreign_keys": table.relationships[:5]  # First 5 FK relationships
                }
                tables_detail.append(table_detail)
                
        self.test_results["discovered_objects"]["tables"] = tables_detail

    def _update_test_results_with_views(self):
        """Add detailed view information to test results"""
        if not self.test_mode:
            return
            
        views_detail = []
        for view_name, view_info in self.view_info.items():
            view_detail = {
                "name": view_name,
                "full_name": view_info['full_name'],
                "definition_available": bool(view_info['definition'] and 'Definition not available' not in view_info['definition']),
                "definition_preview": (view_info['definition'][:200] + "...") if view_info['definition'] and len(view_info['definition']) > 200 else view_info['definition'],
                "execution_success": view_info['execution_success'],
                "error_message": view_info.get('error_message'),
                "columns_returned": view_info['columns_returned'],
                "sample_data": view_info['sample_data'][:2],  # First 2 sample rows
                "business_pattern": view_info['business_pattern'],
                "query_template": view_info['query_template']
            }
            views_detail.append(view_detail)
            
        self.test_results["discovered_objects"]["views"] = views_detail
        
        # Extract business patterns summary
        patterns = {}
        for view_info in self.view_info.values():
            if view_info['execution_success']:
                pattern = view_info['business_pattern']['pattern']
                patterns[pattern] = patterns.get(pattern, 0) + 1
        self.test_results["business_patterns"]["views"] = patterns

    def _update_test_results_with_procedures(self):
        """Add detailed stored procedure information to test results"""
        if not self.test_mode:
            return
            
        procedures_detail = []
        for sp_name, sp_info in self.stored_procedure_info.items():
            procedure_detail = {
                "name": sp_name,
                "full_name": sp_info['full_name'],
                "definition_available": bool(sp_info['definition'] and 'Definition extraction failed' not in sp_info['definition']),
                "definition_preview": (sp_info['definition'][:200] + "...") if sp_info['definition'] and len(sp_info['definition']) > 200 else sp_info['definition'],
                "parameter_count": len(sp_info['parameters']),
                "parameters": sp_info['parameters'][:5],  # First 5 parameters
                "business_pattern": sp_info['business_pattern'],
                "complexity": sp_info['complexity']
            }
            procedures_detail.append(procedure_detail)
            
        self.test_results["discovered_objects"]["stored_procedures"] = procedures_detail
        
        # Extract business patterns summary
        patterns = {}
        for sp_info in self.stored_procedure_info.values():
            pattern = sp_info['business_pattern']['pattern']
            patterns[pattern] = patterns.get(pattern, 0) + 1
        self.test_results["business_patterns"]["stored_procedures"] = patterns

    def _update_test_results_with_relationships(self):
        """Add detailed relationship information to test results"""
        if not self.test_mode:
            return
            
        relationships_detail = []
        for rel in self.relationships[:20]:  # First 20 relationships
            rel_detail = {
                "from_table": rel.from_table,
                "to_table": rel.to_table,
                "relationship_type": rel.relationship_type,
                "confidence": rel.confidence,
                "description": rel.description
            }
            relationships_detail.append(rel_detail)
            
        self.test_results["relationships_detail"] = relationships_detail

    def _get_all_objects(self, test_mode: bool = False) -> List[DatabaseObject]:
        """Get all database objects with priority scoring and name filtering."""
        
        # Enhanced query that includes stored procedures
        query = """
        -- Tables
        SELECT 
            SCHEMA_NAME(t.schema_id) as schema_name,
            t.name as object_name,
            'BASE TABLE' as object_type,
            COALESCE(
                (SELECT SUM(p.rows)
                 FROM sys.partitions p
                 WHERE p.object_id = t.object_id AND p.index_id IN (0,1)
                ), 0
            ) as estimated_rows,
            t.create_date,
            t.modify_date
        FROM sys.tables t
        WHERE t.is_ms_shipped = 0
          AND SCHEMA_NAME(t.schema_id) NOT IN ('sys','information_schema')
          AND LOWER(t.name) NOT LIKE '%bck%'
          AND LOWER(t.name) NOT LIKE '%backup%'
          AND LOWER(t.name) NOT LIKE '%dev%'

        UNION ALL

        -- Views
        SELECT 
            SCHEMA_NAME(v.schema_id) as schema_name,
            v.name as object_name,
            'VIEW' as object_type,
            0 as estimated_rows,
            v.create_date,
            v.modify_date
        FROM sys.views v
        WHERE v.is_ms_shipped = 0
          AND SCHEMA_NAME(v.schema_id) NOT IN ('sys','information_schema')
          AND LOWER(v.name) NOT LIKE '%bck%'
          AND LOWER(v.name) NOT LIKE '%backup%'
          AND LOWER(v.name) NOT LIKE '%dev%'
          AND LOWER(v.name) NOT LIKE '%timingview%'
          AND LOWER(v.name) NOT LIKE '%viewbatchactionanalysis%'
          AND LOWER(v.name) NOT LIKE '%viewfixofferfailure%'
          AND LOWER(v.name) NOT LIKE '%viewrenewalfailure%'
          AND LOWER(v.name) NOT LIKE '%viewsalesmarketactivecampaign%'
          AND LOWER(v.name) NOT LIKE '%viewsalesmarketItempotentialownerbasic%'
          AND LOWER(v.name) NOT LIKE '%viewsalesmarketitempotentialownerforcounter%'
          AND LOWER(v.name) NOT LIKE '%viewtargetgroupiteminfo%'
          AND LOWER(v.name) NOT LIKE '%viewtasklistfordialer%'
          AND LOWER(v.name) NOT LIKE '%vw_locacities%'
          AND LOWER(v.name) NOT LIKE '%vwagora%'
          AND LOWER(v.name) NOT LIKE '%workflowcontractview%'


        UNION ALL

        -- Stored Procedures
        SELECT 
            SCHEMA_NAME(p.schema_id) as schema_name,
            p.name as object_name,
            'STORED PROCEDURE' as object_type,
            0 as estimated_rows,
            p.create_date,
            p.modify_date
        FROM sys.procedures p
        WHERE p.is_ms_shipped = 0
          AND SCHEMA_NAME(p.schema_id) NOT IN ('sys','information_schema')
          AND LOWER(p.name) NOT LIKE '%bck%'
          AND LOWER(p.name) NOT LIKE '%backup%'
          AND LOWER(p.name) NOT LIKE '%dev%'
          AND p.name NOT LIKE 'sp_%'  -- Exclude system stored procedures

        ORDER BY estimated_rows DESC, object_name;
        """
        
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                all_objects = []
                for row in cursor.fetchall():
                    obj = DatabaseObject(
                        schema=row[0],
                        name=row[1],
                        object_type=row[2],
                        estimated_rows=row[3]
                    )
                    # Add extra metadata for test mode
                    if test_mode:
                        obj.create_date = row[4]
                        obj.modify_date = row[5]
                    all_objects.append(obj)
                    
        except Exception as e:
            print(f"❌ Failed to get database objects: {e}")
            return []

        # Safety: double-check in Python as well
        bad = ("bck", "backup", "dev")
        all_objects = [o for o in all_objects if not any(x in o.name.lower() for x in bad)]
        
        # In test mode, limit to 5 of each type
        if test_mode:
            limited_objects = []
            type_counts = {}
            
            for obj in all_objects:
                obj_type = obj.object_type
                if type_counts.get(obj_type, 0) < 5:
                    limited_objects.append(obj)
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                    
            print(f"   🧪 Test mode: Limited to {len(limited_objects)} objects:")
            for obj_type, count in type_counts.items():
                print(f"      • {obj_type}: {count}")
                
            return limited_objects
            
        return all_objects

    async def _analyze_objects(self, objects: List[DatabaseObject]):
        """Analyze objects and collect sample data"""
        pbar = tqdm(
            objects,
            desc="Analyzing objects",
            unit="obj",
            dynamic_ncols=True,
            mininterval=0.2,
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        )
        
        # Shorter timeout in test mode
        timeout_seconds = 60 if self.test_mode else int(self.per_object_timeout_sec)

        for obj in pbar:
            # show the *current* object on the same status line
            current_name = f"{obj.object_type}: {obj.schema}.{obj.name}"
            if len(current_name) > 60:
                current_name = "…" + current_name[-59:]
            pbar.set_postfix_str(current_name)

            try:
                table_info = await asyncio.wait_for(
                    asyncio.to_thread(self._analyze_single_object, obj),
                    timeout=timeout_seconds
                )
                if table_info:
                    self.tables.append(table_info)

                pbar.set_description(f"Analyzed {len(self.tables)}/{len(objects)}")

            except asyncio.TimeoutError:
                error_msg = f"Timeout analyzing {obj.schema}.{obj.name} (>{timeout_seconds}s)"
                pbar.write(f"   ⏰ {error_msg}")
                if self.test_mode:
                    self.test_results["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Failed to analyze {obj.schema}.{obj.name}: {e}"
                pbar.write(f"   ⚠️ {error_msg}")
                if self.test_mode:
                    self.test_results["errors"].append(error_msg)

        pbar.close()

    def _analyze_single_object(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze a single database object"""

        try:
            with self.get_database_connection() as conn:
                # Enforce DB-side statement timeouts
                per_obj = 60 if self.test_mode else int(self.per_object_timeout_sec)
                try:
                    conn.timeout = per_obj
                except Exception:
                    pass

                cursor = conn.cursor()
                try:
                    cursor.timeout = per_obj
                except Exception:
                    pass

                # Don't wait forever on locks
                try:
                    lock_ms = min(per_obj * 1000, 2147483647)
                    cursor.execute(f"SET LOCK_TIMEOUT {lock_ms};")
                except Exception:
                    pass

                # Get column information (skip for stored procedures)
                if obj.object_type != 'STORED PROCEDURE':
                    columns = self._get_columns(cursor, obj.schema, obj.name)
                    if not columns:
                        return None

                    # Get edge sample data (first 3 + last 3)
                    sample_data = self._get_sample_data(cursor, obj, columns)

                    # Get foreign keys
                    foreign_keys = self._get_foreign_keys(cursor, obj.schema, obj.name)
                else:
                    # For stored procedures, get parameter information instead
                    columns = self._get_procedure_parameters(cursor, obj.schema, obj.name)
                    sample_data = []  # No sample data for procedures
                    foreign_keys = []

                return TableInfo(
                    name=obj.name,
                    schema=obj.schema,
                    full_name=f"[{obj.schema}].[{obj.name}]",
                    object_type=obj.object_type,
                    row_count=obj.estimated_rows,
                    columns=columns,
                    sample_data=sample_data,
                    relationships=foreign_keys
                )

        except Exception:
            return None

    def _get_procedure_parameters(self, cursor, schema: str, name: str) -> List[Dict[str, Any]]:
        """Get stored procedure parameters"""
        
        query = """
        SELECT 
            p.name as parameter_name,
            TYPE_NAME(p.system_type_id) as data_type,
            p.max_length,
            p.is_output,
            p.has_default_value,
            p.default_value
        FROM sys.parameters p
        JOIN sys.procedures pr ON p.object_id = pr.object_id
        WHERE SCHEMA_NAME(pr.schema_id) = ? 
          AND pr.name = ?
          AND p.parameter_id > 0  -- Exclude return value
        ORDER BY p.parameter_id
        """
        
        try:
            cursor.execute(query, schema, name)
            parameters = []
            
            for row in cursor.fetchall():
                parameters.append({
                    'name': row[0] or '',
                    'data_type': row[1] or 'unknown',
                    'max_length': row[2],
                    'is_output': bool(row[3]),
                    'has_default': bool(row[4]),
                    'default_value': row[5],
                    'is_parameter': True
                })
            
            return parameters
        except Exception:
            return []

    def _get_columns(self, cursor, schema: str, name: str) -> List[Dict[str, Any]]:
        """Get column information"""

        query = """
        SELECT 
            c.COLUMN_NAME,
            c.DATA_TYPE,
            c.IS_NULLABLE,
            c.COLUMN_DEFAULT,
            CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as is_primary_key
        FROM INFORMATION_SCHEMA.COLUMNS c
        LEFT JOIN (
            SELECT ku.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku 
                ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
              AND tc.TABLE_SCHEMA = ?
              AND tc.TABLE_NAME = ?
        ) pk ON c.COLUMN_NAME = pk.COLUMN_NAME
        WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
        ORDER BY c.ORDINAL_POSITION
        """

        try:
            cursor.execute(query, schema, name, schema, name)
            columns = []

            for row in cursor.fetchall():
                columns.append({
                    'name': row[0],
                    'data_type': row[1],
                    'nullable': row[2] == 'YES',
                    'default': row[3],
                    'is_primary_key': bool(row[4])
                })

            return columns
        except Exception:
            return []

    def _pick_order_column(self, columns: List[Dict[str, Any]]) -> Optional[str]:
        """Choose a good column to ORDER BY for edge sampling."""
        if not columns:
            return None

        def is_datetime(col: Dict[str, Any]) -> bool:
            dt = (col.get('data_type') or '').lower()
            return dt in (
                'date', 'datetime', 'datetime2', 'smalldatetime',
                'datetimeoffset', 'time', 'timestamp'
            )

        # 1) datetime-ish with good names
        preferred_names = ('updated', 'modified', 'last', 'created', 'insert', 'date', 'time')
        named_dt = [c for c in columns if is_datetime(c) and any(p in c['name'].lower() for p in preferred_names)]
        if named_dt:
            return named_dt[0]['name']

        # 2) primary key
        pks = [c for c in columns if c.get('is_primary_key')]
        if pks:
            return pks[0]['name']

        # 3) first column
        return columns[0]['name']

    def _fetch_rows(self, cursor, sql: str, max_rows: int = 3) -> List[Dict[str, Any]]:
        try:
            cursor.execute(sql)
            if not cursor.description:
                return []
            cols = [c[0] for c in cursor.description]
            out = []
            for row in cursor.fetchmany(max_rows):
                d = {cols[i]: self._safe_value(val) for i, val in enumerate(row) if i < len(cols)}
                out.append(d)
            return out
        except Exception:
            return []

    def _get_sample_data(self, cursor, obj: DatabaseObject, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return up to 6 rows: first 3 and last 3, tagged with __edge to help downstream."""
        full_name = f"[{obj.schema}].[{obj.name}]"
        order_col = self._pick_order_column(columns)

        if order_col:
            first_sql = f"SELECT TOP 3 * FROM {full_name} ORDER BY [{order_col}] ASC"
            last_sql = f"SELECT TOP 3 * FROM {full_name} ORDER BY [{order_col}] DESC"
        else:
            first_sql = f"SELECT TOP 3 * FROM {full_name}"
            last_sql = f"SELECT TOP 3 * FROM {full_name}"

        first_rows = self._fetch_rows(cursor, first_sql, 3)
        last_rows = self._fetch_rows(cursor, last_sql, 3)

        # Fallbacks if needed
        if not first_rows:
            for sql in (
                f"SELECT TOP 3 * FROM {full_name}",
                f"SELECT TOP 1 * FROM {full_name}",
            ):
                first_rows = self._fetch_rows(cursor, sql, 3)
                if first_rows:
                    break

        if not last_rows and order_col:
            last_rows = self._fetch_rows(cursor, f"SELECT TOP 3 * FROM {full_name} ORDER BY [{order_col}] DESC", 3)

        # Tag rows to indicate which edge they came from
        tagged = []
        for r in first_rows:
            tr = dict(r)
            tr["__edge"] = "first"
            tagged.append(tr)
        for r in last_rows:
            tr = dict(r)
            tr["__edge"] = "last"
            tagged.append(tr)

        return tagged

    def _get_foreign_keys(self, cursor, schema: str, name: str) -> List[str]:
        """Get foreign key relationships"""

        query = """
        SELECT 
            OBJECT_SCHEMA_NAME(f.referenced_object_id) + '.' + OBJECT_NAME(f.referenced_object_id) as referenced_table,
            COL_NAME(f.parent_object_id, f.parent_column_id) as column_name,
            COL_NAME(f.referenced_object_id, f.referenced_column_id) as referenced_column
        FROM sys.foreign_key_columns f
        WHERE OBJECT_SCHEMA_NAME(f.parent_object_id) = ?
          AND OBJECT_NAME(f.parent_object_id) = ?
        """

        try:
            cursor.execute(query, schema, name)
            relationships = []

            for row in cursor.fetchall():
                relationships.append(f"{row[1]} -> {row[0]}.{row[2]}")

            return relationships
        except Exception:
            return []

    async def _extract_view_definitions_and_samples(self):
        """Extract view definitions AND execute them for sample data and pattern analysis"""
        
        view_tables = [t for t in self.tables if t.object_type == 'VIEW']
        if not view_tables:
            print("   📋 No views found")
            return

        print(f"   👁️ Extracting definitions and samples for {len(view_tables)} views...")
        
        # Enhanced view info structure
        self.view_info = {}
        
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                
                # Set connection-level timeout
                try:
                    cursor.timeout = 30
                except Exception:
                    pass

                successful_definitions = 0
                successful_executions = 0

                for view_table in view_tables:
                    view_name = f"{view_table.schema}.{view_table.name}"
                    full_view_name = f"[{view_table.schema}].[{view_table.name}]"
                    
                    # Try multiple methods to get view definition
                    definition = self._get_view_definition_multiple_methods(cursor, view_table.schema, view_table.name, full_view_name)
                    
                    if definition and 'Definition not available' not in definition and 'Definition extraction failed' not in definition:
                        successful_definitions += 1
                    
                    # Execute view for sample data
                    sample_data, execution_success, error_msg = self._execute_view_safely(
                        cursor, full_view_name
                    )
                    if execution_success:
                        successful_executions += 1
                    
                    # Analyze business pattern
                    business_pattern = self._analyze_view_business_pattern(definition, sample_data)
                    
                    # Store comprehensive view info
                    self.view_info[view_name] = {
                        'full_name': full_view_name,
                        'definition': definition,
                        'sample_data': sample_data,
                        'execution_success': execution_success,
                        'error_message': error_msg,
                        'columns_returned': list(sample_data[0].keys()) if sample_data else [],
                        'business_pattern': business_pattern,
                        'query_template': self._create_query_template(definition, sample_data, business_pattern)
                    }
                    
                    # Maintain backward compatibility
                    if definition:
                        self.view_definitions[view_name] = definition

            print(f"   ✅ Extracted {successful_definitions}/{len(view_tables)} view definitions")
            print(f"   🎯 Successfully executed {successful_executions}/{len(view_tables)} views for samples")

        except Exception as e:
            print(f"   ⚠️ View analysis failed: {e}")

    def _get_view_definition_multiple_methods(self, cursor, schema: str, name: str, full_name: str) -> Optional[str]:
        """Try multiple methods to extract view definition"""
        
        methods = [
            # Method 1: OBJECT_DEFINITION function (most reliable)
            {
                "name": "OBJECT_DEFINITION",
                "query": "SELECT OBJECT_DEFINITION(OBJECT_ID(?))",
                "params": [full_name]
            },
            
            # Method 2: sys.sql_modules join
            {
                "name": "sys.sql_modules",
                "query": """
                    SELECT m.definition
                    FROM sys.views v
                    JOIN sys.sql_modules m ON v.object_id = m.object_id
                    WHERE SCHEMA_NAME(v.schema_id) = ? AND v.name = ?
                """,
                "params": [schema, name]
            },
            
            # Method 3: Direct sys.sql_modules lookup
            {
                "name": "direct_sql_modules",
                "query": """
                    SELECT definition
                    FROM sys.sql_modules
                    WHERE object_id = OBJECT_ID(?)
                """,
                "params": [full_name]
            },
            
            # Method 4: Information schema (may be truncated but worth trying)
            {
                "name": "information_schema",
                "query": """
                    SELECT VIEW_DEFINITION
                    FROM INFORMATION_SCHEMA.VIEWS
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                """,
                "params": [schema, name]
            }
        ]
        
        for method in methods:
            try:
                cursor.execute(method["query"], method["params"])
                row = cursor.fetchone()
                if row and row[0]:
                    definition = row[0].strip()
                    if definition and len(definition) > 10:  # Ensure it's not just whitespace
                        if self.debug_mode:
                            print(f"      ✅ View {schema}.{name} definition extracted using {method['name']}")
                        return definition
            except Exception as e:
                if self.debug_mode:
                    print(f"      ⚠️ Method {method['name']} failed for {schema}.{name}: {e}")
                continue
        
        # If all methods fail, return a placeholder
        if self.debug_mode:
            print(f"      ❌ All definition extraction methods failed for {schema}.{name}")
        
        return f"-- Definition extraction failed for {full_name}\n-- View may be encrypted, corrupted, or have permission issues"

    async def _analyze_stored_procedures(self):
        """Analyze stored procedures and extract their definitions"""
        
        stored_procedure_tables = [t for t in self.tables if t.object_type == 'STORED PROCEDURE']
        if not stored_procedure_tables:
            print("   📋 No stored procedures found")
            return

        print(f"   ⚙️ Analyzing {len(stored_procedure_tables)} stored procedures...")
        
        self.stored_procedure_info = {}
        
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                
                successful_definitions = 0

                for sp_table in stored_procedure_tables:
                    sp_name = f"{sp_table.schema}.{sp_table.name}"
                    full_sp_name = f"[{sp_table.schema}].[{sp_table.name}]"
                    
                    # Get stored procedure definition
                    definition = self._get_stored_procedure_definition(cursor, sp_table.schema, sp_table.name, full_sp_name)
                    
                    if definition and 'Definition extraction failed' not in definition:
                        successful_definitions += 1
                    
                    # Analyze business pattern for stored procedure
                    business_pattern = self._analyze_procedure_business_pattern(definition, sp_table.columns)
                    
                    # Store comprehensive stored procedure info
                    self.stored_procedure_info[sp_name] = {
                        'full_name': full_sp_name,
                        'definition': definition,
                        'parameters': [col for col in sp_table.columns if col.get('is_parameter', False)],
                        'business_pattern': business_pattern,
                        'complexity': self._assess_procedure_complexity(definition)
                    }

            print(f"   ✅ Extracted {successful_definitions}/{len(stored_procedure_tables)} stored procedure definitions")

        except Exception as e:
            print(f"   ⚠️ Stored procedure analysis failed: {e}")

    def _get_stored_procedure_definition(self, cursor, schema: str, name: str, full_name: str) -> Optional[str]:
        """Extract stored procedure definition"""
        
        methods = [
            # Method 1: OBJECT_DEFINITION function
            {
                "query": "SELECT OBJECT_DEFINITION(OBJECT_ID(?))",
                "params": [full_name]
            },
            
            # Method 2: sys.sql_modules
            {
                "query": """
                    SELECT m.definition
                    FROM sys.procedures p
                    JOIN sys.sql_modules m ON p.object_id = m.object_id
                    WHERE SCHEMA_NAME(p.schema_id) = ? AND p.name = ?
                """,
                "params": [schema, name]
            }
        ]
        
        for method in methods:
            try:
                cursor.execute(method["query"], method["params"])
                row = cursor.fetchone()
                if row and row[0]:
                    definition = row[0].strip()
                    if definition and len(definition) > 10:
                        return definition
            except Exception:
                continue
        
        return f"-- Definition extraction failed for {full_name}"

    def _assess_procedure_complexity(self, definition: str) -> Dict[str, Any]:
        """Assess the complexity of a stored procedure"""
        
        if not definition or 'Definition extraction failed' in definition:
            return {"level": "unknown", "indicators": []}
        
        definition_lower = definition.lower()
        complexity_indicators = []
        score = 0
        
        # Count various complexity indicators
        if 'cursor' in definition_lower:
            complexity_indicators.append("uses_cursors")
            score += 3
            
        if 'transaction' in definition_lower or 'begin tran' in definition_lower:
            complexity_indicators.append("uses_transactions")
            score += 2
            
        if 'try' in definition_lower and 'catch' in definition_lower:
            complexity_indicators.append("error_handling")
            score += 1
            
        if 'while' in definition_lower or 'loop' in definition_lower:
            complexity_indicators.append("has_loops")
            score += 2
            
        # Count number of SQL statements (rough estimate)
        statement_count = len([word for word in ['select', 'insert', 'update', 'delete'] if word in definition_lower])
        if statement_count > 5:
            complexity_indicators.append("many_sql_statements")
            score += 1
            
        # Determine complexity level
        if score >= 6:
            level = "high"
        elif score >= 3:
            level = "medium"
        elif score >= 1:
            level = "low"
        else:
            level = "simple"
            
        return {
            "level": level,
            "score": score,
            "indicators": complexity_indicators,
            "statement_count": statement_count
        }

    def _analyze_procedure_business_pattern(self, definition: str, parameters: List[Dict]) -> Dict[str, Any]:
        """Analyze what business pattern this stored procedure represents"""
        
        if not definition:
            return {"pattern": "unknown", "confidence": 0.0}
        
        definition_lower = definition.lower()
        param_names = [p.get('name', '').lower() for p in parameters]
        
        patterns = {}
        confidence = 0.0
        
        # Data manipulation patterns
        if any(word in definition_lower for word in ['insert', 'update', 'delete']):
            patterns['data_manipulation'] = True
            confidence += 0.3
            
        # Reporting patterns
        if 'select' in definition_lower and ('group by' in definition_lower or 'sum(' in definition_lower):
            patterns['reporting'] = True
            confidence += 0.2
            
        # Business logic patterns
        if any(word in definition_lower for word in ['customer', 'payment', 'order', 'invoice']):
            patterns['business_logic'] = True
            confidence += 0.3
            
        # Maintenance patterns
        if any(word in definition_lower for word in ['cleanup', 'purge', 'maintenance', 'archive']):
            patterns['maintenance'] = True
            confidence += 0.2
            
        # Determine primary pattern
        if patterns.get('business_logic') and patterns.get('data_manipulation'):
            primary_pattern = "business_transaction"
        elif patterns.get('reporting'):
            primary_pattern = "reporting_procedure"
        elif patterns.get('data_manipulation'):
            primary_pattern = "data_management"
        elif patterns.get('maintenance'):
            primary_pattern = "system_maintenance"
        else:
            primary_pattern = "general_procedure"
            
        return {
            "pattern": primary_pattern,
            "confidence": min(confidence, 1.0),
            "characteristics": patterns,
            "parameter_count": len(parameters)
        }

    def _execute_view_safely(self, cursor, full_view_name: str) -> tuple:
        """Safely execute a view and return sample data"""
        
        # Multiple execution strategies with increasing safety
        strategies = [
            f"SELECT TOP 3 * FROM {full_view_name} ORDER BY (SELECT NULL)",
            f"SELECT TOP 3 * FROM {full_view_name}",
            f"SELECT TOP 1 * FROM {full_view_name}",
        ]
        
        error_msg = None
        
        for strategy in strategies:
            try:
                cursor.execute(strategy)
                
                if not cursor.description:
                    continue
                    
                columns = [col[0] for col in cursor.description]
                sample_data = []
                
                for row in cursor.fetchmany(3):
                    row_dict = {}
                    for i, value in enumerate(row):
                        if i < len(columns):
                            row_dict[columns[i]] = self._safe_value(value)
                    sample_data.append(row_dict)
                
                if sample_data:
                    return sample_data, True, None
                    
            except Exception as e:
                error_msg = str(e)
                continue
        
        return [], False, error_msg

    def _analyze_view_business_pattern(self, definition: str, sample_data: List[Dict]) -> Dict[str, Any]:
        """Analyze what business pattern this view represents"""
        
        if not definition:
            return {"pattern": "unknown", "confidence": 0.0}
        
        definition_lower = definition.lower()
        columns = list(sample_data[0].keys()) if sample_data else []
        column_names_lower = [c.lower() for c in columns]
        
        patterns = {}
        confidence = 0.0
        
        # Customer analysis pattern
        if any(word in definition_lower for word in ['customer', 'client', 'account', 'contact']):
            patterns['involves_customers'] = True
            confidence += 0.3
            
        # Payment/Financial pattern  
        if any(word in definition_lower for word in ['payment', 'invoice', 'revenue', 'amount', 'price', 'total']):
            patterns['involves_payments'] = True
            confidence += 0.3
            
        # Order/Sales pattern
        if any(word in definition_lower for word in ['order', 'sale', 'purchase', 'transaction']):
            patterns['involves_orders'] = True
            confidence += 0.2
            
        # Time-based analysis pattern
        if any(word in column_names_lower for word in ['date', 'month', 'year', 'period', 'time']):
            patterns['time_based'] = True
            confidence += 0.2
            
        # Aggregation pattern
        if any(word in definition_lower for word in ['sum(', 'count(', 'avg(', 'group by', 'having']):
            patterns['is_aggregated'] = True
            confidence += 0.2
            
        # Join pattern
        if 'join' in definition_lower:
            patterns['has_joins'] = True
            confidence += 0.1
        
        # Determine primary pattern
        if patterns.get('involves_customers') and patterns.get('involves_payments'):
            primary_pattern = "customer_payment_analysis"
            confidence += 0.3
        elif patterns.get('involves_customers') and patterns.get('involves_orders'):
            primary_pattern = "customer_order_analysis"
            confidence += 0.3
        elif patterns.get('involves_payments') and patterns.get('is_aggregated'):
            primary_pattern = "financial_reporting"
            confidence += 0.2
        elif patterns.get('involves_customers'):
            primary_pattern = "customer_analysis"
            confidence += 0.2
        elif patterns.get('involves_payments'):
            primary_pattern = "payment_analysis"
            confidence += 0.2
        elif patterns.get('involves_orders'):
            primary_pattern = "order_analysis"
            confidence += 0.2
        else:
            primary_pattern = "general_reporting"
        
        return {
            "pattern": primary_pattern,
            "confidence": min(confidence, 1.0),
            "characteristics": patterns,
            "sample_columns": columns[:8],
            "estimated_use_case": self._guess_use_case(primary_pattern, columns),
            "join_complexity": "complex" if patterns.get('has_joins') and patterns.get('is_aggregated') else "simple"
        }

    def _guess_use_case(self, pattern: str, columns: List[str]) -> str:
        """Guess what business question this view answers"""
        
        use_cases = {
            "customer_payment_analysis": "Analyze customer payment history, amounts, and payment behavior",
            "customer_order_analysis": "Analyze customer ordering patterns and purchase history", 
            "financial_reporting": "Generate financial reports, revenue analysis, and payment summaries",
            "customer_analysis": "Analyze customer information, demographics, and relationships",
            "payment_analysis": "Analyze payment transactions, amounts, and payment methods",
            "order_analysis": "Analyze order volumes, patterns, and sales performance",
            "general_reporting": "General business reporting and data analysis"
        }
        
        base_use_case = use_cases.get(pattern, "Data analysis and reporting")
        
        # Enhance based on column names
        column_hints = []
        column_names_lower = [c.lower() for c in columns]
        
        if any('total' in c or 'sum' in c or 'amount' in c for c in column_names_lower):
            column_hints.append("with totals/amounts")
        if any('count' in c or 'number' in c for c in column_names_lower):
            column_hints.append("with counts")
        if any('date' in c or 'month' in c or 'year' in c for c in column_names_lower):
            column_hints.append("over time periods")
        
        if column_hints:
            base_use_case += " " + ", ".join(column_hints)
            
        return base_use_case

    def _create_query_template(self, definition: str, sample_data: List[Dict], 
                             business_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create a reusable query template from the view"""
        
        if not definition or not sample_data:
            return {"available": False}
            
        template = {
            "available": True,
            "pattern_type": business_pattern.get('pattern', 'unknown'),
            "confidence": business_pattern.get('confidence', 0.0),
            "use_case": business_pattern.get('estimated_use_case', ''),
            "sample_question": self._generate_sample_question(business_pattern, sample_data),
            "expected_columns": list(sample_data[0].keys()) if sample_data else [],
            "join_pattern": "has_joins" if business_pattern.get('characteristics', {}).get('has_joins') else "single_table",
            "aggregation_pattern": "aggregated" if business_pattern.get('characteristics', {}).get('is_aggregated') else "detailed",
            "time_based": business_pattern.get('characteristics', {}).get('time_based', False)
        }
        
        return template

    def _generate_sample_question(self, business_pattern: Dict[str, Any], sample_data: List[Dict]) -> str:
        """Generate a sample question this view could answer"""
        
        pattern = business_pattern.get('pattern', 'unknown')
        columns = list(sample_data[0].keys()) if sample_data else []
        
        base_questions = {
            "customer_payment_analysis": "How much have customers paid?",
            "customer_order_analysis": "What have customers ordered?",
            "financial_reporting": "What is our revenue?",
            "customer_analysis": "Who are our customers?",
            "payment_analysis": "What payments have been made?",
            "order_analysis": "What orders do we have?",
            "general_reporting": "What does this data show?"
        }
        
        base_question = base_questions.get(pattern, "What information is available?")
        
        # Enhance based on actual columns
        if any('total' in c.lower() or 'amount' in c.lower() for c in columns):
            base_question = base_question.replace("How much", "What is the total amount")
        if any('count' in c.lower() for c in columns):
            base_question = base_question.replace("How much", "How many")
        if any('2024' in str(v) or '2025' in str(v) for row in sample_data for v in row.values()):
            base_question += " in 2024/2025?"
        elif not base_question.endswith('?'):
            base_question += "?"
            
        return base_question

    def _discover_relationships(self):
        """Discover relationships using multiple methods"""

        # Method 1: Foreign key relationships
        self._discover_foreign_key_relationships()

        # Method 2: Column name pattern relationships
        self._discover_pattern_relationships()

        # Method 3: View-based relationships (enhanced)
        self._discover_view_relationships()

        print(f"   ✅ Discovered {len(self.relationships)} relationships")

    def _discover_foreign_key_relationships(self):
        """Discover explicit foreign key relationships"""
        for table in self.tables:
            for fk_info in table.relationships:
                if '->' not in fk_info:
                    continue
                try:
                    _, rhs = fk_info.split('->', 1)
                    rhs = rhs.strip()
                    table_part, _ = rhs.rsplit('.', 1)
                    if '.' in table_part:
                        sch, tbl = table_part.split('.', 1)
                        referenced_table_full = f"[{sch}].[{tbl}]"
                    else:
                        referenced_table_full = table_part

                    self.relationships.append(Relationship(
                        from_table=table.full_name,
                        to_table=referenced_table_full,
                        relationship_type='foreign_key',
                        confidence=0.95,
                        description=f"Foreign key: {fk_info}"
                    ))
                except Exception:
                    continue

    def _discover_pattern_relationships(self):
        """Discover relationships based on column naming patterns"""

        table_lookup = {}
        for table in self.tables:
            table_lookup[table.name.lower()] = table.full_name

        for table in self.tables:
            column_names = [col['name'].lower() for col in table.columns]

            for col_name in column_names:
                if col_name.endswith('id') and col_name != 'id':
                    entity_name = col_name[:-2]

                    for table_name, full_name in table_lookup.items():
                        if (entity_name in table_name or table_name in entity_name) and full_name != table.full_name:
                            self.relationships.append(Relationship(
                                from_table=table.full_name,
                                to_table=full_name,
                                relationship_type='pattern_match',
                                confidence=0.7,
                                description=f"Column pattern: {col_name}"
                            ))
                            break

    def _discover_view_relationships(self):
        """Discover relationships from view dependencies"""
        view_tables = [t for t in self.tables if t.object_type == 'VIEW']
        if not view_tables:
            return

        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()

                for vt in view_tables:
                    view_name_2part = f"{vt.schema}.{vt.name}"
                    refs = set()

                    # Prefer dependency metadata
                    try:
                        cursor.execute(
                            "SELECT referenced_schema_name, referenced_entity_name "
                            "FROM sys.dm_sql_referenced_entities(?, 'OBJECT')",
                            view_name_2part
                        )
                        rows = cursor.fetchall() or []
                        for r in rows:
                            sch, ent = r[0], r[1]
                            if sch and ent:
                                refs.add(f"[{sch}].[{ent}]")
                    except Exception:
                        pass

                    # Fallback: heuristic scan of the stored definition
                    if not refs:
                        definition = self.view_definitions.get(view_name_2part, "") or ""
                        if definition:
                            words = definition.replace('[', '').replace(']', '').upper().split()
                            for i, w in enumerate(words):
                                if w in ('JOIN', 'FROM') and i + 1 < len(words):
                                    refs.add(words[i + 1])

                    # Map back to known objects
                    known_names = {t.full_name.lower(): t.full_name for t in self.tables}
                    for ref in refs:
                        key = ref.lower()
                        if key in known_names and known_names[key] != vt.full_name:
                            self.relationships.append(Relationship(
                                from_table=vt.full_name,
                                to_table=known_names[key],
                                relationship_type='view_dependency',
                                confidence=0.85,
                                description=f"Referenced by view {view_name_2part}"
                            ))
        except Exception as e:
            print(f"   ⚠️ View relationship discovery failed: {e}")

    def _safe_value(self, value) -> Any:
        """Convert database value to safe format"""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)[:200]

    def _show_discovery_summary(self):
        """Show discovery summary including view pattern analysis"""

        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        procedure_count = sum(1 for t in self.tables if t.object_type == 'STORED PROCEDURE')

        total_samples = sum(len(table.sample_data) for table in self.tables)
        successful_views = sum(1 for v in self.view_info.values() if v['execution_success'])
        
        print(f"\n📊 {'TEST MODE ' if self.test_mode else ''}DISCOVERY SUMMARY:")
        print(f"   📋 Objects analyzed: {len(self.tables)} (Tables: {table_count}, Views: {view_count}, Procedures: {procedure_count})")
        print(f"   📝 Sample rows collected: {total_samples}")
        print(f"   👁️ View definitions extracted: {len(self.view_definitions)}")
        print(f"   🎯 Views successfully executed: {successful_views}/{view_count}")
        print(f"   ⚙️ Stored procedures analyzed: {len(self.stored_procedure_info)}")
        print(f"   🔗 Relationships discovered: {len(self.relationships)}")

        # Show relationship types
        if self.relationships:
            rel_types = {}
            for rel in self.relationships:
                rel_types[rel.relationship_type] = rel_types.get(rel.relationship_type, 0) + 1

            print(f"   🔗 Relationship types:")
            for rel_type, count in rel_types.items():
                print(f"      • {rel_type}: {count}")

        # Show business patterns found in views
        if self.view_info:
            business_patterns = {}
            query_templates = 0
            for v in self.view_info.values():
                if v['execution_success']:
                    pattern = v['business_pattern']['pattern']
                    business_patterns[pattern] = business_patterns.get(pattern, 0) + 1
                    if v['query_template']['available']:
                        query_templates += 1
            
            if business_patterns:
                print(f"   📊 Business patterns in executable views:")
                for pattern, count in business_patterns.items():
                    print(f"      • {pattern}: {count} views")
                print(f"   🎯 Query templates created: {query_templates}")

        # Show stored procedure patterns
        if self.stored_procedure_info:
            sp_patterns = {}
            for sp in self.stored_procedure_info.values():
                pattern = sp['business_pattern']['pattern']
                sp_patterns[pattern] = sp_patterns.get(pattern, 0) + 1
            
            if sp_patterns:
                print(f"   ⚙️ Stored procedure patterns:")
                for pattern, count in sp_patterns.items():
                    print(f"      • {pattern}: {count} procedures")

    def _finalize_test_results(self):
        """Finalize test results with comprehensive summary information"""
        if not self.test_mode:
            return
            
        # Create comprehensive summary
        self.test_results["summary"] = {
            "total_objects_found": len(self.tables),
            "tables_analyzed": sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE']),
            "views_analyzed": sum(1 for t in self.tables if t.object_type == 'VIEW'),
            "procedures_analyzed": sum(1 for t in self.tables if t.object_type == 'STORED PROCEDURE'),
            "views_with_working_samples": sum(1 for v in self.view_info.values() if v['execution_success']),
            "views_with_definitions": sum(1 for v in self.view_info.values() if v['definition'] and 'extraction failed' not in v['definition'] and 'Definition not available' not in v['definition']),
            "procedures_with_definitions": sum(1 for sp in self.stored_procedure_info.values() if sp['definition'] and 'extraction failed' not in sp['definition']),
            "relationships_discovered": len(self.relationships),
            "sample_data_rows_collected": sum(len(t.sample_data) for t in self.tables),
            "business_patterns_identified": {
                "views": list(set(v['business_pattern']['pattern'] for v in self.view_info.values() if v['execution_success'])),
                "procedures": list(set(sp['business_pattern']['pattern'] for sp in self.stored_procedure_info.values()))
            },
            "query_templates_available": sum(1 for v in self.view_info.values() if v['query_template']['available']),
            "errors_encountered": len(self.test_results["errors"])
        }
        
        # Add sample successful view for reference
        successful_views = [
            (name, info) for name, info in self.view_info.items() 
            if info['execution_success'] and info['sample_data']
        ]
        
        if successful_views:
            sample_view_name, sample_view_info = successful_views[0]
            self.test_results["sample_successful_view"] = {
                "view_name": sample_view_name,
                "pattern": sample_view_info['business_pattern']['pattern'],
                "sample_data": sample_view_info['sample_data'][:1],  # Just first row
                "use_case": sample_view_info['business_pattern']['estimated_use_case'],
                "confidence": sample_view_info['business_pattern']['confidence']
            }
            
        # Add available query templates
        templates = []
        for view_name, view_info in self.view_info.items():
            if view_info['query_template']['available']:
                templates.append({
                    "view_name": view_name,
                    "pattern_type": view_info['query_template']['pattern_type'],
                    "sample_question": view_info['query_template']['sample_question'],
                    "use_case": view_info['query_template']['use_case']
                })
        self.test_results["query_templates"] = templates[:3]  # Top 3 templates

    def _save_test_results(self):
        """Save comprehensive test results to file"""
        if not self.test_mode or not self.test_output_file:
            return
            
        try:
            self.test_results["end_time"] = datetime.now().isoformat()
            self.test_results["duration_seconds"] = (
                datetime.fromisoformat(self.test_results["end_time"]) - 
                datetime.fromisoformat(self.test_results["start_time"])
            ).total_seconds()
            
            with open(self.test_output_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
                
            print(f"💾 Test results saved to: {self.test_output_file}")
            print(f"📄 File size: {self.test_output_file.stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            print(f"⚠️ Failed to save test results: {e}")

    def _save_to_cache(self):
        """Save discovery results to cache with enhanced view information"""

        cache_file = self.config.get_cache_path("database_structure.json")

        # Convert to dictionary format
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
                'relationships': table.relationships
            })

        relationships_data = []
        for rel in self.relationships:
            relationships_data.append({
                'from_table': rel.from_table,
                'to_table': rel.to_table,
                'relationship_type': rel.relationship_type,
                'confidence': rel.confidence,
                'description': rel.description
            })

        data = {
            'tables': tables_data,
            'relationships': relationships_data,
            'view_definitions': self.view_definitions,  # Legacy compatibility
            'view_info': self.view_info,  # Enhanced view information with patterns
            'stored_procedure_info': self.stored_procedure_info,  # Stored procedure information
            'created': datetime.now().isoformat(),
            'version': '2.5-comprehensive-test-output'
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")

    def _load_from_cache(self) -> bool:
        """Load from cache if available"""

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

            # Load tables
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
                        relationships=table_data.get('relationships', [])
                    )
                    self.tables.append(table)

            # Load relationships
            if 'relationships' in data:
                self.relationships = []
                for rel_data in data['relationships']:
                    self.relationships.append(Relationship(
                        from_table=rel_data['from_table'],
                        to_table=rel_data['to_table'],
                        relationship_type=rel_data['relationship_type'],
                        confidence=rel_data['confidence'],
                        description=rel_data.get('description', '')
                    ))

            # Load view definitions (legacy)
            self.view_definitions = data.get('view_definitions', {})
            
            # Load enhanced view info
            self.view_info = data.get('view_info', {})
            
            # Load stored procedure info
            self.stored_procedure_info = data.get('stored_procedure_info', {})

            return True

        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
            return False

    def load_from_cache(self) -> bool:
        """Public method to load from cache"""
        return self._load_from_cache()

    def get_tables(self) -> List[TableInfo]:
        """Get discovered tables"""
        return self.tables

    def get_relationships(self) -> List[Relationship]:
        """Get discovered relationships"""
        return self.relationships

    def get_view_definitions(self) -> Dict[str, str]:
        """Get view definitions (legacy)"""
        return self.view_definitions
    
    def get_view_info(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced view information with execution results and patterns"""
        return self.view_info
    
    def get_stored_procedure_info(self) -> Dict[str, Dict[str, Any]]:
        """Get stored procedure information"""
        return self.stored_procedure_info
    
    def get_executable_view_patterns(self) -> List[Dict[str, Any]]:
        """Get only the views that executed successfully as query patterns"""
        patterns = []
        for view_name, view_info in self.view_info.items():
            if view_info['execution_success'] and view_info['query_template']['available']:
                patterns.append({
                    'view_name': view_name,
                    'full_name': view_info['full_name'],
                    'definition': view_info['definition'],
                    'sample_data': view_info['sample_data'][:2],  # First 2 rows as example
                    'business_pattern': view_info['business_pattern'],
                    'query_template': view_info['query_template'],
                    'use_case': view_info['business_pattern']['estimated_use_case']
                })
        return patterns

    # Test mode convenience methods
    async def run_test_discovery(self) -> bool:
        """Run discovery in test mode"""
        return await self.discover_database(test_mode=True)
    
    def get_test_results_file(self) -> Optional[str]:
        """Get the path to the test results file"""
        return str(self.test_output_file) if self.test_output_file else None