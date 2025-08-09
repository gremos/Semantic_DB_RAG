#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Database Discovery - Simple and Maintainable
Implements view definition mining and relationship discovery

Improvements vs. the original:
- Skips objects whose names contain bck/backup/dev (case-insensitive)
- Captures "edge samples": first 3 and last 3 rows per object, tagged with __edge
- More robust view relationship mining via sys.dm_sql_referenced_entities
- Fixes FK parsing to retain [schema].[table] instead of only schema
- Adds hard per-object timeouts (Python + DB-side) to avoid lingering threads
- Safe tqdm fallback (won’t crash if tqdm/pip are unavailable)
"""

import pyodbc
import asyncio
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
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
    """Enhanced database discovery with view analysis and relationship discovery"""

    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.view_definitions: Dict[str, str] = {}
        # unified, configurable per-object timeout (seconds). Default: 300s (5 minutes)
        self.per_object_timeout_sec = getattr(self.config, "discovery_timeout_sec", 300)

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

    async def discover_database(self, limit: Optional[int] = None) -> bool:
        """Main discovery method with view analysis"""
        print("🚀 Starting enhanced database discovery...")

        # Check cache first
        if self._load_from_cache():
            print(f"✅ Loaded {len(self.tables)} objects from cache")
            return True

        try:
            # Step 1: Get all database objects
            print("📊 Discovering database objects...")
            objects = self._get_all_objects()

            if not objects:
                print("❌ No database objects found")
                return False

            # Apply limit if specified
            if limit and limit < len(objects):
                objects = objects[:limit]
                print(f"   🎯 Limited to top {limit} objects")

            print(f"   ✅ Found {len(objects)} objects to analyze")

            # Step 2: Analyze objects with progress bar
            print("🔄 Analyzing objects and collecting samples...")
            await self._analyze_objects(objects)

            # Step 3: Get view definitions
            print("👁️ Extracting view definitions...")
            await self._extract_view_definitions()

            # Step 4: Discover relationships
            print("🔗 Discovering relationships...")
            self._discover_relationships()

            # Step 5: Save results
            print("💾 Saving results...")
            self._save_to_cache()

            # Show summary
            self._show_discovery_summary()

            return True

        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False

    def _get_all_objects(self) -> List[DatabaseObject]:
        """Get all database objects with priority scoring and name filtering."""
        # Filter out bck|backup|dev (case-insensitive) directly in SQL
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
            ) as estimated_rows
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
            0 as estimated_rows
        FROM sys.views v
        WHERE v.is_ms_shipped = 0
          AND SCHEMA_NAME(v.schema_id) NOT IN ('sys','information_schema')
          AND LOWER(v.name) NOT LIKE '%bck%'
          AND LOWER(v.name) NOT LIKE '%backup%'
          AND LOWER(v.name) NOT LIKE '%dev%'

        ORDER BY estimated_rows DESC, object_name;
        """
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                objects = [
                    DatabaseObject(
                        schema=row[0],
                        name=row[1],
                        object_type=row[2],
                        estimated_rows=row[3]
                    )
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            print(f"❌ Failed to get database objects: {e}")
            return []

        # Safety: double-check in Python as well
        bad = ("bck", "backup", "dev")
        objects = [o for o in objects if not any(x in o.name.lower() for x in bad)]
        return objects

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
        timeout_seconds = int(self.per_object_timeout_sec)  # seconds

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
                pbar.write(f"   ⏰ Timeout analyzing {obj.schema}.{obj.name} (>{timeout_seconds//60}min) - skipping")
            except Exception as e:
                pbar.write(f"   ⚠️ Failed to analyze {obj.schema}.{obj.name}: {e}")

        pbar.close()

    def _analyze_single_object(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze a single database object"""

        try:
            with self.get_database_connection() as conn:
                # Enforce DB-side statement timeouts so worker thread won't linger
                per_obj = int(self.per_object_timeout_sec)
                try:
                    conn.timeout = per_obj  # default for operations on this connection
                except Exception:
                    pass

                cursor = conn.cursor()
                try:
                    cursor.timeout = per_obj  # some drivers honor cursor-level timeout
                except Exception:
                    pass

                # Don't wait forever on locks (milliseconds)
                try:
                    lock_ms = min(per_obj * 1000, 2147483647)
                    cursor.execute(f"SET LOCK_TIMEOUT {lock_ms};")
                except Exception:
                    pass

                # Get column information
                columns = self._get_columns(cursor, obj.schema, obj.name)
                if not columns:
                    return None

                # Get edge sample data (first 3 + last 3)
                sample_data = self._get_sample_data(cursor, obj, columns)

                # Get foreign keys
                foreign_keys = self._get_foreign_keys(cursor, obj.schema, obj.name)

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

    async def _extract_view_definitions(self):
        """Extract view definitions for relationship mining"""

        view_tables = [t for t in self.tables if t.object_type == 'VIEW']
        if not view_tables:
            print("   📋 No views found")
            return

        print(f"   👁️ Extracting definitions for {len(view_tables)} views...")

        query = """
        SELECT 
            SCHEMA_NAME(v.schema_id) + '.' + v.name as view_name,
            m.definition
        FROM sys.views v
        JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE SCHEMA_NAME(v.schema_id) + '.' + v.name = ?
        """

        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()

                for view_table in view_tables:
                    view_name = f"{view_table.schema}.{view_table.name}"

                    try:
                        cursor.execute(query, view_name)
                        row = cursor.fetchone()
                        if row:
                            self.view_definitions[view_name] = row[1]
                    except Exception:
                        continue

            print(f"   ✅ Extracted {len(self.view_definitions)} view definitions")

        except Exception as e:
            print(f"   ⚠️ View definition extraction failed: {e}")

    def _discover_relationships(self):
        """Discover relationships using multiple methods"""

        # Method 1: Foreign key relationships
        self._discover_foreign_key_relationships()

        # Method 2: Column name pattern relationships
        self._discover_pattern_relationships()

        # Method 3: View-based relationships
        self._discover_view_relationships()

        print(f"   ✅ Discovered {len(self.relationships)} relationships")

    def _discover_foreign_key_relationships(self):
        """Discover explicit foreign key relationships"""
        for table in self.tables:
            for fk_info in table.relationships:
                # Example: "CustomerId -> dbo.Customers.CustomerId"
                if '->' not in fk_info:
                    continue
                try:
                    _, rhs = fk_info.split('->', 1)
                    rhs = rhs.strip()
                    # Split into "schema.table" and "column"
                    table_part, _ = rhs.rsplit('.', 1)
                    # Normalize to [schema].[table]
                    if '.' in table_part:
                        sch, tbl = table_part.split('.', 1)
                        referenced_table_full = f"[{sch}].[{tbl}]"
                    else:
                        referenced_table_full = table_part  # best effort

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

        # Create lookup for tables by name patterns
        table_lookup = {}
        for table in self.tables:
            table_lookup[table.name.lower()] = table.full_name

        for table in self.tables:
            column_names = [col['name'].lower() for col in table.columns]

            for col_name in column_names:
                # Look for ID columns that reference other tables
                if col_name.endswith('id') and col_name != 'id':
                    entity_name = col_name[:-2]  # Remove 'id'

                    # Find matching table
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
        """Discover relationships from view dependencies (authoritative) with fallback to text parse."""
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
                            # Very light heuristic that looks around FROM/JOIN tokens
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
            return str(value)[:200]  # Truncate long values

    def _show_discovery_summary(self):
        """Show discovery summary"""

        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')

        total_samples = sum(len(table.sample_data) for table in self.tables)

        print(f"\n📊 DISCOVERY SUMMARY:")
        print(f"   📋 Objects analyzed: {len(self.tables)} (Tables: {table_count}, Views: {view_count})")
        print(f"   📝 Sample rows collected (first/last): {total_samples}")
        print(f"   👁️ View definitions: {len(self.view_definitions)}")
        print(f"   🔗 Relationships discovered: {len(self.relationships)}")

        # Show relationship types
        if self.relationships:
            rel_types = {}
            for rel in self.relationships:
                rel_types[rel.relationship_type] = rel_types.get(rel.relationship_type, 0) + 1

            print(f"   🔗 Relationship types:")
            for rel_type, count in rel_types.items():
                print(f"      • {rel_type}: {count}")

    def _save_to_cache(self):
        """Save discovery results to cache"""

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
                'sample_data': table.sample_data,  # includes __edge tags
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
            'view_definitions': self.view_definitions,
            'created': datetime.now().isoformat(),
            'version': '2.2-enhanced-edge-samples-timeouts'
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

            # Load view definitions
            self.view_definitions = data.get('view_definitions', {})

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
        """Get view definitions"""
        return self.view_definitions
