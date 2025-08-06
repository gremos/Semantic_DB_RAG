#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Database Discovery - Simple and Maintainable
Implements view definition mining and relationship discovery
"""

import pyodbc
import asyncio
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

from shared.config import Config
from shared.models import TableInfo, Relationship, DatabaseObject

class DatabaseDiscovery:
    """Enhanced database discovery with view analysis and relationship discovery"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.view_definitions: Dict[str, str] = {}
    
    def get_database_connection(self):
        """Get database connection with Greek text support"""
        connection_string = self.config.get_database_connection_string()
        conn = pyodbc.connect(connection_string)
        
        # UTF-8 encoding for international characters
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        
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
        """Get all database objects with priority scoring"""
        
        query = """
        -- Get tables with metadata
        SELECT 
            SCHEMA_NAME(t.schema_id) as schema_name,
            t.name as object_name,
            'BASE TABLE' as object_type,
            COALESCE(
                (SELECT SUM(rows) FROM sys.partitions p WHERE p.object_id = t.object_id AND p.index_id < 2),
                0
            ) as estimated_rows
        FROM sys.tables t
        WHERE t.is_ms_shipped = 0
          AND SCHEMA_NAME(t.schema_id) NOT IN ('sys', 'information_schema')
        
        UNION ALL
        
        -- Get views
        SELECT 
            SCHEMA_NAME(v.schema_id) as schema_name,
            v.name as object_name,
            'VIEW' as object_type,
            0 as estimated_rows
        FROM sys.views v
        WHERE v.is_ms_shipped = 0
          AND SCHEMA_NAME(v.schema_id) NOT IN ('sys', 'information_schema')
        
        ORDER BY estimated_rows DESC, object_name
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
                    objects.append(obj)
                
                return objects
                
        except Exception as e:
            print(f"❌ Failed to get database objects: {e}")
            return []
    
    async def _analyze_objects(self, objects: List[DatabaseObject]):
        """Analyze objects and collect sample data"""
        
        pbar = tqdm(objects, desc="Analyzing objects", unit="obj")
        timeout_seconds = 5 * 60  # Convert minutes to seconds
        
        for obj in pbar:
            try:
                # Add timeout wrapper - skip objects that take too long
                table_info = await asyncio.wait_for(
                    asyncio.to_thread(self._analyze_single_object, obj),
                    timeout=timeout_seconds
                )
                if table_info:
                    self.tables.append(table_info)
                    
                pbar.set_description(f"Analyzed {len(self.tables)}/{len(objects)}")
                
            except asyncio.TimeoutError:
                print(f"   ⏰ Timeout analyzing {obj.name} (>{timeout_seconds//60}min) - skipping")
            except Exception as e:
                print(f"   ⚠️ Failed to analyze {obj.name}: {e}")
        
        pbar.close()
    
    def _analyze_single_object(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze a single database object"""
        
        try:
            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                
                # Get column information
                columns = self._get_columns(cursor, obj.schema, obj.name)
                if not columns:
                    return None
                
                # Get sample data
                sample_data = self._get_sample_data(cursor, obj)
                
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
        except:
            return []
    
    def _get_sample_data(self, cursor, obj: DatabaseObject) -> List[Dict[str, Any]]:
        """Get 5 sample rows with multiple fallback strategies"""
        
        full_name = f"[{obj.schema}].[{obj.name}]"
        
        # Different strategies based on object type
        if obj.object_type == 'VIEW':
            strategies = [
                f"SELECT TOP 5 * FROM {full_name}",
                f"SELECT TOP 3 * FROM {full_name}",
                f"SELECT TOP 1 * FROM {full_name}"
            ]
        else:
            strategies = [
                f"SELECT TOP 5 * FROM {full_name} OPTION (FAST 5)",
                f"SELECT TOP 5 * FROM {full_name} WITH (NOLOCK)",
                f"SELECT TOP 5 * FROM {full_name}",
                f"SELECT TOP 1 * FROM {full_name}"
            ]
        
        for strategy in strategies:
            try:
                cursor.execute(strategy)
                
                if not cursor.description:
                    continue
                
                col_names = [col[0] for col in cursor.description]
                sample_data = []
                
                for row in cursor.fetchmany(5):
                    row_dict = {}
                    for i, value in enumerate(row):
                        if i < len(col_names):
                            row_dict[col_names[i]] = self._safe_value(value)
                    sample_data.append(row_dict)
                
                if sample_data:
                    return sample_data
                    
            except:
                continue
        
        return []
    
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
        except:
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
                    except:
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
                if '->' in fk_info:
                    parts = fk_info.split(' -> ')
                    if len(parts) == 2:
                        referenced_table = parts[1].split('.')[0]
                        
                        self.relationships.append(Relationship(
                            from_table=table.full_name,
                            to_table=referenced_table,
                            relationship_type='foreign_key',
                            confidence=0.95,
                            description=f"Foreign key: {fk_info}"
                        ))
    
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
        """Discover relationships from view definitions"""
        
        for view_name, definition in self.view_definitions.items():
            if not definition:
                continue
            
            # Simple JOIN pattern extraction
            definition_upper = definition.upper()
            
            # Look for JOIN patterns
            if 'JOIN' in definition_upper:
                # Extract table names from JOIN clauses (simplified)
                words = definition_upper.split()
                
                for i, word in enumerate(words):
                    if word == 'JOIN' and i + 1 < len(words):
                        joined_table = words[i + 1].strip('[]')
                        
                        # Find matching table in our list
                        for table in self.tables:
                            if joined_table.lower() in table.full_name.lower():
                                self.relationships.append(Relationship(
                                    from_table=f"[{view_name.split('.')[0]}].[{view_name.split('.')[1]}]",
                                    to_table=table.full_name,
                                    relationship_type='view_join',
                                    confidence=0.8,
                                    description=f"View join pattern in {view_name}"
                                ))
                                break
    
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
        print(f"   📝 Sample rows collected: {total_samples}")
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
            'view_definitions': self.view_definitions,
            'created': datetime.now().isoformat(),
            'version': '2.0-enhanced'
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