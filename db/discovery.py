#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Database Discovery
Following README: Simple, Readable, Maintainable
Core features: Schema + samples + view/SP analysis
"""

import pyodbc
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from shared.config import Config
from shared.models import TableInfo, Relationship, DatabaseObject
from shared.utils import safe_database_value

class DatabaseDiscovery:
    """Simple database discovery with view/SP analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables: List[TableInfo] = []
        self.relationships: List[Relationship] = []
        self.view_info: Dict[str, Dict[str, Any]] = {}
        self.stored_procedure_info: Dict[str, Dict[str, Any]] = {}
    
    def get_connection(self):
        """Get database connection with UTF-8 support"""
        conn = pyodbc.connect(self.config.get_database_connection_string())
        
        # UTF-8 encoding for international characters (README requirement)
        try:
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
        except Exception:
            pass  # Some drivers may not support this
        
        return conn
    
    async def discover_database(self) -> bool:
        """Main discovery method"""
        print("🔍 Starting database discovery...")
        
        # Check cache first
        if self.load_from_cache():
            print(f"✅ Loaded from cache: {len(self.tables)} objects")
            return True
        
        try:
            # Step 1: Get database objects
            objects = self.get_database_objects()
            if not objects:
                print("❌ No database objects found")
                return False
            
            print(f"📊 Found {len(objects)} objects to analyze")
            
            # Step 2: Analyze objects
            await self.analyze_objects(objects)
            
            # Step 3: Extract view definitions (README requirement)
            await self.extract_view_definitions()
            
            # Step 4: Analyze stored procedures (README requirement)
            await self.analyze_stored_procedures()
            
            # Step 5: Discover relationships
            self.discover_relationships()
            
            # Step 6: Save results
            self.save_to_cache()
            
            self.show_summary()
            return True
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    def get_database_objects(self) -> List[DatabaseObject]:
        """Get all database objects with exclusion filtering"""
        
        # Enhanced query including stored procedures (README requirement)
        query = """
        SELECT 
            SCHEMA_NAME(t.schema_id) as schema_name,
            t.name as object_name,
            'BASE TABLE' as object_type,
            COALESCE((SELECT SUM(p.rows) FROM sys.partitions p 
                     WHERE p.object_id = t.object_id AND p.index_id IN (0,1)), 0) as estimated_rows
        FROM sys.tables t
        WHERE t.is_ms_shipped = 0
          AND SCHEMA_NAME(t.schema_id) NOT IN ('sys','information_schema')
          AND LOWER(t.name) NOT LIKE '%bck%'
          AND LOWER(t.name) NOT LIKE '%backup%'
          AND LOWER(t.name) NOT LIKE '%dev%'

        UNION ALL

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

        SELECT 
            SCHEMA_NAME(p.schema_id) as schema_name,
            p.name as object_name,
            'STORED PROCEDURE' as object_type,
            0 as estimated_rows
        FROM sys.procedures p
        WHERE p.is_ms_shipped = 0
          AND SCHEMA_NAME(p.schema_id) NOT IN ('sys','information_schema')
          AND LOWER(p.name) NOT LIKE '%bck%'
          AND LOWER(p.name) NOT LIKE '%backup%'
          AND LOWER(p.name) NOT LIKE '%dev%'
          AND p.name NOT LIKE 'sp_%'

        ORDER BY estimated_rows DESC, object_name
        """
        
        try:
            with self.get_connection() as conn:
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
    
    async def analyze_objects(self, objects: List[DatabaseObject]):
        """Analyze objects and collect sample data"""
        print(f"🔄 Analyzing {len(objects)} objects...")
        
        for i, obj in enumerate(objects, 1):
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(objects)}")
            
            try:
                table_info = self.analyze_single_object(obj)
                if table_info:
                    self.tables.append(table_info)
            except Exception as e:
                print(f"   ⚠️ Failed to analyze {obj.name}: {e}")
    
    def analyze_single_object(self, obj: DatabaseObject) -> Optional[TableInfo]:
        """Analyze a single database object"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.timeout = self.config.query_timeout_seconds
                
                # Get columns (skip for stored procedures)
                if obj.object_type != 'STORED PROCEDURE':
                    columns = self.get_columns(cursor, obj.schema, obj.name)
                    if not columns:
                        return None
                    
                    # Get sample data (README requirement: first 3 + last 3)
                    sample_data = self.get_sample_data(cursor, obj, columns)
                    
                    # Get foreign keys
                    foreign_keys = self.get_foreign_keys(cursor, obj.schema, obj.name)
                else:
                    # For stored procedures, get parameters
                    columns = self.get_procedure_parameters(cursor, obj.schema, obj.name)
                    sample_data = []
                    foreign_keys = []
                
                return TableInfo(
                    name=obj.name,
                    schema=obj.schema,
                    full_name=obj.full_name,
                    object_type=obj.object_type,
                    row_count=obj.estimated_rows,
                    columns=columns,
                    sample_data=sample_data,
                    relationships=foreign_keys
                )
                
        except Exception:
            return None
    
    def get_columns(self, cursor, schema: str, name: str) -> List[Dict[str, Any]]:
        """Get column information"""
        query = """
        SELECT 
            c.COLUMN_NAME,
            c.DATA_TYPE,
            c.IS_NULLABLE,
            CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as is_primary_key
        FROM INFORMATION_SCHEMA.COLUMNS c
        LEFT JOIN (
            SELECT ku.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
              AND tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ?
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
                    'is_primary_key': bool(row[3])
                })
            return columns
        except Exception:
            return []
    
    def get_sample_data(self, cursor, obj: DatabaseObject, columns: List[Dict]) -> List[Dict[str, Any]]:
        """Get sample data: first 3 + last 3 rows (README requirement)"""
        full_name = obj.full_name
        
        # Find best column for ordering
        order_col = self.find_order_column(columns)
        
        # Get first 3 rows
        first_sql = f"SELECT TOP 3 * FROM {full_name}"
        if order_col:
            first_sql += f" ORDER BY [{order_col}] ASC"
        
        # Get last 3 rows  
        last_sql = f"SELECT TOP 3 * FROM {full_name}"
        if order_col:
            last_sql += f" ORDER BY [{order_col}] DESC"
        
        samples = []
        
        # Fetch first rows
        try:
            cursor.execute(first_sql)
            if cursor.description:
                cols = [c[0] for c in cursor.description]
                for row in cursor.fetchmany(3):
                    row_dict = {cols[i]: safe_database_value(val) 
                              for i, val in enumerate(row) if i < len(cols)}
                    row_dict["__edge"] = "first"
                    samples.append(row_dict)
        except Exception:
            pass
        
        # Fetch last rows
        try:
            cursor.execute(last_sql)
            if cursor.description:
                cols = [c[0] for c in cursor.description]
                for row in cursor.fetchmany(3):
                    row_dict = {cols[i]: safe_database_value(val) 
                              for i, val in enumerate(row) if i < len(cols)}
                    row_dict["__edge"] = "last"
                    samples.append(row_dict)
        except Exception:
            pass
        
        return samples
    
    def find_order_column(self, columns: List[Dict]) -> Optional[str]:
        """Find best column for ordering sample data"""
        if not columns:
            return None
        
        # Prefer primary key
        for col in columns:
            if col.get('is_primary_key'):
                return col['name']
        
        # Prefer date columns
        for col in columns:
            col_name = col['name'].lower()
            data_type = col.get('data_type', '').lower()
            if ('date' in col_name or 'time' in col_name or 
                'datetime' in data_type or 'date' in data_type):
                return col['name']
        
        # Use first column
        return columns[0]['name']
    
    def get_foreign_keys(self, cursor, schema: str, name: str) -> List[str]:
        """Get foreign key relationships"""
        query = """
        SELECT 
            OBJECT_SCHEMA_NAME(f.referenced_object_id) + '.' + OBJECT_NAME(f.referenced_object_id) as referenced_table,
            COL_NAME(f.parent_object_id, f.parent_column_id) as column_name,
            COL_NAME(f.referenced_object_id, f.referenced_column_id) as referenced_column
        FROM sys.foreign_key_columns f
        WHERE OBJECT_SCHEMA_NAME(f.parent_object_id) = ? AND OBJECT_NAME(f.parent_object_id) = ?
        """
        
        try:
            cursor.execute(query, schema, name)
            relationships = []
            for row in cursor.fetchall():
                relationships.append(f"{row[1]} -> {row[0]}.{row[2]}")
            return relationships
        except Exception:
            return []
    
    def get_procedure_parameters(self, cursor, schema: str, name: str) -> List[Dict[str, Any]]:
        """Get stored procedure parameters"""
        query = """
        SELECT 
            p.name as parameter_name,
            TYPE_NAME(p.system_type_id) as data_type,
            p.is_output
        FROM sys.parameters p
        JOIN sys.procedures pr ON p.object_id = pr.object_id
        WHERE SCHEMA_NAME(pr.schema_id) = ? AND pr.name = ? AND p.parameter_id > 0
        ORDER BY p.parameter_id
        """
        
        try:
            cursor.execute(query, schema, name)
            parameters = []
            for row in cursor.fetchall():
                parameters.append({
                    'name': row[0] or '',
                    'data_type': row[1] or 'unknown',
                    'is_output': bool(row[2]),
                    'is_parameter': True
                })
            return parameters
        except Exception:
            return []
    
    async def extract_view_definitions(self):
        """Extract view definitions (README requirement)"""
        view_tables = [t for t in self.tables if t.object_type == 'VIEW']
        if not view_tables:
            return
        
        print(f"👁️ Extracting {len(view_tables)} view definitions...")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for view_table in view_tables:
                    view_name = f"{view_table.schema}.{view_table.name}"
                    
                    # Get view definition using OBJECT_DEFINITION
                    definition = self.get_view_definition(cursor, view_table.full_name)
                    
                    # Execute view for sample data
                    sample_data, success = self.execute_view_safely(cursor, view_table.full_name)
                    
                    self.view_info[view_name] = {
                        'full_name': view_table.full_name,
                        'definition': definition,
                        'sample_data': sample_data,
                        'execution_success': success
                    }
                    
        except Exception as e:
            print(f"⚠️ View analysis failed: {e}")
    
    def get_view_definition(self, cursor, full_name: str) -> str:
        """Get view definition text"""
        try:
            cursor.execute("SELECT OBJECT_DEFINITION(OBJECT_ID(?))", full_name)
            row = cursor.fetchone()
            return row[0].strip() if row and row[0] else "Definition not available"
        except Exception:
            return "Definition extraction failed"
    
    def execute_view_safely(self, cursor, full_name: str) -> tuple:
        """Safely execute view for sample data"""
        try:
            cursor.execute(f"SELECT TOP 3 * FROM {full_name}")
            if not cursor.description:
                return [], False
                
            columns = [col[0] for col in cursor.description]
            sample_data = []
            
            for row in cursor.fetchmany(3):
                row_dict = {}
                for i, value in enumerate(row):
                    if i < len(columns):
                        row_dict[columns[i]] = safe_database_value(value)
                sample_data.append(row_dict)
            
            return sample_data, True
            
        except Exception:
            return [], False
    
    async def analyze_stored_procedures(self):
        """Analyze stored procedures (README requirement)"""
        sp_tables = [t for t in self.tables if t.object_type == 'STORED PROCEDURE']
        if not sp_tables:
            return
        
        print(f"⚙️ Analyzing {len(sp_tables)} stored procedures...")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for sp_table in sp_tables:
                    sp_name = f"{sp_table.schema}.{sp_table.name}"
                    
                    # Get procedure definition
                    definition = self.get_procedure_definition(cursor, sp_table.full_name)
                    
                    self.stored_procedure_info[sp_name] = {
                        'full_name': sp_table.full_name,
                        'definition': definition,
                        'parameters': sp_table.columns
                    }
                    
        except Exception as e:
            print(f"⚠️ Stored procedure analysis failed: {e}")
    
    def get_procedure_definition(self, cursor, full_name: str) -> str:
        """Get stored procedure definition"""
        try:
            cursor.execute("SELECT OBJECT_DEFINITION(OBJECT_ID(?))", full_name)
            row = cursor.fetchone()
            return row[0].strip() if row and row[0] else "Definition not available"
        except Exception:
            return "Definition extraction failed"
    
    def discover_relationships(self):
        """Discover relationships between objects"""
        print("🔗 Discovering relationships...")
        
        # Method 1: Foreign key relationships
        for table in self.tables:
            for fk_info in table.relationships:
                if '->' in fk_info:
                    try:
                        parts = fk_info.split('->', 1)
                        referenced = parts[1].strip()
                        
                        self.relationships.append(Relationship(
                            from_table=table.full_name,
                            to_table=referenced.split('.')[0] if '.' in referenced else referenced,
                            relationship_type='foreign_key',
                            confidence=0.95,
                            description=f"Foreign key: {fk_info}"
                        ))
                    except Exception:
                        continue
        
        # Method 2: Pattern-based relationships
        table_lookup = {t.name.lower(): t.full_name for t in self.tables}
        
        for table in self.tables:
            for col in table.columns:
                col_name = col.get('name', '').lower()
                if col_name.endswith('id') and col_name != 'id':
                    entity_name = col_name[:-2]
                    
                    for table_name, full_name in table_lookup.items():
                        if entity_name in table_name and full_name != table.full_name:
                            self.relationships.append(Relationship(
                                from_table=table.full_name,
                                to_table=full_name,
                                relationship_type='pattern_match',
                                confidence=0.7,
                                description=f"Column pattern: {col_name}"
                            ))
                            break
    
    def show_summary(self):
        """Show discovery summary"""
        table_count = sum(1 for t in self.tables if t.object_type in ['BASE TABLE', 'TABLE'])
        view_count = sum(1 for t in self.tables if t.object_type == 'VIEW')
        procedure_count = sum(1 for t in self.tables if t.object_type == 'STORED PROCEDURE')
        
        print(f"\n📊 DISCOVERY SUMMARY:")
        print(f"   📋 Tables: {table_count}")
        print(f"   👁️ Views: {view_count}")
        print(f"   ⚙️ Stored Procedures: {procedure_count}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        print(f"   📝 Sample rows: {sum(len(t.sample_data) for t in self.tables)}")
    
    def save_to_cache(self):
        """Save discovery results to cache"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        data = {
            'tables': [self.table_to_dict(t) for t in self.tables],
            'relationships': [self.relationship_to_dict(r) for r in self.relationships],
            'view_info': self.view_info,
            'stored_procedure_info': self.stored_procedure_info,
            'created': datetime.now().isoformat(),
            'version': '2.0-simple'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def load_from_cache(self) -> bool:
        """Load from cache if available and fresh"""
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
            self.tables = [self.dict_to_table(t) for t in data.get('tables', [])]
            
            # Load relationships  
            self.relationships = [self.dict_to_relationship(r) for r in data.get('relationships', [])]
            
            # Load view info
            self.view_info = data.get('view_info', {})
            self.stored_procedure_info = data.get('stored_procedure_info', {})
            
            return True
            
        except Exception:
            return False
    
    def table_to_dict(self, table: TableInfo) -> Dict:
        """Convert TableInfo to dictionary"""
        return {
            'name': table.name,
            'schema': table.schema,
            'full_name': table.full_name,
            'object_type': table.object_type,
            'row_count': table.row_count,
            'columns': table.columns,
            'sample_data': table.sample_data,
            'relationships': table.relationships,
            'entity_type': table.entity_type,
            'business_role': table.business_role,
            'confidence': table.confidence
        }
    
    def dict_to_table(self, data: Dict) -> TableInfo:
        """Convert dictionary to TableInfo"""
        table = TableInfo(
            name=data['name'],
            schema=data['schema'],
            full_name=data['full_name'],
            object_type=data['object_type'],
            row_count=data['row_count'],
            columns=data['columns'],
            sample_data=data['sample_data'],
            relationships=data.get('relationships', [])
        )
        table.entity_type = data.get('entity_type', 'Unknown')
        table.business_role = data.get('business_role', 'Unknown')
        table.confidence = data.get('confidence', 0.0)
        return table
    
    def relationship_to_dict(self, rel: Relationship) -> Dict:
        """Convert Relationship to dictionary"""
        return {
            'from_table': rel.from_table,
            'to_table': rel.to_table,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'description': rel.description
        }
    
    def dict_to_relationship(self, data: Dict) -> Relationship:
        """Convert dictionary to Relationship"""
        return Relationship(
            from_table=data['from_table'],
            to_table=data['to_table'],
            relationship_type=data['relationship_type'],
            confidence=data['confidence'],
            description=data.get('description', '')
        )
    
    # Public interface methods
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_view_info(self) -> Dict[str, Dict[str, Any]]:
        return self.view_info
    
    def get_stored_procedure_info(self) -> Dict[str, Dict[str, Any]]:
        return self.stored_procedure_info