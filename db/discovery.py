#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Discovery - Enhanced with First 3 + Last 3 Sampling
Following README: Schema + samples + view/SP definitions in JSON
DRY, SOLID, YAGNI principles - simple and maintainable
"""

import asyncio
import json
import pyodbc
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

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

class EnhancedSampleCollector:
    """Enhanced sample collection - First 3 + Last 3 rows"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def collect_samples(self, table_info: TableInfo) -> List[Dict[str, Any]]:
        """Collect first 3 + last 3 sample rows (README requirement)"""
        try:
            # Get primary key or first column for ordering
            order_column = self._get_order_column(table_info)
            
            if order_column:
                # Get first 3 rows
                first_3_sql = f"""
                SELECT TOP (3) * FROM {table_info.full_name} 
                ORDER BY [{order_column}] ASC
                """
                first_3 = self.connector.execute_query(first_3_sql)
                
                # Get last 3 rows  
                last_3_sql = f"""
                SELECT TOP (3) * FROM {table_info.full_name} 
                ORDER BY [{order_column}] DESC
                """
                last_3 = self.connector.execute_query(last_3_sql)
                
                # Combine samples with metadata
                samples = []
                
                # Add first 3 with position markers
                for i, row in enumerate(first_3, 1):
                    row['__sample_position__'] = f'first_{i}'
                    samples.append(row)
                
                # Add last 3 with position markers (reverse to maintain order)
                for i, row in enumerate(reversed(last_3), 1):
                    row['__sample_position__'] = f'last_{i}'
                    samples.append(row)
                
                print(f"   📋 Collected first 3 + last 3 samples for {table_info.name}")
                return samples
            
            else:
                # Fallback: just get first 6 rows
                fallback_sql = f"SELECT TOP (6) * FROM {table_info.full_name}"
                samples = self.connector.execute_query(fallback_sql)
                
                # Add position markers
                for i, row in enumerate(samples, 1):
                    row['__sample_position__'] = f'row_{i}'
                
                return samples
                
        except Exception as e:
            print(f"   ⚠️ Sample collection failed for {table_info.full_name}: {e}")
            return []
    
    def _get_order_column(self, table_info: TableInfo) -> Optional[str]:
        """Get best column for ordering (PK, ID, date, or first column)"""
        if not table_info.columns:
            return None
        
        # Look for primary key or ID columns
        for col in table_info.columns:
            col_name = col.get('name', '').lower()
            if col_name in ['id', 'pk'] or col_name.endswith('id'):
                return col.get('name')
        
        # Look for date/time columns
        for col in table_info.columns:
            col_type = col.get('data_type', '').lower()
            col_name = col.get('name', '').lower()
            if any(t in col_type for t in ['date', 'time']) or any(w in col_name for w in ['date', 'created', 'modified']):
                return col.get('name')
        
        # Use first column as fallback
        return table_info.columns[0].get('name') if table_info.columns else None

class EnhancedViewAnalyzer:
    """Enhanced view analysis with definition storage"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def analyze_views(self) -> Dict[str, Dict[str, Any]]:
        """Analyze views and store complete definitions in JSON"""
        print("🔍 Analyzing views with definitions...")
        
        sql = """
        SELECT 
            s.name as schema_name,
            v.name as view_name,
            m.definition as view_definition,
            v.create_date,
            v.modify_date
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
            
            # Store complete view information
            view_info[full_name] = {
                'schema': schema,
                'name': name,
                'full_name': full_name,
                'object_type': 'VIEW',
                'definition': definition,
                'create_date': row.get('create_date'),
                'modify_date': row.get('modify_date'),
                'referenced_objects': [],
                'parsed_joins': [],
                'parsing_success': False,
                'query_type': self._classify_view_type(definition)
            }
            
            # Parse with SQLGlot if available
            if HAS_SQLGLOT and definition:
                try:
                    parsed = sqlglot.parse_one(definition, dialect="tsql")
                    if parsed:
                        # Extract referenced tables
                        tables = []
                        joins = []
                        
                        # Find all tables
                        for table in parsed.find_all(sqlglot.expressions.Table):
                            if table.this:
                                table_name = str(table.this)
                                if table.db:
                                    table_name = f"[{table.db}].[{table_name}]"
                                elif table.catalog:
                                    table_name = f"[{table.catalog}].[{table_name}]"
                                tables.append(table_name)
                        
                        # Find JOIN patterns
                        for join in parsed.find_all(sqlglot.expressions.Join):
                            if join.this and hasattr(join.this, 'this'):
                                join_info = {
                                    'join_type': str(join.kind) if join.kind else 'INNER',
                                    'table': str(join.this.this) if hasattr(join.this, 'this') else str(join.this),
                                    'condition': str(join.on) if join.on else None
                                }
                                joins.append(join_info)
                        
                        view_info[full_name].update({
                            'referenced_objects': list(set(tables)),
                            'parsed_joins': joins,
                            'parsing_success': True
                        })
                        
                except Exception as e:
                    view_info[full_name]['parsing_error'] = str(e)
        
        print(f"   ✅ Analyzed {len(view_info)} views with definitions")
        return view_info
    
    def _classify_view_type(self, definition: str) -> str:
        """Classify view type based on definition"""
        if not definition:
            return 'unknown'
        
        definition_lower = definition.lower()
        
        if 'union' in definition_lower:
            return 'union_view'
        elif 'join' in definition_lower:
            return 'joined_view'
        elif 'group by' in definition_lower:
            return 'aggregated_view'
        elif 'where' in definition_lower:
            return 'filtered_view'
        else:
            return 'simple_view'

class StoredProcedureAnalyzer:
    """Stored procedure analysis with definition storage"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
    
    def analyze_procedures(self) -> Dict[str, Dict[str, Any]]:
        """Analyze stored procedures and store definitions"""
        print("🔍 Analyzing stored procedures with definitions...")
        
        sql = """
        SELECT 
            s.name as schema_name,
            p.name as procedure_name,
            m.definition as procedure_definition,
            p.create_date,
            p.modify_date,
            p.type_desc
        FROM sys.procedures p
        INNER JOIN sys.schemas s ON p.schema_id = s.schema_id
        INNER JOIN sys.sql_modules m ON p.object_id = m.object_id
        WHERE s.name NOT IN ('sys', 'information_schema')
        ORDER BY s.name, p.name
        """
        
        results = self.connector.execute_query(sql)
        sp_info = {}
        
        for row in results:
            schema = row['schema_name']
            name = row['procedure_name']
            definition = row['procedure_definition']
            full_name = f"[{schema}].[{name}]"
            
            # Store complete procedure information
            sp_info[full_name] = {
                'schema': schema,
                'name': name,
                'full_name': full_name,
                'object_type': 'STORED_PROCEDURE',
                'definition': definition,
                'create_date': row.get('create_date'),
                'modify_date': row.get('modify_date'),
                'type_desc': row.get('type_desc'),
                'referenced_objects': [],
                'select_statements': [],
                'parsing_success': False,
                'procedure_type': self._classify_procedure_type(definition)
            }
            
            # Extract SELECT statements and referenced objects
            if definition:
                select_statements = self._extract_select_statements(definition)
                sp_info[full_name]['select_statements'] = select_statements
                
                # Extract referenced tables from SELECT statements
                referenced_objects = set()
                for select_stmt in select_statements:
                    tables = self._extract_tables_from_select(select_stmt)
                    referenced_objects.update(tables)
                
                sp_info[full_name]['referenced_objects'] = list(referenced_objects)
                sp_info[full_name]['parsing_success'] = True
        
        print(f"   ✅ Analyzed {len(sp_info)} stored procedures with definitions")
        return sp_info
    
    def _classify_procedure_type(self, definition: str) -> str:
        """Classify procedure type based on definition"""
        if not definition:
            return 'unknown'
        
        definition_lower = definition.lower()
        
        if 'insert' in definition_lower and 'update' in definition_lower:
            return 'crud_procedure'
        elif 'select' in definition_lower and 'from' in definition_lower:
            return 'query_procedure'
        elif 'delete' in definition_lower:
            return 'delete_procedure'
        elif 'insert' in definition_lower:
            return 'insert_procedure'
        elif 'update' in definition_lower:
            return 'update_procedure'
        else:
            return 'utility_procedure'
    
    def _extract_select_statements(self, definition: str) -> List[str]:
        """Extract SELECT statements from procedure definition"""
        if not definition:
            return []
        
        # Simple regex-based extraction
        import re
        
        # Find SELECT statements (basic pattern)
        select_pattern = r'SELECT\s+.*?(?=SELECT\s|$)'
        matches = re.findall(select_pattern, definition, re.IGNORECASE | re.DOTALL)
        
        # Clean and filter valid SELECT statements
        select_statements = []
        for match in matches:
            cleaned = match.strip()
            if len(cleaned) > 20 and 'FROM' in cleaned.upper():  # Basic validation
                select_statements.append(cleaned)
        
        return select_statements[:5]  # Limit to first 5 statements
    
    def _extract_tables_from_select(self, select_statement: str) -> List[str]:
        """Extract table names from SELECT statement"""
        if not select_statement:
            return []
        
        # Simple table extraction using regex
        import re
        
        # Pattern to find FROM and JOIN clauses
        table_pattern = r'(?:FROM|JOIN)\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)'
        matches = re.findall(table_pattern, select_statement, re.IGNORECASE)
        
        # Clean table names
        tables = []
        for match in matches:
            cleaned = match.strip()
            if cleaned and not cleaned.lower() in ['where', 'group', 'order', 'having']:
                tables.append(cleaned)
        
        return list(set(tables))  # Remove duplicates

class EnhancedCacheManager:
    """Enhanced cache management with view/SP definitions"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_cache(self, tables: List[TableInfo], view_info: Dict, sp_info: Dict):
        """Save enhanced discovery results to cache"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        data = {
            'metadata': {
                'discovered': datetime.now().isoformat(),
                'version': '2.1-enhanced',
                'sampling_method': 'first_3_plus_last_3',
                'includes_definitions': True
            },
            'discovery_summary': {
                'total_tables': len(tables),
                'total_views': len(view_info),
                'total_procedures': len(sp_info),
                'sqlglot_available': HAS_SQLGLOT
            },
            'tables': [self._table_to_dict(t) for t in tables],
            'views': view_info,
            'stored_procedures': sp_info
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Enhanced discovery cached: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def load_cache(self) -> Tuple[List[TableInfo], Dict, Dict]:
        """Load enhanced discovery results from cache"""
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
            view_info = data.get('views', data.get('view_info', {}))  # Backward compatibility
            sp_info = data.get('stored_procedures', data.get('procedure_info', {}))
            
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
            'relationships': table.relationships,
            'sampling_method': 'first_3_plus_last_3'
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
    """Enhanced database discovery orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize enhanced components
        self.connector = DatabaseConnector(config)
        self.sample_collector = EnhancedSampleCollector(self.connector)
        self.view_analyzer = EnhancedViewAnalyzer(self.connector)
        self.sp_analyzer = StoredProcedureAnalyzer(self.connector)
        self.cache_manager = EnhancedCacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.view_info: Dict = {}
        self.sp_info: Dict = {}
        self.relationships: List[Relationship] = []
    
    async def discover_database(self) -> bool:
        """Enhanced discovery with first 3 + last 3 sampling"""
        print("🔍 ENHANCED DATABASE DISCOVERY")
        print("Following README: First 3 + Last 3 sampling, View/SP definitions")
        print("=" * 60)
        
        # Check cache first
        cached_tables, cached_views, cached_sp = self.cache_manager.load_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.sp_info = cached_sp
            print(f"✅ Loaded from cache: {len(self.tables)} tables, {len(self.view_info)} views, {len(self.sp_info)} procedures")
            return True
        
        try:
            start_time = time.time()
            
            # Step 1: Discover tables
            await self._discover_tables()
            
            # Step 2: Analyze views with definitions
            if self.config.is_view_analysis_enabled():
                self.view_info = self.view_analyzer.analyze_views()
            
            # Step 3: Analyze stored procedures
            self.sp_info = self.sp_analyzer.analyze_procedures()
            
            # Step 4: Build relationships
            self._build_relationships()
            
            # Step 5: Save enhanced cache
            self.cache_manager.save_cache(self.tables, self.view_info, self.sp_info)
            
            # Show enhanced summary
            self._show_enhanced_summary(time.time() - start_time)
            return True
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    async def _discover_tables(self):
        """Discover and analyze tables"""
        print(f"📊 Discovering tables with enhanced sampling...")
        
        # Get table list
        sql = """
        SELECT 
            s.name as schema_name,
            t.name as table_name,
            t.type_desc as object_type,
            ISNULL(p.rows, 0) as estimated_rows
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
        ORDER BY s.name, t.name
        """
        
        results = self.connector.execute_query(sql)
        exclusion_patterns = self.config.get_exclusion_patterns()
        
        for row in results:
            schema = row['schema_name']
            name = row['table_name']
            
            # Apply exclusions
            if should_exclude_table(name, schema, exclusion_patterns):
                continue
            
            try:
                # Get columns and relationships
                columns, relationships = self._get_table_details(schema, name)
                
                if not columns:
                    continue
                
                # Create table info
                table_info = TableInfo(
                    name=name,
                    schema=schema,
                    full_name=f"[{schema}].[{name}]",
                    object_type=row['object_type'],
                    row_count=row['estimated_rows'],
                    columns=columns,
                    sample_data=[],
                    relationships=relationships
                )
                
                # Collect enhanced samples (first 3 + last 3)
                if row['estimated_rows'] > 0:
                    samples = self.sample_collector.collect_samples(table_info)
                    table_info.sample_data = samples
                
                self.tables.append(table_info)
                
            except Exception as e:
                print(f"   ⚠️ Failed to analyze {schema}.{name}: {e}")
        
        print(f"   ✅ Discovered {len(self.tables)} tables with enhanced sampling")
    
    def _get_table_details(self, schema: str, table: str) -> Tuple[List[Dict], List[str]]:
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
    
    def _show_enhanced_summary(self, elapsed_time: float):
        """Show enhanced discovery summary"""
        # Debug: Show what object types we actually have
        object_types = {}
        for t in self.tables:
            obj_type = t.object_type
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        print(f"   🔍 DEBUG: Object types found: {object_types}")
        
        # Count all table-like objects
        table_count = len([t for t in self.tables if t.object_type not in ['VIEW', 'SYSTEM_VIEW']])
        
        print(f"\n📊 ENHANCED DISCOVERY COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Tables: {table_count}")
        print(f"   👁️ Views: {len(self.view_info)}")
        print(f"   ⚙️ Stored Procedures: {len(self.sp_info)}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        print(f"   📋 Total objects: {len(self.tables)}")
        print(f"   📝 Sampling: First 3 + Last 3 rows per table")
        
        # Show parsing success rates
        if self.view_info:
            parsed_views = sum(1 for v in self.view_info.values() if v.get('parsing_success'))
            print(f"   ✅ View parsing: {parsed_views}/{len(self.view_info)} successful")
        
        if self.sp_info:
            parsed_sps = sum(1 for sp in self.sp_info.values() if sp.get('parsing_success'))
            print(f"   ✅ SP parsing: {parsed_sps}/{len(self.sp_info)} successful")
    
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
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_view_info(self) -> Dict:
        return self.view_info
    
    def get_stored_procedure_info(self) -> Dict:
        return self.sp_info
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        table_count = sum(1 for t in self.tables if t.object_type in ['USER_TABLE', 'BASE TABLE', 'TABLE'])
        
        return {
            'total_objects': len(self.tables),
            'tables': table_count,
            'views': len(self.view_info),
            'stored_procedures': len(self.sp_info),
            'relationships': len(self.relationships),
            'sqlglot_available': HAS_SQLGLOT,
            'sampling_method': 'first_3_plus_last_3',
            'includes_definitions': True
        }