#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Discovery - Enhanced with SQL Server sys.* views and Multi-RDL integration
Following Architecture: SQL Server only, sys.* metadata, Multi-RDL parsing, sqlglot validation
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
"""

import asyncio
import json
import pyodbc
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import glob

# SQLGlot for SQL parsing (required by Architecture)
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False
    print("⚠️ sqlglot not available - install with: pip install sqlglot")

from shared.config import Config
from shared.models import TableInfo, DatabaseObject, Relationship
from shared.utils import safe_database_value, should_exclude_table, normalize_table_name

class SqlServerConnector:
    """SQL Server connection with proper UTF-8 and sys.* support"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_connection(self):
        """Get SQL Server connection with UTF-8 support"""
        conn = pyodbc.connect(self.config.get_database_connection_string())
        
        # UTF-8 support for international characters (Greek etc.)
        if self.config.utf8_encoding:
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
        
        return conn
    
    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query and return results with error handling"""
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

class SqlServerMetadata:
    """SQL Server metadata extraction using sys.* views (Architecture requirement)"""
    
    def __init__(self, connector: SqlServerConnector):
        self.connector = connector
    
    def get_tables_with_row_counts(self) -> List[Dict[str, Any]]:
        """Get tables with estimated row counts using sys.dm_db_partition_stats"""
        sql = """
        SELECT 
            s.name as schema_name,
            t.name as table_name,
            t.type_desc as object_type,
            ISNULL(SUM(p.rows), 0) as estimated_rows
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN sys.dm_db_partition_stats p ON t.object_id = p.object_id 
            AND p.index_id IN (0,1)  -- Heap or clustered index
        WHERE s.name NOT IN ('sys', 'information_schema')
        GROUP BY s.name, t.name, t.type_desc
        ORDER BY s.name, t.name
        """
        
        print("   📊 Getting tables with sys.dm_db_partition_stats row counts...")
        return self.connector.execute_query(sql)
    
    def get_table_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get table columns using sys.columns + sys.types"""
        sql = """
        SELECT 
            c.name as column_name,
            t.name as data_type,
            c.max_length,
            c.is_nullable,
            c.collation_name,
            c.is_identity
        FROM sys.columns c
        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE c.object_id = OBJECT_ID(?)
        ORDER BY c.column_id
        """
        
        full_name = f"[{schema}].[{table}]"
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, full_name)
                
                columns = []
                for row in cursor:
                    columns.append({
                        'name': row.column_name,
                        'data_type': row.data_type,
                        'max_length': row.max_length,
                        'is_nullable': row.is_nullable,
                        'collation': row.collation_name,
                        'is_identity': row.is_identity
                    })
                
                return columns
        except Exception as e:
            print(f"   ⚠️ Column query failed for {full_name}: {e}")
            return []
    
    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        """Get primary keys using sys.key_constraints + sys.index_columns"""
        sql = """
        SELECT c.name as column_name
        FROM sys.key_constraints kc
        INNER JOIN sys.index_columns ic ON kc.parent_object_id = ic.object_id 
            AND kc.unique_index_id = ic.index_id
        INNER JOIN sys.columns c ON ic.object_id = c.object_id 
            AND ic.column_id = c.column_id
        WHERE kc.type = 'PK'
          AND kc.parent_object_id = OBJECT_ID(?)
        ORDER BY ic.key_ordinal
        """
        
        full_name = f"[{schema}].[{table}]"
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, full_name)
                
                return [row.column_name for row in cursor]
        except Exception:
            return []
    
    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get foreign keys using sys.foreign_keys + sys.foreign_key_columns (Architecture fix)"""
        sql = """
        SELECT 
            fkc.constraint_column_id,
            COL_NAME(fkc.parent_object_id, fkc.parent_column_id) as column_name,
            SCHEMA_NAME(ref_t.schema_id) as referenced_schema,
            OBJECT_NAME(fkc.referenced_object_id) as referenced_table,
            COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) as referenced_column,
            fk.name as constraint_name
        FROM sys.foreign_key_columns fkc
        INNER JOIN sys.foreign_keys fk ON fkc.constraint_object_id = fk.object_id
        INNER JOIN sys.tables ref_t ON fkc.referenced_object_id = ref_t.object_id
        WHERE fkc.parent_object_id = OBJECT_ID(?)
        ORDER BY fkc.constraint_column_id
        """
        
        full_name = f"[{schema}].[{table}]"
        
        try:
            with self.connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, full_name)
                
                foreign_keys = []
                for row in cursor:
                    foreign_keys.append({
                        'column_name': row.column_name,
                        'referenced_schema': row.referenced_schema,
                        'referenced_table': row.referenced_table,
                        'referenced_column': row.referenced_column,
                        'constraint_name': row.constraint_name
                    })
                
                return foreign_keys
        except Exception as e:
            print(f"   ⚠️ Foreign key query failed for {full_name}: {e}")
            return []
    
    def get_views_with_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get views with definitions using sys.views + sys.sql_modules"""
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
        
        print("   👁️ Getting views with definitions...")
        results = self.connector.execute_query(sql)
        
        view_info = {}
        for row in results:
            full_name = f"[{row['schema_name']}].[{row['view_name']}]"
            
            view_info[full_name] = {
                'schema': row['schema_name'],
                'name': row['view_name'],
                'full_name': full_name,
                'object_type': 'VIEW',
                'definition': row['view_definition'],
                'create_date': row.get('create_date'),
                'modify_date': row.get('modify_date'),
                'referenced_objects': [],
                'parsed_joins': [],
                'parsing_success': False
            }
            
            # Parse with sqlglot if available
            if HAS_SQLGLOT and row['view_definition']:
                try:
                    parsed = sqlglot.parse_one(row['view_definition'], dialect="tsql")
                    if parsed:
                        tables = []
                        joins = []
                        
                        # Extract referenced tables
                        for table in parsed.find_all(sqlglot.expressions.Table):
                            if table.this:
                                table_name = str(table.this)
                                if table.db:
                                    table_name = f"[{table.db}].[{table_name}]"
                                tables.append(table_name)
                        
                        # Extract JOIN information
                        for join in parsed.find_all(sqlglot.expressions.Join):
                            if join.this:
                                join_info = {
                                    'join_type': str(join.kind) if join.kind else 'INNER',
                                    'table': str(join.this),
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
        
        return view_info

class MultiRDLParser:
    """Multi-RDL parser for SSRS reports (Architecture requirement)"""
    
    def __init__(self):
        self.rdl_usage = {}
    
    def parse_all_rdl_files(self, rdl_directory: str = "data_upload") -> Dict[str, Any]:
        """Parse all RDL files in the directory for business insights"""
        print("   📋 Parsing all RDL files for business insights...")
        
        rdl_dir = Path(rdl_directory)
        if not rdl_dir.exists():
            print(f"   ⚠️ RDL directory not found: {rdl_directory}")
            return {}
        
        # Find all RDL files
        rdl_files = list(rdl_dir.glob("*.rdl"))
        if not rdl_files:
            print(f"   ⚠️ No RDL files found in: {rdl_directory}")
            return {}
        
        print(f"   📁 Found {len(rdl_files)} RDL files to process")
        
        # Combined RDL information
        combined_rdl_info = {
            'reports': [],
            'datasets': [],
            'parameters': [],
            'referenced_tables': set(),
            'business_priority_signals': [],
            'report_count': len(rdl_files)
        }
        
        # Process each RDL file
        for rdl_file in rdl_files:
            try:
                print(f"   📄 Processing: {rdl_file.name}")
                rdl_info = self._parse_single_rdl_file(rdl_file)
                
                if rdl_info:
                    # Add report info
                    combined_rdl_info['reports'].append({
                        'filename': rdl_file.name,
                        'title': rdl_info.get('report_title', rdl_file.stem),
                        'datasets_count': len(rdl_info.get('datasets', [])),
                        'parameters_count': len(rdl_info.get('parameters', [])),
                        'referenced_tables_count': len(rdl_info.get('referenced_tables', []))
                    })
                    
                    # Combine datasets
                    combined_rdl_info['datasets'].extend(rdl_info.get('datasets', []))
                    
                    # Combine parameters
                    combined_rdl_info['parameters'].extend(rdl_info.get('parameters', []))
                    
                    # Combine referenced tables
                    combined_rdl_info['referenced_tables'].update(rdl_info.get('referenced_tables', set()))
                    
                    # Combine business signals
                    combined_rdl_info['business_priority_signals'].extend(
                        rdl_info.get('business_priority_signals', [])
                    )
                    
            except Exception as e:
                print(f"   ⚠️ Failed to parse {rdl_file.name}: {e}")
        
        # Remove duplicates and finalize
        combined_rdl_info['referenced_tables'] = list(combined_rdl_info['referenced_tables'])
        combined_rdl_info['business_priority_signals'] = list(set(combined_rdl_info['business_priority_signals']))
        
        print(f"   ✅ Multi-RDL parsing completed:")
        print(f"      📊 Total reports: {len(combined_rdl_info['reports'])}")
        print(f"      📋 Total datasets: {len(combined_rdl_info['datasets'])}")
        print(f"      🔗 Referenced tables: {len(combined_rdl_info['referenced_tables'])}")
        print(f"      🎯 Business signals: {len(combined_rdl_info['business_priority_signals'])}")
        
        return combined_rdl_info
    
    def _parse_single_rdl_file(self, rdl_path: Path) -> Dict[str, Any]:
        """Parse a single RDL file for business insights and table usage"""
        try:
            # Parse XML
            tree = ET.parse(rdl_path)
            root = tree.getroot()
            
            # Handle namespace
            ns = {'': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'}
            if not root.tag.startswith('{'):
                # No namespace in root, try without
                ns = {}
            
            # Extract report metadata
            rdl_info = {
                'report_title': self._get_report_title(root, ns, rdl_path.stem),
                'datasets': [],
                'parameters': [],
                'referenced_tables': set(),
                'business_priority_signals': []
            }
            
            # Extract DataSets and their SQL
            datasets = root.findall('.//DataSet', ns) if ns else root.findall('.//DataSet')
            for dataset in datasets:
                dataset_info = self._extract_dataset_info(dataset, ns)
                if dataset_info:
                    rdl_info['datasets'].append(dataset_info)
                    # Track referenced tables
                    rdl_info['referenced_tables'].update(dataset_info.get('referenced_tables', []))
            
            # Extract Parameters
            params = root.findall('.//ReportParameter', ns) if ns else root.findall('.//ReportParameter')
            for param in params:
                param_info = self._extract_parameter_info(param, ns)
                if param_info:
                    rdl_info['parameters'].append(param_info)
            
            # Determine business priority signals
            rdl_info['business_priority_signals'] = self._analyze_business_priority(rdl_info)
            
            return rdl_info
            
        except Exception as e:
            print(f"   ⚠️ RDL parsing failed for {rdl_path.name}: {e}")
            return {}
    
    def _get_report_title(self, root, ns: Dict, fallback: str) -> str:
        """Extract report title"""
        # Try multiple ways to get title
        title_elements = [
            './/Textbox[@Name="textbox1"]',
            './/TextRun/Value',
            './/Textbox/Paragraphs/Paragraph/TextRuns/TextRun/Value'
        ]
        
        for xpath in title_elements:
            elements = root.findall(xpath, ns) if ns else root.findall(xpath)
            for element in elements:
                if element.text and len(element.text.strip()) > 5:
                    return element.text.strip()
        
        return fallback
    
    def _extract_dataset_info(self, dataset, ns: Dict) -> Dict[str, Any]:
        """Extract dataset information including SQL"""
        try:
            name_elem = dataset.find('Name', ns) if ns else dataset.find('Name')
            query_elem = dataset.find('.//Query', ns) if ns else dataset.find('.//Query')
            
            if not query_elem:
                return None
            
            command_text_elem = query_elem.find('CommandText', ns) if ns else query_elem.find('CommandText')
            if not command_text_elem or not command_text_elem.text:
                return None
            
            command_text = command_text_elem.text.strip()
            
            dataset_info = {
                'name': name_elem.text if name_elem is not None and name_elem.text else 'Unknown',
                'command_text': command_text,
                'referenced_tables': [],
                'fields': []
            }
            
            # Parse SQL to extract referenced tables
            if HAS_SQLGLOT and command_text:
                try:
                    parsed = sqlglot.parse_one(command_text, dialect="tsql")
                    if parsed:
                        tables = []
                        for table in parsed.find_all(sqlglot.expressions.Table):
                            if table.this:
                                table_name = str(table.this)
                                if table.db:
                                    table_name = f"[{table.db}].[{table_name}]"
                                tables.append(table_name)
                        dataset_info['referenced_tables'] = list(set(tables))
                except Exception:
                    # Fallback: simple regex extraction
                    dataset_info['referenced_tables'] = self._extract_tables_regex(command_text)
            
            # Extract fields
            fields_elem = dataset.find('.//Fields', ns) if ns else dataset.find('.//Fields')
            if fields_elem:
                for field in fields_elem.findall('Field', ns) if ns else fields_elem.findall('Field'):
                    name_attr = field.get('Name')
                    if name_attr:
                        dataset_info['fields'].append(name_attr)
            
            return dataset_info
            
        except Exception as e:
            print(f"   ⚠️ Dataset extraction failed: {e}")
            return None
    
    def _extract_tables_regex(self, sql: str) -> List[str]:
        """Fallback regex-based table extraction"""
        import re
        
        # Simple patterns for SQL Server table references
        patterns = [
            r'FROM\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)',
            r'JOIN\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)',
            r'UPDATE\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)',
            r'INSERT\s+INTO\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)'
        ]
        
        tables = set()
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if cleaned and not cleaned.upper() in ['WHERE', 'SELECT', 'GROUP', 'ORDER']:
                    tables.add(cleaned)
        
        return list(tables)
    
    def _extract_parameter_info(self, param, ns: Dict) -> Dict[str, Any]:
        """Extract parameter information"""
        try:
            name_attr = param.get('Name')
            data_type_elem = param.find('DataType', ns) if ns else param.find('DataType')
            prompt_elem = param.find('Prompt', ns) if ns else param.find('Prompt')
            
            return {
                'name': name_attr or 'Unknown',
                'data_type': data_type_elem.text if data_type_elem is not None else 'String',
                'prompt': prompt_elem.text if prompt_elem is not None else ''
            }
        except Exception:
            return None
    
    def _analyze_business_priority(self, rdl_info: Dict) -> List[str]:
        """Analyze RDL for business priority signals"""
        signals = []
        
        title = rdl_info.get('report_title', '').lower()
        
        # Check for executive/high-priority indicators
        if any(word in title for word in ['executive', 'weekly', 'monthly', 'approved', 'συμβόλαια']):
            signals.append('executive_report')
        
        # Check for financial indicators
        if any(word in title for word in ['revenue', 'payment', 'financial', 'contracts']):
            signals.append('financial_data')
        
        # Check dataset complexity
        if len(rdl_info.get('datasets', [])) > 1:
            signals.append('complex_report')
        
        return signals
    
    def generate_rdl_usage_json(self, rdl_info: Dict) -> Dict[str, Any]:
        """Generate RDL usage JSON for boosting table priorities"""
        usage_data = {
            'metadata': {
                'parsed_at': datetime.now().isoformat(),
                'report_count': rdl_info.get('report_count', 0),
                'priority_signals': rdl_info.get('business_priority_signals', [])
            },
            'table_usage': {},
            'common_joins': [],
            'parameters': rdl_info.get('parameters', []),
            'reports_summary': rdl_info.get('reports', [])
        }
        
        # Build table usage statistics
        for table_name in rdl_info.get('referenced_tables', []):
            # Count how many reports reference this table
            usage_count = sum(1 for report in rdl_info.get('reports', []) 
                            if any(dataset.get('referenced_tables', []) and table_name in dataset.get('referenced_tables', []) 
                                   for dataset in rdl_info.get('datasets', [])))
            
            usage_data['table_usage'][table_name] = {
                'usage_count': max(1, usage_count),
                'business_priority': 'high' if 'executive_report' in rdl_info.get('business_priority_signals', []) else 'medium',
                'report_context': f"{rdl_info.get('report_count', 0)} reports"
            }
        
        return usage_data

class EnhancedSampleCollector:
    """Enhanced sample collection - First 3 + Last 3 rows (Architecture requirement)"""
    
    def __init__(self, connector: SqlServerConnector):
        self.connector = connector
    
    def collect_samples(self, table_info: TableInfo) -> List[Dict[str, Any]]:
        """Collect first 3 + last 3 sample rows with intelligent ordering"""
        try:
            # Get best ordering column
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
                
                # Combine with position metadata
                samples = []
                
                for i, row in enumerate(first_3, 1):
                    row['__sample_position__'] = f'first_{i}'
                    samples.append(row)
                
                for i, row in enumerate(reversed(last_3), 1):
                    row['__sample_position__'] = f'last_{i}'
                    samples.append(row)
                
                print(f"   📋 Collected first 3 + last 3 samples for {table_info.name}")
                return samples
            
            else:
                # Fallback: just get first 6 rows
                fallback_sql = f"SELECT TOP (6) * FROM {table_info.full_name}"
                samples = self.connector.execute_query(fallback_sql)
                
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
        
        # Look for primary key or ID columns first
        for col in table_info.columns:
            col_name = col.get('name', '').lower()
            if col.get('is_identity') or col_name in ['id', 'pk'] or col_name.endswith('id'):
                return col.get('name')
        
        # Look for date/time columns
        for col in table_info.columns:
            col_type = col.get('data_type', '').lower()
            col_name = col.get('name', '').lower()
            if any(t in col_type for t in ['date', 'time']) or any(w in col_name for w in ['date', 'created', 'modified']):
                return col.get('name')
        
        # Use first column as fallback
        return table_info.columns[0].get('name') if table_info.columns else None

class EnhancedCacheManager:
    """Enhanced cache with Multi-RDL integration"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_cache(self, tables: List[TableInfo], view_info: Dict, rdl_info: Dict):
        """Save enhanced discovery with Multi-RDL data"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        data = {
            'metadata': {
                'discovered': datetime.now().isoformat(),
                'version': '3.0-sql-server-multi-rdl',
                'sampling_method': 'first_3_plus_last_3',
                'sql_server_metadata': True,
                'multi_rdl_integrated': len(rdl_info) > 0,
                'sqlglot_available': HAS_SQLGLOT
            },
            'discovery_summary': {
                'total_tables': len(tables),
                'total_views': len(view_info),
                'rdl_reports_count': rdl_info.get('report_count', 0),
                'rdl_referenced_tables': len(rdl_info.get('referenced_tables', [])) if rdl_info else 0
            },
            'tables': [self._table_to_dict(t) for t in tables],
            'views': view_info,
            'rdl_info': rdl_info
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Enhanced Multi-RDL discovery cached: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def load_cache(self) -> Tuple[List[TableInfo], Dict, Dict]:
        """Load enhanced discovery results"""
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
            view_info = data.get('views', {})
            rdl_info = data.get('rdl_info', {})
            
            return tables, view_info, rdl_info
            
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
    """Enhanced database discovery with SQL Server sys.* and Multi-RDL integration"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.connector = SqlServerConnector(config)
        self.metadata = SqlServerMetadata(self.connector)
        self.rdl_parser = MultiRDLParser()
        self.sample_collector = EnhancedSampleCollector(self.connector)
        self.cache_manager = EnhancedCacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.view_info: Dict = {}
        self.rdl_info: Dict = {}
        self.relationships: List[Relationship] = []
    
    async def discover_database(self) -> bool:
        """Enhanced discovery with SQL Server sys.* and Multi-RDL integration"""
        print("🔍 ENHANCED SQL SERVER DISCOVERY + MULTI-RDL INTEGRATION")
        print("Architecture: sys.* metadata + sqlglot + Multi-RDL parsing")
        print("=" * 65)
        
        # Check cache first
        cached_tables, cached_views, cached_rdl = self.cache_manager.load_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.rdl_info = cached_rdl
            print(f"✅ Loaded from cache: {len(self.tables)} tables, {len(self.view_info)} views")
            print(f"   📋 RDL reports: {self.rdl_info.get('report_count', 0)}")
            return True
        
        try:
            start_time = time.time()
            
            # Step 1: Parse all RDL files for business insights
            self.rdl_info = self.rdl_parser.parse_all_rdl_files()
            
            # Step 2: Discover tables with SQL Server metadata
            await self._discover_tables_with_metadata()
            
            # Step 3: Analyze views with definitions
            self.view_info = self.metadata.get_views_with_definitions()
            
            # Step 4: Build relationships from foreign keys
            self._build_relationships_from_foreign_keys()
            
            # Step 5: Apply Multi-RDL insights for business priority
            self._apply_multi_rdl_insights()
            
            # Step 6: Save enhanced cache
            self.cache_manager.save_cache(self.tables, self.view_info, self.rdl_info)
            
            # Show summary
            self._show_discovery_summary(time.time() - start_time)
            return True
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    async def _discover_tables_with_metadata(self):
        """Discover tables using SQL Server sys.* metadata"""
        print("📊 Discovering tables with sys.* metadata...")
        
        # Get tables with row counts
        table_results = self.metadata.get_tables_with_row_counts()
        exclusion_patterns = self.config.get_exclusion_patterns()
        
        for row in table_results:
            schema = row['schema_name']
            name = row['table_name']
            
            # Apply exclusions
            if should_exclude_table(name, schema, exclusion_patterns):
                continue
            
            try:
                # Get detailed metadata
                columns = self.metadata.get_table_columns(schema, name)
                primary_keys = self.metadata.get_primary_keys(schema, name)
                foreign_keys = self.metadata.get_foreign_keys(schema, name)
                
                if not columns:
                    continue
                
                # Mark primary key columns
                for col in columns:
                    col['is_primary_key'] = col['name'] in primary_keys
                
                # Build relationship strings
                relationships = []
                for fk in foreign_keys:
                    rel_str = f"{fk['column_name']} -> [{fk['referenced_schema']}].[{fk['referenced_table']}].{fk['referenced_column']}"
                    relationships.append(rel_str)
                
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
                
                # Collect samples if table has data
                if row['estimated_rows'] > 0:
                    samples = self.sample_collector.collect_samples(table_info)
                    table_info.sample_data = samples
                
                self.tables.append(table_info)
                
            except Exception as e:
                print(f"   ⚠️ Failed to analyze {schema}.{name}: {e}")
        
        print(f"   ✅ Discovered {len(self.tables)} tables with enhanced metadata")
    
    def _build_relationships_from_foreign_keys(self):
        """Build relationships from foreign key metadata"""
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
                            to_table=to_ref.split('.')[0] + '.' + to_ref.split('.')[1] if '.' in to_ref else to_ref,
                            relationship_type='foreign_key',
                            confidence=0.95,
                            description=f"FK: {from_col} -> {to_ref}"
                        ))
                    except Exception:
                        continue
    
    def _apply_multi_rdl_insights(self):
        """Apply Multi-RDL insights to boost business priority"""
        if not self.rdl_info:
            return
        
        rdl_tables = set(self.rdl_info.get('referenced_tables', []))
        business_signals = self.rdl_info.get('business_priority_signals', [])
        
        print(f"   📋 Applying Multi-RDL insights: {len(rdl_tables)} referenced tables across {self.rdl_info.get('report_count', 0)} reports")
        
        for table in self.tables:
            # Boost priority for tables referenced in RDL reports
            if table.full_name in rdl_tables or any(table.name.lower() in ref.lower() for ref in rdl_tables):
                table.business_priority = 'high'
                table.confidence = min(1.0, getattr(table, 'confidence', 0.5) + 0.3)
                
                # Additional boost for executive reports
                if 'executive_report' in business_signals:
                    table.business_priority = 'high'
                    print(f"   📈 Boosted priority: {table.name} (Multi-RDL reference)")
    
    def _show_discovery_summary(self, elapsed_time: float):
        """Show enhanced discovery summary"""
        print(f"\n📊 SQL SERVER DISCOVERY WITH MULTI-RDL COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Tables: {len(self.tables)}")
        print(f"   👁️ Views: {len(self.view_info)}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        print(f"   📋 RDL reports processed: {self.rdl_info.get('report_count', 0)}")
        print(f"   🔗 RDL referenced tables: {len(self.rdl_info.get('referenced_tables', []))}")
        print(f"   🧠 Metadata source: SQL Server sys.* views")
        print(f"   📝 Sampling: First 3 + Last 3 rows per table")
        print(f"   ⚙️ SQL parsing: {'✅ sqlglot available' if HAS_SQLGLOT else '❌ sqlglot missing'}")
        
        # Show Multi-RDL insights
        if self.rdl_info:
            reports = self.rdl_info.get('reports', [])
            if reports:
                print(f"   📋 RDL reports:")
                for report in reports[:3]:  # Show first 3 reports
                    print(f"      • {report.get('title', 'Unknown')} ({report.get('datasets_count', 0)} datasets)")
                if len(reports) > 3:
                    print(f"      • ... and {len(reports) - 3} more reports")
            
            priority_signals = self.rdl_info.get('business_priority_signals', [])
            if priority_signals:
                print(f"   🎯 Business signals: {', '.join(list(set(priority_signals)))}")
    
    def load_from_cache(self) -> bool:
        """Load from cache"""
        cached_tables, cached_views, cached_rdl = self.cache_manager.load_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.rdl_info = cached_rdl
            return True
        return False
    
    # Public interface
    def get_tables(self) -> List[TableInfo]:
        return self.tables
    
    def get_relationships(self) -> List[Relationship]:
        return self.relationships
    
    def get_view_info(self) -> Dict:
        return self.view_info
    
    def get_rdl_info(self) -> Dict:
        return self.rdl_info
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        return {
            'total_objects': len(self.tables),
            'tables': len([t for t in self.tables if t.object_type != 'VIEW']),
            'views': len(self.view_info),
            'relationships': len(self.relationships),
            'rdl_reports': self.rdl_info.get('report_count', 0),
            'rdl_references': len(self.rdl_info.get('referenced_tables', [])),
            'sqlglot_available': HAS_SQLGLOT,
            'sampling_method': 'first_3_plus_last_3',
            'metadata_source': 'sql_server_sys_views'
        }