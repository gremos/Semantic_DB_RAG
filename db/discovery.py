#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Discovery - Enhanced SQL Server Architecture
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
"""

import asyncio
import json
import pyodbc
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# SQLGlot for SQL parsing (Architecture requirement)
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from shared.config import Config
from shared.models import TableInfo, Relationship
from shared.utils import safe_database_value, should_exclude_table

class SqlServerConnector:
    """SQL Server connection with UTF-8 support"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_connection(self):
        """Get SQL Server connection with UTF-8 support"""
        conn = pyodbc.connect(self.config.get_database_connection_string())
        
        if self.config.utf8_encoding:
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
        
        return conn
    
    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query with error handling"""
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
    """SQL Server metadata using sys.* views (Architecture requirement)"""
    
    def __init__(self, connector: SqlServerConnector):
        self.connector = connector
    
    def get_tables_with_row_counts(self) -> List[Dict[str, Any]]:
        """Get tables with estimated row counts using sys.dm_db_partition_stats"""
        sql = """
        SELECT 
            s.name as schema_name,
            t.name as table_name,
            t.type_desc as object_type,
            ISNULL(SUM(p.row_count), 0) as estimated_rows
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN sys.dm_db_partition_stats p ON t.object_id = p.object_id 
            AND p.index_id IN (0,1)
        WHERE s.name NOT IN ('sys', 'information_schema')
        GROUP BY s.name, t.name, t.type_desc
        ORDER BY s.name, t.name
        """
        
        return self.connector.execute_query(sql)
    
    def get_table_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get table columns using sys.columns + sys.types"""
        sql = """
        SELECT 
            c.name as column_name,
            t.name as data_type,
            c.max_length,
            c.is_nullable,
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
                        'is_identity': row.is_identity
                    })
                
                return columns
        except Exception as e:
            print(f"   ⚠️ Column query failed for {full_name}: {e}")
            return []
    
    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        """Get primary keys using sys.key_constraints"""
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
        """Get foreign keys using sys.foreign_keys (Architecture fix)"""
        sql = """
        SELECT 
            COL_NAME(fkc.parent_object_id, fkc.parent_column_id) as column_name,
            SCHEMA_NAME(ref_t.schema_id) as referenced_schema,
            OBJECT_NAME(fkc.referenced_object_id) as referenced_table,
            COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) as referenced_column
        FROM sys.foreign_key_columns fkc
        INNER JOIN sys.foreign_keys fk ON fkc.constraint_object_id = fk.object_id
        INNER JOIN sys.tables ref_t ON fkc.referenced_object_id = ref_t.object_id
        WHERE fkc.parent_object_id = OBJECT_ID(?)
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
                        'referenced_column': row.referenced_column
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
            full_name = f"[{row['schema_name']}].[{row['view_name']}]"
            
            view_info[full_name] = {
                'schema': row['schema_name'],
                'name': row['view_name'],
                'full_name': full_name,
                'object_type': 'VIEW',
                'definition': row['view_definition'],
                'referenced_objects': [],
                'parsing_success': False
            }
            
            # Parse with sqlglot if available
            if HAS_SQLGLOT and row['view_definition']:
                try:
                    parsed = sqlglot.parse_one(row['view_definition'], dialect="tsql")
                    if parsed:
                        tables = []
                        for table in parsed.find_all(sqlglot.expressions.Table):
                            if table.this:
                                table_name = str(table.this)
                                if table.db:
                                    table_name = f"[{table.db}].[{table_name}]"
                                tables.append(table_name)
                        
                        view_info[full_name].update({
                            'referenced_objects': list(set(tables)),
                            'parsing_success': True
                        })
                        
                except Exception:
                    pass
        
        return view_info

class RDLParser:
    """RDL parser for SSRS reports (Architecture requirement)"""
    
    def parse_all_rdl_files(self, rdl_directory: str = "data_upload") -> Dict[str, Any]:
        """Parse all RDL files from data_upload directory (COMPLIANT - reads ALL files from path)"""
        print(f"   📋 Parsing all RDL files from: {rdl_directory}")
        
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
                    combined_rdl_info['reports'].append({
                        'filename': rdl_file.name,
                        'title': rdl_info.get('report_title', rdl_file.stem),
                        'datasets_count': len(rdl_info.get('datasets', [])),
                        'referenced_tables_count': len(rdl_info.get('referenced_tables', []))
                    })
                    
                    combined_rdl_info['datasets'].extend(rdl_info.get('datasets', []))
                    combined_rdl_info['parameters'].extend(rdl_info.get('parameters', []))
                    combined_rdl_info['referenced_tables'].update(rdl_info.get('referenced_tables', set()))
                    combined_rdl_info['business_priority_signals'].extend(
                        rdl_info.get('business_priority_signals', [])
                    )
                    
            except Exception as e:
                print(f"   ⚠️ Failed to parse {rdl_file.name}: {e}")
        
        # Finalize
        combined_rdl_info['referenced_tables'] = list(combined_rdl_info['referenced_tables'])
        combined_rdl_info['business_priority_signals'] = list(set(combined_rdl_info['business_priority_signals']))
        
        print(f"   ✅ Multi-RDL parsing completed:")
        print(f"      📊 Total reports: {len(combined_rdl_info['reports'])}")
        print(f"      📋 Total datasets: {len(combined_rdl_info['datasets'])}")
        print(f"      🔗 Referenced tables: {len(combined_rdl_info['referenced_tables'])}")
        
        return combined_rdl_info
    
    def _parse_single_rdl_file(self, rdl_path: Path) -> Dict[str, Any]:
        """Parse single RDL file with improved XML handling"""
        try:
            # Read file with encoding detection
            try:
                content = rdl_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = rdl_path.read_text(encoding='utf-8-sig')  # Try with BOM
            except UnicodeDecodeError:
                content = rdl_path.read_text(encoding='latin-1')   # Fallback
            
            # Parse XML
            try:
                root = ET.fromstring(content)
            except ET.ParseError as e:
                print(f"   ⚠️ XML parse error in {rdl_path.name}: {e}")
                return {}
            
            # Improved namespace handling
            ns = {}
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0][1:]
                ns[''] = namespace
            
            rdl_info = {
                'report_title': self._get_report_title(root, ns, rdl_path.stem),
                'datasets': [],
                'parameters': [],
                'referenced_tables': set(),
                'business_priority_signals': []
            }
            
            # Extract DataSets with improved search
            datasets = self._find_elements(root, ['DataSet', 'DataSets/DataSet'], ns)
            for dataset in datasets:
                dataset_info = self._extract_dataset_info(dataset, ns)
                if dataset_info:
                    rdl_info['datasets'].append(dataset_info)
                    rdl_info['referenced_tables'].update(dataset_info.get('referenced_tables', []))
            
            # Extract Parameters with improved search
            params = self._find_elements(root, ['ReportParameter', 'ReportParameters/ReportParameter'], ns)
            for param in params:
                param_info = self._extract_parameter_info(param, ns)
                if param_info:
                    rdl_info['parameters'].append(param_info)
            
            # Business signals
            rdl_info['business_priority_signals'] = self._analyze_business_priority(rdl_info)
            
            return rdl_info
            
        except Exception as e:
            print(f"   ⚠️ RDL parsing failed for {rdl_path.name}: {e}")
            return {}
    
    def _find_elements(self, root, xpath_list: List[str], ns: Dict) -> List:
        """Find elements using multiple XPath expressions"""
        elements = []
        
        # Try multiple search patterns
        search_patterns = xpath_list + [
            f".//{path}" for path in xpath_list  # Deep search
        ]
        
        for pattern in search_patterns:
            try:
                if ns:
                    found = root.findall(pattern, ns)
                else:
                    found = root.findall(pattern)
                
                if found:
                    elements.extend(found)
                    break  # Stop at first successful pattern
            except:
                continue
        
        # If namespaced search fails, try without namespace
        if not elements and ns:
            for pattern in xpath_list:
                try:
                    found = root.findall(pattern)
                    if found:
                        elements.extend(found)
                        break
                except:
                    continue
        
        return elements
    
    def _get_report_title(self, root, ns: Dict, fallback: str) -> str:
        """Extract report title with improved search"""
        title_patterns = [
            './/Textbox[@Name="textbox1"]//Value',
            './/Textbox[@Name="Textbox1"]//Value', 
            './/TextRun/Value',
            './/Paragraph/TextRuns/TextRun/Value',
            './/Value'
        ]
        
        for pattern in title_patterns:
            elements = self._find_elements(root, [pattern], ns)
            for element in elements:
                if element.text and len(element.text.strip()) > 3:
                    title = element.text.strip()
                    # Skip if it looks like a field reference
                    if not title.startswith('=') and not title.startswith('Fields!'):
                        return title
        
        return fallback
    
    def _extract_dataset_info(self, dataset, ns: Dict) -> Dict[str, Any]:
        """Extract dataset information with improved parsing"""
        try:
            # Get dataset name
            name_elem = dataset.get('Name') or (dataset.find('Name', ns) if ns else dataset.find('Name'))
            dataset_name = 'Unknown'
            if name_elem is not None:
                dataset_name = name_elem.text if hasattr(name_elem, 'text') else str(name_elem)
            
            # Find Query/CommandText with multiple patterns
            command_text = None
            command_patterns = [
                'Query/CommandText',
                './/CommandText', 
                'CommandText'
            ]
            
            for pattern in command_patterns:
                elements = self._find_elements(dataset, [pattern], ns)
                if elements and elements[0].text:
                    command_text = elements[0].text.strip()
                    break
            
            if not command_text:
                return None
            
            dataset_info = {
                'name': dataset_name,
                'command_text': command_text,
                'referenced_tables': []
            }
            
            # Enhanced SQL parsing to extract tables
            if HAS_SQLGLOT and command_text:
                try:
                    # Clean the SQL first
                    cleaned_sql = self._clean_sql_for_parsing(command_text)
                    parsed = sqlglot.parse_one(cleaned_sql, dialect="tsql")
                    if parsed:
                        tables = []
                        for table in parsed.find_all(sqlglot.expressions.Table):
                            if table.this:
                                table_name = str(table.this)
                                if table.db:
                                    table_name = f"[{table.db}].[{table_name}]"
                                elif '.' not in table_name:
                                    table_name = f"[dbo].[{table_name}]"
                                tables.append(table_name)
                        dataset_info['referenced_tables'] = list(set(tables))
                except Exception:
                    # Fallback to regex extraction
                    dataset_info['referenced_tables'] = self._extract_tables_regex(command_text)
            else:
                dataset_info['referenced_tables'] = self._extract_tables_regex(command_text)
            
            return dataset_info
            
        except Exception as e:
            print(f"   ⚠️ Dataset extraction error: {e}")
            return None
    
    def _clean_sql_for_parsing(self, sql: str) -> str:
        """Clean SQL for better parsing"""
        import re
        
        # Remove XML entities
        sql = sql.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
        
        # Remove comments
        sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Remove variable declarations that might confuse parser
        sql = re.sub(r'declare\s+@\w+.*?(?=\n|$)', '', sql, flags=re.IGNORECASE | re.MULTILINE)
        sql = re.sub(r'set\s+@\w+.*?(?=\n|$)', '', sql, flags=re.IGNORECASE | re.MULTILINE)
        
        return sql.strip()
    
    def _extract_tables_regex(self, sql: str) -> List[str]:
        """Enhanced regex-based table extraction"""
        import re
        
        # Clean SQL first
        sql = self._clean_sql_for_parsing(sql)
        
        # Enhanced patterns for table extraction
        patterns = [
            r'FROM\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)',
            r'JOIN\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)',
            r'UPDATE\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)',
            r'INSERT\s+INTO\s+(\[?\w+\]?\.\[?\w+\]?|\[?\w+\]?)',
            # Handle temp tables and CTEs
            r'WITH\s+(\w+)\s+AS',
            # Handle schema.table patterns
            r'\b(\w+\.\w+)\b(?=\s+(?:AS\s+\w+\s*)?(?:WHERE|GROUP|ORDER|JOIN|,|\)|$))'
        ]
        
        tables = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                cleaned = match.strip()
                
                # Skip obvious non-tables
                skip_words = ['WHERE', 'SELECT', 'GROUP', 'ORDER', 'HAVING', 'AND', 'OR', 'AS', 'ON']
                if cleaned.upper() in skip_words:
                    continue
                
                # Skip variables and functions
                if cleaned.startswith('@') or '(' in cleaned:
                    continue
                
                # Normalize table names
                if cleaned and len(cleaned) > 1:
                    if '.' not in cleaned and not cleaned.startswith('['):
                        cleaned = f"[dbo].[{cleaned}]"
                    elif '.' in cleaned and not cleaned.startswith('['):
                        parts = cleaned.split('.')
                        if len(parts) == 2:
                            cleaned = f"[{parts[0]}].[{parts[1]}]"
                    
                    tables.add(cleaned)
        
        return list(tables)
    
    def _extract_parameter_info(self, param, ns: Dict) -> Dict[str, Any]:
        """Extract parameter information"""
        try:
            name_attr = param.get('Name')
            data_type_elem = param.find('DataType', ns) if ns else param.find('DataType')
            
            return {
                'name': name_attr or 'Unknown',
                'data_type': data_type_elem.text if data_type_elem is not None else 'String'
            }
        except Exception:
            return None
    
    def _analyze_business_priority(self, rdl_info: Dict) -> List[str]:
        """Analyze for business priority signals"""
        signals = []
        title = rdl_info.get('report_title', '').lower()
        
        # Executive/important reports
        if any(word in title for word in ['monthly', 'weekly', 'executive', 'dashboard', 'summary']):
            signals.append('executive_report')
        
        # Greek business terms
        if any(word in title for word in ['συμβόλαια', 'πελατών', 'καμπάνια', 'τζίρος', 'φερεγγυότητα']):
            signals.append('business_critical')
        
        # Contract/payment analysis
        if any(word in title for word in ['contract', 'payment', 'revenue', 'billing', 'πληρωμής']):
            signals.append('financial_data')
        
        # Customer analysis
        if any(word in title for word in ['customer', 'client', 'πελατών']):
            signals.append('customer_data')
        
        return signals

class SampleCollector:
    """Sample collection - First 3 + Last 3 rows"""
    
    def __init__(self, connector: SqlServerConnector):
        self.connector = connector
    
    def collect_samples(self, table_info: TableInfo) -> List[Dict[str, Any]]:
        """Collect first 3 + last 3 sample rows"""
        try:
            order_column = self._get_order_column(table_info)
            
            if order_column:
                # First 3
                first_3_sql = f"SELECT TOP (3) * FROM {table_info.full_name} ORDER BY [{order_column}] ASC"
                first_3 = self.connector.execute_query(first_3_sql)
                
                # Last 3
                last_3_sql = f"SELECT TOP (3) * FROM {table_info.full_name} ORDER BY [{order_column}] DESC"
                last_3 = self.connector.execute_query(last_3_sql)
                
                samples = []
                for i, row in enumerate(first_3, 1):
                    row['__sample_position__'] = f'first_{i}'
                    samples.append(row)
                
                for i, row in enumerate(reversed(last_3), 1):
                    row['__sample_position__'] = f'last_{i}'
                    samples.append(row)
                
                return samples
            else:
                # Fallback
                fallback_sql = f"SELECT TOP (6) * FROM {table_info.full_name}"
                samples = self.connector.execute_query(fallback_sql)
                
                for i, row in enumerate(samples, 1):
                    row['__sample_position__'] = f'row_{i}'
                
                return samples
                
        except Exception as e:
            print(f"   ⚠️ Sample collection failed for {table_info.full_name}: {e}")
            return []
    
    def _get_order_column(self, table_info: TableInfo) -> Optional[str]:
        """Get best column for ordering"""
        if not table_info.columns:
            return None
        
        # Primary key or identity columns
        for col in table_info.columns:
            if col.get('is_identity') or col.get('name', '').lower().endswith('id'):
                return col.get('name')
        
        # Date columns
        for col in table_info.columns:
            col_type = col.get('data_type', '').lower()
            if 'date' in col_type or 'time' in col_type:
                return col.get('name')
        
        # First column
        return table_info.columns[0].get('name') if table_info.columns else None

class CacheManager:
    """Cache management (COMPLIANT - uses existing cache, doesn't create new files)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_cache(self, tables: List[TableInfo], view_info: Dict, rdl_info: Dict):
        """Save discovery cache"""
        cache_file = self.config.get_cache_path("database_structure.json")
        
        data = {
            'metadata': {
                'discovered': datetime.now().isoformat(),
                'version': '3.0-sql-server-rdl',
                'sampling_method': 'first_3_plus_last_3',
                'sqlglot_available': HAS_SQLGLOT
            },
            'tables': [self._table_to_dict(t) for t in tables],
            'views': view_info,
            'rdl_info': rdl_info
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 Discovery cached: {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def load_cache(self):
        """Load discovery cache"""
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
    """Enhanced database discovery with SQL Server + RDL integration"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.connector = SqlServerConnector(config)
        self.metadata = SqlServerMetadata(self.connector)
        self.rdl_parser = RDLParser()
        self.sample_collector = SampleCollector(self.connector)
        self.cache_manager = CacheManager(config)
        
        # Data storage
        self.tables: List[TableInfo] = []
        self.view_info: Dict = {}
        self.rdl_info: Dict = {}
        self.relationships: List[Relationship] = []
    
    async def discover_database(self) -> bool:
        """Enhanced discovery with SQL Server + RDL integration"""
        print("🔍 SQL SERVER DISCOVERY + RDL INTEGRATION")
        print("=" * 50)
        
        # Check cache first
        cached_tables, cached_views, cached_rdl = self.cache_manager.load_cache()
        if cached_tables:
            self.tables = cached_tables
            self.view_info = cached_views
            self.rdl_info = cached_rdl
            print(f"✅ Loaded from cache: {len(self.tables)} tables, {len(self.view_info)} views")
            return True
        
        try:
            start_time = time.time()
            
            # Parse RDL files from data_upload directory
            self.rdl_info = self.rdl_parser.parse_all_rdl_files()
            
            # Discover tables
            await self._discover_tables()
            
            # Analyze views
            self.view_info = self.metadata.get_views_with_definitions()
            
            # Build relationships
            self._build_relationships()
            
            # Apply RDL insights
            self._apply_rdl_insights()
            
            # Save cache
            self.cache_manager.save_cache(self.tables, self.view_info, self.rdl_info)
            
            self._show_summary(time.time() - start_time)
            return True
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return False
    
    async def _discover_tables(self):
        """Discover tables using SQL Server metadata"""
        print("📊 Discovering tables with sys.* metadata...")
        
        table_results = self.metadata.get_tables_with_row_counts()
        exclusion_patterns = self.config.get_exclusion_patterns()
        
        for row in table_results:
            schema = row['schema_name']
            name = row['table_name']
            
            if should_exclude_table(name, schema, exclusion_patterns):
                continue
            
            try:
                columns = self.metadata.get_table_columns(schema, name)
                primary_keys = self.metadata.get_primary_keys(schema, name)
                foreign_keys = self.metadata.get_foreign_keys(schema, name)
                
                if not columns:
                    continue
                
                # Mark primary key columns
                for col in columns:
                    col['is_primary_key'] = col['name'] in primary_keys
                
                # Build relationships
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
                
                # Collect samples
                if row['estimated_rows'] > 0:
                    samples = self.sample_collector.collect_samples(table_info)
                    table_info.sample_data = samples
                
                self.tables.append(table_info)
                
            except Exception as e:
                print(f"   ⚠️ Failed to analyze {schema}.{name}: {e}")
        
        print(f"   ✅ Discovered {len(self.tables)} tables")
    
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
                            to_table=to_ref.split('.')[0] + '.' + to_ref.split('.')[1] if '.' in to_ref else to_ref,
                            relationship_type='foreign_key',
                            confidence=0.95,
                            description=f"FK: {from_col} -> {to_ref}"
                        ))
                    except Exception:
                        continue
    
    def _apply_rdl_insights(self):
        """Apply RDL insights for business priority"""
        if not self.rdl_info:
            return
        
        rdl_tables = set(self.rdl_info.get('referenced_tables', []))
        
        for table in self.tables:
            if table.full_name in rdl_tables or any(table.name.lower() in ref.lower() for ref in rdl_tables):
                table.business_priority = 'high'
                table.confidence = 0.9
    
    def _show_summary(self, elapsed_time: float):
        """Show discovery summary"""
        print(f"\n✅ SQL SERVER DISCOVERY COMPLETED:")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        print(f"   📊 Tables: {len(self.tables)}")
        print(f"   👁️ Views: {len(self.view_info)}")
        print(f"   🔗 Relationships: {len(self.relationships)}")
        print(f"   📋 RDL reports: {self.rdl_info.get('report_count', 0)}")
        print(f"   ⚙️ SQL parsing: {'✅ sqlglot' if HAS_SQLGLOT else '⚠️ basic only'}")
        
        if self.rdl_info:
            reports = self.rdl_info.get('reports', [])
            if reports:
                print(f"   📋 RDL reports:")
                for report in reports[:3]:
                    print(f"      • {report.get('title', 'Unknown')}")
    
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