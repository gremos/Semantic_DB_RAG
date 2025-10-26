"""
Database Introspector - Phase 1: Discover database schema, views, stored procedures, and RDLs
"""

import logging
import re
import os
import glob
from typing import List, Dict, Optional
import sqlglot
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from sqlalchemy import types, create_engine
from sqlalchemy.dialects import mssql
from sqlalchemy.exc import SAWarning
import warnings
import logging

logger = logging.getLogger(__name__)


class TableID(types.TypeDecorator):
    """SQL Server TableID user-defined type (typically wraps INT)"""
    impl = types.Integer
    cache_ok = True

class MoneyValue(types.TypeDecorator):
    """SQL Server MoneyValue UDT - wraps DECIMAL for currency"""
    impl = types.DECIMAL(19, 4)
    cache_ok = True

class PercentageValue(types.TypeDecorator):
    """SQL Server PercentageValue UDT - wraps DECIMAL for percentages"""
    impl = types.DECIMAL(5, 2)
    cache_ok = True

mssql.base.ischema_names['TableID'] = TableID
mssql.base.ischema_names['MoneyValue'] = MoneyValue
mssql.base.ischema_names['PercentageValue'] = PercentageValue

class DatabaseIntrospector:
    """
    Introspects database to extract:
    1. Schema metadata (tables, columns, types, keys, FKs)
    2. Views and their SQL definitions
    3. Stored procedures
    4. RDL files from /data_upload directory
    """
    
    def __init__(
        self,
        connection_string: str,
        schema_exclusions: List[str] = None,
        table_exclusions: List[str] = None,
        rdl_directory: str = "data_upload/Reports"
    ):
        self.connection_string = connection_string
        self.schema_exclusions = schema_exclusions or ["sys", "information_schema", "INFORMATION_SCHEMA"]
        self.table_exclusions = table_exclusions or ["temp_", "test_", "backup_", "old_"]
        self.rdl_directory = rdl_directory
        
        # Create SQLAlchemy engine
        self.engine = create_engine(connection_string)
        
        # Detect database vendor
        self.vendor = self._detect_vendor()
        self.dialect = self._map_vendor_to_sqlglot_dialect(self.vendor)
        
        logger.info(f"Initialized introspector for {self.vendor} database")
    
    def introspect(self) -> dict:
        """
        Main entry point: perform complete database introspection.
        
        Returns: Discovery JSON
        """
        logger.info("Starting database introspection...")
        
        discovery = {
            "database": {
                "vendor": self.vendor,
                "version": self._get_db_version()
            },
            "dialect": self.dialect,
            "schemas": [],
            "named_assets": []
        }
        
        # 1. Introspect schemas and tables
        logger.info("Introspecting schemas and tables...")
        discovery["schemas"] = self._introspect_schemas()
        
        # 2. Extract views
        logger.info("Extracting views...")
        views = self._extract_views()
        discovery["named_assets"].extend(views)
        
        # 3. Extract stored procedures
        logger.info("Extracting stored procedures...")
        stored_procs = self._extract_stored_procedures()
        discovery["named_assets"].extend(stored_procs)
        
        # 4. Extract RDL files
        logger.info("Extracting RDL files...")
        rdls = self._extract_rdl_files()
        discovery["named_assets"].extend(rdls)
        
        # 5. Link views/SPs/RDLs back to tables
        logger.info("Linking assets to tables...")
        self._link_assets_to_tables(discovery)
        
        logger.info(f"Introspection complete:")
        logger.info(f"  - Schemas: {len(discovery['schemas'])}")
        logger.info(f"  - Tables/Views: {sum(len(s['tables']) for s in discovery['schemas'])}")
        logger.info(f"  - Named assets: {len(discovery['named_assets'])}")
        
        return discovery
    
    def _detect_vendor(self) -> str:
        """Detect database vendor from connection string."""
        conn_str_lower = self.connection_string.lower()
        
        if "mssql" in conn_str_lower or "sqlserver" in conn_str_lower:
            return "mssql"
        elif "postgres" in conn_str_lower or "postgresql" in conn_str_lower:
            return "postgres"
        elif "mysql" in conn_str_lower:
            return "mysql"
        elif "oracle" in conn_str_lower:
            return "oracle"
        else:
            return "unknown"
    
    def _map_vendor_to_sqlglot_dialect(self, vendor: str) -> str:
        """Map vendor to sqlglot dialect name."""
        mapping = {
            "mssql": "tsql",
            "postgres": "postgres",
            "mysql": "mysql",
            "oracle": "oracle"
        }
        return mapping.get(vendor, "")
    
    def _get_db_version(self) -> str:
        """Get database version."""
        try:
            with self.engine.connect() as conn:
                if self.vendor == "mssql":
                    result = conn.execute(text("SELECT @@VERSION"))
                    version = result.fetchone()[0]
                    return version.split("\n")[0]  # First line
                elif self.vendor == "postgres":
                    result = conn.execute(text("SELECT version()"))
                    return result.fetchone()[0]
                else:
                    return "unknown"
        except Exception as e:
            logger.warning(f"Could not get database version: {e}")
            return "unknown"
    
    def _introspect_schemas(self) -> List[dict]:
        """Introspect all schemas and their tables."""
        inspector = inspect(self.engine)
        schemas = []
        
        for schema_name in inspector.get_schema_names():
            # Skip excluded schemas
            if schema_name in self.schema_exclusions:
                continue
            
            schema_data = {
                "name": schema_name,
                "tables": []
            }
            
            # Get tables in schema
            for table_name in inspector.get_table_names(schema=schema_name):
                # Skip excluded tables
                if any(table_name.startswith(prefix) for prefix in self.table_exclusions):
                    continue
                
                table_data = self._introspect_table(schema_name, table_name, "table")
                if table_data:
                    schema_data["tables"].append(table_data)
            
            # Get views in schema
            for view_name in inspector.get_view_names(schema=schema_name):
                if any(view_name.startswith(prefix) for prefix in self.table_exclusions):
                    continue
                
                view_data = self._introspect_table(schema_name, view_name, "view")
                if view_data:
                    schema_data["tables"].append(view_data)
            
            if schema_data["tables"]:
                schemas.append(schema_data)
        
        return schemas
    
    def _introspect_table(self, schema: str, table_name: str, table_type: str) -> Optional[dict]:
        """Introspect a single table or view."""
        try:
            inspector = inspect(self.engine)
            
            # Get columns
            columns = []
            for col in inspector.get_columns(table_name, schema=schema):
                col_type = col.get('type')
                
                # Handle unrecognized types
                if col_type is None or isinstance(col_type, types.NullType):
                    # Query actual type from DB for custom types
                    actual_type = self._resolve_custom_type(
                        schema, table_name, col['name']
                    )
                    col['type'] = actual_type
                else:
                    col['type'] = str(col_type)
                
                columns.append({
                    "name": col['name'],
                    "type": col['type'],
                    "nullable": col.get('nullable', True)
                })
            
            # Get primary key
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
            primary_key = pk_constraint.get("constrained_columns", []) if pk_constraint else []
            
            # Get foreign keys
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name, schema=schema):
                for col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                    foreign_keys.append({
                        "column": col,
                        "ref_table": f"{fk.get('referred_schema', schema)}.{fk['referred_table']}",
                        "ref_column": ref_col
                    })
            
            # Get row count sample (limit to avoid performance issues)
            rowcount_sample = self._get_row_count(schema, table_name)
            
            return {
                "name": table_name,
                "type": table_type,
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "rowcount_sample": rowcount_sample,
                "source_assets": []  # Will be populated later
            }
        
        except Exception as e:
            logger.warning(f"Failed to introspect {schema}.{table_name}: {e}")
            return None
        
    def _resolve_custom_type(self, schema: str, table: str, column: str) -> str:
        """Query SQL Server for actual user-defined type"""
        query = f"""
        SELECT 
            TYPE_NAME(c.user_type_id) as base_type,
            TYPE_NAME(c.system_type_id) as system_type
        FROM sys.columns c
        JOIN sys.tables t ON c.object_id = t.object_id
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name = :schema 
        AND t.name = :table 
        AND c.name = :column
        """
        
        try:
            result = self.engine.execute(
                query, 
                {"schema": schema, "table": table, "column": column}
            ).fetchone()
            
            if result:
                # Return the system type (e.g., 'int' for TableID)
                return result['system_type'] or result['base_type']
        except:
            pass
        
        # Fallback to generic type
        return "VARCHAR(MAX)"
    
    def _get_row_count(self, schema: str, table_name: str) -> int:
        """Get approximate row count."""
        try:
            with self.engine.connect() as conn:
                # Use COUNT(*) with LIMIT for safety
                query = f'SELECT COUNT(*) FROM "{schema}"."{table_name}"'
                if self.vendor == "mssql":
                    query = f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"
                
                result = conn.execute(text(query))
                count = result.fetchone()[0]
                return count
        except Exception as e:
            logger.warning(f"Could not get row count for {schema}.{table_name}: {e}")
            return 0
    
    def _extract_views(self) -> List[dict]:
        """Extract view definitions and normalize SQL."""
        views = []
        
        try:
            with self.engine.connect() as conn:
                if self.vendor == "mssql":
                    query = """
                    SELECT 
                        SCHEMA_NAME(v.schema_id) as schema_name,
                        v.name as view_name,
                        m.definition as view_definition
                    FROM sys.views v
                    INNER JOIN sys.sql_modules m ON v.object_id = m.object_id
                    WHERE SCHEMA_NAME(v.schema_id) NOT IN ('sys', 'INFORMATION_SCHEMA')
                    """
                elif self.vendor == "postgres":
                    query = """
                    SELECT 
                        schemaname as schema_name,
                        viewname as view_name,
                        definition as view_definition
                    FROM pg_views
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    """
                else:
                    logger.warning(f"View extraction not implemented for {self.vendor}")
                    return []
                
                result = conn.execute(text(query))
                
                for row in result:
                    schema_name = row[0]
                    view_name = row[1]
                    view_def = row[2]
                    
                    # Normalize SQL using sqlglot
                    normalized_sql = self._normalize_sql(view_def)
                    
                    views.append({
                        "kind": "view",
                        "name": f"{schema_name}.{view_name}",
                        "sql_normalized": normalized_sql
                    })
        
        except Exception as e:
            logger.error(f"Failed to extract views: {e}")
        
        return views
    
    def _extract_stored_procedures(self) -> List[dict]:
        """Extract stored procedure definitions."""
        stored_procs = []
        
        try:
            with self.engine.connect() as conn:
                if self.vendor == "mssql":
                    query = """
                    SELECT 
                        SCHEMA_NAME(p.schema_id) as schema_name,
                        p.name as proc_name,
                        m.definition as proc_definition
                    FROM sys.procedures p
                    INNER JOIN sys.sql_modules m ON p.object_id = m.object_id
                    WHERE SCHEMA_NAME(p.schema_id) NOT IN ('sys')
                    """
                elif self.vendor == "postgres":
                    query = """
                    SELECT 
                        n.nspname as schema_name,
                        p.proname as proc_name,
                        pg_get_functiondef(p.oid) as proc_definition
                    FROM pg_proc p
                    JOIN pg_namespace n ON p.pronamespace = n.oid
                    WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                    AND p.prokind = 'p'
                    """
                else:
                    logger.warning(f"Stored procedure extraction not implemented for {self.vendor}")
                    return []
                
                result = conn.execute(text(query))
                
                for row in result:
                    schema_name = row[0]
                    proc_name = row[1]
                    proc_def = row[2]
                    
                    # Normalize SQL
                    normalized_sql = self._normalize_sql(proc_def)
                    
                    stored_procs.append({
                        "kind": "stored_procedure",
                        "name": f"{schema_name}.{proc_name}",
                        "sql_normalized": normalized_sql
                    })
        
        except Exception as e:
            logger.error(f"Failed to extract stored procedures: {e}")
        
        return stored_procs
    
    def _extract_rdl_files(self) -> List[dict]:
        """Extract RDL report files from data_upload directory."""
        rdls = []
        
        if not os.path.exists(self.rdl_directory):
            logger.warning(f"RDL directory not found: {self.rdl_directory}")
            return []
        
        # Find all .rdl files
        rdl_files = glob.glob(os.path.join(self.rdl_directory, "**/*.rdl"), recursive=True)
        
        for rdl_path in rdl_files:
            try:
                # Parse RDL XML to extract dataset queries
                import xml.etree.ElementTree as ET
                tree = ET.parse(rdl_path)
                root = tree.getroot()
                
                # RDL namespace
                ns = {"rdl": "http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition"}
                
                datasets = []
                for dataset in root.findall(".//rdl:DataSet", ns):
                    dataset_name = dataset.get("Name", "unknown")
                    datasets.append(dataset_name)
                
                rdls.append({
                    "kind": "rdl",
                    "path": rdl_path,
                    "datasets": datasets
                })
            
            except Exception as e:
                logger.warning(f"Failed to parse RDL {rdl_path}: {e}")
        
        logger.info(f"Found {len(rdls)} RDL files")
        return rdls
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL using sqlglot."""
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            normalized = parsed.sql(dialect=self.dialect, pretty=True)
            return normalized
        except Exception as e:
            logger.warning(f"Failed to normalize SQL: {e}")
            return sql  # Return original if normalization fails
    
    def _link_assets_to_tables(self, discovery: dict):
        """Link views, stored procedures, and RDLs back to the tables they reference."""
        
        # Build table lookup
        table_lookup = {}
        for schema in discovery["schemas"]:
            for table in schema["tables"]:
                full_name = f"{schema['name']}.{table['name']}"
                table_lookup[full_name] = table
        
        # Scan each asset for table references
        for asset in discovery["named_assets"]:
            if asset["kind"] in ["view", "stored_procedure"]:
                sql = asset.get("sql_normalized", "")
                referenced_tables = self._extract_table_references(sql)
                
                # Link back to tables
                for table_name in referenced_tables:
                    if table_name in table_lookup:
                        table_lookup[table_name]["source_assets"].append({
                            "kind": asset["kind"],
                            "name": asset["name"]
                        })
    
    def _extract_table_references(self, sql: str) -> List[str]:
        """Extract table names referenced in SQL."""
        referenced_tables = []
        
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Find all table references
            for table in parsed.find_all(sqlglot.exp.Table):
                schema = table.args.get("db", {}).get("this") if table.args.get("db") else None
                table_name = table.args.get("this", {}).get("this") if table.args.get("this") else None
                
                if schema and table_name:
                    full_name = f"{schema}.{table_name}"
                    referenced_tables.append(full_name)
                elif table_name:
                    referenced_tables.append(table_name)
        
        except Exception as e:
            logger.warning(f"Failed to extract table references: {e}")
        
        return referenced_tables