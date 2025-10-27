"""
Database introspection using SQLAlchemy.
Discovers schemas, tables, columns, keys, indexes.
"""

import re
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, inspect, MetaData, text
from sqlalchemy.engine import Engine, Inspector
from sqlalchemy.exc import SQLAlchemyError

from config.settings import Settings


class DatabaseIntrospector:
    """Introspect database schema using SQLAlchemy."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine: Optional[Engine] = None
        self.inspector: Optional[Inspector] = None
        
    def connect(self) -> None:
        """Establish database connection with read-only enforcement."""
        try:
            # Create engine with read-only connection
            connect_args = {
                'timeout': self.settings.DISCOVERY_TIMEOUT,
                'readonly': True  # Enforce read-only at driver level if supported
            }
            
            self.engine = create_engine(
                self.settings.DATABASE_CONNECTION_STRING,
                connect_args=connect_args,
                pool_pre_ping=True,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.inspector = inspect(self.engine)
            
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.inspector = None
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database vendor and version information."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        try:
            with self.engine.connect() as conn:
                # Get database vendor
                vendor = self.engine.dialect.name
                
                # Try to get version
                version = None
                if vendor == 'mssql':
                    result = conn.execute(text("SELECT @@VERSION"))
                    version_str = result.scalar()
                    # Extract version number from string
                    match = re.search(r'Microsoft SQL Server (\d+)', version_str)
                    if match:
                        version = match.group(1) + ".0"
                elif vendor == 'postgresql':
                    result = conn.execute(text("SELECT version()"))
                    version = result.scalar()
                elif vendor == 'mysql':
                    result = conn.execute(text("SELECT VERSION()"))
                    version = result.scalar()
                
                return {
                    "vendor": vendor,
                    "version": version or "unknown"
                }
        except SQLAlchemyError as e:
            return {
                "vendor": self.engine.dialect.name,
                "version": "unknown"
            }
    
    def should_exclude_schema(self, schema_name: str) -> bool:
        """Check if schema should be excluded."""
        exclusions = self.settings.schema_exclusions_list
        return schema_name.lower() in [e.lower() for e in exclusions]
    
    def should_exclude_table(self, table_name: str) -> bool:
        """Check if table should be excluded based on prefixes and patterns."""
        # Check prefix exclusions
        prefixes = self.settings.table_exclusions_list
        for prefix in prefixes:
            if table_name.lower().startswith(prefix.lower()):
                return True
        
        # Check regex patterns
        patterns = self.settings.table_exclusion_patterns_list
        for pattern in patterns:
            if re.match(pattern, table_name, re.IGNORECASE):
                return True
        
        return False
    
    def get_schemas(self) -> List[str]:
        """Get list of schemas, excluding system schemas."""
        if not self.inspector:
            raise RuntimeError("Not connected to database")
        
        schemas = self.inspector.get_schema_names()
        return [s for s in schemas if not self.should_exclude_schema(s)]
    
    def get_tables(self, schema: str) -> List[Dict[str, Any]]:
        """Get tables and views in a schema with metadata."""
        if not self.inspector:
            raise RuntimeError("Not connected to database")
        
        tables = []
        
        # Get regular tables
        for table_name in self.inspector.get_table_names(schema=schema):
            if self.should_exclude_table(table_name):
                continue
            
            table_info = {
                "name": table_name,
                "type": "table",
                "schema": schema,
                "full_name": f"{schema}.{table_name}"
            }
            tables.append(table_info)
        
        # Get views
        for view_name in self.inspector.get_view_names(schema=schema):
            if self.should_exclude_table(view_name):
                continue
            
            view_info = {
                "name": view_name,
                "type": "view",
                "schema": schema,
                "full_name": f"{schema}.{view_name}"
            }
            tables.append(view_info)
        
        return tables
    
    def get_columns(self, table_name: str, schema: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        if not self.inspector:
            raise RuntimeError("Not connected to database")
        
        columns = []
        for col in self.inspector.get_columns(table_name, schema=schema):
            column_info = {
                "name": col['name'],
                "type": str(col['type']),
                "nullable": col['nullable'],
                "default": str(col.get('default')) if col.get('default') else None,
                "autoincrement": col.get('autoincrement', False)
            }
            columns.append(column_info)
        
        return columns
    
    def get_primary_key(self, table_name: str, schema: str) -> List[str]:
        """Get primary key columns for a table."""
        if not self.inspector:
            raise RuntimeError("Not connected to database")
        
        pk = self.inspector.get_pk_constraint(table_name, schema=schema)
        return pk.get('constrained_columns', [])
    
    def get_foreign_keys(self, table_name: str, schema: str) -> List[Dict[str, Any]]:
        """Get foreign key constraints for a table."""
        if not self.inspector:
            raise RuntimeError("Not connected to database")
        
        fks = []
        for fk in self.inspector.get_foreign_keys(table_name, schema=schema):
            fk_info = {
                "column": fk['constrained_columns'][0] if fk['constrained_columns'] else None,
                "ref_table": f"{fk['referred_schema']}.{fk['referred_table']}" if fk.get('referred_schema') else fk['referred_table'],
                "ref_column": fk['referred_columns'][0] if fk['referred_columns'] else None,
                "name": fk.get('name')
            }
            fks.append(fk_info)
        
        return fks
    
    def get_indexes(self, table_name: str, schema: str) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        if not self.inspector:
            raise RuntimeError("Not connected to database")
        
        indexes = []
        for idx in self.inspector.get_indexes(table_name, schema=schema):
            index_info = {
                "name": idx['name'],
                "columns": idx['column_names'],
                "unique": idx.get('unique', False)
            }
            indexes.append(index_info)
        
        return indexes
    
    def get_row_count(self, table_name: str, schema: str, limit: int = 1000) -> int:
        """
        Get approximate row count for a table.
        Uses metadata statistics if available, otherwise does actual count with limit.
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        try:
            # Try fast metadata approach first (MSSQL)
            if self.engine.dialect.name == 'mssql':
                with self.engine.connect() as conn:
                    query = text(f"""
                        SELECT SUM(p.rows) as row_count
                        FROM sys.partitions p
                        INNER JOIN sys.tables t ON p.object_id = t.object_id
                        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                        WHERE t.name = :table_name 
                        AND s.name = :schema_name
                        AND p.index_id IN (0, 1)
                    """)
                    result = conn.execute(query, {"table_name": table_name, "schema_name": schema})
                    count = result.scalar()
                    if count is not None:
                        return int(count)
            
            # Fallback: actual count with limit
            with self.engine.connect() as conn:
                query = text(f'SELECT COUNT(*) FROM (SELECT TOP {limit} 1 FROM "{schema}"."{table_name}") AS limited')
                result = conn.execute(query)
                return result.scalar() or 0
                
        except SQLAlchemyError:
            return 0
    
    def introspect_full(self) -> Dict[str, Any]:
        """
        Perform full database introspection.
        Returns discovery structure with all metadata.
        """
        if not self.inspector:
            self.connect()
        
        db_info = self.get_database_info()
        schemas_data = []
        
        for schema_name in self.get_schemas():
            tables_data = []
            
            for table_info in self.get_tables(schema_name):
                table_name = table_info['name']
                
                # Get all metadata
                columns = self.get_columns(table_name, schema_name)
                primary_key = self.get_primary_key(table_name, schema_name)
                foreign_keys = self.get_foreign_keys(table_name, schema_name)
                indexes = self.get_indexes(table_name, schema_name)
                row_count = self.get_row_count(table_name, schema_name)
                
                table_data = {
                    "name": table_name,
                    "type": table_info['type'],
                    "columns": columns,
                    "primary_key": primary_key,
                    "foreign_keys": foreign_keys,
                    "indexes": indexes,
                    "rowcount_sample": row_count,
                    "sample_rows": [],  # Will be populated by sampler
                    "source_assets": []
                }
                
                tables_data.append(table_data)
            
            schema_data = {
                "name": schema_name,
                "tables": tables_data
            }
            schemas_data.append(schema_data)
        
        return {
            "database": db_info,
            "dialect": db_info['vendor'],
            "schemas": schemas_data
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
