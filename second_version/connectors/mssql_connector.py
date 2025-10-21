from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, inspect, Integer, Numeric, String
from sqlalchemy.dialects import mssql
from .base import DatabaseConnector

# Register custom SQL Server types
class TableID(Integer):
    """Custom TableID type - treat as Integer"""
    __visit_name__ = 'TableID'

class MoneyValue(Numeric):
    """Custom MoneyValue type - treat as Numeric"""
    __visit_name__ = 'MoneyValue'

class PercentageValue(Numeric):
    """Custom PercentageValue type - treat as Numeric"""
    __visit_name__ = 'PercentageValue'

# Register with SQLAlchemy's type map
mssql.base.ischema_names['TableID'] = TableID
mssql.base.ischema_names['MoneyValue'] = MoneyValue
mssql.base.ischema_names['PercentageValue'] = PercentageValue

class MSSQLConnector(DatabaseConnector):
    """SQL Server connector implementation."""
    
    def __init__(self, connection_string: str):
        # Add connection parameters to prevent timeouts
        if '?' in connection_string:
            connection_string += '&timeout=120&connect_timeout=30'
        else:
            connection_string += '?timeout=120&connect_timeout=30'
        
        self.engine = create_engine(
            connection_string,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
            connect_args={
                'timeout': 120,
                'connect_timeout': 30
            }
        )
        self.inspector = inspect(self.engine)
    
    def get_vendor_version(self) -> Dict[str, str]:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT @@VERSION")).fetchone()
            version_str = result[0] if result else "unknown"
            return {"vendor": "mssql", "version": version_str.split('\n')[0]}
    
    def get_dialect(self) -> str:
        return "tsql"
    
    def get_schemas(self, exclusions: List[str]) -> List[str]:
        schemas = self.inspector.get_schema_names()
        return [s for s in schemas if s not in exclusions]
    
    def get_tables(self, schema: str, table_exclusions: List[str]) -> List[Dict[str, Any]]:
        tables = []
        
        # Get tables
        for table_name in self.inspector.get_table_names(schema=schema):
            if not any(table_name.startswith(prefix) for prefix in table_exclusions):
                tables.append({"name": table_name, "type": "table"})
        
        # Get views
        for view_name in self.inspector.get_view_names(schema=schema):
            if not any(view_name.startswith(prefix) for prefix in table_exclusions):
                tables.append({"name": view_name, "type": "view"})
        
        return tables
    
    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        columns = self.inspector.get_columns(table, schema=schema)
        return [
            {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col["nullable"]
            }
            for col in columns
        ]
    
    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        pk = self.inspector.get_pk_constraint(table, schema=schema)
        return pk.get("constrained_columns", []) if pk else []
    
    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        fks = self.inspector.get_foreign_keys(table, schema=schema)
        result = []
        for fk in fks:
            ref_schema = fk.get("referred_schema", schema)
            ref_table = fk["referred_table"]
            for idx, col in enumerate(fk["constrained_columns"]):
                result.append({
                    "column": col,
                    "ref_table": f"{ref_schema}.{ref_table}",
                    "ref_column": fk["referred_columns"][idx]
                })
        return result
    
    def get_row_count(self, schema: str, table: str) -> Optional[int]:
        try:
            with self.engine.connect() as conn:
                query = text(f"""
                    SELECT SUM(p.rows) 
                    FROM sys.partitions p
                    INNER JOIN sys.tables t ON p.object_id = t.object_id
                    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE s.name = :schema AND t.name = :table AND p.index_id IN (0,1)
                """)
                result = conn.execute(query, {"schema": schema, "table": table}).fetchone()
                return result[0] if result and result[0] else None
        except:
            return None
    
    def get_view_definition(self, schema: str, view: str) -> Optional[str]:
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT OBJECT_DEFINITION(OBJECT_ID(:full_name))
                """)
                result = conn.execute(query, {"full_name": f"{schema}.{view}"}).fetchone()
                return result[0] if result else None
        except:
            return None
    
    def get_stored_procedures(self, schema: str) -> List[Dict[str, str]]:
        procedures = []
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT p.name, OBJECT_DEFINITION(p.object_id) as definition
                    FROM sys.procedures p
                    INNER JOIN sys.schemas s ON p.schema_id = s.schema_id
                    WHERE s.name = :schema
                """)
                result = conn.execute(query, {"schema": schema})
                for row in result:
                    procedures.append({
                        "name": f"{schema}.{row[0]}",
                        "definition": row[1]
                    })
        except:
            pass
        return procedures
    
    def close(self):
        self.engine.dispose()