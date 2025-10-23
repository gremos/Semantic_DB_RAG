from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, inspect
from .base import DatabaseConnector

class PostgresConnector(DatabaseConnector):
    """PostgreSQL connector implementation."""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.inspector = inspect(self.engine)
    
    def get_vendor_version(self) -> Dict[str, str]:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT version()")).fetchone()
            return {"vendor": "postgres", "version": result[0] if result else "unknown"}
    
    def get_dialect(self) -> str:
        return "postgres"
    
    def get_schemas(self, exclusions: List[str]) -> List[str]:
        schemas = self.inspector.get_schema_names()
        return [s for s in schemas if s not in exclusions and not s.startswith('pg_')]
    
    def get_tables(self, schema: str, table_exclusions: List[str]) -> List[Dict[str, Any]]:
        tables = []
        
        for table_name in self.inspector.get_table_names(schema=schema):
            if not any(table_name.startswith(prefix) for prefix in table_exclusions):
                tables.append({"name": table_name, "type": "table"})
        
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
                    SELECT reltuples::bigint 
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = :schema AND c.relname = :table
                """)
                result = conn.execute(query, {"schema": schema, "table": table}).fetchone()
                return result[0] if result else None
        except:
            return None
    
    def get_view_definition(self, schema: str, view: str) -> Optional[str]:
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT definition 
                    FROM pg_views 
                    WHERE schemaname = :schema AND viewname = :view
                """)
                result = conn.execute(query, {"schema": schema, "view": view}).fetchone()
                return result[0] if result else None
        except:
            return None
    
    def get_stored_procedures(self, schema: str) -> List[Dict[str, str]]:
        procedures = []
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT p.proname, pg_get_functiondef(p.oid) as definition
                    FROM pg_proc p
                    INNER JOIN pg_namespace n ON p.pronamespace = n.oid
                    WHERE n.nspname = :schema AND p.prokind = 'p'
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