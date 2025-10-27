"""
Enhanced schema inspector that captures full column metadata for semantic model authority
"""
import logging
from typing import Dict, List, Any, Optional
from collections import Counter
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


class SchemaInspector:
    """Introspect database with complete column metadata for semantic model"""
    
    def __init__(self, connection_string: str, dialect: str = "mssql"):
        self.connection_string = connection_string
        self.dialect = dialect
        self.exclusions = {
            "schemas": ["sys", "information_schema"],
            "table_prefixes": ["temp_", "test_", "backup_", "old_"]
        }
    
    def discover(self) -> Dict[str, Any]:
        """Run full discovery with enhanced metadata"""
        database_info = self._get_database_info()
        schemas = self._introspect_schemas()
        named_assets = self._discover_named_assets()
        negative_findings = self._detect_negative_findings(schemas)
        
        return {
            "database": database_info,
            "dialect": self.dialect,
            "schemas": schemas,
            "named_assets": named_assets,
            "negative_findings": negative_findings
        }
    
    def _get_database_info(self) -> Dict[str, str]:
        """Get database vendor and version"""
        # Implementation would query @@VERSION or equivalent
        return {
            "vendor": "mssql",
            "version": "15.0.2000"
        }
    
    def _introspect_schemas(self) -> List[Dict[str, Any]]:
        """Introspect all schemas with full column metadata"""
        schemas = []
        
        # Query information_schema.tables
        raw_schemas = self._query_schemas()
        
        for schema_name in raw_schemas:
            if schema_name in self.exclusions["schemas"]:
                continue
            
            tables = self._introspect_tables(schema_name)
            if tables:
                schemas.append({
                    "name": schema_name,
                    "tables": tables
                })
        
        return schemas
    
    def _introspect_tables(self, schema_name: str) -> List[Dict[str, Any]]:
        """Introspect tables with complete column metadata"""
        tables = []
        
        raw_tables = self._query_tables(schema_name)
        
        for table_name, table_type in raw_tables:
            # Skip excluded table prefixes
            if any(table_name.startswith(prefix) for prefix in self.exclusions["table_prefixes"]):
                continue
            
            columns = self._introspect_columns(schema_name, table_name)
            primary_key = self._get_primary_key(schema_name, table_name)
            foreign_keys = self._get_foreign_keys(schema_name, table_name)
            indexes = self._get_indexes(schema_name, table_name)
            rowcount = self._get_rowcount(schema_name, table_name)
            
            # CRITICAL: Sample data to get value distributions
            column_stats = self._sample_column_data(schema_name, table_name, columns)
            
            tables.append({
                "name": table_name,
                "type": table_type,
                "columns": columns,
                "column_stats": column_stats,  # NEW: Full column statistics
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "rowcount_sample": rowcount
            })
        
        return tables
    
    def _introspect_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get column definitions"""
        # Query information_schema.columns
        query = f"""
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
        ORDER BY ORDINAL_POSITION
        """
        
        results = self._execute_query(query)
        columns = []
        
        for row in results:
            col_type = row["DATA_TYPE"]
            if row["CHARACTER_MAXIMUM_LENGTH"]:
                col_type += f"({row['CHARACTER_MAXIMUM_LENGTH']})"
            elif row["NUMERIC_PRECISION"]:
                col_type += f"({row['NUMERIC_PRECISION']},{row['NUMERIC_SCALE']})"
            
            columns.append({
                "name": row["COLUMN_NAME"],
                "type": col_type,
                "nullable": row["IS_NULLABLE"] == "YES",
                "default": row["COLUMN_DEFAULT"]
            })
        
        return columns
    
    def _sample_column_data(self, schema: str, table: str, columns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        CRITICAL: Sample 1000 rows per column to get:
        - Distinct value counts
        - Top values with distributions
        - Null rates
        - Min/max for numeric columns
        - Common patterns for string columns
        """
        stats = {}
        full_table_name = f"{schema}.{table}"
        
        for col in columns:
            col_name = col["name"]
            col_type = col["type"].lower()
            
            try:
                # Sample 1000 rows for this column
                sample_query = f"""
                SELECT TOP 1000 [{col_name}]
                FROM [{schema}].[{table}]
                WHERE [{col_name}] IS NOT NULL
                """
                
                sample_results = self._execute_query(sample_query)
                values = [row[col_name] for row in sample_results]
                
                # Get total row count and null count
                count_query = f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT([{col_name}]) as non_null_rows
                FROM [{schema}].[{table}]
                """
                count_result = self._execute_query(count_query)[0]
                total_rows = count_result["total_rows"]
                non_null_rows = count_result["non_null_rows"]
                null_rate = (total_rows - non_null_rows) / total_rows if total_rows > 0 else 0
                
                # Compute statistics based on data type
                if "int" in col_type or "decimal" in col_type or "numeric" in col_type or "float" in col_type:
                    # Numeric column
                    stats[col_name] = {
                        "distinct_count": len(set(values)),
                        "null_rate": round(null_rate, 4),
                        "min": min(values) if values else None,
                        "max": max(values) if values else None,
                        "avg": sum(values) / len(values) if values else None,
                        "sample_values": list(set(values))[:10]
                    }
                
                elif "char" in col_type or "text" in col_type:
                    # String column - get value distribution
                    value_counts = Counter(values)
                    total_sampled = len(values)
                    
                    # Top 20 values with their distribution
                    top_values = {}
                    for value, count in value_counts.most_common(20):
                        top_values[str(value)] = round(count / total_sampled, 4)
                    
                    stats[col_name] = {
                        "distinct_count": len(value_counts),
                        "null_rate": round(null_rate, 4),
                        "top_values": top_values,
                        "sample_values": list(value_counts.keys())[:10],
                        "avg_length": sum(len(str(v)) for v in values) / len(values) if values else 0
                    }
                
                elif "date" in col_type or "time" in col_type:
                    # Date/datetime column
                    stats[col_name] = {
                        "distinct_count": len(set(values)),
                        "null_rate": round(null_rate, 4),
                        "min": min(values) if values else None,
                        "max": max(values) if values else None,
                        "sample_values": [str(v) for v in list(set(values))[:10]]
                    }
                
                elif "bit" in col_type or col_type == "boolean":
                    # Boolean column
                    value_counts = Counter(values)
                    total_sampled = len(values)
                    distribution = {str(k): round(v / total_sampled, 4) for k, v in value_counts.items()}
                    
                    stats[col_name] = {
                        "distinct_count": len(value_counts),
                        "null_rate": round(null_rate, 4),
                        "distribution": distribution,
                        "sample_values": list(value_counts.keys())
                    }
                
                else:
                    # Generic column
                    stats[col_name] = {
                        "distinct_count": len(set(values)),
                        "null_rate": round(null_rate, 4),
                        "sample_values": [str(v) for v in list(set(values))[:10]]
                    }
                
                logger.info(f"Sampled column {schema}.{table}.{col_name}: {stats[col_name]['distinct_count']} distinct values")
            
            except Exception as e:
                logger.warning(f"Could not sample column {schema}.{table}.{col_name}: {e}")
                stats[col_name] = {
                    "error": str(e),
                    "distinct_count": None,
                    "null_rate": None
                }
        
        return stats
    
    def _get_primary_key(self, schema: str, table: str) -> List[str]:
        """Get primary key columns"""
        query = f"""
        SELECT c.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE c
            ON tc.CONSTRAINT_NAME = c.CONSTRAINT_NAME
            AND tc.TABLE_SCHEMA = c.TABLE_SCHEMA
            AND tc.TABLE_NAME = c.TABLE_NAME
        WHERE tc.TABLE_SCHEMA = '{schema}'
            AND tc.TABLE_NAME = '{table}'
            AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ORDER BY c.ORDINAL_POSITION
        """
        
        results = self._execute_query(query)
        return [row["COLUMN_NAME"] for row in results]
    
    def _get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        """Get foreign key relationships"""
        query = f"""
        SELECT 
            fk.name AS constraint_name,
            c1.name AS column_name,
            OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS ref_schema,
            OBJECT_NAME(fk.referenced_object_id) AS ref_table,
            c2.name AS ref_column
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        JOIN sys.columns c1 ON fkc.parent_object_id = c1.object_id AND fkc.parent_column_id = c1.column_id
        JOIN sys.columns c2 ON fkc.referenced_object_id = c2.object_id AND fkc.referenced_column_id = c2.column_id
        WHERE fk.parent_object_id = OBJECT_ID('{schema}.{table}')
        """
        
        results = self._execute_query(query)
        return [
            {
                "column": row["column_name"],
                "ref_table": f"{row['ref_schema']}.{row['ref_table']}",
                "ref_column": row["ref_column"]
            }
            for row in results
        ]
    
    def _get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get indexes for cardinality hints"""
        query = f"""
        SELECT 
            i.name AS index_name,
            i.type_desc AS index_type,
            COL_NAME(ic.object_id, ic.column_id) AS column_name,
            i.is_unique
        FROM sys.indexes i
        JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        WHERE i.object_id = OBJECT_ID('{schema}.{table}')
            AND i.type > 0  -- Exclude heaps
        ORDER BY i.name, ic.key_ordinal
        """
        
        results = self._execute_query(query)
        indexes = {}
        
        for row in results:
            idx_name = row["index_name"]
            if idx_name not in indexes:
                indexes[idx_name] = {
                    "name": idx_name,
                    "type": row["index_type"],
                    "unique": row["is_unique"],
                    "columns": []
                }
            indexes[idx_name]["columns"].append(row["column_name"])
        
        return list(indexes.values())
    
    def _get_rowcount(self, schema: str, table: str) -> int:
        """Get approximate row count"""
        query = f"SELECT COUNT(*) as cnt FROM [{schema}].[{table}]"
        try:
            result = self._execute_query(query)
            return result[0]["cnt"] if result else 0
        except:
            return 0
    
    def _discover_named_assets(self) -> List[Dict[str, Any]]:
        """Discover views, stored procedures, and RDL reports"""
        assets = []
        
        # Discover views
        views = self._discover_views()
        assets.extend(views)
        
        # Discover stored procedures
        procedures = self._discover_procedures()
        assets.extend(procedures)
        
        # Discover RDL files (if path provided)
        rdls = self._discover_rdl_files()
        assets.extend(rdls)
        
        return assets
    
    def _discover_views(self) -> List[Dict[str, Any]]:
        """Discover views and normalize SQL"""
        query = """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            VIEW_DEFINITION
        FROM INFORMATION_SCHEMA.VIEWS
        WHERE TABLE_SCHEMA NOT IN ('sys', 'information_schema')
        """
        
        results = self._execute_query(query)
        views = []
        
        for row in results:
            view_name = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"
            sql = row["VIEW_DEFINITION"]
            
            # Normalize SQL using sqlglot
            try:
                normalized_sql = self._normalize_sql(sql)
                views.append({
                    "kind": "view",
                    "name": view_name,
                    "sql_normalized": normalized_sql,
                    "sql_raw": sql
                })
            except Exception as e:
                logger.warning(f"Could not normalize view {view_name}: {e}")
                views.append({
                    "kind": "view",
                    "name": view_name,
                    "sql_raw": sql,
                    "parse_error": str(e)
                })
        
        return views
    
    def _discover_procedures(self) -> List[Dict[str, Any]]:
        """Discover stored procedures and normalize SQL"""
        query = """
        SELECT 
            ROUTINE_SCHEMA,
            ROUTINE_NAME,
            ROUTINE_DEFINITION
        FROM INFORMATION_SCHEMA.ROUTINES
        WHERE ROUTINE_TYPE = 'PROCEDURE'
            AND ROUTINE_SCHEMA NOT IN ('sys', 'information_schema')
        """
        
        results = self._execute_query(query)
        procedures = []
        
        for row in results:
            proc_name = f"{row['ROUTINE_SCHEMA']}.{row['ROUTINE_NAME']}"
            sql = row["ROUTINE_DEFINITION"]
            
            try:
                normalized_sql = self._normalize_sql(sql)
                procedures.append({
                    "kind": "stored_procedure",
                    "name": proc_name,
                    "sql_normalized": normalized_sql,
                    "sql_raw": sql
                })
            except Exception as e:
                logger.warning(f"Could not normalize procedure {proc_name}: {e}")
                procedures.append({
                    "kind": "stored_procedure",
                    "name": proc_name,
                    "sql_raw": sql,
                    "parse_error": str(e)
                })
        
        return procedures
    
    def _discover_rdl_files(self) -> List[Dict[str, Any]]:
        """Discover RDL report files (if path provided)"""
        # Implementation would scan /data_upload directory
        # For now, return empty list
        return []
    
    def _detect_negative_findings(self, schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        CRITICAL: Detect what does NOT exist in the database
        This is authority for the Q&A phase to NOT ask about missing things
        """
        all_tables = []
        for schema in schemas:
            for table in schema["tables"]:
                all_tables.append(f"{schema['name']}.{table['name']}")
        
        # Check for common patterns that DON'T exist
        refunds_tables = [t for t in all_tables if "refund" in t.lower() or "return" in t.lower() or "credit" in t.lower()]
        currency_tables = [t for t in all_tables if "currency" in t.lower() or "exchange" in t.lower() or "fx" in t.lower()]
        audit_tables = [t for t in all_tables if "audit" in t.lower() or "log" in t.lower() or "history" in t.lower()]
        
        # Check for multi-currency indicators
        has_multi_currency = False
        for schema in schemas:
            for table in schema["tables"]:
                for col_name, col_stats in table.get("column_stats", {}).items():
                    if "currency" in col_name.lower():
                        # Check if currency column has multiple distinct values
                        if col_stats.get("distinct_count", 0) > 1:
                            has_multi_currency = True
                            break
        
        return {
            "refunds_tables": {
                "found": bool(refunds_tables),
                "tables": refunds_tables,
                "conclusion": "Refunds/returns/credits data is present" if refunds_tables else "No refunds/returns/credits tables detected"
            },
            "currency_conversion": {
                "found": has_multi_currency,
                "tables": currency_tables,
                "conclusion": "Multi-currency detected - conversion may be needed" if has_multi_currency else "Single currency system - no conversion needed"
            },
            "audit_trails": {
                "found": bool(audit_tables),
                "tables": audit_tables,
                "conclusion": "Audit/history tables found" if audit_tables else "No audit/history tables detected"
            }
        }
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL using sqlglot"""
        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            return parsed.sql(dialect=self.dialect, pretty=True)
        except Exception as e:
            logger.warning(f"SQL normalization failed: {e}")
            return sql
    
    def _query_schemas(self) -> List[str]:
        """Query all schema names"""
        # Mock implementation - replace with actual DB query
        return ["dbo"]
    
    def _query_tables(self, schema: str) -> List[tuple]:
        """Query all tables in schema"""
        # Mock implementation - replace with actual DB query
        return [
            ("Customer", "table"),
            ("Orders", "table"),
            ("OrderLines", "table")
        ]
    
    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        # Mock implementation - replace with actual DB connection
        # This should use the connection_string to connect and execute
        logger.info(f"Executing query: {query[:100]}...")
        return []