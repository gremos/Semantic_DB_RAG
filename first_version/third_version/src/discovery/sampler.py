"""
Data sampler for collecting column statistics and sample rows.
Samples up to 1000 rows for stats, stores up to 10 example rows.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config.settings import Settings


class DataSampler:
    """Sample data from tables for statistics and examples."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine: Optional[Engine] = None
        
    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.engine = create_engine(
                self.settings.DATABASE_CONNECTION_STRING,
                pool_pre_ping=True,
                echo=False
            )
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
    
    def sample_column_stats(
        self, 
        table_name: str, 
        schema: str, 
        column_name: str,
        column_type: str,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Collect statistics for a single column.
        Returns: distinct_count, null_rate, min, max, sample_values (up to 5)
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        stats = {
            "distinct_count": 0,
            "null_rate": 0.0,
            "sample_values": []
        }
        
        try:
            with self.engine.connect() as conn:
                full_table = f'"{schema}"."{table_name}"'
                full_column = f'"{column_name}"'
                
                # Get distinct count and null rate
                count_query = text(f"""
                    SELECT 
                        COUNT(DISTINCT {full_column}) as distinct_count,
                        CAST(SUM(CASE WHEN {full_column} IS NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as null_rate,
                        COUNT(*) as total_count
                    FROM (SELECT TOP {limit} {full_column} FROM {full_table}) AS sample
                """)
                
                result = conn.execute(count_query)
                row = result.fetchone()
                
                if row:
                    stats["distinct_count"] = int(row[0] or 0)
                    stats["null_rate"] = float(row[1] or 0.0)
                
                # Get min/max for numeric and date columns
                if self._is_numeric_type(column_type):
                    minmax_query = text(f"""
                        SELECT MIN({full_column}), MAX({full_column})
                        FROM (SELECT TOP {limit} {full_column} FROM {full_table} WHERE {full_column} IS NOT NULL) AS sample
                    """)
                    result = conn.execute(minmax_query)
                    row = result.fetchone()
                    if row and row[0] is not None:
                        stats["min"] = self._serialize_value(row[0])
                        stats["max"] = self._serialize_value(row[1])
                        
                    # Detect currency hints
                    if 'money' in column_type.lower() or 'currency' in column_name.lower():
                        stats["unit_hint"] = "currency"
                        stats["currency_hint"] = "USD"  # Default, could be detected
                
                elif self._is_date_type(column_type):
                    minmax_query = text(f"""
                        SELECT MIN({full_column}), MAX({full_column})
                        FROM (SELECT TOP {limit} {full_column} FROM {full_table} WHERE {full_column} IS NOT NULL) AS sample
                    """)
                    result = conn.execute(minmax_query)
                    row = result.fetchone()
                    if row and row[0] is not None:
                        stats["min"] = self._serialize_value(row[0])
                        stats["max"] = self._serialize_value(row[1])
                
                # Get sample values (up to 5 distinct, stable order)
                # Order by primary key if possible, otherwise by the column itself
                sample_query = text(f"""
                    SELECT DISTINCT TOP 5 {full_column}
                    FROM {full_table}
                    WHERE {full_column} IS NOT NULL
                    ORDER BY {full_column} ASC
                """)
                
                result = conn.execute(sample_query)
                stats["sample_values"] = [
                    self._serialize_value(row[0]) 
                    for row in result.fetchall()
                ]
                
        except SQLAlchemyError as e:
            # Log but don't fail - return partial stats
            pass
        
        return stats
    
    def sample_table_rows(
        self, 
        table_name: str, 
        schema: str,
        columns: List[str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Sample up to 10 rows from a table for display.
        Returns list of row dictionaries.
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        rows = []
        
        try:
            with self.engine.connect() as conn:
                full_table = f'"{schema}"."{table_name}"'
                column_list = ', '.join(f'"{col}"' for col in columns)
                
                query = text(f"""
                    SELECT TOP {limit} {column_list}
                    FROM {full_table}
                """)
                
                result = conn.execute(query)
                
                for row in result.fetchall():
                    row_dict = {}
                    for i, col_name in enumerate(columns):
                        row_dict[col_name] = self._serialize_value(row[i])
                    rows.append(row_dict)
                    
        except SQLAlchemyError as e:
            # Log but don't fail
            pass
        
        return rows
    
    def _is_numeric_type(self, column_type: str) -> bool:
        """Check if column type is numeric."""
        numeric_types = [
            'int', 'integer', 'bigint', 'smallint', 'tinyint',
            'decimal', 'numeric', 'float', 'real', 'double',
            'money', 'smallmoney'
        ]
        return any(nt in column_type.lower() for nt in numeric_types)
    
    def _is_date_type(self, column_type: str) -> bool:
        """Check if column type is date/time."""
        date_types = ['date', 'time', 'datetime', 'timestamp']
        return any(dt in column_type.lower() for dt in date_types)
    
    def _serialize_value(self, value: Any) -> str:
        """Convert database value to JSON-serializable string."""
        if value is None:
            return None
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, bytes):
            return "<binary>"
        else:
            return str(value)
    
    def enrich_table_with_samples(
        self, 
        table_data: Dict[str, Any], 
        schema: str
    ) -> Dict[str, Any]:
        """
        Enrich a table dictionary with column stats and sample rows.
        Modifies table_data in place and returns it.
        """
        if not self.engine:
            self.connect()
        
        table_name = table_data['name']
        
        # Add stats to each column
        for column in table_data['columns']:
            column_name = column['name']
            column_type = column['type']
            
            stats = self.sample_column_stats(
                table_name, 
                schema, 
                column_name,
                column_type,
                limit=1000
            )
            
            column['stats'] = stats
        
        # Add sample rows
        column_names = [col['name'] for col in table_data['columns']]
        sample_rows = self.sample_table_rows(
            table_name,
            schema,
            column_names,
            limit=10
        )
        
        table_data['sample_rows'] = sample_rows
        
        return table_data
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()