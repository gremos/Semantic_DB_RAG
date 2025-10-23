from typing import Dict, Any, List, Tuple
from connectors.base import DatabaseConnector
import logging

logger = logging.getLogger(__name__)

class ColumnSampler:
    """Selectively sample columns that need value examples."""
    
    # Keywords that indicate columns needing samples
    STATUS_KEYWORDS = ['status', 'state', 'type', 'category', 'kind', 'flag', 
                      'active', 'enabled', 'cancelled', 'deleted']
    
    # Types that are candidates for sampling
    SAMPLE_TYPES = ['varchar', 'char', 'nvarchar', 'nchar', 'text', 'enum']
    
    def __init__(self, connector: DatabaseConnector, max_samples: int = 50):
        self.connector = connector
        self.max_samples = max_samples
    
    def identify_sample_targets(
        self, 
        discovery_data: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """
        Identify columns that should be sampled.
        
        Returns:
            List of (schema, table, column) tuples
        """
        targets = []
        
        for schema in discovery_data.get("schemas", []):
            schema_name = schema["name"]
            
            for table in schema.get("tables", []):
                table_name = table["name"]
                
                for column in table.get("columns", []):
                    if self._should_sample_column(column):
                        targets.append((schema_name, table_name, column["name"]))
        
        # Prioritize and limit
        targets = self._prioritize_targets(targets, discovery_data)
        return targets[:self.max_samples]
    
    def _should_sample_column(self, column: Dict[str, Any]) -> bool:
        """Determine if column should be sampled."""
        col_name = column["name"].lower()
        col_type = column["type"].lower()
        
        # Check if type is suitable for sampling
        type_match = any(t in col_type for t in self.SAMPLE_TYPES)
        if not type_match:
            return False
        
        # Check if name suggests categorical/status column
        name_match = any(keyword in col_name for keyword in self.STATUS_KEYWORDS)
        
        return name_match
    
    def _prioritize_targets(
        self, 
        targets: List[Tuple[str, str, str]], 
        discovery_data: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Prioritize targets by importance."""
        # Create lookup for row counts
        table_sizes = {}
        for schema in discovery_data.get("schemas", []):
            for table in schema.get("tables", []):
                key = f"{schema['name']}.{table['name']}"
                table_sizes[key] = table.get("rowcount_sample", 0) or 0
        
        # Sort by table size (larger tables first)
        def sort_key(target):
            schema, table, column = target
            table_key = f"{schema}.{table}"
            return -table_sizes.get(table_key, 0)  # Negative for descending
        
        return sorted(targets, key=sort_key)
    
    def sample_columns(
        self, 
        targets: List[Tuple[str, str, str]]
    ) -> Dict[str, List[Any]]:
        """
        Sample values from identified columns.
        
        Returns:
            Dict mapping "schema.table.column" -> [sample values]
        """
        samples = {}
        
        for schema, table, column in targets:
            try:
                values = self._sample_column_values(schema, table, column)
                key = f"{schema}.{table}.{column}"
                samples[key] = values
                logger.info(f"  Sampled {len(values)} values from {key}")
            except Exception as e:
                logger.warning(f"  Failed to sample {schema}.{table}.{column}: {e}")
        
        return samples
    
    def _sample_column_values(
        self, 
        schema: str, 
        table: str, 
        column: str, 
        limit: int = 10
    ) -> List[Any]:
        """Sample distinct values from a column."""
        try:
            with self.connector.engine.connect() as conn:
                from sqlalchemy import text
                
                # Get distinct values (limit to 10)
                query = text(f"""
                    SELECT DISTINCT TOP {limit} [{column}]
                    FROM [{schema}].[{table}]
                    WHERE [{column}] IS NOT NULL
                    ORDER BY [{column}]
                """)
                
                result = conn.execute(query)
                values = [row[0] for row in result]
                return values
        except Exception as e:
            logger.warning(f"Sample query failed: {e}")
            return []