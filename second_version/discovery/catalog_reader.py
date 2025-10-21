from typing import Dict, Any, List
from connectors.base import DatabaseConnector
from config.settings import settings
from .table_deduplicator import TableDeduplicator
import logging

logger = logging.getLogger(__name__)

class CatalogReader:
    """Read database catalog metadata."""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
        self.deduplicator = TableDeduplicator(
            settings.table_name_patterns_to_exclude
        )
    
    def read_full_catalog(self, max_tables_per_schema: int = 50) -> Dict[str, Any]:
        """
        Read complete database catalog with progress tracking.
        
        Args:
            max_tables_per_schema: Limit tables per schema to reduce payload size
        
        Returns discovery data structure.
        """
        vendor_version = self.connector.get_vendor_version()
        dialect = self.connector.get_dialect()
        
        schemas = self.connector.get_schemas(settings.schema_exclusions)
        logger.info(f"Found {len(schemas)} schemas to process")
        
        schemas_data = []
        total_tables_limit = 200  # Overall limit across all schemas
        total_tables_processed = 0
        
        for schema_idx, schema in enumerate(schemas, 1):
            if total_tables_processed >= total_tables_limit:
                logger.info(f"Reached total table limit ({total_tables_limit}). Stopping discovery.")
                break
            
            logger.info(f"Processing schema {schema_idx}/{len(schemas)}: {schema}")
            
            try:
                # Get all tables for this schema
                raw_tables = self.connector.get_tables(schema, settings.table_exclusions)
                logger.info(f"  Found {len(raw_tables)} tables/views before deduplication")
                
                # Deduplicate tables
                tables = self.deduplicator.filter_tables(raw_tables)
                logger.info(f"  Kept {len(tables)} tables/views after deduplication")
                
                # Sort by row count (prioritize larger tables)
                tables_with_rows = []
                for t in tables:
                    try:
                        rowcount = self.connector.get_row_count(schema, t['name'])
                        tables_with_rows.append((t, rowcount or 0))
                    except:
                        tables_with_rows.append((t, 0))
                
                # Sort descending by row count
                tables_with_rows.sort(key=lambda x: x[1], reverse=True)
                
                # Limit to top N tables per schema
                tables_to_process = [t[0] for t in tables_with_rows[:max_tables_per_schema]]
                
                if len(tables_to_process) < len(tables):
                    logger.info(f"  Limited to top {len(tables_to_process)} tables by row count")
                
                tables_data = []
                
                for table_idx, table_info in enumerate(tables_to_process, 1):
                    if total_tables_processed >= total_tables_limit:
                        break
                    
                    if table_idx % 10 == 0:
                        logger.info(f"    Processing table {table_idx}/{len(tables_to_process)}")
                    
                    table_name = table_info["name"]
                    table_type = table_info["type"]
                    
                    try:
                        columns = self.connector.get_columns(schema, table_name)
                        primary_key = self.connector.get_primary_keys(schema, table_name)
                        foreign_keys = self.connector.get_foreign_keys(schema, table_name)
                        rowcount = self.connector.get_row_count(schema, table_name)
                        
                        table_data = {
                            "name": table_name,
                            "type": table_type,
                            "columns": columns,
                            "primary_key": primary_key,
                            "foreign_keys": foreign_keys,
                            "rowcount_sample": rowcount,
                            "source_assets": []
                        }
                        
                        tables_data.append(table_data)
                        total_tables_processed += 1
                    
                    except Exception as e:
                        logger.warning(f"    Failed to process table {schema}.{table_name}: {e}")
                        continue
                
                if tables_data:  # Only add schema if it has tables
                    schemas_data.append({
                        "name": schema,
                        "tables": tables_data
                    })
                
                logger.info(f"  Completed schema {schema}: {len(tables_data)} tables processed")
            
            except Exception as e:
                logger.error(f"  Failed to process schema {schema}: {e}")
                continue
        
        logger.info(f"Discovery complete: {len(schemas_data)} schemas, "
                f"{sum(len(s['tables']) for s in schemas_data)} tables")
        
        return {
            "database": vendor_version,
            "dialect": dialect,
            "schemas": schemas_data
        }