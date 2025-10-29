"""
Discovery Engine for GPT-5 Semantic Modeling & SQL Q&A System

Phase 1: Database Discovery
- Introspects schemas, tables, columns, keys, indexes
- Samples data with statistics
- Normalizes SQL (views, stored procedures, RDL)
- Detects implicit foreign key relationships (optimized)
- Caches results with fingerprinting
"""
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sqlglot
from sqlalchemy import create_engine, inspect, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config.settings import (
    get_settings,
    get_database_config,
    get_discovery_config,
    get_path_config,
    get_relationship_config
)
from src.discovery.relationship_detector import detect_relationships

logger = logging.getLogger(__name__)


class DiscoveryCache:
    """
    Cache manager for discovery results
    Implements fingerprinting to detect database changes
    """
    
    def __init__(self, cache_dir: Path, cache_hours: int = 168):
        self.cache_dir = cache_dir
        self.cache_hours = cache_hours
        self.cache_file = cache_dir / 'discovery.json'
        self.fingerprint_file = cache_dir / 'discovery_fingerprint.json'
    
    def get_database_fingerprint(self, engine: Engine) -> str:
        """
        Generate fingerprint of database structure
        Used to detect schema changes
        """
        fingerprint_data = []
        
        inspector = inspect(engine)
        
        # Collect schema metadata
        for schema in inspector.get_schema_names():
            for table_name in inspector.get_table_names(schema=schema):
                # Table fingerprint: name + column names + types
                columns = inspector.get_columns(table_name, schema=schema)
                col_signatures = [
                    f"{col['name']}:{col['type']}" 
                    for col in columns
                ]
                
                table_sig = f"{schema}.{table_name}:{','.join(sorted(col_signatures))}"
                fingerprint_data.append(table_sig)
        
        # Generate hash
        fingerprint_str = '|'.join(sorted(fingerprint_data))
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def is_valid(self, engine: Engine) -> bool:
        """
        Check if cached discovery is still valid
        Returns True if cache exists, is fresh, and fingerprint matches
        """
        # Check if cache exists
        if not self.cache_file.exists():
            logger.info("No discovery cache found")
            return False
        
        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
        if cache_age > timedelta(hours=self.cache_hours):
            logger.info(f"Discovery cache expired (age: {cache_age})")
            return False
        
        # Check fingerprint
        if not self.fingerprint_file.exists():
            logger.info("No fingerprint found, assuming cache invalid")
            return False
        
        try:
            with open(self.fingerprint_file, 'r') as f:
                cached_fingerprint = json.load(f).get('fingerprint')
            
            current_fingerprint = self.get_database_fingerprint(engine)
            
            if cached_fingerprint != current_fingerprint:
                logger.info("Database schema changed, cache invalid")
                return False
            
            logger.info("Discovery cache is valid")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def load(self) -> Optional[Dict]:
        """Load cached discovery data"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded discovery cache from {self.cache_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def save(self, data: Dict, engine: Engine):
        """Save discovery data to cache"""
        try:
            # Save discovery data
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Save fingerprint
            fingerprint = self.get_database_fingerprint(engine)
            with open(self.fingerprint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'fingerprint': fingerprint,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved discovery cache to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")


class DiscoveryEngine:
    """
    Main discovery engine
    Orchestrates database introspection, sampling, and relationship detection
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize discovery engine
        
        Args:
            connection_string: Database connection string (uses config if not provided)
        """
        # Load settings
        self.settings = get_settings()
        self.db_config = get_database_config()
        self.discovery_config = get_discovery_config()
        self.path_config = get_path_config()
        self.relationship_config = get_relationship_config()
        
        # Use provided connection string or from config
        self.connection_string = connection_string or self.db_config.connection_string
        
        # Initialize cache
        self.cache = DiscoveryCache(
            cache_dir=self.path_config.cache_dir,
            cache_hours=self.discovery_config.cache_hours
        )
        
        # Engine and inspector (lazy init)
        self._engine: Optional[Engine] = None
        self._inspector = None
        
        # Discovery results
        self.discovery_data: Optional[Dict] = None
    
    @property
    def engine(self) -> Engine:
        """Get or create database engine"""
        if self._engine is None:
            logger.info("Creating database engine...")
            self._engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
        return self._engine
    
    @property
    def inspector(self):
        """Get or create database inspector"""
        if self._inspector is None:
            self._inspector = inspect(self.engine)
        return self._inspector
    
    def discover(self, use_cache: bool = True, skip_relationships: bool = False) -> Dict:
        """
        Main discovery entry point
        
        Args:
            use_cache: If True, use cached discovery if valid
            skip_relationships: If True, skip relationship detection (for testing)
            
        Returns:
            Discovery data dictionary
        """
        logger.info("=" * 80)
        logger.info("STARTING DATABASE DISCOVERY")
        logger.info("=" * 80)
        logger.info(f"Connection: {self.connection_string[:50]}...")
        logger.info(f"Cache: {'enabled' if use_cache else 'disabled'}")
        logger.info(f"Relationships: {'enabled' if not skip_relationships else 'disabled'}")
        
        start_time = time.time()
        
        # Check cache
        if use_cache and self.cache.is_valid(self.engine):
            logger.info("Using cached discovery data")
            self.discovery_data = self.cache.load()
            if self.discovery_data:
                return self.discovery_data
        
        logger.info("Running fresh discovery...")
        
        try:
            # Step 1: Detect database vendor/version
            db_info = self._detect_database_info()
            logger.info(f"Database: {db_info['vendor']} {db_info.get('version', 'unknown')}")
            
            # Step 2: Discover schemas and tables
            schemas_data = self._discover_schemas()
            logger.info(f"Discovered {len(schemas_data)} schemas")
            
            # Step 3: Process named assets (views, stored procedures, RDL)
            named_assets = self._discover_named_assets()
            logger.info(f"Discovered {len(named_assets)} named assets")
            
            # Step 4: Detect relationships (optimized)
            if skip_relationships or not self.relationship_config.enabled:
                logger.info("Skipping relationship detection")
                relationships = []
            else:
                logger.info("Starting optimized relationship detection...")
                relationships = self._detect_relationships(schemas_data, db_info['dialect'])
                logger.info(f"Detected {len(relationships)} relationships")
            
            # Assemble discovery data
            self.discovery_data = {
                'database': db_info,
                'dialect': db_info['dialect'],
                'schemas': schemas_data,
                'named_assets': named_assets,
                'inferred_relationships': relationships,
                'metadata': {
                    'discovered_at': datetime.utcnow().isoformat(),
                    'discovery_duration_seconds': time.time() - start_time,
                    'total_tables': sum(len(s['tables']) for s in schemas_data),
                    'total_columns': sum(
                        len(t['columns']) 
                        for s in schemas_data 
                        for t in s['tables']
                    ),
                }
            }
            
            # Save to cache
            self.cache.save(self.discovery_data, self.engine)
            
            elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"DISCOVERY COMPLETE in {elapsed:.1f}s")
            logger.info("=" * 80)
            
            return self.discovery_data
            
        except Exception as e:
            logger.error(f"Discovery failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            if self._engine:
                self._engine.dispose()
    
    def _detect_database_info(self) -> Dict:
        """Detect database vendor and version"""
        try:
            # Get dialect name
            dialect_name = self.engine.dialect.name.lower()
            
            # Map to standardized names
            dialect_map = {
                'mssql': 'mssql',
                'postgresql': 'postgres',
                'mysql': 'mysql',
                'mariadb': 'mysql',
                'sqlite': 'sqlite',
                'oracle': 'oracle',
            }
            
            dialect = dialect_map.get(dialect_name, dialect_name)
            
            # Try to get version
            version = None
            try:
                if dialect == 'mssql':
                    result = self.engine.execute(text("SELECT @@VERSION")).scalar()
                    version_match = re.search(r'Microsoft SQL Server (\d+)', str(result))
                    if version_match:
                        version = version_match.group(1)
                elif dialect in ['postgres', 'postgresql']:
                    result = self.engine.execute(text("SELECT version()")).scalar()
                    version_match = re.search(r'PostgreSQL ([\d.]+)', str(result))
                    if version_match:
                        version = version_match.group(1)
                elif dialect == 'mysql':
                    result = self.engine.execute(text("SELECT VERSION()")).scalar()
                    version = str(result).split('-')[0]
            except Exception as e:
                logger.warning(f"Could not detect database version: {e}")
            
            return {
                'vendor': dialect,
                'version': version,
                'dialect': dialect,
            }
            
        except Exception as e:
            logger.error(f"Failed to detect database info: {e}")
            return {
                'vendor': 'unknown',
                'dialect': 'generic',
            }
    
    def _should_exclude_schema(self, schema_name: str) -> bool:
        """Check if schema should be excluded"""
        schema_lower = schema_name.lower()
        return any(excl.lower() in schema_lower 
                  for excl in self.discovery_config.schema_exclusions)
    
    def _should_exclude_table(self, table_name: str) -> bool:
        """Check if table should be excluded"""
        table_lower = table_name.lower()
        
        # Check prefix exclusions
        for excl in self.discovery_config.table_exclusions:
            if table_lower.startswith(excl.lower()):
                return True
        
        # Check regex patterns
        for pattern in self.discovery_config.table_exclusion_patterns:
            try:
                if re.match(pattern, table_name, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
        
        return False
    
    def _discover_schemas(self) -> List[Dict]:
        """
        Discover all schemas, tables, columns, keys, and indexes
        Returns list of schema dictionaries
        """
        schemas_data = []
        
        for schema_name in self.inspector.get_schema_names():
            # Skip excluded schemas
            if self._should_exclude_schema(schema_name):
                logger.debug(f"Skipping excluded schema: {schema_name}")
                continue
            
            logger.info(f"Discovering schema: {schema_name}")
            
            # Discover tables in this schema
            tables_data = self._discover_tables(schema_name)
            
            if tables_data:
                schemas_data.append({
                    'name': schema_name,
                    'tables': tables_data
                })
        
        return schemas_data
    
    def _discover_tables(self, schema_name: str) -> List[Dict]:
        """Discover tables in a schema"""
        tables_data = []
        table_names = self.inspector.get_table_names(schema=schema_name)
        
        # Include views
        view_names = self.inspector.get_view_names(schema=schema_name)
        
        # Process tables
        for table_name in table_names:
            if self._should_exclude_table(table_name):
                logger.debug(f"Skipping excluded table: {schema_name}.{table_name}")
                continue
            
            try:
                table_data = self._discover_table(schema_name, table_name, 'table')
                if table_data:
                    tables_data.append(table_data)
            except Exception as e:
                logger.error(f"Error discovering table {schema_name}.{table_name}: {e}")
        
        # Process views
        for view_name in view_names:
            if self._should_exclude_table(view_name):
                logger.debug(f"Skipping excluded view: {schema_name}.{view_name}")
                continue
            
            try:
                view_data = self._discover_table(schema_name, view_name, 'view')
                if view_data:
                    tables_data.append(view_data)
            except Exception as e:
                logger.error(f"Error discovering view {schema_name}.{view_name}: {e}")
        
        return tables_data
    
    def _discover_table(self, schema_name: str, table_name: str, table_type: str) -> Optional[Dict]:
        """
        Discover single table/view with columns, keys, indexes, and samples
        
        Args:
            schema_name: Schema name
            table_name: Table/view name
            table_type: 'table' or 'view'
            
        Returns:
            Table data dictionary or None if error
        """
        logger.debug(f"Discovering {table_type}: {schema_name}.{table_name}")
        
        try:
            # Get columns
            columns_info = self.inspector.get_columns(table_name, schema=schema_name)
            columns_data = []
            
            for col in columns_info:
                col_data = {
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                    'default': str(col.get('default')) if col.get('default') else None,
                }
                
                # Get statistics and samples
                try:
                    stats = self._get_column_stats(schema_name, table_name, col['name'], str(col['type']))
                    if stats:
                        col_data['stats'] = stats
                except Exception as e:
                    logger.debug(f"Could not get stats for {schema_name}.{table_name}.{col['name']}: {e}")
                
                columns_data.append(col_data)
            
            # Get primary key
            pk_constraint = self.inspector.get_pk_constraint(table_name, schema=schema_name)
            primary_key = pk_constraint.get('constrained_columns', []) if pk_constraint else []
            
            # Get foreign keys
            foreign_keys_data = []
            try:
                fk_constraints = self.inspector.get_foreign_keys(table_name, schema=schema_name)
                for fk in fk_constraints:
                    if fk.get('constrained_columns') and fk.get('referred_columns'):
                        for local_col, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
                            ref_schema = fk.get('referred_schema', schema_name)
                            foreign_keys_data.append({
                                'column': local_col,
                                'ref_table': f"{ref_schema}.{fk['referred_table']}",
                                'ref_column': ref_col
                            })
            except Exception as e:
                logger.debug(f"Could not get foreign keys for {schema_name}.{table_name}: {e}")
            
            # Get indexes
            indexes_data = []
            try:
                indexes = self.inspector.get_indexes(table_name, schema=schema_name)
                for idx in indexes:
                    indexes_data.append({
                        'name': idx.get('name'),
                        'columns': idx.get('column_names', []),
                        'unique': idx.get('unique', False)
                    })
            except Exception as e:
                logger.debug(f"Could not get indexes for {schema_name}.{table_name}: {e}")
            
            # Get row count sample
            rowcount_sample = self._get_rowcount_sample(schema_name, table_name)
            
            # Get sample rows
            sample_rows = self._get_sample_rows(schema_name, table_name)
            
            return {
                'name': table_name,
                'type': table_type,
                'columns': columns_data,
                'primary_key': primary_key,
                'foreign_keys': foreign_keys_data,
                'indexes': indexes_data,
                'rowcount_sample': rowcount_sample,
                'sample_rows': sample_rows,
                'source_assets': []  # Populated later if from view/SP/RDL
            }
            
        except Exception as e:
            logger.error(f"Failed to discover {table_type} {schema_name}.{table_name}: {e}")
            return None
    
    def _get_column_stats(self, schema: str, table: str, column: str, col_type: str) -> Optional[Dict]:
        """
        Get statistics for a column
        Returns dict with distinct_count, null_rate, min, max, sample_values
        """
        try:
            # Build query based on dialect
            dialect = self.engine.dialect.name.lower()
            
            if dialect == 'mssql':
                quote_char = '['
                quote_end = ']'
            elif dialect in ['mysql', 'mariadb']:
                quote_char = '`'
                quote_end = '`'
            else:
                quote_char = '"'
                quote_end = '"'
            
            # Quoted identifiers
            schema_q = f"{quote_char}{schema}{quote_end}"
            table_q = f"{quote_char}{table}{quote_end}"
            column_q = f"{quote_char}{column}{quote_end}"
            
            # Stats query with sample limit of 1000
            if dialect == 'mssql':
                query = text(f"""
                    SELECT TOP 1000
                        COUNT(DISTINCT {column_q}) as distinct_count,
                        SUM(CASE WHEN {column_q} IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as null_rate,
                        MIN({column_q}) as min_val,
                        MAX({column_q}) as max_val
                    FROM {schema_q}.{table_q}
                """)
            else:
                query = text(f"""
                    SELECT
                        COUNT(DISTINCT {column_q}) as distinct_count,
                        SUM(CASE WHEN {column_q} IS NULL THEN 1 ELSE 0 END)::float / COUNT(*) as null_rate,
                        MIN({column_q}) as min_val,
                        MAX({column_q}) as max_val
                    FROM {schema_q}.{table_q}
                    LIMIT 1000
                """)
            
            result = self.engine.execute(query).fetchone()
            
            stats = {
                'distinct_count': result[0] if result[0] is not None else 0,
                'null_rate': float(result[1]) if result[1] is not None else 0.0,
            }
            
            # Add min/max for appropriate types
            if any(t in col_type.lower() for t in ['int', 'float', 'decimal', 'numeric', 'date', 'time']):
                stats['min'] = str(result[2]) if result[2] is not None else None
                stats['max'] = str(result[3]) if result[3] is not None else None
            
            # Get sample values (top 5)
            sample_values = self._get_sample_values(schema, table, column, limit=5)
            if sample_values:
                stats['sample_values'] = sample_values
            
            # Detect currency hint for numeric columns
            if any(t in col_type.lower() for t in ['decimal', 'numeric', 'money']):
                if any(kw in column.lower() for kw in ['amount', 'price', 'cost', 'revenue', 'total']):
                    stats['unit_hint'] = 'currency'
                    stats['currency_hint'] = 'USD'  # Default, could be detected
            
            return stats
            
        except Exception as e:
            logger.debug(f"Error getting stats for {schema}.{table}.{column}: {e}")
            return None
    
    def _get_sample_values(self, schema: str, table: str, column: str, limit: int = 5) -> List[str]:
        """Get sample distinct values from a column"""
        try:
            dialect = self.engine.dialect.name.lower()
            
            if dialect == 'mssql':
                query = text(f"""
                    SELECT DISTINCT TOP {limit} [{column}]
                    FROM [{schema}].[{table}]
                    WHERE [{column}] IS NOT NULL
                    ORDER BY [{column}]
                """)
            else:
                query = text(f"""
                    SELECT DISTINCT "{column}"
                    FROM "{schema}"."{table}"
                    WHERE "{column}" IS NOT NULL
                    ORDER BY "{column}"
                    LIMIT {limit}
                """)
            
            result = self.engine.execute(query).fetchall()
            return [str(row[0]) for row in result]
            
        except Exception as e:
            logger.debug(f"Error getting sample values: {e}")
            return []
    
    def _get_rowcount_sample(self, schema: str, table: str) -> int:
        """Get approximate row count (sample, not exact)"""
        try:
            dialect = self.engine.dialect.name.lower()
            
            if dialect == 'mssql':
                query = text(f"SELECT TOP 1000 COUNT(*) FROM [{schema}].[{table}]")
            else:
                query = text(f'SELECT COUNT(*) FROM "{schema}"."{table}" LIMIT 1000')
            
            result = self.engine.execute(query).scalar()
            return int(result) if result else 0
            
        except Exception as e:
            logger.debug(f"Error getting rowcount for {schema}.{table}: {e}")
            return 0
    
    def _get_sample_rows(self, schema: str, table: str, limit: int = 10) -> List[Dict]:
        """Get sample rows from table"""
        try:
            dialect = self.engine.dialect.name.lower()
            
            if dialect == 'mssql':
                query = text(f"SELECT TOP {limit} * FROM [{schema}].[{table}]")
            else:
                query = text(f'SELECT * FROM "{schema}"."{table}" LIMIT {limit}')
            
            result = self.engine.execute(query).fetchall()
            
            if not result:
                return []
            
            # Convert to list of dicts
            columns = result[0].keys()
            return [
                {col: str(row[col]) if row[col] is not None else None for col in columns}
                for row in result
            ]
            
        except Exception as e:
            logger.debug(f"Error getting sample rows from {schema}.{table}: {e}")
            return []
    
    def _discover_named_assets(self) -> List[Dict]:
        """
        Discover named assets: views, stored procedures, RDL files
        Returns list of asset dictionaries
        """
        assets = []
        
        # Discover views (already included in schema discovery, but normalize SQL)
        # This would extract view definitions and normalize them
        
        # Discover stored procedures
        # This would extract SP definitions and normalize them
        
        # Discover RDL files
        rdl_path = self.path_config.rdl_path
        if rdl_path.exists():
            logger.info(f"Scanning RDL files in {rdl_path}")
            # This would parse RDL files and extract datasets
            # For now, just log
            rdl_files = list(rdl_path.rglob('*.rdl'))
            logger.info(f"Found {len(rdl_files)} RDL files")
        
        return assets
    
    def _detect_relationships(self, schemas_data: List[Dict], dialect: str) -> List[Dict]:
        """
        Detect foreign key relationships using optimized detector
        
        Args:
            schemas_data: List of schema dictionaries from discovery
            dialect: Database dialect
            
        Returns:
            List of relationship dictionaries
        """
        # Build discovery data structure for relationship detector
        discovery_for_relationships = {
            'database': {'vendor': dialect, 'version': 'unknown'},
            'dialect': dialect,
            'schemas': schemas_data,
        }
        
        # Call optimized relationship detector
        try:
            relationships = detect_relationships(
                connection_string=self.connection_string,
                discovery_data=discovery_for_relationships,
                config=self.relationship_config
            )
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship detection failed: {e}", exc_info=True)
            return []


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_discovery(use_cache: bool = True, 
                 skip_relationships: bool = False,
                 connection_string: Optional[str] = None) -> Dict:
    """
    Convenience function to run discovery
    
    Args:
        use_cache: Use cached discovery if valid
        skip_relationships: Skip relationship detection
        connection_string: Override connection string from config
        
    Returns:
        Discovery data dictionary
    """
    engine = DiscoveryEngine(connection_string=connection_string)
    return engine.discover(use_cache=use_cache, skip_relationships=skip_relationships)


def clear_discovery_cache():
    """Clear discovery cache"""
    path_config = get_path_config()
    cache_file = path_config.cache_dir / 'discovery.json'
    fingerprint_file = path_config.cache_dir / 'discovery_fingerprint.json'
    
    removed = []
    if cache_file.exists():
        cache_file.unlink()
        removed.append(str(cache_file))
    if fingerprint_file.exists():
        fingerprint_file.unlink()
        removed.append(str(fingerprint_file))
    
    if removed:
        logger.info(f"Cleared discovery cache: {', '.join(removed)}")
    else:
        logger.info("No discovery cache to clear")