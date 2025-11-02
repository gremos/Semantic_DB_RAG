"""
Discovery Engine for GPT-5 Semantic Modeling & SQL Q&A System

Phase 1: Database Discovery
- Introspects schemas, tables, columns, keys, indexes
- Samples data with statistics
- Normalizes SQL (views, stored procedures, RDL)
- Detects implicit foreign key relationships
- Caches results with fingerprinting
"""
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

logger = logging.getLogger(__name__)


class DiscoveryCache:
    """
    Cache manager for discovery results
    Implements fingerprinting to detect database changes
    """
    
    def __init__(self, cache_dir: Path, cache_hours: int = 168):
        self.cache_dir = cache_dir
        self.cache_hours = cache_hours
        
        # Determine cache filename based on sample mode
        try:
            from config.settings import get_discovery_config
            discovery_config = get_discovery_config()
            
            # Ensure we get a string, not a tuple
            if discovery_config.sample_mode_enabled:
                filename = str(getattr(discovery_config, 'sample_output_filename', 'discovery-sample.json'))
            else:
                filename = 'discovery.json'
        except Exception as e:
            # Fallback if config not available or has issues
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not determine sample mode filename: {e}")
            filename = 'discovery.json'
        
        self.cache_file = cache_dir / filename
        self.fingerprint_file = cache_dir / 'discovery_fingerprint.json'
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_database_fingerprint(self, engine: Engine) -> str:
        """
        Generate fingerprint of database structure
        Used to detect schema changes
        """
        fingerprint_data = []
        
        inspector = inspect(engine)
        
        try:
            # Collect schema metadata
            for schema in inspector.get_schema_names():
                try:
                    for table_name in inspector.get_table_names(schema=schema):
                        # Table fingerprint: name + column names + types
                        columns = inspector.get_columns(table_name, schema=schema)
                        col_signatures = [
                            f"{col['name']}:{col['type']}" 
                            for col in columns
                        ]
                        
                        table_sig = f"{schema}.{table_name}:{','.join(sorted(col_signatures))}"
                        fingerprint_data.append(table_sig)
                except Exception as e:
                    logger.warning(f"Error fingerprinting schema {schema}: {e}")
                    continue
            
            # Generate hash
            fingerprint_str = '|'.join(sorted(fingerprint_data))
            return hashlib.sha256(fingerprint_str.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating database fingerprint: {e}")
            # Return a timestamp-based fallback
            return hashlib.sha256(str(time.time()).encode()).hexdigest()
    
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
        try:
            cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
            if cache_age > timedelta(hours=self.cache_hours):
                logger.info(f"Discovery cache expired (age: {cache_age})")
                return False
        except Exception as e:
            logger.warning(f"Error checking cache age: {e}")
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
            # Ensure directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Log sample mode info
            from config.settings import get_discovery_config
            discovery_config = get_discovery_config()
            if discovery_config.sample_mode_enabled:
                logger.info(f"💾 Saving SAMPLE discovery to {self.cache_file}")
                logger.info(f"   Sample settings:")
                if discovery_config.sample_max_tables_per_schema:
                    logger.info(f"     • Max tables/schema: {discovery_config.sample_max_tables_per_schema}")
                if discovery_config.sample_max_views_per_schema:
                    logger.info(f"     • Max views/schema: {discovery_config.sample_max_views_per_schema}")
                if discovery_config.sample_max_stored_procedures:
                    logger.info(f"     • Max stored procedures: {discovery_config.sample_max_stored_procedures}")
                if discovery_config.sample_max_rdl_files:
                    logger.info(f"     • Max RDL files: {discovery_config.sample_max_rdl_files}")
            else:
                logger.info(f"💾 Saving FULL discovery to {self.cache_file}")
            
            # Save discovery data
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save fingerprint
            fingerprint = self.get_database_fingerprint(engine)
            with open(self.fingerprint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'fingerprint': fingerprint,
                    'timestamp': datetime.now().isoformat(),
                    'sample_mode': discovery_config.sample_mode_enabled,  # ADD: Track sample mode in fingerprint
                    'sample_config': {  # ADD: Store sample config for reference
                        'tables_per_schema': discovery_config.sample_max_tables_per_schema,
                        'views_per_schema': discovery_config.sample_max_views_per_schema,
                        'stored_procedures': discovery_config.sample_max_stored_procedures,
                        'rdl_files': discovery_config.sample_max_rdl_files
                    } if discovery_config.sample_mode_enabled else None
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Saved discovery cache to {self.cache_file}")
            logger.info(f"✓ Saved fingerprint to {self.fingerprint_file}")
            
            # Log cache file size
            cache_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Cache file size: {cache_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}", exc_info=True)
            raise  # Re-raise to ensure caller knows save failed


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
        
        # Track expensive objects for logging
        self._expensive_objects = []
    
    @property
    def engine(self) -> Engine:
        """Get or create database engine"""
        if self._engine is None:
            logger.info("Creating database engine...")
            self._engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
                pool_size=20,  # Increase pool size for concurrent operations
                max_overflow=10
            )

            from sqlalchemy import types as satypes

            dialect = self._engine.dialect
            # Add your custom SQL Server UDT aliases here
            dialect.ischema_names.setdefault("TableID", satypes.INTEGER)
            dialect.ischema_names.setdefault("MoneyValue", satypes.DECIMAL)
            dialect.ischema_names.setdefault("PercentageValue", satypes.INTEGER)

            logger.debug("Registered custom SQL types: TableID, MoneyValue")
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
        logger.info(f"Max workers: {self.discovery_config.max_workers}")
        logger.info(f"View timeout: {self.discovery_config.view_sampling_timeout}s")
        
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
            
            # Step 4: Detect relationships
            if skip_relationships or not self.relationship_config.enabled:
                logger.info("Skipping relationship detection")
                relationships = []
            else:
                logger.info("Starting relationship detection...")
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
                    'expensive_objects': self._expensive_objects
                }
            }
            
            # Save to cache
            self.cache.save(self.discovery_data, self.engine)
            
            elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"DISCOVERY COMPLETE in {elapsed:.1f}s")
            logger.info("=" * 80)

            # AFTER LINE ~350 (after introspection stage):
            
            # Log expensive objects summary
            if self._expensive_objects:
                logger.warning(f"Found {len(self._expensive_objects)} expensive objects:")
                for obj in self._expensive_objects[:10]:  # Show top 10
                    logger.warning(f"  - {obj['name']}: {obj['duration']:.1f}s ({obj['reason']})")
            
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
                    with self.engine.connect() as conn:
                        result = conn.execute(text("SELECT @@VERSION")).scalar()
                    version_match = re.search(r'Microsoft SQL Server (\d+)', str(result))
                    if version_match:
                        version = version_match.group(1)
                elif dialect in ['postgres', 'postgresql']:
                    with self.engine.connect() as conn:
                        result = conn.execute(text("SELECT version()")).scalar()
                    version_match = re.search(r'PostgreSQL ([\d.]+)', str(result))
                    if version_match:
                        version = version_match.group(1)
                elif dialect == 'mysql':
                    with self.engine.connect() as conn:
                        result = conn.execute(text("SELECT version()")).scalar()
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
        
        schema_names = self.inspector.get_schema_names()
        logger.info(f"Found {len(schema_names)} schemas to process")
        
        for idx, schema_name in enumerate(schema_names, 1):
            # Skip excluded schemas
            if self._should_exclude_schema(schema_name):
                logger.debug(f"Skipping excluded schema: {schema_name}")
                continue
            
            logger.info(f"[{idx}/{len(schema_names)}] Discovering schema: {schema_name}")
            
            # Discover tables in this schema
            tables_data = self._discover_tables(schema_name)
            
            if tables_data:
                schemas_data.append({
                    'name': schema_name,
                    'tables': tables_data
                })
                logger.info(f"  ✓ Schema {schema_name}: {len(tables_data)} objects")
        
        return schemas_data
    
    # REPLACE ENTIRE METHOD starting at line ~217 with:

    def _discover_tables(self, schema_name: str) -> List[Dict]:
        """
        Discover all tables and views in a schema with optional sampling
        """
        tables_data = []
        
        # Get all table names
        try:
            table_names = self.inspector.get_table_names(schema=schema_name)
        except Exception as e:
            logger.error(f"Failed to get table names for schema {schema_name}: {e}")
            return []
        
        # Get all view names
        try:
            view_names = self.inspector.get_view_names(schema=schema_name)
        except Exception as e:
            logger.warning(f"Failed to get view names for schema {schema_name}: {e}")
            view_names = []
        
        # Apply sampling limits if enabled
        if self.discovery_config.sample_mode_enabled:
            logger.info(f"  📊 SAMPLE MODE ACTIVE for schema '{schema_name}'")
            
            # Sample tables (with type safety)
            max_tables = self.discovery_config.sample_max_tables_per_schema
            if max_tables is not None and isinstance(max_tables, int) and max_tables > 0:
                original_count = len(table_names)
                table_names = table_names[:max_tables]
                logger.info(f"  📉 Tables: {len(table_names)}/{original_count} (sampled)")
            
            # Sample views (with type safety)
            max_views = self.discovery_config.sample_max_views_per_schema
            if max_views is not None and isinstance(max_views, int) and max_views > 0:
                original_count = len(view_names)
                view_names = view_names[:max_views]
                logger.info(f"  📉 Views: {len(view_names)}/{original_count} (sampled)")
        
        # Process tables
        for idx, table_name in enumerate(table_names, 1):
            if self._should_exclude_table(table_name):
                logger.debug(f"Skipping excluded table: {table_name}")
                continue
            
            logger.info(f"    [{idx}/{len(table_names)}] Table: {schema_name}.{table_name}")
            table_data = self._discover_table(schema_name, table_name, 'table')

            
            if table_data:
                tables_data.append(table_data)
        
        # Process views
        for idx, view_name in enumerate(view_names, 1):
            if self._should_exclude_table(view_name):
                logger.debug(f"Skipping excluded view: {view_name}")
                continue
            
            logger.info(f"    [{idx}/{len(view_names)}] View: {schema_name}.{view_name}")
            view_data = self._discover_table(schema_name, view_name, 'view')

            
            if view_data:
                tables_data.append(view_data)
        
        return tables_data
    
    def _discover_table_metadata_only(
        self, 
        schema_name: str, 
        table_name: str, 
        table_type: str
    ) -> Optional[Dict]:
        """
        Discover table/view metadata WITHOUT sampling data
        Used as fallback for expensive views
        
        Args:
            schema_name: Schema name
            table_name: Table/view name
            table_type: 'table' or 'view'
            
        Returns:
            Table metadata dictionary (no samples)
        """
        try:
            # Get columns (metadata only, no sampling)
            columns_info = self.inspector.get_columns(table_name, schema=schema_name)
            columns_data = []
            
            for col in columns_info:
                col_data = {
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                    'default': str(col.get('default')) if col.get('default') else None,
                    'stats': None  # No stats for metadata-only
                }
                columns_data.append(col_data)
            
            # Get primary key
            try:
                pk_constraint = self.inspector.get_pk_constraint(table_name, schema=schema_name)
                primary_key = pk_constraint.get('constrained_columns', []) if pk_constraint else []
            except:
                primary_key = []
            
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
            except:
                pass
            
            return {
                'name': table_name,
                'type': table_type,
                'columns': columns_data,
                'primary_key': primary_key,
                'foreign_keys': foreign_keys_data,
                'indexes': [],
                'rowcount_sample': None,
                'sample_rows': [],  # No samples
                'source_assets': [],
                'sampling_failed': True,
                'failure_reason': 'metadata_only'
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata for {schema_name}.{table_name}: {e}")
            return None

    def _discover_views_concurrent(self, schema_name: str, view_names: List[str]) -> List[Dict]:
        """
        Process views concurrently with timeout protection and fallback strategies
        
        Args:
            schema_name: Schema name
            view_names: List of view names to process
            
        Returns:
            List of view data dictionaries
        """
        if not view_names:
            return []
        
        max_workers = self.discovery_config.max_workers
        logger.info(f"    Processing views with {max_workers} workers")
        
        views_data = []
        completed = 0
        timeout_per_view = self.discovery_config.view_sampling_timeout
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all view discovery tasks
            future_to_view = {
                executor.submit(
                    self._discover_view_with_fallback, 
                    schema_name, 
                    view_name,
                    timeout_per_view
                ): view_name
                for view_name in view_names
            }
            
            # Process results as they complete
            for future in as_completed(future_to_view):
                view_name = future_to_view[future]
                completed += 1
                
                try:
                    view_data = future.result(timeout=timeout_per_view + 10)
                    if view_data:
                        views_data.append(view_data)
                        
                    # Log progress every 10%
                    if completed % max(1, len(view_names) // 10) == 0:
                        pct = 100 * completed / len(view_names)
                        logger.info(
                            f"    Views progress: {completed}/{len(view_names)} ({pct:.0f}%)"
                        )
                        
                except FutureTimeoutError:
                    logger.warning(
                        f"⏱️ Timeout processing view {schema_name}.{view_name} - "
                        f"attempting metadata-only fallback"
                    )
                    # Try metadata-only fallback
                    try:
                        view_data = self._discover_table_metadata_only(schema_name, view_name, 'view')
                        if view_data:
                            views_data.append(view_data)
                            self._expensive_objects.append({
                                'name': f"{schema_name}.{view_name}",
                                'type': 'view',
                                'reason': 'timeout',
                                'duration': timeout_per_view
                            })
                    except Exception as e:
                        logger.error(f"Metadata fallback failed for {schema_name}.{view_name}: {e}")
                        
                except Exception as e:
                    logger.error(f"Error processing view {schema_name}.{view_name}: {e}")
        
        logger.info(f"    ✓ Completed {len(views_data)}/{len(view_names)} views")
        return views_data

    def _discover_view_with_fallback(
        self, 
        schema_name: str, 
        view_name: str, 
        timeout_seconds: int
    ) -> Optional[Dict]:
        """
        Discover a view with timeout protection using thread-based approach
        
        Strategy:
        1. Try full discovery with timeout
        2. If timeout, return None (outer function will handle fallback)
        
        Args:
            schema_name: Schema name
            view_name: View name
            timeout_seconds: Timeout in seconds
            
        Returns:
            View data dictionary or None
        """
        start_time = time.time()
        
        try:
            # Use a nested executor with timeout for the actual discovery
            with ThreadPoolExecutor(max_workers=1) as inner_executor:
                future = inner_executor.submit(self._discover_table, schema_name, view_name, 'view')
                view_data = future.result(timeout=timeout_seconds)
                
                elapsed = time.time() - start_time
                if elapsed > 10:  # Log if took more than 10 seconds
                    logger.info(f"      View {schema_name}.{view_name}: {elapsed:.1f}s")
                    if elapsed > 60:  # Track expensive objects
                        self._expensive_objects.append({
                            'name': f"{schema_name}.{view_name}",
                            'type': 'view',
                            'reason': 'slow_query',
                            'duration': elapsed
                        })
                
                return view_data
                
        except FutureTimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"⏱️ View {schema_name}.{view_name} timed out after {elapsed:.1f}s")
            raise  # Re-raise to be caught by outer function
            
        except Exception as e:
            logger.error(f"Error discovering view {schema_name}.{view_name}: {e}")
            return None
    
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
        start_time = time.time()
        
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
                
                # Get statistics and samples (OPTIONAL - can be disabled)
                if getattr(self.discovery_config, 'collect_stats', True):
                    try:
                        stats = self._get_column_stats(schema_name, table_name, col['name'], str(col['type']))
                        if stats:
                            col_data['stats'] = stats
                    except Exception as e:
                        logger.debug(f"Could not get stats for {schema_name}.{table_name}.{col['name']}: {e}")
                
                columns_data.append(col_data)
            
            # Get primary key
            try:
                pk_constraint = self.inspector.get_pk_constraint(table_name, schema=schema_name)
                primary_key = pk_constraint.get('constrained_columns', []) if pk_constraint else []
            except:
                primary_key = []
            
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
            
            # Get row count sample (optional, can timeout)
            rowcount_sample = None
            if getattr(self.discovery_config, 'collect_row_counts', False):
                try:
                    rowcount_sample = self._get_rowcount_sample(schema_name, table_name)
                except Exception as e:
                    logger.debug(f"Could not get rowcount for {schema_name}.{table_name}: {e}")
            
            # Get sample rows
            sample_rows = []
            if getattr(self.discovery_config, 'collect_samples', True):
                try:
                    sample_rows = self._get_sample_rows(schema_name, table_name)
                except Exception as e:
                    logger.debug(f"Could not get sample rows for {schema_name}.{table_name}: {e}")
            
            elapsed = time.time() - start_time
            if elapsed > 30:  # Track slow tables
                self._expensive_objects.append({
                    'name': f"{schema_name}.{table_name}",
                    'type': table_type,
                    'reason': 'slow_discovery',
                    'duration': elapsed
                })
            
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
        Get statistics for a column with timeout protection
        Returns dict with distinct_count, null_rate, min, max, sample_values
        
        NOTE: This can be VERY expensive. Consider disabling via config.
        """
        timeout_seconds = 30  # Per-column timeout
        
        def _execute_stats_query():
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
            
            schema_q = f"{quote_char}{schema}{quote_end}"
            table_q = f"{quote_char}{table}{quote_end}"
            column_q = f"{quote_char}{column}{quote_end}"
            

            # Fast stats query - sam`ple rows FIRST, then aggregate (much faster!)
            if dialect == 'mssql':
                query = text(f"""
                    WITH sample AS (
                        SELECT TOP 1000 {column_q}
                        FROM {schema_q}.{table_q} WITH (NOLOCK)
                    )
                    SELECT 
                        COUNT(DISTINCT {column_q}) as distinct_count,
                        SUM(CASE WHEN {column_q} IS NULL THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as null_rate,
                        MIN({column_q}) as min_val,
                        MAX({column_q}) as max_val
                    FROM sample
                """)
            else:
                query = text(f"""
                    WITH sample AS (
                        SELECT {column_q}
                        FROM {schema_q}.{table_q}
                        LIMIT 1000
                    )
                    SELECT
                        COUNT(DISTINCT {column_q}) as distinct_count,
                        CAST(SUM(CASE WHEN {column_q} IS NULL THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) as null_rate,
                        MIN({column_q}) as min_val,
                        MAX({column_q}) as max_val
                    FROM sample
                """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
            return result
        
        try:
            # Execute with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_stats_query)
                result = future.result(timeout=timeout_seconds)
            
            stats = {
                'distinct_count': result[0] if result[0] is not None else 0,
                'null_rate': float(result[1]) if result[1] is not None else 0.0,
            }
            
            # Add min/max for appropriate types
            if any(t in col_type.lower() for t in ['int', 'float', 'decimal', 'numeric', 'date', 'time']):
                stats['min'] = str(result[2]) if result[2] is not None else None
                stats['max'] = str(result[3]) if result[3] is not None else None
            
            # Get sample values (top 5) - separate query with timeout
            sample_values = self._get_sample_values(schema, table, column, limit=5)
            if sample_values:
                stats['sample_values'] = sample_values
            
            # Detect currency hint
            if any(t in col_type.lower() for t in ['decimal', 'numeric', 'money']):
                if any(kw in column.lower() for kw in ['amount', 'price', 'cost', 'revenue', 'total', 'discount']):
                    stats['unit_hint'] = 'currency'
                    stats['currency_hint'] = 'USD'
            
            return stats
            
        except FutureTimeoutError:
            logger.debug(f"Stats query timed out for {schema}.{table}.{column}")
            return None
        except Exception as e:
            logger.debug(f"Error getting stats for {schema}.{table}.{column}: {e}")
            return None
    
    def _get_sample_values(self, schema: str, table: str, column: str, limit: int = 5) -> List[str]:
        """Get sample distinct values from a column with timeout"""
        timeout_seconds = 10
        
        def _execute_query():
            dialect = self.engine.dialect.name.lower()
            
            # if dialect == 'mssql':
            #     query = text(f"""
            #         SELECT DISTINCT TOP {limit} [{column}]
            #         FROM [{schema}].[{table}] WITH (NOLOCK)
            #         WHERE [{column}] IS NOT NULL
            #         ORDER BY [{column}]
            #     """)
            # else:
            #     query = text(f"""
            #         SELECT DISTINCT "{column}"
            #         FROM "{schema}"."{table}"
            #         WHERE "{column}" IS NOT NULL
            #         ORDER BY "{column}"
            #         LIMIT {limit}
            #     """)
            if dialect == 'mssql':
                query = text(f"""
                    SELECT TOP {limit} [{column}]
                    FROM [{schema}].[{table}] WITH (NOLOCK)
                """)
            else:
                query = text(f"""
                    SELECT "{column}"
                    FROM "{schema}"."{table}"
                    LIMIT {limit}
                """)

            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            return result
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_query)
                result = future.result(timeout=timeout_seconds)
            return [str(row[0]) for row in result]
        except:
            return []
    
    def _get_rowcount_sample(self, schema: str, table: str) -> int:
        """Get approximate row count with timeout"""
        timeout_seconds = 30
        
        def _execute_query():
            dialect = self.engine.dialect.name.lower()
            
            if dialect == 'mssql':
                # Use sys.partitions for fast estimate
                query = text(f"""
                    SELECT SUM(p.rows) 
                    FROM sys.partitions p
                    JOIN sys.tables t ON p.object_id = t.object_id
                    JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE s.name = :schema AND t.name = :table AND p.index_id IN (0,1)
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(query, {"schema": schema, "table": table}).scalar()
            else:
                query = text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
                with self.engine.connect() as conn:
                    result = conn.execute(query).scalar()
            
            return result
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_query)
                result = future.result(timeout=timeout_seconds)
            return int(result) if result else 0
        except:
            return 0
    
    def _get_sample_rows(self, schema: str, table: str, limit: int = 10) -> List[Dict]:
        """Get sample rows from table/view with timeout protection"""
        timeout_seconds = self.discovery_config.view_sampling_timeout or 60
        
        def _execute_query():
            dialect = self.engine.dialect.name.lower()
            
            if dialect == 'mssql':
                query = text(f"SELECT TOP {limit} * FROM [{schema}].[{table}] WITH (NOLOCK)")
            else:
                query = text(f'SELECT * FROM "{schema}"."{table}" LIMIT {limit}')
            
            with self.engine.connect() as conn:
                result = conn.execute(query).mappings().fetchall()
            return result
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_query)
                result = future.result(timeout=timeout_seconds)
            
            if not result:
                return []
            
            rows = []
            for r in result:
                rows.append({k: (None if v is None else str(v)) for k, v in r.items()})
            
            return rows
            
        except FutureTimeoutError:
            logger.debug(f"Sample query timed out for {schema}.{table}")
            return []
        except Exception as e:
            logger.debug(f"Error sampling {schema}.{table}: {e}")
            return []
    
    def _discover_named_assets(self) -> List[Dict]:
        """
        Discover named assets: views, stored procedures, RDL files
        With optional sampling for testing mode
        Returns list of asset dictionaries
        """
        if hasattr(self, '_named_assets_cache'):
            return self._named_assets_cache
        
        assets = []
        
        # Sample mode tracking
        sp_count = 0
        sp_limit = None
        rdl_count = 0
        rdl_limit = None
        
        if self.discovery_config.sample_mode_enabled:
            sp_limit = self.discovery_config.sample_max_stored_procedures
            rdl_limit = self.discovery_config.sample_max_rdl_files
            logger.info(f"📊 SAMPLE MODE: Stored Procedures limit={sp_limit}, RDL limit={rdl_limit}")
        
        # ====================
        # 1. DISCOVER RDL FILES
        # ====================
        rdl_path = self.path_config.rdl_path
        if rdl_path and rdl_path.exists():
            logger.info(f"Scanning RDL files in {rdl_path}")
            try:
                from src.discovery.rdl_parser import RDLParser

                parser = RDLParser(self.settings)
                all_rdl_files = parser.find_rdl_files()

                logger.info(f"  Found {len(all_rdl_files)} RDL files")
                for rdl_file in all_rdl_files:
                    if rdl_limit and rdl_count >= rdl_limit:
                        logger.info(f"  📉 Reached RDL file limit ({rdl_limit})")
                        break

                    try:
                        rdl_asset = parser.parse_rdl_file(rdl_file)
                        if not rdl_asset:
                            continue

                        # 1) Keep the file-level asset (optional but useful for context)
                        assets.append(rdl_asset)
                        rdl_count += 1

                        # 2) Emit one named asset per dataset with normalized SQL, fields, params
                        for ds in rdl_asset.get("datasets", []):
                            assets.append({
                                "kind": "rdl_dataset",
                                "name": f"{rdl_asset['name']}.{ds.get('name')}",
                                "source_file": rdl_asset["path"],
                                "sql_raw": ds.get("query"),
                                "sql_normalized": ds.get("sql_normalized"),
                                "fields": ds.get("fields", []),
                                "parameters": rdl_asset.get("parameters", []),
                                "data_sources": rdl_asset.get("data_sources", [])
                            })

                    except Exception as e:
                        logger.warning(f"  Failed to parse RDL file {rdl_file}: {e}")
                
            except Exception as e:
                logger.error(f"Error parsing RDL files: {e}")
        
        # ====================
        # 2. DISCOVER VIEWS (SQL NORMALIZATION)
        # ====================
        logger.info("Discovering views with SQL normalization...")
        try:
            import sqlglot
            
            view_count = 0
            
            for schema_name in self.inspector.get_schema_names():
                if self._should_exclude_schema(schema_name):
                    continue
                
                try:
                    view_names = self.inspector.get_view_names(schema=schema_name)
                except Exception as e:
                    logger.debug(f"Could not get views for schema {schema_name}: {e}")
                    continue
                
                for view_name in view_names:
                    if self._should_exclude_table(view_name):
                        continue
                    
                    try:
                        # Get view definition
                        view_def = self.inspector.get_view_definition(view_name, schema=schema_name)
                        
                        if not view_def:
                            continue
                        
                        # Normalize SQL with sqlglot
                        try:
                            parsed = sqlglot.parse_one(view_def, dialect=self.engine.dialect.name)
                            normalized_sql = parsed.sql(dialect=self.engine.dialect.name, pretty=True)
                        except Exception as e:
                            logger.debug(f"Could not normalize view SQL for {schema_name}.{view_name}: {e}")
                            normalized_sql = view_def  # Use raw SQL if normalization fails
                        
                        # Add view as named asset
                        assets.append({
                            'kind': 'view',
                            'name': f"{schema_name}.{view_name}",
                            'sql_normalized': normalized_sql,
                            'sql_raw': view_def
                        })
                        
                        view_count += 1
                        
                    except Exception as e:
                        logger.debug(f"Could not process view {schema_name}.{view_name}: {e}")
            
            logger.info(f"  ✓ Found {view_count} views with SQL")
            
        except Exception as e:
            logger.error(f"View discovery failed: {e}")
        
        # ====================
        # 3. DISCOVER STORED PROCEDURES
        # ====================
        logger.info("Discovering stored procedures...")
        
        try:
            dialect = self.engine.dialect.name.lower()
            
            if dialect == 'mssql':
                # SQL Server stored procedure discovery
                query = text("""
                    SELECT 
                        SCHEMA_NAME(p.schema_id) as schema_name,
                        p.name as procedure_name,
                        m.definition as procedure_definition
                    FROM sys.procedures p
                    INNER JOIN sys.sql_modules m ON p.object_id = m.object_id
                    WHERE p.is_ms_shipped = 0
                    ORDER BY schema_name, procedure_name
                """)
                
                with self.engine.connect() as conn:
                    result = conn.execute(query).fetchall()
                
                for row in result:
                    # Check sample limit
                    if sp_limit and sp_count >= sp_limit:
                        logger.info(f"  📉 Reached stored procedure limit ({sp_limit})")
                        break
                    
                    schema_name = row[0]
                    sp_name = row[1]
                    sp_definition = row[2]
                    
                    if self._should_exclude_schema(schema_name):
                        continue
                    
                    if self._should_exclude_table(sp_name):
                        continue
                    
                    try:
                        # Normalize SQL with sqlglot
                        import sqlglot
                        
                        try:
                            parsed = sqlglot.parse_one(sp_definition, dialect='tsql')
                            normalized_sql = parsed.sql(dialect='tsql', pretty=True)
                        except Exception as e:
                            logger.debug(f"Could not normalize SP SQL for {schema_name}.{sp_name}: {e}")
                            normalized_sql = sp_definition
                        
                        # Add stored procedure as named asset
                        assets.append({
                            'kind': 'stored_procedure',
                            'name': f"{schema_name}.{sp_name}",
                            'sql_normalized': normalized_sql,
                            'sql_raw': sp_definition
                        })
                        
                        sp_count += 1
                        
                    except Exception as e:
                        logger.debug(f"Could not process stored procedure {schema_name}.{sp_name}: {e}")
                
                logger.info(f"  ✓ Found {sp_count} stored procedures")
            
            elif dialect in ['postgresql', 'postgres']:
                # PostgreSQL function discovery
                query = text("""
                    SELECT 
                        n.nspname as schema_name,
                        p.proname as function_name,
                        pg_get_functiondef(p.oid) as function_definition
                    FROM pg_proc p
                    INNER JOIN pg_namespace n ON p.pronamespace = n.oid
                    WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY schema_name, function_name
                """)
                
                with self.engine.connect() as conn:
                    result = conn.execute(query).fetchall()
                
                for row in result:
                    # Check sample limit
                    if sp_limit and sp_count >= sp_limit:
                        logger.info(f"  📉 Reached stored procedure limit ({sp_limit})")
                        break
                    
                    schema_name = row[0]
                    func_name = row[1]
                    func_definition = row[2]
                    
                    if self._should_exclude_schema(schema_name):
                        continue
                    
                    assets.append({
                        'kind': 'stored_procedure',
                        'name': f"{schema_name}.{func_name}",
                        'sql_normalized': func_definition,
                        'sql_raw': func_definition
                    })
                    
                    sp_count += 1
                
                logger.info(f"  ✓ Found {sp_count} stored procedures (PostgreSQL)")
            
            elif dialect == 'mysql':
                # MySQL stored procedure discovery
                query = text("""
                    SELECT 
                        ROUTINE_SCHEMA as schema_name,
                        ROUTINE_NAME as routine_name,
                        ROUTINE_DEFINITION as routine_definition
                    FROM information_schema.ROUTINES
                    WHERE ROUTINE_SCHEMA NOT IN ('mysql', 'information_schema', 'performance_schema', 'sys')
                    ORDER BY schema_name, routine_name
                """)
                
                with self.engine.connect() as conn:
                    result = conn.execute(query).fetchall()
                
                for row in result:
                    # Check sample limit
                    if sp_limit and sp_count >= sp_limit:
                        logger.info(f"  📉 Reached stored procedure limit ({sp_limit})")
                        break
                    
                    schema_name = row[0]
                    routine_name = row[1]
                    routine_definition = row[2]
                    
                    if self._should_exclude_schema(schema_name):
                        continue
                    
                    assets.append({
                        'kind': 'stored_procedure',
                        'name': f"{schema_name}.{routine_name}",
                        'sql_normalized': routine_definition,
                        'sql_raw': routine_definition
                    })
                    
                    sp_count += 1
                
                logger.info(f"  ✓ Found {sp_count} stored procedures (MySQL)")
            
            else:
                logger.warning(f"Stored procedure discovery not implemented for dialect: {dialect}")
        
        except Exception as e:
            logger.error(f"Stored procedure discovery failed: {e}", exc_info=True)
        
        # Cache for reuse
        self._named_assets_cache = assets
        
        # Summary
        logger.info("=" * 80)
        logger.info(f"NAMED ASSETS SUMMARY")
        logger.info(f"  Views:              {len([a for a in assets if a.get('kind') == 'view'])}")
        logger.info(f"  Stored Procedures:  {len([a for a in assets if a.get('kind') == 'stored_procedure'])}")
        logger.info(f"  RDL Files:          {len([a for a in assets if a.get('kind') == 'rdl'])}")
        logger.info(f"  Total Assets:       {len(assets)}")
        if self.discovery_config.sample_mode_enabled:
            logger.info(f"  📊 Sample Mode:     ACTIVE")
        logger.info("=" * 80)
        
        return assets
    
    def _detect_relationships(self, schemas_data: List[Dict], dialect: str) -> List[Dict]:
        """
        Detect foreign key relationships using multiple methods
        
        Args:
            schemas_data: List of schema dictionaries from discovery
            dialect: Database dialect
            
        Returns:
            List of unique relationship dictionaries
        """
        start_time = time.time()
        all_relationships = []
        
        # Build discovery data for relationship detector
        discovery_for_relationships = {
            'database': {'vendor': dialect, 'version': 'unknown'},
            'dialect': dialect,
            'schemas': schemas_data,
            'named_assets': getattr(self, '_named_assets_cache', [])
        }
        
        # Method 1 & 2: Standard detection (explicit + implicit FKs)
        try:
            from src.discovery.relationship_detector import detect_relationships
            
            relationships = detect_relationships(
                connection_string=self.connection_string,
                discovery_data=discovery_for_relationships,
                config=self.relationship_config
            )
            all_relationships.extend(relationships)
            logger.info(f"  ✓ Standard detection: {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Standard relationship detection failed: {e}", exc_info=True)
        
        # Method 3: View join analysis
        if self.relationship_config.detect_views:
            try:
                from src.discovery.relationship_detector import detect_relationships_from_views
                
                view_relationships = detect_relationships_from_views(
                    discovery_data=discovery_for_relationships,
                    config=self.relationship_config
                )
                all_relationships.extend(view_relationships)
                logger.info(f"  ✓ View join analysis: {len(view_relationships)} relationships")
                
            except Exception as e:
                logger.error(f"View relationship detection failed: {e}", exc_info=True)
        
        # Method 4: RDL join analysis
        if self.relationship_config.detect_rdl_joins:
            try:
                from src.discovery.rdl_parser import extract_relationships_from_rdl
                
                rdl_relationships = extract_relationships_from_rdl(
                    rdl_assets=discovery_for_relationships['named_assets'],
                    config=self.relationship_config
                )
                all_relationships.extend(rdl_relationships)
                logger.info(f"  ✓ RDL join analysis: {len(rdl_relationships)} relationships")
                
            except Exception as e:
                logger.error(f"RDL relationship detection failed: {e}", exc_info=True)
        
        # Deduplicate
        unique_relationships = self._deduplicate_relationships(all_relationships)
        
        elapsed = time.time() - start_time
        logger.info(f"  ═══════════════════════════════════")
        logger.info(f"  Relationships: {len(all_relationships)} found, {len(unique_relationships)} unique")
        logger.info(f"  Duration: {elapsed:.2f}s")
        logger.info(f"  ═══════════════════════════════════")
        
        return unique_relationships

    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """
        Deduplicate relationships, preferring higher confidence sources
        
        Priority order:
        1. RDL join analysis (most curated)
        2. View join analysis (curated)
        3. Explicit FKs (database-defined)
        4. Implicit FKs (value overlap)
        """
        seen = {}
        
        priority = {
            "rdl_join_analysis": 4,
            "view_join_analysis": 3,
            "explicit_fk": 2,
            "value_overlap": 1,
            "name_pattern": 1
        }
        
        for rel in relationships:
            key = (rel["from"], rel["to"])
            method = rel.get("method", "unknown")
            current_priority = priority.get(method, 0)
            
            if key not in seen:
                seen[key] = rel
            else:
                existing_priority = priority.get(seen[key].get("method", ""), 0)
                if current_priority > existing_priority:
                    seen[key] = rel
        
        return list(seen.values())


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_discovery(
    use_cache: bool = True, 
    skip_relationships: bool = False,
    connection_string: Optional[str] = None
) -> Dict:
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