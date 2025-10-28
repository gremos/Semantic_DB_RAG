"""
Discovery Engine - Main orchestrator for Phase 1.
Coordinates introspection, sampling, relationship detection, and caching.
"""

import json
import hashlib
import time
import concurrent.futures
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import queue
import threading
import re

from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError

from config.settings import Settings
from src.discovery.introspector import DatabaseIntrospector
from src.discovery.sampler import DataSampler
from src.discovery.relationship_detector import RelationshipDetector
from src.discovery.rdl_parser import RDLParser
from src.utils.cache import CacheManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DiscoveryEngine:
    """Main engine for database discovery (Phase 1)."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_manager = CacheManager(settings)
        self.max_concurrency = 10  # Max concurrent sampling operations
        self.progress_lock = threading.Lock()
        self.completed_tables = 0
        self.total_tables = 0
        self.partial_results_queue = queue.Queue()  # For tables with partial results
        
        # Create a shared connection engine to reuse
        self.engine = self._create_engine()
        
        # Extract dialect from connection string
        self.db_dialect = self._detect_database_dialect()
        self.db_version = self._detect_database_version()
        
        logger.info(f"Detected database dialect: {self.db_dialect}, version: {self.db_version}")
    
    def _create_engine(self):
        """
        Create a SQLAlchemy engine with appropriate timeout settings.
        """
        try:
            # Get connection string from settings
            conn_str = self.settings.DATABASE_CONNECTION_STRING
            
            # Create engine with read-only connection parameters
            if 'sqlserver' in conn_str.lower() or 'mssql' in conn_str.lower():
                # SQL Server specific params for read-only
                if 'ApplicationIntent' not in conn_str:
                    if '?' in conn_str:
                        conn_str += '&ApplicationIntent=ReadOnly'
                    else:
                        conn_str += '?ApplicationIntent=ReadOnly'
            
            # Create engine with appropriate connection pooling
            engine = create_engine(
                conn_str,
                pool_size=self.max_concurrency + 2,  # Add a few extra connections
                max_overflow=5,
                pool_timeout=60,  # 60 seconds timeout for getting a connection
                pool_recycle=300  # Recycle connections every 5 minutes
            )
            
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logger.info(f"Successfully created database engine")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create database engine: {str(e)}")
            # Still return None so initialization continues, we'll handle this later
            return None
    
    def _detect_database_dialect(self) -> str:
        """
        Detect the database dialect from the connection string and engine.
        
        Returns:
            Database dialect string: 'mssql', 'mysql', 'postgresql', 'oracle', or 'generic'
        """
        # First try to get dialect from the engine
        if self.engine is not None:
            try:
                dialect_name = self.engine.dialect.name
                if dialect_name:
                    return dialect_name.lower()
            except:
                pass  # Fall back to connection string detection
        
        # Fall back to connection string detection
        conn_str = self.settings.DATABASE_CONNECTION_STRING.lower()
        
        if 'sqlserver' in conn_str or 'mssql' in conn_str:
            return 'mssql'
        elif 'mysql' in conn_str or 'mariadb' in conn_str:
            return 'mysql'
        elif 'postgresql' in conn_str or 'postgres' in conn_str:
            return 'postgresql'
        elif 'oracle' in conn_str:
            return 'oracle'
        elif 'sqlite' in conn_str:
            return 'sqlite'
        else:
            logger.warning(f"Could not detect database dialect from connection string. Using 'generic'.")
            return 'generic'
    
    def _detect_database_version(self) -> str:
        """
        Detect the database version using a simple query.
        
        Returns:
            Database version string or 'unknown'
        """
        if self.engine is None:
            return 'unknown'
            
        try:
            with self.engine.connect() as conn:
                if self.db_dialect == 'mssql':
                    result = conn.execute(text("SELECT @@VERSION AS version"))
                elif self.db_dialect == 'mysql':
                    result = conn.execute(text("SELECT VERSION() AS version"))
                elif self.db_dialect == 'postgresql':
                    result = conn.execute(text("SHOW server_version"))
                elif self.db_dialect == 'oracle':
                    result = conn.execute(text("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%'"))
                else:
                    return 'unknown'
                    
                row = result.fetchone()
                if row:
                    # Extract just the major version number
                    version_str = str(row[0])
                    # Look for the first numeric part
                    match = re.search(r'(\d+)\.(\d+)', version_str)
                    if match:
                        return f"{match.group(1)}.{match.group(2)}"
                    else:
                        return version_str[:20]  # Truncate to avoid huge version strings
                        
        except Exception as e:
            logger.warning(f"Could not detect database version: {str(e)}")
            
        return 'unknown'
        
    def _generate_fingerprint(self) -> str:
        """
        Generate a fingerprint for the database connection.
        Used to detect if the database has changed.
        """
        # Use connection string as basis (without password)
        conn_str = self.settings.DATABASE_CONNECTION_STRING
        # Remove password from connection string for fingerprint
        conn_str_safe = conn_str.split('@')[-1] if '@' in conn_str else conn_str
        
        fingerprint_data = f"{conn_str_safe}:{self.settings.SCHEMA_EXCLUSIONS}:{self.settings.TABLE_EXCLUSIONS}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    def discover(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run full discovery process with concurrent sampling.
        
        Steps:
        1. Check cache
        2. Introspect database schema
        3. Sample data and collect statistics (now concurrent)
        4. Detect implicit relationships
        5. Parse RDL files
        6. Cache results
        
        Args:
            force_refresh: If True, bypass cache and force fresh discovery
            
        Returns:
            Discovery JSON structure
        """
        logger.info("Starting discovery process...")
        
        # Check cache unless force refresh
        if not force_refresh:
            cached = self.cache_manager.get_discovery()
            if cached:
                logger.info("Using cached discovery results")
                return cached
        
        # Verify engine is available
        if self.engine is None:
            logger.error("Database engine is not available. Attempting to recreate.")
            self.engine = self._create_engine()
            if self.engine is None:
                logger.error("Failed to create database engine. Cannot proceed with discovery.")
                raise RuntimeError("Database connection failed")
        
        # Generate database fingerprint
        fingerprint = self._generate_fingerprint()
        logger.info(f"Database fingerprint: {fingerprint}")
        
        # Step 1: Introspect database
        logger.info("Step 1/4: Introspecting database schema...")
        with DatabaseIntrospector(self.settings) as introspector:
            discovery_data = introspector.introspect_full()
        
        logger.info(f"Found {len(discovery_data['schemas'])} schemas")
        
        # Add database info to discovery data
        if 'database' not in discovery_data:
            discovery_data['database'] = {}
        
        discovery_data['database']['dialect'] = self.db_dialect
        discovery_data['database']['version'] = self.db_version
        
        # Step 2: Sample data and collect statistics concurrently
        logger.info(f"Step 2/4: Sampling data and collecting statistics with {self.max_concurrency} concurrent workers...")
        self._sample_data_concurrently(discovery_data)
        
        # Step 3: Detect implicit relationships
        logger.info("Step 3/4: Detecting implicit relationships...")
        detector = RelationshipDetector(self.settings)
        discovery_data['inferred_relationships'] = detector.detect_relationships(discovery_data)
        
        # Step 4: Parse RDL files (if configured)
        logger.info("Step 4/4: Parsing RDL files...")
        rdl_path = Path(self.settings.RDL_PATH)
        if rdl_path.exists():
            parser = RDLParser(self.settings)
            discovery_data['rdl_assets'] = parser.parse_directory(rdl_path)
        else:
            discovery_data['rdl_assets'] = []
        
        # Record tables with partial or failed results
        partial_results = []
        while not self.partial_results_queue.empty():
            partial_results.append(self.partial_results_queue.get())
        
        if partial_results:
            discovery_data['partial_results'] = partial_results
            logger.warning(f"Discovery completed with {len(partial_results)} partial/failed results")
        
        # Cache the results
        logger.info("Caching discovery results...")
        self.cache_manager.save_discovery(discovery_data)
        
        logger.info("Discovery complete!")
        return discovery_data
    
    def _sample_data_concurrently(self, discovery_data: Dict[str, Any]) -> None:
        """
        Sample data from tables and views concurrently using a thread pool.
        
        Args:
            discovery_data: The discovery data structure to update
        """
        # Collect all tables and views for processing
        all_objects = []
        for schema in discovery_data['schemas']:
            for table in schema['tables']:
                all_objects.append((schema['name'], table))
        
        # Sort objects - prioritize regular tables before views
        # This helps with dependencies where views rely on tables
        all_objects.sort(key=lambda x: 0 if x[1].get('type', 'table') == 'table' else 1)
        
        self.total_tables = len(all_objects)
        self.completed_tables = 0
        
        logger.info(f"Processing {self.total_tables} objects with max {self.max_concurrency} concurrent operations")
        
        # Process objects concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # Submit each object for processing - use own shared engine
            futures = {
                executor.submit(self._sample_object, schema_name, table_obj): 
                (schema_name, table_obj) 
                for schema_name, table_obj in all_objects
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                schema_name, table_obj = futures[future]
                try:
                    # Get the result and update discovery_data
                    result = future.result()
                    if result:
                        # Update the original object in discovery_data
                        self._update_table_in_discovery_data(discovery_data, schema_name, result)
                except Exception as e:
                    logger.error(f"Error processing {schema_name}.{table_obj['name']}: {str(e)}")
                    # Record failed processing
                    self.partial_results_queue.put({
                        "schema": schema_name,
                        "table": table_obj['name'],
                        "error": str(e)
                    })
    
    def _sample_object(self, schema_name: str, table_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sample a single table or view with appropriate optimizations.
        
        Args:
            schema_name: Schema name
            table_obj: Table dictionary to enrich
            
        Returns:
            Updated table dictionary or None if failed
        """
        object_type = table_obj.get('type', 'table')
        object_name = table_obj['name']
        full_name = f"{schema_name}.{object_name}"
        
        logger.debug(f"Sampling {object_type} {full_name}")
        start_time = time.time()
        
        try:
            # Use different sampling approaches based on object type
            if object_type == 'view':
                # Views get optimized sampling with timeouts
                success = self._sample_view_optimized(schema_name, table_obj)
            else:
                # Regular tables use standard sampling
                success = self._sample_table_standard(schema_name, table_obj)
            
            # Update progress
            with self.progress_lock:
                self.completed_tables += 1
                completed = self.completed_tables
                total = self.total_tables
            
            # Log progress periodically
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {completed}/{total} objects ({completed/total:.1%}) - {full_name} took {elapsed:.2f}s")
            
            return table_obj if success else None
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to sample {full_name} after {elapsed:.2f}s: {str(e)}")
            
            # Update progress counter even for failures
            with self.progress_lock:
                self.completed_tables += 1
            
            # Record partial results
            table_obj['sampling_error'] = str(e)
            table_obj['partial_results'] = True
            self.partial_results_queue.put({
                "schema": schema_name,
                "table": object_name,
                "error": str(e)
            })
            
            return table_obj
    
    def _execute_query_with_timeout(self, query: str, timeout_seconds: int) -> List[Dict[str, Any]]:
        """
        Execute a query with a timeout and return the results.
        
        Args:
            query: SQL query to execute
            timeout_seconds: Maximum execution time in seconds
            
        Returns:
            List of row dictionaries
        """
        # Verify engine is available
        if self.engine is None:
            logger.error("Database engine is not available for query execution")
            raise RuntimeError("Database engine is not available")
            
        try:
            # Start timer for manual timeout check
            start_time = time.time()
            
            with self.engine.connect() as connection:
                # Set session timeout if supported by dialect
                try:
                    if self.db_dialect == 'postgresql':
                        connection.execute(text(f"SET statement_timeout = {timeout_seconds * 1000}"))
                    # Don't use QUERY_GOVERNOR_COST_LIMIT as it's not reliable
                except Exception as e:
                    logger.debug(f"Could not set timeout parameters: {str(e)}")
                
                # Execute the query
                result = connection.execute(text(query))
                
                # Collect results as dictionaries
                rows = []
                for row in result:
                    # Check for timeout during iteration
                    if time.time() - start_time > timeout_seconds:
                        logger.warning(f"Query execution timed out after {timeout_seconds}s")
                        break
                    
                    # Convert row to dictionary
                    row_dict = {}
                    for key, value in row._mapping.items():
                        row_dict[key] = value
                    
                    rows.append(row_dict)
                    
                    # Limit to maximum 100 rows regardless
                    if len(rows) >= 100:
                        break
                
                return rows
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            # Determine if it's a timeout error
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.warning(f"Query timed out after {timeout_seconds}s")
            raise
    
    def _sample_view_optimized(self, schema_name: str, view_obj: Dict[str, Any]) -> bool:
        """
        Sample a view with optimizations for performance.
        Uses LIMIT/TOP clauses and query hints to improve performance.
        
        Args:
            schema_name: Schema name
            view_obj: View dictionary to update
            
        Returns:
            True if sampling was successful (even partially), False otherwise
        """
        view_name = view_obj['name']
        full_name = f"{schema_name}.{view_name}"
        
        try:
            # Build optimized query with LIMIT/TOP but without problematic hints
            if self.db_dialect == 'mssql':
                # SQL Server optimization - removed problematic FAST hint
                query = f"""
                SELECT TOP 10 * 
                FROM {schema_name}.{view_name} WITH (NOLOCK)
                """
            elif self.db_dialect == 'mysql':
                # MySQL optimization
                query = f"""
                SELECT * FROM {schema_name}.{view_name} LIMIT 10
                """
            elif self.db_dialect == 'postgresql':
                # PostgreSQL optimization
                query = f"""
                SELECT * FROM {schema_name}.{view_name} LIMIT 10
                """
            elif self.db_dialect == 'oracle':
                # Oracle optimization
                query = f"""
                SELECT * FROM {schema_name}.{view_name}
                WHERE ROWNUM <= 10
                """
            elif self.db_dialect == 'sqlite':
                # SQLite optimization
                query = f"SELECT * FROM {schema_name}.{view_name} LIMIT 10"
            else:
                # Generic optimization
                query = f"SELECT * FROM {schema_name}.{view_name} LIMIT 10"
            
            # Execute the optimized query with timeout
            sample_rows = self._execute_query_with_timeout(
                query, 
                self.settings.DISCOVERY_TIMEOUT
            )
            
            if sample_rows:
                # Successfully got some rows
                view_obj['sample_rows'] = sample_rows[:10]  # Limit to 10 rows
                view_obj['rowcount_sample'] = len(sample_rows)
                
                # Sample each column for statistics
                self._sample_columns_optimized(schema_name, view_obj)
                
                return True
            else:
                # No rows returned, mark as partial
                view_obj['sample_rows'] = []
                view_obj['rowcount_sample'] = 0
                view_obj['partial_results'] = True
                
                # Try to at least sample columns
                self._sample_columns_optimized(schema_name, view_obj)
                
                return False
                
        except Exception as e:
            # Handle timeout or query error
            logger.warning(f"Optimized view sampling failed for {full_name}: {str(e)}")
            
            # Try simplified fallback query to get at least something
            try:
                # Use even more restricted query as fallback with bare minimum syntax
                if self.db_dialect == 'mssql':
                    fallback_query = f"SELECT TOP 2 * FROM {schema_name}.{view_name}"
                else:
                    fallback_query = f"SELECT * FROM {schema_name}.{view_name} LIMIT 2"
                
                sample_rows = self._execute_query_with_timeout(fallback_query, 300)  # 30 second timeout
                
                if sample_rows:
                    # Got something with fallback
                    view_obj['sample_rows'] = sample_rows
                    view_obj['rowcount_sample'] = len(sample_rows)
                    view_obj['partial_results'] = True
                    logger.info(f"Got partial results for {full_name} with fallback query")
                    
                    # Try to sample at least a few columns
                    self._sample_columns_optimized(schema_name, view_obj)
                    
                    return True
                    
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed for {full_name}: {str(fallback_error)}")
            
            # Both attempts failed
            view_obj['sampling_error'] = str(e)
            view_obj['partial_results'] = True
            self.partial_results_queue.put({
                "schema": schema_name,
                "table": view_name,
                "error": str(e)
            })
            
            return False
    
    def _sample_columns_optimized(self, schema_name: str, table_obj: Dict[str, Any]) -> None:
        """
        Sample columns for statistics with optimized queries.
        Uses separate queries with LIMIT/TOP for each column.
        
        Args:
            schema_name: Schema name
            table_obj: Table dictionary to update
        """
        table_name = table_obj['name']
        full_name = f"{schema_name}.{table_name}"
        
        # Process each column with optimized queries
        for column in table_obj.get('columns', []):
            col_name = column['name']
            
            try:
                # Construct optimized query for this column - simplified for compatibility
                if self.db_dialect == 'mssql':
                    # SQL Server optimized column sampling - removed problematic hints
                    query = f"""
                    SELECT TOP 5 [{col_name}] 
                    FROM {schema_name}.{table_name} 
                    WHERE [{col_name}] IS NOT NULL
                    """
                elif self.db_dialect in ('mysql', 'postgresql', 'sqlite'):
                    # Generic optimized column sampling
                    query = f"""
                    SELECT "{col_name}" FROM {schema_name}.{table_name}
                    WHERE "{col_name}" IS NOT NULL
                    LIMIT 5
                    """
                elif self.db_dialect == 'oracle':
                    # Oracle
                    query = f"""
                    SELECT "{col_name}" FROM {schema_name}.{table_name}
                    WHERE "{col_name}" IS NOT NULL AND ROWNUM <= 5
                    """
                else:
                    # Generic
                    query = f"""
                    SELECT {col_name} FROM {schema_name}.{table_name}
                    WHERE {col_name} IS NOT NULL
                    LIMIT 5
                    """
                
                # Execute with short timeout (300 seconds max per column)
                column_samples = self._execute_query_with_timeout(query, 300)
                
                if column_samples:
                    # Extract values for statistics
                    values = [row.get(col_name) for row in column_samples if col_name in row and row.get(col_name) is not None]
                    
                    # Create basic stats
                    stats = {
                        "sample_values": [str(v) for v in values if v is not None],
                        "null_rate": 0.0  # Approximate since we filtered nulls
                    }
                    
                    # Add to column
                    column['stats'] = stats
                else:
                    # No samples, add empty stats
                    column['stats'] = {"sample_values": [], "null_rate": None}
                
            except Exception as col_error:
                logger.warning(f"Error sampling column {full_name}.{col_name}: {str(col_error)}")
                column['stats'] = {"sample_values": [], "null_rate": None}
                column['sampling_error'] = str(col_error)
    
    def _sample_table_standard(self, schema_name: str, table_obj: Dict[str, Any]) -> bool:
        """
        Sample a regular table using the DataSampler approach.
        
        Args:
            schema_name: Schema name
            table_obj: Table dictionary to update
            
        Returns:
            True if sampling was successful, False otherwise
        """
        try:
            # Use direct queries rather than sampler class for more control
            table_name = table_obj['name']
            full_name = f"{schema_name}.{table_name}"
            
            # 1. Get a small sample of rows (up to 10)
            if self.db_dialect == 'mssql':
                query = f"SELECT TOP 10 * FROM {schema_name}.{table_name}"
            else:
                query = f"SELECT * FROM {schema_name}.{table_name} LIMIT 10"
                
            sample_rows = self._execute_query_with_timeout(query, 60)  # 60 second timeout
            
            if sample_rows:
                # Store sample rows
                table_obj['sample_rows'] = sample_rows
                table_obj['rowcount_sample'] = len(sample_rows)
                
                # 2. Get column statistics
                self._sample_columns_optimized(schema_name, table_obj)
                
                return True
            else:
                # No rows, add empty samples
                table_obj['sample_rows'] = []
                table_obj['rowcount_sample'] = 0
                table_obj['partial_results'] = True
                
                # Still try to sample columns
                self._sample_columns_optimized(schema_name, table_obj)
                
                return False
                
        except Exception as e:
            logger.error(f"Error in standard table sampling for {schema_name}.{table_obj['name']}: {str(e)}")
            table_obj['sampling_error'] = str(e)
            table_obj['partial_results'] = True
            return False
    
    def _update_table_in_discovery_data(self, discovery_data: Dict[str, Any], 
                                       schema_name: str, updated_table: Dict[str, Any]) -> None:
        """
        Update a table in the discovery data structure with thread safety.
        
        Args:
            discovery_data: Discovery data structure
            schema_name: Schema name
            updated_table: Updated table dictionary
        """
        with self.progress_lock:
            # Find and update the table in discovery_data
            for schema in discovery_data['schemas']:
                if schema['name'] == schema_name:
                    for i, table in enumerate(schema['tables']):
                        if table['name'] == updated_table['name']:
                            schema['tables'][i] = updated_table
                            return

    def _identify_negative_findings(self, discovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify what was NOT found (critical for Q&A validation).
        """
        findings = {}
        
        # Check for missing primary keys
        tables_without_pk = []
        for schema in discovery_data['schemas']:
            for table in schema['tables']:
                if not table.get('primary_key'):
                    tables_without_pk.append(f"{schema['name']}.{table['name']}")
        
        if tables_without_pk:
            findings['tables_without_primary_key'] = tables_without_pk
        
        # Check for tables with sampling errors
        tables_with_errors = []
        for schema in discovery_data['schemas']:
            for table in schema['tables']:
                if 'sampling_error' in table:
                    tables_with_errors.append({
                        'table': f"{schema['name']}.{table['name']}",
                        'error': table['sampling_error']
                    })
        
        if tables_with_errors:
            findings['tables_with_sampling_errors'] = tables_with_errors
        
        return findings
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        # Close engine if it exists
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                self.engine.dispose()
            except:
                pass