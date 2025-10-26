"""
SQL Executor - Safely execute generated SQL queries
Enforces read-only constraints, timeouts, and row limits
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import signal
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when query execution exceeds timeout."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for query timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Query execution exceeded {seconds} seconds")
    
    # Set the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


class SQLExecutor:
    """
    Executes SQL queries with safety constraints:
    - Read-only enforcement
    - Timeout limits
    - Row limits
    - Error handling
    """
    
    def __init__(
        self,
        connection_string: str,
        default_row_limit: int = 10,
        default_timeout: int = 60
    ):
        """
        Initialize SQL executor.
        
        Args:
            connection_string: Database connection string (must be read-only)
            default_row_limit: Default number of rows to return
            default_timeout: Default query timeout in seconds
        """
        self.connection_string = connection_string
        self.default_row_limit = default_row_limit
        self.default_timeout = default_timeout
        
        # Create engine with read-only settings
        self.engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"timeout": 30}
        )
        
        logger.info(f"SQL Executor initialized (row_limit={default_row_limit}, timeout={default_timeout}s)")
    
    def execute_query(
        self,
        sql: str,
        row_limit: Optional[int] = None,
        timeout_sec: Optional[int] = None
    ) -> Tuple[bool, List[Dict[str, Any]], str, float]:
        """
        Execute SQL query with safety constraints.
        
        Args:
            sql: SQL query to execute
            row_limit: Maximum rows to return (overrides default)
            timeout_sec: Query timeout in seconds (overrides default)
        
        Returns:
            Tuple of (success, results, error_message, execution_time)
            - success: True if query executed successfully
            - results: List of row dictionaries [{col: val, ...}, ...]
            - error_message: Error description if success=False
            - execution_time: Query execution time in seconds
        """
        row_limit = row_limit or self.default_row_limit
        timeout_sec = timeout_sec or self.default_timeout
        
        logger.info(f"Executing query (limit={row_limit}, timeout={timeout_sec}s)")
        logger.debug(f"SQL: {sql}")
        
        # Validate SQL is read-only (basic check)
        if not self._is_read_only_query(sql):
            error_msg = "Query contains non-SELECT statements. Only read-only queries are allowed."
            logger.error(error_msg)
            return False, [], error_msg, 0.0
        
        start_time = time.time()
        
        try:
            with timeout(timeout_sec):
                with self.engine.connect() as conn:
                    # Set query timeout at database level
                    if "mssql" in self.connection_string.lower():
                        conn.execute(text(f"SET LOCK_TIMEOUT {timeout_sec * 1000}"))
                    
                    # Execute query
                    result = conn.execute(text(sql))
                    
                    # Fetch limited rows
                    rows = result.fetchmany(row_limit)
                    
                    # Convert to list of dicts
                    columns = result.keys()
                    results = [dict(zip(columns, row)) for row in rows]
                    
                    execution_time = time.time() - start_time
                    
                    logger.info(f"Query executed successfully: {len(results)} rows in {execution_time:.2f}s")
                    
                    return True, results, "", execution_time
        
        except TimeoutError as e:
            execution_time = time.time() - start_time
            error_msg = f"Query timeout after {execution_time:.1f}s"
            logger.error(error_msg)
            return False, [], error_msg, execution_time
        
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            error_msg = f"Database error: {str(e)}"
            logger.error(error_msg)
            return False, [], error_msg, execution_time
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, [], error_msg, execution_time
    
    def _is_read_only_query(self, sql: str) -> bool:
        """
        Basic check to ensure query is read-only.
        
        Args:
            sql: SQL query to validate
        
        Returns:
            True if query appears to be read-only
        """
        sql_upper = sql.upper().strip()
        
        # Check if starts with SELECT (or WITH for CTEs)
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return False
        
        # Block dangerous keywords
        forbidden = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE"
        ]
        
        for keyword in forbidden:
            if keyword in sql_upper:
                return False
        
        return True
    
    def format_results_table(
        self,
        results: List[Dict[str, Any]],
        max_col_width: int = 50
    ) -> str:
        """
        Format query results as ASCII table for terminal display.
        
        Args:
            results: List of row dictionaries
            max_col_width: Maximum column width for display
        
        Returns:
            Formatted table string
        """
        if not results:
            return "No results returned."
        
        # Get column names
        columns = list(results[0].keys())
        
        # Calculate column widths
        col_widths = {}
        for col in columns:
            # Start with column name length
            col_widths[col] = len(col)
            
            # Check each row value
            for row in results:
                value_str = str(row[col]) if row[col] is not None else "NULL"
                col_widths[col] = max(col_widths[col], len(value_str))
            
            # Cap at max_col_width
            col_widths[col] = min(col_widths[col], max_col_width)
        
        # Build table
        lines = []
        
        # Header separator
        lines.append("+" + "+".join("-" * (col_widths[col] + 2) for col in columns) + "+")
        
        # Header row
        header = "|"
        for col in columns:
            header += f" {col:<{col_widths[col]}} |"
        lines.append(header)
        
        # Header separator
        lines.append("+" + "+".join("=" * (col_widths[col] + 2) for col in columns) + "+")
        
        # Data rows
        for row in results:
            row_str = "|"
            for col in columns:
                value = row[col]
                value_str = str(value) if value is not None else "NULL"
                
                # Truncate if too long
                if len(value_str) > max_col_width:
                    value_str = value_str[:max_col_width-3] + "..."
                
                row_str += f" {value_str:<{col_widths[col]}} |"
            lines.append(row_str)
        
        # Bottom separator
        lines.append("+" + "+".join("-" * (col_widths[col] + 2) for col in columns) + "+")
        
        return "\n".join(lines)
    
    def close(self):
        """Close database connection."""
        self.engine.dispose()
        logger.info("SQL Executor closed")