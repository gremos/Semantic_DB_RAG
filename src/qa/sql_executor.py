"""
SQL Executor for Q&A Phase
Executes generated SQL safely with timeout and result formatting
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings, get_database_config

logger = logging.getLogger(__name__)


class SQLExecutor:
    """Execute SQL queries safely with timeout protection"""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize executor with database connection

        Args:
            connection_string: Database connection string (uses config if not provided)
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            db_config = get_database_config()
            self.connection_string = db_config.connection_string

        self.engine = create_engine(
            self.connection_string,
            pool_pre_ping=True,
            pool_recycle=300
        )

    def execute(
        self,
        sql: str,
        timeout_sec: int = 60,
        row_limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute SQL query with timeout protection

        Args:
            sql: SQL statement to execute
            timeout_sec: Maximum execution time in seconds
            row_limit: Maximum rows to return

        Returns:
            {
                'status': 'ok'|'error'|'timeout',
                'data': [...] list of row dicts,
                'columns': [...] column names,
                'row_count': int,
                'execution_time_ms': float,
                'error': str if status='error'
            }
        """
        logger.info(f"Executing SQL (timeout={timeout_sec}s, limit={row_limit})")
        logger.debug(f"SQL: {sql[:200]}...")

        start_time = datetime.now()

        # Execute with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_query, sql, row_limit)

            try:
                result = future.result(timeout=timeout_sec)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                result['execution_time_ms'] = execution_time
                logger.info(f"Query completed in {execution_time:.0f}ms, {result.get('row_count', 0)} rows")
                return result

            except FuturesTimeoutError:
                logger.warning(f"Query timed out after {timeout_sec}s")
                return {
                    'status': 'timeout',
                    'data': [],
                    'columns': [],
                    'row_count': 0,
                    'execution_time_ms': timeout_sec * 1000,
                    'error': f"Query timed out after {timeout_sec} seconds"
                }

    def _execute_query(self, sql: str, row_limit: int) -> Dict[str, Any]:
        """
        Internal query execution

        Args:
            sql: SQL to execute
            row_limit: Max rows

        Returns:
            Result dict
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))

                # Get column names
                columns = list(result.keys())

                # Fetch rows up to limit
                rows = []
                for i, row in enumerate(result):
                    if i >= row_limit:
                        break
                    # Convert row to dict
                    row_dict = {}
                    for col, val in zip(columns, row):
                        # Handle special types
                        if hasattr(val, 'isoformat'):
                            row_dict[col] = val.isoformat()
                        elif isinstance(val, bytes):
                            row_dict[col] = val.hex()
                        else:
                            row_dict[col] = val
                    rows.append(row_dict)

                return {
                    'status': 'ok',
                    'data': rows,
                    'columns': columns,
                    'row_count': len(rows),
                    'truncated': len(rows) >= row_limit
                }

        except SQLAlchemyError as e:
            logger.error(f"SQL execution error: {e}")
            return {
                'status': 'error',
                'data': [],
                'columns': [],
                'row_count': 0,
                'error': str(e)
            }

    def format_results(
        self,
        result: Dict[str, Any],
        format_type: str = 'table'
    ) -> str:
        """
        Format query results for display

        Args:
            result: Execution result dict
            format_type: 'table', 'json', or 'csv'

        Returns:
            Formatted string
        """
        if result['status'] != 'ok':
            return f"Error: {result.get('error', 'Unknown error')}"

        if not result['data']:
            return "No results found."

        columns = result['columns']
        rows = result['data']

        if format_type == 'json':
            import json
            return json.dumps(rows, indent=2, default=str)

        elif format_type == 'csv':
            lines = [','.join(columns)]
            for row in rows:
                values = [str(row.get(c, '')) for c in columns]
                lines.append(','.join(values))
            return '\n'.join(lines)

        else:  # table format
            return self._format_table(columns, rows)

    def _format_table(self, columns: List[str], rows: List[Dict]) -> str:
        """Format as ASCII table"""
        if not rows:
            return "No results"

        # Calculate column widths
        widths = {}
        for col in columns:
            widths[col] = max(
                len(col),
                max(len(str(row.get(col, ''))[:50]) for row in rows)
            )

        # Build table
        lines = []

        # Header
        header = ' | '.join(col.ljust(widths[col])[:50] for col in columns)
        lines.append(header)
        lines.append('-' * len(header))

        # Rows
        for row in rows[:100]:  # Limit display to 100 rows
            row_str = ' | '.join(
                str(row.get(col, '')).ljust(widths[col])[:50]
                for col in columns
            )
            lines.append(row_str)

        if len(rows) > 100:
            lines.append(f"... and {len(rows) - 100} more rows")

        return '\n'.join(lines)

    def validate_read_only(self, sql: str) -> bool:
        """
        Validate SQL is read-only (SELECT only)

        Args:
            sql: SQL statement

        Returns:
            True if read-only, False otherwise
        """
        sql_upper = sql.upper().strip()

        # Must start with SELECT or WITH (for CTEs)
        if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
            return False

        # Check for forbidden keywords
        forbidden = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
            'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE', 'MERGE'
        ]

        for keyword in forbidden:
            # Check for keyword as whole word
            if f' {keyword} ' in f' {sql_upper} ':
                return False

        return True

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()

    def __del__(self):
        """Cleanup on destruction"""
        self.close()
