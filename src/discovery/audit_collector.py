"""
Remote Production Database Audit Collector

Collects access metrics from SQL Server DMVs to enhance discovery:
- Table access frequency and recency
- Query patterns (joins, filters, aggregations)
- Execution statistics for duplicate/history table detection

Non-invasive: Uses existing DMV data, no schema changes required.
Supports separate production database connection for audit collection.
"""

import json
import hashlib
import logging
import re
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TableAccessMetrics:
    """Access metrics for a single table"""
    schema_name: str
    table_name: str
    full_name: str

    # Access frequency
    total_reads: int = 0
    total_writes: int = 0
    total_seeks: int = 0
    total_scans: int = 0
    total_lookups: int = 0

    # Recency (ISO format strings)
    last_user_seek: Optional[str] = None
    last_user_scan: Optional[str] = None
    last_user_lookup: Optional[str] = None
    last_user_update: Optional[str] = None

    # Derived metrics
    access_score: float = 0.0  # 0-100 normalized score
    days_since_last_access: Optional[int] = None
    access_pattern: str = "unknown"  # 'hot', 'warm', 'cold', 'archive', 'unused'

    # Duplicate detection hints
    similar_tables: List[str] = field(default_factory=list)
    is_likely_history: bool = False
    is_likely_archive: bool = False
    history_base_table: Optional[str] = None  # The "main" table this is a history of


@dataclass
class QueryPattern:
    """Extracted query pattern for generalization"""
    pattern_hash: str
    pattern_template: str  # Normalized SQL with placeholders

    # Tables involved
    tables_referenced: List[str] = field(default_factory=list)
    joins_used: List[Dict[str, str]] = field(default_factory=list)

    # Aggregations and filters
    aggregations: List[str] = field(default_factory=list)
    filter_columns: List[str] = field(default_factory=list)
    group_by_columns: List[str] = field(default_factory=list)

    # Execution stats
    execution_count: int = 0
    total_elapsed_time_ms: int = 0
    avg_elapsed_time_ms: float = 0.0
    total_rows_returned: int = 0

    # Source tracking
    source_type: str = "adhoc"  # 'adhoc', 'stored_procedure', 'application'
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class AuditTimeRange:
    """Time range configuration for audit collection"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    lookback_days: int = 30

    def get_effective_range(self) -> Tuple[datetime, datetime]:
        """Get effective date range for audit"""
        end = self.end_date or datetime.utcnow()

        if self.start_date:
            start = self.start_date
        else:
            start = end - timedelta(days=self.lookback_days)

        return start, end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "lookback_days": self.lookback_days
        }


@dataclass
class AuditConfig:
    """Audit collector configuration"""
    connection_string: str
    enabled: bool = True
    lookback_days: int = 30
    min_execution_count: int = 5
    max_queries: int = 1000
    cache_hours: int = 24
    query_timeout: int = 30
    execution_window_start: Optional[dt_time] = None
    execution_window_end: Optional[dt_time] = None

    # Time range for specific period audit
    time_range: Optional[AuditTimeRange] = None

    @classmethod
    def from_env(cls) -> 'AuditConfig':
        """Load from environment variables"""
        import os

        def get_env(key: str, default: str = None) -> Optional[str]:
            return os.getenv(key, default)

        def get_env_bool(key: str, default: bool = False) -> bool:
            return os.getenv(key, str(default)).lower() in ('true', '1', 'yes')

        def get_env_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default

        def parse_time(value: Optional[str]) -> Optional[dt_time]:
            if not value:
                return None
            try:
                parts = value.split(':')
                return dt_time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
            except (ValueError, IndexError):
                return None

        # Get audit connection string, fall back to main database
        conn_str = get_env('AUDIT_DATABASE_CONNECTION_STRING')
        if not conn_str:
            conn_str = get_env('DATABASE_CONNECTION_STRING', '')

        return cls(
            connection_string=conn_str,
            enabled=get_env_bool('AUDIT_ENABLED', True),
            lookback_days=get_env_int('AUDIT_LOOKBACK_DAYS', 30),
            min_execution_count=get_env_int('AUDIT_MIN_EXECUTION_COUNT', 5),
            max_queries=get_env_int('AUDIT_MAX_QUERIES', 1000),
            cache_hours=get_env_int('AUDIT_CACHE_HOURS', 24),
            query_timeout=get_env_int('AUDIT_QUERY_TIMEOUT', 30),
            execution_window_start=parse_time(get_env('AUDIT_EXECUTION_WINDOW_START')),
            execution_window_end=parse_time(get_env('AUDIT_EXECUTION_WINDOW_END'))
        )


@dataclass
class AuditReport:
    """Complete audit report"""
    collected_at: str
    database_name: str
    source_server: str
    collection_duration_seconds: float = 0.0

    # Time range used
    audit_start_date: Optional[str] = None
    audit_end_date: Optional[str] = None

    # Metrics
    table_metrics: List[TableAccessMetrics] = field(default_factory=list)
    query_patterns: List[QueryPattern] = field(default_factory=list)

    # Summary
    total_tables_analyzed: int = 0
    hot_tables_count: int = 0
    warm_tables_count: int = 0
    cold_tables_count: int = 0
    unused_tables_count: int = 0
    likely_duplicates_count: int = 0
    likely_history_tables_count: int = 0

    # Relationship hints from query patterns
    join_frequency: Dict[str, int] = field(default_factory=dict)

    # Table recommendations
    tables_to_prioritize: List[str] = field(default_factory=list)
    tables_to_skip: List[str] = field(default_factory=list)
    history_table_mappings: Dict[str, str] = field(default_factory=dict)  # history -> base

    # Errors/warnings
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# AUDIT COLLECTOR
# ============================================================================

class AuditCollector:
    """
    Collects database access metrics from SQL Server DMVs

    Supports:
    - Remote production database connection
    - Specific time range auditing
    - Execution window enforcement
    - History/duplicate table detection

    Data Sources (all read-only):
    - sys.dm_db_index_usage_stats - Table access patterns
    - sys.dm_exec_query_stats - Query execution statistics
    - sys.dm_exec_sql_text - Query text for pattern extraction
    """

    # History table naming patterns
    HISTORY_PATTERNS = [
        (r'(.+)_hist$', 'suffix'),
        (r'(.+)_history$', 'suffix'),
        (r'(.+)_archive$', 'suffix'),
        (r'(.+)_backup$', 'suffix'),
        (r'(.+)_bak$', 'suffix'),
        (r'(.+)_old$', 'suffix'),
        (r'(.+)_copy$', 'suffix'),
        (r'(.+)_(\d{8})$', 'date_suffix'),  # _20241201
        (r'(.+)_(\d{6})$', 'date_suffix'),  # _202412
        (r'(.+)_v(\d+)$', 'version'),       # _v1, _v2
        (r'hist_(.+)$', 'prefix'),
        (r'history_(.+)$', 'prefix'),
        (r'archive_(.+)$', 'prefix'),
    ]

    def __init__(self, config: AuditConfig, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.audit_file = cache_dir / "audit_metrics.json"

    @contextmanager
    def _get_connection(self):
        """Get database connection with timeout and isolation"""
        engine = create_engine(
            self.config.connection_string,
            poolclass=NullPool,
            connect_args={
                "timeout": self.config.query_timeout,
                "login_timeout": 10
            }
        )

        try:
            with engine.connect() as conn:
                # Set query timeout
                conn.execute(text(f"SET LOCK_TIMEOUT {self.config.query_timeout * 1000}"))
                yield conn
        finally:
            engine.dispose()

    def _check_execution_window(self) -> Tuple[bool, str]:
        """Check if current time is within allowed execution window"""
        if not self.config.execution_window_start or not self.config.execution_window_end:
            return True, "No execution window configured"

        now = datetime.now().time()
        start = self.config.execution_window_start
        end = self.config.execution_window_end

        # Handle overnight window (e.g., 22:00 - 06:00)
        if start > end:
            in_window = now >= start or now <= end
        else:
            in_window = start <= now <= end

        if in_window:
            return True, f"Within execution window ({start} - {end})"
        else:
            return False, f"Outside execution window ({start} - {end}). Current: {now}"

    def _check_permissions(self) -> Tuple[bool, List[str]]:
        """Verify required permissions on database"""
        missing = []

        try:
            with self._get_connection() as conn:
                # Check VIEW SERVER STATE
                result = conn.execute(text("""
                    SELECT HAS_PERMS_BY_NAME(null, null, 'VIEW SERVER STATE') AS has_perm
                """)).scalar()

                if not result:
                    missing.append("VIEW SERVER STATE")

                # Check VIEW DATABASE STATE
                result = conn.execute(text("""
                    SELECT HAS_PERMS_BY_NAME(DB_NAME(), 'DATABASE', 'VIEW DATABASE STATE') AS has_perm
                """)).scalar()

                if not result:
                    missing.append("VIEW DATABASE STATE")

        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False, ["CONNECTION_FAILED"]

        return len(missing) == 0, missing

    def _get_server_info(self) -> Dict[str, str]:
        """Get server information"""
        with self._get_connection() as conn:
            result = conn.execute(text("""
                SELECT
                    @@SERVERNAME AS server_name,
                    DB_NAME() AS database_name,
                    @@VERSION AS version
            """)).fetchone()

            return {
                "server_name": result.server_name,
                "database_name": result.database_name,
                "version": result.version.split('\n')[0] if result.version else "Unknown"
            }

    def collect(
        self,
        force: bool = False,
        time_range: Optional[AuditTimeRange] = None
    ) -> AuditReport:
        """
        Collect audit metrics from production database

        Args:
            force: Bypass execution window check
            time_range: Specific time range to audit (overrides config)

        Returns:
            AuditReport with collected metrics
        """
        start_time = datetime.utcnow()

        logger.info("=" * 80)
        logger.info("PRODUCTION DATABASE AUDIT COLLECTION")
        logger.info("=" * 80)

        # Check if enabled
        if not self.config.enabled:
            raise RuntimeError("Audit collection disabled. Set AUDIT_ENABLED=true")

        # Check execution window
        if not force:
            in_window, msg = self._check_execution_window()
            if not in_window:
                raise RuntimeError(f"Audit blocked: {msg}. Use --force to override.")
            logger.info(f"Execution window: {msg}")
        else:
            logger.warning("Execution window check bypassed (--force)")

        # Check permissions
        logger.info("Checking permissions...")
        has_perms, missing = self._check_permissions()
        if not has_perms:
            raise PermissionError(
                f"Missing permissions: {', '.join(missing)}. "
                f"Run: GRANT VIEW SERVER STATE, VIEW DATABASE STATE TO [audit_user];"
            )
        logger.info("  ✓ Permissions verified")

        # Get server info
        server_info = self._get_server_info()
        logger.info(f"  Server: {server_info['server_name']}")
        logger.info(f"  Database: {server_info['database_name']}")

        # Determine time range
        effective_time_range = time_range or self.config.time_range or AuditTimeRange(
            lookback_days=self.config.lookback_days
        )
        audit_start, audit_end = effective_time_range.get_effective_range()
        logger.info(f"  Audit period: {audit_start.date()} to {audit_end.date()}")

        # Initialize report
        report = AuditReport(
            collected_at=start_time.isoformat(),
            database_name=server_info['database_name'],
            source_server=server_info['server_name'],
            audit_start_date=audit_start.isoformat(),
            audit_end_date=audit_end.isoformat()
        )

        try:
            # Step 1: Collect table access metrics
            logger.info("Step 1: Collecting table access metrics...")
            report.table_metrics = self._collect_table_metrics()
            logger.info(f"  ✓ {len(report.table_metrics)} tables analyzed")

            # Step 2: Classify access patterns
            logger.info("Step 2: Classifying access patterns...")
            self._classify_access_patterns(report.table_metrics)

            # Step 3: Detect history/duplicate tables
            logger.info("Step 3: Detecting history/duplicate tables...")
            self._detect_history_tables(report.table_metrics)

            # Step 4: Collect query patterns
            logger.info("Step 4: Extracting query patterns...")
            report.query_patterns = self._collect_query_patterns(audit_start, audit_end)
            logger.info(f"  ✓ {len(report.query_patterns)} patterns extracted")

            # Step 5: Build join frequency map
            logger.info("Step 5: Building join frequency map...")
            report.join_frequency = self._build_join_frequency(report.query_patterns)
            logger.info(f"  ✓ {len(report.join_frequency)} join patterns found")

            # Step 6: Generate recommendations
            logger.info("Step 6: Generating recommendations...")
            self._generate_recommendations(report)

            # Compute summary
            report.total_tables_analyzed = len(report.table_metrics)
            report.hot_tables_count = sum(1 for t in report.table_metrics if t.access_pattern == 'hot')
            report.warm_tables_count = sum(1 for t in report.table_metrics if t.access_pattern == 'warm')
            report.cold_tables_count = sum(1 for t in report.table_metrics if t.access_pattern == 'cold')
            report.unused_tables_count = sum(1 for t in report.table_metrics if t.access_pattern == 'unused')
            report.likely_duplicates_count = sum(1 for t in report.table_metrics if t.similar_tables)
            report.likely_history_tables_count = sum(1 for t in report.table_metrics if t.is_likely_history)

        except Exception as e:
            logger.error(f"Audit collection error: {e}")
            report.warnings.append(f"Collection error: {str(e)}")
            raise

        finally:
            report.collection_duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        # Save report
        self._save_report(report)

        logger.info("=" * 80)
        logger.info("AUDIT COLLECTION COMPLETE")
        logger.info(f"  Duration:        {report.collection_duration_seconds:.1f}s")
        logger.info(f"  Hot tables:      {report.hot_tables_count}")
        logger.info(f"  Warm tables:     {report.warm_tables_count}")
        logger.info(f"  Cold tables:     {report.cold_tables_count}")
        logger.info(f"  Unused tables:   {report.unused_tables_count}")
        logger.info(f"  History tables:  {report.likely_history_tables_count}")
        logger.info(f"  To prioritize:   {len(report.tables_to_prioritize)}")
        logger.info(f"  To skip:         {len(report.tables_to_skip)}")
        logger.info("=" * 80)

        return report

    def _collect_table_metrics(self) -> List[TableAccessMetrics]:
        """Collect table access metrics from DMVs"""
        query = text("""
            SELECT
                s.name AS schema_name,
                t.name AS table_name,
                COALESCE(SUM(ius.user_seeks), 0) AS total_seeks,
                COALESCE(SUM(ius.user_scans), 0) AS total_scans,
                COALESCE(SUM(ius.user_lookups), 0) AS total_lookups,
                COALESCE(SUM(ius.user_updates), 0) AS total_updates,
                MAX(ius.last_user_seek) AS last_user_seek,
                MAX(ius.last_user_scan) AS last_user_scan,
                MAX(ius.last_user_lookup) AS last_user_lookup,
                MAX(ius.last_user_update) AS last_user_update
            FROM sys.tables t
            INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
            LEFT JOIN sys.dm_db_index_usage_stats ius
                ON t.object_id = ius.object_id
                AND ius.database_id = DB_ID()
            WHERE t.is_ms_shipped = 0
              AND s.name NOT IN ('sys', 'INFORMATION_SCHEMA', 'guest')
            GROUP BY s.name, t.name
            ORDER BY
                COALESCE(SUM(ius.user_seeks), 0) +
                COALESCE(SUM(ius.user_scans), 0) DESC
        """)

        metrics = []

        with self._get_connection() as conn:
            result = conn.execute(query)

            for row in result:
                total_reads = (row.total_seeks or 0) + (row.total_scans or 0) + (row.total_lookups or 0)

                # Calculate days since last access
                last_dates = [row.last_user_seek, row.last_user_scan, row.last_user_lookup, row.last_user_update]
                last_access = max((d for d in last_dates if d), default=None)
                days_since = (datetime.utcnow() - last_access).days if last_access else None

                metric = TableAccessMetrics(
                    schema_name=row.schema_name,
                    table_name=row.table_name,
                    full_name=f"{row.schema_name}.{row.table_name}",
                    total_reads=total_reads,
                    total_writes=row.total_updates or 0,
                    total_seeks=row.total_seeks or 0,
                    total_scans=row.total_scans or 0,
                    total_lookups=row.total_lookups or 0,
                    last_user_seek=row.last_user_seek.isoformat() if row.last_user_seek else None,
                    last_user_scan=row.last_user_scan.isoformat() if row.last_user_scan else None,
                    last_user_lookup=row.last_user_lookup.isoformat() if row.last_user_lookup else None,
                    last_user_update=row.last_user_update.isoformat() if row.last_user_update else None,
                    days_since_last_access=days_since
                )
                metrics.append(metric)

        return metrics

    def _classify_access_patterns(self, metrics: List[TableAccessMetrics]) -> None:
        """
        Classify tables by access frequency and recency.

        Uses BOTH percentile ranking AND absolute score thresholds to ensure
        meaningful differentiation between hot/warm/cold tables.

        Classification rules:
        - hot: Top 15% by reads AND access_score >= 10.0
        - warm: Top 15-50% by reads AND access_score >= 1.0
        - cold: access_score >= 0.1 but not hot/warm
        - archive: cold + read-only + not accessed in 30+ days
        - unused: access_score < 0.1 OR no reads in 90+ days
        """
        if not metrics:
            return

        sorted_metrics = sorted(metrics, key=lambda m: m.total_reads, reverse=True)
        total = len(sorted_metrics)
        max_reads = max(m.total_reads for m in metrics) or 1

        for idx, metric in enumerate(sorted_metrics):
            percentile = (idx / total) * 100
            metric.access_score = (metric.total_reads / max_reads) * 100

            # Unused tables
            if metric.total_reads == 0:
                metric.access_pattern = "unused"
                continue

            # Check for truly unused (not accessed in 90+ days)
            if metric.days_since_last_access and metric.days_since_last_access > 90:
                metric.access_pattern = "unused"
                continue

            # Hot: Must be in top 15% AND have meaningful score (>= 10%)
            if percentile <= 15 and metric.access_score >= 10.0:
                metric.access_pattern = "hot"

            # Warm: Top 15-50% with decent score (>= 1%)
            elif percentile <= 50 and metric.access_score >= 1.0:
                metric.access_pattern = "warm"

            # Cold but not archive: has some activity (>= 0.1%)
            elif metric.access_score >= 0.1:
                # Check if it's archive (read-only, stale)
                if metric.total_writes == 0 and metric.days_since_last_access and metric.days_since_last_access > 30:
                    metric.access_pattern = "archive"
                else:
                    metric.access_pattern = "cold"

            # Very low activity
            else:
                if metric.days_since_last_access and metric.days_since_last_access > 60:
                    metric.access_pattern = "unused"
                else:
                    metric.access_pattern = "archive"

    def _detect_history_tables(self, metrics: List[TableAccessMetrics]) -> None:
        """Detect history/archive/duplicate tables"""
        # Build table lookup
        table_lookup = {m.table_name.lower(): m for m in metrics}
        full_name_lookup = {m.full_name.lower(): m for m in metrics}

        for metric in metrics:
            table_name_lower = metric.table_name.lower()

            # Check against history patterns
            for pattern, pattern_type in self.HISTORY_PATTERNS:
                match = re.match(pattern, table_name_lower, re.IGNORECASE)
                if match:
                    metric.is_likely_history = True

                    # Extract base table name
                    if pattern_type in ('suffix', 'date_suffix', 'version'):
                        base_name = match.group(1)
                    else:  # prefix
                        base_name = match.group(1)

                    # Look for base table
                    if base_name in table_lookup:
                        base_metric = table_lookup[base_name]
                        metric.history_base_table = base_metric.full_name
                        metric.similar_tables.append(base_metric.full_name)

                    # Also check with same schema
                    full_base = f"{metric.schema_name}.{base_name}".lower()
                    if full_base in full_name_lookup and full_base != metric.full_name.lower():
                        base_metric = full_name_lookup[full_base]
                        if base_metric.full_name not in metric.similar_tables:
                            metric.history_base_table = base_metric.full_name
                            metric.similar_tables.append(base_metric.full_name)

                    break

            # Additional archive detection
            if (metric.total_writes == 0 and
                metric.days_since_last_access and
                metric.days_since_last_access > 60):
                metric.is_likely_archive = True

    def _collect_query_patterns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[QueryPattern]:
        """Extract query patterns from plan cache"""
        query = text("""
            SELECT TOP (:max_queries)
                qs.query_hash,
                qs.execution_count,
                qs.total_elapsed_time / 1000 AS total_elapsed_time_ms,
                qs.total_rows AS total_rows_returned,
                qs.creation_time,
                qs.last_execution_time,
                SUBSTRING(
                    st.text,
                    (qs.statement_start_offset/2) + 1,
                    ((CASE qs.statement_end_offset
                        WHEN -1 THEN DATALENGTH(st.text)
                        ELSE qs.statement_end_offset
                    END - qs.statement_start_offset)/2) + 1
                ) AS query_text
            FROM sys.dm_exec_query_stats qs
            CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
            WHERE
                qs.execution_count >= :min_count
                AND qs.last_execution_time >= :start_date
                AND qs.last_execution_time <= :end_date
                AND st.text NOT LIKE '%sys.%'
                AND st.text NOT LIKE '%INFORMATION_SCHEMA%'
                AND st.text NOT LIKE '%dm_exec%'
                AND st.text NOT LIKE '%sp_executesql%'
            ORDER BY qs.execution_count DESC
        """)

        patterns = []

        with self._get_connection() as conn:
            result = conn.execute(query, {
                "max_queries": self.config.max_queries,
                "min_count": self.config.min_execution_count,
                "start_date": start_date,
                "end_date": end_date
            })

            for row in result:
                if not row.query_text:
                    continue

                pattern = self._extract_pattern(row.query_text)
                if not pattern:
                    continue

                pattern.execution_count = row.execution_count
                pattern.total_elapsed_time_ms = row.total_elapsed_time_ms or 0
                pattern.avg_elapsed_time_ms = (
                    pattern.total_elapsed_time_ms / pattern.execution_count
                    if pattern.execution_count > 0 else 0
                )
                pattern.total_rows_returned = row.total_rows_returned or 0
                pattern.first_seen = row.creation_time.isoformat() if row.creation_time else None
                pattern.last_seen = row.last_execution_time.isoformat() if row.last_execution_time else None

                patterns.append(pattern)

        return patterns

    def _extract_pattern(self, query_text: str) -> Optional[QueryPattern]:
        """Extract and normalize query pattern"""
        try:
            import sqlglot
            from sqlglot import exp

            parsed = sqlglot.parse_one(query_text, dialect='tsql')

            if not isinstance(parsed, exp.Select):
                return None

            # Extract tables
            tables = []
            for table in parsed.find_all(exp.Table):
                name = f"{table.db}.{table.name}" if table.db else table.name
                tables.append(name)

            # Extract joins
            joins = []
            for join in parsed.find_all(exp.Join):
                on_clause = join.args.get("on")
                if on_clause:
                    for eq in on_clause.find_all(exp.EQ):
                        left = eq.args.get("this")
                        right = eq.args.get("expression")
                        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                            joins.append({"left": left.sql(), "right": right.sql()})

            # Extract aggregations
            aggregations = list(set(
                func.__class__.__name__.upper()
                for func in parsed.find_all(exp.AggFunc)
            ))

            # Extract filter columns
            filter_columns = []
            where = parsed.find(exp.Where)
            if where:
                filter_columns = list(set(col.sql() for col in where.find_all(exp.Column)))

            # Extract GROUP BY
            group_by = []
            group = parsed.find(exp.Group)
            if group:
                group_by = [col.sql() for col in group.find_all(exp.Column)]

            # Normalize query for hashing
            normalized = self._normalize_query(parsed)
            pattern_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]

            return QueryPattern(
                pattern_hash=pattern_hash,
                pattern_template=normalized[:1000],
                tables_referenced=tables,
                joins_used=joins,
                aggregations=aggregations,
                filter_columns=filter_columns,
                group_by_columns=group_by
            )

        except Exception as e:
            logger.debug(f"Pattern extraction failed: {e}")
            return None

    def _normalize_query(self, parsed) -> str:
        """Replace literals with placeholders for pattern matching"""
        from sqlglot import exp

        normalized = parsed.copy()
        for literal in normalized.find_all(exp.Literal):
            literal.replace(exp.Placeholder())

        return normalized.sql(dialect='tsql', pretty=False)

    def _build_join_frequency(self, patterns: List[QueryPattern]) -> Dict[str, int]:
        """Build frequency map of join patterns"""
        join_freq = defaultdict(int)

        for pattern in patterns:
            for join in pattern.joins_used:
                key = "=".join(sorted([join["left"], join["right"]]))
                join_freq[key] += pattern.execution_count

        return dict(sorted(join_freq.items(), key=lambda x: x[1], reverse=True))

    def _generate_recommendations(self, report: AuditReport) -> None:
        """Generate table prioritization recommendations"""
        # Tables to prioritize (hot + warm, not history)
        report.tables_to_prioritize = [
            m.full_name for m in report.table_metrics
            if m.access_pattern in ('hot', 'warm') and not m.is_likely_history
        ]

        # Tables to skip (unused + history)
        report.tables_to_skip = [
            m.full_name for m in report.table_metrics
            if m.access_pattern == 'unused' or m.is_likely_history
        ]

        # History table mappings
        report.history_table_mappings = {
            m.full_name: m.history_base_table
            for m in report.table_metrics
            if m.is_likely_history and m.history_base_table
        }

    def _save_report(self, report: AuditReport) -> None:
        """Save report to JSON file"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "collected_at": report.collected_at,
            "database_name": report.database_name,
            "source_server": report.source_server,
            "collection_duration_seconds": report.collection_duration_seconds,
            "audit_period": {
                "start": report.audit_start_date,
                "end": report.audit_end_date
            },
            "table_metrics": [asdict(m) for m in report.table_metrics],
            "query_patterns": [asdict(p) for p in report.query_patterns],
            "summary": {
                "total_tables_analyzed": report.total_tables_analyzed,
                "hot_tables_count": report.hot_tables_count,
                "warm_tables_count": report.warm_tables_count,
                "cold_tables_count": report.cold_tables_count,
                "unused_tables_count": report.unused_tables_count,
                "likely_duplicates_count": report.likely_duplicates_count,
                "likely_history_tables_count": report.likely_history_tables_count
            },
            "join_frequency": report.join_frequency,
            "recommendations": {
                "tables_to_prioritize": report.tables_to_prioritize,
                "tables_to_skip": report.tables_to_skip,
                "history_table_mappings": report.history_table_mappings
            },
            "warnings": report.warnings
        }

        with open(self.audit_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved audit report to {self.audit_file}")

    def load_cached_report(self) -> Optional[AuditReport]:
        """Load cached report if valid"""
        if not self.audit_file.exists():
            return None

        try:
            with open(self.audit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check cache age
            collected_at = datetime.fromisoformat(data["collected_at"])
            age_hours = (datetime.utcnow() - collected_at).total_seconds() / 3600

            if age_hours > self.config.cache_hours:
                logger.info(f"Cached audit expired ({age_hours:.1f}h > {self.config.cache_hours}h)")
                return None

            # Reconstruct report
            report = AuditReport(
                collected_at=data["collected_at"],
                database_name=data["database_name"],
                source_server=data.get("source_server", "unknown")
            )

            report.audit_start_date = data.get("audit_period", {}).get("start")
            report.audit_end_date = data.get("audit_period", {}).get("end")

            report.table_metrics = [
                TableAccessMetrics(**m) for m in data.get("table_metrics", [])
            ]
            report.query_patterns = [
                QueryPattern(**p) for p in data.get("query_patterns", [])
            ]
            report.join_frequency = data.get("join_frequency", {})

            summary = data.get("summary", {})
            report.total_tables_analyzed = summary.get("total_tables_analyzed", 0)
            report.hot_tables_count = summary.get("hot_tables_count", 0)
            report.warm_tables_count = summary.get("warm_tables_count", 0)
            report.cold_tables_count = summary.get("cold_tables_count", 0)
            report.unused_tables_count = summary.get("unused_tables_count", 0)
            report.likely_duplicates_count = summary.get("likely_duplicates_count", 0)
            report.likely_history_tables_count = summary.get("likely_history_tables_count", 0)

            recs = data.get("recommendations", {})
            report.tables_to_prioritize = recs.get("tables_to_prioritize", [])
            report.tables_to_skip = recs.get("tables_to_skip", [])
            report.history_table_mappings = recs.get("history_table_mappings", {})

            logger.info(f"Loaded cached audit ({age_hours:.1f}h old)")
            return report

        except Exception as e:
            logger.warning(f"Failed to load cached report: {e}")
            return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def collect_audit(
    force: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: Optional[int] = None
) -> AuditReport:
    """
    Convenience function to collect audit with optional time range

    Args:
        force: Bypass execution window
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        lookback_days: Days to look back (ignored if start_date set)

    Returns:
        AuditReport
    """
    from config.settings import get_path_config

    config = AuditConfig.from_env()
    path_config = get_path_config()

    # Build time range
    time_range = None
    if start_date or end_date or lookback_days:
        time_range = AuditTimeRange(
            start_date=datetime.fromisoformat(start_date) if start_date else None,
            end_date=datetime.fromisoformat(end_date) if end_date else None,
            lookback_days=lookback_days or config.lookback_days
        )

    collector = AuditCollector(config, path_config.cache_dir)
    return collector.collect(force=force, time_range=time_range)
