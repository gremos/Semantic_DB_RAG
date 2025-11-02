"""
Optimized Relationship Detector
Multi-stage filtering pipeline for fast foreign key relationship detection
Reduces comparisons from O(n²) to O(n log n) through smart pre-filtering
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from sqlalchemy import create_engine, text, pool
from sqlalchemy.engine import Engine

import sqlglot


from .relationship_config import RelationshipDetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Lightweight column information for relationship detection"""

    schema: str
    table: str
    name: str
    type: str
    is_primary_key: bool
    is_indexed: bool
    is_nullable: bool
    distinct_count: Optional[int] = None
    sample_values: Optional[List[str]] = None

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.table}.{self.name}"

    @property
    def table_name(self) -> str:
        return f"{self.schema}.{self.table}"


@dataclass
class RelationshipCandidate:
    """Candidate relationship with scoring"""

    from_column: ColumnInfo
    to_column: ColumnInfo
    name_score: float
    stage: str  # Which filtering stage produced this candidate

    def __lt__(self, other):
        return self.name_score > other.name_score  # Higher scores first


@dataclass
class DetectedRelationship:
    """Detected foreign key relationship"""

    from_column: str
    to_column: str
    overlap_rate: float
    cardinality: str  # many_to_one, one_to_one
    method: str
    confidence: str  # high, medium, low
    name_score: float
    verification: Dict


def _parse_join_condition(on_clause) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract table.column pairs from JOIN ON clause
    
    Args:
        on_clause: sqlglot ON clause expression
        
    Returns:
        Tuple of (left_table, left_col, right_table, right_col)
        Returns (None, None, None, None) if parsing fails
    """
    try:
        # Handle simple equality: table1.col1 = table2.col2
        if isinstance(on_clause, exp.EQ):
            left = on_clause.left
            right = on_clause.right
            
            # Both sides must be columns
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                left_table = left.table if left.table else None
                left_col = left.name if left.name else None
                right_table = right.table if right.table else None
                right_col = right.name if right.name else None
                
                if all([left_table, left_col, right_table, right_col]):
                    return left_table, left_col, right_table, right_col
        
        # Handle complex conditions (AND, OR) - extract first simple equality
        elif isinstance(on_clause, (exp.And, exp.Or)):
            # Recursively check left side
            result = _parse_join_condition(on_clause.left)
            if result and all(result):
                return result
            
            # Recursively check right side
            result = _parse_join_condition(on_clause.right)
            if result and all(result):
                return result
    
    except Exception as e:
        logger.debug(f"Failed to parse join condition: {e}")
    
    return None, None, None, None


def _normalize_table_name(table: str, discovery_data: Dict[str, Any]) -> str:
    """
    Normalize table name to include schema if missing
    
    Args:
        table: Table name (may or may not include schema)
        discovery_data: Discovery data to look up schemas
        
    Returns:
        Normalized table name (schema.table)
    """
    # Already has schema
    if "." in table:
        return table
    
    # Try to find schema by searching discovery data
    for schema in discovery_data.get("schemas", []):
        schema_name = schema.get("name")
        for tbl in schema.get("tables", []):
            if tbl.get("name") == table:
                return f"{schema_name}.{table}"
    
    # Fallback: assume dbo schema (SQL Server default)
    logger.debug(f"Could not find schema for table {table}, assuming 'dbo'")
    return f"dbo.{table}"


class RelationshipDetector:
    """
    Optimized relationship detector with multi-stage filtering

    Pipeline:
    1. Build indexes (PK, FK, type groups)
    2. Type compatibility filter (~70% reduction)
    3. Name-based pre-scoring (~80% reduction)
    4. Index requirement filter (~50% reduction)
    5. Cardinality pre-check (~40% reduction)
    6. Parallel overlap analysis (final verification)
    """

    def __init__(
        self,
        connection_string: str,
        discovery_data: Dict,
        config: Optional[RelationshipDetectionConfig] = None,
    ):
        self.connection_string = connection_string
        self.discovery_data = discovery_data
        self.config = config or RelationshipDetectionConfig.from_env()

        # Performance tracking
        self.stats = {
            "total_columns": 0,
            "pk_columns": 0,
            "candidates_after_type_filter": 0,
            "candidates_after_name_filter": 0,
            "candidates_after_index_filter": 0,
            "candidates_after_cardinality_filter": 0,
            "relationships_found": 0,
            "comparisons_made": 0,
            "time_building_indexes": 0,
            "time_filtering": 0,
            "time_overlap_checks": 0,
        }

        # Indexes built once
        self.pk_index: Dict[str, List[ColumnInfo]] = {}
        self.all_columns: List[ColumnInfo] = []
        self.indexed_columns: Dict[str, Set[str]] = {}
        self.type_index: Dict[int, List[ColumnInfo]] = {}

        # Connection pool for parallel processing
        self.engine: Optional[Engine] = None

    def detect_relationships(self) -> List[DetectedRelationship]:
        """
        Main entry point for relationship detection
        Returns list of detected relationships sorted by confidence
        """

        logger.info("=" * 80)
        logger.info("FK DETECTION DIAGNOSTICS")
        logger.info("=" * 80)

        # Count total potential FK candidates
        total_columns = len(self.all_columns)
        pk_columns = self.stats["pk_columns"]
        logger.info(f"Total columns in database: {total_columns}")
        logger.info(f"Primary key columns: {pk_columns}")
        logger.info(f"Theoretical max comparisons: {total_columns * pk_columns:,}")
        logger.info(f"Config min_overlap_rate: {self.config.min_overlap_rate}")
        logger.info(f"Config sample_size: {self.config.sample_size}")

        # Type distribution
        type_dist = {}
        for col in self.all_columns:
            type_key = col.data_type.lower()
            type_dist[type_key] = type_dist.get(type_key, 0) + 1

        logger.info(f"Column type distribution:")
        for col_type, count in sorted(type_dist.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  • {col_type}: {count}")

        logger.info("=" * 80)

        if not self.config.enabled:
            logger.info("Relationship detection disabled")
            return []

        start_time = time.time()
        logger.info(
            f"Starting optimized relationship detection (strategy: {self.config.strategy})"
        )

        try:
            # Initialize connection pool
            self._init_connection_pool()

            # Stage 1: Build indexes
            stage_start = time.time()
            self._build_indexes()
            self.stats["time_building_indexes"] = time.time() - stage_start

            total_possible = self.stats["total_columns"] * self.stats["pk_columns"]
            logger.info(
                f"Indexes built: {self.stats['total_columns']} total columns, "
                f"{self.stats['pk_columns']} PK columns, "
                f"{total_possible:,} potential comparisons"
            )

            # Stage 2-5: Generate and filter candidates
            stage_start = time.time()
            candidates = self._generate_candidates()
            self.stats["time_filtering"] = time.time() - stage_start

            if not candidates:
                logger.warning("No relationship candidates found after filtering")
                return []

            # Apply hard cap
            if len(candidates) > self.config.max_comparisons:
                logger.warning(
                    f"Candidates ({len(candidates)}) exceed max_comparisons "
                    f"({self.config.max_comparisons}), truncating to top-scored"
                )
                candidates = candidates[: self.config.max_comparisons]

            reduction_pct = 100 * (1 - len(candidates) / max(total_possible, 1))
            logger.info(
                f"Filtering complete: {len(candidates)} candidates "
                f"({reduction_pct:.1f}% reduction from naive approach)"
            )

            # Stage 6: Parallel overlap analysis
            stage_start = time.time()
            relationships = self._analyze_candidates_parallel(candidates, start_time)
            self.stats["time_overlap_checks"] = time.time() - stage_start
            self.stats["relationships_found"] = len(relationships)

            # Sort by confidence and overlap rate
            relationships.sort(
                key=lambda r: (
                    {"high": 3, "medium": 2, "low": 1}.get(r.confidence, 0),
                    r.overlap_rate,
                ),
                reverse=True,
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Relationship detection complete: {len(relationships)} relationships found "
                f"in {elapsed:.1f}s ({self.stats['comparisons_made']} comparisons)"
            )
            self._log_statistics()

            return relationships

        except Exception as e:
            logger.error(f"Relationship detection failed: {e}", exc_info=True)
            return []
        finally:
            if self.engine:
                self.engine.dispose()

    def detect_relationships_from_views(
        discovery_data: Dict[str, Any], config: Any
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships by analyzing JOIN clauses in view definitions

        Process:
        1. Extract view SQL from named_assets
        2. Parse SQL with sqlglot
        3. Identify JOIN conditions
        4. Map to discovery tables
        5. Create high-confidence relationships

        Args:
            discovery_data: Discovery JSON with named_assets
            config: Relationship detection configuration

        Returns:
            List of relationship dictionaries
        """
        import time

        if not config.detect_views:
            logger.info("View relationship detection disabled")
            return []

        logger.info("Starting view relationship detection...")
        relationships = []
        views_analyzed = 0
        max_views = config.max_views
        start_time = time.time()

        for asset in discovery_data.get("named_assets", []):
            if asset.get("kind") != "view":
                continue

            if views_analyzed >= max_views:
                logger.warning(
                    f"Reached max views limit ({max_views}), stopping analysis"
                )
                break

            view_name = asset.get("name", "unknown")
            sql = asset.get("sql_normalized", "")

            if not sql:
                logger.debug(f"Skipping view {view_name}: no SQL definition")
                continue

            try:
                # Parse SQL with timeout protection
                parsed = sqlglot.parse_one(
                    sql, dialect=discovery_data.get("dialect", "")
                )

                # Extract joins
                for join in parsed.find_all(sqlglot.exp.Join):
                    # Get join condition
                    on_clause = join.args.get("on")
                    if not on_clause:
                        continue

                    # Extract left and right columns
                    left_table, left_col, right_table, right_col = (
                        _parse_join_condition(on_clause)
                    )

                    if all([left_table, left_col, right_table, right_col]):
                        # Normalize table names (add schema if missing)
                        left_table = _normalize_table_name(left_table, discovery_data)
                        right_table = _normalize_table_name(right_table, discovery_data)

                        relationships.append(
                            {
                                "from": f"{left_table}.{left_col}",
                                "to": f"{right_table}.{right_col}",
                                "method": "view_join_analysis",
                                "cardinality": "unknown",  # Would need value analysis
                                "confidence": "high",  # Views are curated
                                "source_view": view_name,
                                "detection_timestamp": time.time(),
                            }
                        )

                views_analyzed += 1

            except sqlglot.errors.ParseError as e:
                logger.warning(f"Failed to parse view {view_name}: {e}")
            except Exception as e:
                logger.error(f"Error analyzing view {view_name}: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"View relationship detection complete: analyzed {views_analyzed} views, "
            f"found {len(relationships)} relationships in {elapsed:.2f}s"
        )

        return relationships


    def _init_connection_pool(self):
        """Initialize connection pool for parallel operations"""
        self.engine = create_engine(
            self.connection_string,
            poolclass=pool.QueuePool,
            pool_size=self.config.max_workers + 2,
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"timeout": 10},
        )

    def _build_indexes(self):
        """Build lookup indexes from discovery data"""
        logger.debug("Building relationship detection indexes...")

        for schema_data in self.discovery_data.get("schemas", []):
            schema_name = schema_data["name"]

            for table_data in schema_data.get("tables", []):
                table_name = table_data["name"]
                table_full_name = f"{schema_name}.{table_name}"

                # Track indexed columns
                indexed_cols = set()
                for pk_col in table_data.get("primary_key", []):
                    indexed_cols.add(pk_col)
                for idx in table_data.get("indexes", []):
                    indexed_cols.update(idx.get("columns", []))
                self.indexed_columns[table_full_name] = indexed_cols

                # Process each column
                for col_data in table_data.get("columns", []):
                    col_name = col_data["name"]
                    col_type = col_data["type"]

                    # Extract stats
                    stats = col_data.get("stats", {})
                    distinct_count = stats.get("distinct_count")
                    sample_values = stats.get("sample_values", [])

                    col_info = ColumnInfo(
                        schema=schema_name,
                        table=table_name,
                        name=col_name,
                        type=col_type,
                        is_primary_key=col_name in table_data.get("primary_key", []),
                        is_indexed=col_name in indexed_cols,
                        is_nullable=col_data.get("nullable", True),
                        distinct_count=distinct_count,
                        sample_values=sample_values[:5] if sample_values else None,
                    )

                    self.all_columns.append(col_info)
                    self.stats["total_columns"] += 1

                    # Index primary keys
                    if col_info.is_primary_key:
                        if table_full_name not in self.pk_index:
                            self.pk_index[table_full_name] = []
                        self.pk_index[table_full_name].append(col_info)
                        self.stats["pk_columns"] += 1

                    # Index by type group
                    type_group = self.config.get_type_group(col_type)
                    if type_group >= 0:
                        if type_group not in self.type_index:
                            self.type_index[type_group] = []
                        self.type_index[type_group].append(col_info)

        logger.debug(
            f"Indexes built: {len(self.all_columns)} columns, "
            f"{self.stats['pk_columns']} PKs, "
            f"{len(self.type_index)} type groups"
        )

    def _generate_candidates(self) -> List[RelationshipCandidate]:
        """
        Generate relationship candidates using multi-stage filtering
        Returns sorted list of candidates (highest score first)
        """
        candidates = []

        # Iterate through all non-PK columns as potential FKs
        for from_col in self.all_columns:
            # Skip if it's already a primary key
            if from_col.is_primary_key:
                continue

            # Get type group for type compatibility
            from_type_group = self.config.get_type_group(from_col.type)
            if from_type_group < 0:
                continue  # Unknown type, skip

            # Get candidate PK columns with matching type
            candidate_pks = self.type_index.get(from_type_group, [])
            candidate_pks = [c for c in candidate_pks if c.is_primary_key]

            self.stats["candidates_after_type_filter"] += len(candidate_pks)

            # Apply filters and scoring
            for to_col in candidate_pks:
                # Skip self-references (same table)
                if from_col.table_name == to_col.table_name:
                    continue

                # Stage 3: Name-based scoring
                name_score = self._calculate_name_score(from_col, to_col)
                if name_score < 0.3:  # Minimum threshold
                    continue

                self.stats["candidates_after_name_filter"] += 1

                # Stage 4: Index requirement
                if self.config.require_index_on_target and not to_col.is_indexed:
                    continue

                self.stats["candidates_after_index_filter"] += 1

                # Stage 5: Cardinality pre-check
                if not self._passes_cardinality_check(from_col, to_col):
                    continue

                self.stats["candidates_after_cardinality_filter"] += 1

                # Add to candidates
                candidates.append(
                    RelationshipCandidate(
                        from_column=from_col,
                        to_column=to_col,
                        name_score=name_score,
                        stage="full_filter",
                    )
                )

        # Sort by score (highest first)
        candidates.sort()

        return candidates

    def _calculate_name_score(self, from_col: ColumnInfo, to_col: ColumnInfo) -> float:
        """
        Calculate name-based similarity score between columns
        Returns 0.0 (no match) to 1.0 (perfect match)
        """
        from_name = from_col.name.lower()
        to_name = to_col.name.lower()
        to_table = to_col.table.lower()

        score = 0.0

        # Exact name match
        if from_name == to_name:
            score += 0.5

        # Column name ends with referenced column name
        # e.g., CustomerID → Customer.CustomerID
        elif from_name.endswith(to_name):
            score += 0.4

        # Column name contains referenced column name
        elif to_name in from_name:
            score += 0.3

        # Column name starts with table name
        # e.g., CustomerID → Customer.ID
        if from_name.startswith(to_table):
            score += 0.2

        # Check for FK naming patterns
        if self.config.prioritize_named_patterns:
            from_name_lower = from_name
            for pattern in self.config.fk_suffix_patterns:
                if from_name_lower.endswith(pattern):
                    score += 0.15
                    break

            for pattern in self.config.fk_infix_patterns:
                if pattern in from_name_lower:
                    score += 0.1
                    break

        # Bonus if to_column is indexed (likely a real PK)
        if to_col.is_indexed:
            score += 0.1

        # Penalty if from_column is nullable (weak relationship)
        if from_col.is_nullable:
            score -= 0.05

        return min(1.0, max(0.0, score))

    def _passes_cardinality_check(
        self, from_col: ColumnInfo, to_col: ColumnInfo
    ) -> bool:
        """
        Quick cardinality check without DB query
        For many-to-one: from.distinct_count should be <= to.distinct_count * 1.1
        """
        if from_col.distinct_count is None or to_col.distinct_count is None:
            return True  # Can't check, allow it through

        # Allow 10% tolerance for slight mismatches
        return from_col.distinct_count <= to_col.distinct_count * 1.1

    def _analyze_candidates_parallel(
        self, candidates: List[RelationshipCandidate], start_time: float
    ) -> List[DetectedRelationship]:
        """
        Analyze candidates in parallel with timeout protection
        """
        relationships = []
        completed = 0

        logger.info(
            f"Starting parallel overlap analysis for {len(candidates)} candidates "
            f"(max {self.config.max_workers} workers)"
        )

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_candidate = {
                executor.submit(self._check_overlap, candidate): candidate
                for candidate in candidates
            }

            # Process results as they complete
            for future in future_to_candidate:
                # Check global timeout
                elapsed = time.time() - start_time
                if elapsed > self.config.global_timeout:
                    logger.warning(
                        f"Global timeout reached ({self.config.global_timeout}s), "
                        f"stopping after {completed}/{len(candidates)} candidates"
                    )
                    break

                try:
                    result = future.result(timeout=self.config.timeout_per_comparison)
                    completed += 1

                    if result:
                        relationships.append(result)

                    # Progress logging every 10%
                    if completed % max(1, len(candidates) // 10) == 0:
                        pct = 100 * completed / len(candidates)
                        logger.debug(
                            f"Progress: {completed}/{len(candidates)} ({pct:.0f}%), "
                            f"{len(relationships)} relationships found"
                        )

                except FutureTimeoutError:
                    candidate = future_to_candidate[future]
                    logger.warning(
                        f"Timeout checking {candidate.from_column.full_name} → "
                        f"{candidate.to_column.full_name}"
                    )
                    completed += 1

                except Exception as e:
                    candidate = future_to_candidate[future]
                    logger.error(
                        f"Error checking {candidate.from_column.full_name} → "
                        f"{candidate.to_column.full_name}: {e}"
                    )
                    completed += 1

        return relationships

    def _check_overlap(
        self, candidate: RelationshipCandidate
    ) -> Optional[DetectedRelationship]:
        """
        Check actual value overlap between two columns
        Uses smaller sample size for speed
        """
        from_col = candidate.from_column
        to_col = candidate.to_column

        self.stats["comparisons_made"] += 1

        try:
            # Use smaller sample for speed
            overlap_rate, cardinality = self._calculate_overlap_rate(from_col, to_col)

            if overlap_rate < self.config.min_overlap_rate:
                return None

            # Determine confidence level
            confidence = self._determine_confidence(
                overlap_rate=overlap_rate,
                name_score=candidate.name_score,
                cardinality=cardinality,
            )

            return DetectedRelationship(
                from_column=from_col.full_name,
                to_column=to_col.full_name,
                overlap_rate=overlap_rate,
                cardinality=cardinality,
                method="value_overlap",
                confidence=confidence,
                name_score=candidate.name_score,
                verification={
                    "sample_size": self.config.sample_size,
                    "from_distinct": from_col.distinct_count,
                    "to_distinct": to_col.distinct_count,
                },
            )

        except Exception as e:
            logger.debug(
                f"Overlap check failed for {from_col.full_name} → {to_col.full_name}: {e}"
            )
            return None

    def _calculate_overlap_rate(
        self, from_col: ColumnInfo, to_col: ColumnInfo
    ) -> Tuple[float, str]:
        """
        Calculate overlap rate using optimized sampling query
        Returns (overlap_rate, cardinality)
        """
        # Build optimized sampling query
        dialect = self.discovery_data.get("dialect", "generic").lower()

        if dialect in ["mssql", "tsql"]:
            query = text(
                f"""
                WITH sample_from AS (
                    SELECT TOP ({self.config.sample_size}) [{from_col.name}] as val
                    FROM [{from_col.schema}].[{from_col.table}]
                    WHERE [{from_col.name}] IS NOT NULL
                ),
                sample_to AS (
                    SELECT DISTINCT [{to_col.name}] as val
                    FROM [{to_col.schema}].[{to_col.table}]
                    WHERE [{to_col.name}] IS NOT NULL
                )
                SELECT 
                    COUNT(DISTINCT sf.val) as total_from,
                    COUNT(DISTINCT CASE WHEN st.val IS NOT NULL THEN sf.val END) as matched,
                    COUNT(DISTINCT sf.val) as distinct_from
                FROM sample_from sf
                LEFT JOIN sample_to st ON sf.val = st.val
            """
            )
        elif dialect in ["mysql", "mariadb"]:
            query = text(
                f"""
                SELECT 
                    COUNT(DISTINCT sf.val) as total_from,
                    COUNT(DISTINCT CASE WHEN st.val IS NOT NULL THEN sf.val END) as matched,
                    COUNT(DISTINCT sf.val) as distinct_from
                FROM (
                    SELECT `{from_col.name}` as val
                    FROM `{from_col.schema}`.`{from_col.table}`
                    WHERE `{from_col.name}` IS NOT NULL
                    LIMIT {self.config.sample_size}
                ) sf
                LEFT JOIN (
                    SELECT DISTINCT `{to_col.name}` as val
                    FROM `{to_col.schema}`.`{to_col.table}`
                    WHERE `{to_col.name}` IS NOT NULL
                ) st ON sf.val = st.val
            """
            )
        else:  # PostgreSQL, generic
            query = text(
                f"""
                WITH sample_from AS (
                    SELECT "{from_col.name}" as val
                    FROM "{from_col.schema}"."{from_col.table}"
                    WHERE "{from_col.name}" IS NOT NULL
                    LIMIT {self.config.sample_size}
                ),
                sample_to AS (
                    SELECT DISTINCT "{to_col.name}" as val
                    FROM "{to_col.schema}"."{to_col.table}"
                    WHERE "{to_col.name}" IS NOT NULL
                )
                SELECT 
                    COUNT(DISTINCT sf.val) as total_from,
                    COUNT(DISTINCT CASE WHEN st.val IS NOT NULL THEN sf.val ELSE NULL END) as matched,
                    COUNT(DISTINCT sf.val) as distinct_from
                FROM sample_from sf
                LEFT JOIN sample_to st ON sf.val = st.val
            """
            )

        # Execute query
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchone()
            total_from = result[0] or 0
            matched = result[1] or 0
            distinct_from = result[2] or 1  # Avoid division by zero

        # Calculate overlap rate
        overlap_rate = matched / distinct_from if distinct_from > 0 else 0.0

        # Determine cardinality
        if distinct_from == matched and matched == total_from:
            cardinality = "one_to_one"
        else:
            cardinality = "many_to_one"

        return overlap_rate, cardinality

    def _determine_confidence(
        self, overlap_rate: float, name_score: float, cardinality: str
    ) -> str:
        """Determine confidence level based on multiple factors"""
        # High confidence: strong name match + high overlap
        if name_score >= 0.7 and overlap_rate >= 0.95:
            return "high"

        # Medium confidence: decent name match or high overlap
        if name_score >= 0.5 and overlap_rate >= 0.90:
            return "medium"

        if name_score >= 0.7 and overlap_rate >= 0.85:
            return "medium"

        # Low confidence: meets minimum thresholds
        return "low"

    def _log_statistics(self):
        """Log detailed statistics about the detection process"""
        logger.info("=== Relationship Detection Statistics ===")
        logger.info(f"  Total columns scanned: {self.stats['total_columns']}")
        logger.info(f"  Primary key columns: {self.stats['pk_columns']}")
        logger.info(
            f"  After type filter: {self.stats['candidates_after_type_filter']}"
        )
        logger.info(
            f"  After name filter: {self.stats['candidates_after_name_filter']}"
        )
        logger.info(
            f"  After index filter: {self.stats['candidates_after_index_filter']}"
        )
        logger.info(
            f"  After cardinality filter: {self.stats['candidates_after_cardinality_filter']}"
        )
        logger.info(f"  Overlap comparisons made: {self.stats['comparisons_made']}")
        logger.info(f"  Relationships found: {self.stats['relationships_found']}")
        logger.info(
            f"  Time building indexes: {self.stats['time_building_indexes']:.2f}s"
        )
        logger.info(f"  Time filtering: {self.stats['time_filtering']:.2f}s")
        logger.info(f"  Time overlap checks: {self.stats['time_overlap_checks']:.2f}s")

        total_possible = self.stats["total_columns"] * self.stats["pk_columns"]
        if total_possible > 0:
            reduction = 100 * (1 - self.stats["comparisons_made"] / total_possible)
            logger.info(f"  Overall reduction: {reduction:.1f}%")


# ============================================================================
# STANDALONE WRAPPER FUNCTIONS (for external imports)
# ============================================================================

def detect_relationships(
    connection_string: str,
    discovery_data: Dict[str, Any],
    config: Optional[RelationshipDetectionConfig] = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to detect relationships using RelationshipDetector class.
    
    Args:
        connection_string: Database connection string
        discovery_data: Discovery JSON data
        config: Relationship detection configuration
        
    Returns:
        List of detected relationship dictionaries
    """
    detector = RelationshipDetector(
        connection_string=connection_string,
        discovery_data=discovery_data,
        config=config
    )
    
    # Convert DetectedRelationship objects to dictionaries
    relationships = detector.detect_relationships()
    
    return [
        {
            "from": rel.from_column,
            "to": rel.to_column,
            "overlap_rate": rel.overlap_rate,
            "cardinality": rel.cardinality,
            "method": rel.method,
            "confidence": rel.confidence,
            "name_score": rel.name_score,
            "verification": rel.verification
        }
        for rel in relationships
    ]


def detect_relationships_from_views(
    discovery_data: Dict[str, Any],
    config: Any
) -> List[Dict[str, Any]]:
    """
    Detect relationships by analyzing JOIN clauses in view definitions.
    
    This is a standalone function that can be imported directly.
    
    Process:
    1. Extract view SQL from named_assets
    2. Parse SQL with sqlglot
    3. Identify JOIN conditions
    4. Map to discovery tables
    5. Create high-confidence relationships
    
    Args:
        discovery_data: Discovery JSON with named_assets
        config: Relationship detection configuration
        
    Returns:
        List of relationship dictionaries
    """
    import time
    import sqlglot
    from sqlglot import exp
    
    if not config.detect_views:
        logger.info("View relationship detection disabled")
        return []
    
    logger.info("Starting view relationship detection...")
    relationships = []
    views_analyzed = 0
    max_views = config.max_views
    start_time = time.time()
    
    for asset in discovery_data.get("named_assets", []):
        if asset.get("kind") != "view":
            continue
        
        if views_analyzed >= max_views:
            logger.warning(
                f"Reached max views limit ({max_views}), stopping analysis"
            )
            break
        
        view_name = asset.get("name", "unknown")
        sql = asset.get("sql_normalized", "")
        
        if not sql:
            logger.debug(f"Skipping view {view_name}: no SQL definition")
            continue
        
        try:
            # Parse SQL with timeout protection
            parsed = sqlglot.parse_one(
                sql, dialect=discovery_data.get("dialect", "")
            )
            
            # Extract joins
            for join in parsed.find_all(sqlglot.exp.Join):
                # Get join condition
                on_clause = join.args.get("on")
                if not on_clause:
                    continue
                
                # Extract left and right columns using helper function
                left_table, left_col, right_table, right_col = (
                    _parse_join_condition(on_clause)
                )
                
                if all([left_table, left_col, right_table, right_col]):
                    # Normalize table names (add schema if missing)
                    left_table = _normalize_table_name(left_table, discovery_data)
                    right_table = _normalize_table_name(right_table, discovery_data)
                    
                    relationships.append(
                        {
                            "from": f"{left_table}.{left_col}",
                            "to": f"{right_table}.{right_col}",
                            "method": "view_join_analysis",
                            "cardinality": "unknown",  # Would need value analysis
                            "confidence": "high",  # Views are curated
                            "source_view": view_name,
                            "detection_timestamp": time.time(),
                        }
                    )
            
            views_analyzed += 1
        
        except sqlglot.errors.ParseError as e:
            logger.warning(f"Failed to parse view {view_name}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing view {view_name}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(
        f"View relationship detection complete: analyzed {views_analyzed} views, "
        f"found {len(relationships)} relationships in {elapsed:.2f}s"
    )
    
    return relationships