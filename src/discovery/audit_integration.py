"""
Audit Integration with Discovery

This module provides integration between the audit collector and discovery engine:
1. Enhance discovery with access patterns
2. Filter out unused/history tables
3. Boost relationship confidence based on actual query patterns
4. Prioritize tables for semantic model building
5. Map production audit data to development discovery databases

Usage:
    from src.discovery.audit_integration import AuditEnhancedDiscovery

    enhanced = AuditEnhancedDiscovery()
    discovery_data = enhanced.run_enhanced_discovery()

    # With database mapping (prod audit -> dev discovery)
    enhanced = AuditEnhancedDiscovery()
    enhanced.load_audit_with_mapping(dev_server="cardinal03", dev_database="YPS2907")
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass

from src.discovery.audit_collector import (
    AuditCollector,
    AuditConfig,
    AuditReport,
    AuditTimeRange,
    TableAccessMetrics
)

logger = logging.getLogger(__name__)


# ============================================================================
# AUDIT-ENHANCED TABLE RANKING
# ============================================================================

@dataclass
class TableRanking:
    """Ranking information for a table"""
    full_name: str
    schema_name: str
    table_name: str

    # Base ranking (from asset type)
    base_rank: int  # 1=highest priority

    # Audit-based adjustments
    access_pattern: str  # hot, warm, cold, archive, unused
    access_score: float  # 0-100
    is_history_table: bool
    history_of: Optional[str]  # Base table if this is history

    # Final ranking
    final_rank: int
    include_in_model: bool
    reason: str


class AuditTableRanker:
    """
    Ranks tables for semantic model inclusion based on audit data

    Priority order:
    1. Hot tables (actively queried) - always include
    2. Warm tables with views/SPs - include
    3. Cold tables referenced in joins - conditional
    4. History/archive tables - skip (link to base)
    5. Unused tables - skip

    Integration points:
    - discovery_engine.py: Filter tables before discovery
    - model_builder.py: Prioritize entities/facts
    - relationship_detector.py: Boost join confidence
    """

    # Rank values (lower = higher priority)
    RANK_HOT = 1
    RANK_WARM = 2
    RANK_COLD = 3
    RANK_ARCHIVE = 4
    RANK_HISTORY = 5
    RANK_UNUSED = 6

    def __init__(self, audit_report: AuditReport):
        self.audit_report = audit_report
        self._build_lookups()

    def _build_lookups(self):
        """Build lookup dictionaries for fast access"""
        self.metrics_by_table = {
            m.full_name.lower(): m for m in self.audit_report.table_metrics
        }
        self.join_tables = self._extract_join_tables()
        self.prioritized_set = set(t.lower() for t in self.audit_report.tables_to_prioritize)
        self.skip_set = set(t.lower() for t in self.audit_report.tables_to_skip)

    def _extract_join_tables(self) -> Set[str]:
        """Extract tables referenced in frequent joins"""
        tables = set()
        for join_key in self.audit_report.join_frequency.keys():
            # Parse "table.col=table.col" format
            parts = join_key.replace('=', '.').split('.')
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    tables.add(parts[i].lower())
        return tables

    def rank_table(
        self,
        full_name: str,
        asset_type: str = "table"
    ) -> TableRanking:
        """
        Rank a single table for semantic model inclusion

        Args:
            full_name: Schema.TableName
            asset_type: 'table', 'view', 'stored_procedure', 'rdl'

        Returns:
            TableRanking with inclusion decision
        """
        full_name_lower = full_name.lower()
        parts = full_name.split('.')
        schema_name = parts[0] if len(parts) > 1 else 'dbo'
        table_name = parts[-1]

        # Get audit metrics
        metrics = self.metrics_by_table.get(full_name_lower)

        # Base rank from asset type
        base_rank_map = {
            'view': 1,
            'stored_procedure': 2,
            'rdl': 3,
            'table': 4
        }
        base_rank = base_rank_map.get(asset_type, 4)

        # Default values if no audit data
        if not metrics:
            return TableRanking(
                full_name=full_name,
                schema_name=schema_name,
                table_name=table_name,
                base_rank=base_rank,
                access_pattern="unknown",
                access_score=50.0,
                is_history_table=False,
                history_of=None,
                final_rank=base_rank,
                include_in_model=True,
                reason=f"No audit data - include by default ({asset_type})"
            )

        # Determine access-based rank
        access_rank_map = {
            'hot': self.RANK_HOT,
            'warm': self.RANK_WARM,
            'cold': self.RANK_COLD,
            'archive': self.RANK_ARCHIVE,
            'unused': self.RANK_UNUSED
        }
        access_rank = access_rank_map.get(metrics.access_pattern, self.RANK_COLD)

        # History table handling
        if metrics.is_likely_history:
            access_rank = self.RANK_HISTORY

        # Compute final rank (weighted)
        # Asset type matters more for views/SPs
        if asset_type in ('view', 'stored_procedure'):
            final_rank = min(base_rank, access_rank)
        else:
            final_rank = access_rank

        # Inclusion decision
        include = True
        reason = ""

        if metrics.is_likely_history:
            include = False
            base = metrics.history_base_table or "unknown"
            reason = f"History table of {base} - skip"
        elif metrics.access_pattern == 'unused':
            include = False
            reason = "Unused table - skip"
        elif metrics.access_pattern == 'archive' and metrics.total_writes == 0:
            # Archive tables with no writes might be old snapshots
            if full_name_lower not in self.join_tables:
                include = False
                reason = "Archive table (read-only, not in joins) - skip"
            else:
                reason = f"Archive table but used in joins - include (rank {final_rank})"
        elif metrics.access_pattern == 'hot':
            reason = f"Hot table (score={metrics.access_score:.1f}) - prioritize"
        elif metrics.access_pattern == 'warm':
            reason = f"Warm table (score={metrics.access_score:.1f}) - include"
        else:
            reason = f"Cold table (score={metrics.access_score:.1f}) - include"

        return TableRanking(
            full_name=full_name,
            schema_name=schema_name,
            table_name=table_name,
            base_rank=base_rank,
            access_pattern=metrics.access_pattern,
            access_score=metrics.access_score,
            is_history_table=metrics.is_likely_history,
            history_of=metrics.history_base_table,
            final_rank=final_rank,
            include_in_model=include,
            reason=reason
        )

    def rank_all_tables(
        self,
        tables: List[Dict[str, Any]]
    ) -> List[TableRanking]:
        """
        Rank all tables from discovery

        Args:
            tables: List of table dicts with 'name' and optionally 'type'

        Returns:
            Sorted list of TableRanking (highest priority first)
        """
        rankings = []

        for table in tables:
            name = table.get('name') or table.get('full_name', '')
            asset_type = table.get('type', 'table')

            ranking = self.rank_table(name, asset_type)
            rankings.append(ranking)

        # Sort by final_rank (ascending = higher priority)
        return sorted(rankings, key=lambda r: (r.final_rank, -r.access_score))

    def get_tables_to_include(self) -> List[str]:
        """Get list of tables that should be included in semantic model"""
        return [
            m.full_name for m in self.audit_report.table_metrics
            if m.access_pattern in ('hot', 'warm', 'cold')
            and not m.is_likely_history
        ]

    def get_tables_to_skip(self) -> List[str]:
        """Get list of tables to skip"""
        return self.audit_report.tables_to_skip

    def get_history_mappings(self) -> Dict[str, str]:
        """Get mapping of history tables to their base tables"""
        return self.audit_report.history_table_mappings


# ============================================================================
# RELATIONSHIP CONFIDENCE BOOSTER
# ============================================================================

class AuditRelationshipBooster:
    """
    Boost relationship confidence based on actual query patterns

    If a join is frequently used in production queries,
    it should have higher confidence in the semantic model.
    """

    # Confidence boost thresholds
    HIGH_FREQUENCY_THRESHOLD = 1000  # executions
    MEDIUM_FREQUENCY_THRESHOLD = 100

    def __init__(self, audit_report: AuditReport):
        self.audit_report = audit_report
        self.join_frequency = audit_report.join_frequency

    def get_join_confidence_boost(
        self,
        from_col: str,
        to_col: str
    ) -> Dict[str, Any]:
        """
        Get confidence boost for a relationship based on query patterns

        Args:
            from_col: Source column (schema.table.column)
            to_col: Target column (schema.table.column)

        Returns:
            Dict with boost factor and evidence
        """
        # Normalize join key (sort for bidirectional match)
        key1 = f"{from_col}={to_col}"
        key2 = f"{to_col}={from_col}"

        # Try both directions
        frequency = self.join_frequency.get(key1, 0) or self.join_frequency.get(key2, 0)

        # Also try with just table.column (without schema)
        if frequency == 0:
            short_from = '.'.join(from_col.split('.')[-2:])
            short_to = '.'.join(to_col.split('.')[-2:])
            key3 = "=".join(sorted([short_from, short_to]))

            for join_key, count in self.join_frequency.items():
                if key3 in join_key.lower():
                    frequency = max(frequency, count)
                    break

        # Determine boost
        if frequency >= self.HIGH_FREQUENCY_THRESHOLD:
            boost = "high"
            confidence_adjustment = 0.2  # +20% confidence
        elif frequency >= self.MEDIUM_FREQUENCY_THRESHOLD:
            boost = "medium"
            confidence_adjustment = 0.1  # +10% confidence
        elif frequency > 0:
            boost = "low"
            confidence_adjustment = 0.05  # +5% confidence
        else:
            boost = "none"
            confidence_adjustment = 0.0

        return {
            "frequency": frequency,
            "boost_level": boost,
            "confidence_adjustment": confidence_adjustment,
            "evidence": f"Used in {frequency} query executions" if frequency > 0 else "No query evidence"
        }

    def boost_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply confidence boost to a list of relationships

        Args:
            relationships: List of relationship dicts with 'from', 'to', 'confidence'

        Returns:
            Updated relationships with boosted confidence
        """
        boosted = []

        for rel in relationships:
            from_col = rel.get('from', '')
            to_col = rel.get('to', '')

            boost_info = self.get_join_confidence_boost(from_col, to_col)

            # Clone and update
            updated = rel.copy()

            # Apply boost
            if boost_info['boost_level'] != 'none':
                original_confidence = rel.get('confidence', 'medium')
                confidence_rank = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}

                current_rank = confidence_rank.get(original_confidence, 2)
                new_rank = min(4, current_rank + 1) if boost_info['boost_level'] in ('high', 'medium') else current_rank

                rank_to_confidence = {1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}
                updated['confidence'] = rank_to_confidence.get(new_rank, original_confidence)

                updated['audit_boost'] = {
                    "original_confidence": original_confidence,
                    "boosted_confidence": updated['confidence'],
                    "query_frequency": boost_info['frequency'],
                    "boost_level": boost_info['boost_level']
                }

            boosted.append(updated)

        return boosted


# ============================================================================
# ENHANCED DISCOVERY INTEGRATION
# ============================================================================

class AuditEnhancedDiscovery:
    """
    Combines audit data with discovery for better semantic model building

    Supports mapping between production audit databases and development discovery databases.

    Usage:
        enhanced = AuditEnhancedDiscovery()

        # Option 1: Collect fresh audit then run discovery
        enhanced.collect_audit(lookback_days=30)
        discovery_data = enhanced.run_enhanced_discovery()

        # Option 2: Use existing audit for discovery filtering
        enhanced.load_audit()
        filtered_tables = enhanced.filter_discovery_tables(discovery_data)

        # Option 3: Use audit from production mapped to dev database
        enhanced.load_audit_with_mapping(dev_server="cardinal03", dev_database="YPS2907")
        filtered_tables = enhanced.filter_discovery_tables(discovery_data)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        from config.settings import get_path_config, get_database_mappings

        self.path_config = get_path_config()
        self.cache_dir = cache_dir or self.path_config.cache_dir
        self.audit_config = AuditConfig.from_env()
        self.audit_collector = AuditCollector(self.audit_config, self.cache_dir)
        self.audit_report: Optional[AuditReport] = None
        self.ranker: Optional[AuditTableRanker] = None
        self.booster: Optional[AuditRelationshipBooster] = None

        # Database mapping support
        self.database_mappings = get_database_mappings()
        self.active_mapping: Optional['DatabaseMapping'] = None

    def collect_audit(
        self,
        force: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> AuditReport:
        """
        Collect audit from production database

        Args:
            force: Bypass execution window
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            lookback_days: Days to look back

        Returns:
            AuditReport
        """
        from datetime import datetime

        time_range = None
        if start_date or end_date or lookback_days:
            time_range = AuditTimeRange(
                start_date=datetime.fromisoformat(start_date) if start_date else None,
                end_date=datetime.fromisoformat(end_date) if end_date else None,
                lookback_days=lookback_days or self.audit_config.lookback_days
            )

        self.audit_report = self.audit_collector.collect(force=force, time_range=time_range)
        self._initialize_helpers()

        return self.audit_report

    def load_audit(self) -> Optional[AuditReport]:
        """Load cached audit report"""
        self.audit_report = self.audit_collector.load_cached_report()

        if self.audit_report:
            self._initialize_helpers()

        return self.audit_report

    def load_audit_with_mapping(
        self,
        dev_server: Optional[str] = None,
        dev_database: Optional[str] = None
    ) -> Optional[AuditReport]:
        """
        Load audit and apply database mapping for development discovery.

        This method:
        1. Loads the cached audit from production database
        2. Finds the mapping configuration for the dev database
        3. Validates that audit source matches expected production database

        Args:
            dev_server: Development server name (optional, auto-detected from connection string)
            dev_database: Development database name (optional, auto-detected from connection string)

        Returns:
            AuditReport if loaded and mapping valid, None otherwise
        """
        # Load the audit first
        self.audit_report = self.audit_collector.load_cached_report()

        if not self.audit_report:
            logger.warning("No cached audit report found")
            return None

        # Auto-detect dev server/database from connection string if not provided
        if not dev_server or not dev_database:
            detected = self._parse_connection_string()
            dev_server = dev_server or detected[0]
            dev_database = dev_database or detected[1]

        if not dev_server or not dev_database:
            logger.warning("Could not determine development server/database")
            return self.audit_report

        # Find mapping for this dev database
        mapping = self.database_mappings.get_prod_for_dev(dev_server, dev_database)

        if not mapping:
            logger.info(
                f"No database mapping configured for {dev_server}/{dev_database}. "
                f"Using audit data without mapping."
            )
            self._initialize_helpers()
            return self.audit_report

        # Validate audit source matches production database in mapping
        audit_server = self.audit_report.source_server.lower()
        audit_database = self.audit_report.database_name.lower()

        if not mapping.matches_prod(audit_server, audit_database):
            logger.warning(
                f"Audit source mismatch! "
                f"Expected: {mapping.prod_server}/{mapping.prod_database}, "
                f"Got: {self.audit_report.source_server}/{self.audit_report.database_name}"
            )
            # Still use audit, but warn
            self._initialize_helpers()
            return self.audit_report

        # Mapping is valid
        self.active_mapping = mapping
        logger.info(
            f"Database mapping active: "
            f"{mapping.prod_server}/{mapping.prod_database} -> "
            f"{mapping.dev_server}/{mapping.dev_database}"
        )

        self._initialize_helpers()
        return self.audit_report

    def _parse_connection_string(self) -> Tuple[Optional[str], Optional[str]]:
        """Parse server and database from DATABASE_CONNECTION_STRING"""
        import os
        import re

        conn_str = os.getenv('DATABASE_CONNECTION_STRING', '')

        # Extract server from connection string
        # Format: mssql+pyodbc://user:pass@server:port/database?...
        server_match = re.search(r'@([^:/]+)', conn_str)
        server = server_match.group(1) if server_match else None

        # Extract database from connection string
        db_match = re.search(r'/([^?]+)\?', conn_str)
        if not db_match:
            db_match = re.search(r'/([^/]+)$', conn_str)
        database = db_match.group(1) if db_match else None

        return server, database

    def get_mapping_info(self) -> Optional[Dict[str, str]]:
        """Get information about the active database mapping"""
        if not self.active_mapping:
            return None

        return {
            "prod_server": self.active_mapping.prod_server,
            "prod_database": self.active_mapping.prod_database,
            "dev_server": self.active_mapping.dev_server,
            "dev_database": self.active_mapping.dev_database,
            "audit_source": f"{self.audit_report.source_server}/{self.audit_report.database_name}" if self.audit_report else "N/A"
        }

    def list_available_mappings(self) -> List[Dict[str, str]]:
        """List all configured database mappings"""
        return self.database_mappings.list_mappings()

    def _initialize_helpers(self):
        """Initialize ranker and booster with audit data"""
        if self.audit_report:
            self.ranker = AuditTableRanker(self.audit_report)
            self.booster = AuditRelationshipBooster(self.audit_report)

    def filter_discovery_tables(
        self,
        discovery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter discovery data based on audit recommendations

        Args:
            discovery_data: Raw discovery JSON

        Returns:
            Filtered discovery data with audit metadata
        """
        if not self.audit_report or not self.ranker:
            logger.warning("No audit data - returning unfiltered discovery")
            return discovery_data

        # Clone discovery data
        filtered = discovery_data.copy()
        filtered['audit_metadata'] = {
            'audit_date': self.audit_report.collected_at,
            'source_server': self.audit_report.source_server,
            'tables_filtered': 0,
            'tables_prioritized': 0
        }

        tables_filtered = 0
        tables_prioritized = 0

        # Filter tables in each schema
        for schema in filtered.get('schemas', []):
            original_tables = schema.get('tables', [])
            filtered_tables = []

            for table in original_tables:
                full_name = f"{schema['name']}.{table['name']}"
                ranking = self.ranker.rank_table(full_name, 'table')

                if ranking.include_in_model:
                    # Add audit metadata to table
                    table['audit'] = {
                        'access_pattern': ranking.access_pattern,
                        'access_score': ranking.access_score,
                        'rank': ranking.final_rank,
                        'reason': ranking.reason
                    }
                    filtered_tables.append(table)

                    if ranking.access_pattern == 'hot':
                        tables_prioritized += 1
                else:
                    tables_filtered += 1
                    logger.debug(f"Filtered out: {full_name} - {ranking.reason}")

            schema['tables'] = filtered_tables

        filtered['audit_metadata']['tables_filtered'] = tables_filtered
        filtered['audit_metadata']['tables_prioritized'] = tables_prioritized

        logger.info(f"Audit filtering: {tables_prioritized} prioritized, {tables_filtered} filtered out")

        return filtered

    def boost_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Boost relationship confidence based on query patterns

        Args:
            relationships: List from relationship_detector

        Returns:
            Boosted relationships
        """
        if not self.booster:
            logger.warning("No audit data - returning unboosted relationships")
            return relationships

        return self.booster.boost_relationships(relationships)

    def get_table_exclusions(self) -> List[str]:
        """Get list of tables to exclude from discovery"""
        if not self.audit_report:
            return []

        return self.audit_report.tables_to_skip

    def get_priority_tables(self) -> List[str]:
        """Get list of high-priority tables"""
        if not self.audit_report:
            return []

        return self.audit_report.tables_to_prioritize

    def get_history_table_info(self) -> Dict[str, str]:
        """Get mapping of history tables to base tables"""
        if not self.audit_report:
            return {}

        return self.audit_report.history_table_mappings

    def generate_discovery_config_overrides(self) -> Dict[str, Any]:
        """
        Generate config overrides for discovery based on audit

        Returns dict that can be merged with discovery config
        """
        if not self.audit_report:
            return {}

        # Get tables to skip
        skip_tables = self.get_table_exclusions()

        # Convert to exclusion patterns
        table_exclusions = []
        for table in skip_tables:
            # Extract just table name
            parts = table.split('.')
            table_name = parts[-1] if parts else table
            table_exclusions.append(table_name)

        return {
            'additional_table_exclusions': table_exclusions,
            'priority_tables': self.get_priority_tables(),
            'history_mappings': self.get_history_table_info()
        }

    def print_summary(self):
        """Print audit summary to console"""
        if not self.audit_report:
            print("No audit data loaded")
            return

        r = self.audit_report

        print("\n" + "=" * 60)
        print("AUDIT SUMMARY")
        print("=" * 60)
        print(f"Source:        {r.source_server} / {r.database_name}")
        print(f"Collected:     {r.collected_at}")
        print(f"Period:        {r.audit_start_date} to {r.audit_end_date}")

        # Show mapping info if active
        if self.active_mapping:
            print()
            print("DATABASE MAPPING:")
            print(f"  Production:  {self.active_mapping.prod_server} / {self.active_mapping.prod_database}")
            print(f"  Development: {self.active_mapping.dev_server} / {self.active_mapping.dev_database}")
            print(f"  Status:      Active - audit data will be applied to dev discovery")

        print()
        print("TABLE ACCESS PATTERNS:")
        print(f"  Hot:         {r.hot_tables_count:>5} tables (actively queried)")
        print(f"  Warm:        {r.warm_tables_count:>5} tables (moderately used)")
        print(f"  Cold:        {r.cold_tables_count:>5} tables (rarely used)")
        print(f"  Unused:      {r.unused_tables_count:>5} tables (no recent access)")
        print()
        print("HISTORY/DUPLICATE DETECTION:")
        print(f"  History:     {r.likely_history_tables_count:>5} tables detected")
        print(f"  Duplicates:  {r.likely_duplicates_count:>5} tables detected")
        print()
        print("RECOMMENDATIONS:")
        print(f"  Prioritize:  {len(r.tables_to_prioritize):>5} tables")
        print(f"  Skip:        {len(r.tables_to_skip):>5} tables")
        print()

        if r.join_frequency:
            print("TOP 10 JOIN PATTERNS:")
            for join, count in list(r.join_frequency.items())[:10]:
                print(f"  {count:>8}x  {join}")

        # Show configured mappings
        mappings = self.list_available_mappings()
        if mappings:
            print()
            print("CONFIGURED DATABASE MAPPINGS:")
            for m in mappings:
                status = "(active)" if self.active_mapping and \
                    m['production'].lower() == f"{self.active_mapping.prod_server}/{self.active_mapping.prod_database}".lower() \
                    else ""
                print(f"  {m['production']} -> {m['development']} {status}")

        print("=" * 60)
