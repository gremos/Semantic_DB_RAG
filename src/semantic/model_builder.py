"""
Semantic Model Builder for GPT-5 Semantic Modeling & SQL Q&A System

Phase 2: Semantic Model Construction
- Analyzes discovery JSON using LLM
- Classifies tables into entities, dimensions, and facts
- Enriches columns with semantic metadata
- Builds relationships and rankings
- Enforces JSON schema validation with retries
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel, ValidationError, Field, validator

from config.settings import (
    get_settings,
    get_llm_config,
    get_path_config,
    get_discovery_config
)

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR VALIDATION (QuadRails - Constraint)
# ============================================================================

from typing import Union

class ColumnMetadata(BaseModel):
    """Column metadata in semantic model"""
    name: str
    role: Union[str, List[str]] = "attribute"  # Default to attribute
    semantic_type: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    
    @validator('role', pre=True)
    def validate_role(cls, v):
        """Ensure role is valid and normalized"""
        valid_roles = {'primary_key', 'foreign_key', 'attribute', 'label', 'measure', 'dimension_key'}
        if isinstance(v, str):
            if v not in valid_roles:
                logger.warning(f"Invalid role: {v}, defaulting to 'attribute'")
                return 'attribute'
            return v
        elif isinstance(v, list):
            validated = []
            for role in v:
                if role in valid_roles:
                    validated.append(role)
                else:
                    logger.warning(f"Skipping invalid role: {role}")
            return validated if validated else ['attribute']
        return 'attribute'


class DisplayConfig(BaseModel):
    """Display configuration for entities/dimensions"""
    display_name: Optional[str] = None
    default_label_column: Optional[str] = None
    default_search_columns: List[str] = []
    default_sort: Optional[Dict[str, str]] = None
    attribute_order: List[str] = []


class MeasureDefinition(BaseModel):
    """Measure definition for facts"""
    name: str
    expression: str
    unit: Optional[str] = None
    currency: Optional[str] = None
    format_hint: Optional[str] = None
    description: Optional[str] = None


class EntityDefinition(BaseModel):
    """Entity (lookup/reference table)"""
    name: str
    source: str
    primary_key: List[str]
    display: DisplayConfig
    columns: List[ColumnMetadata]


class DimensionDefinition(BaseModel):
    """Dimension table"""
    name: str
    source: str
    keys: List[str]
    attributes: List[ColumnMetadata]
    display: DisplayConfig


class FactDefinition(BaseModel):
    """Fact table"""
    name: str
    source: str
    grain: List[str]
    measures: List[MeasureDefinition]
    foreign_keys: List[Dict[str, str]]
    display: Optional[Dict[str, Any]] = None




class RelationshipDefinition(BaseModel):
    """Relationship between tables"""
    from_: str = Field(..., alias='from')  # Required field with 'from' alias
    to: str
    cardinality: str
    confidence: str
    verification: Optional[Dict[str, Any]] = None
    
    class Config:
        populate_by_name = True  # Allow both 'from_' and 'from' during parsing
        by_alias = True  # Serialize using alias ('from') instead of field name


class TableRanking(BaseModel):
    """Table ranking by quality"""
    table: str
    duplicate_of: Optional[str] = None
    rank: int
    reason: str


class SemanticModel(BaseModel):
    """Complete semantic model"""
    entities: List[EntityDefinition]
    dimensions: List[DimensionDefinition]
    facts: List[FactDefinition]
    relationships: List[RelationshipDefinition]
    table_rankings: List[TableRanking]
    audit: Dict[str, Any]


# ============================================================================
# DISCOVERY DATA COMPRESSOR
# ============================================================================

class DiscoveryCompressor:
    """
    Compresses discovery data for efficient LLM processing
    Implements multiple compression strategies
    """
    
    def __init__(self, strategy: str = "tldr"):
        """
        Args:
            strategy: Compression strategy - 'tldr' (default), 'detailed', 'map_reduce'
        """
        self.strategy = strategy
    
    def compress_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress a single table for LLM consumption
        
        Returns compact representation with:
        - Column names and types
        - Primary/foreign keys
        - Top 5 sample values per column
        - Basic stats (null rate, distinct count)
        """
        if self.strategy == "detailed":
            return table  # Return full data
        
        # TLDR strategy - compact essentials
        compressed = {
            "name": table["name"],
            "type": table["type"],
            "columns": []
        }
        
        # Compress columns
        for col in table.get("columns", []):
            col_compressed = {
                "name": col["name"],
                "type": col["type"],
                "nullable": col.get("nullable", True)
            }
            
            # Add stats if available
            if "stats" in col and col["stats"]:
                stats = col["stats"]
                col_compressed["stats"] = {
                    "null_rate": stats.get("null_rate", 0),
                    "distinct_count": stats.get("distinct_count", 0),
                    "sample_values": stats.get("sample_values", [])[:5]  # Top 5
                }
                
                # Add hints if present
                if "unit_hint" in stats:
                    col_compressed["stats"]["unit_hint"] = stats["unit_hint"]
                if "currency_hint" in stats:
                    col_compressed["stats"]["currency_hint"] = stats["currency_hint"]
            
            compressed["columns"].append(col_compressed)
        
        # Add keys
        if "primary_key" in table:
            compressed["primary_key"] = table["primary_key"]
        if "foreign_keys" in table:
            compressed["foreign_keys"] = table["foreign_keys"]
        
        # Add sample rows (first 3 for context)
        if "sample_rows" in table and table["sample_rows"]:
            compressed["sample_rows"] = table["sample_rows"][:3]
        
        return compressed
    
    def compress_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Compress entire schema"""
        return {
            "name": schema["name"],
            "tables": [self.compress_table(t) for t in schema.get("tables", [])]
        }
    
    def compress_discovery(self, discovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress entire discovery data"""
        return {
            "database": discovery_data.get("database", {}),
            "dialect": discovery_data.get("dialect", "generic"),
            "schemas": [self.compress_schema(s) for s in discovery_data.get("schemas", [])],
            "inferred_relationships": discovery_data.get("inferred_relationships", []),
            "metadata": {
                "total_tables": discovery_data.get("metadata", {}).get("total_tables", 0),
                "total_columns": discovery_data.get("metadata", {}).get("total_columns", 0)
            }
        }


# ============================================================================
# LLM CLIENT WITH RETRY LOGIC
# ============================================================================

class LLMClient:
    """
    LLM client with retry logic and JSON schema validation
    Implements QuadRails constraint checking
    """

    def __init__(self, max_retries: int = 3):
        """
        Args:
            max_retries: Maximum number of retries for failed LLM calls
        """
        from src.llm.client import get_llm_client

        # Use the unified LLM client factory
        self._client = get_llm_client()
        self.max_retries = max_retries
    
    def call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[BaseModel] = None
    ) -> Dict[str, Any]:
        """
        Call LLM with exponential backoff retry on validation failures

        Args:
            system_prompt: System message
            user_prompt: User message
            response_model: Optional Pydantic model for validation

        Returns:
            Parsed JSON response

        Raises:
            ValidationError: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                # Call unified LLM client with JSON response handling
                logger.debug(f"LLM call attempt {attempt + 1}/{self.max_retries}")
                parsed = self._client.invoke_with_json(
                    user_prompt,
                    system_prompt=system_prompt,
                    max_retries=1  # We handle retries here
                )

                # Validate against Pydantic model if provided
                if response_model:
                    validated = response_model(**parsed)
                    return validated.dict()

                return parsed

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"LLM response validation failed (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in LLM call: {e}")
                raise

        raise RuntimeError("Should not reach here")


# ============================================================================
# SEMANTIC MODEL BUILDER
# ============================================================================

class SemanticModelBuilder:
    """
    Main semantic model builder
    Orchestrates LLM-based analysis and model construction

    Now supports audit data integration for:
    - Table ranking based on access patterns (hot/warm/cold)
    - Relationship confidence boosting from join frequency
    - Prioritizing hot tables in classification
    """

    def __init__(self, audit_report: Optional[Any] = None):
        """
        Initialize builder with config and clients

        Args:
            audit_report: Optional AuditReport for enhanced ranking and relationships
        """
        self.settings = get_settings()
        self.llm_config = get_llm_config()
        self.path_config = get_path_config()
        self.discovery_config = get_discovery_config()

        # Initialize components
        self.compressor = DiscoveryCompressor(strategy="tldr")
        self.llm_client = LLMClient(max_retries=3)

        # Batch sizes from config
        self.entity_batch_size = getattr(self.settings, 'entity_batch_size', 2)
        self.dimension_batch_size = getattr(self.settings, 'dimension_batch_size', 2)
        self.fact_batch_size = getattr(self.settings, 'fact_batch_size', 1)

        # Cache
        self.cache_file = self.path_config.cache_dir / 'semantic_model.json'
        self.incremental_cache_file = self.path_config.cache_dir / 'semantic_model_incremental.json'
        self.cache_hours = getattr(self.settings, 'semantic_cache_hours', 168)

        # Audit integration
        self.audit_report = audit_report
        self._audit_metrics_lookup: Dict[str, Any] = {}
        self._join_frequency: Dict[str, int] = {}

        if audit_report:
            self._initialize_audit_lookups()

    def _initialize_audit_lookups(self):
        """Build lookup dictionaries from audit report for fast access"""
        if not self.audit_report:
            return

        # Build table metrics lookup (by lowercase full_name)
        for metric in getattr(self.audit_report, 'table_metrics', []):
            full_name = getattr(metric, 'full_name', '').lower()
            if full_name:
                self._audit_metrics_lookup[full_name] = metric

        # Copy join frequency
        self._join_frequency = getattr(self.audit_report, 'join_frequency', {})

        logger.info(f"Audit integration: {len(self._audit_metrics_lookup)} table metrics, "
                    f"{len(self._join_frequency)} join patterns loaded")

    def _get_table_access_pattern(self, full_name: str) -> Dict[str, Any]:
        """
        Get audit access pattern for a table

        Returns:
            Dict with access_pattern, access_score, is_hot, is_history
        """
        metric = self._audit_metrics_lookup.get(full_name.lower())
        if not metric:
            return {
                'access_pattern': 'unknown',
                'access_score': 50.0,
                'is_hot': False,
                'is_history': False
            }

        return {
            'access_pattern': getattr(metric, 'access_pattern', 'unknown'),
            'access_score': getattr(metric, 'access_score', 50.0),
            'is_hot': getattr(metric, 'access_pattern', '') == 'hot',
            'is_history': getattr(metric, 'is_likely_history', False)
        }

    def _get_join_confidence_boost(self, from_col: str, to_col: str) -> float:
        """
        Get confidence boost based on join frequency from audit

        Args:
            from_col: Source column (schema.table.column)
            to_col: Target column (schema.table.column)

        Returns:
            Confidence adjustment (0.0 to 0.2)
        """
        if not self._join_frequency:
            return 0.0

        # Try both directions
        key1 = f"{from_col}={to_col}"
        key2 = f"{to_col}={from_col}"

        frequency = self._join_frequency.get(key1, 0) or self._join_frequency.get(key2, 0)

        # Also try with just table.column (without schema)
        if frequency == 0:
            short_from = '.'.join(from_col.split('.')[-2:])
            short_to = '.'.join(to_col.split('.')[-2:])
            key3 = "=".join(sorted([short_from, short_to]))

            for join_key, count in self._join_frequency.items():
                if key3 in join_key.lower():
                    frequency = max(frequency, count)
                    break

        # Return boost based on frequency thresholds
        if frequency >= 1000:
            return 0.2  # High frequency: +20%
        elif frequency >= 100:
            return 0.1  # Medium frequency: +10%
        elif frequency > 0:
            return 0.05  # Low frequency: +5%
        return 0.0

    def _get_audit_measure_hints(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get measure hints from audit query patterns

        Analyzes production queries to find which columns are actually
        aggregated (SUM, COUNT, AVG, etc.) - these are verified measures.

        Args:
            table_name: Full table name (schema.table)

        Returns:
            List of measure hints with column, aggregation, and execution count
        """
        if not self.audit_report:
            return []

        query_patterns = getattr(self.audit_report, 'query_patterns', [])
        if not query_patterns:
            return []

        table_lower = table_name.lower()
        measure_hints = {}  # column -> {aggregations: set, execution_count: int}

        for pattern in query_patterns:
            # Check if this pattern references our table
            tables_ref = [t.lower() for t in getattr(pattern, 'tables_referenced', [])]

            # Match by full name or just table name
            table_matches = any(
                table_lower in t or table_lower.split('.')[-1] in t
                for t in tables_ref
            )

            if not table_matches:
                continue

            # Extract aggregations
            aggregations = getattr(pattern, 'aggregations', [])
            exec_count = getattr(pattern, 'execution_count', 0)

            for agg in aggregations:
                # Parse aggregation: "SUM(Amount)", "COUNT(OrderID)", etc.
                import re
                match = re.match(r'(\w+)\s*\(\s*(\w+)\s*\)', agg.strip())
                if match:
                    func_name = match.group(1).upper()
                    col_name = match.group(2)

                    if col_name not in measure_hints:
                        measure_hints[col_name] = {
                            'aggregations': set(),
                            'execution_count': 0
                        }

                    measure_hints[col_name]['aggregations'].add(func_name)
                    measure_hints[col_name]['execution_count'] += exec_count

        # Convert to list sorted by execution count
        result = []
        for col_name, info in measure_hints.items():
            result.append({
                'column': col_name,
                'aggregations': list(info['aggregations']),
                'execution_count': info['execution_count'],
                'verified': True  # These are from actual production usage
            })

        result.sort(key=lambda x: x['execution_count'], reverse=True)

        if result:
            logger.debug(f"    Audit measure hints for {table_name}: {len(result)} columns")

        return result

    def build(self, discovery_data: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
        """
        Build semantic model from discovery data
        
        Args:
            discovery_data: Discovery JSON from Phase 1
            use_cache: Use cached model if valid
            
        Returns:
            Semantic model JSON
        """
        logger.info("=" * 80)
        logger.info("STARTING SEMANTIC MODEL BUILDING")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Check cache
        if use_cache and self._is_cache_valid():
            logger.info("Using cached semantic model")
            return self._load_cache()
        
        logger.info("Building fresh semantic model...")
        
        try:
            # Step 1: Compress discovery data
            logger.info("Step 1: Compressing discovery data...")
            compressed = self.compressor.compress_discovery(discovery_data)

            # Step 2: Classify tables
            logger.info("Step 2: Classifying tables...")
            classification = self._classify_tables(compressed)

            # Save incremental progress after classification
            self._save_incremental({
                "classification": classification,
                "entities": [],
                "dimensions": [],
                "facts": [],
                "relationships": [],
                "table_rankings": []
            }, "classification")

            # Step 3: Build entities
            logger.info("Step 3: Building entities...")
            entities = self._build_entities(compressed, classification)

            # Save incremental progress after entities
            self._save_incremental({
                "classification": classification,
                "entities": entities,
                "dimensions": [],
                "facts": [],
                "relationships": [],
                "table_rankings": []
            }, "entities")

            # Step 4: Build dimensions
            logger.info("Step 4: Building dimensions...")
            dimensions = self._build_dimensions(compressed, classification)

            # Save incremental progress after dimensions
            self._save_incremental({
                "classification": classification,
                "entities": entities,
                "dimensions": dimensions,
                "facts": [],
                "relationships": [],
                "table_rankings": []
            }, "dimensions")

            # Step 5: Build facts
            logger.info("Step 5: Building facts...")
            facts = self._build_facts(compressed, classification)

            # Save incremental progress after facts
            self._save_incremental({
                "classification": classification,
                "entities": entities,
                "dimensions": dimensions,
                "facts": facts,
                "relationships": [],
                "table_rankings": []
            }, "facts")

            # Validate facts have measures
            facts_without_measures = []
            for fact in facts:
                if not fact.get('measures') or len(fact['measures']) == 0:
                    facts_without_measures.append(fact['name'])
                    logger.warning(f"Fact '{fact['name']}' has no measures defined")
                    
                    # Try to auto-detect measures from numeric columns
                    source_table = fact.get('source', '')
                    table_data = self._get_tables_by_names(compressed, [source_table])
                    if table_data:
                        auto_measures = self._auto_detect_measures(table_data[0])
                        if auto_measures:
                            fact['measures'] = auto_measures
                            logger.info(f"Auto-detected {len(auto_measures)} measures for '{fact['name']}'")

            if facts_without_measures and not any(f.get('measures') for f in facts):
                logger.error("No facts have measures! Q&A will fail.")
                raise ValueError("Semantic model invalid: No facts with measures")
            
            # Step 6: Build relationships
            logger.info("Step 6: Building relationships...")
            relationships = self._build_relationships(discovery_data, entities, dimensions, facts)
            
            # Step 7: Rank tables
            logger.info("Step 7: Ranking tables...")
            rankings = self._rank_tables(discovery_data)
            
            # Build audit integration metadata
            audit_metadata = {
                "dialect": discovery_data.get("dialect"),
                "built_at": datetime.utcnow().isoformat(),
                "build_duration_seconds": time.time() - start_time,
                "discovery_timestamp": discovery_data.get("metadata", {}).get("discovered_at")
            }

            # Add production audit info if available
            if self.audit_report:
                audit_metadata["production_audit"] = {
                    "source_server": getattr(self.audit_report, 'source_server', None),
                    "database_name": getattr(self.audit_report, 'database_name', None),
                    "audit_period": {
                        "start": getattr(self.audit_report, 'audit_start_date', None),
                        "end": getattr(self.audit_report, 'audit_end_date', None)
                    },
                    "collected_at": getattr(self.audit_report, 'collected_at', None),
                    "table_metrics_count": len(self._audit_metrics_lookup),
                    "join_patterns_count": len(self._join_frequency),
                    "access_pattern_summary": {
                        "hot": getattr(self.audit_report, 'hot_tables_count', 0),
                        "warm": getattr(self.audit_report, 'warm_tables_count', 0),
                        "cold": getattr(self.audit_report, 'cold_tables_count', 0),
                        "unused": getattr(self.audit_report, 'unused_tables_count', 0)
                    }
                }
                logger.info("  Audit data integrated into semantic model")

            # Assemble model
            semantic_model = {
                "entities": entities,
                "dimensions": dimensions,
                "facts": facts,
                "relationships": relationships,
                "table_rankings": rankings,
                "audit": audit_metadata
            }
            
            # Validate entire model (QuadRails - Constraint)
            logger.info("Validating semantic model...")
            try:
                validated = SemanticModel(**semantic_model)
                semantic_model = validated.dict()
            except ValidationError as e:
                logger.error(f"Semantic model validation failed: {e}")
                raise
            
            # Save to cache
            self._save_cache(semantic_model)
            
            elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"SEMANTIC MODEL COMPLETE in {elapsed:.1f}s")
            logger.info(f"  Entities: {len(entities)}")
            logger.info(f"  Dimensions: {len(dimensions)}")
            logger.info(f"  Facts: {len(facts)}")
            logger.info(f"  Relationships: {len(relationships)}")
            logger.info("=" * 80)
            
            return semantic_model
            
        except Exception as e:
            logger.error(f"Semantic model building failed: {e}", exc_info=True)
            raise
    
    def _classify_tables(self, compressed_discovery: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Classify tables into entities, dimensions, and facts using LLM

        Now with audit integration:
        - Prioritizes hot/warm tables (actively used in production)
        - Filters out unused/history tables from primary classification
        - Includes access pattern hints for LLM context

        Returns:
            Dict with keys 'entities', 'dimensions', 'facts' mapping to table names
        """
        logger.info("  Classifying tables with LLM (audit-guided)...")

        # Build table list with metadata AND audit access patterns
        hot_warm_tables = []
        cold_unused_tables = []
        skipped_tables = []

        for schema in compressed_discovery.get("schemas", []):
            for table in schema.get("tables", []):
                full_name = f"{schema['name']}.{table['name']}"

                # Get audit access pattern
                audit_info = self._get_table_access_pattern(full_name)
                access_pattern = audit_info['access_pattern']
                access_score = audit_info['access_score']
                is_history = audit_info['is_history']

                # Skip history tables entirely
                if is_history:
                    skipped_tables.append(full_name)
                    continue

                table_info = {
                    "full_name": full_name,
                    "name": table["name"],
                    "type": table["type"],
                    "columns": len(table.get("columns", [])),
                    "has_pk": bool(table.get("primary_key")),
                    "has_fk": bool(table.get("foreign_keys")),
                    "sample_columns": [c["name"] for c in table.get("columns", [])[:10]],
                    # Audit info for LLM context
                    "access_pattern": access_pattern,
                    "access_score": round(access_score, 1)
                }

                # Separate hot/warm from cold/unused
                if access_pattern in ('hot', 'warm', 'unknown'):
                    hot_warm_tables.append(table_info)
                else:
                    cold_unused_tables.append(table_info)

        # Log audit filtering stats
        if self.audit_report:
            logger.info(f"    Audit filtering: {len(hot_warm_tables)} hot/warm, "
                       f"{len(cold_unused_tables)} cold/unused, {len(skipped_tables)} history (skipped)")

        # Prioritize hot/warm tables - send these first to LLM
        # Cold/unused tables are secondary and may be excluded from facts
        tables = hot_warm_tables + cold_unused_tables

        # Build audit context for prompt
        audit_context = ""
        if self.audit_report:
            audit_context = """
**PRODUCTION USAGE DATA AVAILABLE**
Tables are marked with access patterns from production audit:
- HOT (score 70-100): Actively queried - HIGH PRIORITY for classification
- WARM (score 30-70): Moderately used - include in model
- COLD (score 10-30): Rarely used - lower priority, may be deprecated
- UNUSED (score 0-10): Never queried - likely deprecated, classify cautiously
- UNKNOWN: No audit data - use schema analysis

**PRIORITIZE hot/warm tables** - these represent actual business usage.
Cold/unused tables should only be classified as facts if they have clear transactional patterns.
"""

        system_prompt = f"""You are a data modeling expert. Classify database tables into three categories:
{audit_context}

1. **ENTITIES**: Lookup/reference tables (customers, products, employees)
   - Typically have: primary key, descriptive columns, low cardinality
   - Few foreign keys, stable data
   - Examples: Customer, Product, Employee, Supplier

2. **DIMENSIONS**: Analysis dimensions for slicing/filtering data
   - **CRITICAL**: Look for these patterns:
     * TIME dimensions: Date, Calendar, Year, Month, Period (have Year/Month/Day columns)
     * GEOGRAPHY dimensions: Location, Region, Country, City (have hierarchical geo columns)
     * CATEGORY dimensions: ProductCategory, Classification (have hierarchical category columns)
   - Hierarchical structure (year -> quarter -> month OR country -> state -> city)
   - Low-to-medium cardinality
   - Used for slicing/filtering/grouping in analytics
   - Column names include: Year, Month, Quarter, Date, Country, Region, Category, Type
   - Examples: DimDate, DimTime, DimGeography, DimProductCategory

3. **FACTS**: Transaction/measurement tables (sales, orders, events)
   - Many foreign keys to entities/dimensions (3+ FK columns)
   - Contain numeric measures (amounts, quantities, counts)
   - High cardinality, many rows
   - Examples: FactSales, Orders, Transactions, Events

**PRIORITIZE** identifying time dimensions (tables with Date/Year/Month columns) as these are critical for temporal analysis.

Respond with valid JSON only:
{{
  "entities": ["schema.table1", "schema.table2"],
  "dimensions": ["schema.table3"],
  "facts": ["schema.table4"]
}}"""
        
        user_prompt = f"""Classify these tables:

Database: {compressed_discovery.get('dialect')}
Total tables: {len(tables)}

Tables:
{json.dumps(tables, indent=2)}

Respond with classification JSON."""
        
        try:
            result = self.llm_client.call_with_retry(system_prompt, user_prompt)
            
            # Validate structure
            entities = result.get("entities", [])
            dimensions = result.get("dimensions", [])
            facts = result.get("facts", [])
            
            logger.info(f"    ✓ Classified: {len(entities)} entities, {len(dimensions)} dimensions, {len(facts)} facts")
            
            return {
                "entities": entities,
                "dimensions": dimensions,
                "facts": facts
            }
            
        except Exception as e:
            logger.error(f"Table classification failed: {e}")
            # Fallback to empty classification
            return {"entities": [], "dimensions": [], "facts": []}
    
    def _build_entities(
        self, 
        compressed_discovery: Dict[str, Any], 
        classification: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Build entity definitions using LLM in batches
        """
        entity_tables = classification.get("entities", [])
        if not entity_tables:
            logger.info("    No entities to build")
            return []
        
        logger.info(f"    Building {len(entity_tables)} entities in batches of {self.entity_batch_size}...")
        
        entities = []
        
        # Process in batches
        for i in range(0, len(entity_tables), self.entity_batch_size):
            batch = entity_tables[i:i + self.entity_batch_size]
            logger.info(f"      Processing entity batch {i // self.entity_batch_size + 1} ({len(batch)} tables)...")
            
            # Get table data for batch
            batch_tables = self._get_tables_by_names(compressed_discovery, batch)
            
            # Build prompt
            system_prompt = """You are a semantic data modeler. Enrich entity tables with business-friendly metadata.

For each entity, provide:
- Meaningful name (e.g., "Customer" not "tbl_cust")
- Column roles: primary_key, label, attribute, foreign_key
- Semantic types: id, person_or_org_name, email, phone, address, etc.
- Aliases for each column (at least 1-2 per column if applicable)
- Descriptions (must not be null)
- Display configuration (ALL fields are REQUIRED - do not use null)

**CRITICAL DISPLAY REQUIREMENTS:**
1. display_name: MUST be a user-friendly version of the table name
2. default_label_column: MUST be the most human-readable column (Name, Description, Title, or Code)
3. default_search_columns: MUST include at least 2-3 searchable text columns
4. default_sort: MUST specify column and direction (asc/desc)

Respond with valid JSON array:
[
  {
    "name": "Customer",
    "source": "dbo.Customer",
    "primary_key": ["CustomerID"],
    "display": {
      "display_name": "Customer",
      "default_label_column": "CustomerName",
      "default_search_columns": ["CustomerName", "Email", "CustomerID"],
      "default_sort": {"column": "CustomerName", "direction": "asc"}
    },
    "columns": [
      {
        "name": "CustomerID",
        "role": "primary_key",
        "semantic_type": "id",
        "aliases": ["CustID", "ID"],
        "description": "Unique customer identifier"
      }
    ]
  }
]

RULES:
- NEVER return null for display fields
- If no obvious label column exists, use the primary key
- Always include at least 2 search columns
- Prefer string/text columns for searching"""
            
            user_prompt = f"""Enrich these entity tables:

{json.dumps(batch_tables, indent=2)}

Respond with entity definitions JSON array."""
            
            try:
                batch_result = self.llm_client.call_with_retry(system_prompt, user_prompt)

                # Validate each entity
                for entity_data in batch_result:
                    try:
                        validated = EntityDefinition(**entity_data)
                        entities.append(validated.dict())
                    except ValidationError as e:
                        logger.warning(f"Entity validation failed: {e}")

                # Save incremental progress after each batch
                self._save_incremental({
                    "entities": entities,
                    "dimensions": [],
                    "facts": [],
                    "relationships": [],
                    "table_rankings": []
                }, f"entities_batch_{i // self.entity_batch_size + 1}")

            except Exception as e:
                logger.error(f"Entity batch processing failed: {e}")
                # Still save what we have so far
                self._save_incremental({
                    "entities": entities,
                    "dimensions": [],
                    "facts": [],
                    "relationships": [],
                    "table_rankings": []
                }, f"entities_batch_{i // self.entity_batch_size + 1}_failed")

        logger.info(f"      ✓ Built {len(entities)} entities")
        return entities

    def _fix_dimension_output(self, dim_data: Dict[str, Any], table_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Post-process LLM dimension output to fix common schema mismatches.

        Fixes:
        - Missing 'source' field
        - Missing 'keys' field
        - String 'default_sort' instead of dict
        - Missing 'name' in attributes
        """
        # Ensure we have a copy to modify
        dim_data = dict(dim_data)

        # Fix missing 'source' field
        if 'source' not in dim_data or not dim_data['source']:
            if table_data and 'full_name' in table_data:
                dim_data['source'] = table_data['full_name']
            elif 'name' in dim_data:
                # Use name as fallback source
                dim_data['source'] = f"dbo.{dim_data['name']}"
            else:
                dim_data['source'] = 'unknown'

        # Fix missing 'keys' field
        if 'keys' not in dim_data or not dim_data['keys']:
            # Try to extract from table_data primary_key
            if table_data and 'primary_key' in table_data:
                dim_data['keys'] = table_data['primary_key']
            elif 'columns' in dim_data:
                # Look for ID/Key columns
                key_cols = [c['name'] for c in dim_data.get('columns', [])
                           if c.get('role') == 'primary_key' or
                           'ID' in c.get('name', '') or 'Key' in c.get('name', '')]
                dim_data['keys'] = key_cols[:1] if key_cols else ['ID']
            elif 'attributes' in dim_data:
                # Look for ID/Key columns in attributes
                key_cols = [a['name'] for a in dim_data.get('attributes', [])
                           if 'ID' in a.get('name', '') or 'Key' in a.get('name', '')]
                dim_data['keys'] = key_cols[:1] if key_cols else ['ID']
            else:
                dim_data['keys'] = ['ID']

        # Fix 'display.default_sort' if it's a string instead of dict
        if 'display' in dim_data and dim_data['display']:
            display = dim_data['display']
            if isinstance(display.get('default_sort'), str):
                sort_col = display['default_sort']
                display['default_sort'] = {'column': sort_col, 'direction': 'asc'}
            elif display.get('default_sort') is None:
                # Provide a default sort
                if display.get('default_label_column'):
                    display['default_sort'] = {'column': display['default_label_column'], 'direction': 'asc'}
                elif dim_data.get('keys'):
                    display['default_sort'] = {'column': dim_data['keys'][0], 'direction': 'asc'}
        else:
            # Create default display config
            label_col = dim_data['keys'][0] if dim_data.get('keys') else 'ID'
            dim_data['display'] = {
                'display_name': dim_data.get('name', 'Dimension'),
                'default_label_column': label_col,
                'default_search_columns': [label_col],
                'default_sort': {'column': label_col, 'direction': 'asc'},
                'attribute_order': []
            }

        # Fix attributes - ensure each has 'name' field
        if 'attributes' in dim_data:
            fixed_attrs = []
            for attr in dim_data['attributes']:
                if isinstance(attr, dict):
                    if 'name' not in attr or not attr['name']:
                        # Skip attributes without names
                        continue
                    # Ensure required fields have defaults
                    attr.setdefault('role', 'attribute')
                    attr.setdefault('aliases', [])
                    attr.setdefault('description', f"Attribute: {attr['name']}")
                    fixed_attrs.append(attr)
                elif isinstance(attr, str):
                    # Convert string to attribute dict
                    fixed_attrs.append({
                        'name': attr,
                        'role': 'attribute',
                        'aliases': [],
                        'description': f"Attribute: {attr}"
                    })
            dim_data['attributes'] = fixed_attrs
        else:
            # Create attributes from table columns if available
            if table_data and 'columns' in table_data:
                dim_data['attributes'] = [
                    {
                        'name': col['name'],
                        'role': 'attribute',
                        'aliases': [],
                        'description': f"Column: {col['name']}"
                    }
                    for col in table_data['columns'][:20]  # Limit to first 20 columns
                ]
            else:
                dim_data['attributes'] = []

        return dim_data

    def _build_dimensions(
        self,
        compressed_discovery: Dict[str, Any],
        classification: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Build dimension definitions using LLM in batches
        """
        dimension_tables = classification.get("dimensions", [])
        if not dimension_tables:
            logger.info("    No dimensions to build")
            dimensions = []
        else:
            logger.info(f"    Building {len(dimension_tables)} dimensions in batches of {self.dimension_batch_size}...")
            
            dimensions = []
            
            for i in range(0, len(dimension_tables), self.dimension_batch_size):
                batch = dimension_tables[i:i + self.dimension_batch_size]
                logger.info(f"      Processing dimension batch {i // self.dimension_batch_size + 1} ({len(batch)} tables)...")
                
                batch_tables = self._get_tables_by_names(compressed_discovery, batch)
                
                system_prompt = """You are a dimensional modeler. Enrich dimension tables with semantic metadata.

For each dimension, provide:
- Meaningful name
- Source table (full name: schema.table)
- Key columns (the primary key columns)
- Attributes with semantic types (date, year, month_name, country, etc.)
- Display configuration (attribute order for drill-down) - REQUIRED, NOT NULL

**CRITICAL SCHEMA REQUIREMENTS:**
1. "source" field is REQUIRED - must be the full table name (schema.table)
2. "keys" field is REQUIRED - must be an array of key column names
3. "attributes" must be an array of objects, each with a "name" field
4. "display.default_sort" MUST be a dictionary with "column" and "direction" keys

Respond with valid JSON array:
[
  {
    "name": "Geography",
    "source": "dbo.DimGeography",
    "keys": ["GeographyKey"],
    "attributes": [
      {
        "name": "Country",
        "role": "attribute",
        "semantic_type": "country",
        "aliases": ["CountryName", "Nation"],
        "description": "Country name"
      },
      {
        "name": "City",
        "role": "attribute",
        "semantic_type": "city",
        "aliases": ["CityName", "Town"],
        "description": "City name"
      }
    ],
    "display": {
      "display_name": "Geography",
      "default_label_column": "Country",
      "default_search_columns": ["Country", "City"],
      "default_sort": {"column": "Country", "direction": "asc"},
      "attribute_order": ["Country", "Region", "City"]
    }
  }
]

RULES:
- NEVER omit source or keys fields
- NEVER return null for display fields
- Each attribute MUST have a "name" field
- default_sort MUST be a dictionary, NOT a string"""

                user_prompt = f"""Enrich these dimension tables:

{json.dumps(batch_tables, indent=2)}

Respond with array of dimension definition JSON objects following the exact schema shown above."""

                try:
                    batch_result = self.llm_client.call_with_retry(system_prompt, user_prompt)

                    for idx, dim_data in enumerate(batch_result):
                        try:
                            # Post-process to fix common LLM output issues
                            dim_data = self._fix_dimension_output(dim_data, batch_tables[idx] if idx < len(batch_tables) else None)
                            validated = DimensionDefinition(**dim_data)
                            dimensions.append(validated.dict())
                        except ValidationError as e:
                            logger.warning(f"Dimension validation failed: {e}")
                            # Log the problematic data for debugging
                            logger.debug(f"Problematic dimension data: {json.dumps(dim_data, indent=2)[:500]}")

                except Exception as e:
                    logger.error(f"Dimension batch processing failed: {e}")
            
            logger.info(f"      ✓ Built {len(dimensions)} dimensions")
        
        # ✅ NEW: Auto-generate time dimensions from fact tables
        logger.info("    Checking for temporal columns to create time dimensions...")
        time_dimensions = self._auto_generate_time_dimensions(compressed_discovery, classification)
        
        if time_dimensions:
            logger.info(f"    ✅ Auto-generated {len(time_dimensions)} time dimension(s)")
            dimensions.extend(time_dimensions)
        else:
            logger.warning("    ⚠️  No temporal columns found - time-based queries will fail!")
        
        return dimensions
    
    def _auto_generate_time_dimensions(
        self,
        compressed_discovery: Dict[str, Any],
        classification: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Auto-generate time dimensions from datetime columns in fact tables
        
        Critical for temporal queries like "last quarter", "this year", etc.
        """
        time_dimensions = []
        fact_tables = classification.get("facts", [])
        
        # Scan fact tables for datetime columns
        datetime_columns = []
        for fact_name in fact_tables:
            fact_data = self._get_tables_by_names(compressed_discovery, [fact_name])
            if not fact_data:
                continue
            
            for col in fact_data[0].get('columns', []):
                col_type = col.get('type', '').lower()
                col_name = col.get('name', '')
                
                # Detect datetime columns
                if any(dt_type in col_type for dt_type in ['date', 'time', 'timestamp']):
                    datetime_columns.append({
                        'source_table': fact_name,
                        'column_name': col_name,
                        'column_type': col['type'],
                        'stats': col.get('stats', {})
                    })
        
        if not datetime_columns:
            return []
        
        # Create virtual time dimension
        # Pick the most common datetime column name (CreatedOn, OrderDate, TransactionDate, etc.)
        from collections import Counter
        col_names = [dc['column_name'] for dc in datetime_columns]
        most_common_col = Counter(col_names).most_common(1)[0][0]
        
        logger.info(f"      Generating time dimension based on column: {most_common_col}")
        
        # Create virtual DimDate dimension
        time_dim = {
            "name": "Date",
            "source": f"VIRTUAL_DIM_{most_common_col}",  # Virtual dimension marker
            "keys": ["Date"],
            "attributes": [
                {
                    "name": "Date",
                    "role": "attribute",
                    "semantic_type": "date",
                    "aliases": ["DateKey", "FullDate"],
                    "description": f"Full date value (derived from {most_common_col})"
                },
                {
                    "name": "Year",
                    "role": "attribute",
                    "semantic_type": "year",
                    "aliases": ["CalendarYear", "Yr"],
                    "description": "Calendar year (e.g., 2024, 2025)"
                },
                {
                    "name": "Quarter",
                    "role": "attribute",
                    "semantic_type": "quarter",
                    "aliases": ["Qtr", "QuarterNumber"],
                    "description": "Calendar quarter (1-4)"
                },
                {
                    "name": "Month",
                    "role": "attribute",
                    "semantic_type": "month_number",
                    "aliases": ["MonthNumber", "Mo"],
                    "description": "Month number (1-12)"
                },
                {
                    "name": "MonthName",
                    "role": "attribute",
                    "semantic_type": "month_name",
                    "aliases": ["Month Name", "MonthText"],
                    "description": "Month name (January, February, etc.)"
                },
                {
                    "name": "Week",
                    "role": "attribute",
                    "semantic_type": "week",
                    "aliases": ["WeekNumber", "ISOWeek"],
                    "description": "Week number (1-53)"
                },
                {
                    "name": "DayOfMonth",
                    "role": "attribute",
                    "semantic_type": "day",
                    "aliases": ["Day", "DayNumber"],
                    "description": "Day of month (1-31)"
                },
                {
                    "name": "DayOfWeek",
                    "role": "attribute",
                    "semantic_type": "day_of_week",
                    "aliases": ["Weekday", "DayName"],
                    "description": "Day name (Monday, Tuesday, etc.)"
                }
            ],
            "display": {
                "display_name": "Date",
                "default_label_column": "Date",
                "default_search_columns": ["Date", "Year", "MonthName"],
                "default_sort": {
                    "column": "Date",
                    "direction": "desc"
                },
                "attribute_order": ["Year", "Quarter", "MonthName", "Week", "DayOfMonth", "DayOfWeek"]
            },
            "metadata": {
                "is_virtual": True,
                "source_columns": datetime_columns,
                "generation_logic": f"Derived from datetime columns in fact tables. SQL generation should use DATEPART/EXTRACT functions on {most_common_col}."
            }
        }
        
        time_dimensions.append(time_dim)
        return time_dimensions
    
    def _build_facts(
        self,
        compressed_discovery: Dict[str, Any],
        classification: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Build fact definitions using LLM in batches
        Facts are processed ONE at a time due to complexity
        """
        fact_tables = classification.get("facts", [])
        if not fact_tables:
            logger.info("    No facts to build")
            return []
        
        logger.info(f"    Building {len(fact_tables)} facts (one at a time)...")
        
        facts = []
        
        for idx, fact_name in enumerate(fact_tables, 1):
            logger.info(f"      Processing fact {idx}/{len(fact_tables)}: {fact_name}")

            fact_tables_data = self._get_tables_by_names(compressed_discovery, [fact_name])
            if not fact_tables_data:
                continue

            fact_table = fact_tables_data[0]

            # Get audit measure hints (verified from production queries)
            audit_measures = self._get_audit_measure_hints(fact_name)

            # Detect potential measure columns automatically
            potential_measures = []
            audit_columns = {m['column'].lower() for m in audit_measures}

            for col in fact_table.get('columns', []):
                col_type = col.get('type', '').lower()
                col_name = col.get('name', '').lower()

                # Numeric columns likely to be measures
                if any(t in col_type for t in ['int', 'decimal', 'numeric', 'float', 'money']):
                    # Skip ID columns
                    if not any(keyword in col_name for keyword in ['id', 'key', 'code']):
                        measure_hint = {
                            'column': col['name'],
                            'type': col['type'],
                            'stats': col.get('stats', {}),
                            # Mark if verified by audit
                            'verified_by_audit': col_name in audit_columns
                        }
                        potential_measures.append(measure_hint)

            # Build user prompt with audit-enhanced measure hints
            user_prompt = f"""Enrich this fact table:

{json.dumps(fact_table, indent=2)}

Respond with fact definition JSON object."""

            # Add audit-verified measures (highest priority)
            if audit_measures:
                audit_context = f"""

**VERIFIED MEASURES FROM PRODUCTION QUERIES:**
These columns are actually aggregated in production - PRIORITIZE these as measures:
{json.dumps(audit_measures, indent=2)}"""
                user_prompt += audit_context

            # Add potential measures (secondary)
            if potential_measures:
                measure_context = f"\n\nPotential measure columns detected:\n{json.dumps(potential_measures, indent=2)}"
                user_prompt += measure_context
            
            
            system_prompt = """You are a fact table modeler. Enrich fact tables with measures and metadata.

For the fact table, provide:
- Meaningful name
- Grain (unique key columns)
- Measures with:
  * Expression (e.g., "SUM(ExtendedAmount)")
  * Unit (currency, count, percentage, etc.)
  * Currency code if applicable
  * Format hint (e.g., "currency(2)", "percentage(1)")
  * Description
- Foreign key references
- Display hints (default breakdowns, filters)

Respond with valid JSON object:
{
  "name": "Sales",
  "source": "dbo.FactSales",
  "grain": ["OrderID", "LineID"],
  "measures": [
    {
      "name": "Revenue",
      "expression": "SUM(ExtendedAmount)",
      "unit": "currency",
      "currency": "USD",
      "format_hint": "currency(2)",
      "description": "Total sales revenue"
    }
  ],
  "foreign_keys": [
    {"column": "CustomerID", "references": "Customer.CustomerID"}
  ],
  "display": {
    "default_breakdowns": ["Customer", "Date"],
    "default_filters": [{"column": "Date.Year", "op": ">=", "value": "2024"}]
  }
}"""
            
            

            
            try:
                fact_result = self.llm_client.call_with_retry(system_prompt, user_prompt)

                try:
                    validated = FactDefinition(**fact_result)
                    facts.append(validated.dict())
                    logger.info(f"        ✓ Fact {idx}/{len(fact_tables)} built: {fact_name}")
                except ValidationError as e:
                    logger.warning(f"Fact validation failed: {e}")

                # Save incremental progress after each fact
                self._save_incremental({
                    "entities": [],  # Will be populated from main build
                    "dimensions": [],
                    "facts": facts,
                    "relationships": [],
                    "table_rankings": []
                }, f"facts_{idx}_of_{len(fact_tables)}")

            except Exception as e:
                logger.error(f"Fact processing failed for {fact_name}: {e}")
                # Still save what we have so far
                self._save_incremental({
                    "entities": [],
                    "dimensions": [],
                    "facts": facts,
                    "relationships": [],
                    "table_rankings": []
                }, f"facts_{idx}_of_{len(fact_tables)}_failed")

        logger.info(f"      ✓ Built {len(facts)} facts")
        return facts
    
    def _discover_relationships_from_audit(
        self,
        entities: List[Dict[str, Any]],
        dimensions: List[Dict[str, Any]],
        facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Discover new relationships from audit join patterns

        High-frequency joins in production queries are strong indicators
        of actual relationships, even if not explicitly defined as FKs.

        Returns:
            List of relationship definitions from audit join patterns
        """
        if not self._join_frequency:
            return []

        logger.info("    Discovering relationships from audit join patterns...")

        # Build source table -> semantic object lookup
        source_to_object = {}
        for obj in entities + dimensions + facts:
            source = obj.get('source', '').lower()
            if source:
                source_to_object[source] = obj

        audit_relationships = []
        processed_pairs = set()

        for join_key, frequency in self._join_frequency.items():
            # Only process high-frequency joins (100+)
            if frequency < 100:
                continue

            # Parse join key: "schema.table.column=schema.table.column"
            if '=' not in join_key:
                continue

            parts = join_key.split('=')
            if len(parts) != 2:
                continue

            from_col, to_col = parts[0].strip(), parts[1].strip()

            # Extract table names
            from_parts = from_col.split('.')
            to_parts = to_col.split('.')

            if len(from_parts) < 2 or len(to_parts) < 2:
                continue

            # Get table portion (handle schema.table.column or table.column)
            if len(from_parts) >= 3:
                from_table = f"{from_parts[0]}.{from_parts[1]}".lower()
                from_column = from_parts[-1]
            else:
                from_table = from_parts[0].lower()
                from_column = from_parts[-1]

            if len(to_parts) >= 3:
                to_table = f"{to_parts[0]}.{to_parts[1]}".lower()
                to_column = to_parts[-1]
            else:
                to_table = to_parts[0].lower()
                to_column = to_parts[-1]

            # Skip self-joins
            if from_table == to_table:
                continue

            # Avoid duplicate relationships (either direction)
            pair_key = tuple(sorted([from_table, to_table]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # Find corresponding semantic objects
            from_obj = source_to_object.get(from_table)
            to_obj = source_to_object.get(to_table)

            if not from_obj or not to_obj:
                continue

            # Determine confidence based on frequency
            if frequency >= 1000:
                confidence = 'very_high'
            elif frequency >= 500:
                confidence = 'high'
            else:
                confidence = 'medium'

            relationship = {
                'from': f"{from_obj['name']}.{from_column}",
                'to': f"{to_obj['name']}.{to_column}",
                'cardinality': 'many_to_one',  # Default assumption
                'confidence': confidence,
                'verification': {
                    'method': 'audit_join_pattern',
                    'execution_count': frequency
                },
                'audit_source': {
                    'join_key': join_key,
                    'frequency': frequency
                }
            }

            audit_relationships.append(relationship)
            logger.debug(f"    Audit relationship: {from_obj['name']} -> {to_obj['name']} "
                        f"(frequency={frequency}, confidence={confidence})")

        logger.info(f"    ✓ Discovered {len(audit_relationships)} relationships from audit joins")
        return audit_relationships

    def _build_relationships(
        self,
        discovery_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        dimensions: List[Dict[str, Any]],
        facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build relationships from discovery data + audit join patterns

        This method:
        1. Creates lookup of all semantic model objects (entities, dimensions, facts)
        2. Discovers relationships from audit join patterns (NEW - high priority)
        3. Filters relationships from discovery data
        4. Maps source table relationships to semantic model object relationships
        5. Applies confidence-based filtering to reduce false positives
        6. Validates that both sides of relationship exist in semantic model

        Returns:
            List of validated relationship definitions with confidence scores
        """
        logger.info("    Building relationships...")

        relationships = []

        # Step 0: Discover relationships from audit join patterns (NEW)
        audit_rels = self._discover_relationships_from_audit(entities, dimensions, facts)
        relationships.extend(audit_rels)

        # Track which object pairs already have relationships (to avoid duplicates)
        existing_pairs = set()
        for rel in audit_rels:
            from_obj = rel['from'].split('.')[0]
            to_obj = rel['to'].split('.')[0]
            existing_pairs.add(tuple(sorted([from_obj, to_obj])))

        # Step 1: Create lookup of all object columns (for validation)
        object_columns = {}

        for entity in entities:
            for pk_col in entity.get('primary_key', []):
                key = f"{entity['name']}.{pk_col}"
                object_columns[key] = {'type': 'entity', 'object': entity['name']}

        for dimension in dimensions:
            for key_col in dimension.get('keys', []):
                key = f"{dimension['name']}.{key_col}"
                object_columns[key] = {'type': 'dimension', 'object': dimension['name']}

        # Step 2: Extract relationships from discovery data
        discovered_rels = discovery_data.get('inferred_relationships', [])
        
        # Step 3: Define confidence-based filtering thresholds
        # This prevents low-quality relationships from polluting the model
        CONFIDENCE_THRESHOLDS = {
            'very_high': 1.0,   # Always include (explicit FKs from schema)
            'high': 1.0,        # Always include (curated assets like views/SPs)
            'medium': 0.5,      # Include 50% (review overlap + cardinality)
            'low': 0.0          # Never include (too risky - likely false positives)
        }
        
        # Step 4: Build semantic domain map for entities
        # Used to reject cross-domain relationships that don't make sense
        entity_domains = {}
        for entity in entities:
            source = entity.get('source', '').lower()
            # Extract semantic domain from table name (e.g., "Payment", "Customer", "Product")
            name_parts = source.split('.')[-1].replace('_', ' ').split()
            entity_domains[entity['name']] = set(name_parts)
        
        filtered_count = 0
        accepted_count = 0
        
        # Step 5: Process each discovered relationship
        for rel in discovered_rels:
            from_col = rel.get('from', '')
            to_col = rel.get('to', '')
            confidence = rel.get('confidence', 'low')
            cardinality = rel.get('cardinality', 'one_to_one')
            overlap_rate = rel.get('overlap_rate', 0.0)
            
            # Filter 5a: Reject based on confidence threshold
            if CONFIDENCE_THRESHOLDS.get(confidence, 0.0) == 0.0:
                filtered_count += 1
                logger.debug(f"  ❌ Filtered {confidence} confidence: {from_col} -> {to_col}")
                continue
            
            # Filter 5b: Reject suspicious one-to-one relationships
            # One-to-one with 99.9%+ overlap usually means duplicate/mirror tables
            # which should NOT be treated as foreign key relationships
            if cardinality == 'one_to_one' and overlap_rate >= 0.999:
                # Extract table names
                from_table_parts = from_col.split('.')[:-1]
                to_table_parts = to_col.split('.')[:-1]
                
                # Check if tables are from same semantic domain
                from_table = from_table_parts[-1] if from_table_parts else ''
                to_table = to_table_parts[-1] if to_table_parts else ''
                
                # Reject if tables don't share any semantic relationship
                # (e.g., "PaymentMethod" and "User" shouldn't have 1:1 with 99.9% overlap)
                if from_table.lower() not in to_table.lower() and \
                to_table.lower() not in from_table.lower():
                    filtered_count += 1
                    logger.debug(f"  ❌ Filtered suspicious 1:1: {from_col} -> {to_col} (overlap={overlap_rate:.3f})")
                    continue
            
            # Step 6: Convert schema.table.column to ObjectName.column
            # This maps database-level relationships to semantic model relationships
            from_parts = from_col.split('.')
            to_parts = to_col.split('.')
            
            # Both must be fully qualified (schema.table.column)
            if len(from_parts) >= 3 and len(to_parts) >= 3:
                from_table = f"{from_parts[0]}.{from_parts[1]}"
                to_table = f"{to_parts[0]}.{to_parts[1]}"
                
                # Find corresponding objects in semantic model
                from_obj = self._find_model_object(from_table, entities, dimensions, facts)
                to_obj = self._find_model_object(to_table, entities, dimensions, facts)
                
                # Filter 6a: Only include if BOTH objects exist in semantic model
                # This ensures we don't create dangling relationships
                if from_obj and to_obj:
                    from_obj_name = from_obj['name']
                    to_obj_name = to_obj['name']
                    
                    # Filter 6b: Semantic domain validation for medium confidence
                    # Medium confidence relationships need extra validation
                    if confidence == 'medium':
                        from_domain = entity_domains.get(from_obj_name, set())
                        to_domain = entity_domains.get(to_obj_name, set())
                        
                        # Require at least some overlap in semantic domain
                        # or high overlap rate (97%+) to proceed
                        if not (from_domain & to_domain) and overlap_rate < 0.97:
                            filtered_count += 1
                            logger.debug(f"  ❌ Filtered cross-domain medium confidence: {from_obj_name} -> {to_obj_name}")
                            continue
                    
                    # Apply audit-based confidence boost from join frequency
                    confidence_boost = self._get_join_confidence_boost(from_col, to_col)
                    boosted_confidence = confidence

                    if confidence_boost > 0:
                        # Upgrade confidence level based on audit join frequency
                        confidence_rank = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
                        rank_to_confidence = {1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}

                        current_rank = confidence_rank.get(confidence, 2)
                        boost_levels = int(confidence_boost / 0.1)  # 0.2 = 2 levels, 0.1 = 1 level
                        new_rank = min(4, current_rank + boost_levels)
                        boosted_confidence = rank_to_confidence.get(new_rank, confidence)

                    # Create validated relationship
                    relationship = {
                        'from': f"{from_obj['name']}.{from_parts[-1]}",
                        'to': f"{to_obj['name']}.{to_parts[-1]}",
                        'cardinality': rel.get('cardinality', 'many_to_one'),
                        'confidence': boosted_confidence,
                        'verification': {
                            'overlap_rate': overlap_rate,
                            'method': rel.get('method', 'inferred')
                        }
                    }

                    # Add audit boost info if applicable
                    if confidence_boost > 0:
                        relationship['audit_boost'] = {
                            'original_confidence': confidence,
                            'boosted_confidence': boosted_confidence,
                            'boost_amount': confidence_boost
                        }

                    relationships.append(relationship)
                    accepted_count += 1
                else:
                    filtered_count += 1
                    logger.debug(f"  ❌ Filtered - objects not in model: {from_col} -> {to_col}")
        
        # Step 7: Log results
        logger.info(f"    ✓ Accepted {accepted_count} relationships, filtered {filtered_count}")
        logger.info(f"    ✓ Built {len(relationships)} validated relationships")
        
        return relationships

    def _find_model_object(
        self,
        source_table: str,
        entities: List[Dict],
        dimensions: List[Dict],
        facts: List[Dict]
    ) -> Optional[Dict]:
        """Find model object by source table name"""
        for obj in entities + dimensions + facts:
            if obj.get('source') == source_table:
                return obj
        return None
    
    def _rank_tables(self, discovery_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank tables by quality with audit data integration

        Priority order (combined asset type + audit access pattern):
        1. Hot views (actively queried curated views)
        2. Hot tables (actively queried raw tables)
        3. Warm views/SPs
        4. Cold views/SPs
        5. Warm tables
        6. Cold tables
        7. Unused/history tables (lowest priority)
        """
        logger.info("    Ranking tables by quality...")

        rankings = []

        # Build index of views and SPs from named_assets
        curated_assets = {}
        for asset in discovery_data.get("named_assets", []):
            asset_name = asset.get("name", "")
            asset_kind = asset.get("kind", "")
            if asset_name:
                curated_assets[asset_name] = asset_kind

        logger.debug(f"      Found {len(curated_assets)} curated assets (views/SPs)")

        # Collect all tables with their types
        all_tables = []

        for schema in discovery_data.get("schemas", []):
            for table in schema.get("tables", []):
                full_name = f"{schema['name']}.{table['name']}"

                # Check multiple sources for table type
                table_type = table.get("type", "table").lower()

                # Override with curated asset type if available
                if full_name in curated_assets:
                    table_type = curated_assets[full_name]
                    logger.debug(f"      Upgraded {full_name} type to: {table_type}")

                all_tables.append({
                    "name": full_name,
                    "type": table_type,
                    "original_type": table.get("type", "table")
                })

        # Base rank by asset type (lower = better)
        asset_rank_map = {
            "view": 1,
            "stored_procedure": 2,
            "storedprocedure": 2,
            "rdl": 3,
            "rdl_dataset": 3,
            "table": 4
        }

        # Access pattern rank modifier (added to base rank)
        # Hot tables get bonus, unused/history get penalty
        access_rank_modifier = {
            "hot": -1,      # Boost hot tables
            "warm": 0,      # No change
            "cold": 1,      # Slight penalty
            "archive": 2,   # Larger penalty
            "unused": 3,    # Large penalty
            "unknown": 0    # No change (no audit data)
        }

        reason_map = {
            "view": "curated view",
            "stored_procedure": "stored procedure",
            "storedprocedure": "stored procedure",
            "rdl": "RDL dataset",
            "rdl_dataset": "RDL dataset",
            "table": "raw table"
        }

        # Track rank distribution for logging
        rank_distribution = {}
        audit_stats = {"hot": 0, "warm": 0, "cold": 0, "unused": 0, "unknown": 0}

        for table in all_tables:
            table_type = table["type"]
            table_name = table["name"]

            # Get base rank from asset type
            base_rank = asset_rank_map.get(table_type, 4)
            reason = reason_map.get(table_type, "raw table")

            # Get audit access pattern
            audit_info = self._get_table_access_pattern(table_name)
            access_pattern = audit_info['access_pattern']
            access_score = audit_info['access_score']
            is_history = audit_info['is_history']

            # Track audit stats
            audit_stats[access_pattern] = audit_stats.get(access_pattern, 0) + 1

            # Calculate final rank
            access_modifier = access_rank_modifier.get(access_pattern, 0)

            # History tables get extra penalty
            if is_history:
                access_modifier += 3

            final_rank = max(1, base_rank + access_modifier)  # Ensure rank >= 1

            # Build reason string with audit info
            if access_pattern != 'unknown':
                reason = f"{reason} ({access_pattern}, score={access_score:.0f})"
                if is_history:
                    reason += " [HISTORY]"

            # Track distribution
            rank_distribution[final_rank] = rank_distribution.get(final_rank, 0) + 1

            rankings.append({
                "table": table_name,
                "duplicate_of": None,
                "rank": final_rank,
                "reason": reason,
                "audit": {
                    "access_pattern": access_pattern,
                    "access_score": access_score,
                    "is_history": is_history
                } if access_pattern != 'unknown' else None
            })

        # Sort by rank (ascending), then by access_score (descending) for ties
        rankings.sort(key=lambda x: (
            x["rank"],
            -(x.get("audit", {}) or {}).get("access_score", 0)
        ))
        
        # Log rank distribution
        logger.info(f"      ✓ Ranked {len(rankings)} tables")
        logger.info(f"      Rank distribution: {rank_distribution}")

        # Log audit integration stats
        if self.audit_report:
            logger.info(f"      Audit patterns: hot={audit_stats.get('hot', 0)}, "
                        f"warm={audit_stats.get('warm', 0)}, cold={audit_stats.get('cold', 0)}, "
                        f"unused={audit_stats.get('unused', 0)}")
        else:
            logger.info("      No audit data - using asset type ranking only")

        if rank_distribution.get(1, 0) == 0:
            logger.warning("      ⚠ No high-priority tables (rank 1) found")

        return rankings
    
    def _get_tables_by_names(
        self, 
        compressed_discovery: Dict[str, Any], 
        table_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract table data for given full table names
        
        Args:
            compressed_discovery: Compressed discovery data
            table_names: List of full table names (schema.table)
            
        Returns:
            List of table data dictionaries
        """
        tables = []
        
        for schema in compressed_discovery.get("schemas", []):
            for table in schema.get("tables", []):
                full_name = f"{schema['name']}.{table['name']}"
                if full_name in table_names:
                    # Add schema context
                    table_with_context = table.copy()
                    table_with_context["schema"] = schema["name"]
                    table_with_context["full_name"] = full_name
                    tables.append(table_with_context)
        
        return tables
    
    def _is_cache_valid(self) -> bool:
        """Check if semantic model cache is valid"""
        if not self.cache_file.exists():
            return False
        
        # Check age
        from datetime import timedelta
        cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
        if cache_age > timedelta(hours=self.cache_hours):
            logger.info(f"Semantic cache expired (age: {cache_age})")
            return False
        
        return True
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load semantic model from cache"""
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_cache(self, semantic_model: Dict[str, Any]):
        """Save semantic model to cache"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_model, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Saved semantic model to {self.cache_file}")

    def _save_incremental(self, partial_model: Dict[str, Any], phase: str):
        """
        Save partial semantic model incrementally as batches complete.
        This ensures progress is not lost if the build fails partway through.

        Args:
            partial_model: Current state of the semantic model
            phase: Current build phase (classification, entities, dimensions, facts, etc.)
        """
        try:
            self.incremental_cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata about the incremental save
            incremental_data = {
                "phase": phase,
                "saved_at": datetime.utcnow().isoformat(),
                "is_complete": False,
                **partial_model
            }

            with open(self.incremental_cache_file, 'w', encoding='utf-8') as f:
                json.dump(incremental_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"  Incremental save: {phase} -> {self.incremental_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save incremental cache: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_semantic_model(
    discovery_data: Optional[Dict[str, Any]] = None,
    audit_report: Optional[Any] = None,
    use_cache: bool = True,
    use_audit: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to build semantic model with optional audit integration

    Args:
        discovery_data: Discovery JSON (loads from cache if not provided)
        audit_report: AuditReport for enhanced ranking/relationships (auto-loads if None and use_audit=True)
        use_cache: Use cached semantic model if valid
        use_audit: Attempt to load audit data if audit_report not provided

    Returns:
        Semantic model JSON
    """
    # Load discovery data if not provided
    if discovery_data is None:
        path_config = get_path_config()
        discovery_file = path_config.cache_dir / 'discovery.json'

        if not discovery_file.exists():
            raise FileNotFoundError(
                "Discovery data not found. Run discovery first: python main.py discovery"
            )

        with open(discovery_file, 'r', encoding='utf-8') as f:
            discovery_data = json.load(f)

    # Auto-load audit report if requested and not provided
    if audit_report is None and use_audit:
        try:
            from src.discovery.audit_integration import AuditEnhancedDiscovery
            enhanced = AuditEnhancedDiscovery()
            audit_report = enhanced.load_audit_with_mapping()
            if audit_report:
                logger.info("Auto-loaded audit report for semantic model building")
        except Exception as e:
            logger.debug(f"Could not load audit report: {e}")
            audit_report = None

    # Build model with audit integration
    builder = SemanticModelBuilder(audit_report=audit_report)
    return builder.build(discovery_data, use_cache=use_cache)


def clear_semantic_cache():
    """Clear semantic model cache"""
    path_config = get_path_config()
    cache_file = path_config.cache_dir / 'semantic_model.json'
    
    if cache_file.exists():
        cache_file.unlink()
        logger.info(f"Cleared semantic model cache: {cache_file}")
    else:
        logger.info("No semantic model cache to clear")


def _auto_detect_measures(self, table_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Auto-detect measures from numeric columns"""
    measures = []
    for col in table_data.get('columns', []):
        col_type = col.get('type', '').lower()
        col_name = col['name']
        
        if any(t in col_type for t in ['int', 'decimal', 'numeric', 'float', 'money']):
            if not any(kw in col_name.lower() for kw in ['id', 'key', 'code']):
                measures.append({
                    'name': col_name,
                    'expression': f"SUM({col_name})",
                    'unit': 'numeric',
                    'description': f"Auto-detected measure from {col_name}"
                })
    return measures[:5]  # Limit to top 5 auto-detected measures