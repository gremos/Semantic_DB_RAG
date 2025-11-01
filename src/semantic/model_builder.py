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

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

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
    role: Union[str, List[str]]  # Can be single role or multiple roles
    semantic_type: Optional[str] = None
    aliases: List[str] = []
    description: Optional[str] = None


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
    from_field: str = None  # Rename to avoid Python keyword
    to: str
    cardinality: str
    confidence: str
    verification: Optional[Dict[str, Any]] = None
    
    class Config:
        fields = {'from_field': 'from'}


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
        llm_config = get_llm_config()
        
        # Initialize Azure OpenAI client
        # NOTE: gpt-5-mini doesn't support temperature parameter
        self.llm = AzureChatOpenAI(
            deployment_name=llm_config.deployment_name,
            api_version=llm_config.api_version,
            azure_endpoint=llm_config.endpoint,
            api_key=llm_config.api_key,
            # temperature NOT supported by gpt-5-mini
        )
        
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
                # Build messages
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                # Call LLM
                logger.debug(f"LLM call attempt {attempt + 1}/{self.max_retries}")
                response = self.llm.invoke(messages)
                
                # Parse JSON
                content = response.content.strip()
                
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content.split("```json")[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                
                content = content.strip()
                parsed = json.loads(content)
                
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
    """
    
    def __init__(self):
        """Initialize builder with config and clients"""
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
        self.cache_hours = getattr(self.settings, 'semantic_cache_hours', 168)
    
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
            
            # Step 3: Build entities
            logger.info("Step 3: Building entities...")
            entities = self._build_entities(compressed, classification)
            
            # Step 4: Build dimensions
            logger.info("Step 4: Building dimensions...")
            dimensions = self._build_dimensions(compressed, classification)
            
            # Step 5: Build facts
            logger.info("Step 5: Building facts...")
            facts = self._build_facts(compressed, classification)
            
            # Step 6: Build relationships
            logger.info("Step 6: Building relationships...")
            relationships = self._build_relationships(discovery_data, entities, dimensions, facts)
            
            # Step 7: Rank tables
            logger.info("Step 7: Ranking tables...")
            rankings = self._rank_tables(discovery_data)
            
            # Assemble model
            semantic_model = {
                "entities": entities,
                "dimensions": dimensions,
                "facts": facts,
                "relationships": relationships,
                "table_rankings": rankings,
                "audit": {
                    "dialect": discovery_data.get("dialect"),
                    "built_at": datetime.utcnow().isoformat(),
                    "build_duration_seconds": time.time() - start_time,
                    "discovery_timestamp": discovery_data.get("metadata", {}).get("discovered_at")
                }
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
        
        Returns:
            Dict with keys 'entities', 'dimensions', 'facts' mapping to table names
        """
        logger.info("  Classifying tables with LLM...")
        
        # Build table list with metadata
        tables = []
        for schema in compressed_discovery.get("schemas", []):
            for table in schema.get("tables", []):
                tables.append({
                    "full_name": f"{schema['name']}.{table['name']}",
                    "name": table["name"],
                    "type": table["type"],
                    "columns": len(table.get("columns", [])),
                    "has_pk": bool(table.get("primary_key")),
                    "has_fk": bool(table.get("foreign_keys")),
                    "sample_columns": [c["name"] for c in table.get("columns", [])[:10]]
                })
        
        system_prompt = """You are a data modeling expert. Classify database tables into three categories:

1. **ENTITIES**: Lookup/reference tables (customers, products, employees)
   - Typically have: primary key, descriptive columns, low cardinality
   - Few foreign keys, stable data

2. **DIMENSIONS**: Analysis dimensions (date, geography, categories)
   - Hierarchical structure (year -> quarter -> month)
   - Used for slicing/filtering data

3. **FACTS**: Transaction/measurement tables (sales, orders, events)
   - Many foreign keys to entities/dimensions
   - Contain numeric measures
   - High cardinality, many rows

Respond with valid JSON only:
{
  "entities": ["schema.table1", "schema.table2"],
  "dimensions": ["schema.table3"],
  "facts": ["schema.table4"]
}"""
        
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
- Aliases for each column
- Descriptions
- Display configuration (label column, search columns, sort order)

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
]"""
            
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
                
            except Exception as e:
                logger.error(f"Entity batch processing failed: {e}")
        
        logger.info(f"      ✓ Built {len(entities)} entities")
        return entities
    
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
            return []
        
        logger.info(f"    Building {len(dimension_tables)} dimensions in batches of {self.dimension_batch_size}...")
        
        dimensions = []
        
        for i in range(0, len(dimension_tables), self.dimension_batch_size):
            batch = dimension_tables[i:i + self.dimension_batch_size]
            logger.info(f"      Processing dimension batch {i // self.dimension_batch_size + 1} ({len(batch)} tables)...")
            
            batch_tables = self._get_tables_by_names(compressed_discovery, batch)
            
            system_prompt = """You are a dimensional modeler. Enrich dimension tables with semantic metadata.

For each dimension, provide:
- Meaningful name
- Key columns
- Attributes with semantic types (date, year, month_name, country, etc.)
- Display configuration (attribute order for drill-down)

Respond with valid JSON array:
[
  {
    "name": "Date",
    "source": "dbo.DimDate",
    "keys": ["DateKey"],
    "attributes": [
      {"name": "Date", "semantic_type": "date", "role": "attribute"},
      {"name": "Year", "semantic_type": "year", "role": "attribute"},
      {"name": "Month", "semantic_type": "month_name", "role": "attribute"}
    ],
    "display": {
      "attribute_order": ["Year", "Month", "Date"]
    }
  }
]"""
            
            user_prompt = f"""Enrich these dimension tables:

{json.dumps(batch_tables, indent=2)}

Respond with dimension definitions JSON array."""
            
            try:
                batch_result = self.llm_client.call_with_retry(system_prompt, user_prompt)
                
                for dim_data in batch_result:
                    try:
                        validated = DimensionDefinition(**dim_data)
                        dimensions.append(validated.dict())
                    except ValidationError as e:
                        logger.warning(f"Dimension validation failed: {e}")
            
            except Exception as e:
                logger.error(f"Dimension batch processing failed: {e}")
        
        logger.info(f"      ✓ Built {len(dimensions)} dimensions")
        return dimensions
    
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
            
            user_prompt = f"""Enrich this fact table:

{json.dumps(fact_table, indent=2)}

Respond with fact definition JSON object."""
            
            try:
                fact_result = self.llm_client.call_with_retry(system_prompt, user_prompt)
                
                try:
                    validated = FactDefinition(**fact_result)
                    facts.append(validated.dict())
                except ValidationError as e:
                    logger.warning(f"Fact validation failed: {e}")
            
            except Exception as e:
                logger.error(f"Fact processing failed for {fact_name}: {e}")
        
        logger.info(f"      ✓ Built {len(facts)} facts")
        return facts
    
    def _build_relationships(
        self,
        discovery_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        dimensions: List[Dict[str, Any]],
        facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build semantic relationships from discovery relationships
        Maps to semantic model object names
        """
        logger.info("    Mapping relationships to semantic model...")
        
        inferred_rels = discovery_data.get("inferred_relationships", [])
        
        # Build mapping from source table to semantic name
        source_to_semantic = {}
        
        for entity in entities:
            source_to_semantic[entity["source"]] = entity["name"]
        for dimension in dimensions:
            source_to_semantic[dimension["source"]] = dimension["name"]
        for fact in facts:
            source_to_semantic[fact["source"]] = fact["name"]
        
        relationships = []
        
        for rel in inferred_rels:
            try:
                # Parse from/to
                from_parts = rel["from"].split(".")
                to_parts = rel["to"].split(".")
                
                if len(from_parts) >= 2 and len(to_parts) >= 2:
                    from_table = f"{from_parts[0]}.{from_parts[1]}"
                    to_table = f"{to_parts[0]}.{to_parts[1]}"
                    
                    # Map to semantic names
                    from_semantic = source_to_semantic.get(from_table)
                    to_semantic = source_to_semantic.get(to_table)
                    
                    if from_semantic and to_semantic:
                        # Build semantic relationship
                        rel_obj = {
                            "from": f"{from_semantic}.{from_parts[2]}",
                            "to": f"{to_semantic}.{to_parts[2]}",
                            "cardinality": rel.get("cardinality", "many_to_one"),
                            "confidence": rel.get("confidence", "medium")
                        }
                        
                        # Add verification if present
                        if "overlap_rate" in rel:
                            rel_obj["verification"] = {
                                "overlap_rate": rel["overlap_rate"]
                            }
                        
                        relationships.append(rel_obj)
            
            except Exception as e:
                logger.debug(f"Could not map relationship: {rel} - {e}")
        
        logger.info(f"      ✓ Mapped {len(relationships)} relationships")
        return relationships
    
    def _rank_tables(self, discovery_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank tables by quality
        Priority: views > stored procedures > RDL datasets > raw tables
        """
        logger.info("    Ranking tables by quality...")
        
        rankings = []
        
        # Collect all tables with their types
        all_tables = []
        
        for schema in discovery_data.get("schemas", []):
            for table in schema.get("tables", []):
                full_name = f"{schema['name']}.{table['name']}"
                all_tables.append({
                    "name": full_name,
                    "type": table["type"]
                })
        
        # Assign ranks
        rank_map = {
            "view": 1,
            "stored_procedure": 2,
            "rdl_dataset": 3,
            "table": 4
        }
        
        reason_map = {
            "view": "curated view",
            "stored_procedure": "stored procedure",
            "rdl_dataset": "RDL dataset",
            "table": "raw table"
        }
        
        for table in all_tables:
            table_type = table["type"]
            rank = rank_map.get(table_type, 4)
            reason = reason_map.get(table_type, "raw table")
            
            rankings.append({
                "table": table["name"],
                "duplicate_of": None,  # TODO: Detect duplicates
                "rank": rank,
                "reason": reason
            })
        
        # Sort by rank
        rankings.sort(key=lambda x: x["rank"])
        
        logger.info(f"      ✓ Ranked {len(rankings)} tables")
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


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_semantic_model(
    discovery_data: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to build semantic model
    
    Args:
        discovery_data: Discovery JSON (loads from cache if not provided)
        use_cache: Use cached semantic model if valid
        
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
    
    # Build model
    builder = SemanticModelBuilder()
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