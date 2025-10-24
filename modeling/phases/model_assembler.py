from typing import Dict, Any, List, Tuple
from llm.azure_client import AzureLLMClient
from utils.json_extractor import JSONExtractor
import json
import logging

logger = logging.getLogger(__name__)

class ModelAssembler:
    """Phase 5: Hierarchical incremental assembly - small focused LLM calls."""
    
    # Batch sizes to stay under token limits
    ENTITY_BATCH_SIZE = 10
    DIMENSION_BATCH_SIZE = 10
    FACT_BATCH_SIZE = 5  # Facts are most complex
    
    def __init__(self, llm_client: AzureLLMClient):
        self.llm = llm_client
        self.json_extractor = JSONExtractor()
        self._load_prompts()
    
    def _load_prompts(self):
        """Load specialized prompts for each assembly phase."""
        # Entity assembly prompt (lightweight)
        self.entity_prompt = """You are assembling ENTITY records for a semantic model.

Input: Table classifications and column info for ENTITY tables.
Output: JSON array of entity objects.

Each entity MUST have:
- name: Business name (e.g., "Customer")
- source: Full table name (e.g., "dbo.Customers")
- primary_key: Array of PK columns
- description: One sentence explaining this entity
- columns: Array with name, type, semantic_role, description

Return ONLY JSON array: [{"name": "...", "source": "...", ...}]"""

        # Dimension assembly prompt
        self.dimension_prompt = """You are assembling DIMENSION records for a semantic model.

Input: Table classifications and column info for DIMENSION tables.
Output: JSON array of dimension objects.

Each dimension MUST have:
- name: Business name
- source: Full table name
- keys: Array of key columns
- attributes: Array of descriptive columns
- columns: Array with metadata

Return ONLY JSON array: [{"name": "...", "source": "...", ...}]"""

        # Fact assembly prompt (most complex)
        self.fact_prompt = """You are assembling FACT records for a semantic model.

Input: 
- Table classification (fact)
- Columns with semantic roles
- Pre-identified measures
- Status columns with filter logic

Output: JSON array of fact objects.

Each fact MUST have:
- name: Business name
- source: Full table name
- grain: Array defining granularity
- columns: Array with metadata
- measures: Array from pre-identified measures
- foreign_keys: Array of FK references

CRITICAL: For measures, ADD filters_applied from status columns:
- If measure depends on columns with status_indicator role
- Look up the active_filter from status column metadata
- Add to filters_applied array

Return ONLY JSON array: [{"name": "...", "source": "...", ...}]"""

        # Metrics generation prompt
        self.metrics_prompt = """You are generating business METRICS from fact measures.

Input: All assembled facts with their measures.
Output: JSON array of metrics.

Generate 3-5 meaningful metrics that combine measures or add business logic.

Each metric MUST have:
- name: KPI name (e.g., "Total Active Revenue")
- purpose: What it measures
- logic: How to calculate it
- inputs: Which fact.measure combinations
- constraints: Filters to apply
- explain: Detailed explanation

Return ONLY JSON array: [{"name": "...", "purpose": "...", ...}]"""

    def assemble_model(
        self,
        classifications: Dict[str, Dict[str, Any]],
        measures: Dict[str, List[Dict[str, Any]]],
        status_columns: Dict[str, Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Hierarchical incremental assembly - multiple small LLM calls.
        
        Returns:
            (success, semantic_model, error)
        """
        try:
            logger.info("Starting hierarchical incremental assembly...")
            
            # Separate classifications by type
            entities_cls = {k: v for k, v in classifications.items() if v["classification"] == "entity"}
            dimensions_cls = {k: v for k, v in classifications.items() if v["classification"] == "dimension"}
            facts_cls = {k: v for k, v in classifications.items() if v["classification"] == "fact"}
            
            logger.info(f"  {len(entities_cls)} entities, {len(dimensions_cls)} dimensions, {len(facts_cls)} facts")
            
            # Phase 5a: Assemble entities in batches
            logger.info("Phase 5a: Assembling entities...")
            entities = self._assemble_entities_batched(entities_cls, compressed_discovery)
            logger.info(f"  ✓ Assembled {len(entities)} entities")
            
            # Phase 5b: Assemble dimensions in batches
            logger.info("Phase 5b: Assembling dimensions...")
            dimensions = self._assemble_dimensions_batched(dimensions_cls, compressed_discovery)
            logger.info(f"  ✓ Assembled {len(dimensions)} dimensions")
            
            # Phase 5c: Assemble facts in batches (most complex)
            logger.info("Phase 5c: Assembling facts...")
            facts = self._assemble_facts_batched(facts_cls, measures, status_columns, compressed_discovery)
            logger.info(f"  ✓ Assembled {len(facts)} facts")
            
            # Phase 5d: Generate metrics from facts
            logger.info("Phase 5d: Generating metrics...")
            metrics = self._generate_metrics(facts)
            logger.info(f"  ✓ Generated {len(metrics)} metrics")
            
            # Phase 5e: Build final model
            logger.info("Phase 5e: Building final model...")
            model = {
                "entities": entities,
                "dimensions": dimensions,
                "facts": facts,
                "relationships": relationships,
                "metrics": metrics,
                "audit": {
                    "dialect": compressed_discovery.get("dialect", "tsql"),
                    "source_assets_used": self._build_source_assets(entities, dimensions, facts),
                    "assumptions": [
                        "Generated via hierarchical incremental assembly",
                        f"Processed {len(classifications)} tables in batches"
                    ]
                }
            }
            
            logger.info(f"✓ Model assembly complete: {len(entities)} entities, {len(dimensions)} dimensions, {len(facts)} facts, {len(metrics)} metrics")
            return (True, model, "")
        
        except Exception as e:
            logger.error(f"Error in hierarchical assembly: {e}", exc_info=True)
            return (False, {}, str(e))
    
    def _assemble_entities_batched(
        self,
        entities_cls: Dict[str, Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assemble entities in small batches."""
        all_entities = []
        entity_names = list(entities_cls.keys())
        
        for i in range(0, len(entity_names), self.ENTITY_BATCH_SIZE):
            batch = entity_names[i:i + self.ENTITY_BATCH_SIZE]
            logger.info(f"  Processing entity batch {i//self.ENTITY_BATCH_SIZE + 1} ({len(batch)} tables)...")
            
            # Build compact batch context
            batch_context = self._build_entity_batch_context(batch, compressed_discovery)
            
            # Call LLM
            try:
                user_prompt = f"""Process these {len(batch)} ENTITY tables:

{json.dumps(batch_context, indent=2)}

Return JSON array of entities."""
                
                response = self.llm.generate(self.entity_prompt, user_prompt)
                result, method = self.json_extractor.extract(response, log_failures=False)
                
                if result and isinstance(result, list):
                    all_entities.extend(result)
                    logger.info(f"    ✓ Assembled {len(result)} entities")
                elif result and isinstance(result, dict) and "entities" in result:
                    all_entities.extend(result["entities"])
                    logger.info(f"    ✓ Assembled {len(result['entities'])} entities")
                else:
                    # Fallback for this batch
                    logger.warning(f"    ⚠ LLM failed, using fallback for batch")
                    all_entities.extend(self._fallback_entities(batch, compressed_discovery))
            
            except Exception as e:
                logger.error(f"    ✗ Batch failed: {e}")
                all_entities.extend(self._fallback_entities(batch, compressed_discovery))
        
        return all_entities
    
    def _assemble_dimensions_batched(
        self,
        dimensions_cls: Dict[str, Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assemble dimensions in small batches."""
        all_dimensions = []
        dimension_names = list(dimensions_cls.keys())
        
        for i in range(0, len(dimension_names), self.DIMENSION_BATCH_SIZE):
            batch = dimension_names[i:i + self.DIMENSION_BATCH_SIZE]
            logger.info(f"  Processing dimension batch {i//self.DIMENSION_BATCH_SIZE + 1} ({len(batch)} tables)...")
            
            batch_context = self._build_dimension_batch_context(batch, compressed_discovery)
            
            try:
                user_prompt = f"""Process these {len(batch)} DIMENSION tables:

{json.dumps(batch_context, indent=2)}

Return JSON array of dimensions."""
                
                response = self.llm.generate(self.dimension_prompt, user_prompt)
                result, method = self.json_extractor.extract(response, log_failures=False)
                
                if result and isinstance(result, list):
                    all_dimensions.extend(result)
                    logger.info(f"    ✓ Assembled {len(result)} dimensions")
                elif result and isinstance(result, dict) and "dimensions" in result:
                    all_dimensions.extend(result["dimensions"])
                    logger.info(f"    ✓ Assembled {len(result['dimensions'])} dimensions")
                else:
                    logger.warning(f"    ⚠ LLM failed, using fallback for batch")
                    all_dimensions.extend(self._fallback_dimensions(batch, compressed_discovery))
            
            except Exception as e:
                logger.error(f"    ✗ Batch failed: {e}")
                all_dimensions.extend(self._fallback_dimensions(batch, compressed_discovery))
        
        return all_dimensions
    
    def _assemble_facts_batched(
        self,
        facts_cls: Dict[str, Dict[str, Any]],
        measures: Dict[str, List[Dict[str, Any]]],
        status_columns: Dict[str, Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assemble facts in small batches (most complex)."""
        all_facts = []
        fact_names = list(facts_cls.keys())
        
        for i in range(0, len(fact_names), self.FACT_BATCH_SIZE):
            batch = fact_names[i:i + self.FACT_BATCH_SIZE]
            logger.info(f"  Processing fact batch {i//self.FACT_BATCH_SIZE + 1} ({len(batch)} tables)...")
            
            batch_context = self._build_fact_batch_context(
                batch, 
                measures, 
                status_columns, 
                compressed_discovery
            )
            
            try:
                user_prompt = f"""Process these {len(batch)} FACT tables:

{json.dumps(batch_context, indent=2)}

IMPORTANT: For each measure, check if it depends on status indicator columns.
If yes, add the active_filter from status column metadata to filters_applied.

Return JSON array of facts."""
                
                response = self.llm.generate(self.fact_prompt, user_prompt)
                result, method = self.json_extractor.extract(response, log_failures=False)
                
                if result and isinstance(result, list):
                    all_facts.extend(result)
                    logger.info(f"    ✓ Assembled {len(result)} facts")
                elif result and isinstance(result, dict) and "facts" in result:
                    all_facts.extend(result["facts"])
                    logger.info(f"    ✓ Assembled {len(result['facts'])} facts")
                else:
                    logger.warning(f"    ⚠ LLM failed, using fallback for batch")
                    all_facts.extend(self._fallback_facts(batch, measures, status_columns, compressed_discovery))
            
            except Exception as e:
                logger.error(f"    ✗ Batch failed: {e}")
                all_facts.extend(self._fallback_facts(batch, measures, status_columns, compressed_discovery))
        
        return all_facts
    
    def _generate_metrics(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate business metrics from assembled facts."""
        if not facts:
            return []
        
        # Build compact fact summary for metrics generation
        fact_summary = []
        for fact in facts[:10]:  # Limit to first 10 facts to avoid token limit
            measure_names = [m["name"] for m in fact.get("measures", [])]
            fact_summary.append({
                "name": fact["name"],
                "source": fact["source"],
                "measures": measure_names
            })
        
        try:
            user_prompt = f"""Generate business metrics from these facts:

{json.dumps(fact_summary, indent=2)}

Create 3-5 meaningful KPIs that combine measures or add business logic.

Return JSON array of metrics."""
            
            response = self.llm.generate(self.metrics_prompt, user_prompt)
            result, method = self.json_extractor.extract(response, log_failures=False)
            
            if result and isinstance(result, list):
                return result[:5]  # Max 5 metrics
            elif result and isinstance(result, dict) and "metrics" in result:
                return result["metrics"][:5]
            else:
                # Fallback: Generate basic metrics
                return self._fallback_metrics(facts)
        
        except Exception as e:
            logger.error(f"Metrics generation failed: {e}")
            return self._fallback_metrics(facts)
    
    # ==================== CONTEXT BUILDERS ====================
    
    def _build_entity_batch_context(
        self,
        entity_names: List[str],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build compact context for entity batch."""
        batch_context = []
        
        for table_name in entity_names:
            table_data = compressed_discovery["tables"].get(table_name, {})
            
            # Only include essential info
            context = {
                "table_name": table_name,
                "pk": table_data.get("pk", []),
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "semantic_role": c.get("semantic_role"),
                        "is_pk": c.get("is_pk", False)
                    }
                    for c in table_data.get("columns", [])[:15]  # Limit to 15 columns
                ]
            }
            
            batch_context.append(context)
        
        return batch_context
    
    def _build_dimension_batch_context(
        self,
        dimension_names: List[str],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build compact context for dimension batch."""
        batch_context = []
        
        for table_name in dimension_names:
            table_data = compressed_discovery["tables"].get(table_name, {})
            
            # Identify key vs attribute columns
            pk_cols = set(table_data.get("pk", []))
            
            context = {
                "table_name": table_name,
                "keys": list(pk_cols),
                "attributes": [
                    c["name"] for c in table_data.get("columns", [])
                    if c["name"] not in pk_cols
                ][:10],  # Limit attributes
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "semantic_role": c.get("semantic_role")
                    }
                    for c in table_data.get("columns", [])[:15]
                ]
            }
            
            batch_context.append(context)
        
        return batch_context
    
    def _build_fact_batch_context(
        self,
        fact_names: List[str],
        measures: Dict[str, List[Dict[str, Any]]],
        status_columns: Dict[str, Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build compact context for fact batch."""
        batch_context = []
        
        for table_name in fact_names:
            table_data = compressed_discovery["tables"].get(table_name, {})
            
            # Get measures for this fact
            fact_measures = measures.get(table_name, [])
            
            # Get status columns for this fact
            fact_status = {}
            for col_key, col_status in status_columns.items():
                if col_key.startswith(f"{table_name}."):
                    col_name = col_key.split('.')[-1]
                    fact_status[col_name] = {
                        "active_filter": col_status.get("active_filter", ""),
                        "description": col_status.get("description", "")
                    }
            
            context = {
                "table_name": table_name,
                "grain": table_data.get("pk", []),
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "semantic_role": c.get("semantic_role"),
                        "is_fk": c.get("is_fk", False)
                    }
                    for c in table_data.get("columns", [])[:20]  # Facts may have more columns
                ],
                "measures": fact_measures,
                "status_columns": fact_status,
                "fks": table_data.get("fks", [])
            }
            
            batch_context.append(context)
        
        return batch_context
    
    # ==================== FALLBACK BUILDERS ====================
    
    def _fallback_entities(
        self,
        entity_names: List[str],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback entity builder."""
        entities = []
        
        for table_name in entity_names:
            table_data = compressed_discovery["tables"].get(table_name, {})
            
            entities.append({
                "name": table_name.split('.')[-1],
                "source": table_name,
                "primary_key": table_data.get("pk", ["ID"]),
                "description": f"Entity: {table_name}",
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "nullable": c.get("nullable", True),
                        "semantic_role": c.get("semantic_role", "metadata"),
                        "description": c["name"]
                    }
                    for c in table_data.get("columns", [])
                ]
            })
        
        return entities
    
    def _fallback_dimensions(
        self,
        dimension_names: List[str],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback dimension builder."""
        dimensions = []
        
        for table_name in dimension_names:
            table_data = compressed_discovery["tables"].get(table_name, {})
            pk_cols = set(table_data.get("pk", []))
            
            dimensions.append({
                "name": table_name.split('.')[-1],
                "source": table_name,
                "keys": list(pk_cols) if pk_cols else ["ID"],
                "attributes": [c["name"] for c in table_data.get("columns", []) if c["name"] not in pk_cols],
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "nullable": c.get("nullable", True),
                        "semantic_role": c.get("semantic_role", "dimension_attribute"),
                        "description": c["name"]
                    }
                    for c in table_data.get("columns", [])
                ]
            })
        
        return dimensions
    
    def _fallback_facts(
        self,
        fact_names: List[str],
        measures: Dict[str, List[Dict[str, Any]]],
        status_columns: Dict[str, Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback fact builder with status column integration."""
        facts = []
        
        for table_name in fact_names:
            table_data = compressed_discovery["tables"].get(table_name, {})
            fact_measures = measures.get(table_name, [])
            
            # Add filters to measures
            for measure in fact_measures:
                filters = []
                for col_name in measure.get("depends_on", []):
                    col_key = f"{table_name}.{col_name}"
                    if col_key in status_columns:
                        filter_cond = status_columns[col_key].get("active_filter", "")
                        if filter_cond:
                            filters.append(filter_cond)
                measure["filters_applied"] = filters
            
            facts.append({
                "name": table_name.split('.')[-1],
                "source": table_name,
                "grain": table_data.get("pk", ["ID"]),
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "nullable": c.get("nullable", True),
                        "semantic_role": c.get("semantic_role", "measure_component"),
                        "description": c["name"]
                    }
                    for c in table_data.get("columns", [])
                ],
                "measures": fact_measures,
                "foreign_keys": [
                    {"column": fk.split('→')[0], "references": fk.split('→')[1].split('.')[0]}
                    for fk in table_data.get("fks", [])
                ]
            })
        
        return facts
    
    def _fallback_metrics(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate basic fallback metrics."""
        metrics = []
        
        for fact in facts[:3]:  # First 3 facts
            for measure in fact.get("measures", [])[:2]:  # First 2 measures per fact
                metrics.append({
                    "name": f"{fact['name']} {measure['name']}",
                    "purpose": f"Track {measure['name']}",
                    "logic": measure["expression"],
                    "inputs": [f"{fact['name']}.{measure['name']}"],
                    "constraints": measure.get("filters_applied", []),
                    "explain": f"Aggregate {measure['name']} from {fact['name']}"
                })
        
        return metrics[:5]  # Max 5 metrics
    
    def _build_source_assets(
        self,
        entities: List[Dict],
        dimensions: List[Dict],
        facts: List[Dict]
    ) -> List[Dict[str, str]]:
        """Build source assets list."""
        assets = []
        
        for entity in entities:
            assets.append({"kind": "table", "name_or_path": entity["source"]})
        
        for dim in dimensions:
            assets.append({"kind": "table", "name_or_path": dim["source"]})
        
        for fact in facts:
            assets.append({"kind": "table", "name_or_path": fact["source"]})
        
        return assets