from typing import Dict, Any, List, Tuple
from llm.azure_client import AzureLLMClient
from utils.json_extractor import JSONExtractor
import json
import logging

logger = logging.getLogger(__name__)

class ModelAssembler:
    """Phase 5: Hierarchical incremental assembly - small focused LLM calls."""
    
    # Batch sizes to stay under token limits
    ENTITY_BATCH_SIZE = 2
    DIMENSION_BATCH_SIZE = 2
    FACT_BATCH_SIZE = 1  # Facts are most complex
    
    def __init__(self, llm_client: AzureLLMClient):
        self.llm = llm_client
        self.json_extractor = JSONExtractor()
        from config.settings import settings
        self.ENTITY_BATCH_SIZE = settings.entity_batch_size
        self.DIMENSION_BATCH_SIZE = settings.dimension_batch_size
        self.FACT_BATCH_SIZE = settings.fact_batch_size
        
        # Add logging
        logger.info(f"Batch sizes: entities={self.ENTITY_BATCH_SIZE}, "
                   f"dimensions={self.DIMENSION_BATCH_SIZE}, facts={self.FACT_BATCH_SIZE}")
        
        self._load_prompts()
    
    def _load_prompts(self):
        """Load specialized prompts for each assembly phase."""
        # Entity assembly prompt (lightweight)
        self.entity_prompt = """You are assembling ENTITY records for a semantic model.

Input: Table classifications and column info for ENTITY tables.
Output: JSON array of entity objects.

**OUTPUT RULES:**
1. Your response must be ONLY a JSON array
2. Start with [ and end with ]
3. NO explanations, NO markdown, NO text before or after the JSON
4. Each entity must have: name, source, primary_key, description, columns

**EXAMPLE OUTPUT:**
[
  {
    "name": "Customer",
    "source": "dbo.Customers",
    "primary_key": ["CustomerID"],
    "description": "Customer master data",
    "columns": [
      {
        "name": "CustomerID",
        "type": "int",
        "nullable": false,
        "semantic_role": "primary_key",
        "description": "Unique customer identifier"
      },
      {
        "name": "CustomerName",
        "type": "nvarchar(100)",
        "nullable": false,
        "semantic_role": "name",
        "description": "Customer business name"
      }
    ]
  }
]

**IMPORTANT:**
- Keep descriptions concise (one sentence)
- Use business-friendly names
- If uncertain about primary_key, use ["ID"]
- Only include up to 10 most important columns
- semantic_role should be one of: primary_key, foreign_key, name, identifier, metadata

START YOUR RESPONSE WITH: [
DO NOT include any text before the opening bracket [
"""

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
        self.fact_prompt = """"You are assembling FACT records for a semantic model.

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

CRITICAL FOREIGN KEY FORMAT:
foreign_keys MUST be: [{"column": "string", "references": "string"}]
Do NOT use nested objects for column or references!

Example:
{
  "foreign_keys": [
    {"column": "CustomerID", "references": "dbo.Customer"},
    {"column": "ProductID", "references": "dbo.Product"}
  ]
}

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
- constraints: ARRAY of filter strings (e.g., ["Status = 'Active'", "Amount > 0"])
- explain: Detailed explanation

CRITICAL: constraints MUST be an array, NOT a string!

Example:
[
  {
    "name": "Total Active Revenue",
    "purpose": "Revenue from active contracts",
    "logic": "Sum all active contract amounts",
    "inputs": ["Sales.Revenue"],
    "constraints": ["Status = 'Active'", "Amount > 0"],
    "explain": "..."
  }
]

Return ONLY JSON array starting with ["""

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
            
            # FIX: Monitor prompt size
            user_prompt = f"""Process these {len(batch)} ENTITY tables:

    {json.dumps(batch_context, indent=2)}

    Return JSON array of entities."""
            
            prompt_size = len(self.entity_prompt) + len(user_prompt)
            logger.info(f"    Prompt size: {prompt_size:,} chars")
            
            # FIX: Warn and skip if too large
            if prompt_size > 20000:
                logger.error(f"    ✗ Prompt too large ({prompt_size:,} chars > 20,000 limit)")
                logger.error(f"    Using fallback for this batch")
                all_entities.extend(self._fallback_entities(batch, compressed_discovery))
                continue
            
            # Call LLM with retry logic
            success = False
            for attempt in range(2):
                try:
                    response = self.llm.generate(self.entity_prompt, user_prompt)
                    logger.info(f"    LLM response length: {len(response)} chars")
                    
                    # FIX: Check for empty response immediately
                    if len(response) == 0:
                        logger.error(f"    ✗ LLM returned EMPTY response (attempt {attempt + 1}/2)")
                        if attempt == 0:
                            logger.info(f"    Retrying with simplified prompt...")
                            # Reduce batch size in half and retry
                            half_batch = batch[:len(batch)//2] if len(batch) > 1 else batch
                            batch_context = self._build_entity_batch_context(half_batch, compressed_discovery)
                            user_prompt = f"Process these {len(half_batch)} ENTITY tables:\n{json.dumps(batch_context, indent=2)}\nReturn JSON array."
                            continue
                        else:
                            logger.error(f"    Using fallback after 2 empty responses")
                            all_entities.extend(self._fallback_entities(batch, compressed_discovery))
                            success = True
                            break
                    
                    # Try to extract JSON
                    result, method = self.json_extractor.extract(response, log_failures=True)
                    
                    if result and isinstance(result, list):
                        all_entities.extend(result)
                        logger.info(f"    ✓ Assembled {len(result)} entities")
                        success = True
                        break
                    elif result and isinstance(result, dict) and "entities" in result:
                        all_entities.extend(result["entities"])
                        logger.info(f"    ✓ Assembled {len(result['entities'])} entities")
                        success = True
                        break
                    else:
                        logger.warning(f"    ⚠ JSON extraction failed (attempt {attempt + 1}/2)")
                        if attempt == 0:
                            continue
                
                except Exception as e:
                    logger.error(f"    ✗ Batch failed (attempt {attempt + 1}/2): {e}")
                    if attempt == 0:
                        continue
            
            if not success:
                logger.warning(f"    Using fallback for batch after all attempts")
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

                # ADD THESE LINES:
                logger.info(f"    LLM response length: {len(response)} chars")
                logger.debug(f"    Response preview (first 500): {response[:500]}")
                logger.debug(f"    Response preview (last 500): {response[-500:]}")

                # If still failing, log the FULL response:
                if len(response) < 5000:  # Only log full response if reasonable size
                    logger.debug(f"    Full response:\n{response}")
                else:
                    logger.debug(f"    Response too long to log fully ({len(response)} chars)")


                result, method = self.json_extractor.extract(response, log_failures=True)

                if not result or (not isinstance(result, list) and not (isinstance(result, dict) and "facts" in result)):
                    # Save failed response for debugging
                    with open(f"failed_fact_batch_{i//self.FACT_BATCH_SIZE + 1}.txt", "w") as f:
                        f.write(response)
                    logger.error(f"    Saved failed response to failed_fact_batch_{i//self.FACT_BATCH_SIZE + 1}.txt")
                
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
            batch_num = i//self.FACT_BATCH_SIZE + 1
            logger.info(f"  Processing fact batch {batch_num} ({len(batch)} tables)...")
            
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
                
                # 🔍 ADD: Log prompt size
                logger.info(f"    Prompt size: {len(user_prompt):,} chars")
                
                response = self.llm.generate(self.fact_prompt, user_prompt)
                
                # 🔍 ADD: Log response details
                logger.info(f"    LLM response length: {len(response)} chars")
                logger.debug(f"    Response starts with: {response[:200]}")
                logger.debug(f"    Response ends with: {response[-200:]}")
                
                # 🔍 CHANGE: Enable failure logging
                result, method = self.json_extractor.extract(response, log_failures=True)
                
                # 🔍 ADD: Log extraction result
                logger.info(f"    JSON extraction method: {method}")
                
                if result and isinstance(result, list):
                    all_facts.extend(result)
                    logger.info(f"    ✓ Assembled {len(result)} facts")
                elif result and isinstance(result, dict) and "facts" in result:
                    all_facts.extend(result["facts"])
                    logger.info(f"    ✓ Assembled {len(result['facts'])} facts")
                else:
                    # 🔍 ADD: Save failed response to file
                    logger.warning(f"    ⚠ LLM failed, using fallback for batch")
                    logger.warning(f"    Result type: {type(result)}, Method: {method}")
                    
                    failed_file = f"failed_fact_batch_{batch_num}.txt"
                    with open(failed_file, "w", encoding="utf-8") as f:
                        f.write("="*80 + "\n")
                        f.write(f"BATCH {batch_num} - FAILED\n")
                        f.write("="*80 + "\n\n")
                        f.write("TABLES IN BATCH:\n")
                        f.write(", ".join(batch) + "\n\n")
                        f.write("="*80 + "\n")
                        f.write("LLM RESPONSE:\n")
                        f.write("="*80 + "\n")
                        f.write(response)
                        f.write("\n\n")
                        f.write("="*80 + "\n")
                        f.write("EXTRACTION RESULT:\n")
                        f.write("="*80 + "\n")
                        f.write(f"Method: {method}\n")
                        f.write(f"Result: {result}\n")
                    
                    logger.error(f"    Saved failed response to {failed_file}")
                    
                    all_facts.extend(self._fallback_facts(batch, measures, status_columns, compressed_discovery))
            
            except Exception as e:
                logger.error(f"    ✗ Batch failed with exception: {e}", exc_info=True)
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
                # ✅ ADD: Fix constraints if they're strings
                for metric in result[:5]:
                    if isinstance(metric.get("constraints"), str):
                        # Split string into array
                        constraints_str = metric["constraints"]
                        if ";" in constraints_str:
                            metric["constraints"] = [c.strip() for c in constraints_str.split(";")]
                        elif constraints_str:
                            metric["constraints"] = [constraints_str]
                        else:
                            metric["constraints"] = []
                        logger.warning(f"Auto-corrected constraints from string to array for metric: {metric['name']}")
                
                return result[:5]
        except Exception as e:
            print(e)
    
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
            
            # FIX: Reduce from 10 to 5 columns max per table
            columns = table_data.get("columns", [])[:5]  # Was [:10]
            
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
                    for c in columns
                ],
                "column_count": len(table_data.get("columns", [])),
                "total_columns_note": f"Showing first 5 of {len(table_data.get('columns', []))} columns"
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
        """Build compact context for fact batch - REDUCED SIZE."""
        batch_context = []
        
        for table_name in fact_names:
            table_data = compressed_discovery["tables"].get(table_name, {})
            
            # Get measures for this fact - LIMIT to first 5
            fact_measures = measures.get(table_name, [])[:5]
            
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
                # LIMIT columns to 10 max
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "semantic_role": c.get("semantic_role"),
                        "is_fk": c.get("is_fk", False)
                    }
                    for c in table_data.get("columns", [])[:5]  
                ],
                "measures": fact_measures,  # Already limited to 5
                "status_columns": fact_status,
                "fks": table_data.get("fks", [])[:5]  # LIMIT FKs to 5
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
            
            # FIX: Parse FK format properly
            foreign_keys = []
            for fk_str in table_data.get("fks", []):
                try:
                    # Parse "column→ref_table.ref_column"
                    parts = fk_str.split('→')
                    if len(parts) == 2:
                        fk_column = parts[0]
                        ref_parts = parts[1].split('.')
                        if len(ref_parts) >= 1:
                            ref_table = ref_parts[0]
                            
                            # ✅ CRITICAL: Ensure both are strings
                            fk_obj = {
                                "column": str(fk_column),  # Force string
                                "references": str(ref_table)  # Force string
                            }
                            foreign_keys.append(fk_obj)
                except Exception as e:
                    logger.warning(f"Failed to parse FK '{fk_str}': {e}")
                    continue
            
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
                "foreign_keys": foreign_keys  # FIX: Use properly parsed FKs
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