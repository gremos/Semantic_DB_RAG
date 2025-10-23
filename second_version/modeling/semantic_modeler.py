import json
import time
from typing import Dict, Any, Tuple, List
from llm.azure_client import AzureLLMClient
from validation.schema_validator import SchemaValidator
from validation.grounding_validator import GroundingValidator
from utils.json_extractor import JSONExtractor
from .schema_batcher import SchemaBatcher
import logging

logger = logging.getLogger(__name__)

class SemanticModeler:
    """Phase 2: Semantic Model Creation using LLM."""
    
    def __init__(self, llm_client: AzureLLMClient, validator: SchemaValidator):
        self.llm = llm_client
        self.validator = validator
        self.json_extractor = JSONExtractor()
        self._load_prompt()
    
    def _load_prompt(self):
        """Load modeling system prompt."""
        with open("prompts/modeling_prompt.txt", 'r') as f:
            self.system_prompt = f.read()
    
    def create_model(
        self, 
        discovery_data: Dict[str, Any],
        domain_hints: str = ""
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Create semantic model from discovery data.
        Uses batching to handle large datasets.
        
        Returns:
            (success, model_json, error_message)
        """
        # Check size and decide on batching
        estimated_tokens = SchemaBatcher.estimate_tokens(discovery_data)
        logger.info(f"Estimated discovery tokens: {estimated_tokens:,}")
        
        # If payload is large, use batching
        if estimated_tokens > 25000:  # Threshold for batching
            logger.info("Discovery data is large. Using batched processing...")
            return self._create_model_batched(discovery_data, domain_hints)
        else:
            logger.info("Discovery data is small enough for single request")
            return self._create_model_single(discovery_data, domain_hints)
    
    def _create_model_single(
        self,
        discovery_data: Dict[str, Any],
        domain_hints: str = ""
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Create model with single API call."""
        user_prompt = self._build_user_prompt(discovery_data, domain_hints)
        
        for attempt in range(2):
            try:
                logger.info(f"Modeling attempt {attempt + 1}/2")
                
                response = self.llm.generate(self.system_prompt, user_prompt)
                
                # ENHANCED: Log response details
                logger.info(f"Received response of length: {len(response)} characters")
                logger.debug(f"Response starts with: {response[:200]}")
                logger.debug(f"Response ends with: {response[-200:]}")
                
                # ENHANCED: Use robust JSON extraction
                model_json, extraction_method = self.json_extractor.extract(response, log_failures=True)
                
                if not model_json:
                    logger.error("Failed to extract JSON from response")
                    logger.error(f"Full response:\n{response}")
                    
                    if attempt == 0:
                        logger.info("Retrying with more explicit instructions...")
                        continue
                    else:
                        return (False, {}, "Could not extract valid JSON from LLM response after 2 attempts")
                
                logger.info(f"JSON extracted successfully using method: {extraction_method}")
                logger.debug(f"Extracted JSON keys: {list(model_json.keys())}")
                
                # POST-PROCESS: Enrich with column metadata
                model_json = self._enrich_with_columns(model_json, discovery_data)
                
                # Validate schema
                valid, schema_error = self.validator.validate(model_json, "semantic_model")
                if not valid:
                    logger.warning(f"Schema validation failed: {schema_error}")
                    logger.debug(f"Invalid model structure: {json.dumps(model_json, indent=2)[:1000]}")
                    if attempt == 0:
                        continue
                    return (False, {}, f"Schema validation failed: {schema_error}")
                
                # Validate grounding
                grounding_validator = GroundingValidator(discovery_data)
                grounded, grounding_errors = grounding_validator.validate_semantic_model(model_json)
                
                if not grounded:
                    logger.warning(f"Grounding validation failed: {grounding_errors}")
                    if attempt == 0:
                        continue
                    return (False, {}, f"Grounding errors: {'; '.join(grounding_errors)}")
                
                logger.info("Semantic model created successfully")
                return (True, model_json, "")
            
            except Exception as e:
                logger.error(f"Modeling attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == 1:
                    return (False, {}, str(e))
        
        return (False, {}, "Failed to generate valid semantic model after 2 attempts")
    
    def _create_model_batched(
        self,
        discovery_data: Dict[str, Any],
        domain_hints: str = ""
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Create model using batched processing."""
        # Create batches
        batches = SchemaBatcher.create_batches(discovery_data, max_tokens_per_batch=20000)
        logger.info(f"Split discovery into {len(batches)} batches")
        
        models = []
        
        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_idx}/{len(batches)}")
            logger.info(f"  Batch contains {len(batch.get('schemas', []))} schema(s)")
            
            # Add delay between batches to respect rate limits
            if batch_idx > 1:
                delay = 5  # 5 seconds between batches
                logger.info(f"  Waiting {delay}s before next batch...")
                time.sleep(delay)
            
            # Process batch
            success, model, error = self._create_model_single(batch, domain_hints)
            
            if not success:
                logger.error(f"  Batch {batch_idx} failed: {error}")
                # Continue with other batches rather than failing completely
                continue
            
            models.append(model)
            logger.info(f"  Batch {batch_idx} completed successfully")
        
        if not models:
            return (False, {}, "All batches failed to produce valid models")
        
        # Merge models
        logger.info(f"Merging {len(models)} models...")
        merged_model = SchemaBatcher.merge_semantic_models(models)
        
        logger.info("Semantic model created successfully via batching")
        return (True, merged_model, "")
    
    def _enrich_with_columns(
        self, 
        model_json: Dict[str, Any], 
        discovery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich semantic model with column metadata from discovery.
        This is a fallback if LLM doesn't include columns.
        """
        # Build lookup: source -> columns
        source_columns = {}
        for schema in discovery_data.get("schemas", []):
            schema_name = schema["name"]
            for table in schema.get("tables", []):
                table_name = table["name"]
                full_name = f"{schema_name}.{table_name}"
                source_columns[full_name] = table.get("columns", [])
        
        # Get column samples
        column_samples = discovery_data.get("column_samples", {})
        
        # Enrich entities
        for entity in model_json.get("entities", []):
            if "columns" not in entity or not entity["columns"]:
                source = entity.get("source")
                if source in source_columns:
                    entity["columns"] = self._create_column_metadata(
                        source_columns[source],
                        source,
                        column_samples,
                        entity.get("primary_key", [])
                    )
        
        # Enrich dimensions
        for dimension in model_json.get("dimensions", []):
            if "columns" not in dimension or not dimension["columns"]:
                source = dimension.get("source")
                if source in source_columns:
                    dimension["columns"] = self._create_column_metadata(
                        source_columns[source],
                        source,
                        column_samples,
                        dimension.get("keys", [])
                    )
        
        # Enrich facts
        for fact in model_json.get("facts", []):
            if "columns" not in fact or not fact["columns"]:
                source = fact.get("source")
                if source in source_columns:
                    fact["columns"] = self._create_column_metadata(
                        source_columns[source],
                        source,
                        column_samples,
                        fact.get("grain", []),
                        fact.get("measures", [])
                    )
        
        return model_json
    
    def _create_column_metadata(
        self,
        columns: List[Dict[str, Any]],
        source: str,
        column_samples: Dict[str, List],
        key_columns: List[str],
        measures: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Create column metadata with semantic roles."""
        enriched = []
        
        # Get measure dependencies
        measure_deps = set()
        if measures:
            for measure in measures:
                measure_deps.update(measure.get("depends_on", []))
        
        for col in columns:
            col_name = col["name"]
            col_type = col["type"]
            
            # Determine semantic role
            semantic_role = "metadata"  # Default
            description = f"Column {col_name}"
            
            if col_name in key_columns:
                semantic_role = "primary_key"
                description = f"Primary key column: {col_name}"
            elif col_name in measure_deps:
                semantic_role = "measure_component"
                description = f"Used in measure calculations: {col_name}"
            elif self._is_status_column(col_name):
                semantic_role = "status_indicator"
                description = self._generate_status_description(col_name, col_type)
            elif "date" in col_type.lower() or "time" in col_type.lower():
                semantic_role = "timestamp"
                description = f"Timestamp column: {col_name}"
            
            # Get sample values if available
            sample_key = f"{source}.{col_name}"
            sample_values = column_samples.get(sample_key, None)
            
            enriched.append({
                "name": col_name,
                "type": col_type,
                "nullable": col.get("nullable", True),
                "semantic_role": semantic_role,
                "description": description,
                "sample_values": sample_values,
                "cardinality_estimate": self._estimate_cardinality(sample_values)
            })
        
        return enriched
    
    def _is_status_column(self, col_name: str) -> bool:
        """Check if column name suggests a status indicator."""
        status_keywords = [
            'status', 'state', 'cancelled', 'deleted', 'active', 
            'enabled', 'disabled', 'flag', 'is', 'has'
        ]
        col_lower = col_name.lower()
        return any(kw in col_lower for kw in status_keywords)
    
    def _generate_status_description(self, col_name: str, col_type: str) -> str:
        """Generate description for status indicator columns."""
        col_lower = col_name.lower()
        
        if 'cancelled' in col_lower:
            if 'date' in col_type.lower() or 'time' in col_type.lower():
                return f"Status indicator: When NULL, record is ACTIVE. When populated, record is CANCELLED."
            else:
                return f"Status indicator: Check if record is cancelled."
        
        if 'deleted' in col_lower:
            if 'date' in col_type.lower() or 'time' in col_type.lower():
                return f"Status indicator: When NULL, record is ACTIVE. When populated, record is DELETED."
            else:
                return f"Status indicator: Check if record is deleted."
        
        if 'active' in col_lower or 'enabled' in col_lower:
            return f"Status indicator: Indicates if record is active/enabled."
        
        return f"Status indicator column: {col_name}"
    
    def _estimate_cardinality(self, sample_values: List) -> str:
        """Estimate cardinality based on sample values."""
        if not sample_values:
            return None
        
        unique_count = len(set(str(v) for v in sample_values))
        
        if unique_count <= 3:
            return "low"
        elif unique_count <= 7:
            return "medium"
        else:
            return "high"
    
    def _build_user_prompt(self, discovery_data: Dict[str, Any], domain_hints: str) -> str:
        """Build user prompt with discovery data."""
        prompt_parts = [
            "# Discovery Data",
            json.dumps(discovery_data, indent=2)
        ]
        
        if domain_hints:
            prompt_parts.extend([
                "",
                "# Domain Hints",
                domain_hints
            ])
        
        prompt_parts.extend([
            "",
            "# Task",
            "Generate a semantic model following the schema.",
            "",
            "IMPORTANT: You can optionally include column metadata, but it's not required.",
            "Focus on getting the basic structure correct first:",
            "- entities, dimensions, facts",
            "- relationships",
            "- measures with clear expressions",
            "- metrics",
            "",
            "Return ONLY valid JSON. No explanations, no markdown formatting."
        ])
        
        return "\n".join(prompt_parts)