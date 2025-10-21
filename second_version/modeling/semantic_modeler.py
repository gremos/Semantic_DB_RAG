import json
import time
from typing import Dict, Any, Tuple, List
from llm.azure_client import AzureLLMClient
from validation.schema_validator import SchemaValidator
from validation.grounding_validator import GroundingValidator
from .schema_batcher import SchemaBatcher
import logging

logger = logging.getLogger(__name__)

class SemanticModeler:
    """Phase 2: Semantic Model Creation using LLM."""
    
    def __init__(self, llm_client: AzureLLMClient, validator: SchemaValidator):
        self.llm = llm_client
        self.validator = validator
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
                
                # DEBUG: Log the raw response
                logger.debug(f"Raw LLM response (first 500 chars): {response[:500]}")
                
                model_json = self._extract_json(response)
                
                if not model_json:
                    logger.warning("Failed to extract JSON from response")
                    logger.debug(f"Full response: {response}")
                    continue
                
                # DEBUG: Log what was extracted
                logger.debug(f"Extracted JSON keys: {list(model_json.keys())}")
                
                # Validate
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
            "Generate a semantic model following the schema. Return ONLY valid JSON."
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response (handles markdown code blocks)."""
        try:
            return json.loads(response)
        except:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
        
        return None