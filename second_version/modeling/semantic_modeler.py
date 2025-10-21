import json
from typing import Dict, Any, Tuple
from llm.azure_client import AzureLLMClient
from validation.schema_validator import SchemaValidator
from validation.grounding_validator import GroundingValidator
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
        
        Returns:
            (success, model_json, error_message)
        """
        # Build user prompt
        user_prompt = self._build_user_prompt(discovery_data, domain_hints)
        
        # Try up to 2 times (initial + 1 retry)
        for attempt in range(2):
            try:
                logger.info(f"Modeling attempt {attempt + 1}/2")
                
                # Generate model
                response = self.llm.generate(self.system_prompt, user_prompt)
                
                # Parse JSON
                model_json = self._extract_json(response)
                if not model_json:
                    continue
                
                # Validate against schema
                valid, schema_error = self.validator.validate(model_json, "semantic_model")
                if not valid:
                    logger.warning(f"Schema validation failed: {schema_error}")
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
                logger.error(f"Modeling attempt {attempt + 1} failed: {e}")
                if attempt == 1:
                    return (False, {}, str(e))
        
        return (False, {}, "Failed to generate valid semantic model after 2 attempts")
    
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
            # Try direct parse
            return json.loads(response)
        except:
            # Try extracting from code block
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