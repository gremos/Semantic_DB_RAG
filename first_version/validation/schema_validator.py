import json
import jsonschema
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SchemaValidator:
    """JSON Schema validation (Rail 2: Constraint Rail)."""
    
    def __init__(self):
        self.schemas = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load all JSON schemas."""
        schema_files = {
            "discovery": "schemas/discovery_schema.json",
            "semantic_model": "schemas/semantic_model_schema.json",
            "answer": "schemas/answer_schema.json"
        }
        
        for name, path in schema_files.items():
            try:
                with open(path, 'r') as f:
                    self.schemas[name] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load schema {name}: {e}")
                raise
    
    def validate(self, data: Dict[str, Any], schema_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate data against schema.
        
        Returns:
            (is_valid, error_message)
        """
        if schema_name not in self.schemas:
            return (False, f"Unknown schema: {schema_name}")
        
        try:
            jsonschema.validate(instance=data, schema=self.schemas[schema_name])
            return (True, None)
        except jsonschema.exceptions.ValidationError as e:
            return (False, f"Validation error: {e.message}")
        except Exception as e:
            return (False, f"Unexpected error: {str(e)}")