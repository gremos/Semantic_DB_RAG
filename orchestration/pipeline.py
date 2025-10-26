"""
File: orchestration/pipeline.py
Main orchestration pipeline for Discovery → Modeling → Q&A
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from qa.question_handler import QuestionHandler

logger = logging.getLogger(__name__)


class SemanticPipeline:
    """
    Orchestrates the three-phase pipeline:
    1. Discovery (database introspection)
    2. Semantic Modeling (business layer creation)
    3. Question Answering (LLM-driven SQL generation)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        File: orchestration/pipeline.py, Line: 23
        
        Args:
            config: Configuration dictionary from environment
        """
        self.config = config
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            deployment_name=config["DEPLOYMENT_NAME"],
            api_version=config["API_VERSION"],
            azure_endpoint=config["AZURE_ENDPOINT"],
            api_key=config["AZURE_OPENAI_API_KEY"],
        )
        
        # Cache storage
        self.cache_dir = config.get("cache_dir", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Discovery and semantic model caches
        self.discovery_json: Optional[Dict[str, Any]] = None
        self.semantic_model: Optional[Dict[str, Any]] = None

    def run_full_pipeline(self, question: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run complete pipeline: Discovery → Modeling → Q&A
        
        File: orchestration/pipeline.py, Line: 51
        
        Args:
            question: Natural language question
            force_refresh: If True, bypass cache and regenerate discovery/model
            
        Returns:
            Answer JSON or Refusal JSON
        """
        try:
            # Phase 1: Discovery
            logger.info("=== PHASE 1: DISCOVERY ===")
            discovery_json = self._get_or_create_discovery(force_refresh=force_refresh)
            
            # Phase 2: Semantic Modeling
            logger.info("=== PHASE 2: SEMANTIC MODELING ===")
            semantic_model = self._get_or_create_semantic_model(
                discovery_json, 
                force_refresh=force_refresh
            )
            
            # Phase 3: Question Answering
            logger.info("=== PHASE 3: QUESTION ANSWERING ===")
            handler = QuestionHandler(self.llm, semantic_model)
            answer = handler.answer_question(question)
            
            return answer
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                "status": "refuse",
                "refusal": {
                    "reason": f"System error: {str(e)}",
                    "suggestions": ["Please try again or contact support"]
                }
            }

    def _get_or_create_discovery(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get cached discovery or create new one via LLM-driven introspection.
        
        File: orchestration/pipeline.py, Line: 85
        
        Args:
            force_refresh: If True, bypass cache and regenerate
            
        Returns:
            Discovery JSON
        """
        cache_path = os.path.join(self.cache_dir, "discovery.json")
        cache_meta_path = os.path.join(self.cache_dir, "discovery_meta.json")
        
        # Check cache validity (unless force_refresh)
        if not force_refresh and os.path.exists(cache_path) and os.path.exists(cache_meta_path):
            with open(cache_meta_path, 'r') as f:
                meta = json.load(f)
                cache_time = datetime.fromisoformat(meta["created_at"])
                cache_hours = int(self.config.get("discovery_cache_hours", 168))
                
                if datetime.now() - cache_time < timedelta(hours=cache_hours):
                    logger.info("Using cached discovery JSON")
                    with open(cache_path, 'r') as df:
                        return json.load(df)
                else:
                    logger.info("Discovery cache expired")
        elif force_refresh:
            logger.info("Force refresh enabled - bypassing cache")
        
        # Create new discovery
        logger.info("Running discovery introspection")
        discovery_json = self._run_discovery_via_llm()
        
        # Cache it
        with open(cache_path, 'w') as f:
            json.dump(discovery_json, f, indent=2)
        with open(cache_meta_path, 'w') as f:
            json.dump({"created_at": datetime.now().isoformat()}, f)
        
        return discovery_json

    def _run_discovery_via_llm(self) -> Dict[str, Any]:
        """
        Introspect database using LLM to guide the process.
        
        File: orchestration/pipeline.py, Line: 121
        
        Returns:
            Discovery JSON with normalized metadata
        """
        # NOTE: In production, this would connect to the actual database
        # For this spec, we'll simulate with a sample discovery
        
        logger.info("Introspecting database schema...")
        
        # Sample discovery structure (replace with actual DB introspection)
        discovery_json = {
            "database": {
                "vendor": "mssql",
                "version": "2019",
                "fingerprint": "sample_db_v1"
            },
            "dialect": "mssql",
            "schemas": [
                {
                    "name": "dbo",
                    "tables": [
                        {
                            "name": "Customer",
                            "type": "table",
                            "columns": [
                                {"name": "CustomerID", "type": "int", "nullable": False},
                                {"name": "CustomerName", "type": "varchar(100)", "nullable": False},
                                {"name": "Email", "type": "varchar(100)", "nullable": True},
                                {"name": "CreatedDate", "type": "datetime", "nullable": False}
                            ],
                            "primary_key": ["CustomerID"],
                            "foreign_keys": [],
                            "rowcount_sample": 5000,
                            "source_assets": []
                        },
                        {
                            "name": "Orders",
                            "type": "table",
                            "columns": [
                                {"name": "OrderID", "type": "int", "nullable": False},
                                {"name": "CustomerID", "type": "int", "nullable": False},
                                {"name": "OrderDate", "type": "datetime", "nullable": False},
                                {"name": "TotalAmount", "type": "decimal(18,2)", "nullable": False},
                                {"name": "Status", "type": "varchar(50)", "nullable": False}
                            ],
                            "primary_key": ["OrderID"],
                            "foreign_keys": [
                                {
                                    "column": "CustomerID",
                                    "ref_table": "dbo.Customer",
                                    "ref_column": "CustomerID"
                                }
                            ],
                            "rowcount_sample": 50000,
                            "source_assets": []
                        },
                        {
                            "name": "OrderLines",
                            "type": "table",
                            "columns": [
                                {"name": "OrderLineID", "type": "int", "nullable": False},
                                {"name": "OrderID", "type": "int", "nullable": False},
                                {"name": "ProductID", "type": "int", "nullable": False},
                                {"name": "Quantity", "type": "int", "nullable": False},
                                {"name": "UnitPrice", "type": "decimal(18,2)", "nullable": False},
                                {"name": "LineTotal", "type": "decimal(18,2)", "nullable": False}
                            ],
                            "primary_key": ["OrderLineID"],
                            "foreign_keys": [
                                {
                                    "column": "OrderID",
                                    "ref_table": "dbo.Orders",
                                    "ref_column": "OrderID"
                                }
                            ],
                            "rowcount_sample": 150000,
                            "source_assets": []
                        }
                    ]
                }
            ],
            "named_assets": [
                {
                    "kind": "view",
                    "name": "dbo.vCustomerSales",
                    "sql_normalized": "SELECT c.CustomerID, c.CustomerName, SUM(o.TotalAmount) AS TotalSales FROM dbo.Customer c INNER JOIN dbo.Orders o ON c.CustomerID = o.CustomerID GROUP BY c.CustomerID, c.CustomerName"
                }
            ]
        }
        
        logger.info(f"Discovered {len(discovery_json['schemas'][0]['tables'])} tables")
        return discovery_json

    def _get_or_create_semantic_model(
        self, discovery_json: Dict[str, Any], force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get cached semantic model or create via LLM.
        
        File: orchestration/pipeline.py, Line: 218
        
        Args:
            discovery_json: Discovery JSON from Phase 1
            force_refresh: If True, bypass cache and regenerate
            
        Returns:
            Semantic Model JSON
        """
        cache_path = os.path.join(self.cache_dir, "semantic_model.json")
        cache_meta_path = os.path.join(self.cache_dir, "semantic_model_meta.json")
        
        # Check cache validity (unless force_refresh)
        if not force_refresh and os.path.exists(cache_path) and os.path.exists(cache_meta_path):
            with open(cache_meta_path, 'r') as f:
                meta = json.load(f)
                cache_time = datetime.fromisoformat(meta["created_at"])
                cache_hours = int(self.config.get("semantic_cache_hours", 168))
                
                if datetime.now() - cache_time < timedelta(hours=cache_hours):
                    logger.info("Using cached semantic model")
                    with open(cache_path, 'r') as sf:
                        return json.load(sf)
                else:
                    logger.info("Semantic model cache expired")
        elif force_refresh:
            logger.info("Force refresh enabled - bypassing semantic model cache")
        
        # Create new semantic model
        logger.info("Building semantic model via LLM")
        semantic_model = self._build_semantic_model_via_llm(discovery_json)
        
        # Cache it
        with open(cache_path, 'w') as f:
            json.dump(semantic_model, f, indent=2)
        with open(cache_meta_path, 'w') as f:
            json.dump({"created_at": datetime.now().isoformat()}, f)
        
        return semantic_model

    def _build_semantic_model_via_llm(
        self, discovery_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build semantic model using LLM with strict grounding to discovery.
        
        File: orchestration/pipeline.py, Line: 261
        
        Args:
            discovery_json: Discovery JSON
            
        Returns:
            Semantic Model JSON
        """
        prompt = f"""You are building a business-friendly semantic data model from database metadata.

Discovery JSON:
{json.dumps(discovery_json, indent=2)}

Your task:
1. Classify tables as entities (master data), dimensions (descriptive), or facts (transactional)
2. Generate friendly business names for each
3. Infer measures from numeric columns in facts (SUM, AVG, COUNT, etc.)
4. Map relationships using foreign keys
5. Suggest useful business metrics

STRICT RULES:
- ONLY use tables/columns from Discovery JSON
- DO NOT invent tables or columns
- Every source reference must be valid
- Rate confidence (high/medium/low) for each inference

Return JSON matching this schema:
{{
  "entities": [
    {{
      "name": "Customer",
      "source": "dbo.Customer",
      "primary_key": ["CustomerID"],
      "business_name": "Customer",
      "description": "Customer master data",
      "confidence": "high"
    }}
  ],
  "dimensions": [
    {{
      "name": "Date",
      "source": "dbo.DimDate",
      "keys": ["DateKey"],
      "attributes": ["Year", "Month", "Quarter"],
      "business_name": "Date Dimension",
      "confidence": "high"
    }}
  ],
  "facts": [
    {{
      "name": "Sales",
      "source": "dbo.Orders",
      "grain": ["OrderID"],
      "measures": [
        {{
          "name": "Revenue",
          "expression": "SUM(TotalAmount)",
          "aggregation": "sum",
          "column": "TotalAmount",
          "business_name": "Total Revenue",
          "confidence": "high"
        }}
      ],
      "foreign_keys": [
        {{
          "column": "CustomerID",
          "references": "Customer.CustomerID"
        }}
      ],
      "business_name": "Sales Transactions",
      "confidence": "high"
    }}
  ],
  "relationships": [
    {{
      "from": "Sales.CustomerID",
      "to": "Customer.CustomerID",
      "cardinality": "many_to_one",
      "confidence": "high"
    }}
  ],
  "metrics": [
    {{
      "name": "Customer Lifetime Value",
      "logic": "Sum of all sales per customer over their lifetime",
      "required_objects": ["Customer", "Sales"],
      "confidence": "medium"
    }}
  ],
  "audit": {{
    "dialect": "{discovery_json.get('dialect', 'unknown')}",
    "created_at": "{datetime.now().isoformat()}"
  }}
}}

Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        try:
            semantic_model = json.loads(response.content)
            
            # Validate that all sources exist in discovery
            valid = self._validate_semantic_model(semantic_model, discovery_json)
            if not valid["is_valid"]:
                logger.error(f"Semantic model validation failed: {valid['errors']}")
                # Retry once
                retry_prompt = f"""{prompt}

VALIDATION ERRORS FROM PREVIOUS ATTEMPT:
{json.dumps(valid['errors'], indent=2)}

Fix these errors and return corrected JSON."""
                retry_response = self.llm.invoke(retry_prompt)
                semantic_model = json.loads(retry_response.content)
            
            return semantic_model
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse semantic model: {e}")
            raise

    def _validate_semantic_model(
        self, semantic_model: Dict[str, Any], discovery_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that semantic model only references discovered objects.
        
        File: orchestration/pipeline.py, Line: 373
        
        Args:
            semantic_model: Proposed semantic model
            discovery_json: Ground truth discovery
            
        Returns:
            Validation result with errors
        """
        errors = []
        
        # Build valid sources from discovery
        valid_tables = set()
        valid_columns = {}
        
        for schema in discovery_json.get("schemas", []):
            for table in schema.get("tables", []):
                table_full_name = f"{schema['name']}.{table['name']}"
                valid_tables.add(table_full_name)
                valid_columns[table_full_name] = {col["name"] for col in table["columns"]}
        
        # Validate entities
        for entity in semantic_model.get("entities", []):
            if entity["source"] not in valid_tables:
                errors.append(f"Entity '{entity['name']}' references invalid source: {entity['source']}")
        
        # Validate facts
        for fact in semantic_model.get("facts", []):
            if fact["source"] not in valid_tables:
                errors.append(f"Fact '{fact['name']}' references invalid source: {fact['source']}")
            
            # Validate measures
            for measure in fact.get("measures", []):
                column = measure.get("column")
                if column and column not in valid_columns.get(fact["source"], set()):
                    errors.append(
                        f"Measure '{measure['name']}' references invalid column: "
                        f"{fact['source']}.{column}"
                    )
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }

    def invalidate_cache(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Invalidate cache files (discovery, semantic model, or both).
        
        File: orchestration/pipeline.py, Line: 410
        
        Args:
            cache_type: Type of cache to clear ('discovery', 'semantic', or None for both)
            
        Returns:
            Dictionary with cleared cache info
        """
        cleared = []
        
        cache_files = {
            "discovery": ["discovery.json", "discovery_meta.json"],
            "semantic": ["semantic_model.json", "semantic_model_meta.json"]
        }
        
        # Determine which caches to clear
        if cache_type is None:
            types_to_clear = ["discovery", "semantic"]
        elif cache_type in cache_files:
            types_to_clear = [cache_type]
        else:
            return {
                "status": "error",
                "message": f"Invalid cache_type: {cache_type}. Use 'discovery', 'semantic', or None"
            }
        
        # Clear the specified caches
        for cache_type_name in types_to_clear:
            for filename in cache_files[cache_type_name]:
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        cleared.append(filename)
                        logger.info(f"Removed cache file: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to remove {filename}: {e}")
        
        return {
            "status": "success",
            "cleared": cleared,
            "cache_dir": self.cache_dir
        }


def load_config_from_env() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    File: orchestration/pipeline.py, Line: 460
    
    Returns:
        Configuration dictionary
    """
    required_vars = [
        "DEPLOYMENT_NAME",
        "API_VERSION",
        "AZURE_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "DATABASE_CONNECTION_STRING"
    ]
    
    config = {}
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        config[var] = value
    
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    
    # Optional config with defaults
    config["discovery_cache_hours"] = os.getenv("DISCOVERY_CACHE_HOURS", "168")
    config["semantic_cache_hours"] = os.getenv("SEMANTIC_CACHE_HOURS", "168")
    config["schema_exclusions"] = os.getenv("SCHEMA_EXCLUSIONS", "sys,information_schema").split(",")
    config["table_exclusions"] = os.getenv("TABLE_EXCLUSIONS", "temp_,test_,backup_,old_").split(",")
    config["cache_dir"] = os.getenv("CACHE_DIR", "cache")
    
    return config


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = load_config_from_env()
    
    # Initialize pipeline
    pipeline = SemanticPipeline(config)
    
    # Example 1: Normal query (uses cache)
    question = "What are the total sales by customer?"
    result = pipeline.run_full_pipeline(question)
    
    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    # Example 2: Force refresh (bypasses cache)
    # result = pipeline.run_full_pipeline(question, force_refresh=True)
    
    # Example 3: Clear cache
    # cache_result = pipeline.invalidate_cache()
    # print(f"Cache cleared: {cache_result['cleared']}")
    
    # Example 4: Clear only discovery cache
    # cache_result = pipeline.invalidate_cache(cache_type='discovery')
    
    # Example 5: Clear only semantic model cache
    # cache_result = pipeline.invalidate_cache(cache_type='semantic')