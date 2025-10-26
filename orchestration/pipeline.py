"""
Orchestration Pipeline - Main execution flow
Discovery → Semantic Relationships → Modeling → Q&A
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

# Import phase modules
from discovery.db_introspector import DatabaseIntrospector
from modeling.semantic_relationships import enhance_discovery_with_semantic_relationships
from modeling.incremental_modeler import IncrementalModeler
from qa.question_handler import QuestionHandler

# LangChain Azure OpenAI
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)


class SemanticPipeline:
    """
    Main orchestration pipeline for the GPT-5 Semantic Modeling & SQL Q&A System.
    
    Flow:
    1. Phase 1: Discovery (DB introspection)
    2. Phase 1.5: Semantic Relationship Extraction (from views/RDLs)
    3. Phase 2: Semantic Model Creation
    4. Phase 3: Question Answering with Disambiguation
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            deployment_name=config["DEPLOYMENT_NAME"],
            api_version=config["API_VERSION"],
            azure_endpoint=config["AZURE_ENDPOINT"],
            api_key=config["AZURE_OPENAI_API_KEY"],
            # temperature=0.0  # Deterministic for consistency
        )
        
        # Cache paths
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.discovery_cache_path = os.path.join(self.cache_dir, "discovery.json")
        self.semantic_model_cache_path = os.path.join(self.cache_dir, "semantic_model.json")
        
        # State
        self.discovery_json: Optional[dict] = None
        self.semantic_model: Optional[dict] = None
    
    def run_discovery(self, force_refresh: bool = False) -> dict:
        """
        Phase 1: Database Discovery
        
        Returns: Discovery JSON
        """
        logger.info("=== PHASE 1: DISCOVERY ===")
        
        # Check cache
        if not force_refresh and os.path.exists(self.discovery_cache_path):
            cache_age_hours = self._get_file_age_hours(self.discovery_cache_path)
            if cache_age_hours < float(self.config.get("DISCOVERY_CACHE_HOURS", 168)):
                logger.info(f"Using cached discovery (age: {cache_age_hours:.1f} hours)")
                with open(self.discovery_cache_path, "r") as f:
                    self.discovery_json = json.load(f)
                return self.discovery_json
        
        logger.info("Running fresh discovery...")
        
        # Initialize introspector
        introspector = DatabaseIntrospector(
            connection_string=self.config["DATABASE_CONNECTION_STRING"],
            schema_exclusions=self.config.get("SCHEMA_EXCLUSIONS", "").split(","),
            table_exclusions=self.config.get("TABLE_EXCLUSIONS", "").split(",")
        )
        
        # Run discovery
        self.discovery_json = introspector.introspect()
        
        # Save to cache
        with open(self.discovery_cache_path, "w") as f:
            json.dump(self.discovery_json, f, indent=2)
        
        logger.info(f"Discovery complete: {len(self.discovery_json.get('schemas', []))} schemas")
        
        return self.discovery_json
    
    def run_semantic_relationship_extraction(self) -> dict:
        """
        Phase 1.5: Extract semantic relationships from views, stored procedures, and RDLs.
        This is NEW and critical for grounding.
        
        Returns: Enhanced Discovery JSON with semantic_relationships
        """
        logger.info("=== PHASE 1.5: SEMANTIC RELATIONSHIP EXTRACTION ===")
        
        if not self.discovery_json:
            raise ValueError("Must run discovery first")
        
        # Enhance discovery with semantic relationships
        self.discovery_json = enhance_discovery_with_semantic_relationships(self.discovery_json)
        
        # Save enhanced discovery
        enhanced_path = os.path.join(self.cache_dir, "discovery_enhanced.json")
        with open(enhanced_path, "w") as f:
            json.dump(self.discovery_json, f, indent=2)
        
        relationship_count = len(self.discovery_json.get("semantic_relationships", []))
        logger.info(f"Extracted {relationship_count} semantic relationships from views/RDLs")
        
        # Log sample for debugging
        if relationship_count > 0:
            sample = self.discovery_json["semantic_relationships"][0]
            logger.info(f"Sample relationship: {sample}")
        
        return self.discovery_json
    
    def run_semantic_modeling(self, force_refresh: bool = False) -> dict:
        """
        Phase 2: Semantic Model Creation
        
        Returns: Semantic Model JSON
        """
        logger.info("=== PHASE 2: SEMANTIC MODEL CREATION ===")
        
        if not self.discovery_json:
            raise ValueError("Must run discovery first")
        
        # Check cache
        if not force_refresh and os.path.exists(self.semantic_model_cache_path):
            cache_age_hours = self._get_file_age_hours(self.semantic_model_cache_path)
            if cache_age_hours < float(self.config.get("SEMANTIC_CACHE_HOURS", 168)):
                logger.info(f"Using cached semantic model (age: {cache_age_hours:.1f} hours)")
                with open(self.semantic_model_cache_path, "r") as f:
                    self.semantic_model = json.load(f)
                return self.semantic_model
        
        logger.info("Running fresh semantic modeling...")
        
        # Initialize modeler
        modeler = IncrementalModeler(
            llm=self.llm,
            discovery_json=self.discovery_json
        )
        
        # Run modeling
        self.semantic_model = modeler.build_semantic_model()
        
        # Validate schema
        if not self._validate_semantic_model_schema(self.semantic_model):
            logger.error("Semantic model failed schema validation")
            raise ValueError("Invalid semantic model schema")
        
        # Save to cache
        with open(self.semantic_model_cache_path, "w") as f:
            json.dump(self.semantic_model, f, indent=2)
        
        logger.info(f"Semantic modeling complete:")
        logger.info(f"  - Entities: {len(self.semantic_model.get('entities', []))}")
        logger.info(f"  - Dimensions: {len(self.semantic_model.get('dimensions', []))}")
        logger.info(f"  - Facts: {len(self.semantic_model.get('facts', []))}")
        logger.info(f"  - Relationships: {len(self.semantic_model.get('relationships', []))}")
        
        return self.semantic_model
    
    def answer_question(self, question: str) -> dict:
        """
        Phase 3: Question Answering with Smart Disambiguation
        
        Args:
            question: Natural language question
        
        Returns: Answer JSON with SQL or refusal
        """
        logger.info("=== PHASE 3: QUESTION ANSWERING ===")
        
        if not self.semantic_model:
            raise ValueError("Must run semantic modeling first")
        
        # Initialize Q&A handler
        handler = QuestionHandler(
            llm=self.llm,
            semantic_model=self.semantic_model,
            discovery_json=self.discovery_json
        )
        
        # Process question
        answer = handler.answer_question(question)
        
        # Log result
        logger.info(f"Answer status: {answer.get('status')}")
        if answer.get("status") == "ok":
            sql_count = len(answer.get("sql", []))
            logger.info(f"Generated {sql_count} SQL statement(s)")
        else:
            logger.warning(f"Refusal reason: {answer.get('refusal', {}).get('reason')}")
        
        return answer
    
    def run_full_pipeline(self, question: str, force_refresh: bool = False) -> dict:
        """
        Run the complete pipeline: Discovery → Modeling → Q&A
        
        Args:
            question: Natural language question
            force_refresh: Skip cache and regenerate everything
        
        Returns: Answer JSON
        """
        logger.info("=== RUNNING FULL PIPELINE ===")
        start_time = datetime.now()
        
        try:
            # Phase 1: Discovery
            self.run_discovery(force_refresh=force_refresh)
            
            # Phase 1.5: Semantic Relationship Extraction (NEW)
            self.run_semantic_relationship_extraction()
            
            # Phase 2: Semantic Modeling
            self.run_semantic_modeling(force_refresh=force_refresh)
            
            # Phase 3: Question Answering
            answer = self.answer_question(question)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"=== PIPELINE COMPLETE ({elapsed:.2f}s) ===")
            
            return answer
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "status": "refuse",
                "refusal": {
                    "reason": f"Pipeline error: {str(e)}",
                    "clarifying_questions": ["Please contact support or check logs."]
                }
            }
    
    def _validate_semantic_model_schema(self, model: dict) -> bool:
        """
        Validate semantic model against required schema.
        
        This is where the 'type' is a required property error was happening.
        We need to ensure all relationships have a 'type' field.
        """
        logger.info("Validating semantic model schema...")
        
        # Check required top-level keys
        required_keys = ["entities", "dimensions", "facts", "relationships", "audit"]
        for key in required_keys:
            if key not in model:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Validate relationships specifically (this is where the error occurs)
        for idx, rel in enumerate(model.get("relationships", [])):
            # Check required fields
            required_rel_fields = ["from", "to", "cardinality", "type"]
            for field in required_rel_fields:
                if field not in rel:
                    logger.error(f"Relationship {idx} missing required field: {field}")
                    logger.error(f"Relationship data: {rel}")
                    return False
            
            # Validate cardinality values
            valid_cardinalities = ["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
            if rel["cardinality"] not in valid_cardinalities:
                logger.error(f"Invalid cardinality in relationship {idx}: {rel['cardinality']}")
                return False
        
        # Validate facts have measures
        for fact in model.get("facts", []):
            if "measures" not in fact or len(fact["measures"]) == 0:
                logger.warning(f"Fact {fact.get('name')} has no measures")
        
        logger.info("Schema validation passed")
        return True
    
    def _get_file_age_hours(self, filepath: str) -> float:
        """Get file age in hours."""
        if not os.path.exists(filepath):
            return float('inf')
        
        mtime = os.path.getmtime(filepath)
        age_seconds = datetime.now().timestamp() - mtime
        return age_seconds / 3600.0
    
    def invalidate_cache(self):
        """Force cache invalidation (e.g., when RDL changes detected)."""
        logger.info("Invalidating all caches...")
        
        if os.path.exists(self.discovery_cache_path):
            os.remove(self.discovery_cache_path)
        
        if os.path.exists(self.semantic_model_cache_path):
            os.remove(self.semantic_model_cache_path)
        
        logger.info("Cache invalidated")


def main():
    """Example usage"""
    import sys
    from dotenv import load_dotenv
    
    # Load config
    load_dotenv()
    config = {
        "DEPLOYMENT_NAME": os.getenv("DEPLOYMENT_NAME"),
        "API_VERSION": os.getenv("API_VERSION"),
        "AZURE_ENDPOINT": os.getenv("AZURE_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "DATABASE_CONNECTION_STRING": os.getenv("DATABASE_CONNECTION_STRING"),
        "SCHEMA_EXCLUSIONS": os.getenv("SCHEMA_EXCLUSIONS", "sys,information_schema"),
        "TABLE_EXCLUSIONS": os.getenv("TABLE_EXCLUSIONS", "temp_,test_,backup_"),
        "DISCOVERY_CACHE_HOURS": os.getenv("DISCOVERY_CACHE_HOURS", "168"),
        "SEMANTIC_CACHE_HOURS": os.getenv("SEMANTIC_CACHE_HOURS", "168"),
    }
    
    # Validate required config
    required = ["DEPLOYMENT_NAME", "API_VERSION", "AZURE_ENDPOINT", 
                "AZURE_OPENAI_API_KEY", "DATABASE_CONNECTION_STRING"]
    for key in required:
        if not config.get(key):
            print(f"ERROR: Missing required config: {key}")
            sys.exit(1)
    
    # Initialize pipeline
    pipeline = SemanticPipeline(config)
    
    # Get question from command line or use default
    question = sys.argv[1] if len(sys.argv) > 1 else "What are the total sales by customer?"
    
    # Run pipeline
    answer = pipeline.run_full_pipeline(question, force_refresh=False)
    
    # Print result
    print("\n=== ANSWER ===")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()