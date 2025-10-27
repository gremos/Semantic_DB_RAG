"""
Orchestration pipeline that ensures semantic model as single source of truth
"""
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SemanticPipeline:
    """
    Orchestrates Discovery → Semantic Model → Q&A
    Ensures semantic model is single source of truth
    """
    
    def __init__(
        self,
        db_connection_string: str,
        llm_client: Any,
        cache_dir: Path,
        discovery_cache_hours: int = 168,
        semantic_cache_hours: int = 168
    ):
        self.db_connection_string = db_connection_string
        self.llm_client = llm_client
        self.cache_dir = cache_dir
        self.discovery_cache_hours = discovery_cache_hours
        self.semantic_cache_hours = semantic_cache_hours
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Main entry point - execute full pipeline
        """
        
        # Phase 1: Discovery
        logger.info("=== PHASE 1: DISCOVERY ===")
        discovery_json = self._get_or_create_discovery()
        
        # Phase 2: Semantic Model
        logger.info("=== PHASE 2: SEMANTIC MODELING ===")
        semantic_model = self._get_or_create_semantic_model(discovery_json)
        
        # Phase 3: Q&A
        logger.info("=== PHASE 3: QUESTION ANSWERING ===")
        answer = self._answer_question(question, semantic_model)
        
        return answer
    
    def _get_or_create_discovery(self) -> Dict[str, Any]:
        """
        Get cached discovery or create new one
        Cache key is based on database fingerprint
        """
        
        # Compute cache key from connection string
        cache_key = self._compute_discovery_cache_key()
        cache_file = self.cache_dir / f"discovery_{cache_key}.json"
        
        # Check cache
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=self.discovery_cache_hours):
                logger.info("Using cached discovery JSON")
                with open(cache_file, "r") as f:
                    return json.load(f)
            else:
                logger.info(f"Discovery cache expired (age: {cache_age})")
        
        # Run discovery
        logger.info("Running discovery on database")
        from discovery.schema_inspector import SchemaInspector
        
        inspector = SchemaInspector(self.db_connection_string)
        discovery_json = inspector.discover()
        
        # Cache it
        with open(cache_file, "w") as f:
            json.dump(discovery_json, f, indent=2, default=str)
        
        logger.info(f"Discovery complete - cached to {cache_file}")
        return discovery_json
    
    def _get_or_create_semantic_model(self, discovery_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cached semantic model or create new one
        Cache key is based on discovery fingerprint
        """
        
        # Compute discovery fingerprint
        discovery_fingerprint = self._compute_discovery_fingerprint(discovery_json)
        cache_file = self.cache_dir / f"semantic_model_{discovery_fingerprint}.json"
        
        # Check cache
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=self.semantic_cache_hours):
                logger.info("Using cached semantic model")
                with open(cache_file, "r") as f:
                    return json.load(f)
            else:
                logger.info(f"Semantic model cache expired (age: {cache_age})")
        
        # Build semantic model
        logger.info("Building semantic model via LLM")
        from modeling.semantic_builder import SemanticModelBuilder
        
        builder = SemanticModelBuilder(discovery_json, self.llm_client)
        semantic_model = builder.build()
        
        # CRITICAL: Validate model completeness
        validation_result = self._validate_model_completeness(semantic_model)
        if not validation_result["is_complete"]:
            logger.warning(f"Semantic model incomplete: {validation_result['missing']}")
            # Add warnings to model
            semantic_model["warnings"] = validation_result["missing"]
        
        # Cache it
        with open(cache_file, "w") as f:
            json.dump(semantic_model, f, indent=2, default=str)
        
        logger.info(f"Semantic model complete - cached to {cache_file}")
        return semantic_model
    
    def _answer_question(self, question: str, semantic_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer question using semantic model as AUTHORITY
        """
        from qa.question_handler import QuestionHandler
        
        handler = QuestionHandler(semantic_model, self.llm_client)
        answer = handler.answer(question)
        
        return answer
    
    def _validate_model_completeness(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that semantic model has all required metadata
        to serve as single source of truth
        """
        
        validation = {
            "is_complete": True,
            "missing": []
        }
        
        # Check entities have column metadata
        for entity in model.get("entities", []):
            if not entity.get("columns"):
                validation["is_complete"] = False
                validation["missing"].append(f"Entity '{entity['name']}' missing column metadata")
        
        # Check facts have column metadata and filter columns
        for fact in model.get("facts", []):
            if not fact.get("columns"):
                validation["is_complete"] = False
                validation["missing"].append(f"Fact '{fact['name']}' missing column metadata")
            
            if not fact.get("filter_columns"):
                validation["missing"].append(f"Fact '{fact['name']}' missing filter column metadata (may limit Q&A)")
            
            # Check measures have column stats
            for measure in fact.get("measures", []):
                if not measure.get("column_stats"):
                    validation["missing"].append(f"Measure '{measure['name']}' missing column statistics")
        
        # Check negative findings exist
        if not model.get("negative_findings"):
            validation["is_complete"] = False
            validation["missing"].append("Missing negative findings (what doesn't exist in database)")
        
        return validation
    
    def _compute_discovery_cache_key(self) -> str:
        """
        Compute cache key from connection string
        """
        # Hash connection string (without password for security)
        # In production, extract server/database only
        hash_input = self.db_connection_string[:100]  # Truncate for safety
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _compute_discovery_fingerprint(self, discovery_json: Dict[str, Any]) -> str:
        """
        Compute fingerprint from discovery JSON
        Used to invalidate semantic model cache when discovery changes
        """
        fingerprint_data = []
        
        for schema in discovery_json.get("schemas", []):
            for table in schema.get("tables", []):
                # Include table name + rowcount + column count
                col_count = len(table.get("columns", []))
                row_count = table.get("rowcount_sample", 0)
                fingerprint_data.append(f"{schema['name']}.{table['name']}:{col_count}:{row_count}")
        
        fingerprint_str = "|".join(sorted(fingerprint_data))
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    def invalidate_caches(self):
        """
        Force invalidate all caches
        Useful when database schema changes
        """
        logger.info("Invalidating all caches")
        
        for cache_file in self.cache_dir.glob("discovery_*.json"):
            cache_file.unlink()
            logger.info(f"Deleted {cache_file}")
        
        for cache_file in self.cache_dir.glob("semantic_model_*.json"):
            cache_file.unlink()
            logger.info(f"Deleted {cache_file}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of current semantic model for debugging
        """
        
        # Try to load cached model
        cached_models = list(self.cache_dir.glob("semantic_model_*.json"))
        if not cached_models:
            return {"error": "No semantic model cached"}
        
        # Load most recent
        latest_model_file = max(cached_models, key=lambda p: p.stat().st_mtime)
        with open(latest_model_file, "r") as f:
            model = json.load(f)
        
        summary = {
            "cache_file": str(latest_model_file),
            "cache_age_hours": (datetime.now() - datetime.fromtimestamp(latest_model_file.stat().st_mtime)).total_seconds() / 3600,
            "entity_count": len(model.get("entities", [])),
            "dimension_count": len(model.get("dimensions", [])),
            "fact_count": len(model.get("facts", [])),
            "relationship_count": len(model.get("relationships", [])),
            "metric_count": len(model.get("metrics", [])),
            "facts": []
        }
        
        # Summarize facts (most important for Q&A)
        for fact in model.get("facts", []):
            fact_summary = {
                "name": fact.get("name"),
                "source": fact.get("source"),
                "measure_count": len(fact.get("measures", [])),
                "filter_column_count": len(fact.get("filter_columns", [])),
                "measures": [m.get("name") for m in fact.get("measures", [])],
                "filter_columns": [f.get("name") for f in fact.get("filter_columns", [])]
            }
            summary["facts"].append(fact_summary)
        
        return summary
    
    def explain_refusal(self, question: str, answer: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of why a question was refused
        Useful for debugging
        """
        
        if answer.get("status") != "refuse":
            return "Question was not refused"
        
        refusal = answer.get("refusal", {})
        
        explanation = f"""
        Question: {question}
        
        Status: REFUSED
        
        Reason: {refusal.get("reason", "Unknown")}
        
        Missing from semantic model:
        {json.dumps(refusal.get("missing", []), indent=2)}
        
        Ambiguities detected:
        {json.dumps(refusal.get("ambiguities", []), indent=2)}
        
        Clarifying questions:
        {json.dumps(refusal.get("clarifying_questions", []), indent=2)}
        
        To fix this:
        1. Check if semantic model has the required entities/measures/filters
        2. Run get_model_summary() to see what's available
        3. If model is incomplete, invalidate caches and rebuild
        4. If user question is ambiguous, rephrase with more specifics
        """
        
        return explanation.strip()


def create_pipeline_from_env() -> SemanticPipeline:
    """
    Factory method to create pipeline from environment variables
    """
    import os
    from pathlib import Path
    
    # Load environment
    db_connection_string = os.getenv("DATABASE_CONNECTION_STRING")
    if not db_connection_string:
        raise ValueError("DATABASE_CONNECTION_STRING environment variable required")
    
    # Azure OpenAI config
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-5-mini")
    
    if not azure_endpoint or not azure_api_key:
        raise ValueError("AZURE_ENDPOINT and AZURE_OPENAI_API_KEY environment variables required")
    
    # Create LLM client
    from langchain_openai import AzureChatOpenAI
    
    llm_client = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=deployment_name,
        api_version="2025-01-01-preview"
    )
    
    # Cache config
    discovery_cache_hours = int(os.getenv("DISCOVERY_CACHE_HOURS", "168"))
    semantic_cache_hours = int(os.getenv("SEMANTIC_CACHE_HOURS", "168"))
    cache_dir = Path(os.getenv("CACHE_DIR", "./cache"))
    
    return SemanticPipeline(
        db_connection_string=db_connection_string,
        llm_client=llm_client,
        cache_dir=cache_dir,
        discovery_cache_hours=discovery_cache_hours,
        semantic_cache_hours=semantic_cache_hours
    )