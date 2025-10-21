from typing import Dict, Any, Optional, Tuple
from connectors.dialect_registry import DialectRegistry
from discovery.discovery_service import DiscoveryService
from modeling.semantic_modeler import SemanticModeler
from qa.sql_answerer import SQLAnswerer
from validation.schema_validator import SchemaValidator
from llm.azure_client import AzureLLMClient
from caching.cache_manager import CacheManager
from normalization.sql_normalizer import SQLNormalizer
from utils.fingerprint import calculate_db_fingerprint
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class SemanticPipeline:
    """Main orchestration pipeline: Discovery → Modeling → Q&A."""
    
    def __init__(self):
        self.llm = AzureLLMClient()
        self.validator = SchemaValidator()
        self.normalizer = SQLNormalizer()
        self.cache = CacheManager(
            settings.discovery_cache_hours,
            settings.semantic_cache_hours
        )
        self.connector = None
        self.discovery_data = None
        self.semantic_model = None
    
    def initialize(self, bypass_cache: bool = False) -> Tuple[bool, str]:
        """
        Phase 1: Discovery
        
        Returns:
            (success, error_message)
        """
        try:
            # Connect to database
            logger.info("Connecting to database...")
            self.connector = DialectRegistry.get_connector(settings.database_connection_string)
            
            # Check cache
            fingerprint_key = settings.database_connection_string
            
            if not bypass_cache:
                cached_discovery = self.cache.get_discovery(fingerprint_key)
                if cached_discovery:
                    logger.info("Using cached discovery data")
                    self.discovery_data = cached_discovery
                    return (True, "")
            
            # Run discovery
            logger.info("Running database discovery...")
            discovery_service = DiscoveryService(self.connector)
            self.discovery_data = discovery_service.discover()
            
            # Validate discovery JSON
            valid, error = self.validator.validate(self.discovery_data, "discovery")
            if not valid:
                return (False, f"Discovery validation failed: {error}")
            
            # Cache it
            db_fingerprint = calculate_db_fingerprint(
                settings.database_connection_string,
                self.discovery_data
            )
            self.cache.set_discovery(db_fingerprint, self.discovery_data)
            
            logger.info(f"Discovery complete: {len(self.discovery_data.get('schemas', []))} schemas")
            return (True, "")
        
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return (False, str(e))
    
    def create_semantic_model(
        self, 
        domain_hints: str = "",
        bypass_cache: bool = False
    ) -> Tuple[bool, str]:
        """
        Phase 2: Semantic Modeling
        
        Returns:
            (success, error_message)
        """
        if not self.discovery_data:
            return (False, "Discovery data not available. Run initialize() first.")
        
        try:
            # Check cache
            model_key = calculate_db_fingerprint(
                settings.database_connection_string,
                self.discovery_data
            ) + f"__{domain_hints}"
            
            if not bypass_cache:
                cached_model = self.cache.get_semantic(model_key)
                if cached_model:
                    logger.info("Using cached semantic model")
                    self.semantic_model = cached_model
                    return (True, "")
            
            # Create model
            logger.info("Creating semantic model...")
            modeler = SemanticModeler(self.llm, self.validator)
            success, model, error = modeler.create_model(self.discovery_data, domain_hints)
            
            if not success:
                return (False, error)
            
            self.semantic_model = model
            
            # Cache it
            self.cache.set_semantic(model_key, model)
            
            logger.info("Semantic model created successfully")
            return (True, "")
        
        except Exception as e:
            logger.error(f"Modeling failed: {e}")
            return (False, str(e))
    
    def answer_question(self, question: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Phase 3: Question Answering
        
        Returns:
            (success, answer_json, error_message)
        """
        if not self.semantic_model:
            return (False, {}, "Semantic model not available. Run create_semantic_model() first.")
        
        try:
            logger.info(f"Answering question: {question}")
            answerer = SQLAnswerer(self.llm, self.validator, self.normalizer)
            
            success, answer, error = answerer.answer_question(
                question,
                self.semantic_model,
                self.discovery_data
            )
            
            if not success:
                return (False, {}, error)
            
            logger.info(f"Answer status: {answer.get('status')}")
            return (True, answer, "")
        
        except Exception as e:
            logger.error(f"Q&A failed: {e}")
            return (False, {}, str(e))
    
    def get_discovery_data(self) -> Optional[Dict[str, Any]]:
        """Return current discovery data."""
        return self.discovery_data
    
    def get_semantic_model(self) -> Optional[Dict[str, Any]]:
        """Return current semantic model."""
        return self.semantic_model
    
    def cleanup(self):
        """Close connections."""
        if self.connector:
            self.connector.close()