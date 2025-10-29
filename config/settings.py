"""
Configuration Management for GPT-5 Semantic Modeling & SQL Q&A System
Loads all settings from environment variables with validation and defaults
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get environment variable with validation
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: If True, raise error if not found
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(key, default)
    
    if required and not value:
        raise ValueError(f"Required environment variable '{key}' is not set")
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable"""
    value = os.getenv(key, str(default))
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {key}={value}, using default {default}")
        return default


def get_env_float(key: str, default: float) -> float:
    """Get float environment variable"""
    value = os.getenv(key, str(default))
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {key}={value}, using default {default}")
        return default


def get_env_list(key: str, default: Optional[List[str]] = None, separator: str = ',') -> List[str]:
    """Get list environment variable"""
    value = os.getenv(key, '')
    if not value and default:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]


# ============================================================================
# AZURE OPENAI CONFIGURATION
# ============================================================================

@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration"""
    deployment_name: str
    api_version: str
    endpoint: str
    api_key: str
    
    @classmethod
    def from_env(cls) -> 'AzureOpenAIConfig':
        return cls(
            deployment_name=get_env('DEPLOYMENT_NAME', required=True),
            api_version=get_env('API_VERSION', required=True),
            endpoint=get_env('AZURE_ENDPOINT', required=True),
            api_key=get_env('AZURE_OPENAI_API_KEY', required=True),
        )
    
    def validate(self):
        """Validate configuration"""
        if not self.endpoint.startswith('https://'):
            raise ValueError(f"Invalid Azure endpoint: {self.endpoint}")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required")


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    connection_string: str
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            connection_string=get_env('DATABASE_CONNECTION_STRING', required=True)
        )
    
    def validate(self):
        """Validate configuration"""
        if not self.connection_string:
            raise ValueError("Database connection string is required")


# ============================================================================
# FILE PATHS CONFIGURATION
# ============================================================================

@dataclass
class PathConfig:
    """File paths configuration"""
    rdl_path: Path
    cache_dir: Path
    log_dir: Path
    
    @classmethod
    def from_env(cls) -> 'PathConfig':
        config = cls(
            rdl_path=Path(get_env('RDL_PATH', './data_upload')),
            cache_dir=Path(get_env('CACHE_DIR', './cache')),
            log_dir=Path(get_env('LOG_DIR', './logs')),
        )
        
        # Create directories if they don't exist
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)
        if not config.rdl_path.exists():
            logger.warning(f"RDL path does not exist: {config.rdl_path}")
        
        return config


# ============================================================================
# DISCOVERY CONFIGURATION
# ============================================================================

@dataclass
class DiscoveryConfig:
    """Discovery phase configuration"""
    # Timeouts
    timeout: int
    cache_hours: int
    
    # Encoding
    utf8_encoding: bool
    
    # Exclusions
    schema_exclusions: List[str]
    table_exclusions: List[str]
    table_exclusion_patterns: List[str]
    
    # Concurrency
    max_workers: int
    
    @classmethod
    def from_env(cls) -> 'DiscoveryConfig':
        return cls(
            timeout=get_env_int('DISCOVERY_TIMEOUT', 300),
            cache_hours=get_env_int('DISCOVERY_CACHE_HOURS', 168),
            utf8_encoding=get_env_bool('UTF8_ENCODING', True),
            schema_exclusions=get_env_list('SCHEMA_EXCLUSIONS', 
                                          ['sys', 'information_schema', 'guest', 'INFORMATION_SCHEMA']),
            table_exclusions=get_env_list('TABLE_EXCLUSIONS',
                                         ['temp_', 'test_', 'backup_', 'old_', 'archive_', 'copy_']),
            table_exclusion_patterns=get_env_list('TABLE_EXCLUSION_PATTERNS',
                                                  ['.*_\\d{8}$', '.*_\\d{6}$', '.*_backup.*', 
                                                   '.*_archive.*', '.*_copy.*', '.*_old.*']),
            max_workers=get_env_int('CONCURRENCY_MAX_WORKERS', 10),
        )


# ============================================================================
# RELATIONSHIP DETECTION CONFIGURATION
# ============================================================================

@dataclass
class RelationshipDetectionConfig:
    """Optimized relationship detection configuration"""
    # Core settings
    enabled: bool
    strategy: str
    min_overlap_rate: float
    sample_size: int
    
    # Performance tuning
    max_workers: int
    timeout_per_comparison: int
    max_comparisons: int
    global_timeout: int
    
    # Filtering strategies
    prioritize_named_patterns: bool
    require_index_on_target: bool
    confidence_threshold: float
    
    @classmethod
    def from_env(cls) -> 'RelationshipDetectionConfig':
        return cls(
            enabled=get_env_bool('RELATIONSHIP_DETECTION_ENABLED', True),
            strategy=get_env('RELATIONSHIP_DETECTION_STRATEGY', 'smart_filter'),
            min_overlap_rate=get_env_float('RELATIONSHIP_MIN_OVERLAP_RATE', 0.80),
            sample_size=get_env_int('RELATIONSHIP_SAMPLE_SIZE', 100),
            max_workers=get_env_int('RELATIONSHIP_MAX_WORKERS', 10),
            timeout_per_comparison=get_env_int('RELATIONSHIP_TIMEOUT_PER_COMPARISON', 5),
            max_comparisons=get_env_int('RELATIONSHIP_MAX_COMPARISONS', 5000),
            global_timeout=get_env_int('RELATIONSHIP_GLOBAL_TIMEOUT', 300),
            prioritize_named_patterns=get_env_bool('RELATIONSHIP_PRIORITIZE_PATTERNS', True),
            require_index_on_target=get_env_bool('RELATIONSHIP_REQUIRE_INDEX', True),
            confidence_threshold=get_env_float('RELATIONSHIP_CONFIDENCE_THRESHOLD', 0.7),
        )
    
    def validate(self):
        """Validate configuration"""
        if self.strategy not in ['smart_filter', 'full', 'name_only']:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        if not 0.0 <= self.min_overlap_rate <= 1.0:
            raise ValueError(f"Invalid min_overlap_rate: {self.min_overlap_rate}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"Invalid confidence_threshold: {self.confidence_threshold}")


# ============================================================================
# SEMANTIC MODEL CONFIGURATION
# ============================================================================

@dataclass
class SemanticModelConfig:
    """Semantic model generation configuration"""
    # Cache
    cache_hours: int
    
    # Batch sizes for LLM processing
    entity_batch_size: int
    dimension_batch_size: int
    fact_batch_size: int
    
    # Assembly
    max_retries: int
    
    # Compression
    compression_strategy: str
    
    @classmethod
    def from_env(cls) -> 'SemanticModelConfig':
        return cls(
            cache_hours=get_env_int('SEMANTIC_CACHE_HOURS', 168),
            entity_batch_size=get_env_int('ENTITY_BATCH_SIZE', 2),
            dimension_batch_size=get_env_int('DIMENSION_BATCH_SIZE', 2),
            fact_batch_size=get_env_int('FACT_BATCH_SIZE', 1),
            max_retries=get_env_int('ASSEMBLY_MAX_RETRIES', 3),
            compression_strategy=get_env('COMPRESSION_STRATEGY', 'tldr'),
        )
    
    def validate(self):
        """Validate configuration"""
        if self.compression_strategy not in ['tldr', 'detailed', 'map_reduce', 'recap']:
            raise ValueError(f"Invalid compression_strategy: {self.compression_strategy}")


# ============================================================================
# QUERY EXECUTION CONFIGURATION
# ============================================================================

@dataclass
class QueryExecutionConfig:
    """Query execution defaults and limits"""
    default_row_limit: int
    max_row_limit: int
    timeout: int
    
    @classmethod
    def from_env(cls) -> 'QueryExecutionConfig':
        return cls(
            default_row_limit=get_env_int('DEFAULT_ROW_LIMIT', 10),
            max_row_limit=get_env_int('MAX_ROW_LIMIT', 1000),
            timeout=get_env_int('QUERY_TIMEOUT', 60),
        )
    
    def validate(self):
        """Validate configuration"""
        if self.default_row_limit > self.max_row_limit:
            raise ValueError(
                f"default_row_limit ({self.default_row_limit}) cannot exceed "
                f"max_row_limit ({self.max_row_limit})"
            )


# ============================================================================
# CONFIDENCE SCORING CONFIGURATION
# ============================================================================

@dataclass
class ConfidenceConfig:
    """Confidence scoring thresholds"""
    high_threshold: float
    medium_threshold: float
    low_threshold: float
    
    @classmethod
    def from_env(cls) -> 'ConfidenceConfig':
        return cls(
            high_threshold=get_env_float('CONFIDENCE_HIGH_THRESHOLD', 0.85),
            medium_threshold=get_env_float('CONFIDENCE_MEDIUM_THRESHOLD', 0.70),
            low_threshold=get_env_float('CONFIDENCE_LOW_THRESHOLD', 0.50),
        )
    
    def validate(self):
        """Validate configuration"""
        if not (self.low_threshold < self.medium_threshold < self.high_threshold):
            raise ValueError(
                f"Thresholds must be ordered: "
                f"low ({self.low_threshold}) < "
                f"medium ({self.medium_threshold}) < "
                f"high ({self.high_threshold})"
            )


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    format: str
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        return cls(
            level=get_env('LOG_LEVEL', 'INFO'),
            format=get_env('LOG_FORMAT', 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        )
    
    def configure(self):
        """Configure Python logging"""
        level = getattr(logging, self.level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format=self.format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )


# ============================================================================
# MASTER SETTINGS CLASS
# ============================================================================

@dataclass
class Settings:
    """Master settings container"""
    azure_openai: AzureOpenAIConfig
    database: DatabaseConfig
    paths: PathConfig
    discovery: DiscoveryConfig
    relationships: RelationshipDetectionConfig
    semantic_model: SemanticModelConfig
    query_execution: QueryExecutionConfig
    confidence: ConfidenceConfig
    logging: LoggingConfig
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Load all settings from environment variables"""
        return cls(
            azure_openai=AzureOpenAIConfig.from_env(),
            database=DatabaseConfig.from_env(),
            paths=PathConfig.from_env(),
            discovery=DiscoveryConfig.from_env(),
            relationships=RelationshipDetectionConfig.from_env(),
            semantic_model=SemanticModelConfig.from_env(),
            query_execution=QueryExecutionConfig.from_env(),
            confidence=ConfidenceConfig.from_env(),
            logging=LoggingConfig.from_env(),
        )
    
    def validate(self):
        """Validate all configurations"""
        self.azure_openai.validate()
        self.database.validate()
        self.relationships.validate()
        self.semantic_model.validate()
        self.query_execution.validate()
        self.confidence.validate()
    
    def summary(self) -> str:
        """Return a formatted summary of key settings"""
        return f"""
Configuration Summary:
=====================
Azure OpenAI:
  Deployment:        {self.azure_openai.deployment_name}
  API Version:       {self.azure_openai.api_version}

Database:
  Connection:        {self.database.connection_string[:50]}...

Paths:
  RDL Path:          {self.paths.rdl_path}
  Cache Dir:         {self.paths.cache_dir}
  Log Dir:           {self.paths.log_dir}

Discovery:
  Timeout:           {self.discovery.timeout}s
  Cache:             {self.discovery.cache_hours}h
  Max Workers:       {self.discovery.max_workers}

Relationship Detection:
  Enabled:           {self.relationships.enabled}
  Strategy:          {self.relationships.strategy}
  Sample Size:       {self.relationships.sample_size}
  Max Workers:       {self.relationships.max_workers}
  Max Comparisons:   {self.relationships.max_comparisons}
  Min Overlap:       {self.relationships.min_overlap_rate}

Semantic Model:
  Entity Batch:      {self.semantic_model.entity_batch_size}
  Dimension Batch:   {self.semantic_model.dimension_batch_size}
  Fact Batch:        {self.semantic_model.fact_batch_size}
  Compression:       {self.semantic_model.compression_strategy}

Query Execution:
  Default Limit:     {self.query_execution.default_row_limit}
  Max Limit:         {self.query_execution.max_row_limit}
  Timeout:           {self.query_execution.timeout}s

Confidence:
  High:              {self.confidence.high_threshold}
  Medium:            {self.confidence.medium_threshold}
  Low:               {self.confidence.low_threshold}

Logging:
  Level:             {self.logging.level}
=====================
        """


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get or create settings singleton
    
    Args:
        reload: If True, reload settings from environment
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None or reload:
        try:
            _settings = Settings.from_env()
            _settings.validate()
            _settings.logging.configure()
            
            logger.info("Configuration loaded successfully")
            logger.debug(_settings.summary())
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    return _settings


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_azure_openai_config() -> AzureOpenAIConfig:
    """Get Azure OpenAI configuration"""
    return get_settings().azure_openai


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return get_settings().database


def get_path_config() -> PathConfig:
    """Get paths configuration"""
    return get_settings().paths


def get_discovery_config() -> DiscoveryConfig:
    """Get discovery configuration"""
    return get_settings().discovery


def get_relationship_config() -> RelationshipDetectionConfig:
    """Get relationship detection configuration"""
    return get_settings().relationships


def get_semantic_model_config() -> SemanticModelConfig:
    """Get semantic model configuration"""
    return get_settings().semantic_model


def get_query_execution_config() -> QueryExecutionConfig:
    """Get query execution configuration"""
    return get_settings().query_execution


def get_confidence_config() -> ConfidenceConfig:
    """Get confidence configuration"""
    return get_settings().confidence


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize settings on import (can be disabled by setting env var)
if not get_env_bool('SKIP_SETTINGS_INIT', False):
    try:
        get_settings()
    except Exception as e:
        logger.warning(f"Failed to initialize settings on import: {e}")
        logger.warning("Settings will be loaded on first access")