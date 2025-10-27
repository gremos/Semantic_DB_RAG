"""
Environment settings loader using pydantic-settings.
Fails fast if required environment variables are missing.
"""

import os
import sys
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Azure OpenAI Configuration
    DEPLOYMENT_NAME: str = Field(..., description="Azure OpenAI deployment name")
    API_VERSION: str = Field(..., description="Azure OpenAI API version")
    AZURE_ENDPOINT: str = Field(..., description="Azure OpenAI endpoint URL")
    AZURE_OPENAI_API_KEY: str = Field(..., description="Azure OpenAI API key")
    
    # Database Configuration
    DATABASE_CONNECTION_STRING: str = Field(..., description="SQLAlchemy connection string")
    
    # Paths
    RDL_PATH: str = Field(default="./data_upload", description="Path to RDL files")
    CACHE_DIR: str = Field(default="./cache", description="Cache directory")
    LOG_DIR: str = Field(default="./logs", description="Log directory")
    
    # Encoding & Filtering
    UTF8_ENCODING: bool = Field(default=True, description="Use UTF-8 encoding")
    SCHEMA_EXCLUSIONS: str = Field(
        default="sys,information_schema",
        description="Comma-separated schemas to exclude"
    )
    TABLE_EXCLUSIONS: str = Field(
        default="temp_,test_,backup_,old_",
        description="Comma-separated table prefixes to exclude"
    )
    TABLE_EXCLUSION_PATTERNS: str = Field(
        default=".*_\\d{8}$,.*_\\d{6}$,.*_backup.*,.*_archive.*,.*_copy.*,.*_old.*",
        description="Regex patterns for table exclusions"
    )
    
    # Cache TTL (hours)
    DISCOVERY_CACHE_HOURS: int = Field(default=168, description="Discovery cache TTL in hours")
    SEMANTIC_CACHE_HOURS: int = Field(default=168, description="Semantic model cache TTL in hours")
    
    # Connection timeout (seconds)
    DISCOVERY_TIMEOUT: int = Field(default=300, description="Discovery timeout in seconds")
    
    # Batching
    ENTITY_BATCH_SIZE: int = Field(default=2, description="Batch size for entity processing")
    DIMENSION_BATCH_SIZE: int = Field(default=2, description="Batch size for dimension processing")
    FACT_BATCH_SIZE: int = Field(default=1, description="Batch size for fact processing")
    
    # Compression strategy
    COMPRESSION_STRATEGY: str = Field(
        default="tldr",
        description="Compression strategy: detailed, tldr, map_reduce, recap"
    )
    
    # Assembly retries
    ASSEMBLY_MAX_RETRIES: int = Field(default=3, description="Max retries for model assembly")
    
    # Query execution defaults
    DEFAULT_ROW_LIMIT: int = Field(default=10, description="Default row limit for queries")
    MAX_ROW_LIMIT: int = Field(default=1000, description="Maximum row limit allowed")
    QUERY_TIMEOUT_SEC: int = Field(default=60, description="Query execution timeout")
    
    # Confidence thresholds
    CONFIDENCE_HIGH: float = Field(default=0.85, description="High confidence threshold")
    CONFIDENCE_MEDIUM: float = Field(default=0.70, description="Medium confidence threshold")
    CONFIDENCE_LOW: float = Field(default=0.50, description="Low confidence threshold")
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )
    
    @field_validator('COMPRESSION_STRATEGY')
    @classmethod
    def validate_compression_strategy(cls, v: str) -> str:
        """Validate compression strategy is one of allowed values."""
        allowed = ['detailed', 'tldr', 'map_reduce', 'recap']
        if v not in allowed:
            raise ValueError(f"COMPRESSION_STRATEGY must be one of {allowed}")
        return v
    
    @property
    def schema_exclusions_list(self) -> List[str]:
        """Get schema exclusions as list."""
        return [s.strip() for s in self.SCHEMA_EXCLUSIONS.split(',') if s.strip()]
    
    @property
    def table_exclusions_list(self) -> List[str]:
        """Get table exclusions as list."""
        return [t.strip() for t in self.TABLE_EXCLUSIONS.split(',') if t.strip()]
    
    @property
    def table_exclusion_patterns_list(self) -> List[str]:
        """Get table exclusion patterns as list."""
        return [p.strip() for p in self.TABLE_EXCLUSION_PATTERNS.split(',') if p.strip()]
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for dir_path in [self.RDL_PATH, self.CACHE_DIR, self.LOG_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def validate_required_fields(self) -> None:
        """
        Validate that all required fields are set.
        Fails fast with descriptive error if any are missing.
        """
        missing = []
        
        for field_name, field_info in self.model_fields.items():
            if field_info.is_required():
                value = getattr(self, field_name, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing.append(field_name)
        
        if missing:
            print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
            print("Please set these variables in your .env file")
            sys.exit(1)


# Create a singleton instance
def get_settings() -> Settings:
    """Get settings instance and ensure directories exist."""
    settings = Settings()
    settings.validate_required_fields()
    settings.ensure_directories()
    return settings