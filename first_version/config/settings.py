import os
from typing import List
from dotenv import load_dotenv

class Settings:
    """Environment configuration with fail-fast validation."""

    # Cache TTL (Time To Live) in hours
    discovery_cache_hours: int = 168  # 7 days
    semantic_cache_hours: int = 168   # 7 days
    entity_batch_size: int = int(os.getenv('ENTITY_BATCH_SIZE', '3'))     # Was 10
    dimension_batch_size: int = int(os.getenv('DIMENSION_BATCH_SIZE', '3')) # Was 10  
    fact_batch_size: int = int(os.getenv('FACT_BATCH_SIZE', '1'))        # Was 5
    
    # Assembly retry settings
    assembly_max_retries: int = int(os.getenv('ASSEMBLY_MAX_RETRIES', '2'))
    
    def __init__(self):
        load_dotenv()
        self._validate_required()
        
    def _validate_required(self):
        """Fail fast if required env vars are missing."""
        required = [
            'DEPLOYMENT_NAME',
            'API_VERSION',
            'AZURE_ENDPOINT',
            'AZURE_OPENAI_API_KEY',
            'DATABASE_CONNECTION_STRING'
        ]
        
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    @property
    def deployment_name(self) -> str:
        return os.getenv('DEPLOYMENT_NAME')
    
    @property
    def api_version(self) -> str:
        return os.getenv('API_VERSION', '2024-02-15-preview')
    
    @property
    def azure_endpoint(self) -> str:
        return os.getenv('AZURE_ENDPOINT')
    
    @property
    def azure_api_key(self) -> str:
        return os.getenv('AZURE_OPENAI_API_KEY')
    
    @property
    def database_connection_string(self) -> str:
        return os.getenv('DATABASE_CONNECTION_STRING')
    
    @property
    def utf8_encoding(self) -> bool:
        return os.getenv('UTF8_ENCODING', 'true').lower() == 'true'
    
    @property
    def schema_exclusions(self) -> List[str]:
        return os.getenv('SCHEMA_EXCLUSIONS', 'sys,information_schema').split(',')
    
    @property
    def table_exclusions(self) -> List[str]:
        return os.getenv('TABLE_EXCLUSIONS', 'temp_,test_,backup_,old_').split(',')
    
    @property
    def discovery_cache_hours(self) -> int:
        return int(os.getenv('DISCOVERY_CACHE_HOURS', '168'))
    
    @property
    def semantic_cache_hours(self) -> int:
        return int(os.getenv('SEMANTIC_CACHE_HOURS', '168'))
    
    @property
    def table_exclusions(self) -> List[str]:
        """Patterns for table names to exclude."""
        return os.getenv(
            'TABLE_EXCLUSIONS', 
            'temp_,test_,backup_,old_,archive_,copy_,bak_,_copy,_backup,_old,_archive,_bak,_staging,staging_'
        ).split(',')

    @property
    def table_name_patterns_to_exclude(self) -> List[str]:
        """Regex patterns for excluding similar tables."""
        default_patterns = r'.*_\d{8}$,.*_\d{6}$,.*_backup.*,.*_archive.*,.*_copy.*,.*_old.*'
        return os.getenv('TABLE_EXCLUSION_PATTERNS', default_patterns).split(',')


settings = Settings()