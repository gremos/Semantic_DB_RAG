"""
Relationship Detection Configuration
Optimized settings for fast and accurate foreign key relationship detection
"""
import os
from dataclasses import dataclass
from typing import List, Set


@dataclass
class RelationshipDetectionConfig:
    """Configuration for optimized relationship detection"""
    
    # Core detection settings
    enabled: bool = True
    strategy: str = "smart_filter"  # Options: full, smart_filter, name_only
    min_overlap_rate: float = 0.90  # Lowered from 0.80 to catch more candidates
    sample_size: int = 500  # Increased from 100 for better accuracy
    
    # Performance tuning
    max_workers: int = 10
    timeout_per_comparison: int = 5  # seconds
    max_comparisons: int = 5000  # Hard cap on total comparisons
    global_timeout: int = 300  # 5 minutes total timeout
    
    # Filtering strategies
    prioritize_named_patterns: bool = True
    require_index_on_target: bool = True
    confidence_threshold: float = 0.7  # For name-based pre-scoring
    
    # Type compatibility groups
    compatible_type_groups: List[Set[str]] = None
    
    # FK naming patterns (case-insensitive)
    fk_suffix_patterns: List[str] = None
    fk_infix_patterns: List[str] = None
    
    def __post_init__(self):
        """Initialize default patterns if not provided"""
        if self.compatible_type_groups is None:
            self.compatible_type_groups = [
                # Integer types (expanded)
                {'int', 'integer', 'bigint', 'smallint', 'tinyint', 'number', 'numeric', 'decimal'},
                # String types (expanded)
                {'varchar', 'char', 'nvarchar', 'nchar', 'text', 'ntext', 'string', 
                'character', 'character varying', 'varchar2'},
                # UUID/GUID types
                {'uuid', 'uniqueidentifier', 'guid'},
                # Date types
                {'date', 'datetime', 'datetime2', 'timestamp', 'smalldatetime', 'datetimeoffset'},
                # Binary types (sometimes used for IDs)
                {'binary', 'varbinary', 'image'},
            ]
        
        if self.fk_suffix_patterns is None:
            self.fk_suffix_patterns = [
                'id', 'key', 'fk', 'code', 'num', 'number', 
                '_id', '_key', '_fk', '_code', '_num'
            ]
        
        if self.fk_infix_patterns is None:
            self.fk_infix_patterns = [
                '_id_', '_key_', '_fk_', 'id_', 'key_', 'fk_'
            ]
    
    @classmethod
    def from_env(cls) -> 'RelationshipDetectionConfig':
        """Load configuration from environment variables"""
        return cls(
            enabled=os.getenv('RELATIONSHIP_DETECTION_ENABLED', 'true').lower() == 'true',
            strategy=os.getenv('RELATIONSHIP_DETECTION_STRATEGY', 'smart_filter'),
            min_overlap_rate=float(os.getenv('RELATIONSHIP_MIN_OVERLAP_RATE', '0.80')),
            sample_size=int(os.getenv('RELATIONSHIP_SAMPLE_SIZE', '100')),
            max_workers=int(os.getenv('RELATIONSHIP_MAX_WORKERS', '10')),
            timeout_per_comparison=int(os.getenv('RELATIONSHIP_TIMEOUT_PER_COMPARISON', '5')),
            max_comparisons=int(os.getenv('RELATIONSHIP_MAX_COMPARISONS', '5000')),
            global_timeout=int(os.getenv('RELATIONSHIP_GLOBAL_TIMEOUT', '300')),
            prioritize_named_patterns=os.getenv('RELATIONSHIP_PRIORITIZE_PATTERNS', 'true').lower() == 'true',
            require_index_on_target=os.getenv('RELATIONSHIP_REQUIRE_INDEX', 'true').lower() == 'true',
            confidence_threshold=float(os.getenv('RELATIONSHIP_CONFIDENCE_THRESHOLD', '0.7')),
        )
    
    def get_type_group(self, sql_type: str) -> int:
        """Get the type group index for a SQL type"""
        sql_type_lower = sql_type.lower().split('(')[0]  # Strip precision/scale
        for idx, group in enumerate(self.compatible_type_groups):
            if any(t in sql_type_lower for t in group):
                return idx
        return -1  # No group found
    
    def types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two SQL types can be used in a foreign key relationship"""
        group1 = self.get_type_group(type1)
        group2 = self.get_type_group(type2)
        return group1 >= 0 and group1 == group2