import hashlib
import json
import time
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
from cachetools import TTLCache

class CacheManager:
    """TTL-based caching for discovery and semantic models."""
    
    def __init__(self, discovery_ttl_hours: int, semantic_ttl_hours: int):
        # Use TTLCache with appropriate TTL
        self.discovery_cache = TTLCache(
            maxsize=100, 
            ttl=discovery_ttl_hours * 3600
        )
        self.semantic_cache = TTLCache(
            maxsize=100, 
            ttl=semantic_ttl_hours * 3600
        )
    
    @staticmethod
    def generate_fingerprint(data: Dict[str, Any]) -> str:
        """Generate cache key from data."""
        # Create deterministic hash
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def get_discovery(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached discovery data."""
        return self.discovery_cache.get(fingerprint)
    
    def set_discovery(self, fingerprint: str, data: Dict[str, Any]):
        """Cache discovery data."""
        self.discovery_cache[fingerprint] = data
    
    def get_semantic(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached semantic model."""
        return self.semantic_cache.get(fingerprint)
    
    def set_semantic(self, fingerprint: str, data: Dict[str, Any]):
        """Cache semantic model."""
        self.semantic_cache[fingerprint] = data
    
    def invalidate_discovery(self, fingerprint: str):
        """Remove discovery from cache."""
        if fingerprint in self.discovery_cache:
            del self.discovery_cache[fingerprint]
    
    def invalidate_semantic(self, fingerprint: str):
        """Remove semantic model from cache."""
        if fingerprint in self.semantic_cache:
            del self.semantic_cache[fingerprint]
    
    def clear_all(self):
        """Clear all caches."""
        self.discovery_cache.clear()
        self.semantic_cache.clear()