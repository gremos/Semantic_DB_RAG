"""
Cache manager for discovery and semantic model results.
Uses file-based caching with TTL support.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from config.settings import Settings


class CacheManager:
    """Manage file-based caching for discovery and semantic model."""
    
    DISCOVERY_CACHE_FILE = 'discovery.json'
    SEMANTIC_CACHE_FILE = 'semantic_model.json'
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, cache_type: str) -> Path:
        """Get path for cache file."""
        if cache_type == 'discovery':
            return self.cache_dir / self.DISCOVERY_CACHE_FILE
        elif cache_type == 'semantic':
            return self.cache_dir / self.SEMANTIC_CACHE_FILE
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _is_cache_valid(self, cache_path: Path, ttl_hours: int) -> bool:
        """Check if cache file exists and is within TTL."""
        if not cache_path.exists():
            return False
        
        # Check TTL
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        ttl = timedelta(hours=ttl_hours)
        
        return datetime.now() - mtime < ttl
    
    def get_discovery(self) -> Optional[Dict[str, Any]]:
        """Get cached discovery data if valid."""
        cache_path = self._get_cache_path('discovery')
        
        if self._is_cache_valid(cache_path, self.settings.DISCOVERY_CACHE_HOURS):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        
        return None
    
    def set_discovery(self, data: Dict[str, Any]) -> None:
        """Cache discovery data."""
        cache_path = self._get_cache_path('discovery')
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_semantic_model(self) -> Optional[Dict[str, Any]]:
        """Get cached semantic model if valid."""
        cache_path = self._get_cache_path('semantic')
        
        if self._is_cache_valid(cache_path, self.settings.SEMANTIC_CACHE_HOURS):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        
        return None
    
    def set_semantic_model(self, data: Dict[str, Any]) -> None:
        """Cache semantic model data."""
        cache_path = self._get_cache_path('semantic')
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear_discovery(self) -> bool:
        """Clear discovery cache."""
        cache_path = self._get_cache_path('discovery')
        
        if cache_path.exists():
            cache_path.unlink()
            return True
        
        return False
    
    def clear_semantic_model(self) -> bool:
        """Clear semantic model cache."""
        cache_path = self._get_cache_path('semantic')
        
        if cache_path.exists():
            cache_path.unlink()
            return True
        
        return False
    
    def clear_all(self) -> bool:
        """Clear all caches."""
        discovery_cleared = self.clear_discovery()
        semantic_cleared = self.clear_semantic_model()
        
        return discovery_cleared or semantic_cleared