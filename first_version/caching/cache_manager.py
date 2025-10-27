import hashlib
import json
import pickle
import os
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """TTL-based caching with persistent disk storage."""
    
    def __init__(self, discovery_ttl_hours: int, semantic_ttl_hours: int):
        self.discovery_ttl = timedelta(hours=discovery_ttl_hours)
        self.semantic_ttl = timedelta(hours=semantic_ttl_hours)
        
        # Create cache directory
        self.cache_dir = ".cache"
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cache directory initialized: {os.path.abspath(self.cache_dir)}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise
        
        # Cache files
        self.discovery_cache_file = os.path.join(self.cache_dir, "discovery_cache.pkl")
        self.semantic_cache_file = os.path.join(self.cache_dir, "semantic_cache.pkl")
        
        # Load existing caches
        logger.info("Loading existing caches...")
        self.discovery_cache = self._load_cache(self.discovery_cache_file)
        self.semantic_cache = self._load_cache(self.semantic_cache_file)
        
        logger.info(f"Discovery cache entries: {len(self.discovery_cache)}")
        logger.info(f"Semantic cache entries: {len(self.semantic_cache)}")
    
    def _load_cache(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    cache = pickle.load(f)
                    logger.info(f"Loaded cache from {filepath}: {len(cache)} entries")
                    return cache
            except Exception as e:
                logger.warning(f"Failed to load cache from {filepath}: {e}")
                return {}
        else:
            logger.info(f"No existing cache file: {filepath}")
            return {}
    
    def _save_cache(self, cache: Dict[str, Dict[str, Any]], filepath: str):
        """Save cache to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write to temp file first (atomic write)
            temp_file = filepath + ".tmp"
            with open(temp_file, 'wb') as f:
                pickle.dump(cache, f)
            
            # Rename to actual file (atomic on Unix)
            os.replace(temp_file, filepath)
            
            logger.info(f"Saved cache to {filepath}: {len(cache)} entries, {os.path.getsize(filepath):,} bytes")
        except Exception as e:
            logger.error(f"Failed to save cache to {filepath}: {e}", exc_info=True)
    
    @staticmethod
    def generate_fingerprint(data: str) -> str:
        """Generate cache key from string."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get_discovery(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached discovery data if not expired."""
        logger.debug(f"Looking up discovery cache: {fingerprint[:16]}...")
        
        entry = self.discovery_cache.get(fingerprint)
        if not entry:
            logger.info("Discovery cache miss")
            return None
        
        # Check expiration
        if datetime.now() > entry['expires']:
            logger.info("Discovery cache expired")
            del self.discovery_cache[fingerprint]
            self._save_cache(self.discovery_cache, self.discovery_cache_file)
            return None
        
        logger.info(f"Discovery cache HIT (created: {entry['created']})")
        return entry['data']
    
    def set_discovery(self, fingerprint: str, data: Dict[str, Any]):
        """Cache discovery data with expiration."""
        logger.info(f"Caching discovery data: {fingerprint[:16]}...")
        
        self.discovery_cache[fingerprint] = {
            'data': data,
            'expires': datetime.now() + self.discovery_ttl,
            'created': datetime.now()
        }
        
        self._save_cache(self.discovery_cache, self.discovery_cache_file)
    
    def get_semantic(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached semantic model if not expired."""
        logger.debug(f"Looking up semantic cache: {fingerprint[:16]}...")
        
        entry = self.semantic_cache.get(fingerprint)
        if not entry:
            logger.info("Semantic model cache miss")
            return None
        
        # Check expiration
        if datetime.now() > entry['expires']:
            logger.info("Semantic model cache expired")
            del self.semantic_cache[fingerprint]
            self._save_cache(self.semantic_cache, self.semantic_cache_file)
            return None
        
        logger.info(f"Semantic model cache HIT (created: {entry['created']})")
        return entry['data']
    
    def set_semantic(self, fingerprint: str, data: Dict[str, Any]):
        """Cache semantic model with expiration."""
        logger.info(f"Caching semantic model: {fingerprint[:16]}...")
        
        self.semantic_cache[fingerprint] = {
            'data': data,
            'expires': datetime.now() + self.semantic_ttl,
            'created': datetime.now()
        }
        
        self._save_cache(self.semantic_cache, self.semantic_cache_file)
    
    def invalidate_discovery(self, fingerprint: str):
        """Remove discovery from cache."""
        if fingerprint in self.discovery_cache:
            logger.info(f"Invalidating discovery cache: {fingerprint[:16]}")
            del self.discovery_cache[fingerprint]
            self._save_cache(self.discovery_cache, self.discovery_cache_file)
    
    def invalidate_semantic(self, fingerprint: str):
        """Remove semantic model from cache."""
        if fingerprint in self.semantic_cache:
            logger.info(f"Invalidating semantic cache: {fingerprint[:16]}")
            del self.semantic_cache[fingerprint]
            self._save_cache(self.semantic_cache, self.semantic_cache_file)
    
    def clear_all(self):
        """Clear all caches."""
        logger.info("Clearing all caches")
        
        self.discovery_cache.clear()
        self.semantic_cache.clear()
        
        # Delete cache files
        try:
            if os.path.exists(self.discovery_cache_file):
                os.remove(self.discovery_cache_file)
                logger.info(f"Deleted {self.discovery_cache_file}")
            
            if os.path.exists(self.semantic_cache_file):
                os.remove(self.semantic_cache_file)
                logger.info(f"Deleted {self.semantic_cache_file}")
        except Exception as e:
            logger.error(f"Error clearing cache files: {e}")