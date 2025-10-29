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
        self.cache_dir = self._resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- internal helpers (compat with old & new Settings) ----------

    def _resolve_cache_dir(self) -> Path:
        # Preferred (new): settings.paths.cache_dir
        if hasattr(self.settings, "paths") and hasattr(self.settings.paths, "cache_dir"):
            return Path(self.settings.paths.cache_dir)

        # Legacy: settings.cache_dir
        if hasattr(self.settings, "cache_dir"):
            return Path(getattr(self.settings, "cache_dir"))

        raise AttributeError(
            "No cache_dir found on Settings. Expected settings.paths.cache_dir or settings.cache_dir."
        )

    def _discovery_ttl_hours(self) -> int:
        # Preferred (new): settings.discovery.cache_hours
        if hasattr(self.settings, "discovery") and hasattr(self.settings.discovery, "cache_hours"):
            return int(self.settings.discovery.cache_hours)

        # Legacy: settings.DISCOVERY_CACHE_HOURS
        if hasattr(self.settings, "DISCOVERY_CACHE_HOURS"):
            return int(getattr(self.settings, "DISCOVERY_CACHE_HOURS"))

        # Sensible default
        return 24

    def _semantic_ttl_hours(self) -> int:
        # Preferred (new): settings.semantic.cache_hours (if you have it)
        if hasattr(self.settings, "semantic") and hasattr(self.settings.semantic, "cache_hours"):
            return int(self.settings.semantic.cache_hours)

        # Legacy: settings.SEMANTIC_CACHE_HOURS
        if hasattr(self.settings, "SEMANTIC_CACHE_HOURS"):
            return int(getattr(self.settings, "SEMANTIC_CACHE_HOURS"))

        # Fallback to discovery TTL if semantic TTL isn't defined
        return self._discovery_ttl_hours()

    def _get_cache_path(self, cache_type: str) -> Path:
        if cache_type == 'discovery':
            return self.cache_dir / self.DISCOVERY_CACHE_FILE
        elif cache_type == 'semantic':
            return self.cache_dir / self.SEMANTIC_CACHE_FILE
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

    def _is_cache_valid(self, cache_path: Path, ttl_hours: int) -> bool:
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=ttl_hours)

    # ------------------------------- API -----------------------------------

    def get_discovery(self) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path('discovery')
        if self._is_cache_valid(cache_path, self._discovery_ttl_hours()):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def set_discovery(self, data: Dict[str, Any]) -> None:
        cache_path = self._get_cache_path('discovery')
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_semantic_model(self) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path('semantic')
        if self._is_cache_valid(cache_path, self._semantic_ttl_hours()):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def set_semantic_model(self, data: Dict[str, Any]) -> None:
        cache_path = self._get_cache_path('semantic')
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_discovery(self) -> bool:
        cache_path = self._get_cache_path('discovery')
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear_semantic_model(self) -> bool:
        cache_path = self._get_cache_path('semantic')
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear_all(self) -> bool:
        discovery_cleared = self.clear_discovery()
        semantic_cleared = self.clear_semantic_model()
        return discovery_cleared or semantic_cleared
