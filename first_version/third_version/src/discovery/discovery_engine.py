"""
Discovery Engine - Main orchestrator for Phase 1.
Coordinates introspection, sampling, relationship detection, and caching.
"""

import json
import hashlib
from typing import Dict, Any
from pathlib import Path

from config.settings import Settings
from src.discovery.introspector import DatabaseIntrospector
from src.discovery.sampler import DataSampler
from src.discovery.relationship_detector import RelationshipDetector
from src.discovery.rdl_parser import RDLParser
from src.utils.cache import CacheManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DiscoveryEngine:
    """Main engine for database discovery (Phase 1)."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_manager = CacheManager(settings)
        
    def _generate_fingerprint(self) -> str:
        """
        Generate a fingerprint for the database connection.
        Used to detect if the database has changed.
        """
        # Use connection string as basis (without password)
        conn_str = self.settings.DATABASE_CONNECTION_STRING
        # Remove password from connection string for fingerprint
        conn_str_safe = conn_str.split('@')[-1] if '@' in conn_str else conn_str
        
        fingerprint_data = f"{conn_str_safe}:{self.settings.SCHEMA_EXCLUSIONS}:{self.settings.TABLE_EXCLUSIONS}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    def discover(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run full discovery process.
        
        Steps:
        1. Check cache
        2. Introspect database schema
        3. Sample data and collect statistics
        4. Detect implicit relationships
        5. Parse RDL files
        6. Cache results
        
        Args:
            force_refresh: If True, bypass cache and force fresh discovery
            
        Returns:
            Discovery JSON structure
        """
        logger.info("Starting discovery process...")
        
        # Check cache unless force refresh
        if not force_refresh:
            cached = self.cache_manager.get_discovery()
            if cached:
                logger.info("Using cached discovery results")
                return cached
        
        # Generate database fingerprint
        fingerprint = self._generate_fingerprint()
        logger.info(f"Database fingerprint: {fingerprint}")
        
        # Step 1: Introspect database
        logger.info("Step 1/4: Introspecting database schema...")
        with DatabaseIntrospector(self.settings) as introspector:
            discovery_data = introspector.introspect_full()
        
        logger.info(f"Found {len(discovery_data['schemas'])} schemas")
        
        # Step 2: Sample data and collect statistics
        logger.info("Step 2/4: Sampling data and collecting statistics...")
        with DataSampler(self.settings) as sampler:
            total_tables = 0
            for schema in discovery_data['schemas']:
                for table in schema['tables']:
                    sampler.enrich_table_with_samples(table, schema['name'])
                    total_tables += 1
                    logger.debug(f"Sampled {schema['name']}.{table['name']}")
        
        logger.info(f"Sampled {total_tables} tables")
        
        # Step 3: Detect implicit relationships
        logger.info("Step 3/4: Detecting implicit relationships...")
        with RelationshipDetector(self.settings) as detector:
            inferred_relationships = detector.detect_relationships(discovery_data)
            discovery_data['inferred_relationships'] = inferred_relationships
        
        logger.info(f"Found {len(inferred_relationships)} inferred relationships")
        
        # Step 4: Parse RDL files
        logger.info("Step 4/4: Parsing RDL files...")
        rdl_parser = RDLParser(self.settings)
        rdl_assets = rdl_parser.parse_all_rdl_files()
        
        # Combine with other named assets
        discovery_data['named_assets'] = rdl_assets
        logger.info(f"Parsed {len(rdl_assets)} RDL files")
        
        # Add metadata
        discovery_data['fingerprint'] = fingerprint
        discovery_data['discovery_version'] = '1.0'
        
        # Cache results
        logger.info("Caching discovery results...")
        self.cache_manager.set_discovery(discovery_data)
        
        logger.info("Discovery process completed successfully")
        return discovery_data
    
    def get_discovery_summary(self, discovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of discovery results."""
        total_tables = sum(
            len(schema.get('tables', [])) 
            for schema in discovery_data.get('schemas', [])
        )
        
        total_columns = sum(
            len(table.get('columns', []))
            for schema in discovery_data.get('schemas', [])
            for table in schema.get('tables', [])
        )
        
        total_relationships = len(discovery_data.get('inferred_relationships', []))
        total_assets = len(discovery_data.get('named_assets', []))
        
        return {
            'database': discovery_data.get('database', {}),
            'schemas': len(discovery_data.get('schemas', [])),
            'tables': total_tables,
            'columns': total_columns,
            'relationships': total_relationships,
            'named_assets': total_assets,
            'fingerprint': discovery_data.get('fingerprint'),
        }