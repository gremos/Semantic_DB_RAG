from typing import Dict, Any, List
from connectors.base import DatabaseConnector
from .catalog_reader import CatalogReader
from .column_sampler import ColumnSampler
from .discovery_compressor import DiscoveryCompressor
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DiscoveryService:
    """Orchestrate discovery process - simplified for incremental approach."""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
        self.catalog_reader = CatalogReader(connector)
        self.column_sampler = ColumnSampler(connector, max_samples=30)
        self.compressor = DiscoveryCompressor()
    
    def discover(self) -> Dict[str, Any]:
        """
        Execute discovery and return compressed format.
        
        Returns: Compressed discovery JSON
        """
        # Phase 1: Catalog
        logger.info("Phase 1: Reading database catalog...")
        discovery = self.catalog_reader.read_full_catalog()
        
        # Phase 2: Selective column sampling
        logger.info("Phase 2: Sampling key columns...")
        sample_targets = self.column_sampler.identify_sample_targets(discovery)
        logger.info(f"  Sampling {len(sample_targets)} columns")
        
        if sample_targets:
            column_samples = self.column_sampler.sample_columns(sample_targets)
            discovery["column_samples"] = column_samples
        else:
            discovery["column_samples"] = {}
        
        # Phase 3: Compress for incremental processing
        logger.info("Phase 3: Compressing discovery data...")
        compressed = self.compressor.compress(discovery)
        
        logger.info(f"Discovery complete: {len(compressed['tables'])} tables")
        return compressed