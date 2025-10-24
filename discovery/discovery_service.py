from typing import Dict, Any, List
from connectors.base import DatabaseConnector
from .catalog_reader import CatalogReader
from .column_sampler import ColumnSampler  # UPDATED import
from .discovery_compressor import DiscoveryCompressor
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DiscoveryService:
    """Orchestrate discovery process with enhanced column intelligence."""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
        self.catalog_reader = CatalogReader(connector)
        self.column_sampler = ColumnSampler(connector, max_samples_per_table=100)  # INCREASED
        self.compressor = DiscoveryCompressor()
    
    def discover(self) -> Dict[str, Any]:
        """
        Execute discovery and return compressed format with semantic intelligence.
        
        Returns: Compressed discovery JSON with enhanced column metadata
        """
        # Phase 1: Catalog
        logger.info("Phase 1: Reading database catalog...")
        discovery = self.catalog_reader.read_full_catalog()
        
        # Phase 2: ENHANCED column sampling and classification
        logger.info("Phase 2: Classifying and sampling columns...")
        
        # STEP 2A: Classify ALL columns by semantic role
        sample_targets, column_classifications = self.column_sampler.identify_and_classify_columns(discovery)
        logger.info(f"  Classified {len(column_classifications)} columns")
        logger.info(f"  Selected {len(sample_targets)} columns for value sampling")
        
        # STEP 2B: Sample values for high-priority columns
        if sample_targets:
            enhanced_samples = self.column_sampler.sample_columns(sample_targets, column_classifications)
        else:
            enhanced_samples = {}
        
        # Phase 3: Compress with PRESERVED semantic metadata
        logger.info("Phase 3: Compressing discovery data with semantic preservation...")
        compressed = self.compressor.compress(discovery, enhanced_samples)
        
        logger.info(f"Discovery complete: {len(compressed['tables'])} tables")
        logger.info(f"  - {len(compressed['column_classifications'])} columns classified")
        logger.info(f"  - {len(compressed['nl_mappings'])} NL mappings created")
        logger.info(f"  - {len(compressed['column_samples'])} columns sampled")
        
        return compressed