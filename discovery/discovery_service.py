from typing import Dict, Any, List
from connectors.base import DatabaseConnector
from .catalog_reader import CatalogReader
from .column_sampler import ColumnSampler
from .rdl_reader import RDLReader
from .discovery_compressor import DiscoveryCompressor
from config.settings import settings
import logging
import os
import glob


logger = logging.getLogger(__name__)

class DiscoveryService:
    """Orchestrate discovery process with enhanced column intelligence."""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
        self.catalog_reader = CatalogReader(connector)
        self.column_sampler = ColumnSampler(connector, max_samples_per_table=100)  # INCREASED
        self.compressor = DiscoveryCompressor()
        self.rdl_reader = RDLReader()
    
    def discover(self, rdl_directory: str = "data_upload") -> Dict[str, Any]:
        """Discovery with RDL enhancement."""
        
        # Phase 1: Catalog
        logger.info("Phase 1: Reading database catalog...")
        discovery = self.catalog_reader.read_full_catalog()
        
        # Phase 2: Column sampling
        logger.info("Phase 2: Classifying and sampling columns...")
        sample_targets, column_classifications = self.column_sampler.identify_and_classify_columns(discovery)
        enhanced_samples = self.column_sampler.sample_columns(sample_targets, column_classifications)
        
        # Phase 2.5: RDL enrichment (NEW)
        logger.info("Phase 2.5: Reading RDL reports for business context...")
        rdl_metadata = self._read_rdl_files(rdl_directory)
        discovery = self._enrich_with_rdl(discovery, rdl_metadata)
        
        # Phase 3: Compression
        logger.info("Phase 3: Compressing discovery data...")
        compressed = self.compressor.compress(discovery, enhanced_samples)
        
        # Add RDL metadata to compressed discovery
        compressed["rdl_reports"] = rdl_metadata
        
        return compressed
    
    def _read_rdl_files(self, directory: str) -> List[Dict[str, Any]]:
        """Read all RDL files from directory."""
        rdl_files = glob.glob(os.path.join(directory, "*.rdl"))
        rdl_metadata = []
        
        for rdl_path in rdl_files:
            logger.info(f"  Parsing RDL: {os.path.basename(rdl_path)}")
            metadata = self.rdl_reader.parse_rdl_file(rdl_path)
            if metadata:
                rdl_metadata.append(metadata)
        
        logger.info(f"  Parsed {len(rdl_metadata)} RDL reports")
        return rdl_metadata
    
    def _enrich_with_rdl(self, discovery: Dict[str, Any], rdl_metadata: List[Dict]) -> Dict[str, Any]:
        """Enrich discovery data with RDL business context."""
        # Add RDL queries as named assets
        for rdl in rdl_metadata:
            for dataset in rdl.get("datasets", []):
                discovery.setdefault("named_assets", []).append({
                    "kind": "report_query",
                    "name": f"{rdl['report_name']} - {dataset['name']}",
                    "sql_normalized": dataset["sql"],
                    "available": True,
                    "tables_referenced": dataset["tables_referenced"],
                    "business_context": rdl.get("business_context", {})
                })
        
        # Enrich table descriptions with report usage
        for schema in discovery.get("schemas", []):
            for table in schema.get("tables", []):
                table_name = f"{schema['name']}.{table['name']}"
                
                # Find RDLs that reference this table
                referencing_reports = []
                for rdl in rdl_metadata:
                    for dataset in rdl.get("datasets", []):
                        if any(table_name in ref for ref in dataset.get("tables_referenced", [])):
                            referencing_reports.append(rdl["report_name"])
                
                if referencing_reports:
                    table.setdefault("source_assets", []).extend([
                        {
                            "kind": "rdl_report",
                            "path": report,
                            "note": f"Used in {report}"
                        }
                        for report in referencing_reports
                    ])
        
        return discovery