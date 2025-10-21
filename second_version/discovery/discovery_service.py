from typing import Dict, Any
from connectors.base import DatabaseConnector
from normalization.sql_normalizer import SQLNormalizer
from .catalog_reader import CatalogReader
from .asset_parser import AssetParser
from .rdl_parser import RDLParser
from config.settings import settings

class DiscoveryService:
    """Orchestrate full discovery process."""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
        self.normalizer = SQLNormalizer()
        self.catalog_reader = CatalogReader(connector)
        self.asset_parser = AssetParser(connector, self.normalizer)
        self.rdl_parser = RDLParser()
    
    def discover(self) -> Dict[str, Any]:
        """
        Execute full discovery process.
        
        Returns: Discovery JSON matching schema.
        """
        # Phase 1: Catalog
        discovery = self.catalog_reader.read_full_catalog()
        
        # Phase 2: Named assets (views, SPs)
        schemas = [s["name"] for s in discovery["schemas"]]
        
        views = self.asset_parser.parse_views(schemas)
        stored_procs = self.asset_parser.parse_stored_procedures(schemas)
        
        # Phase 3: RDL files
        rdl_assets = []
        rdl_files = self.rdl_parser.find_rdl_files("./data_upload")
        
        for rdl_file in rdl_files:
            rdl_data = self.rdl_parser.parse_rdl(rdl_file)
            rdl_assets.append({
                "kind": "rdl",
                "name": rdl_data["path"],
                "path": rdl_data["path"],
                "datasets": rdl_data["datasets"]
            })
            
            # Also add queries as named assets
            for query_info in rdl_data.get("queries", []):
                success, normalized, error = self.normalizer.normalize(
                    query_info["query"], 
                    discovery["dialect"]
                )
                views.append({
                    "kind": "rdl_query",
                    "name": f"RDL:{query_info['name']}",
                    "sql_normalized": normalized if success else query_info["query"]
                })
        
        # Combine all assets
        discovery["named_assets"] = views + stored_procs + rdl_assets
        
        return discovery