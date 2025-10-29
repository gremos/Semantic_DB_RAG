"""
RDL (Report Definition Language) file parser.
Extracts datasets, queries, joins, and parameters from .rdl files.
"""

import os
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from pathlib import Path

from config.settings import Settings

from src.discovery.relationship_detector import _parse_join_condition

logger = logging.getLogger(__name__)

class RDLParser:
    """Parse RDL files to extract data definitions."""
    
    # RDL namespace
    NAMESPACES = {
        'rd': 'http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition',
        'rd2010': 'http://schemas.microsoft.com/sqlserver/reporting/2010/01/reportdefinition',
        'rd2008': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'
    }
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rdl_path = Path(settings.RDL_PATH)
    
    def find_rdl_files(self) -> List[Path]:
        """Find all .rdl files in the RDL_PATH directory."""
        if not self.rdl_path.exists():
            return []
        
        return list(self.rdl_path.rglob('*.rdl'))
    
    def parse_rdl_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a single RDL file and extract datasets, queries, etc.
        Returns asset dictionary or None if parsing fails.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Try to determine namespace
            ns = self._detect_namespace(root)
            
            # Extract datasets
            datasets = self._extract_datasets(root, ns)
            
            # Extract data sources
            data_sources = self._extract_data_sources(root, ns)
            
            # Extract parameters
            parameters = self._extract_parameters(root, ns)
            
            asset = {
                'kind': 'rdl',
                'path': str(file_path.relative_to(self.rdl_path)),
                'name': file_path.stem,
                'datasets': datasets,
                'data_sources': data_sources,
                'parameters': parameters
            }
            
            return asset
            
        except Exception as e:
            # Log error but don't fail
            return None
    
    def _detect_namespace(self, root: ET.Element) -> str:
        """Detect RDL namespace from root element."""
        tag = root.tag
        if '}' in tag:
            ns = tag.split('}')[0].strip('{')
            # Map to prefix
            for prefix, uri in self.NAMESPACES.items():
                if uri == ns:
                    return prefix
        return 'rd'  # Default
    
    def _extract_datasets(self, root: ET.Element, ns: str) -> List[Dict[str, Any]]:
        """Extract dataset definitions from RDL."""
        datasets = []
        
        # Find DataSets element
        datasets_elem = root.find(f'.//{{{self.NAMESPACES[ns]}}}DataSets')
        if datasets_elem is None:
            return datasets
        
        for dataset_elem in datasets_elem.findall(f'{{{self.NAMESPACES[ns]}}}DataSet'):
            name_elem = dataset_elem.find(f'{{{self.NAMESPACES[ns]}}}Name')
            query_elem = dataset_elem.find(f'.//{{{self.NAMESPACES[ns]}}}CommandText')
            
            if name_elem is not None:
                dataset = {
                    'name': name_elem.text,
                    'query': query_elem.text if query_elem is not None else None
                }
                
                # Extract fields
                fields = self._extract_fields(dataset_elem, ns)
                if fields:
                    dataset['fields'] = fields
                
                datasets.append(dataset)
        
        return datasets
    
    def _extract_fields(self, dataset_elem: ET.Element, ns: str) -> List[Dict[str, str]]:
        """Extract field definitions from a dataset."""
        fields = []
        
        fields_elem = dataset_elem.find(f'{{{self.NAMESPACES[ns]}}}Fields')
        if fields_elem is None:
            return fields
        
        for field_elem in fields_elem.findall(f'{{{self.NAMESPACES[ns]}}}Field'):
            name_elem = field_elem.find(f'{{{self.NAMESPACES[ns]}}}Name')
            datafield_elem = field_elem.find(f'{{{self.NAMESPACES[ns]}}}DataField')
            
            if name_elem is not None:
                field = {
                    'name': name_elem.text,
                    'data_field': datafield_elem.text if datafield_elem is not None else None
                }
                fields.append(field)
        
        return fields
    
    def _extract_data_sources(self, root: ET.Element, ns: str) -> List[Dict[str, str]]:
        """Extract data source definitions."""
        sources = []
        
        sources_elem = root.find(f'.//{{{self.NAMESPACES[ns]}}}DataSources')
        if sources_elem is None:
            return sources
        
        for source_elem in sources_elem.findall(f'{{{self.NAMESPACES[ns]}}}DataSource'):
            name_elem = source_elem.find(f'{{{self.NAMESPACES[ns]}}}Name')
            conn_elem = source_elem.find(f'.//{{{self.NAMESPACES[ns]}}}ConnectString')
            
            if name_elem is not None:
                source = {
                    'name': name_elem.text,
                    'connection_string': conn_elem.text if conn_elem is not None else None
                }
                sources.append(source)
        
        return sources
    
    def _extract_parameters(self, root: ET.Element, ns: str) -> List[Dict[str, Any]]:
        """Extract report parameters."""
        parameters = []
        
        params_elem = root.find(f'.//{{{self.NAMESPACES[ns]}}}ReportParameters')
        if params_elem is None:
            return parameters
        
        for param_elem in params_elem.findall(f'{{{self.NAMESPACES[ns]}}}ReportParameter'):
            name_elem = param_elem.find(f'{{{self.NAMESPACES[ns]}}}Name')
            datatype_elem = param_elem.find(f'{{{self.NAMESPACES[ns]}}}DataType')
            
            if name_elem is not None:
                param = {
                    'name': name_elem.text,
                    'data_type': datatype_elem.text if datatype_elem is not None else None
                }
                parameters.append(param)
        
        return parameters
    
    def parse_all_rdl_files(self) -> List[Dict[str, Any]]:
        """Parse all RDL files in RDL_PATH."""
        assets = []
        
        for rdl_file in self.find_rdl_files():
            asset = self.parse_rdl_file(rdl_file)
            if asset:
                assets.append(asset)
        
        return assets
    

    def extract_relationships_from_rdl(
            rdl_assets: List[Dict[str, Any]],
            config: Any
        ) -> List[Dict[str, Any]]:
            """
            Extract relationships from RDL dataset join definitions
            
            RDL datasets often define joins between tables explicitly in queries.
            These are high-confidence relationships since they're curated.
            
            Args:
                rdl_assets: List of RDL asset dictionaries
                config: Relationship detection configuration
                
            Returns:
                List of relationship dictionaries
            """
            import xml.etree.ElementTree as ET
            from pathlib import Path
            import sqlglot
            
            if not config.detect_rdl_joins:
                logger.info("RDL relationship detection disabled")
                return []
            
            logger.info("Starting RDL relationship detection...")
            relationships = []
            rdls_analyzed = 0
            
            for asset in rdl_assets:
                if asset.get("kind") != "rdl":
                    continue
                
                rdl_path = asset.get("path")
                if not rdl_path or not Path(rdl_path).exists():
                    logger.debug(f"Skipping RDL asset: invalid path {rdl_path}")
                    continue
                
                try:
                    tree = ET.parse(rdl_path)
                    root = tree.getroot()
                    
                    # Find DataSets in multiple RDL namespaces
                    namespaces = [
                        "{http://schemas.microsoft.com/sqlserver/reporting/2010/01/reportdefinition}",
                        "{http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition}",
                        "{http://schemas.microsoft.com/sqlserver/reporting/2005/01/reportdefinition}"
                    ]
                    
                    for ns in namespaces:
                        datasets = root.findall(f".//{ns}DataSet")
                        
                        for dataset in datasets:
                            # Get dataset name
                            name_elem = dataset.find(f"{ns}Name")
                            dataset_name = name_elem.text if name_elem is not None else "Unknown"
                            
                            # Look for query with JOINs
                            query_elem = dataset.find(f".//{ns}CommandText")
                            if query_elem is not None and query_elem.text:
                                query = query_elem.text.strip()
                                
                                # Parse joins from query
                                try:
                                    parsed = sqlglot.parse_one(query)
                                    
                                    for join in parsed.find_all(sqlglot.exp.Join):
                                        on_clause = join.args.get("on")
                                        if not on_clause:
                                            continue
                                        
                                        # Use same parsing logic as views
                                        left_table, left_col, right_table, right_col = \
                                            _parse_join_condition(on_clause)
                                        
                                        if all([left_table, left_col, right_table, right_col]):
                                            # Map confidence based on config
                                            confidence_map = {
                                                "high": "high",
                                                "medium": "medium",
                                                "low": "low"
                                            }
                                            confidence = confidence_map.get(
                                                config.rdl_trust_level.lower(), 
                                                "high"
                                            )
                                            
                                            relationships.append({
                                                "from": f"{left_table}.{left_col}",
                                                "to": f"{right_table}.{right_col}",
                                                "method": "rdl_join_analysis",
                                                "cardinality": "unknown",
                                                "confidence": confidence,
                                                "source_rdl": Path(rdl_path).name,
                                                "source_dataset": dataset_name
                                            })
                                
                                except sqlglot.errors.ParseError as e:
                                    logger.debug(f"Failed to parse RDL query in {rdl_path}/{dataset_name}: {e}")
                                except Exception as e:
                                    logger.warning(f"Error parsing RDL query in {rdl_path}/{dataset_name}: {e}")
                    
                    rdls_analyzed += 1
                
                except ET.ParseError as e:
                    logger.error(f"Failed to parse RDL XML file {rdl_path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing RDL file {rdl_path}: {e}")
            
            logger.info(f"RDL relationship detection complete: analyzed {rdls_analyzed} files, "
                        f"found {len(relationships)} relationships")
            
            return relationships