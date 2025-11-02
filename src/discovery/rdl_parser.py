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
        self.rdl_path = Path(settings.paths.rdl_path)
    
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
            logger.debug(f"Parsing RDL file: {file_path}")
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Try to determine namespace
            ns = self._detect_namespace(root)
            logger.debug(f"Using namespace: {ns} for file {file_path.name}")
            
            # Extract datasets
            datasets = self._extract_datasets(root, ns)
            if not datasets:
                logger.warning(f"No datasets found in RDL file: {file_path}")
            
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
            
            logger.info(f"✓ Parsed RDL: {file_path.name} - {len(datasets)} datasets, {len(parameters)} parameters")
            return asset
            
        except ET.ParseError as e:
            logger.error(f"XML parse error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse RDL file {file_path}: {e}", exc_info=True)
            return None
    
    def _detect_namespace(self, root: ET.Element) -> str:
        """Detect RDL namespace from root element."""
        tag = root.tag
        if '}' in tag:
            ns = tag.split('}')[0].strip('{')
            # Map to prefix
            for prefix, uri in self.NAMESPACES.items():
                if uri == ns:
                    logger.debug(f"Detected RDL namespace: {prefix} ({ns})")
                    return prefix
            # Log if namespace is unknown
            logger.warning(f"Unknown RDL namespace: {ns}, falling back to 'rd2008'")
        
        # Default to rd2008 (most common)
        return 'rd2008'
    

    # in _extract_datasets
    def _extract_datasets(self, root: ET.Element, ns: str) -> List[Dict[str, Any]]:
        """Extract dataset definitions from RDL (2005/2008/2010/2016)."""
        uri = self.NAMESPACES[ns]
        datasets: List[Dict[str, Any]] = []

        # Find *all* DataSet nodes anywhere (don’t depend on DataSets wrapper)
        for dataset_elem in root.findall(f'.//{{{uri}}}DataSet'):
            # Name can be attribute or child element depending on RDL version/tooling
            dataset_name = dataset_elem.get('Name')
            if not dataset_name:
                name_elem = dataset_elem.find(f'{{{uri}}}Name')
                dataset_name = name_elem.text if name_elem is not None else 'UnnamedDataSet'

            # Query text
            query_elem = dataset_elem.find(f'.//{{{uri}}}CommandText')
            raw_query = query_elem.text if query_elem is not None else None

            normalized_query = None
            if raw_query:
                cleaned_query = self._clean_sql(raw_query)
                try:
                    import sqlglot
                    parsed = sqlglot.parse_one(cleaned_query, dialect='tsql')
                    normalized_query = parsed.sql(dialect='tsql', pretty=True)
                    logger.debug(f"✓ Normalized SQL for dataset '{dataset_name}'")
                except Exception as e:
                    logger.debug(f"Could not normalize SQL for dataset '{dataset_name}': {e}")
                    normalized_query = cleaned_query

            dataset = {
                'name': dataset_name,
                'query': raw_query,
                'sql_normalized': normalized_query
            }

            fields = self._extract_fields(dataset_elem, ns)
            if fields:
                dataset['fields'] = fields

            datasets.append(dataset)

        logger.debug(f"Extracted {len(datasets)} datasets")
        return datasets


    def _clean_sql(self, sql: str) -> str:
        """Clean SQL text - remove RDL-specific artifacts"""
        if not sql:
            return sql
        
        # Remove XML entities
        sql = sql.replace('&gt;', '>')
        sql = sql.replace('&lt;', '<')
        sql = sql.replace('&amp;', '&')
        
        # Remove excess whitespace
        import re
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.strip()
        
        return sql
        
    def _extract_fields(self, dataset_elem: ET.Element, ns: str) -> List[Dict[str, str]]:
        """Extract field definitions from a dataset."""
        fields: List[Dict[str, str]] = []
        uri = self.NAMESPACES[ns]

        fields_elem = dataset_elem.find(f'{{{uri}}}Fields')
        if fields_elem is None:
            return fields

        for field_elem in fields_elem.findall(f'{{{uri}}}Field'):
            # In RDL 2008+, field name is usually an attribute
            name = field_elem.get('Name')
            if not name:
                name_elem = field_elem.find(f'{{{uri}}}Name')
                name = name_elem.text if name_elem is not None else None

            datafield_elem = field_elem.find(f'{{{uri}}}DataField')
            fields.append({
                'name': name,
                'data_field': datafield_elem.text if datafield_elem is not None else None
            })

        return fields

    def _extract_data_sources(self, root: ET.Element, ns: str) -> List[Dict[str, str]]:
        """Extract data source definitions."""
        sources: List[Dict[str, str]] = []
        uri = self.NAMESPACES[ns]

        sources_elem = root.find(f'.//{{{uri}}}DataSources')
        if sources_elem is None:
            return sources

        for source_elem in sources_elem.findall(f'{{{uri}}}DataSource'):
            name = source_elem.get('Name')
            if not name:
                name_elem = source_elem.find(f'{{{uri}}}Name')
                name = name_elem.text if name_elem is not None else None

            conn_elem = source_elem.find(f'.//{{{uri}}}ConnectString')
            sources.append({
                'name': name,
                'connection_string': conn_elem.text if conn_elem is not None else None
            })

        return sources

    
    def _extract_parameters(self, root: ET.Element, ns: str) -> List[Dict[str, Any]]:
        """Extract report parameters."""
        parameters: List[Dict[str, Any]] = []
        uri = self.NAMESPACES[ns]

        params_elem = root.find(f'.//{{{uri}}}ReportParameters')
        if params_elem is None:
            return parameters

        for param_elem in params_elem.findall(f'{{{uri}}}ReportParameter'):
            # Name is usually an attribute in RDL 2008+
            name = param_elem.get('Name')
            if not name:
                name_elem = param_elem.find(f'{{{uri}}}Name')
                name = name_elem.text if name_elem is not None else None

            datatype_elem = param_elem.find(f'{{{uri}}}DataType')
            parameters.append({
                'name': name,
                'data_type': datatype_elem.text if datatype_elem is not None else None
            })

        return parameters
    
    def parse_all_rdl_files(self) -> List[Dict[str, Any]]:
        """Parse all RDL files in RDL_PATH."""
        assets = []
        
        for rdl_file in self.find_rdl_files():
            asset = self.parse_rdl_file(rdl_file)
            if asset:
                assets.append(asset)
        
        return assets
    


    def extract_relationships_from_rdl(self, rdl_assets: List[Dict[str, Any]], config: Any) -> List[Dict[str, Any]]:
        import xml.etree.ElementTree as ET
        from pathlib import Path
        import sqlglot

        if not config.detect_rdl_joins:
            logger.info("RDL relationship detection disabled")
            return []

        relationships = []
        rdls_analyzed = 0

        for asset in rdl_assets:
            if asset.get("kind") != "rdl":
                continue

            rel_path = asset.get("path")
            if not rel_path:
                continue

            # asset['path'] is relative to self.rdl_path
            full_path = (self.rdl_path / rel_path).resolve()
            if not full_path.exists():
                logger.debug(f"Skipping RDL asset: missing file {full_path}")
                continue

            try:
                root = ET.parse(full_path).getroot()
                namespaces = [
                    "{http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition}",
                    "{http://schemas.microsoft.com/sqlserver/reporting/2010/01/reportdefinition}",
                    "{http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition}",
                    "{http://schemas.microsoft.com/sqlserver/reporting/2005/01/reportdefinition}",
                ]

                for ns in namespaces:
                    for dataset in root.findall(f".//{ns}DataSet"):
                        name_attr = dataset.get("Name")
                        name_elem = dataset.find(f"{ns}Name")
                        dataset_name = name_attr or (name_elem.text if name_elem is not None else "Unknown")

                        query_elem = dataset.find(f".//{ns}CommandText")
                        if query_elem is None or not query_elem.text:
                            continue

                        query = self._clean_sql(query_elem.text)

                        try:
                            parsed = sqlglot.parse_one(query, dialect="tsql")
                            for join in parsed.find_all(sqlglot.exp.Join):
                                on_clause = join.args.get("on")
                                if not on_clause:
                                    continue
                                left_table, left_col, right_table, right_col = _parse_join_condition(on_clause)
                                if all([left_table, left_col, right_table, right_col]):
                                    confidence = {
                                        "high": "high",
                                        "medium": "medium",
                                        "low": "low",
                                    }.get(getattr(config, "rdl_trust_level", "high").lower(), "high")

                                    relationships.append({
                                        "from": f"{left_table}.{left_col}",
                                        "to": f"{right_table}.{right_col}",
                                        "method": "rdl_join_analysis",
                                        "cardinality": "unknown",
                                        "confidence": confidence,
                                        "source_rdl": full_path.name,
                                        "source_dataset": dataset_name,
                                    })
                        except Exception as e:
                            logger.debug(f"Join parse failed for {full_path}/{dataset_name}: {e}")

                rdls_analyzed += 1

            except ET.ParseError as e:
                logger.error(f"Failed to parse RDL XML file {full_path}: {e}")
            except Exception as e:
                logger.error(f"Error processing RDL file {full_path}: {e}")

        logger.info(f"RDL relationship detection complete: analyzed {rdls_analyzed} files, "
                    f"found {len(relationships)} relationships")
        return relationships
