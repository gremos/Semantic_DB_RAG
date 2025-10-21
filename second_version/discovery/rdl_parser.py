from typing import List, Dict, Any
import os
from lxml import etree

class RDLParser:
    """Parser for SSRS RDL (Report Definition Language) XML files."""
    
    NAMESPACE = {
        'rdl': 'http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition',
        'rdl2005': 'http://schemas.microsoft.com/sqlserver/reporting/2005/01/reportdefinition',
        'rdl2010': 'http://schemas.microsoft.com/sqlserver/reporting/2010/01/reportdefinition'
    }
    
    @staticmethod
    def find_rdl_files(directory: str) -> List[str]:
        """Find all .rdl files in directory."""
        rdl_files = []
        if not os.path.exists(directory):
            return rdl_files
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.rdl'):
                    rdl_files.append(os.path.join(root, file))
        
        return rdl_files
    
    @staticmethod
    def parse_rdl(file_path: str) -> Dict[str, Any]:
        """
        Parse RDL file and extract datasets.
        
        Returns:
            {
                "path": "file_path",
                "datasets": ["dataset1", "dataset2", ...],
                "queries": [{"name": "dataset1", "query": "SELECT ..."}]
            }
        """
        try:
            tree = etree.parse(file_path)
            root = tree.getroot()
            
            # Detect namespace
            ns = None
            for prefix, uri in RDLParser.NAMESPACE.items():
                if uri in root.nsmap.values():
                    ns = {'rdl': uri}
                    break
            
            if not ns:
                ns = {'rdl': list(root.nsmap.values())[0]} if root.nsmap else {}
            
            datasets = []
            queries = []
            
            # Find DataSets
            dataset_elements = root.findall('.//rdl:DataSet', ns)
            for dataset in dataset_elements:
                name_elem = dataset.find('rdl:Name', ns)
                if name_elem is not None:
                    dataset_name = name_elem.text
                    datasets.append(dataset_name)
                    
                    # Try to extract query
                    query_elem = dataset.find('.//rdl:CommandText', ns)
                    if query_elem is not None and query_elem.text:
                        queries.append({
                            "name": dataset_name,
                            "query": query_elem.text.strip()
                        })
            
            return {
                "path": file_path,
                "datasets": datasets,
                "queries": queries
            }
        
        except Exception as e:
            return {
                "path": file_path,
                "datasets": [],
                "queries": [],
                "error": str(e)
            }