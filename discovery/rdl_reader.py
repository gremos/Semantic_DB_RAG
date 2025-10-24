import xml.etree.ElementTree as ET
from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

class RDLReader:
    """Extract semantic metadata from SQL Server Reporting Services (.rdl) files."""
    
    NAMESPACES = {
        'rd': 'http://schemas.microsoft.com/SQLServer/reporting/reportdesigner',
        '': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'
    }
    
    def parse_rdl_file(self, rdl_path: str) -> Dict[str, Any]:
        """
        Parse RDL file and extract semantic information.
        
        Returns:
            {
                "report_name": "Absentees ( Monthly Report )",
                "datasets": [
                    {
                        "name": "AbsenteesDataset",
                        "sql": "SELECT ...",
                        "tables_referenced": ["cardinal12.HrMaster.dbo.T0X_PERSON", ...],
                        "fields": [
                            {"name": "Personnel_Number", "type": "System.String"},
                            {"name": "LAST_NAME", "type": "System.String"}
                        ],
                        "parameters": [
                            {"name": "datefrom", "type": "DateTime", "prompt": "Ημ. Από"},
                            {"name": "dateto", "type": "DateTime", "prompt": "Ημ. Έως"}
                        ]
                    }
                ],
                "business_context": {
                    "domain": "HR",  # Inferred from Personnel, LAST_NAME
                    "purpose": "Track employee absences by date range",
                    "key_entities": ["Person", "Absences"]
                }
            }
        """
        try:
            tree = ET.parse(rdl_path)
            root = tree.getroot()
            
            # Extract report name
            report_name = self._extract_report_title(root)
            
            # Extract datasets
            datasets = []
            for dataset in root.findall('.//DataSet', self.NAMESPACES):
                dataset_info = self._parse_dataset(dataset)
                datasets.append(dataset_info)
            
            # Extract business context
            business_context = self._infer_business_context(report_name, datasets)
            
            return {
                "report_name": report_name,
                "datasets": datasets,
                "business_context": business_context
            }
        
        except Exception as e:
            logger.error(f"Failed to parse RDL file {rdl_path}: {e}")
            return None
    
    def _parse_dataset(self, dataset_elem) -> Dict[str, Any]:
        """Extract dataset information."""
        name = dataset_elem.get('Name')
        
        # Extract SQL query
        query_elem = dataset_elem.find('.//CommandText', self.NAMESPACES)
        sql = query_elem.text if query_elem is not None else ""
        
        # Extract tables from SQL
        tables = self._extract_tables_from_sql(sql)
        
        # Extract field definitions
        fields = []
        fields_elem = dataset_elem.find('.//Fields', self.NAMESPACES)
        if fields_elem is not None:
            for field in fields_elem.findall('.//Field', self.NAMESPACES):
                field_name = field.get('Name')
                data_field = field.find('.//DataField', self.NAMESPACES)
                type_name = field.find('.//rd:TypeName', self.NAMESPACES)
                
                fields.append({
                    "name": field_name,
                    "source_field": data_field.text if data_field is not None else field_name,
                    "type": type_name.text if type_name is not None else "Unknown"
                })
        
        # Extract parameters
        parameters = []
        query_params = dataset_elem.findall('.//QueryParameter', self.NAMESPACES)
        for param in query_params:
            param_name = param.get('Name')
            value_elem = param.find('.//Value', self.NAMESPACES)
            parameters.append({
                "name": param_name.lstrip('@'),
                "value_expression": value_elem.text if value_elem is not None else ""
            })
        
        return {
            "name": name,
            "sql": sql,
            "tables_referenced": tables,
            "fields": fields,
            "parameters": parameters
        }
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table references from SQL using regex."""
        # Pattern: FROM/JOIN table_name
        pattern = r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+|[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+|[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+|[a-zA-Z0-9_]+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return list(set(matches))
    
    def _extract_report_title(self, root) -> str:
        """Extract report title from Textbox elements."""
        for textbox in root.findall('.//Textbox', self.NAMESPACES):
            value_elem = textbox.find('.//Value', self.NAMESPACES)
            if value_elem is not None and value_elem.text:
                # First substantial text is usually the title
                if len(value_elem.text) > 10 and not value_elem.text.startswith('='):
                    return value_elem.text
        return "Unknown Report"
    
    def _infer_business_context(self, report_name: str, datasets: List[Dict]) -> Dict[str, Any]:
        """Infer business domain and purpose from report metadata."""
        # Collect all field names
        all_fields = []
        for ds in datasets:
            all_fields.extend([f["name"] for f in ds.get("fields", [])])
        
        # Domain inference based on field patterns
        domains = {
            "HR": ["personnel", "employee", "firstname", "lastname", "absence"],
            "Sales": ["customer", "order", "product", "amount", "revenue"],
            "Finance": ["invoice", "payment", "balance", "account"],
            "Marketing": ["campaign", "lead", "conversion"]
        }
        
        detected_domain = "General"
        for domain, keywords in domains.items():
            if any(any(kw in field.lower() for kw in keywords) for field in all_fields):
                detected_domain = domain
                break
        
        return {
            "domain": detected_domain,
            "purpose": f"Report: {report_name}",
            "key_fields": all_fields[:10]  # First 10 fields
        }