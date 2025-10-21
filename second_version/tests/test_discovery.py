import unittest
from unittest.mock import Mock, patch
from discovery.discovery_service import DiscoveryService
from connectors.base import DatabaseConnector

class TestDiscovery(unittest.TestCase):
    """Test discovery phase."""
    
    def setUp(self):
        self.mock_connector = Mock(spec=DatabaseConnector)
        self.mock_connector.get_vendor_version.return_value = {
            "vendor": "mssql",
            "version": "SQL Server 2022"
        }
        self.mock_connector.get_dialect.return_value = "tsql"
        self.mock_connector.get_schemas.return_value = ["dbo"]
        self.mock_connector.get_tables.return_value = [
            {"name": "Customer", "type": "table"}
        ]
        self.mock_connector.get_columns.return_value = [
            {"name": "CustomerID", "type": "int", "nullable": False},
            {"name": "CustomerName", "type": "varchar", "nullable": False}
        ]
        self.mock_connector.get_primary_keys.return_value = ["CustomerID"]
        self.mock_connector.get_foreign_keys.return_value = []
        self.mock_connector.get_row_count.return_value = 1000
        self.mock_connector.get_view_definition.return_value = None
        self.mock_connector.get_stored_procedures.return_value = []
    
    @patch('discovery.discovery_service.RDLParser')
    def test_discovery_basic(self, mock_rdl_parser):
        """Test basic discovery flow."""
        mock_rdl_parser.return_value.find_rdl_files.return_value = []
        
        service = DiscoveryService(self.mock_connector)
        result = service.discover()
        
        self.assertEqual(result["database"]["vendor"], "mssql")
        self.assertEqual(result["dialect"], "tsql")
        self.assertEqual(len(result["schemas"]), 1)
        self.assertEqual(result["schemas"][0]["name"], "dbo")
        self.assertEqual(len(result["schemas"][0]["tables"]), 1)
        self.assertEqual(result["schemas"][0]["tables"][0]["name"], "Customer")

if __name__ == '__main__':
    unittest.main()