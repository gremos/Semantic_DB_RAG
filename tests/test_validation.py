import unittest
from validation.schema_validator import SchemaValidator
from validation.grounding_validator import GroundingValidator

class TestValidation(unittest.TestCase):
    """Test validation components."""
    
    def setUp(self):
        self.validator = SchemaValidator()
    
    def test_schema_validation_valid(self):
        """Test valid discovery JSON."""
        valid_discovery = {
            "database": {"vendor": "mssql", "version": "2022"},
            "dialect": "tsql",
            "schemas": [],
            "named_assets": []
        }
        
        is_valid, error = self.validator.validate(valid_discovery, "discovery")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_schema_validation_invalid(self):
        """Test invalid discovery JSON."""
        invalid_discovery = {
            "database": {"vendor": "mssql"},  # Missing version
            "dialect": "tsql"
        }
        
        is_valid, error = self.validator.validate(invalid_discovery, "discovery")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_grounding_validation(self):
        """Test grounding validator."""
        discovery = {
            "database": {"vendor": "mssql", "version": "2022"},
            "dialect": "tsql",
            "schemas": [{
                "name": "dbo",
                "tables": [{
                    "name": "Customer",
                    "type": "table",
                    "columns": [
                        {"name": "CustomerID", "type": "int", "nullable": False}
                    ],
                    "primary_key": ["CustomerID"],
                    "foreign_keys": []
                }]
            }],
            "named_assets": []
        }
        
        grounding = GroundingValidator(discovery)
        
        valid_model = {
            "entities": [{
                "name": "Customer",
                "source": "dbo.Customer",
                "primary_key": ["CustomerID"],
                "description": "Customer entity"
            }]
        }
        
        is_valid, errors = grounding.validate_semantic_model(valid_model)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid reference
        invalid_model = {
            "entities": [{
                "name": "Product",
                "source": "dbo.Product",  # Doesn't exist
                "primary_key": ["ProductID"],
                "description": "Product entity"
            }]
        }
        
        is_valid, errors = grounding.validate_semantic_model(invalid_model)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

if __name__ == '__main__':
    unittest.main()