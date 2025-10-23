from typing import Type
from .base import DatabaseConnector
from .mssql_connector import MSSQLConnector
from .postgres_connector import PostgresConnector

class DialectRegistry:
    """Registry for database connectors."""
    
    _connectors = {
        'mssql': MSSQLConnector,
        'postgresql': PostgresConnector,
        'postgres': PostgresConnector,
    }
    
    @classmethod
    def get_connector(cls, connection_string: str) -> DatabaseConnector:
        """Factory method to create appropriate connector."""
        dialect = connection_string.split(':')[0].split('+')[0]
        
        connector_class = cls._connectors.get(dialect)
        if not connector_class:
            raise ValueError(f"Unsupported database dialect: {dialect}")
        
        return connector_class(connection_string)
    
    @classmethod
    def register(cls, dialect: str, connector_class: Type[DatabaseConnector]):
        """Register a new connector (Open/Closed principle)."""
        cls._connectors[dialect] = connector_class