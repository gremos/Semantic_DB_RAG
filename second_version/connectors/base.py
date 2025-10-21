from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""
    
    @abstractmethod
    def get_vendor_version(self) -> Dict[str, str]:
        """Return vendor name and version."""
        pass
    
    @abstractmethod
    def get_dialect(self) -> str:
        """Return sqlglot dialect name."""
        pass
    
    @abstractmethod
    def get_schemas(self, exclusions: List[str]) -> List[str]:
        """Return list of schema names, excluding specified schemas."""
        pass
    
    @abstractmethod
    def get_tables(self, schema: str, table_exclusions: List[str]) -> List[Dict[str, Any]]:
        """Return tables/views in schema with metadata."""
        pass
    
    @abstractmethod
    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Return columns with type and nullable info."""
        pass
    
    @abstractmethod
    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        """Return primary key column names."""
        pass
    
    @abstractmethod
    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        """Return foreign key relationships."""
        pass
    
    @abstractmethod
    def get_row_count(self, schema: str, table: str) -> Optional[int]:
        """Return approximate row count."""
        pass
    
    @abstractmethod
    def get_view_definition(self, schema: str, view: str) -> Optional[str]:
        """Return view SQL definition."""
        pass
    
    @abstractmethod
    def get_stored_procedures(self, schema: str) -> List[Dict[str, str]]:
        """Return stored procedure names and definitions."""
        pass
    
    @abstractmethod
    def close(self):
        """Close database connection."""
        pass