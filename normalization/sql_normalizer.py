import sqlglot
from typing import Optional, Tuple

class SQLNormalizer:
    """Wrapper for sqlglot to normalize SQL across dialects."""
    
    @staticmethod
    def normalize(sql: str, source_dialect: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Normalize SQL to a canonical form.
        
        Returns:
            (success, normalized_sql, error_message)
        """
        try:
            parsed = sqlglot.parse_one(sql, dialect=source_dialect)
            normalized = parsed.sql(dialect=source_dialect, pretty=True)
            return (True, normalized, None)
        except Exception as e:
            return (False, None, str(e))
    
    @staticmethod
    def transpile(sql: str, source_dialect: str, target_dialect: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Transpile SQL from source to target dialect.
        
        Returns:
            (success, transpiled_sql, error_message)
        """
        try:
            transpiled = sqlglot.transpile(sql, read=source_dialect, write=target_dialect)[0]
            return (True, transpiled, None)
        except Exception as e:
            return (False, None, str(e))
    
    @staticmethod
    def parse(sql: str, dialect: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL can be parsed.
        
        Returns:
            (success, error_message)
        """
        try:
            sqlglot.parse_one(sql, dialect=dialect)
            return (True, None)
        except Exception as e:
            return (False, str(e))