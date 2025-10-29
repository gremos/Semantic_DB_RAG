"""
Relationship Detector - FIXED VERSION with SQL Server Collation Conflict Handling

This version handles collation conflicts that occur when comparing columns
with different collations (e.g., Greek_CI_AS vs SQL_Latin1_General_CP1_CI_AI)
"""

from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import re

from config.settings import Settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RelationshipDetector:
    """Detect implicit foreign key relationships with collation conflict handling."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = None
        self.db_dialect = None
        
        # Table exclusions
        self.excluded_prefixes = getattr(settings, 'TABLE_EXCLUSIONS', [])
        self.excluded_patterns = []
        
        # Compile exclusion patterns
        pattern_str = getattr(settings, 'TABLE_EXCLUSION_PATTERNS', '')
        if pattern_str:
            patterns = [p.strip() for p in pattern_str.split(',') if p.strip()]
            self.excluded_patterns = [re.compile(p) for p in patterns]
    
    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.engine = create_engine(
                self.settings.DATABASE_CONNECTION_STRING,
                pool_pre_ping=True,
                echo=False
            )
            # Detect dialect
            self.db_dialect = self.engine.dialect.name.lower()
            logger.info(f"RelationshipDetector connected to {self.db_dialect} database")
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
    
    def _calculate_overlap(
        self,
        schema_a: str, table_a: str, col_a: str,
        schema_b: str, table_b: str, col_b: str
    ) -> float:
        """
        Calculate value overlap rate with COLLATION CONFLICT HANDLING.
        
        Returns percentage of values in col_a that exist in col_b.
        
        FIX: Uses COLLATE DATABASE_DEFAULT for SQL Server to handle columns
        with different collations (e.g., Greek_CI_AS vs SQL_Latin1_General_CP1_CI_AI)
        """
        if not self.engine:
            return 0.0
        
        try:
            with self.engine.connect() as conn:
                # Set query timeout (10 seconds per query)
                if self.db_dialect == 'mssql':
                    conn.execute(text("SET LOCK_TIMEOUT 10000"))  # 10s in ms
                elif self.db_dialect == 'postgresql':
                    conn.execute(text("SET statement_timeout = '10s'"))
                elif self.db_dialect == 'mysql':
                    conn.execute(text("SET SESSION max_execution_time = 10000"))
                
                # Build dialect-appropriate query with collation handling
                if self.db_dialect == 'mssql':
                    # SQL Server with COLLATE DATABASE_DEFAULT to handle collation conflicts
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT DISTINCT TOP 1000 [{col_a}] AS val
                            FROM [{schema_a}].[{table_a}]
                            WHERE [{col_a}] IS NOT NULL
                        ),
                        sample_b AS (
                            SELECT DISTINCT [{col_b}] AS val
                            FROM [{schema_b}].[{table_b}]
                            WHERE [{col_b}] IS NOT NULL
                        )
                        SELECT 
                            COUNT(DISTINCT a.val) AS total_a,
                            COUNT(DISTINCT CASE 
                                WHEN b.val IS NOT NULL THEN a.val 
                            END) AS overlap
                        FROM sample_a a
                        LEFT JOIN sample_b b 
                            ON a.val COLLATE DATABASE_DEFAULT = b.val COLLATE DATABASE_DEFAULT
                    """)
                    
                elif self.db_dialect == 'postgresql':
                    # PostgreSQL uses text collation
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT DISTINCT "{col_a}" AS val
                            FROM "{schema_a}"."{table_a}"
                            WHERE "{col_a}" IS NOT NULL
                            LIMIT 1000
                        ),
                        sample_b AS (
                            SELECT DISTINCT "{col_b}" AS val
                            FROM "{schema_b}"."{table_b}"
                            WHERE "{col_b}" IS NOT NULL
                        )
                        SELECT 
                            COUNT(DISTINCT a.val) AS total_a,
                            COUNT(DISTINCT CASE WHEN b.val IS NOT NULL THEN a.val END) AS overlap
                        FROM sample_a a
                        LEFT JOIN sample_b b ON a.val::text = b.val::text
                    """)
                    
                elif self.db_dialect == 'mysql':
                    # MySQL uses binary comparison for collation safety
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT DISTINCT `{col_a}` AS val
                            FROM `{schema_a}`.`{table_a}`
                            WHERE `{col_a}` IS NOT NULL
                            LIMIT 1000
                        ),
                        sample_b AS (
                            SELECT DISTINCT `{col_b}` AS val
                            FROM `{schema_b}`.`{table_b}`
                            WHERE `{col_b}` IS NOT NULL
                        )
                        SELECT 
                            COUNT(DISTINCT a.val) AS total_a,
                            COUNT(DISTINCT CASE WHEN b.val IS NOT NULL THEN a.val END) AS overlap
                        FROM sample_a a
                        LEFT JOIN sample_b b ON BINARY a.val = BINARY b.val
                    """)
                    
                else:
                    # Generic SQL (SQLite, etc.)
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT DISTINCT "{col_a}" AS val
                            FROM "{schema_a}"."{table_a}"
                            WHERE "{col_a}" IS NOT NULL
                            LIMIT 1000
                        ),
                        sample_b AS (
                            SELECT DISTINCT "{col_b}" AS val
                            FROM "{schema_b}"."{table_b}"
                            WHERE "{col_b}" IS NOT NULL
                        )
                        SELECT 
                            COUNT(DISTINCT a.val) AS total_a,
                            COUNT(DISTINCT CASE WHEN b.val IS NOT NULL THEN a.val END) AS overlap
                        FROM sample_a a
                        LEFT JOIN sample_b b ON a.val = b.val
                    """)
                
                result = conn.execute(query)
                row = result.fetchone()
                
                if row and row[0] > 0:
                    overlap_rate = row[1] / row[0]
                    logger.debug(
                        f"Overlap: {schema_a}.{table_a}.{col_a} -> "
                        f"{schema_b}.{table_b}.{col_b} = {overlap_rate:.2%}"
                    )
                    return overlap_rate
                
        except OperationalError as e:
            # Query timeout
            logger.debug(f"Query timeout for overlap calculation: {e}")
        except SQLAlchemyError as e:
            error_msg = str(e).lower()
            
            # Log collation errors specifically
            if 'collation' in error_msg or 'collate' in error_msg:
                logger.warning(
                    f"Collation conflict between {schema_a}.{table_a}.{col_a} "
                    f"and {schema_b}.{table_b}.{col_b}: {e}"
                )
            else:
                logger.debug(f"SQL error in overlap calculation: {e}")
        
        return 0.0
    
    def _determine_cardinality(
        self,
        schema_a: str, table_a: str, col_a: str,
        schema_b: str, table_b: str, col_b: str
    ) -> str:
        """
        Determine cardinality of relationship with COLLATION HANDLING.
        Returns: one_to_one, one_to_many, many_to_one, many_to_many
        """
        if not self.engine:
            return 'many_to_many'
        
        try:
            with self.engine.connect() as conn:
                # Set query timeout
                if self.db_dialect == 'mssql':
                    conn.execute(text("SET LOCK_TIMEOUT 10000"))
                elif self.db_dialect == 'postgresql':
                    conn.execute(text("SET statement_timeout = '10s'"))
                elif self.db_dialect == 'mysql':
                    conn.execute(text("SET SESSION max_execution_time = 10000"))
                
                # Build dialect-appropriate query with collation handling
                if self.db_dialect == 'mssql':
                    # SQL Server with COLLATE DATABASE_DEFAULT
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT TOP 1000 [{col_a}] AS val
                            FROM [{schema_a}].[{table_a}]
                            WHERE [{col_a}] IS NOT NULL
                        ),
                        sample_b AS (
                            SELECT TOP 1000 [{col_b}] AS val
                            FROM [{schema_b}].[{table_b}]
                            WHERE [{col_b}] IS NOT NULL
                        ),
                        joined AS (
                            SELECT 
                                a.val AS val_a,
                                b.val AS val_b
                            FROM sample_a a
                            INNER JOIN sample_b b 
                                ON a.val COLLATE DATABASE_DEFAULT = b.val COLLATE DATABASE_DEFAULT
                        )
                        SELECT 
                            COUNT(DISTINCT val_a) AS distinct_a,
                            COUNT(DISTINCT val_b) AS distinct_b,
                            COUNT(*) AS total_matches
                        FROM joined
                    """)
                    
                elif self.db_dialect == 'postgresql':
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT "{col_a}" AS val
                            FROM "{schema_a}"."{table_a}"
                            WHERE "{col_a}" IS NOT NULL
                            LIMIT 1000
                        ),
                        sample_b AS (
                            SELECT "{col_b}" AS val
                            FROM "{schema_b}"."{table_b}"
                            WHERE "{col_b}" IS NOT NULL
                            LIMIT 1000
                        ),
                        joined AS (
                            SELECT 
                                a.val AS val_a,
                                b.val AS val_b
                            FROM sample_a a
                            INNER JOIN sample_b b ON a.val::text = b.val::text
                        )
                        SELECT 
                            COUNT(DISTINCT val_a) AS distinct_a,
                            COUNT(DISTINCT val_b) AS distinct_b,
                            COUNT(*) AS total_matches
                        FROM joined
                    """)
                    
                elif self.db_dialect == 'mysql':
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT `{col_a}` AS val
                            FROM `{schema_a}`.`{table_a}`
                            WHERE `{col_a}` IS NOT NULL
                            LIMIT 1000
                        ),
                        sample_b AS (
                            SELECT `{col_b}` AS val
                            FROM `{schema_b}`.`{table_b}`
                            WHERE `{col_b}` IS NOT NULL
                            LIMIT 1000
                        ),
                        joined AS (
                            SELECT 
                                a.val AS val_a,
                                b.val AS val_b
                            FROM sample_a a
                            INNER JOIN sample_b b ON BINARY a.val = BINARY b.val
                        )
                        SELECT 
                            COUNT(DISTINCT val_a) AS distinct_a,
                            COUNT(DISTINCT val_b) AS distinct_b,
                            COUNT(*) AS total_matches
                        FROM joined
                    """)
                    
                else:
                    # Generic SQL
                    query = text(f"""
                        WITH sample_a AS (
                            SELECT "{col_a}" AS val
                            FROM "{schema_a}"."{table_a}"
                            WHERE "{col_a}" IS NOT NULL
                            LIMIT 1000
                        ),
                        sample_b AS (
                            SELECT "{col_b}" AS val
                            FROM "{schema_b}"."{table_b}"
                            WHERE "{col_b}" IS NOT NULL
                            LIMIT 1000
                        ),
                        joined AS (
                            SELECT 
                                a.val AS val_a,
                                b.val AS val_b
                            FROM sample_a a
                            INNER JOIN sample_b b ON a.val = b.val
                        )
                        SELECT 
                            COUNT(DISTINCT val_a) AS distinct_a,
                            COUNT(DISTINCT val_b) AS distinct_b,
                            COUNT(*) AS total_matches
                        FROM joined
                    """)
                
                result = conn.execute(query)
                row = result.fetchone()
                
                if not row or row[2] == 0:
                    return 'many_to_many'
                
                distinct_a = row[0]
                distinct_b = row[1]
                total = row[2]
                
                # Determine cardinality based on ratios
                ratio_a = total / distinct_a if distinct_a > 0 else 0
                ratio_b = total / distinct_b if distinct_b > 0 else 0
                
                if ratio_a <= 1.1 and ratio_b <= 1.1:
                    return 'one_to_one'
                elif ratio_a <= 1.1:
                    return 'many_to_one'
                elif ratio_b <= 1.1:
                    return 'one_to_many'
                else:
                    return 'many_to_many'
        
        except SQLAlchemyError as e:
            logger.debug(f"Error determining cardinality: {e}")
        
        return 'many_to_many'
    
    def _types_compatible(self, type_a: str, type_b: str) -> bool:
        """Check if two column types are compatible for FK relationship."""
        # Normalize types
        type_a = type_a.lower()
        type_b = type_b.lower()
        
        # Integer types
        int_types = ['int', 'integer', 'bigint', 'smallint', 'tinyint']
        if any(t in type_a for t in int_types) and any(t in type_b for t in int_types):
            return True
        
        # String types
        str_types = ['char', 'varchar', 'text', 'string', 'nchar', 'nvarchar']
        if any(t in type_a for t in str_types) and any(t in type_b for t in str_types):
            return True
        
        # UUID types
        if 'uuid' in type_a and 'uuid' in type_b:
            return True
        
        return False
    
    def _calculate_confidence(
        self,
        overlap_rate: float,
        cardinality: str,
        fk_candidate: Dict[str, Any]
    ) -> str:
        """Calculate confidence level for relationship."""
        score = 0.0
        
        # Overlap rate contribution (max 50 points)
        score += overlap_rate * 50
        
        # Cardinality contribution (max 20 points)
        if cardinality in ('many_to_one', 'one_to_one'):
            score += 20
        elif cardinality == 'one_to_many':
            score += 10
        
        # Name matching contribution (max 30 points)
        if fk_candidate.get('suffix_match'):
            score += 20
        if fk_candidate.get('name_similarity', 0) > 0.7:
            score += 10
        
        # Determine confidence level
        if score >= 70:
            return 'high'
        elif score >= 50:
            return 'medium'
        else:
            return 'low'
    
    def detect_relationships(
        self,
        discovery_data: Dict[str, Any],
        overlap_threshold: float = 0.80
    ) -> List[Dict[str, Any]]:
        """
        Detect implicit relationships with COLLATION CONFLICT HANDLING.
        
        Args:
            discovery_data: Discovery JSON from Phase 1
            overlap_threshold: Minimum overlap rate (default 0.80)
        
        Returns:
            List of inferred relationships
        """
        if not self.engine:
            self.connect()
        
        relationships = []
        
        try:
            # Build list of all tables
            all_tables = []
            for schema in discovery_data.get('schemas', []):
                schema_name = schema['name']
                for table in schema.get('tables', []):
                    table_info = {
                        'schema': schema_name,
                        'name': table['name'],
                        'full_name': f"{schema_name}.{table['name']}",
                        'columns': table.get('columns', []),
                        'primary_key': table.get('primary_key', [])
                    }
                    all_tables.append(table_info)
            
            total_pairs = len(all_tables) * (len(all_tables) - 1)
            processed = 0
            
            logger.info(f"Analyzing {total_pairs} table pairs for relationships...")
            
            # Compare each table pair
            for table_a in all_tables:
                for table_b in all_tables:
                    if table_a['full_name'] == table_b['full_name']:
                        continue
                    
                    # Find FK candidates
                    for col_a in table_a['columns']:
                        for col_b in table_b['columns']:
                            # Check if col_b is a primary key
                            if col_b['name'] not in table_b['primary_key']:
                                continue
                            
                            # Check type compatibility
                            if not self._types_compatible(
                                col_a.get('type', ''),
                                col_b.get('type', '')
                            ):
                                continue
                            
                            # Calculate overlap with collation handling
                            overlap = self._calculate_overlap(
                                table_a['schema'], table_a['name'], col_a['name'],
                                table_b['schema'], table_b['name'], col_b['name']
                            )
                            
                            if overlap >= overlap_threshold:
                                # Determine cardinality with collation handling
                                cardinality = self._determine_cardinality(
                                    table_a['schema'], table_a['name'], col_a['name'],
                                    table_b['schema'], table_b['name'], col_b['name']
                                )
                                
                                fk_candidate = {
                                    'col_a': col_a['name'],
                                    'col_b': col_b['name']
                                }
                                
                                relationship = {
                                    'from': f"{table_a['full_name']}.{col_a['name']}",
                                    'to': f"{table_b['full_name']}.{col_b['name']}",
                                    'method': 'value_overlap',
                                    'overlap_rate': round(overlap, 3),
                                    'cardinality': cardinality,
                                    'confidence': self._calculate_confidence(
                                        overlap, cardinality, fk_candidate
                                    )
                                }
                                
                                relationships.append(relationship)
                                logger.info(
                                    f"Found relationship: {relationship['from']} -> "
                                    f"{relationship['to']} (overlap={overlap:.2%}, {cardinality})"
                                )
                    
                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Progress: {processed}/{total_pairs} pairs analyzed")
            
            logger.info(f"Relationship detection complete: {len(relationships)} relationships found")
        
        except Exception as e:
            logger.error(f"Error in relationship detection: {e}")
        
        return relationships


# Usage note:
"""
COLLATION CONFLICT FIX:

The key change is in the JOIN conditions:

SQL Server:
    ON a.val COLLATE DATABASE_DEFAULT = b.val COLLATE DATABASE_DEFAULT

PostgreSQL:
    ON a.val::text = b.val::text

MySQL:
    ON BINARY a.val = BINARY b.val

This forces both sides of the comparison to use the same collation,
preventing errors like:
    "Cannot resolve the collation conflict between Greek_CI_AS 
     and SQL_Latin1_General_CP1_CI_AI"

The DATABASE_DEFAULT collation is a safe choice as it uses the 
database's default collation, which both columns can be cast to.

Alternative approaches:
1. Use specific collation: COLLATE Latin1_General_CI_AS
2. Use case-sensitive collation: COLLATE Latin1_General_CS_AS
3. Use binary comparison: CAST(a.val AS VARBINARY) = CAST(b.val AS VARBINARY)

DATABASE_DEFAULT is preferred because:
- Works across different database collations
- Doesn't require knowing specific collation names
- Handles mixed collations gracefully
- Standard SQL Server recommendation
"""