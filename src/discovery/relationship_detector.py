"""
Relationship detector for inferring implicit foreign keys.
Uses value overlap (>80%), m:1 cardinality pattern, and suffix matching.
"""

from typing import Dict, List, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import Settings


class RelationshipDetector:
    """Detect implicit foreign key relationships between tables."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = None
    
    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.engine = create_engine(
                self.settings.DATABASE_CONNECTION_STRING,
                pool_pre_ping=True,
                echo=False
            )
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
    
    def detect_relationships(
        self, 
        discovery_data: Dict[str, Any],
        overlap_threshold: float = 0.80
    ) -> List[Dict[str, Any]]:
        """
        Detect implicit relationships across all tables.
        Returns list of inferred relationships.
        """
        if not self.engine:
            self.connect()
        
        relationships = []
        
        # Build list of all tables with their columns
        all_tables = []
        for schema in discovery_data.get('schemas', []):
            schema_name = schema['name']
            for table in schema.get('tables', []):
                table_info = {
                    'schema': schema_name,
                    'name': table['name'],
                    'full_name': f"{schema_name}.{table['name']}",
                    'columns': table['columns'],
                    'primary_key': table.get('primary_key', [])
                }
                all_tables.append(table_info)
        
        # Compare all table pairs
        for i, table_a in enumerate(all_tables):
            for table_b in all_tables[i+1:]:
                # Check for potential FK columns between these tables
                potential_fks = self._find_potential_fks(table_a, table_b)
                
                for fk_candidate in potential_fks:
                    # Calculate overlap
                    overlap = self._calculate_overlap(
                        table_a['schema'], table_a['name'], fk_candidate['col_a'],
                        table_b['schema'], table_b['name'], fk_candidate['col_b']
                    )
                    
                    if overlap >= overlap_threshold:
                        # Check cardinality
                        cardinality = self._determine_cardinality(
                            table_a['schema'], table_a['name'], fk_candidate['col_a'],
                            table_b['schema'], table_b['name'], fk_candidate['col_b']
                        )
                        
                        # Determine direction (many-to-one preferred)
                        from_col = f"{table_a['full_name']}.{fk_candidate['col_a']}"
                        to_col = f"{table_b['full_name']}.{fk_candidate['col_b']}"
                        
                        if cardinality == 'one_to_many':
                            # Reverse direction
                            from_col, to_col = to_col, from_col
                            cardinality = 'many_to_one'
                        
                        relationship = {
                            'from': from_col,
                            'to': to_col,
                            'method': 'value_overlap',
                            'overlap_rate': round(overlap, 3),
                            'cardinality': cardinality,
                            'confidence': self._calculate_confidence(overlap, cardinality, fk_candidate)
                        }
                        
                        relationships.append(relationship)
        
        return relationships
    
    def _find_potential_fks(
        self, 
        table_a: Dict[str, Any], 
        table_b: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Find potential FK column pairs based on:
        - Name similarity (suffix matching)
        - Type compatibility
        """
        candidates = []
        
        for col_a in table_a['columns']:
            for col_b in table_b['columns']:
                # Check if names suggest a relationship
                if self._names_suggest_relationship(
                    col_a['name'], 
                    col_b['name'],
                    table_b['primary_key']
                ):
                    # Check type compatibility
                    if self._types_compatible(col_a['type'], col_b['type']):
                        candidates.append({
                            'col_a': col_a['name'],
                            'col_b': col_b['name'],
                            'name_match': True
                        })
        
        return candidates
    
    def _names_suggest_relationship(
        self, 
        name_a: str, 
        name_b: str,
        pk_columns: List[str]
    ) -> bool:
        """Check if column names suggest a FK relationship."""
        # Exact match
        if name_a.lower() == name_b.lower():
            return True
        
        # name_b is a PK and name_a contains it
        if name_b in pk_columns:
            if name_b.lower() in name_a.lower():
                return True
            # Common patterns: CustomerID -> Customer, CustomerId -> CustomerId
            if name_a.lower().endswith('id') and name_b.lower().endswith('id'):
                base_a = name_a[:-2].lower()
                base_b = name_b[:-2].lower()
                if base_a == base_b or base_a in base_b or base_b in base_a:
                    return True
        
        return False
    
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
        str_types = ['char', 'varchar', 'text', 'string']
        if any(t in type_a for t in str_types) and any(t in type_b for t in str_types):
            return True
        
        # UUID types
        if 'uuid' in type_a and 'uuid' in type_b:
            return True
        
        return False
    
    def _calculate_overlap(
        self,
        schema_a: str, table_a: str, col_a: str,
        schema_b: str, table_b: str, col_b: str
    ) -> float:
        """
        Calculate value overlap rate between two columns.
        Returns percentage of values in col_a that exist in col_b.
        """
        if not self.engine:
            return 0.0
        
        try:
            with self.engine.connect() as conn:
                query = text(f"""
                    WITH sample_a AS (
                        SELECT DISTINCT TOP 1000 "{col_a}" AS val
                        FROM "{schema_a}"."{table_a}"
                        WHERE "{col_a}" IS NOT NULL
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
                    return row[1] / row[0]
                
        except SQLAlchemyError:
            pass
        
        return 0.0
    
    def _determine_cardinality(
        self,
        schema_a: str, table_a: str, col_a: str,
        schema_b: str, table_b: str, col_b: str
    ) -> str:
        """
        Determine cardinality of relationship.
        Returns: one_to_one, one_to_many, many_to_one, many_to_many
        """
        if not self.engine:
            return 'many_to_many'
        
        try:
            with self.engine.connect() as conn:
                # Check distinctness in both directions
                query = text(f"""
                    WITH sample_a AS (
                        SELECT TOP 1000 "{col_a}" AS val
                        FROM "{schema_a}"."{table_a}"
                        WHERE "{col_a}" IS NOT NULL
                    ),
                    sample_b AS (
                        SELECT TOP 1000 "{col_b}" AS val
                        FROM "{schema_b}"."{table_b}"
                        WHERE "{col_b}" IS NOT NULL
                    )
                    SELECT
                        COUNT(*) AS count_a,
                        COUNT(DISTINCT "{col_a}") AS distinct_a,
                        (SELECT COUNT(DISTINCT val) FROM sample_b) AS distinct_b,
                        (SELECT COUNT(*) FROM sample_b) AS count_b
                    FROM sample_a
                """)
                
                result = conn.execute(query)
                row = result.fetchone()
                
                if row:
                    count_a, distinct_a, distinct_b, count_b = row
                    
                    # Calculate uniqueness ratios
                    unique_a = distinct_a / count_a if count_a > 0 else 0
                    unique_b = distinct_b / count_b if count_b > 0 else 0
                    
                    # Determine cardinality
                    if unique_a > 0.95 and unique_b > 0.95:
                        return 'one_to_one'
                    elif unique_a > 0.95:
                        return 'one_to_many'  # A is unique, B has duplicates
                    elif unique_b > 0.95:
                        return 'many_to_one'  # B is unique, A has duplicates
                    else:
                        return 'many_to_many'
        
        except SQLAlchemyError:
            pass
        
        return 'many_to_many'
    
    def _calculate_confidence(
        self, 
        overlap: float, 
        cardinality: str,
        fk_candidate: Dict[str, Any]
    ) -> str:
        """
        Calculate confidence level for inferred relationship.
        Returns: high, medium, low
        """
        score = 0.0
        
        # Overlap score (50%)
        score += overlap * 0.5
        
        # Cardinality score (30%)
        if cardinality in ['many_to_one', 'one_to_one']:
            score += 0.3
        elif cardinality == 'one_to_many':
            score += 0.2
        
        # Name match score (20%)
        if fk_candidate.get('name_match'):
            score += 0.2
        
        if score >= 0.85:
            return 'high'
        elif score >= 0.70:
            return 'medium'
        else:
            return 'low'
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
