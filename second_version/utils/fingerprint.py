import hashlib
from typing import Dict, Any

def calculate_db_fingerprint(connection_string: str, discovery_data: Dict[str, Any]) -> str:
    """
    Calculate fingerprint for database state.
    
    Uses connection string + row counts + schema structure.
    """
    components = [connection_string]
    
    # Add schema/table structure
    for schema in discovery_data.get("schemas", []):
        components.append(schema["name"])
        for table in schema.get("tables", []):
            components.append(f"{table['name']}:{table.get('rowcount_sample', 0)}")
    
    fingerprint_str = "|".join(str(c) for c in components)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()