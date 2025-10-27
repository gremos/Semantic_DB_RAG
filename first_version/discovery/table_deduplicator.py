from typing import List, Dict, Any, Set
import re
from difflib import SequenceMatcher

class TableDeduplicator:
    """Identify and exclude duplicate/copy tables."""
    
    def __init__(self, exclusion_patterns: List[str]):
        self.exclusion_patterns = [re.compile(p) for p in exclusion_patterns if p]
    
    def filter_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out duplicate and backup tables.
        
        Returns: Filtered list of tables
        """
        # Step 1: Filter by regex patterns
        filtered = []
        for table in tables:
            table_name = table['name'].lower()
            
            # Check exclusion patterns
            if any(pattern.match(table_name) for pattern in self.exclusion_patterns):
                continue
            
            filtered.append(table)
        
        # Step 2: Detect structural duplicates
        filtered = self._remove_structural_duplicates(filtered)
        
        return filtered
    
    def _remove_structural_duplicates(
        self, 
        tables: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove tables that are structural copies of others.
        Keeps the table with the most rows.
        """
        # Group by similar names
        groups = self._group_similar_names(tables)
        
        result = []
        for group in groups:
            if len(group) == 1:
                result.append(group[0])
            else:
                # Keep the one with most rows, or alphabetically first
                best = max(group, key=lambda t: (
                    t.get('rowcount_sample', 0) or 0,
                    -len(t['name'])  # Prefer shorter names
                ))
                result.append(best)
        
        return result
    
    def _group_similar_names(
        self, 
        tables: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group tables with similar names."""
        groups = []
        processed = set()
        
        for i, table1 in enumerate(tables):
            if table1['name'] in processed:
                continue
            
            group = [table1]
            processed.add(table1['name'])
            
            # Find similar tables
            for j, table2 in enumerate(tables[i+1:], i+1):
                if table2['name'] in processed:
                    continue
                
                if self._are_similar(table1['name'], table2['name']):
                    group.append(table2)
                    processed.add(table2['name'])
            
            groups.append(group)
        
        return groups
    
    def _are_similar(self, name1: str, name2: str) -> bool:
        """Check if two table names are similar (likely copies)."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Remove common suffixes/prefixes
        suffixes = ['_copy', '_backup', '_old', '_archive', '_bak', '_new']
        prefixes = ['copy_', 'backup_', 'old_', 'archive_', 'bak_', 'new_']
        
        clean1 = name1_lower
        clean2 = name2_lower
        
        for suffix in suffixes:
            clean1 = clean1.replace(suffix, '')
            clean2 = clean2.replace(suffix, '')
        
        for prefix in prefixes:
            clean1 = clean1.replace(prefix, '')
            clean2 = clean2.replace(prefix, '')
        
        # Remove dates (8 digits)
        clean1 = re.sub(r'_?\d{6,8}', '', clean1)
        clean2 = re.sub(r'_?\d{6,8}', '', clean2)
        
        # Check similarity
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return similarity > 0.85  # 85% similar