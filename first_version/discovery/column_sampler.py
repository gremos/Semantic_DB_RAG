from typing import Dict, Any, List, Tuple, Set
from connectors.base import DatabaseConnector
import logging
import re

logger = logging.getLogger(__name__)

class ColumnSampler:
    """Intelligently sample and classify columns for semantic understanding."""
    
    # EXPANDED: Keywords for different semantic roles
    NAME_KEYWORDS = ['name', 'title', 'label', 'description', 'desc']
    ID_KEYWORDS = ['id', 'code', 'number', 'num', 'key']
    STATUS_KEYWORDS = ['status', 'state', 'type', 'category', 'kind', 'flag', 
                      'active', 'enabled', 'cancelled', 'deleted', 'is_', 'has_']
    DATE_KEYWORDS = ['date', 'time', 'timestamp', 'created', 'modified', 'updated']
    AMOUNT_KEYWORDS = ['amount', 'price', 'cost', 'total', 'sum', 'revenue', 
                       'value', 'quantity', 'qty', 'count']
    
    # Text types that should be sampled
    TEXT_TYPES = ['varchar', 'char', 'nvarchar', 'nchar', 'text', 'string']
    
    def __init__(self, connector: DatabaseConnector, max_samples_per_table: int = 100):
        self.connector = connector
        self.max_samples_per_table = max_samples_per_table
    
    def enrich_discovery_with_samples(
        self, 
        discovery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        MAIN ENTRY POINT: Enrich discovery JSON with column samples and classifications.
        This ensures column metadata flows through to semantic model.
        
        Returns:
            Enhanced discovery_data with column_samples added to each table
        """
        targets, classifications = self.identify_and_classify_columns(discovery_data)
        samples = self.sample_columns(targets, classifications)
        
        # CRITICAL: Inject samples back into discovery JSON
        for schema in discovery_data.get("schemas", []):
            for table in schema.get("tables", []):
                full_table = f"{schema['name']}.{table['name']}"
                
                # Add enriched column metadata
                for column in table.get("columns", []):
                    full_col = f"{full_table}.{column['name']}"
                    
                    # Add classification
                    if full_col in classifications:
                        column["classification"] = classifications[full_col]
                    
                    # Add sample values and stats
                    if full_col in samples and samples[full_col]["values"]:
                        column["sample_values"] = samples[full_col]["values"]
                        column["distinct_count"] = samples[full_col].get("distinct_count", 0)
                        
                        # Calculate distribution for categorical columns
                        if samples[full_col]["values"]:
                            value_counts = {}
                            total = len(samples[full_col]["values"])
                            for val in samples[full_col]["values"]:
                                value_counts[str(val)] = value_counts.get(str(val), 0) + 1
                            
                            column["value_distribution"] = {
                                k: round(v / total, 3) 
                                for k, v in value_counts.items()
                            }
        
        # Add negative findings
        discovery_data["negative_findings"] = self._identify_negative_findings(discovery_data)
        
        logger.info(f"Enriched discovery with column samples and classifications")
        return discovery_data
    
    def _identify_negative_findings(self, discovery_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Identify what was NOT found (critical for Q&A validation).
        """
        findings = {}
        
        # Check for common tables that don't exist
        all_tables = set()
        for schema in discovery_data.get("schemas", []):
            for table in schema.get("tables", []):
                all_tables.add(table["name"].lower())
        
        # Common business tables
        missing_tables = []
        for expected in ['refunds', 'returns', 'credits', 'adjustments']:
            if not any(expected in t for t in all_tables):
                missing_tables.append(expected)
        
        if missing_tables:
            findings["missing_tables"] = f"No tables found matching: {', '.join(missing_tables)}"
        
        # Check for multi-currency indicators
        has_currency_col = False
        for schema in discovery_data.get("schemas", []):
            for table in schema.get("tables", []):
                for col in table.get("columns", []):
                    if 'currency' in col["name"].lower():
                        has_currency_col = True
                        break
        
        if not has_currency_col:
            findings["currency"] = "NO CURRENCY COLUMNS DETECTED - assuming single currency"
        
        return findings
    
    def identify_and_classify_columns(
        self, 
        discovery_data: Dict[str, Any]
    ) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Dict[str, Any]]]:
        """
        Identify columns to sample AND classify them by semantic role.
        
        Returns:
            (sample_targets, column_classifications)
            
            sample_targets: List of (schema, table, column, semantic_role)
            column_classifications: {
                "schema.table.column": {
                    "semantic_role": "name|identifier|status|date|amount|description",
                    "priority": "critical|high|medium|low",
                    "nl_aliases": ["product name", "name of product", ...]
                }
            }
        """
        targets = []
        classifications = {}
        
        for schema in discovery_data.get("schemas", []):
            schema_name = schema["name"]
            
            for table in schema.get("tables", []):
                table_name = table["name"]
                full_table = f"{schema_name}.{table_name}"
                
                # Get primary keys and foreign keys for this table
                pk_cols = set(table.get("primary_key", []))
                fk_cols = set(fk["column"] for fk in table.get("foreign_keys", []))
                
                table_targets = []
                
                for column in table.get("columns", []):
                    col_name = column["name"]
                    col_type = column["type"]
                    full_col = f"{full_table}.{col_name}"
                    
                    # Classify this column
                    classification = self._classify_column(
                        col_name, col_type, 
                        col_name in pk_cols, 
                        col_name in fk_cols,
                        table_name
                    )
                    
                    if classification:
                        classifications[full_col] = classification
                        
                        # Add to sample targets
                        table_targets.append((
                            schema_name, 
                            table_name, 
                            col_name, 
                            classification["semantic_role"]
                        ))
                
                # Prioritize and limit per table
                table_targets = self._prioritize_targets(
                    table_targets, 
                    classifications,
                    self.max_samples_per_table
                )
                
                targets.extend(table_targets)
        
        logger.info(f"Identified {len(targets)} columns to sample across all tables")
        logger.info(f"Classified {len(classifications)} columns by semantic role")
        
        return targets, classifications
    
    def _classify_column(
        self,
        col_name: str,
        col_type: str,
        is_pk: bool,
        is_fk: bool,
        table_name: str
    ) -> Dict[str, Any]:
        """
        Classify column by semantic role and generate NL aliases.
        
        Returns classification dict or None if not interesting.
        """
        col_lower = col_name.lower()
        type_lower = col_type.lower()
        
        # Determine semantic role and priority
        semantic_role = None
        priority = "low"
        nl_aliases = []
        
        # PRIMARY KEY
        if is_pk:
            semantic_role = "primary_key"
            priority = "critical"
            nl_aliases = [
                f"{table_name} id",
                f"{table_name} identifier",
                f"id of {table_name}"
            ]
        
        # FOREIGN KEY
        elif is_fk:
            semantic_role = "foreign_key"
            priority = "high"
            # Extract referenced entity name (e.g., CustomerID -> Customer)
            ref_entity = self._extract_entity_from_fk(col_name)
            nl_aliases = [
                f"{ref_entity} id",
                f"id of {ref_entity}",
                f"{ref_entity} identifier"
            ]
        
        # NAME COLUMNS (highest priority for NL queries)
        elif any(kw in col_lower for kw in self.NAME_KEYWORDS):
            if 'name' in col_lower:
                semantic_role = "name"
                priority = "critical"
                # Generate rich aliases
                entity_name = table_name.lower()
                nl_aliases = [
                    f"{entity_name} name",
                    f"name of {entity_name}",
                    f"{entity_name} title",
                    "name",
                    "title"
                ]
            elif 'desc' in col_lower:
                semantic_role = "description"
                priority = "high"
                nl_aliases = [
                    f"{table_name} description",
                    f"description of {table_name}",
                    "description",
                    "details"
                ]
        
        # STATUS INDICATORS - CRITICAL FOR FILTERING
        elif any(kw in col_lower for kw in self.STATUS_KEYWORDS):
            semantic_role = "status_indicator"
            priority = "critical"  # Changed from "high" to "critical"
            nl_aliases = [
                "status",
                "state",
                f"{table_name} status",
                "active",
                "cancelled",
                "completed",
                "pending"
            ]
        
        # DATE COLUMNS
        elif any(kw in col_lower for kw in self.DATE_KEYWORDS):
            semantic_role = "timestamp"
            priority = "medium"
            nl_aliases = [
                self._generate_date_aliases(col_name, table_name)
            ][0]
        
        # AMOUNT COLUMNS (measures)
        elif any(kw in col_lower for kw in self.AMOUNT_KEYWORDS):
            if any(num_type in type_lower for num_type in ['int', 'decimal', 'numeric', 'float', 'money']):
                semantic_role = "measure_component"
                priority = "high"
                nl_aliases = [
                    self._generate_amount_aliases(col_name, table_name)
                ][0]
        
        # GENERIC TEXT COLUMNS (might contain searchable data)
        elif any(t in type_lower for t in self.TEXT_TYPES):
            semantic_role = "text_attribute"
            priority = "low"
            nl_aliases = [col_name.lower()]
        
        # Skip columns we don't care about
        else:
            return None
        
        return {
            "semantic_role": semantic_role,
            "priority": priority,
            "nl_aliases": list(set(nl_aliases)),  # Remove duplicates
            "is_pk": is_pk,
            "is_fk": is_fk,
            "should_sample": self._should_sample_values(semantic_role, col_type)
        }
    
    def _extract_entity_from_fk(self, fk_name: str) -> str:
        """Extract entity name from FK column (e.g., CustomerID -> Customer)."""
        # Remove common suffixes
        name = re.sub(r'(ID|Id|_id|_ID)$', '', fk_name)
        return name if name else fk_name
    
    def _generate_date_aliases(self, col_name: str, table_name: str) -> List[str]:
        """Generate natural language aliases for date columns."""
        col_lower = col_name.lower()
        aliases = []
        
        if 'created' in col_lower:
            aliases.extend([
                f"when was {table_name} created",
                "creation date",
                "created on",
                "date created"
            ])
        elif 'modified' in col_lower or 'updated' in col_lower:
            aliases.extend([
                f"when was {table_name} updated",
                "last updated",
                "modification date"
            ])
        else:
            aliases.append(col_name.lower().replace('_', ' '))
        
        return aliases
    
    def _generate_amount_aliases(self, col_name: str, table_name: str) -> List[str]:
        """Generate natural language aliases for amount columns."""
        col_lower = col_name.lower()
        aliases = []
        
        if 'price' in col_lower:
            aliases.extend(["price", "cost", "unit price"])
        elif 'quantity' in col_lower or 'qty' in col_lower:
            aliases.extend(["quantity", "amount", "count", "number of"])
        elif 'total' in col_lower:
            aliases.extend(["total", "sum", f"total {table_name}"])
        elif 'revenue' in col_lower:
            aliases.extend(["revenue", "sales", "income"])
        else:
            aliases.append(col_name.lower().replace('_', ' '))
        
        return aliases
    
    def _should_sample_values(self, semantic_role: str, col_type: str) -> bool:
        """Determine if we should sample actual values for this column."""
        # Sample values for these roles
        sample_roles = ['status_indicator', 'name', 'description']
        if semantic_role in sample_roles:
            return True
        
        # Sample text columns under 255 chars
        if any(t in col_type.lower() for t in ['varchar', 'char', 'text']):
            # Extract length if specified (e.g., varchar(50))
            match = re.search(r'\((\d+)\)', col_type)
            if match:
                length = int(match.group(1))
                return length <= 255
            return True
        
        return False
    
    def _prioritize_targets(
        self,
        targets: List[Tuple[str, str, str, str]],
        classifications: Dict[str, Dict[str, Any]],
        limit: int
    ) -> List[Tuple[str, str, str, str]]:
        """Prioritize columns within a table based on priority and role."""
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        def sort_key(target):
            schema, table, col, role = target
            full_col = f"{schema}.{table}.{col}"
            classification = classifications.get(full_col, {})
            priority = classification.get("priority", "low")
            return priority_order.get(priority, 99)
        
        sorted_targets = sorted(targets, key=sort_key)
        return sorted_targets[:limit]
    
    def sample_columns(
        self, 
        targets: List[Tuple[str, str, str, str]],
        classifications: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Sample values from identified columns.
        
        Returns:
            Enhanced column samples with classifications
        """
        samples = {}
        
        for schema, table, column, semantic_role in targets:
            full_col = f"{schema}.{table}.{column}"
            classification = classifications.get(full_col, {})
            
            # Only sample if needed
            if not classification.get("should_sample", False):
                samples[full_col] = {
                    "values": None,
                    "classification": classification
                }
                continue
            
            try:
                values = self._sample_column_values(schema, table, column)
                samples[full_col] = {
                    "values": values,
                    "classification": classification,
                    "distinct_count": len(set(values)) if values else 0
                }
                
                if values:
                    logger.debug(f"  Sampled {len(values)} values from {full_col}")
            
            except Exception as e:
                logger.warning(f"  Failed to sample {full_col}: {e}")
                samples[full_col] = {
                    "values": None,
                    "classification": classification
                }
        
        logger.info(f"Sampled {sum(1 for s in samples.values() if s['values'])} columns successfully")
        return samples
    
    def _sample_column_values(
        self, 
        schema: str, 
        table: str, 
        column: str, 
        limit: int = 20
    ) -> List[Any]:
        """Sample distinct values from a column."""
        try:
            with self.connector.engine.connect() as conn:
                from sqlalchemy import text
                
                # Get distinct values (limit to 20)
                query = text(f"""
                    SELECT DISTINCT TOP {limit} [{column}]
                    FROM [{schema}].[{table}]
                    WHERE [{column}] IS NOT NULL
                    ORDER BY [{column}]
                """)
                
                result = conn.execute(query)
                values = [row[0] for row in result]
                return values
        except Exception as e:
            logger.warning(f"Sample query failed for {schema}.{table}.{column}: {e}")
            return []