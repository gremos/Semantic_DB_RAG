"""
Incremental Semantic Modeler - Phase 2: Build Semantic Model from Discovery
Uses LLM for entity identification and incorporates semantic relationships from views/RDLs.
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class IncrementalModeler:
    """
    Builds semantic model incrementally:
    1. Identify entities (ALWAYS use LLM, never code heuristics)
    2. Identify dimensions
    3. Identify facts and measures
    4. Infer relationships (FK + naming + semantic from views/RDLs)
    5. Generate metrics
    6. Assemble and validate
    """
    
    def __init__(self, llm, discovery_json: dict):
        self.llm = llm
        self.discovery_json = discovery_json
        
        # Extract all tables from all schemas
        self.all_tables = []
        for schema in discovery_json.get("schemas", []):
            for table in schema.get("tables", []):
                # Add schema prefix to table name for uniqueness
                table_with_schema = table.copy()
                table_with_schema["full_name"] = f"{schema['name']}.{table['name']}"
                table_with_schema["schema"] = schema["name"]
                self.all_tables.append(table_with_schema)
        
        # Build lookup maps
        self.table_lookup = {t["full_name"]: t for t in self.all_tables}
        
        # Extract semantic relationships (from views/RDLs)
        self.semantic_relationships = discovery_json.get("semantic_relationships", [])
        
        # Build view usage frequency map
        self.view_usage = self._build_view_usage_map()
        
        # Components to build
        self.entities = []
        self.dimensions = []
        self.facts = []
        self.relationships = []
        self.metrics = []
    
    def build_semantic_model(self) -> dict:
        """
        Main entry point: build complete semantic model.
        
        Returns: Semantic Model JSON
        """
        logger.info("Starting incremental semantic modeling...")
        
        # Phase 1: Identify entities (LLM-based, NEVER code-based)
        logger.info("=== PHASE 1: ENTITY IDENTIFICATION (LLM) ===")
        self.entities = self._identify_entities_with_llm()
        logger.info(f"Identified {len(self.entities)} entities")
        
        # Phase 2: Identify dimensions
        logger.info("=== PHASE 2: DIMENSION IDENTIFICATION ===")
        self.dimensions = self._identify_dimensions()
        logger.info(f"Identified {len(self.dimensions)} dimensions")
        
        # Phase 3: Identify facts and measures
        logger.info("=== PHASE 3: FACT & MEASURE IDENTIFICATION ===")
        self.facts = self._identify_facts_and_measures()
        logger.info(f"Identified {len(self.facts)} facts")
        
        # Phase 4: Infer relationships
        logger.info("=== PHASE 4: RELATIONSHIP INFERENCE ===")
        self.relationships = self._infer_all_relationships()
        logger.info(f"Inferred {len(self.relationships)} relationships")
        
        # Phase 4.5: Add semantic relationships from views/RDLs
        logger.info("=== PHASE 4.5: SEMANTIC RELATIONSHIP ENRICHMENT ===")
        self._enrich_with_semantic_relationships()
        logger.info(f"Total relationships after enrichment: {len(self.relationships)}")
        
        # Phase 5: Generate metrics
        logger.info("=== PHASE 5: METRIC GENERATION ===")
        self.metrics = self._generate_metrics()
        logger.info(f"Generated {len(self.metrics)} metrics")
        
        # Phase 6: Assemble and validate
        logger.info("=== PHASE 6: MODEL ASSEMBLY & VALIDATION ===")
        semantic_model = self._assemble_final_model()
        
        return semantic_model
    
    def _build_view_usage_map(self) -> dict:
        """
        Build map of table -> usage statistics.
        Tables used in more views/SPs/RDLs are more authoritative.
        """
        usage = {}
        
        for table in self.all_tables:
            table_name = table["full_name"]
            source_assets = table.get("source_assets", [])
            
            view_count = len([a for a in source_assets if a.get("kind") == "view"])
            sp_count = len([a for a in source_assets if a.get("kind") == "stored_procedure"])
            rdl_count = len([a for a in source_assets if a.get("kind") == "rdl"])
            
            usage[table_name] = {
                "view_count": view_count,
                "stored_procedure_count": sp_count,
                "rdl_count": rdl_count,
                "total_references": view_count + sp_count + rdl_count
            }
        
        return usage
    
    def _identify_entities_with_llm(self) -> List[dict]:
        """
        PHASE 1: Use LLM to identify entities with full context.
        NEVER use code-based heuristics for entity classification.
        
        Returns: List of entity definitions
        """
        
        # Build rich context for LLM
        context = {
            "tables": [
                {
                    "name": t["full_name"],
                    "columns": [c["name"] for c in t.get("columns", [])],
                    "column_types": {c["name"]: c["type"] for c in t.get("columns", [])},
                    "row_count": t.get("rowcount_sample", 0),
                    "primary_key": t.get("primary_key", []),
                    "foreign_keys": [
                        f"{fk['column']} -> {fk['ref_table']}.{fk['ref_column']}"
                        for fk in t.get("foreign_keys", [])
                    ],
                    "view_usage": self.view_usage.get(t["full_name"], {})
                }
                for t in self.all_tables[:50]  # Limit to avoid token overflow
            ],
            "semantic_relationships": self.semantic_relationships[:20],  # Sample
            "total_table_count": len(self.all_tables)
        }
        
        prompt = f"""
You are a data modeling expert analyzing a database to identify ENTITIES (master data, dimensions).

Context (showing first 50 of {len(self.all_tables)} tables):
{json.dumps(context, indent=2)}

**ENTITY IDENTIFICATION RULES:**

1. **Entities are master data tables** containing relatively stable, reference data:
   - Customers, Products, Employees, Locations, Categories, etc.
   - NOT transactional tables (Orders, Invoices, Logs, Events)

2. **Key indicators of entities:**
   - High view_usage (frequently referenced in views/stored procedures)
   - Referenced by many foreign keys from other tables
   - Descriptive column names (Name, Title, Description, etc.)
   - Relatively low row counts (master data, not transaction data)
   - Contains identification/classification data

3. **Look for these patterns:**
   - Tables with "Dim" prefix (e.g., DimCustomer, DimProduct)
   - Tables ending in "Type", "Category", "Status" (lookup tables)
   - Tables with people/organization names (Customer, Employee, Vendor)
   - Tables with geographic data (Location, Region, Country)
   - Tables with product/service data (Product, Service, Item)

4. **Avoid classifying these as entities:**
   - Fact/transaction tables (high row counts, temporal data)
   - Junction/bridge tables (only foreign keys, no descriptive attributes)
   - Audit/log tables (CreatedAt, ModifiedAt, LogEntry, etc.)
   - Temporary/staging tables

**OUTPUT JSON SCHEMA:**
{{
  "entities": [
    {{
      "name": "Customer",
      "source": "dbo.BusinessPoint",
      "primary_key": ["ID"],
      "business_meaning": "Represents a customer organization",
      "confidence": 0.95,
      "evidence": [
        "Referenced in 12 views",
        "Has descriptive columns: BrandName, ContactInfo",
        "Stable master data pattern",
        "Foreign key target for 5 other tables"
      ]
    }}
  ]
}}

**IMPORTANT:** Only include tables you are highly confident are entities (confidence > 0.7).
Output valid JSON only, no markdown or explanations.
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            entities = result.get("entities", [])
            
            # Log sample for debugging
            if entities:
                logger.info(f"Sample entity: {entities[0]}")
            
            return entities
        
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON for entity identification: {e}")
            logger.error(f"Response content: {response.content[:500]}")
            return []
        
        except Exception as e:
            logger.error(f"Entity identification failed: {e}", exc_info=True)
            return []
    
    def _identify_dimensions(self) -> List[dict]:
        """
        PHASE 2: Identify dimension tables (specialized entities for analytical slicing).
        
        Dimensions are:
        - Time/Date dimensions
        - Geography dimensions
        - Product hierarchies
        - Organization hierarchies
        """
        dimensions = []
        
        # Look for date/time dimensions
        for table in self.all_tables:
            table_name = table["full_name"]
            columns = [c["name"].lower() for c in table.get("columns", [])]
            
            # Date dimension patterns
            is_date_dim = (
                any(keyword in table_name.lower() for keyword in ["date", "time", "calendar"]) or
                all(col in columns for col in ["year", "month", "day"])
            )
            
            if is_date_dim:
                date_attributes = [
                    c["name"] for c in table.get("columns", [])
                    if any(kw in c["name"].lower() for kw in ["year", "quarter", "month", "week", "day", "date"])
                ]
                
                dimensions.append({
                    "name": table["name"],
                    "source": table_name,
                    "type": "date",
                    "keys": table.get("primary_key", []),
                    "attributes": date_attributes,
                    "business_meaning": "Time dimension for date-based analysis"
                })
                continue
            
            # Geography dimension patterns
            is_geo_dim = any(
                keyword in table_name.lower() 
                for keyword in ["geography", "location", "region", "country", "city"]
            )
            
            if is_geo_dim:
                geo_attributes = [
                    c["name"] for c in table.get("columns", [])
                    if any(kw in c["name"].lower() for kw in ["country", "region", "city", "state", "postal", "latitude", "longitude"])
                ]
                
                dimensions.append({
                    "name": table["name"],
                    "source": table_name,
                    "type": "geography",
                    "keys": table.get("primary_key", []),
                    "attributes": geo_attributes,
                    "business_meaning": "Geographic dimension for location-based analysis"
                })
        
        return dimensions
    
    def _identify_facts_and_measures(self) -> List[dict]:
        """
        PHASE 3: Identify fact tables and their measures.
        
        Facts are transactional tables with:
        - High row counts
        - Multiple foreign keys (to dimensions/entities)
        - Numeric measures (amounts, quantities, etc.)
        - Temporal columns (date/time)
        """
        facts = []
        processed_count = 0
        
        for table in self.all_tables:
            table_name = table["full_name"]
            
            # Skip if already identified as entity or dimension
            if self._is_entity_or_dimension(table_name):
                continue
            
            # Check for fact patterns
            has_fks = len(table.get("foreign_keys", [])) >= 2  # At least 2 FKs
            has_numeric_cols = self._has_numeric_columns(table)
            high_row_count = table.get("rowcount_sample", 0) > 1000  # Rough threshold
            
            is_fact = has_fks and has_numeric_cols
            
            if is_fact:
                # Identify measures
                measures = self._identify_measures_for_fact(table)
                
                if not measures:
                    logger.warning(f"Failed to identify measures for {table_name}, using defaults")
                    measures = self._get_default_measures(table)
                
                # Identify grain (composite key or all FKs)
                grain = table.get("primary_key", [])
                if not grain:
                    # Use foreign keys as grain
                    grain = [fk["column"] for fk in table.get("foreign_keys", [])]
                
                facts.append({
                    "name": table["name"],
                    "source": table_name,
                    "grain": grain,
                    "measures": measures,
                    "foreign_keys": [
                        {
                            "column": fk["column"],
                            "references": f"{fk['ref_table']}.{fk['ref_column']}"
                        }
                        for fk in table.get("foreign_keys", [])
                    ],
                    "business_meaning": f"Transactional data at {', '.join(grain)} grain"
                })
            
            processed_count += 1
            if processed_count % 10 == 0:
                logger.info(f"    Progress: {processed_count}/{len(self.all_tables)} facts processed")
        
        return facts
    
    def _identify_measures_for_fact(self, table: dict) -> List[dict]:
        """
        Use LLM to identify appropriate measures for a fact table.
        """
        
        numeric_columns = [
            {"name": c["name"], "type": c["type"]}
            for c in table.get("columns", [])
            if self._is_numeric_type(c["type"])
        ]
        
        if not numeric_columns:
            return []
        
        # Use LLM for measure identification
        context = {
            "table_name": table["full_name"],
            "numeric_columns": numeric_columns,
            "all_columns": [c["name"] for c in table.get("columns", [])]
        }
        
        prompt = f"""
You are a data modeling expert. Identify appropriate measures for this fact table.

Context:
{json.dumps(context, indent=2)}

Rules:
1. Measures are numeric columns that should be aggregated (SUM, AVG, COUNT, etc.)
2. Common measure patterns:
   - Amounts: TotalPrice, Revenue, Cost, Amount (use SUM)
   - Quantities: Quantity, Count, Units (use SUM)
   - Rates: Rate, Percentage, Ratio (use AVG)
3. Avoid: IDs, foreign keys, status codes (not measures)

Output JSON:
{{
  "measures": [
    {{
      "name": "Revenue",
      "expression": "SUM(TotalPrice)",
      "column": "TotalPrice",
      "aggregation": "SUM",
      "business_meaning": "Total revenue"
    }}
  ]
}}

Output valid JSON only.
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            return result.get("measures", [])
        
        except Exception as e:
            logger.warning(f"LLM measure identification failed: {e}")
            return []
    
    def _get_default_measures(self, table: dict) -> List[dict]:
        """
        Fallback: create default measures from numeric columns.
        """
        measures = []
        
        for col in table.get("columns", []):
            if self._is_numeric_type(col["type"]):
                col_name = col["name"].lower()
                
                # Skip ID columns
                if "id" in col_name:
                    continue
                
                # Guess aggregation
                if any(kw in col_name for kw in ["count", "quantity", "qty", "units"]):
                    agg = "SUM"
                elif any(kw in col_name for kw in ["price", "amount", "revenue", "cost", "total"]):
                    agg = "SUM"
                elif any(kw in col_name for kw in ["rate", "percent", "ratio", "average"]):
                    agg = "AVG"
                else:
                    agg = "SUM"  # Default
                
                measures.append({
                    "name": col["name"],
                    "expression": f"{agg}({col['name']})",
                    "column": col["name"],
                    "aggregation": agg,
                    "business_meaning": f"{agg} of {col['name']}"
                })
        
        return measures
    
    def _infer_all_relationships(self) -> List[dict]:
        """
        PHASE 4: Infer relationships from multiple sources:
        1. Foreign key constraints (highest confidence)
        2. Naming patterns (medium confidence)
        3. Semantic relationships from views/RDLs (added in Phase 4.5)
        """
        all_relationships = []
        
        # 1. From foreign key constraints
        for table in self.all_tables:
            fk_rels = self._infer_relationships_from_fks(table)
            all_relationships.extend(fk_rels)
        
        logger.info(f"    Found {len(all_relationships)} relationships from foreign keys")
        
        # 2. From naming patterns
        naming_rels = []
        for table in self.all_tables:
            naming_rels.extend(self._infer_relationships_from_naming(table))
        
        # Deduplicate (FK takes precedence over naming)
        existing_pairs = {(r["from"], r["to"]) for r in all_relationships}
        for rel in naming_rels:
            pair = (rel["from"], rel["to"])
            if pair not in existing_pairs:
                all_relationships.append(rel)
                existing_pairs.add(pair)
        
        logger.info(f"    Added {len(naming_rels)} relationships from naming patterns")
        logger.info(f"    Total relationships: {len(all_relationships)}")
        
        return all_relationships
    
    def _infer_relationships_from_fks(self, table: dict) -> List[dict]:
        """Infer relationships from foreign key constraints."""
        relationships = []
        
        for fk in table.get("foreign_keys", []):
            rel = {
                "from": f"{table['full_name']}.{fk['column']}",
                "to": f"{fk['ref_table']}.{fk['ref_column']}",
                "cardinality": "many-to-one",  # FK implies many-to-one
                "type": "foreign_key",  # FIX: Add type field
                "business_meaning": f"Each {table['name']} references one {fk['ref_table'].split('.')[-1]} via {fk['column']}",
                "confidence": 1.0,
                "source": "foreign_key_constraint"
            }
            relationships.append(rel)
        
        return relationships
    
    def _infer_relationships_from_naming(self, table: dict) -> List[dict]:
        """
        Infer relationships from column naming patterns.
        Example: CustomerID column likely references a Customer table.
        """
        relationships = []
        
        for col in table.get("columns", []):
            col_name = col["name"]
            
            # Skip if this is the primary key
            if col_name in table.get("primary_key", []):
                continue
            
            # Look for ID suffix pattern
            if col_name.endswith("ID"):
                ref_table_name = col_name[:-2]  # Remove "ID" suffix
                
                # Try to find matching table
                possible_refs = [
                    t["full_name"] for t in self.all_tables
                    if t["name"].lower() == ref_table_name.lower()
                ]
                
                if possible_refs:
                    ref_table = possible_refs[0]
                    
                    # Assume referenced table has "ID" as PK
                    rel = {
                        "from": f"{table['full_name']}.{col_name}",
                        "to": f"{ref_table}.ID",
                        "cardinality": "many-to-one",
                        "type": "inferred_from_naming",  # FIX: Add type field
                        "business_meaning": f"Each {table['name']} references one {ref_table_name} via {col_name}",
                        "confidence": 0.7,
                        "source": "naming_pattern"
                    }
                    relationships.append(rel)
        
        return relationships
    
    def _enrich_with_semantic_relationships(self):
        """
        PHASE 4.5: Add relationships extracted from views, stored procedures, and RDLs.
        These are HIGH CONFIDENCE because they represent actual usage.
        """
        
        if not self.semantic_relationships:
            logger.warning("No semantic relationships available from views/RDLs")
            return
        
        # Build set of existing relationship pairs
        existing_pairs = {(r["from"], r["to"]) for r in self.relationships}
        
        added_count = 0
        for sem_rel in self.semantic_relationships:
            pair = (sem_rel["from"], sem_rel["to"])
            
            if pair not in existing_pairs:
                # Add new relationship from semantic analysis
                self.relationships.append({
                    "from": sem_rel["from"],
                    "to": sem_rel["to"],
                    "cardinality": self._infer_cardinality_from_join(sem_rel.get("join_type", "INNER")),
                    "type": "semantic_from_views",  # FIX: Add type field
                    "business_meaning": f"Relationship used in {sem_rel.get('usage_count', 1)} views/stored procedures",
                    "confidence": sem_rel.get("confidence", 0.9),
                    "source": sem_rel.get("source_asset", "unknown"),
                    "usage_count": sem_rel.get("usage_count", 1)
                })
                added_count += 1
                existing_pairs.add(pair)
            else:
                # Relationship already exists - upgrade confidence and add metadata
                for rel in self.relationships:
                    if rel["from"] == sem_rel["from"] and rel["to"] == sem_rel["to"]:
                        rel["confidence"] = max(rel.get("confidence", 0), sem_rel.get("confidence", 0.9))
                        rel["validated_by_views"] = True
                        rel["view_usage_count"] = sem_rel.get("usage_count", 1)
                        break
        
        logger.info(f"    Added {added_count} new relationships from semantic analysis")
        logger.info(f"    Enriched existing relationships with view usage metadata")
    
    def _infer_cardinality_from_join(self, join_type: str) -> str:
        """Infer cardinality from JOIN type."""
        join_type = join_type.upper()
        
        if join_type == "INNER":
            return "many-to-one"  # Most common
        elif join_type == "LEFT":
            return "many-to-one"
        elif join_type == "RIGHT":
            return "one-to-many"
        else:
            return "many-to-one"  # Default
    
    def _generate_metrics(self) -> List[dict]:
        """
        PHASE 5: Generate business metrics using LLM.
        Metrics combine measures with business logic.
        """
        
        if not self.facts:
            logger.warning("No facts available for metric generation")
            return []
        
        # Build context
        context = {
            "entities": [
                {"name": e["name"], "source": e["source"]}
                for e in self.entities
            ],
            "facts": [
                {
                    "name": f["name"],
                    "measures": [m["name"] for m in f.get("measures", [])]
                }
                for f in self.facts
            ],
            "dimensions": [
                {"name": d["name"], "type": d.get("type", "generic")}
                for d in self.dimensions
            ]
        }
        
        prompt = f"""
You are a business intelligence expert. Generate useful metrics based on this semantic model.

Context:
{json.dumps(context, indent=2)}

Generate 3-5 business metrics that would be valuable for analytics. Examples:
- "Customer Lifetime Value": Total revenue per customer over time
- "Sales Growth Rate": Period-over-period revenue change
- "Upsell Opportunities": Customers who purchased recently but missing certain product categories

Output JSON:
{{
  "metrics": [
    {{
      "name": "Customer Lifetime Value",
      "logic": "SUM(Revenue) grouped by Customer over all time",
      "required_entities": ["Customer", "Sales"],
      "required_measures": ["Revenue"],
      "business_value": "Identifies most valuable customers"
    }}
  ]
}}

Output valid JSON only.
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            return result.get("metrics", [])
        
        except Exception as e:
            logger.warning(f"Metric generation failed: {e}")
            return []
    
    def _assemble_final_model(self) -> dict:
        """
        PHASE 6: Assemble all components into final semantic model JSON.
        CRITICAL: Validate that all relationships have required 'type' field.
        """
        logger.info("Assembling final semantic model...")
        
        # FIX: Ensure all relationships have required 'type' field
        validated_relationships = []
        
        for idx, rel in enumerate(self.relationships):
            # Ensure 'type' field exists
            if "type" not in rel:
                # Infer type based on available fields
                if rel.get("source") == "foreign_key_constraint":
                    rel["type"] = "foreign_key"
                elif rel.get("source") == "naming_pattern":
                    rel["type"] = "inferred_from_naming"
                elif "validated_by_views" in rel:
                    rel["type"] = "semantic_from_views"
                else:
                    rel["type"] = "derived"  # Default
                
                logger.warning(f"Added missing 'type' field to relationship {idx}: {rel['type']}")
            
            # Validate required fields
            required_fields = ["from", "to", "cardinality", "type"]
            if all(field in rel for field in required_fields):
                validated_relationships.append(rel)
            else:
                missing = [f for f in required_fields if f not in rel]
                logger.warning(f"Skipping invalid relationship (missing: {missing}): {rel}")
        
        logger.info(f"Validated {len(validated_relationships)}/{len(self.relationships)} relationships")
        
        # Log sample relationship for debugging
        if validated_relationships:
            logger.info(f"Sample relationship: {validated_relationships[0]}")
        
        # Assemble final model
        semantic_model = {
            "entities": self.entities,
            "dimensions": self.dimensions,
            "facts": self.facts,
            "relationships": validated_relationships,  # Use validated list
            "metrics": self.metrics,
            "audit": {
                "dialect": self.discovery_json.get("dialect", "unknown"),
                "generated_at": datetime.now().isoformat(),
                "table_count": len(self.all_tables),
                "entity_count": len(self.entities),
                "dimension_count": len(self.dimensions),
                "fact_count": len(self.facts),
                "relationship_count": len(validated_relationships),
                "semantic_relationship_count": len(self.semantic_relationships),
                "view_enriched": len([r for r in validated_relationships if r.get("type") == "semantic_from_views"])
            }
        }
        
        logger.info("Model assembly complete:")
        logger.info(f"  - Entities: {len(self.entities)}")
        logger.info(f"  - Dimensions: {len(self.dimensions)}")
        logger.info(f"  - Facts: {len(self.facts)}")
        logger.info(f"  - Relationships: {len(validated_relationships)}")
        logger.info(f"  - View-enriched relationships: {semantic_model['audit']['view_enriched']}")
        logger.info(f"  - Metrics: {len(self.metrics)}")
        
        return semantic_model
    
    # Helper methods
    
    def _is_entity_or_dimension(self, table_name: str) -> bool:
        """Check if table is already classified as entity or dimension."""
        entity_sources = [e["source"] for e in self.entities]
        dim_sources = [d["source"] for d in self.dimensions]
        return table_name in entity_sources or table_name in dim_sources
    
    def _has_numeric_columns(self, table: dict) -> bool:
        """Check if table has numeric columns."""
        for col in table.get("columns", []):
            if self._is_numeric_type(col["type"]):
                return True
        return False
    
    def _is_numeric_type(self, col_type: str) -> bool:
        """Check if column type is numeric."""
        col_type_lower = col_type.lower()
        numeric_keywords = [
            "int", "integer", "bigint", "smallint", "tinyint",
            "decimal", "numeric", "float", "real", "double",
            "money", "smallmoney"
        ]
        return any(keyword in col_type_lower for keyword in numeric_keywords)
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        return any(
            t["name"].lower() == table_name.lower()
            for t in self.all_tables
        )