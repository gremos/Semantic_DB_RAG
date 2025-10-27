"""
Semantic Model Builder - Simple adapter for pipeline integration
Builds semantic model from discovery data using LLM
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class SemanticModelBuilder:
    """
    Builds semantic model from discovery data.
    Uses LLM to identify entities, facts, dimensions, measures, and relationships.
    """
    
    def __init__(self, discovery_json: Dict[str, Any], llm_client: Any):
        """
        Initialize builder.
        
        Args:
            discovery_json: Complete discovery data
            llm_client: LangChain AzureChatOpenAI instance
        """
        self.discovery = discovery_json
        self.llm = llm_client
        self.dialect = discovery_json.get("dialect", "mssql")
        
        # Build table index for quick lookup
        self.tables = {}
        for schema in discovery_json.get("schemas", []):
            for table in schema.get("tables", []):
                full_name = f"{schema['name']}.{table['name']}"
                self.tables[full_name] = table
    
    def build(self) -> Dict[str, Any]:
        """
        Build complete semantic model.
        
        Returns:
            Semantic model JSON following the spec
        """
        logger.info("Building semantic model from discovery data...")
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Build user prompt with discovery context
        user_prompt = self._build_user_prompt()
        
        # Call LLM
        logger.info("Calling LLM to generate semantic model...")
        response = self._call_llm(system_prompt, user_prompt)
        
        # Parse response
        model = self._parse_response(response)
        
        # Enrich with column metadata
        model = self._enrich_with_columns(model)
        
        # Add audit metadata
        model["audit"] = {
            "dialect": self.dialect,
            "created_at": datetime.utcnow().isoformat(),
            "tables_analyzed": len(self.tables),
            "generator": "SemanticModelBuilder"
        }
        
        logger.info(f"Semantic model built: {len(model.get('entities', []))} entities, "
                   f"{len(model.get('facts', []))} facts, "
                   f"{len(model.get('relationships', []))} relationships")
        
        return model
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for semantic modeling."""
        return f"""You are a semantic data modeling expert. Your job is to analyze database schemas and create a business-friendly semantic model.

CRITICAL RULES:
1. Only use tables and columns that exist in the provided discovery data
2. Classify tables as either ENTITIES (master data) or FACTS (transactional data)
3. Identify meaningful MEASURES for facts (SUM, AVG, COUNT, etc.)
4. Identify RELATIONSHIPS between tables (foreign keys)
5. Generate business-friendly METRICS that users might want to query
6. Return valid JSON ONLY - no markdown, no explanation

OUTPUT SCHEMA:
{{
  "entities": [
    {{
      "name": "Customer",
      "source": "dbo.Customer",
      "primary_key": ["CustomerID"],
      "business_name": "Customer",
      "description": "Customer master data",
      "confidence": "high"
    }}
  ],
  "dimensions": [
    {{
      "name": "Date",
      "source": "dbo.DimDate",
      "keys": ["DateKey"],
      "attributes": ["Year", "Month", "Quarter"],
      "business_name": "Date Dimension",
      "confidence": "high"
    }}
  ],
  "facts": [
    {{
      "name": "Sales",
      "source": "dbo.Orders",
      "grain": ["OrderID"],
      "measures": [
        {{
          "name": "TotalRevenue",
          "expression": "SUM(TotalAmount)",
          "aggregation": "sum",
          "column": "TotalAmount",
          "business_name": "Total Revenue",
          "confidence": "high"
        }}
      ],
      "foreign_keys": [
        {{
          "column": "CustomerID",
          "references": "Customer.CustomerID"
        }}
      ],
      "business_name": "Sales Transactions",
      "description": "Order-level sales data",
      "confidence": "high"
    }}
  ],
  "relationships": [
    {{
      "from": "Sales.CustomerID",
      "to": "Customer.CustomerID",
      "cardinality": "many_to_one",
      "confidence": "high"
    }}
  ],
  "metrics": [
    {{
      "name": "Total Revenue",
      "logic": "SUM(TotalAmount) from Orders",
      "required_objects": ["Sales"],
      "confidence": "high"
    }}
  ]
}}

CLASSIFICATION RULES:
- ENTITIES: Master data tables (Customer, Product, Employee) - low FK count, descriptive columns
- DIMENSIONS: Date/time tables or lookup tables with hierarchies
- FACTS: Transactional tables (Orders, Sales, Transactions) - many FKs, numeric measures
- MEASURES: Numeric columns that can be aggregated (SUM, AVG, COUNT, MIN, MAX)
- RELATIONSHIPS: Foreign key references between tables

DATABASE DIALECT: {self.dialect}
"""
    
    def _build_user_prompt(self) -> str:
        """Build user prompt with discovery context."""
        
        # Summarize tables
        table_summary = []
        for table_name, table_data in self.tables.items():
            columns = table_data.get("columns", [])
            fks = table_data.get("foreign_keys", [])
            
            # Sample a few columns
            col_sample = columns[:10]  # First 10 columns
            
            summary = {
                "table": table_name,
                "column_count": len(columns),
                "sample_columns": [
                    {
                        "name": c.get("name"),
                        "type": c.get("type"),
                        "nullable": c.get("nullable", True)
                    }
                    for c in col_sample
                ],
                "primary_key": table_data.get("primary_key", []),
                "foreign_keys": fks,
                "row_count": table_data.get("rowcount_sample", 0)
            }
            table_summary.append(summary)
        
        prompt = f"""Analyze the following {len(table_summary)} tables and create a semantic model:

{json.dumps(table_summary, indent=2)}

TASK:
1. Classify each table as entity, dimension, or fact
2. For facts, identify measures (aggregatable numeric columns)
3. Identify relationships based on foreign keys and naming patterns
4. Generate business-friendly names and descriptions
5. Suggest useful metrics that combine measures

Return ONLY valid JSON following the output schema. No markdown, no explanation."""
        
        return prompt
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with prompts."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON."""
        
        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        try:
            model = json.loads(response)
            return model
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {response[:500]}")
            raise ValueError(f"LLM returned invalid JSON: {e}")
    
    def _enrich_with_columns(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich entities and facts with full column metadata from discovery.
        This ensures the semantic model has complete information.
        """
        
        # Enrich entities
        for entity in model.get("entities", []):
            source = entity.get("source")
            if source in self.tables:
                table_data = self.tables[source]
                entity["columns"] = table_data.get("columns", [])
        
        # Enrich facts
        for fact in model.get("facts", []):
            source = fact.get("source")
            if source in self.tables:
                table_data = self.tables[source]
                fact["columns"] = table_data.get("columns", [])
                
                # Add filter columns (all non-numeric columns can be used for filtering)
                filter_cols = []
                for col in table_data.get("columns", []):
                    col_type = col.get("type", "").lower()
                    # Skip numeric types (those are measures)
                    if not any(t in col_type for t in ["int", "decimal", "float", "numeric", "money"]):
                        filter_cols.append({
                            "name": col.get("name"),
                            "type": col.get("type"),
                            "role": "filter",
                            "business_name": col.get("name")
                        })
                
                fact["filter_columns"] = filter_cols
        
        # Enrich dimensions
        for dim in model.get("dimensions", []):
            source = dim.get("source")
            if source in self.tables:
                table_data = self.tables[source]
                dim["columns"] = table_data.get("columns", [])
        
        return model