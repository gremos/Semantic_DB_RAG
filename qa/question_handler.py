"""
Question Handler - Phase 3: Natural Language to SQL with Smart Disambiguation
Handles ambiguity resolution using LLM + evidence-based ranking.
"""

import json
import logging
from typing import Dict, List, Optional
from qa.table_disambiguator import TableDisambiguator

logger = logging.getLogger(__name__)


class QuestionHandler:
    """
    Converts natural language questions to SQL using semantic model.
    Automatically resolves ambiguities instead of refusing.
    """
    
    def __init__(self, llm, semantic_model: dict, discovery_json: dict):
        self.llm = llm
        self.semantic_model = semantic_model
        self.discovery_json = discovery_json
        
        # NEW: Disambiguator for handling multiple candidates
        self.disambiguator = TableDisambiguator(llm)
        
        # Extract view usage stats for ranking
        self.view_usage = self._build_view_usage_map()
        
        # Build entity/dimension/fact lookups for fast access
        self.entity_map = {e["name"]: e for e in semantic_model.get("entities", [])}
        self.dimension_map = {d["name"]: d for d in semantic_model.get("dimensions", [])}
        self.fact_map = {f["name"]: f for f in semantic_model.get("facts", [])}
    
    def answer_question(self, question: str) -> dict:
        """
        Main entry point: convert question to SQL with automatic disambiguation.
        
        Returns Answer JSON:
        {
          "status": "ok|refuse",
          "sql": [...],
          "refusal": {...}  # only if status=refuse
        }
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Step 1: Extract required concepts from question
            concepts = self._extract_concepts(question)
            logger.info(f"Extracted concepts: {concepts}")
            
            # Step 2: Find all candidate tables/measures for each concept
            candidates = self._find_candidates(concepts)
            logger.info(f"Found candidates: {candidates}")
            
            # Step 3: Disambiguate using LLM + evidence (if needed)
            resolved = self._resolve_ambiguity(question, candidates)
            logger.info(f"Resolved entities: {resolved}")
            
            # Step 4: Check if we have enough information
            if resolved.get("needs_clarification"):
                return self._build_refusal_response(resolved)
            
            # Step 5: Generate SQL with resolved entities
            sql_result = self._generate_sql(question, resolved)
            
            return sql_result
        
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            return {
                "status": "refuse",
                "refusal": {
                    "reason": f"Internal error: {str(e)}",
                    "clarifying_questions": ["Please rephrase your question or contact support."]
                }
            }
    
    def _extract_concepts(self, question: str) -> dict:
        """
        Use LLM to identify what concepts the user is asking about.
        
        Returns:
        {
          "needs": ["customer_name", "sales_measure", "time_period"],
          "filters": ["active_only"],
          "aggregations": ["sum", "group_by_customer"]
        }
        """
        prompt = f"""
You are a semantic query analyzer. Extract the key concepts from this question:

Question: "{question}"

Identify:
1. What entities are needed (customer, product, employee, etc.)
2. What measures are needed (sales, revenue, quantity, etc.)
3. What dimensions are needed (date, geography, category, etc.)
4. What filters are implied (active, recent, specific date range, etc.)
5. What aggregations are needed (sum, count, average, group by, etc.)

Output JSON Schema:
{{
  "needs": [
    {{"type": "entity", "concept": "customer", "keywords": ["customer", "client", "buyer"]}},
    {{"type": "measure", "concept": "sales", "keywords": ["sales", "revenue", "total"]}},
    {{"type": "dimension", "concept": "date", "keywords": ["when", "date", "period"]}}
  ],
  "filters": [
    {{"type": "status", "value": "active"}},
    {{"type": "date_range", "value": "last_90_days"}}
  ],
  "aggregations": [
    {{"function": "sum", "field": "sales"}},
    {{"function": "group_by", "field": "customer"}}
  ]
}}

Output valid JSON only.
"""
        
        response = self.llm.invoke(prompt)
        
        try:
            concepts = json.loads(response.content)
            return concepts
        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON for concept extraction")
            return {
                "needs": [],
                "filters": [],
                "aggregations": []
            }
    
    def _find_candidates(self, concepts: dict) -> dict:
        """
        Find all possible tables/columns that could satisfy each concept.
        
        Returns:
        {
          "customer_sources": ["dbo.BusinessPoint.BrandName", "dbo.Contact.Firstname", ...],
          "sales_measures": ["dbo.ContractProduct.TotalPrice", ...],
          "date_dimensions": ["dbo.DimDate", ...]
        }
        """
        candidates = {
            "customer_sources": [],
            "sales_measures": [],
            "date_dimensions": [],
            "other_entities": []
        }
        
        for need in concepts.get("needs", []):
            concept_type = need.get("type")
            concept = need.get("concept", "").lower()
            keywords = [k.lower() for k in need.get("keywords", [])]
            
            if concept_type == "entity":
                if any(k in ["customer", "client", "buyer"] for k in keywords):
                    candidates["customer_sources"] = self._find_customer_candidates()
                else:
                    # Generic entity search
                    candidates["other_entities"].extend(
                        self._find_entity_by_keywords(keywords)
                    )
            
            elif concept_type == "measure":
                if any(k in ["sales", "revenue", "total", "amount", "price"] for k in keywords):
                    candidates["sales_measures"] = self._find_sales_measure_candidates()
            
            elif concept_type == "dimension":
                if any(k in ["date", "time", "when", "period"] for k in keywords):
                    candidates["date_dimensions"] = self._find_date_dimension_candidates()
        
        return candidates
    
    def _find_customer_candidates(self) -> List[str]:
        """Find all possible customer name sources."""
        candidates = []
        
        # Search entities
        for entity_name, entity in self.entity_map.items():
            source = entity.get("source", "")
            
            # Check if entity name suggests customer
            if any(keyword in entity_name.lower() for keyword in ["customer", "client", "contact", "business", "account"]):
                # Find name-like columns
                table = self._get_table_from_discovery(source)
                if table:
                    for col in table.get("columns", []):
                        col_name = col["name"].lower()
                        if any(keyword in col_name for keyword in ["name", "title", "brand", "firstname", "lastname"]):
                            candidates.append(f"{source}.{col['name']}")
        
        # Also search facts that might have denormalized customer names
        for fact_name, fact in self.fact_map.items():
            source = fact.get("source", "")
            table = self._get_table_from_discovery(source)
            if table:
                for col in table.get("columns", []):
                    col_name = col["name"].lower()
                    if "customer" in col_name and "name" in col_name:
                        candidates.append(f"{source}.{col['name']}")
        
        return candidates
    
    def _find_sales_measure_candidates(self) -> List[str]:
        """Find all possible sales/revenue measures."""
        candidates = []
        
        # Search facts for monetary measures
        for fact_name, fact in self.fact_map.items():
            # Check measures defined in semantic model
            for measure in fact.get("measures", []):
                measure_name = measure.get("name", "").lower()
                if any(keyword in measure_name for keyword in ["revenue", "sales", "total", "amount", "price"]):
                    candidates.append(f"{fact['source']}.{measure['name']}")
            
            # Also check raw columns
            source = fact.get("source", "")
            table = self._get_table_from_discovery(source)
            if table:
                for col in table.get("columns", []):
                    col_name = col["name"].lower()
                    col_type = col["type"].lower()
                    
                    # Numeric columns with revenue-like names
                    if any(keyword in col_name for keyword in ["price", "amount", "revenue", "total", "sales"]):
                        if any(t in col_type for t in ["decimal", "money", "numeric", "float", "int"]):
                            candidates.append(f"{source}.{col['name']}")
        
        return candidates
    
    def _find_date_dimension_candidates(self) -> List[str]:
        """Find all date dimension tables."""
        candidates = []
        
        for dim_name, dim in self.dimension_map.items():
            if any(keyword in dim_name.lower() for keyword in ["date", "time", "calendar"]):
                candidates.append(dim.get("source", ""))
        
        return candidates
    
    def _find_entity_by_keywords(self, keywords: List[str]) -> List[str]:
        """Generic entity search by keywords."""
        candidates = []
        
        for entity_name, entity in self.entity_map.items():
            if any(kw in entity_name.lower() for kw in keywords):
                candidates.append(entity.get("source", ""))
        
        return candidates
    
    def _resolve_ambiguity(self, question: str, candidates: dict) -> dict:
        """
        Use disambiguation logic to pick best candidates.
        
        Returns:
        {
          "customer": {"selected": "dbo.BusinessPoint.BrandName", "confidence": 0.92, ...},
          "sales": {"selected": "dbo.ContractProduct.TotalPrice", "aggregation": "SUM", ...},
          "needs_clarification": false
        }
        """
        resolved = {}
        
        # Disambiguate customer sources
        customer_sources = candidates.get("customer_sources", [])
        if len(customer_sources) == 0:
            resolved["needs_clarification"] = True
            resolved["missing"] = ["customer_source"]
        elif len(customer_sources) == 1:
            resolved["customer"] = {
                "selected": customer_sources[0],
                "confidence": 1.0,
                "reasoning": "Only one customer source available"
            }
        else:
            # Multiple candidates - use disambiguator
            resolved["customer"] = self.disambiguator.disambiguate_customer_source(
                candidates=customer_sources,
                question=question,
                semantic_model=self.semantic_model,
                view_usage=self.view_usage
            )
        
        # Disambiguate sales measures
        sales_measures = candidates.get("sales_measures", [])
        if len(sales_measures) == 0:
            resolved["needs_clarification"] = True
            resolved["missing"] = resolved.get("missing", []) + ["sales_measure"]
        elif len(sales_measures) == 1:
            resolved["sales"] = {
                "selected": sales_measures[0],
                "aggregation": "SUM",
                "confidence": 1.0,
                "reasoning": "Only one sales measure available"
            }
        else:
            # Multiple candidates - use disambiguator
            resolved["sales"] = self.disambiguator.disambiguate_measure(
                candidates=sales_measures,
                question=question,
                semantic_model=self.semantic_model,
                view_usage=self.view_usage
            )
        
        # Check if disambiguation confidence is too low
        if resolved.get("customer", {}).get("confidence", 1.0) < 0.5:
            resolved["needs_clarification"] = True
        if resolved.get("sales", {}).get("confidence", 1.0) < 0.5:
            resolved["needs_clarification"] = True
        
        return resolved
    
    def _generate_sql(self, question: str, resolved: dict) -> dict:
        """
        Generate final SQL using resolved entities.
        """
        
        # Build context for LLM
        context = {
            "semantic_model": self.semantic_model,
            "resolved_entities": resolved,
            "relationships": self.discovery_json.get("semantic_relationships", []),
            "dialect": self.discovery_json.get("dialect", "mssql")
        }
        
        prompt = f"""
You are an expert SQL generator. Convert this question to SQL using the resolved entities.

Question: "{question}"

Context:
{json.dumps(context, indent=2)}

Rules:
1. Use ONLY the resolved entities (don't search for alternatives)
2. Use semantic_relationships to determine JOINs
3. Apply appropriate aggregations (SUM for sales/revenue)
4. Include GROUP BY for dimension columns
5. Output dialect-specific SQL ({context['dialect']})

Output JSON Schema:
{{
  "status": "ok",
  "sql": [
    {{
      "dialect": "mssql",
      "statement": "SELECT ... FROM ... JOIN ... WHERE ... GROUP BY ...",
      "explanation": "Using BusinessPoint.BrandName as customer (confidence 0.92) and ContractProduct.TotalPrice as sales measure",
      "evidence": {{
        "customer_source": {{
          "selected": "dbo.BusinessPoint.BrandName",
          "confidence": 0.92,
          "reasoning": "Most authoritative source"
        }},
        "sales_measure": {{
          "selected": "dbo.ContractProduct.TotalPrice",
          "aggregation": "SUM",
          "confidence": 0.88,
          "reasoning": "Primary sales measure"
        }}
      }},
      "limits": {{
        "row_limit": 1000,
        "timeout_sec": 60
      }}
    }}
  ]
}}

Output valid JSON only.
"""
        
        response = self.llm.invoke(prompt)
        
        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON for SQL generation")
            return {
                "status": "refuse",
                "refusal": {
                    "reason": "Failed to generate valid SQL",
                    "clarifying_questions": ["Please rephrase your question."]
                }
            }
    
    def _build_refusal_response(self, resolved: dict) -> dict:
        """Build a refusal response when clarification is needed."""
        missing = resolved.get("missing", [])
        
        clarifying_questions = []
        if "customer_source" in missing:
            clarifying_questions.append(
                "Could not identify a customer source. Please specify which table/view contains customer names."
            )
        if "sales_measure" in missing:
            clarifying_questions.append(
                "Could not identify a sales measure. Please specify which field represents sales/revenue."
            )
        
        return {
            "status": "refuse",
            "refusal": {
                "reason": "Insufficient information to generate query",
                "missing_objects": missing,
                "clarifying_questions": clarifying_questions
            }
        }
    
    def _get_table_from_discovery(self, table_name: str) -> Optional[dict]:
        """Get table metadata from discovery JSON."""
        for schema in self.discovery_json.get("schemas", []):
            for table in schema.get("tables", []):
                full_name = f"{schema['name']}.{table['name']}"
                if full_name == table_name or table["name"] == table_name:
                    return table
        return None
    
    def _build_view_usage_map(self) -> dict:
        """Build a map of table -> usage count from discovery."""
        usage = {}
        
        for schema in self.discovery_json.get("schemas", []):
            for table in schema.get("tables", []):
                table_name = f"{schema['name']}.{table['name']}"
                source_assets = table.get("source_assets", [])
                
                view_count = len([a for a in source_assets if a.get("kind") == "view"])
                sp_count = len([a for a in source_assets if a.get("kind") == "stored_procedure"])
                rdl_count = len([a for a in source_assets if a.get("kind") == "rdl"])
                
                usage[table_name] = {
                    "view_count": view_count,
                    "sp_count": sp_count,
                    "rdl_count": rdl_count,
                    "total_references": view_count + sp_count + rdl_count
                }
        
        return usage