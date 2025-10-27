"""
File: qa/question_handler.py
Question Answering Handler - Uses SEMANTIC MODEL as single source of truth
"""

import json
import logging
from typing import Dict, List, Any, Optional
from langchain_openai import AzureChatOpenAI
import sqlglot

logger = logging.getLogger(__name__)


class QuestionHandler:
    """
    Handles natural language questions by:
    1. Using semantic model as single source of truth
    2. Validating ALL references against model
    3. Generating grounded SQL with evidence chain
    """

    def __init__(self, llm: AzureChatOpenAI, semantic_model: Dict[str, Any]):
        """
        Initialize question handler.
        
        Args:
            llm: Azure OpenAI LLM instance
            semantic_model: Validated semantic model JSON with COMPLETE column metadata
        """
        self.llm = llm
        self.semantic_model = semantic_model
        self.dialect = semantic_model.get("audit", {}).get("dialect", "")
        
        # BUILD LOOKUP INDEX for fast validation
        self._build_model_index()

    def _build_model_index(self):
        """
        Build fast lookup index from semantic model.
        This is the SINGLE SOURCE OF TRUTH for validation.
        """
        self.table_index = {}
        self.column_index = {}
        self.measure_index = {}
        
        # Index entities
        for entity in self.semantic_model.get("entities", []):
            source = entity["source"]
            self.table_index[source] = {
                "type": "entity",
                "name": entity["name"],
                "columns": entity.get("columns", []),
                "primary_key": entity.get("primary_key", [])
            }
            
            # Index columns
            for col in entity.get("columns", []):
                col_key = f"{source}.{col['name']}"
                self.column_index[col_key] = {
                    "table": source,
                    "column": col["name"],
                    "type": col["type"],
                    "sample_values": col.get("sample_values", []),
                    "value_distribution": col.get("value_distribution", {}),
                    "classification": col.get("classification", {})
                }
        
        # Index facts
        for fact in self.semantic_model.get("facts", []):
            source = fact["source"]
            self.table_index[source] = {
                "type": "fact",
                "name": fact["name"],
                "columns": fact.get("columns", []),
                "measures": fact.get("measures", []),
                "grain": fact.get("grain", [])
            }
            
            # Index columns
            for col in fact.get("columns", []):
                col_key = f"{source}.{col['name']}"
                self.column_index[col_key] = {
                    "table": source,
                    "column": col["name"],
                    "type": col["type"],
                    "sample_values": col.get("sample_values", []),
                    "value_distribution": col.get("value_distribution", {}),
                    "classification": col.get("classification", {})
                }
            
            # Index measures
            for measure in fact.get("measures", []):
                measure_key = f"{fact['name']}.{measure['name']}"
                self.measure_index[measure_key] = {
                    "measure": measure["name"],
                    "expression": measure["expression"],
                    "base_column": measure.get("column"),
                    "from_fact": fact["name"],
                    "source_table": source
                }
        
        logger.info(f"Built model index: {len(self.table_index)} tables, {len(self.column_index)} columns, {len(self.measure_index)} measures")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Main entry point for question answering.
        ALWAYS validates against semantic model index.
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Parse question and extract explicit references
            parsed = self._parse_question_references(question)
            logger.info(f"Parsed references: {parsed}")
            
            # Step 2: VALIDATE all references against model index
            validation = self._validate_references(parsed)
            
            if not validation["valid"]:
                # References don't exist in model → immediate refusal
                return self._generate_refusal_with_alternatives(
                    question, 
                    validation["errors"],
                    validation["suggestions"]
                )
            
            # Step 3: Extract semantic intent using LLM
            concepts = self._extract_concepts(question)
            logger.info(f"Extracted concepts: {concepts}")
            
            # Step 4: Find matching candidates using MODEL INDEX
            candidates = self._find_candidates_from_index(question, concepts, parsed)
            
            # Step 5: If multiple candidates, choose best OR offer alternatives
            if len(candidates.get("alternatives", [])) > 1:
                return self._offer_alternatives(question, candidates)
            
            # Step 6: Assess confidence
            confidence = self._assess_confidence(candidates)
            
            if confidence < 0.70:
                return self._generate_refusal(question, candidates, confidence)
            
            # Step 7: Generate SQL using ONLY validated objects
            sql_result = self._generate_sql_via_llm(question, candidates)
            
            # Step 8: Final SQL validation
            sql_validation = self._validate_sql(sql_result["sql"])
            if not sql_validation["valid"]:
                return self._generate_refusal(
                    question, 
                    candidates, 
                    confidence,
                    validation_error=sql_validation["error"]
                )
            
            # Step 9: Build evidence chain
            evidence = self._build_evidence_chain(candidates, sql_result)
            
            # Step 10: Generate explanation
            explanation = self._generate_explanation_via_llm(
                question, sql_result["sql"], candidates
            )
            
            return {
                "status": "ok",
                "sql": [{
                    "dialect": self.dialect,
                    "statement": sql_result["sql"],
                    "explanation": explanation,
                    "evidence": evidence,
                    "confidence": confidence,
                    "limits": {
                        "row_limit": 1000,
                        "timeout_sec": 60
                    }
                }],
                "alternatives": candidates.get("alternatives", []),
                "next_steps": sql_result.get("next_steps", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            return {
                "status": "refuse",
                "refusal": {
                    "reason": f"Internal error: {str(e)}",
                    "suggestions": ["Please rephrase your question", "Provide more specific details"]
                }
            }

    def _parse_question_references(self, question: str) -> Dict[str, Any]:
        """
        Parse explicit table/column references from question.
        Example: "SUM(Orders.TotalAmount)" → {table: "Orders", column: "TotalAmount"}
        """
        import re
        
        # Pattern: Table.Column
        table_col_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b'
        
        # Pattern: status = 'Value'
        status_pattern = r"(?:status|state)\s*=\s*['\"]([^'\"]+)['\"]"
        
        matches = re.findall(table_col_pattern, question, re.IGNORECASE)
        status_matches = re.findall(status_pattern, question, re.IGNORECASE)
        
        return {
            "explicit_refs": [
                {"table": m[0], "column": m[1]} 
                for m in matches
            ],
            "status_filters": status_matches
        }

    def _validate_references(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ALL explicit references against model index.
        This is the CRITICAL validation step.
        """
        errors = []
        suggestions = []
        
        for ref in parsed["explicit_refs"]:
            table = ref["table"]
            column = ref["column"]
            
            # Find matching table (case-insensitive, partial match)
            matched_table = None
            for table_key in self.table_index.keys():
                if table.lower() in table_key.lower():
                    matched_table = table_key
                    break
            
            if not matched_table:
                errors.append(f"Table '{table}' not found in semantic model")
                # Suggest similar tables
                similar = [
                    t for t in self.table_index.keys() 
                    if self._is_similar(table, t)
                ]
                if similar:
                    suggestions.append(f"Did you mean: {', '.join(similar)}?")
                continue
            
            # Check column exists
            col_key = f"{matched_table}.{column}"
            if col_key not in self.column_index:
                errors.append(f"Column '{column}' not found in table '{matched_table}'")
                # Suggest similar columns
                table_cols = [
                    c for c in self.column_index.keys() 
                    if c.startswith(matched_table + ".")
                ]
                similar_cols = [
                    c.split(".")[-1] for c in table_cols 
                    if self._is_similar(column, c.split(".")[-1])
                ]
                if similar_cols:
                    suggestions.append(f"Did you mean: {', '.join(similar_cols)}?")
        
        # Validate status filters
        for status_val in parsed["status_filters"]:
            # Find Status columns
            status_cols = [
                col_key for col_key, col_info in self.column_index.items()
                if col_info.get("classification", {}).get("semantic_role") == "status_indicator"
            ]
            
            if not status_cols:
                errors.append(f"No status column found in semantic model")
                continue
            
            # Check if value exists in any status column
            found = False
            for col_key in status_cols:
                col_info = self.column_index[col_key]
                sample_values = [str(v).lower() for v in col_info.get("sample_values", [])]
                if status_val.lower() in sample_values:
                    found = True
                    break
            
            if not found:
                errors.append(f"Status value '{status_val}' not found in sampled values")
                # Show actual values
                if status_cols:
                    actual_vals = self.column_index[status_cols[0]].get("sample_values", [])
                    suggestions.append(f"Valid status values: {', '.join(map(str, actual_vals))}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "suggestions": suggestions
        }

    def _is_similar(self, str1: str, str2: str, threshold: float = 0.6) -> bool:
        """Check if two strings are similar using simple heuristic."""
        s1 = str1.lower()
        s2 = str2.lower()
        
        # Exact match
        if s1 == s2:
            return True
        
        # Substring match
        if s1 in s2 or s2 in s1:
            return True
        
        # Levenshtein-like check
        shorter = min(len(s1), len(s2))
        longer = max(len(s1), len(s2))
        
        if shorter / longer < threshold:
            return False
        
        # Count matching characters
        matches = sum(1 for a, b in zip(s1, s2) if a == b)
        return matches / longer >= threshold

    def _find_candidates_from_index(
        self, 
        question: str, 
        concepts: Dict[str, Any],
        parsed_refs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find candidates using MODEL INDEX (not LLM guessing).
        Returns definitive matches with alternatives if multiple options exist.
        """
        candidates = {
            "matched_entities": [],
            "matched_facts": [],
            "matched_measures": [],
            "alternatives": [],
            "filters": []
        }
        
        # Use explicit references first
        for ref in parsed_refs["explicit_refs"]:
            table = ref["table"]
            column = ref["column"]
            
            # Find in index
            for table_key, table_info in self.table_index.items():
                if table.lower() in table_key.lower():
                    col_key = f"{table_key}.{column}"
                    
                    if col_key in self.column_index:
                        col_info = self.column_index[col_key]
                        
                        # Is this a measure column?
                        if col_info.get("classification", {}).get("semantic_role") == "measure_component":
                            # Find measure definition
                            for meas_key, meas_info in self.measure_index.items():
                                if meas_info["base_column"] == column and meas_info["source_table"] == table_key:
                                    candidates["matched_measures"].append({
                                        "name": meas_info["measure"],
                                        "expression": meas_info["expression"],
                                        "column": column,
                                        "table": table_key,
                                        "relevance": "high",
                                        "reason": "Explicitly referenced in question"
                                    })
                        
                        # Is this a fact table?
                        if table_info["type"] == "fact":
                            candidates["matched_facts"].append({
                                "name": table_info["name"],
                                "source": table_key,
                                "columns": table_info["columns"],
                                "relevance": "high",
                                "reason": "Explicitly referenced in question"
                            })
                        
                        # Is this an entity?
                        if table_info["type"] == "entity":
                            candidates["matched_entities"].append({
                                "name": table_info["name"],
                                "source": table_key,
                                "columns": table_info["columns"],
                                "relevance": "high",
                                "reason": "Explicitly referenced in question"
                            })
        
        # Add status filters with validated values
        for status_val in parsed_refs["status_filters"]:
            status_cols = [
                col_key for col_key, col_info in self.column_index.items()
                if col_info.get("classification", {}).get("semantic_role") == "status_indicator"
            ]
            
            for col_key in status_cols:
                col_info = self.column_index[col_key]
                sample_values = [str(v).lower() for v in col_info.get("sample_values", [])]
                
                if status_val.lower() in sample_values:
                    candidates["filters"].append({
                        "column": col_key,
                        "operator": "=",
                        "value": status_val,
                        "validated": True,
                        "reason": "Status value exists in sampled data"
                    })
        
        # If no explicit refs, look for sales/revenue measures (semantic matching)
        if not candidates["matched_measures"]:
            needs = concepts.get("needs", [])
            for need in needs:
                if need.get("type") == "measure":
                    # Find revenue-related measures
                    for meas_key, meas_info in self.measure_index.items():
                        if any(kw in meas_info["measure"].lower() for kw in ["revenue", "sales", "total", "amount"]):
                            candidates["matched_measures"].append({
                                "name": meas_info["measure"],
                                "expression": meas_info["expression"],
                                "column": meas_info["base_column"],
                                "table": meas_info["source_table"],
                                "relevance": "medium",
                                "reason": f"Semantic match for '{need.get('concept')}'"
                            })
                            
                            # Also add the fact table
                            fact_table = meas_info["source_table"]
                            if fact_table in self.table_index:
                                candidates["matched_facts"].append({
                                    "name": self.table_index[fact_table]["name"],
                                    "source": fact_table,
                                    "columns": self.table_index[fact_table]["columns"],
                                    "relevance": "high",
                                    "reason": "Source table for matched measure"
                                })
        
        # Detect if multiple measure options exist (alternatives)
        if len(candidates["matched_measures"]) > 1:
            candidates["alternatives"] = [
                {
                    "measure": m["name"],
                    "expression": m["expression"],
                    "description": f"Use {m['column']} from {m['table']}"
                }
                for m in candidates["matched_measures"]
            ]
        
        return candidates

    def _offer_alternatives(
        self, 
        question: str, 
        candidates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        When multiple valid options exist, present them to user.
        This is EXPLORATION mode.
        """
        alternatives = candidates.get("alternatives", [])
        
        return {
            "status": "clarify",
            "message": f"I found {len(alternatives)} ways to calculate this. Which should I use?",
            "alternatives": alternatives,
            "suggestion": f"Please specify which measure: {' or '.join([a['measure'] for a in alternatives])}"
        }

    def _generate_refusal_with_alternatives(
        self,
        question: str,
        errors: List[str],
        suggestions: List[str]
    ) -> Dict[str, Any]:
        """
        Generate refusal when explicit references don't validate.
        Provide concrete alternatives from semantic model.
        """
        return {
            "status": "refuse",
            "refusal": {
                "reason": f"Cannot find the requested objects in semantic model: {'; '.join(errors)}",
                "errors": errors,
                "suggestions": suggestions,
                "available_tables": list(self.table_index.keys()),
                "available_measures": [
                    f"{m['measure']} ({m['expression']})"
                    for m in self.measure_index.values()
                ]
            }
        }

    # ... rest of methods (_extract_concepts, _assess_confidence, _generate_sql_via_llm, etc.) remain similar
    # but now they operate on VALIDATED candidates from the index

    def _extract_concepts(self, question: str) -> Dict[str, Any]:
        """Extract semantic concepts using LLM (no hardcoded keywords)."""
        prompt = f"""Analyze this business question and extract semantic concepts.

Question: "{question}"

Extract the following as JSON:
{{
  "needs": [
    {{"type": "entity|measure|dimension", "concept": "name", "purpose": "why needed"}}
  ],
  "filters": [
    {{"type": "filter_type", "value": "constraint", "implicit": true/false}}
  ],
  "aggregations": [
    {{"function": "sum|count|avg|group_by", "field": "concept", "note": "optional"}}
  ],
  "timeframe": {{"type": "all_time|recent|specific", "value": "description"}},
  "ambiguities": ["any unclear aspects"]
}}

Return ONLY valid JSON, no markdown."""

        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse concept extraction: {e}")
            retry_prompt = f"{prompt}\n\nPrevious response was invalid JSON. Please return ONLY valid JSON."
            retry_response = self.llm.invoke(retry_prompt)
            return json.loads(retry_response.content)

    def _assess_confidence(self, candidates: Dict[str, Any]) -> float:
        """Assess confidence - now based on validated matches."""
        matched_measures = len(candidates.get("matched_measures", []))
        matched_facts = len(candidates.get("matched_facts", []))
        matched_entities = len(candidates.get("matched_entities", []))
        validated_filters = sum(1 for f in candidates.get("filters", []) if f.get("validated"))
        
        # If we have explicit validated matches, confidence is high
        if matched_measures > 0 and matched_facts > 0:
            return 0.9
        elif matched_facts > 0:
            return 0.75
        else:
            return 0.5

    def _generate_sql_via_llm(
        self, question: str, candidates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SQL using LLM with strict grounding to matched candidates."""
        prompt = f"""Generate SQL to answer this business question.

Question: "{question}"

STRICT CONSTRAINTS:
1. ONLY use these matched objects (DO NOT invent tables/columns):
{json.dumps(candidates, indent=2)}

2. Use dialect: {self.dialect}
3. Include proper JOINs based on required_relationships
4. Add appropriate WHERE clauses for filters
5. Use matched measures with their exact expressions
6. Add LIMIT 1000
7. Add appropriate GROUP BY if aggregating

Return JSON:
{{
  "sql": "SELECT ... (complete SQL statement)",
  "tables_used": ["list of schema.table"],
  "columns_used": ["list of column references"],
  "joins_used": ["list of join conditions"],
  "reasoning": "step-by-step explanation of SQL construction",
  "next_steps": ["optional suggestions for user"]
}}

Return ONLY valid JSON. SQL must be executable."""

        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SQL generation: {e}")
            retry_prompt = f"{prompt}\n\nPrevious response was invalid JSON. Return ONLY valid JSON."
            retry_response = self.llm.invoke(retry_prompt)
            return json.loads(retry_response.content)

    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate SQL syntax using sqlglot parser."""
        try:
            # Parse SQL
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            
            # Extract all table references
            tables_in_sql = set()
            for table in parsed.find_all(sqlglot.exp.Table):
                tables_in_sql.add(f"{table.db}.{table.name}" if table.db else str(table.name))
            
            # Verify all tables exist in semantic model
            valid_tables = set(self.table_index.keys())
            
            invalid_tables = tables_in_sql - valid_tables
            if invalid_tables:
                return {
                    "valid": False,
                    "error": f"SQL references tables not in semantic model: {invalid_tables}"
                }
            
            # Check for dangerous operations
            if any(parsed.find_all(exp_type) for exp_type in [
                sqlglot.exp.Insert,
                sqlglot.exp.Update,
                sqlglot.exp.Delete,
                sqlglot.exp.Drop,
                sqlglot.exp.Create,
                sqlglot.exp.Alter
            ]):
                return {
                    "valid": False,
                    "error": "SQL contains forbidden write operations"
                }
            
            return {
                "valid": True,
                "parsed": parsed,
                "tables_used": list(tables_in_sql)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"SQL parsing failed: {str(e)}"
            }

    def _build_evidence_chain(
        self, candidates: Dict[str, Any], sql_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build evidence chain showing how SQL maps to semantic model."""
        evidence = {
            "entities": [e["name"] for e in candidates.get("matched_entities", [])],
            "facts": [f["name"] for f in candidates.get("matched_facts", [])],
            "measures": [m["name"] for m in candidates.get("matched_measures", [])],
            "filters": candidates.get("filters", []),
            "sources": {
                "tables": sql_result.get("tables_used", []),
                "columns": sql_result.get("columns_used", []),
                "joins": sql_result.get("joins_used", [])
            }
        }
        return evidence

    def _generate_explanation_via_llm(
        self, question: str, sql: str, candidates: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation of the SQL using LLM."""
        prompt = f"""Generate a clear, business-friendly explanation of this SQL query.

Question: "{question}"

SQL:
{sql}

Matched Semantic Objects:
{json.dumps(candidates, indent=2)}

Write a 2-3 sentence explanation that:
1. States what data we're retrieving
2. Mentions key business entities/measures used
3. Notes any important filters or aggregations
4. Avoids technical jargon

Return ONLY the explanation text, no JSON, no markdown."""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _generate_refusal(
        self,
        question: str,
        candidates: Dict[str, Any],
        confidence: float,
        validation_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate refusal response with clarifying questions using LLM."""
        prompt = f"""The system cannot confidently answer this question. Generate helpful clarifying questions.

Question: "{question}"

Confidence: {confidence:.2f}

Matched Candidates:
{json.dumps(candidates, indent=2)}

{"Validation Error: " + validation_error if validation_error else ""}

Generate 2-3 specific clarifying questions that would help answer the original question.
Focus on:
1. Missing entities or measures
2. Ambiguous time periods or filters
3. Unclear aggregation levels

Return JSON:
{{
  "reason": "brief explanation of why we can't answer",
  "missing": ["list specific missing elements"],
  "ambiguities": ["list unclear aspects"],
  "clarifying_questions": [
    "Specific question 1?",
    "Specific question 2?",
    "Specific question 3?"
  ],
  "suggestions": ["alternative approaches user could try"]
}}

Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        try:
            refusal_data = json.loads(response.content)
            return {
                "status": "refuse",
                "refusal": refusal_data,
                "confidence": confidence
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse refusal generation: {e}")
            # Fallback refusal
            return {
                "status": "refuse",
                "refusal": {
                    "reason": f"Cannot answer with confidence {confidence:.2f}. " + 
                             (validation_error or "Insufficient information in semantic model."),
                    "clarifying_questions": [
                        "Could you provide more specific details about what you're looking for?",
                        "Which specific metrics or entities are you interested in?",
                        "What time period should be considered?"
                    ]
                },
                "confidence": confidence
            }