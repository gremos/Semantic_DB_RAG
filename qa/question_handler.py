"""
File: qa/question_handler.py
Question Answering Handler - Uses LLM for all semantic matching (no hardcoded keywords)
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
    1. Using LLM to extract semantic intent
    2. Using LLM to match against semantic model
    3. Generating grounded SQL with evidence chain
    """

    def __init__(self, llm: AzureChatOpenAI, semantic_model: Dict[str, Any]):
        """
        Initialize question handler.
        
        Args:
            llm: Azure OpenAI LLM instance
            semantic_model: Validated semantic model JSON
        """
        self.llm = llm
        self.semantic_model = semantic_model
        self.dialect = semantic_model.get("audit", {}).get("dialect", "")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Main entry point for question answering.
        
        File: qa/question_handler.py, Line: 37
        
        Args:
            question: Natural language question
            
        Returns:
            Answer JSON with SQL, explanation, evidence, or refusal
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Extract semantic intent using LLM
            concepts = self._extract_concepts(question)
            logger.info(f"Extracted concepts: {concepts}")
            
            # Step 2: Find matching candidates in semantic model using LLM
            candidates = self._find_candidates_via_llm(question, concepts)
            
            # Step 3: Assess confidence and determine if we can answer
            confidence = self._assess_confidence(candidates)
            
            if confidence < 0.70:
                return self._generate_refusal(question, candidates, confidence)
            
            # Step 4: Generate SQL using LLM with strict grounding
            sql_result = self._generate_sql_via_llm(question, candidates)
            
            # Step 5: Validate SQL
            validation = self._validate_sql(sql_result["sql"])
            if not validation["valid"]:
                return self._generate_refusal(
                    question, 
                    candidates, 
                    confidence,
                    validation_error=validation["error"]
                )
            
            # Step 6: Build evidence chain
            evidence = self._build_evidence_chain(candidates, sql_result)
            
            # Step 7: Generate explanation using LLM
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

    def _extract_concepts(self, question: str) -> Dict[str, Any]:
        """
        Extract semantic concepts using LLM (no hardcoded keywords).
        
        File: qa/question_handler.py, Line: 110
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with needs, filters, aggregations
        """
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
            # Retry once
            retry_prompt = f"{prompt}\n\nPrevious response was invalid JSON. Please return ONLY valid JSON."
            retry_response = self.llm.invoke(retry_prompt)
            return json.loads(retry_response.content)

    def _find_candidates_via_llm(
        self, question: str, concepts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find matching candidates in semantic model using LLM for semantic matching.
        NO hardcoded keyword matching - LLM determines relevance.
        
        File: qa/question_handler.py, Line: 157
        
        Args:
            question: Original question
            concepts: Extracted concepts
            
        Returns:
            Dictionary with matched entities, measures, relationships
        """
        # Build semantic model context
        model_context = {
            "entities": self.semantic_model.get("entities", []),
            "dimensions": self.semantic_model.get("dimensions", []),
            "facts": self.semantic_model.get("facts", []),
            "relationships": self.semantic_model.get("relationships", []),
            "metrics": self.semantic_model.get("metrics", [])
        }
        
        prompt = f"""You are matching a business question to a semantic data model.

Question: "{question}"

Extracted Concepts:
{json.dumps(concepts, indent=2)}

Available Semantic Model:
{json.dumps(model_context, indent=2)}

Your task: Find the BEST matches for the question. For each concept, identify which model objects (entities, dimensions, facts, measures, metrics) are relevant.

Rules:
1. ONLY suggest objects that exist in the semantic model
2. Rate each match's relevance (high/medium/low)
3. Identify required relationships between objects
4. Flag if critical information is missing

Return JSON:
{{
  "matched_entities": [
    {{"name": "EntityName", "source": "schema.table", "relevance": "high|medium|low", "reason": "why matched"}}
  ],
  "matched_dimensions": [
    {{"name": "DimensionName", "source": "schema.table", "relevance": "high|medium|low", "reason": "why matched"}}
  ],
  "matched_facts": [
    {{"name": "FactName", "source": "schema.table", "grain": ["keys"], "relevance": "high|medium|low", "reason": "why matched"}}
  ],
  "matched_measures": [
    {{"name": "MeasureName", "expression": "SQL expression", "from_fact": "FactName", "relevance": "high|medium|low", "reason": "why matched"}}
  ],
  "matched_metrics": [
    {{"name": "MetricName", "logic": "description", "relevance": "high|medium|low", "reason": "why matched"}}
  ],
  "required_relationships": [
    {{"from": "Object.Column", "to": "Object.Column", "exists": true/false}}
  ],
  "missing_critical": ["list any critical missing pieces"],
  "ambiguities": ["any unclear mappings"]
}}

Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse candidate matching: {e}")
            retry_prompt = f"{prompt}\n\nPrevious response was invalid JSON. Return ONLY valid JSON."
            retry_response = self.llm.invoke(retry_prompt)
            return json.loads(retry_response.content)

    def _assess_confidence(self, candidates: Dict[str, Any]) -> float:
        """
        Assess confidence that we can answer the question using LLM judgment.
        
        File: qa/question_handler.py, Line: 235
        
        Args:
            candidates: Matched candidates from semantic model
            
        Returns:
            Confidence score 0.0-1.0
        """
        prompt = f"""Assess the confidence that we can answer the user's question based on these matches.

Matched Candidates:
{json.dumps(candidates, indent=2)}

Evaluate:
1. Are all required entities/measures available? (40% weight)
2. Are relationships properly defined? (30% weight)
3. Are there any critical ambiguities? (20% weight)
4. Is the data grain appropriate? (10% weight)

Return JSON:
{{
  "confidence_score": 0.0-1.0,
  "reasoning": "explanation of score",
  "strengths": ["what we have"],
  "weaknesses": ["what's missing or unclear"],
  "critical_gaps": ["blockers if any"]
}}

Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            return float(result.get("confidence_score", 0.0))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to assess confidence: {e}")
            # Conservative default
            return 0.5

    def _generate_sql_via_llm(
        self, question: str, candidates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate SQL using LLM with strict grounding to matched candidates.
        
        File: qa/question_handler.py, Line: 278
        
        Args:
            question: Original question
            candidates: Matched candidates with evidence
            
        Returns:
            Dictionary with SQL and metadata
        """
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
        """
        Validate SQL syntax using sqlglot parser.
        
        File: qa/question_handler.py, Line: 329
        
        Args:
            sql: SQL statement to validate
            
        Returns:
            Dictionary with validation status and any errors
        """
        try:
            # Parse SQL
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            
            # Extract all table references
            tables_in_sql = set()
            for table in parsed.find_all(sqlglot.exp.Table):
                tables_in_sql.add(f"{table.db}.{table.name}" if table.db else str(table.name))
            
            # Verify all tables exist in semantic model
            valid_tables = set()
            for entity in self.semantic_model.get("entities", []):
                valid_tables.add(entity["source"])
            for dim in self.semantic_model.get("dimensions", []):
                valid_tables.add(dim["source"])
            for fact in self.semantic_model.get("facts", []):
                valid_tables.add(fact["source"])
            
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
        """
        Build evidence chain showing how SQL maps to semantic model.
        
        File: qa/question_handler.py, Line: 391
        
        Args:
            candidates: Matched candidates
            sql_result: Generated SQL result
            
        Returns:
            Evidence dictionary
        """
        evidence = {
            "entities": [e["name"] for e in candidates.get("matched_entities", [])],
            "dimensions": [d["name"] for d in candidates.get("matched_dimensions", [])],
            "facts": [f["name"] for f in candidates.get("matched_facts", [])],
            "measures": [m["name"] for m in candidates.get("matched_measures", [])],
            "metrics": [m["name"] for m in candidates.get("matched_metrics", [])],
            "relationships": candidates.get("required_relationships", []),
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
        """
        Generate human-readable explanation of the SQL using LLM.
        
        File: qa/question_handler.py, Line: 421
        
        Args:
            question: Original question
            sql: Generated SQL
            candidates: Matched candidates
            
        Returns:
            Plain English explanation
        """
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
        """
        Generate refusal response with clarifying questions using LLM.
        
        File: qa/question_handler.py, Line: 457
        
        Args:
            question: Original question
            candidates: Matched candidates (may be incomplete)
            confidence: Confidence score
            validation_error: Optional validation error message
            
        Returns:
            Refusal JSON with suggestions
        """
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