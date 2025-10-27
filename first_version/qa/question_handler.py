"""
Phase 3: Question Answering with QuadRails (Anti-Hallucination Guardrails)

QuadRails:
1. Grounding: Only use objects from semantic model
2. Constraint: Strict JSON schema validation  
3. Verification: SQL linting via sqlglot
4. Escalation: Refuse with clarifying questions when uncertain
"""

import json
import logging
import sqlglot
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class QuestionHandler:
    """
    Answer questions using semantic model as SINGLE SOURCE OF TRUTH.
    Implements QuadRails to prevent hallucination.
    """
    
    def __init__(self, semantic_model: Dict[str, Any], llm_client: Any):
        self.model = semantic_model
        self.llm = llm_client
        self.dialect = semantic_model.get("audit", {}).get("dialect", "mssql")
        
        # Build whitelist of allowed objects
        self._build_whitelist()
        
        # Load Q&A prompt
        self._load_prompt()
    
    def _build_whitelist(self):
        """
        Build whitelist of all allowed entities, tables, columns, measures.
        CRITICAL: This is QuadRail #1 - Grounding
        """
        self.whitelist = {
            "entities": {},
            "dimensions": {},
            "facts": {},
            "measures": {},
            "columns": set(),
            "tables": set(),
            "relationships": []
        }
        
        # Extract entities
        for entity in self.model.get("entities", []):
            name = entity.get("name")
            self.whitelist["entities"][name] = entity
            self.whitelist["tables"].add(entity.get("source"))
            
            # Extract columns if available
            for col in entity.get("columns", []):
                self.whitelist["columns"].add(f"{name}.{col['name']}")
        
        # Extract dimensions
        for dim in self.model.get("dimensions", []):
            name = dim.get("name")
            self.whitelist["dimensions"][name] = dim
            self.whitelist["tables"].add(dim.get("source"))
            
            # Extract attributes
            for attr in dim.get("attributes", []):
                self.whitelist["columns"].add(f"{name}.{attr}")
        
        # Extract facts
        for fact in self.model.get("facts", []):
            name = fact.get("name")
            self.whitelist["facts"][name] = fact
            self.whitelist["tables"].add(fact.get("source"))
            
            # Extract measures
            for measure in fact.get("measures", []):
                measure_name = measure.get("name")
                self.whitelist["measures"][f"{name}.{measure_name}"] = measure
            
            # Extract grain columns
            for grain_col in fact.get("grain", []):
                self.whitelist["columns"].add(f"{name}.{grain_col}")
            
            # Extract foreign keys
            for fk in fact.get("foreign_keys", []):
                col = fk.get("column")
                self.whitelist["columns"].add(f"{name}.{col}")
        
        # Extract relationships
        self.whitelist["relationships"] = self.model.get("relationships", [])
        
        logger.info(f"Whitelist built: {len(self.whitelist['entities'])} entities, "
                   f"{len(self.whitelist['facts'])} facts, "
                   f"{len(self.whitelist['measures'])} measures, "
                   f"{len(self.whitelist['columns'])} columns")
    
    def _load_prompt(self):
        """Load Q&A system prompt."""
        self.system_prompt = f"""You are a SQL query generator. Your job is to convert natural language questions into SQL queries.

CRITICAL RULES:
1. You can ONLY use objects that exist in the provided semantic model
2. You MUST return valid JSON in the exact schema specified
3. If you're uncertain or if required objects are missing, you MUST refuse with clarifying questions
4. Never invent tables, columns, or measures that don't exist in the model

SEMANTIC MODEL SUMMARY:
- Entities: {list(self.whitelist['entities'].keys())}
- Facts: {list(self.whitelist['facts'].keys())}
- Dimensions: {list(self.whitelist['dimensions'].keys())}
- Available Measures: {list(self.whitelist['measures'].keys())}

DIALECT: {self.dialect}

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "ok" | "refuse",
  "sql": [
    {{
      "dialect": "{self.dialect}",
      "statement": "SELECT ...",
      "explanation": "brief explanation",
      "evidence": {{
        "entities": ["Entity1"],
        "facts": ["Fact1"],
        "measures": ["Measure1"]
      }},
      "limits": {{
        "row_limit": 1000,
        "timeout_sec": 60
      }}
    }}
  ],
  "next_steps": ["optional suggestions"],
  "refusal": {{
    "reason": "why refused (only if status=refuse)",
    "missing": ["list of missing objects"],
    "clarifying_questions": ["q1", "q2"],
    "suggestions": ["how to fix"]
  }}
}}

REFUSAL CRITERIA:
- If the question requires entities/measures not in the model → REFUSE
- If there's ambiguity in entity/measure mapping → REFUSE
- If the question is too vague → REFUSE
- Provide 2-3 specific clarifying questions

CONFIDENCE THRESHOLD: Only generate SQL if you're >90% confident all required objects exist.
"""
    
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer question with grounded SQL or refuse with explanation.
        
        Returns:
            Answer JSON following the spec schema
        """
        logger.info(f"Answering question: {question}")
        
        # Step 1: Quick check for obvious missing concepts
        missing_concepts = self._detect_missing_concepts(question)
        if missing_concepts:
            return self._create_refusal(
                reason=f"Required concepts not found in semantic model: {', '.join(missing_concepts)}",
                missing=missing_concepts,
                clarifying_questions=self._generate_clarifying_questions(question, missing_concepts)
            )
        
        # Step 2: Call LLM to generate SQL
        try:
            answer_json = self._call_llm(question)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._create_error_response(str(e))
        
        # Step 3: Validate response (QuadRail #2 - Constraint)
        is_valid, validation_errors = self._validate_answer_schema(answer_json)
        if not is_valid:
            logger.error(f"Schema validation failed: {validation_errors}")
            return self._create_error_response(f"Invalid response schema: {validation_errors}")
        
        # Step 4: If OK status, perform grounding checks (QuadRail #1 - Grounding)
        if answer_json.get("status") == "ok":
            is_grounded, grounding_errors = self._verify_grounding(answer_json)
            if not is_grounded:
                logger.warning(f"Grounding check failed: {grounding_errors}")
                return self._create_refusal(
                    reason="Generated SQL uses objects not in semantic model (hallucination detected)",
                    missing=grounding_errors,
                    clarifying_questions=[
                        "Which specific entities or measures do you want to query?",
                        "Can you rephrase the question using terms from the available data?"
                    ]
                )
            
            # Step 5: SQL validation (QuadRail #3 - Verification)
            is_sql_valid, sql_errors = self._validate_sql(answer_json)
            if not is_sql_valid:
                logger.error(f"SQL validation failed: {sql_errors}")
                return self._create_refusal(
                    reason=f"Generated SQL has syntax or semantic errors: {sql_errors}",
                    missing=[],
                    clarifying_questions=["Can you rephrase your question more specifically?"]
                )
        
        return answer_json
    
    def _detect_missing_concepts(self, question: str) -> List[str]:
        """
        Quick heuristic check for concepts not in the model.
        This is a fast pre-filter before calling the LLM.
        """
        question_lower = question.lower()
        missing = []
        
        # Common concept keywords to check
        concept_keywords = {
            "product": ["product", "item", "sku"],
            "customer": ["customer", "client", "buyer"],
            "date": ["date", "time", "month", "year", "day"],
            "sales": ["sales", "revenue", "orders"],
            "quantity": ["quantity", "units", "count"],
            "price": ["price", "cost", "amount"]
        }
        
        for concept, keywords in concept_keywords.items():
            # If question mentions this concept
            if any(kw in question_lower for kw in keywords):
                # Check if we have it in model
                found = False
                
                # Check entities
                for entity_name in self.whitelist["entities"].keys():
                    if concept in entity_name.lower():
                        found = True
                        break
                
                # Check facts
                if not found:
                    for fact_name in self.whitelist["facts"].keys():
                        if concept in fact_name.lower():
                            found = True
                            break
                
                # Check dimensions
                if not found:
                    for dim_name in self.whitelist["dimensions"].keys():
                        if concept in dim_name.lower():
                            found = True
                            break
                
                if not found:
                    missing.append(concept)
        
        return missing
    
    def _call_llm(self, question: str) -> Dict[str, Any]:
        """Call LLM with semantic model context."""
        
        # Prepare model context
        model_context = {
            "entities": list(self.whitelist["entities"].keys()),
            "facts": [
                {
                    "name": f_name,
                    "source": f_data.get("source"),
                    "measures": [m.get("name") for m in f_data.get("measures", [])]
                }
                for f_name, f_data in self.whitelist["facts"].items()
            ],
            "relationships": self.whitelist["relationships"],
            "metrics": self.model.get("metrics", [])
        }
        
        user_message = f"""Question: {question}

Semantic Model Context:
{json.dumps(model_context, indent=2)}

Generate SQL or refuse with clarifying questions. Return JSON only, no markdown."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Extract JSON (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            answer_json = json.loads(response_text)
            return answer_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse: {response_text}")
            raise ValueError(f"LLM returned invalid JSON: {e}")
    
    def _validate_answer_schema(self, answer: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate answer follows the spec schema (QuadRail #2)."""
        errors = []
        
        # Check required fields
        if "status" not in answer:
            errors.append("Missing 'status' field")
            return False, errors
        
        status = answer["status"]
        if status not in ["ok", "refuse"]:
            errors.append(f"Invalid status: {status} (must be 'ok' or 'refuse')")
        
        # Validate based on status
        if status == "ok":
            if "sql" not in answer:
                errors.append("Status 'ok' requires 'sql' field")
            elif not isinstance(answer["sql"], list):
                errors.append("'sql' must be a list")
            else:
                for idx, sql_obj in enumerate(answer["sql"]):
                    if "statement" not in sql_obj:
                        errors.append(f"SQL object {idx} missing 'statement'")
                    if "dialect" not in sql_obj:
                        errors.append(f"SQL object {idx} missing 'dialect'")
        
        elif status == "refuse":
            if "refusal" not in answer:
                errors.append("Status 'refuse' requires 'refusal' field")
            else:
                refusal = answer["refusal"]
                if "reason" not in refusal:
                    errors.append("Refusal missing 'reason'")
        
        return len(errors) == 0, errors
    
    def _verify_grounding(self, answer: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Verify all SQL uses only whitelisted objects (QuadRail #1).
        This is the core anti-hallucination check.
        """
        errors = []
        
        for sql_obj in answer.get("sql", []):
            sql = sql_obj.get("statement", "")
            
            # Parse SQL using sqlglot
            try:
                parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            except Exception as e:
                errors.append(f"Failed to parse SQL: {e}")
                continue
            
            # Extract tables referenced in SQL
            tables_in_sql = set()
            for table in parsed.find_all(sqlglot.exp.Table):
                table_name = table.name
                if table.db:
                    table_name = f"{table.db}.{table_name}"
                tables_in_sql.add(table_name)
            
            # Check if all tables are in whitelist
            for table in tables_in_sql:
                if table not in self.whitelist["tables"]:
                    errors.append(f"Table '{table}' not in semantic model")
        
        return len(errors) == 0, errors
    
    def _validate_sql(self, answer: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate SQL syntax and basic semantics (QuadRail #3).
        """
        errors = []
        
        for sql_obj in answer.get("sql", []):
            sql = sql_obj.get("statement", "")
            dialect = sql_obj.get("dialect", self.dialect)
            
            # Parse SQL
            try:
                parsed = sqlglot.parse_one(sql, dialect=dialect)
            except Exception as e:
                errors.append(f"SQL syntax error: {e}")
                continue
            
            # Check for dangerous operations (read-only enforcement)
            dangerous_ops = [sqlglot.exp.Insert, sqlglot.exp.Update, 
                           sqlglot.exp.Delete, sqlglot.exp.Drop, 
                           sqlglot.exp.Create, sqlglot.exp.Alter]
            
            for op in dangerous_ops:
                if parsed.find(op):
                    errors.append(f"Dangerous operation detected: {op.__name__}")
        
        return len(errors) == 0, errors
    
    def _generate_clarifying_questions(self, question: str, missing_concepts: List[str]) -> List[str]:
        """Generate helpful clarifying questions for refusals."""
        questions = []
        
        if "product" in missing_concepts:
            questions.append("The model doesn't have Product data. Did you mean to ask about OrderLines instead?")
        
        if "customer" in missing_concepts:
            questions.append("Which customer-related information are you looking for? (The model has a Customer entity)")
        
        if missing_concepts:
            questions.append(f"Can you rephrase your question without referring to: {', '.join(missing_concepts)}?")
        
        # Generic fallback
        if not questions:
            questions.append("Can you be more specific about what you want to query?")
            questions.append("Which specific metrics or dimensions are you interested in?")
        
        return questions[:3]  # Max 3 questions
    
    def _create_refusal(self, reason: str, missing: List[str], 
                       clarifying_questions: List[str],
                       suggestions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a refusal response."""
        return {
            "status": "refuse",
            "sql": [],
            "refusal": {
                "reason": reason,
                "missing": missing,
                "clarifying_questions": clarifying_questions,
                "suggestions": suggestions or [
                    "Review the available entities and measures",
                    "Try running 'python main.py explain-model' to see what's available"
                ]
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "status": "refuse",
            "sql": [],
            "refusal": {
                "reason": f"Internal error: {error_message}",
                "missing": [],
                "clarifying_questions": ["Please try rephrasing your question"],
                "suggestions": ["Contact support if this error persists"]
            }
        }