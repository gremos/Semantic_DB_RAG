import json
from typing import Dict, Any, Tuple, Optional
from llm.azure_client import AzureLLMClient
from validation.schema_validator import SchemaValidator
from validation.sql_verifier import SQLVerifier
from validation.escalation_handler import EscalationHandler
from normalization.sql_normalizer import SQLNormalizer
import logging

logger = logging.getLogger(__name__)

class SQLAnswerer:
    """Phase 3: Question Answering using LLM."""
    
    def __init__(
        self, 
        llm_client: AzureLLMClient, 
        validator: SchemaValidator,
        normalizer: SQLNormalizer
    ):
        self.llm = llm_client
        self.validator = validator
        self.normalizer = normalizer
        self._load_prompt()
    
    def _load_prompt(self):
        """Load Q&A system prompt."""
        with open("prompts/qa_prompt.txt", 'r') as f:
            self.system_prompt = f.read()
    
    def answer_question(
        self,
        question: str,
        semantic_model: Dict[str, Any],
        discovery_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Answer natural language question with SQL.
        
        Args:
            question: Natural language question
            semantic_model: Semantic model JSON
            discovery_data: Optional discovery data for verification
        
        Returns:
            (success, answer_json, error_message)
        """
        # Build user prompt (semantic model only)
        user_prompt = self._build_user_prompt(question, semantic_model)
        
        # Try up to 2 times
        for attempt in range(2):
            try:
                logger.info(f"Q&A attempt {attempt + 1}/2")
                
                # Generate answer
                response = self.llm.generate(self.system_prompt, user_prompt)
                
                # DEBUG: Log response
                logger.debug(f"Raw LLM response (first 500 chars): {response[:500]}")
                
                # Parse JSON
                answer_json = self._extract_json(response)
                if not answer_json:
                    logger.warning("Failed to extract JSON from response")
                    logger.debug(f"Full response: {response}")
                    continue
                
                # DEBUG: Log extracted structure
                logger.debug(f"Extracted JSON keys: {list(answer_json.keys())}")
                
                # Validate schema
                valid, schema_error = self.validator.validate(answer_json, "answer")
                if not valid:
                    logger.warning(f"Schema validation failed: {schema_error}")
                    logger.debug(f"Invalid answer structure: {json.dumps(answer_json, indent=2)[:1000]}")
                    if attempt == 0:
                        continue
                    return (False, {}, f"Schema validation failed: {schema_error}")
                
                # If refusal, return it (this is valid)
                if answer_json.get("status") == "refuse":
                    logger.info("LLM refused to answer - requesting clarification")
                    return (True, answer_json, "")
                
                # Verify SQL only if we have discovery data
                if discovery_data:
                    dialect = semantic_model.get("audit", {}).get("dialect", "tsql")
                    sql_verifier = SQLVerifier(semantic_model, discovery_data)
                    
                    all_valid = True
                    issues = []
                    
                    for sql_obj in answer_json.get("sql", []):
                        sql_stmt = sql_obj.get("statement", "")
                        
                        # Parse check
                        success, error = self.normalizer.parse(sql_stmt, dialect)
                        if not success:
                            issues.append(f"SQL parse error: {error}")
                            all_valid = False
                            continue
                        
                        # Verification check
                        verified, verify_issues = sql_verifier.verify_sql(sql_stmt, dialect)
                        if not verified:
                            issues.extend(verify_issues)
                            all_valid = False
                    
                    if not all_valid:
                        logger.warning(f"SQL verification failed: {issues}")
                        if attempt == 0:
                            continue
                        
                        # Return refusal with issues
                        refusal = EscalationHandler.create_refusal(
                            reason="; ".join(issues),
                            missing_objects=[],
                            clarifying_questions=[]
                        )
                        return (True, refusal, "")
                else:
                    # Skip verification if no discovery data
                    logger.info("Skipping SQL verification (no discovery data provided)")
                    
                    # Still do basic parse check
                    dialect = semantic_model.get("audit", {}).get("dialect", "tsql")
                    for sql_obj in answer_json.get("sql", []):
                        sql_stmt = sql_obj.get("statement", "")
                        success, error = self.normalizer.parse(sql_stmt, dialect)
                        if not success:
                            logger.warning(f"SQL parse error: {error}")
                            if attempt == 0:
                                continue
                            return (True, EscalationHandler.create_refusal(
                                reason=f"SQL syntax error: {error}",
                                missing_objects=[],
                                clarifying_questions=[]
                            ), "")
                
                logger.info("Answer generated successfully")
                return (True, answer_json, "")
            
            except Exception as e:
                logger.error(f"Q&A attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == 1:
                    return (False, {}, str(e))
        
        # Escalate with clarifications
        refusal = EscalationHandler.create_refusal(
            reason="Could not generate valid SQL after 2 attempts",
            clarifying_questions=EscalationHandler.suggest_clarifications(question, semantic_model)
        )
        return (True, refusal, "")
    
    def _build_user_prompt(self, question: str, semantic_model: Dict[str, Any]) -> str:
        """Build user prompt with question and model ONLY (no discovery data)."""
        
        # Extract only what's needed from semantic model
        model_summary = {
            "entities": semantic_model.get("entities", []),
            "dimensions": semantic_model.get("dimensions", []),
            "facts": semantic_model.get("facts", []),
            "relationships": semantic_model.get("relationships", []),
            "metrics": semantic_model.get("metrics", []),
            "dialect": semantic_model.get("audit", {}).get("dialect", "tsql")
        }
        
        # Add specific guidance for this model
        guidance = """
IMPORTANT: In this semantic model:
- Revenue data is in the SalesItem fact (source: dbo.ContractProduct)
- Use the NetAmount measure for revenue calculations
- Customers are represented by the BusinessPoint entity
- Join SalesItem to BusinessPoint via BusinessPointID foreign key
"""
        
        return f"""# Semantic Model
{json.dumps(model_summary, indent=2)}

{guidance}

# Question
{question}

# Task
Generate SQL to answer the question. Return ONLY valid JSON matching the Answer schema.
"""
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response."""
        try:
            return json.loads(response)
        except:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
        
        return None