import json
from typing import Dict, Any, Tuple, Optional, List
from llm.azure_client import AzureLLMClient
from validation.schema_validator import SchemaValidator
from validation.sql_verifier import SQLVerifier
from validation.escalation_handler import EscalationHandler
from normalization.sql_normalizer import SQLNormalizer
import logging

logger = logging.getLogger(__name__)

class SQLAnswerer:
    """Phase 3: Question Answering with Natural Language Understanding."""
    
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


    def _fix_group_by_issues(self, sql: str) -> str:
        """
        Auto-fix common GROUP BY issues where SELECT expressions don't match GROUP BY.
        
        This is a best-effort fix for the common case where COALESCE() is used in SELECT
        but the GROUP BY references the underlying column.
        """
        import re
        
        # Pattern: Find SELECT with COALESCE and GROUP BY without it
        # This is a simplified fix - for production, use proper SQL parsing
        
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:ORDER|HAVING|$)', sql, re.IGNORECASE | re.DOTALL)
        
        if not select_match or not group_match:
            return sql  # Can't fix, return as-is
        
        select_clause = select_match.group(1)
        group_clause = group_match.group(1).strip()
        
        # Look for COALESCE expressions in SELECT
        coalesce_pattern = r'(COALESCE\([^)]+\))\s+AS\s+(\w+)'
        coalesces = re.findall(coalesce_pattern, select_clause, re.IGNORECASE)
        
        if not coalesces:
            return sql  # No COALESCE to fix
        
        # For each COALESCE with an alias, check if GROUP BY uses the base column
        for coalesce_expr, alias in coalesces:
            # If GROUP BY references the alias or a simple column, replace with full expression
            if alias in group_clause or re.search(r'\b' + alias + r'\b', group_clause):
                # Replace alias in GROUP BY with the full COALESCE expression
                group_clause = re.sub(r'\b' + alias + r'\b', coalesce_expr, group_clause)
                logger.info(f"Auto-fixed GROUP BY: replaced '{alias}' with '{coalesce_expr}'")
        
        # Rebuild SQL with fixed GROUP BY
        fixed_sql = re.sub(
            r'GROUP\s+BY\s+.*?(?=ORDER|HAVING|$)',
            f'GROUP BY {group_clause}',
            sql,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        return fixed_sql
    
    def answer_question(
        self,
        question: str,
        semantic_model: Dict[str, Any],
        compressed_discovery: Optional[Dict[str, Any]] = None  # NEW: Need this for NL mappings
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Answer natural language question with SQL using enhanced semantic understanding.
        
        Args:
            question: Natural language question
            semantic_model: Semantic model JSON
            compressed_discovery: REQUIRED for NL mappings
        
        Returns:
            (success, answer_json, error_message)
        """
        if not compressed_discovery:
            logger.warning("No compressed_discovery provided - NL mapping disabled")
        
        # STEP 1: Extract key terms from question and map to columns
        column_hints = self._extract_column_hints(question, compressed_discovery) if compressed_discovery else {}
        
        # STEP 2: Build enhanced user prompt with column hints
        user_prompt = self._build_enhanced_user_prompt(
            question, 
            semantic_model, 
            compressed_discovery,
            column_hints
        )
        
        # Try up to 2 times
        for attempt in range(2):
            try:
                logger.info(f"Q&A attempt {attempt + 1}/2")
                
                # Generate answer
                response = self.llm.generate(self.system_prompt, user_prompt)
                
                logger.debug(f"Raw LLM response (first 500 chars): {response[:500]}")
                
                # Parse JSON
                answer_json = self._extract_json(response)
                if not answer_json:
                    logger.warning("Failed to extract JSON from response")
                    continue
                
                # Auto-correct status field if needed (documented issue)
                if answer_json.get("status") == "success":
                    logger.warning("Auto-correcting status 'success' to 'ok'")
                    answer_json["status"] = "ok"
                
                logger.debug(f"Extracted JSON keys: {list(answer_json.keys())}")
                
                # Validate schema
                valid, schema_error = self.validator.validate(answer_json, "answer")
                if not valid:
                    logger.warning(f"Schema validation failed: {schema_error}")
                    if attempt == 0:
                        continue
                    return (False, {}, f"Schema validation failed: {schema_error}")
                
                # If refusal, return it (this is valid)
                if answer_json.get("status") == "refuse":
                    logger.info("LLM refused to answer - requesting clarification")
                    return (True, answer_json, "")
                
                if answer_json.get("status") == "ok":
                    for sql_obj in answer_json.get("sql", []):
                        sql_stmt = sql_obj.get("statement", "")
                        # Auto-fix: Detect and repair GROUP BY mismatches
                        sql_stmt = self._fix_group_by_issues(sql_stmt)
                        sql_obj["statement"] = sql_stmt
                
                # Verify SQL if we have discovery data
                if compressed_discovery:
                    # Reconstruct discovery format for verifier
                    discovery_for_verifier = self._reconstruct_discovery(compressed_discovery)
                    
                    dialect = semantic_model.get("audit", {}).get("dialect", "tsql")
                    sql_verifier = SQLVerifier(semantic_model, discovery_for_verifier)
                    
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
                    # Basic parse check only
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
    
    def _extract_column_hints(
        self, 
        question: str, 
        compressed_discovery: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        NEW: Extract key terms from question and map to columns using NL mappings.
        
        Returns:
            {
                "matched_terms": ["product name", "total sales", ...],
                "column_suggestions": ["dbo.Product.ProductName", "dbo.Sales.TotalAmount", ...]
            }
        """
        from discovery.discovery_compressor import DiscoveryCompressor
        
        question_lower = question.lower()
        hints = {
            "matched_terms": [],
            "column_suggestions": []
        }
        
        # Extract potential column references from question
        # Common patterns: "get product name", "total sales", "customer email"
        potential_terms = [
            "product name", "name", "title",
            "customer name", "customer email",
            "total", "amount", "price", "quantity",
            "date", "status", "description"
        ]
        
        for term in potential_terms:
            if term in question_lower:
                columns = DiscoveryCompressor.get_nl_mapping(compressed_discovery, term)
                if columns:
                    hints["matched_terms"].append(term)
                    hints["column_suggestions"].extend(columns)
        
        # Remove duplicates
        hints["column_suggestions"] = list(set(hints["column_suggestions"]))
        
        if hints["column_suggestions"]:
            logger.info(f"Mapped question terms to {len(hints['column_suggestions'])} columns: {hints['column_suggestions'][:5]}...")
        
        return hints
    
    def _build_enhanced_user_prompt(
        self, 
        question: str, 
        semantic_model: Dict[str, Any],
        compressed_discovery: Optional[Dict[str, Any]],
        column_hints: Dict[str, List[str]]
    ) -> str:
        """Build user prompt with ENHANCED column information and NL hints."""
        
        # Build focused column guide based on question
        focused_guide = self._build_focused_column_guide(
            semantic_model, 
            compressed_discovery, 
            column_hints
        )
        
        # Extract model summary (lighter than before)
        model_summary = {
            "facts": [
                {
                    "name": f["name"],
                    "source": f["source"],
                    "measures": [m["name"] for m in f.get("measures", [])]
                }
                for f in semantic_model.get("facts", [])
            ],
            "dimensions": [
                {
                    "name": d["name"],
                    "source": d["source"],
                    "attributes": d.get("attributes", [])[:10]  # Limit
                }
                for d in semantic_model.get("dimensions", [])
            ],
            "entities": [e["name"] for e in semantic_model.get("entities", [])],
            "dialect": semantic_model.get("audit", {}).get("dialect", "tsql")
        }
        
        prompt_parts = [
            "# Semantic Model Summary",
            json.dumps(model_summary, indent=2),
            "",
            "# IMPORTANT: Focused Column Guide for This Question",
            focused_guide,
            "",
            "# Natural Language Hints",
            f"Detected terms: {', '.join(column_hints.get('matched_terms', []))}",
            f"Suggested columns: {', '.join(column_hints.get('column_suggestions', [])[:10])}",
            "",
            "# Question",
            question,
            "",
            "# Task",
            "Generate SQL to answer the question.",
            "",
            "CRITICAL INSTRUCTIONS:",
            "1. Use the 'Focused Column Guide' above - it shows the EXACT columns you need",
            "2. Pay attention to 'Natural Language Hints' - these map question terms to columns",
            "3. For 'product name' queries, use columns with semantic_role='name'",
            "4. For status questions, use columns with semantic_role='status_indicator'",
            "5. Build WHERE clauses using the column descriptions provided",
            "",
            "Return ONLY valid JSON matching the Answer schema."
        ]
        
        return "\n".join(prompt_parts)
    
    def _build_focused_column_guide(
        self,
        semantic_model: Dict[str, Any],
        compressed_discovery: Optional[Dict[str, Any]],
        column_hints: Dict[str, List[str]]
    ) -> str:
        """
        NEW: Build a FOCUSED column guide showing only relevant columns for this question.
        """
        if not compressed_discovery:
            return "No column metadata available"
        
        suggested_cols = column_hints.get("column_suggestions", [])
        
        guide_lines = []
        guide_lines.append("## Key Columns for This Query:")
        guide_lines.append("")
        
        # Get classifications
        classifications = compressed_discovery.get("column_classifications", {})
        column_samples = compressed_discovery.get("column_samples", {})
        
        # Show suggested columns first (high priority)
        if suggested_cols:
            guide_lines.append("### Columns Matching Your Question:")
            for full_col in suggested_cols[:15]:  # Limit to 15
                classification = classifications.get(full_col, {})
                sample_data = column_samples.get(full_col, {})
                
                # Parse full_col: schema.table.column
                parts = full_col.split('.')
                if len(parts) == 3:
                    table_name = parts[1]
                    col_name = parts[2]
                    
                    role = classification.get("semantic_role", "unknown")
                    aliases = classification.get("nl_aliases", [])
                    values = sample_data.get("values", [])
                    
                    guide_lines.append(f"- **{table_name}.{col_name}**")
                    guide_lines.append(f"  - Role: {role}")
                    guide_lines.append(f"  - Natural language: {', '.join(aliases[:3])}")
                    if values:
                        guide_lines.append(f"  - Sample values: {', '.join(str(v) for v in values[:5])}")
                    guide_lines.append("")
        
        # Always show status indicators (critical for filtering)
        guide_lines.append("### Status Indicator Columns (for filtering active/cancelled):")
        for fact in semantic_model.get("facts", []):
            fact_name = fact.get("name")
            for col in fact.get("columns", []):
                if col.get("semantic_role") == "status_indicator":
                    guide_lines.append(
                        f"- {fact_name}.{col['name']}: {col.get('description', 'No description')}"
                    )
        
        # Show available measures
        guide_lines.append("\n### Pre-Calculated Measures:")
        for fact in semantic_model.get("facts", []):
            fact_name = fact.get("name")
            for measure in fact.get("measures", []):
                filters = measure.get("filters_applied", [])
                if filters:
                    guide_lines.append(
                        f"- {fact_name}.{measure['name']}: {measure['expression']} (Filters: {', '.join(filters)})"
                    )
                else:
                    guide_lines.append(
                        f"- {fact_name}.{measure['name']}: {measure['expression']}"
                    )
        
        return "\n".join(guide_lines)
    
    def _reconstruct_discovery(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct discovery format for SQL verifier."""
        schemas = {}
        for table_name, table_data in compressed.get("tables", {}).items():
            schema_name = table_data["schema"]
            if schema_name not in schemas:
                schemas[schema_name] = {"name": schema_name, "tables": []}
            
            schemas[schema_name]["tables"].append({
                "name": table_data["name"],
                "type": table_data["type"],
                "columns": table_data["columns"]
            })
        
        return {
            "database": compressed.get("database"),
            "dialect": compressed.get("dialect"),
            "schemas": list(schemas.values()),
            "named_assets": []
        }
    
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