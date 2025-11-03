"""
SQL Generator for Q&A Phase
Generates grounded SQL from natural language questions using semantic model
"""

import logging
from typing import Dict, Any, List, Optional
import sqlglot

from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

class SQLGenerator:
    """Generate SQL from NL questions using semantic model"""
    
    def __init__(self, semantic_model: Dict[str, Any], dialect: str = "mssql"):
        self.semantic_model = semantic_model
        self.dialect = dialect
        self.llm_client = LLMClient()
        
        # Build lookup indexes
        self.entities = {e['name']: e for e in semantic_model.get('entities', [])}
        self.dimensions = {d['name']: d for d in semantic_model.get('dimensions', [])}
        self.facts = {f['name']: f for f in semantic_model.get('facts', [])}
        self.relationships = semantic_model.get('relationships', [])
    
    def generate_sql(self, question: str, row_limit: int = 10) -> Dict[str, Any]:
        """
        Generate SQL from natural language question
        
        Returns:
            {
                'status': 'ok'|'refuse',
                'sql': [...],
                'confidence': 0.0-1.0,
                'evidence': {...},
                'refusal': {...} if status='refuse'
            }
        """
        logger.info(f"Generating SQL for: {question}")
        
        # Step 1: Analyze question intent
        intent = self._analyze_intent(question)
        
        # Step 2: Calculate confidence
        confidence = self._calculate_confidence(intent)
        
        # Step 3: If low confidence, refuse with clarifications
        if confidence < 0.50:
            return self._create_refusal(question, intent, confidence)
        
        # Step 4: Generate SQL
        sql_response = self._generate_grounded_sql(question, intent, row_limit)
        
        # Step 5: Validate SQL
        validation = self._validate_sql(sql_response['statement'])
        if not validation['valid']:
            logger.error(f"Generated invalid SQL: {validation['errors']}")
            return self._create_refusal(question, intent, 0.0, validation['errors'])
        
        return {
            'status': 'ok',
            'sql': [sql_response],
            'confidence': confidence,
            'evidence': intent['evidence']
        }
    
    def _analyze_intent(self, question: str) -> Dict[str, Any]:
        """Analyze question to identify entities, measures, filters"""
        
        system_prompt = f"""You are a semantic query analyzer. Given a question and semantic model,
identify the entities, measures, dimensions, and filters needed.

Semantic Model Available:
Entities: {list(self.entities.keys())}
Dimensions: {list(self.dimensions.keys())}
Facts: {list(self.facts.keys())}

For each fact, these measures are available:
{self._format_measures()}

Respond with JSON:
{{
  "entities": ["Customer"],
  "dimensions": ["Date"],
  "facts": ["Sales"],
  "measures": ["Revenue", "Units"],
  "filters": [{{"dimension": "Date", "column": "Year", "op": ">=", "value": "2024"}}],
  "breakdowns": ["Customer.Name", "Date.Month"],
  "aggregation": "sum"|"count"|"avg",
  "temporal_context": "last_month"|"this_year"|null,
  "confidence_factors": {{
    "entity_clarity": 0.9,
    "measure_clarity": 0.8,
    "temporal_clarity": 0.7
  }}
}}"""
        
        response = self.llm_client.call_with_retry(
            system_prompt,
            f"Analyze this question:\n\n{question}"
        )
        
        # Build evidence
        response['evidence'] = {
            'entities': response.get('entities', []),
            'measures': response.get('measures', []),
            'relationships': self._find_required_relationships(response)
        }
        
        return response
    
    def _calculate_confidence(self, intent: Dict[str, Any]) -> float:
        """Calculate confidence score for query"""
        factors = intent.get('confidence_factors', {})
        
        # Weighted scoring
        weights = {
            'entity_clarity': 0.3,
            'measure_clarity': 0.3,
            'temporal_clarity': 0.2,
            'relationship_clarity': 0.2
        }
        
        score = sum(factors.get(k, 0.5) * w for k, w in weights.items())
        
        # Penalty if no measures found
        if not intent.get('measures'):
            score *= 0.5
        
        return min(1.0, max(0.0, score))
    
    def _generate_grounded_sql(
        self,
        question: str,
        intent: Dict[str, Any],
        row_limit: int
    ) -> Dict[str, Any]:
        """Generate SQL grounded in semantic model"""
        
        # Build context from semantic model
        context = self._build_sql_context(intent)
        
        system_prompt = f"""You are a SQL generator. Generate ONLY valid {self.dialect} SQL.

Available Tables and Columns:
{context}

Rules:
1. Use ONLY tables/columns listed above
2. Always use TOP({row_limit}) for SQL Server
3. Join tables using the relationships provided
4. Use proper aggregation functions (SUM, COUNT, AVG)
5. Add WHERE clauses for filters

Respond with JSON:
{{
  "dialect": "{self.dialect}",
  "statement": "SELECT TOP(10) ...",
  "explanation": "This query joins Customer and Sales tables...",
  "evidence": {{
    "tables_used": ["dbo.Customer", "dbo.FactSales"],
    "joins_used": ["Customer.CustomerID = Sales.CustomerID"],
    "measures_used": ["SUM(ExtendedAmount) as Revenue"]
  }}
}}"""
        
        response = self.llm_client.call_with_retry(
            system_prompt,
            f"Generate SQL for: {question}\n\nIntent: {intent}"
        )
        
        response['limits'] = {'row_limit': row_limit, 'timeout_sec': 60}
        return response
    
    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate SQL using sqlglot"""
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Check for DML/DDL
            forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE']
            sql_upper = sql.upper()
            for keyword in forbidden:
                if keyword in sql_upper:
                    return {
                        'valid': False,
                        'errors': [f"Forbidden keyword: {keyword}"]
                    }
            
            return {'valid': True, 'errors': []}
            
        except Exception as e:
            return {'valid': False, 'errors': [str(e)]}
    
    # Helper methods...
    def _format_measures(self) -> str:
        """Format available measures for prompt"""
        lines = []
        for fact_name, fact in self.facts.items():
            measures = [m['name'] for m in fact.get('measures', [])]
            lines.append(f"  {fact_name}: {', '.join(measures)}")
        return "\n".join(lines)
    
    def _find_required_relationships(self, intent: Dict[str, Any]) -> List[str]:
        """Find relationships needed for query"""
        # Implementation: match entities/dims/facts to relationships
        return []
    
    def _build_sql_context(self, intent: Dict[str, Any]) -> str:
        """Build SQL context from intent"""
        # Implementation: extract relevant tables/columns
        return ""
    
    def _create_refusal(
        self,
        question: str,
        intent: Dict[str, Any],
        confidence: float,
        errors: List[str] = None
    ) -> Dict[str, Any]:
        """Create refusal response with clarifying questions"""
        clarifications = []
        
        if not intent.get('measures'):
            clarifications.append("What metric would you like to analyze? (revenue, count, average?)")
        
        if not intent.get('entities') and not intent.get('dimensions'):
            clarifications.append("Which entities should I focus on? (customers, products, dates?)")
        
        if confidence < 0.30:
            clarifications.append("Can you rephrase with more specific details?")
        
        return {
            'status': 'refuse',
            'confidence': confidence,
            'refusal': {
                'reason': f"Low confidence ({confidence:.0%}) or validation errors",
                'errors': errors or [],
                'clarifying_questions': clarifications
            }
        }