"""
SQL Generator for Q&A Phase
Generates grounded SQL from natural language questions using semantic model

Enhanced with Template-Based Generation:
- Uses verified SQL patterns from RDL/Views
- LLM adapts templates instead of inventing SQL from scratch
- Deterministic template matching for consistent results
- Falls back to schema-based generation if no template matches

Per REFACTORING_PLAN.md Phase 3: Deterministic Matching
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import sqlglot

from src.llm.client import get_llm_client

logger = logging.getLogger(__name__)


# Keywords for deterministic extraction (no LLM needed)
TEMPORAL_KEYWORDS = {
    'this month': 'this_month',
    'last month': 'last_month',
    'this year': 'this_year',
    'last year': 'last_year',
    'this quarter': 'this_quarter',
    'last quarter': 'last_quarter',
    'today': 'today',
    'yesterday': 'yesterday',
    'this week': 'this_week',
    'last week': 'last_week',
    # Greek
    'αυτόν τον μήνα': 'this_month',
    'τον προηγούμενο μήνα': 'last_month',
    'φέτος': 'this_year',
    'πέρυσι': 'last_year',
}

AGGREGATION_KEYWORDS = {
    'total': 'SUM',
    'sum': 'SUM',
    'count': 'COUNT',
    'average': 'AVG',
    'avg': 'AVG',
    'maximum': 'MAX',
    'max': 'MAX',
    'minimum': 'MIN',
    'min': 'MIN',
    # Greek
    'σύνολο': 'SUM',
    'πλήθος': 'COUNT',
    'μέσος': 'AVG',
}


class SQLGenerator:
    """
    Generate SQL from NL questions using semantic model.

    Enhanced with template-based generation:
    1. Extract keywords from question (deterministic - no LLM)
    2. Match against SQL templates from RDL/Views
    3. If high-confidence match: LLM adapts template (constrained)
    4. Else: Fall back to schema-based generation (less reliable)
    """

    # Template matching threshold
    TEMPLATE_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        semantic_model: Dict[str, Any],
        dialect: str = "tsql",
        templates_path: Optional[Path] = None
    ):
        self.semantic_model = semantic_model
        self.dialect = dialect
        self.llm_client = get_llm_client()

        # Build lookup indexes
        self.entities = {e['name']: e for e in semantic_model.get('entities', [])}
        self.dimensions = {d['name']: d for d in semantic_model.get('dimensions', [])}
        self.facts = {f['name']: f for f in semantic_model.get('facts', [])}
        self.relationships = semantic_model.get('relationships', [])

        # Load SQL templates (for template-based generation)
        # Templates now also include column_index for term->column mapping
        self._template_index = None
        self._column_index = None
        self._templates_data = None
        self._load_templates(templates_path)

        # Build reverse lookup: source table -> model object
        self._source_to_object: Dict[str, Dict[str, Any]] = {}
        for obj in list(self.entities.values()) + list(self.dimensions.values()) + list(self.facts.values()):
            source = obj.get('source', '')
            if source:
                self._source_to_object[source.lower()] = obj

        # Build table rankings lookup for audit-aware context building
        # First try semantic model rankings
        self._table_rankings: Dict[str, Dict[str, Any]] = {}
        for ranking in semantic_model.get('table_rankings', []):
            table_name = ranking.get('table', '').lower()
            if table_name:
                self._table_rankings[table_name] = ranking

        # Then load actual audit metrics from cache (has access_pattern)
        self._load_audit_metrics()

        # Extract audit metadata for context
        self._audit_metadata = semantic_model.get('audit', {}).get('production_audit', {})
        self._has_audit_data = bool(self._audit_metadata) or bool(self._table_rankings)

    def _load_audit_metrics(self):
        """Load audit metrics from cache to get table access patterns (HOT/WARM/COLD)"""
        try:
            from config.settings import get_path_config
            import json

            path_config = get_path_config()
            audit_file = path_config.cache_dir / 'audit_metrics.json'

            if audit_file.exists():
                with open(audit_file, 'r', encoding='utf-8') as f:
                    audit_data = json.load(f)

                # Build lookup from table_metrics
                for metric in audit_data.get('table_metrics', []):
                    full_name = metric.get('full_name', '').lower()
                    table_name = metric.get('table_name', '').lower()

                    # Store with access_pattern
                    ranking_data = {
                        'table': full_name,
                        'access_pattern': metric.get('access_pattern', ''),
                        'access_score': metric.get('access_score', 0),
                        'total_reads': metric.get('total_reads', 0)
                    }

                    # Add to rankings (prefer audit data over model rankings)
                    if full_name:
                        self._table_rankings[full_name] = ranking_data
                    if table_name:
                        self._table_rankings[table_name] = ranking_data

                logger.info(f"Loaded audit metrics: {len(audit_data.get('table_metrics', []))} tables with access patterns")
        except Exception as e:
            logger.debug(f"Could not load audit metrics: {e}")

    def _load_templates(self, templates_path: Optional[Path] = None):
        """Load SQL templates for template-based generation (includes column_index)"""
        try:
            from src.semantic.sql_templates import load_templates, TemplateIndex, ColumnIndex
            from config.settings import get_path_config

            if templates_path is None:
                path_config = get_path_config()
                templates_path = path_config.cache_dir / 'sql_templates.json'

            if templates_path.exists():
                self._templates_data = load_templates(templates_path)
                self._template_index = TemplateIndex(self._templates_data)
                logger.info(f"Loaded {len(self._templates_data.get('templates', []))} SQL templates")

                # Also load column_index if present (consolidated from column_fingerprints)
                column_index_data = self._templates_data.get('column_index', {})
                if column_index_data:
                    self._column_index = ColumnIndex(column_index_data)
                    logger.info(f"Loaded column_index with {len(column_index_data)} business terms")
            else:
                logger.warning(f"Templates file not found: {templates_path}")
        except Exception as e:
            logger.warning(f"Failed to load SQL templates: {e}")

    def _extract_keywords_deterministic(self, question: str) -> Dict[str, Any]:
        """
        Extract keywords from question WITHOUT using LLM.

        This is deterministic - same question always gives same keywords.
        Critical for consistent SQL generation (same Q = same SQL).

        Returns:
            {
                'terms': ['contract', 'campaign', ...],
                'temporal': 'this_month' | None,
                'aggregation': 'SUM' | 'COUNT' | None
            }
        """
        q_lower = question.lower()

        # Extract terms by splitting and cleaning
        words = re.findall(r'\b\w+\b', q_lower)
        # Filter stopwords and short words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'by', 'for', 'to', 'in', 'of',
                    'show', 'me', 'give', 'list', 'get', 'find', 'what', 'how', 'many',
                    'all', 'and', 'or', 'with', 'from', 'that', 'this', 'which'}
        terms = [w for w in words if len(w) > 2 and w not in stopwords]

        # Detect temporal context
        temporal = None
        for phrase, temporal_type in TEMPORAL_KEYWORDS.items():
            if phrase in q_lower:
                temporal = temporal_type
                break

        # Detect aggregation
        aggregation = None
        for keyword, agg_type in AGGREGATION_KEYWORDS.items():
            if keyword in q_lower:
                aggregation = agg_type
                break

        return {
            'terms': terms,
            'temporal': temporal,
            'aggregation': aggregation
        }

    def _find_matching_template(
        self,
        keywords: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Find best matching SQL template for given keywords.

        Enhanced with audit-aware ranking:
        - Templates using HOT tables get score boost
        - Templates using COLD/empty tables get score penalty

        Returns:
            (template, score) tuple or None if no good match
        """
        if not self._template_index:
            return None

        terms = keywords.get('terms', [])
        if not terms:
            return None

        # Search templates by keywords (get more candidates for re-ranking with audit data)
        matches = self._template_index.find_by_keywords(terms, limit=20)

        if not matches:
            return None

        # Re-rank with audit data, entity relevance, and aggregation mismatch
        reranked = []
        for template, score in matches:
            # Calculate audit boost based on tables used in template
            audit_boost = self._calculate_audit_boost(template) if self._table_rankings else 0.0

            # Calculate entity relevance boost - templates that have the main entity
            # as their PRIMARY table should score higher
            entity_boost = self._calculate_entity_relevance(template, terms)

            # Calculate aggregation mismatch penalty - dimension-only templates
            # get penalized when user wants aggregated data
            agg_penalty = self._calculate_aggregation_mismatch_penalty(template, keywords)

            adjusted_score = score + audit_boost + entity_boost + agg_penalty
            reranked.append((template, adjusted_score, score))
            if audit_boost != 0 or entity_boost != 0 or agg_penalty != 0:
                logger.debug(f"Template {template['name']}: base={score:.2f}, audit={audit_boost:.2f}, entity={entity_boost:.2f}, agg_penalty={agg_penalty:.2f}, final={adjusted_score:.2f}")

        # Sort by adjusted score
        reranked.sort(key=lambda x: -x[1])
        matches = [(t, adj_score) for t, adj_score, _ in reranked]

        if matches:
            best_template, score = matches[0]
            if score >= self.TEMPLATE_CONFIDENCE_THRESHOLD:
                logger.info(f"Template match: {best_template['name']} (score={score:.2f})")
                return (best_template, score)

        return None

    def _calculate_entity_relevance(self, template: Dict[str, Any], terms: List[str]) -> float:
        """
        Calculate boost based on whether template tables directly contain search terms.

        If user asks about "contracts", templates with Contract table should rank higher
        than templates that just mention contracts in their name but use other tables.

        Returns:
            Boost value (0.0 to 0.5)
        """
        tables = template.get('tables', [])
        if not tables:
            return 0.0

        boost = 0.0

        for term in terms:
            term_lower = term.lower()
            # Handle plurals
            term_singular = term_lower.rstrip('s') if term_lower.endswith('s') else term_lower

            for table in tables:
                table_lower = table.lower()
                # Strip schema prefix for comparison
                table_name = table_lower.split('.')[-1]

                # Check if term matches table name (accounting for View prefix)
                clean_table = table_name.replace('view', '').replace('dbo', '').strip('.')

                if term_singular in clean_table or clean_table in term_singular:
                    # Strong match - the entity is a primary table
                    boost += 0.3
                    break  # Only count once per term

        return min(boost, 0.5)  # Cap at 0.5

    def _is_dimension_only_template(self, template: Dict[str, Any]) -> bool:
        """
        Detect templates that only return dimensional/lookup data without fact measures.

        These templates are typically:
        - Year/Month/Date lookup tables (just return a list of dates)
        - Reference table lookups (just return codes/names)
        - No aggregations (SUM, COUNT, AVG) in the SQL

        Returns:
            True if template appears to be dimension-only (no fact data)
        """
        sql = template.get('sql', '').upper()
        tables = template.get('tables', [])
        name = template.get('name', '').lower()

        # Check if SQL has any aggregation functions
        has_aggregation = any(agg in sql for agg in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN('])

        # Check if tables are only CTEs/virtual (like yearlist)
        real_tables = [t for t in tables if not t.lower().startswith('yearlist') and
                       not t.lower().startswith('monthlist') and
                       not t.lower().startswith('datelist') and
                       len(t) > 3]

        # Check for dimension-only patterns in name
        dimension_patterns = ['year', 'month', 'date', 'calendar', 'lookup', 'reference', 'list']
        name_is_dimension = any(p in name for p in dimension_patterns)

        # It's a dimension-only template if:
        # - No aggregations AND (no real tables OR name suggests dimension lookup)
        if not has_aggregation and (len(real_tables) == 0 or name_is_dimension):
            return True

        return False

    def _calculate_aggregation_mismatch_penalty(
        self,
        template: Dict[str, Any],
        keywords: Dict[str, Any]
    ) -> float:
        """
        Calculate penalty for templates that don't match query intent.

        Key insight: If user asks for "revenue by year" they want:
        - Aggregated fact data (revenue/sales/amounts)
        - Grouped by a dimension (year)

        NOT:
        - A dimension lookup (just list of years)

        This method penalizes dimension-only templates when the query
        contains aggregation keywords (total, revenue, sales, count, etc.)

        Returns:
            Penalty value (negative float, 0.0 to -1.0)
        """
        terms = keywords.get('terms', [])
        aggregation = keywords.get('aggregation')

        # Measure/aggregation terms that indicate user wants fact data
        measure_terms = {'revenue', 'sales', 'total', 'amount', 'value', 'price',
                        'count', 'sum', 'average', 'profit', 'cost', 'discount',
                        'quantity', 'volume', 'income', 'expense'}

        # Check if query contains measure terms
        has_measure_terms = any(t.lower() in measure_terms for t in terms)

        # Check if query has explicit aggregation keyword detected
        wants_aggregation = aggregation is not None or has_measure_terms

        if not wants_aggregation:
            return 0.0  # No penalty - user might want dimension lookup

        # Check if template is dimension-only
        if self._is_dimension_only_template(template):
            # Heavy penalty - user wants aggregated data but template only returns dimensions
            logger.debug(f"Template {template.get('name')} is dimension-only but query wants aggregation - penalty applied")
            return -0.8

        return 0.0

    def _calculate_audit_boost(self, template: Dict[str, Any]) -> float:
        """
        Calculate score boost/penalty based on audit data for tables in template.

        Enhanced scoring:
        - HOT tables: +0.5 boost (increased from 0.3)
        - WARM tables: +0.2 boost (increased from 0.1)
        - COLD tables: -0.3 penalty (increased from -0.2)
        - ARCHIVE tables: -0.5 penalty
        - Very low read tables (< 1000 total): -0.4 penalty
        - Cross-database refs: -1.0 penalty (likely won't work in dev)
        - No audit data: 0 (neutral)
        """
        sql = template.get('sql', '')

        # Heavy penalty for cross-database references (won't work in dev environment)
        if self._has_cross_db_reference(sql):
            return -1.0

        tables = template.get('tables', [])
        if not tables:
            # Try to extract from SQL
            tables = self._extract_tables_from_sql(sql)

        if not tables:
            return 0.0

        total_boost = 0.0
        matched_tables = 0
        has_very_low_read_table = False

        for table in tables:
            table_lower = table.lower()

            # Skip view names that aren't real tables (they reference other tables in the list)
            # For example, if tables=['ViewContract', 'dbo.Contract', 'dbo.BusinessPoint'],
            # we should skip ViewContract and only count the actual tables
            if table_lower.startswith('view') and len(tables) > 1:
                continue

            # Try with and without schema prefix
            ranking = (
                self._table_rankings.get(table_lower) or
                self._table_rankings.get(f'dbo.{table_lower}')
            )

            if ranking:
                tier = ranking.get('tier', '').upper()
                access_pattern = ranking.get('access_pattern', '').lower()
                total_reads = ranking.get('total_reads', 0)

                # Check for very low read count (essentially unused table)
                if total_reads < 1000:
                    has_very_low_read_table = True
                    logger.debug(f"Table {table} has very low reads: {total_reads}")

                # Use tier or access_pattern with increased weights
                if tier == 'HOT' or access_pattern == 'hot':
                    total_boost += 0.5
                    matched_tables += 1
                elif tier == 'WARM' or access_pattern == 'warm':
                    total_boost += 0.2
                    matched_tables += 1
                elif tier == 'COLD' or access_pattern == 'cold':
                    total_boost -= 0.3
                    matched_tables += 1
                elif access_pattern == 'archive':
                    total_boost -= 0.5
                    matched_tables += 1

        # Apply penalty for templates using very low read tables
        # These are likely historical/unused tables without current data
        if has_very_low_read_table:
            total_boost -= 0.4

        # Average boost if multiple tables
        if matched_tables > 0:
            return total_boost / matched_tables
        return 0.0

    def _has_cross_db_reference(self, sql: str) -> bool:
        """Check if SQL contains cross-database references that won't work in dev"""
        if not sql:
            return False

        sql_lower = sql.lower()
        # Common cross-database patterns
        cross_db_patterns = [
            'yps.dbo.',
            'itapps.dbo.',
            'itapps.',
            'yps.',
            # Three-part names like [DatabaseName].[schema].[table]
            r'\w+\.\w+\.\w+',  # But handled via regex below
        ]

        for pattern in cross_db_patterns[:4]:  # Check string patterns
            if pattern in sql_lower:
                return True

        # Check for three-part names (database.schema.table) that aren't dbo
        # Pattern: word.dbo.word where first word is NOT dbo
        three_part = re.findall(r'(\w+)\.dbo\.\w+', sql_lower)
        for db_name in three_part:
            if db_name not in ('dbo', ''):
                return True

        return False

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL (simple regex extraction)"""
        if not sql:
            return []

        tables = set()
        # Match FROM/JOIN table patterns
        patterns = [
            r'\bFROM\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?',
            r'\bJOIN\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, sql, re.IGNORECASE):
                tables.add(match.group(1))

        return list(tables)

    def generate_sql(self, question: str, row_limit: int = 10) -> Dict[str, Any]:
        """
        Generate SQL from natural language question.

        New Flow (Template-Based):
        1. Extract keywords deterministically (no LLM)
        2. Match against SQL templates
        3. If template match: Use constrained LLM adaptation
        4. Else: Fall back to schema-based generation

        Returns:
            {
                'status': 'ok'|'refuse',
                'sql': [...],
                'confidence': 0.0-1.0,
                'evidence': {...},
                'generation_method': 'template'|'schema',
                'refusal': {...} if status='refuse'
            }
        """
        logger.info(f"Generating SQL for: {question}")

        # Step 1: Extract keywords deterministically
        keywords = self._extract_keywords_deterministic(question)
        logger.debug(f"Extracted keywords: {keywords}")

        # Step 2: Try template-based generation first
        template_match = self._find_matching_template(keywords)

        if template_match:
            template, template_score = template_match
            logger.info(f"Using template-based generation: {template['name']}")

            # Generate SQL by adapting template
            sql_response = self._generate_from_template(
                question, template, keywords, row_limit
            )

            # Validate
            validation = self._validate_sql(sql_response.get('statement', ''))
            if validation['valid']:
                return {
                    'status': 'ok',
                    'sql': [sql_response],
                    'confidence': min(0.95, template_score + 0.2),  # Template = higher confidence
                    'evidence': {
                        'template_used': template['name'],
                        'template_source': template['source'],
                        'tables': template.get('tables', []),
                    },
                    'generation_method': 'template'
                }
            else:
                logger.warning(f"Template adaptation failed: {validation['errors']}")
                # Fall through to schema-based

        # Step 3: Fall back to schema-based generation
        logger.info("Using schema-based generation (no template match)")

        # Analyze intent with LLM
        intent = self._analyze_intent(question)
        confidence = self._calculate_confidence(intent)

        # If low confidence, refuse
        if confidence < 0.50:
            return self._create_refusal(question, intent, confidence)

        # Generate from schema
        sql_response = self._generate_grounded_sql(question, intent, row_limit)

        # Validate
        validation = self._validate_sql(sql_response.get('statement', ''))
        if not validation['valid']:
            logger.error(f"Generated invalid SQL: {validation['errors']}")
            return self._create_refusal(question, intent, 0.0, validation['errors'])

        return {
            'status': 'ok',
            'sql': [sql_response],
            'confidence': confidence * 0.8,  # Lower confidence for schema-based
            'evidence': intent.get('evidence', {}),
            'generation_method': 'schema'
        }

    def _generate_from_template(
        self,
        question: str,
        template: Dict[str, Any],
        keywords: Dict[str, Any],
        row_limit: int
    ) -> Dict[str, Any]:
        """
        Generate SQL by adapting a verified template.

        The LLM is constrained to:
        - Keep table names exactly as in template
        - Keep JOIN conditions exactly as in template
        - Only modify SELECT, WHERE, GROUP BY, ORDER BY

        This prevents hallucination of table/column names.
        """
        # Build constrained prompt
        columns = template.get('columns', {})
        selectable = columns.get('selectable', [])
        filterable = columns.get('filterable', [])
        groupable = columns.get('groupable', [])

        temporal = keywords.get('temporal')
        temporal_hint = ""
        if temporal:
            temporal_hint = f"\nTemporal filter needed: {temporal}"
            if temporal == 'this_month':
                temporal_hint += "\n  Use: WHERE date_column >= DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0)"
            elif temporal == 'last_month':
                temporal_hint += "\n  Use: WHERE date_column >= DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) - 1, 0) AND date_column < DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0)"
            elif temporal == 'this_year':
                temporal_hint += "\n  Use: WHERE date_column >= DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0)"
            elif temporal == 'last_year':
                temporal_hint += "\n  Use: WHERE YEAR(date_column) = YEAR(GETDATE()) - 1"

        system_prompt = f"""You are a SQL adapter. Adapt the VERIFIED template below to answer the user's question.

TEMPLATE: {template['name']}
SOURCE: {template['source']} (VERIFIED PRODUCTION SQL)

TEMPLATE SQL:
{template.get('sql', '')}

AVAILABLE COLUMNS (use ONLY these):
- Selectable: {', '.join(selectable[:20]) if selectable else 'all columns from template'}
- Filterable: {', '.join(filterable[:10]) if filterable else 'all columns'}
- Groupable: {', '.join(groupable[:10]) if groupable else 'all columns'}
{temporal_hint}

RULES - CRITICAL:
1. DO NOT change table names - use exactly: {', '.join(template.get('tables', []))}
2. DO NOT change JOIN conditions - they are verified
3. DO NOT invent columns - use ONLY columns from the template
4. ONLY modify: SELECT columns, WHERE filters, GROUP BY, ORDER BY
5. Always use TOP({row_limit}) for SQL Server
6. Use proper column quoting with square brackets []

Respond with JSON:
{{
  "dialect": "tsql",
  "statement": "SELECT TOP({row_limit}) ...",
  "explanation": "Adapted template to ...",
  "evidence": {{
    "template_used": "{template['name']}",
    "tables_used": [...],
    "columns_selected": [...]
  }}
}}"""

        response = self.llm_client.invoke_with_json(
            f"Adapt this template to answer: {question}",
            system_prompt=system_prompt
        )

        response['limits'] = {'row_limit': row_limit, 'timeout_sec': 60}
        return response

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

        response = self.llm_client.invoke_with_json(
            f"Analyze this question:\n\n{question}",
            system_prompt=system_prompt
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
6. Use schema-qualified table names (e.g., dbo.TableName)

Respond with JSON:
{{
  "dialect": "{self.dialect}",
  "statement": "SELECT TOP({row_limit}) ...",
  "explanation": "This query joins Customer and Sales tables...",
  "evidence": {{
    "tables_used": ["dbo.Customer", "dbo.FactSales"],
    "joins_used": ["Customer.CustomerID = Sales.CustomerID"],
    "measures_used": ["SUM(ExtendedAmount) as Revenue"]
  }}
}}"""

        response = self.llm_client.invoke_with_json(
            f"Generate SQL for: {question}\n\nIntent: {intent}",
            system_prompt=system_prompt
        )

        response['limits'] = {'row_limit': row_limit, 'timeout_sec': 60}
        return response

    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate SQL using sqlglot"""
        if not sql:
            return {'valid': False, 'errors': ['Empty SQL statement']}

        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)

            # Check for DML/DDL using word boundaries to avoid false positives
            # (e.g., CreatedOn contains CREATE but is a valid column name)
            forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
            sql_upper = sql.upper()
            for keyword in forbidden:
                # Use regex with word boundaries
                pattern = r'\b' + keyword + r'\b'
                if re.search(pattern, sql_upper):
                    return {
                        'valid': False,
                        'errors': [f"Forbidden keyword: {keyword}"]
                    }

            return {'valid': True, 'errors': []}

        except Exception as e:
            return {'valid': False, 'errors': [str(e)]}

    def _format_measures(self) -> str:
        """Format available measures for prompt"""
        lines = []
        for fact_name, fact in self.facts.items():
            measures = [m['name'] for m in fact.get('measures', [])]
            if measures:
                lines.append(f"  {fact_name}: {', '.join(measures)}")
        return "\n".join(lines) if lines else "  (no measures defined)"

    def _get_table_access_info(self, source_table: str) -> Dict[str, Any]:
        """
        Get audit access information for a table from rankings

        Args:
            source_table: Full table name (schema.table)

        Returns:
            Dict with access_pattern, access_score, rank, is_hot
        """
        ranking = self._table_rankings.get(source_table.lower(), {})
        audit_info = ranking.get('audit', {})

        if audit_info:
            return {
                'access_pattern': audit_info.get('access_pattern', 'unknown'),
                'access_score': audit_info.get('access_score', 50.0),
                'rank': ranking.get('rank', 5),
                'is_hot': audit_info.get('access_pattern') == 'hot',
                'is_warm': audit_info.get('access_pattern') == 'warm',
                'is_cold': audit_info.get('access_pattern') in ('cold', 'unused'),
                'is_history': audit_info.get('is_history', False)
            }
        else:
            return {
                'access_pattern': 'unknown',
                'access_score': 50.0,
                'rank': ranking.get('rank', 5),
                'is_hot': False,
                'is_warm': False,
                'is_cold': False,
                'is_history': False
            }

    def _sort_objects_by_priority(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort semantic model objects by production usage priority

        Hot/warm tables come first, cold/unused tables come last.
        This ensures the LLM sees the most relevant tables first.

        Args:
            objects: List of semantic model objects (entities, dimensions, facts)

        Returns:
            Sorted list with hot tables first
        """
        if not self._has_audit_data:
            return objects

        def get_priority(obj: Dict[str, Any]) -> tuple:
            source = obj.get('source', '')
            access_info = self._get_table_access_info(source)

            # Priority order: hot (0), warm (1), unknown (2), cold (3), unused (4)
            pattern_priority = {
                'hot': 0,
                'warm': 1,
                'unknown': 2,
                'cold': 3,
                'unused': 4,
                'archive': 5
            }
            pattern = access_info['access_pattern']
            primary_priority = pattern_priority.get(pattern, 2)

            # Secondary sort by access score (higher = better)
            secondary_priority = -access_info['access_score']

            return (primary_priority, secondary_priority)

        return sorted(objects, key=get_priority)

    def _find_required_relationships(self, intent: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Find relationships needed to join entities/dimensions/facts in the query

        Args:
            intent: Parsed intent with entities, dimensions, facts lists

        Returns:
            List of relationship dicts with from/to/confidence
        """
        required_rels = []

        # Get all objects mentioned in intent
        mentioned_objects: Set[str] = set()
        mentioned_objects.update(intent.get('entities', []))
        mentioned_objects.update(intent.get('dimensions', []))
        mentioned_objects.update(intent.get('facts', []))

        if len(mentioned_objects) < 2:
            return []

        # Find relationships connecting these objects
        for rel in self.relationships:
            from_obj = rel.get('from', '').split('.')[0]  # Get object name before column
            to_obj = rel.get('to', '').split('.')[0]

            # Check if this relationship connects mentioned objects
            if from_obj in mentioned_objects and to_obj in mentioned_objects:
                required_rels.append({
                    'from': rel.get('from', ''),
                    'to': rel.get('to', ''),
                    'confidence': rel.get('confidence', 'medium')
                })

        return required_rels

    def _build_sql_context(self, intent: Dict[str, Any]) -> str:
        """
        Build SQL context from intent - extract relevant tables, columns, and relationships

        Now with audit-aware filtering:
        - Prioritizes hot/warm tables (actively used in production)
        - Includes access pattern hints for LLM context
        - Filters out cold/unused tables when better alternatives exist

        Args:
            intent: Parsed intent with entities, dimensions, facts

        Returns:
            Formatted string with tables, columns, and join paths
        """
        lines = []

        # Add audit context header if available
        if self._has_audit_data:
            lines.append("-- PRODUCTION AUDIT DATA AVAILABLE")
            lines.append("-- Tables marked with [HOT] are actively queried in production - prefer these")
            lines.append("-- Tables marked with [COLD] are rarely used - use only if no alternative")
            lines.append("")

        # Collect all relevant objects
        relevant_objects: List[Dict[str, Any]] = []

        for entity_name in intent.get('entities', []):
            if entity_name in self.entities:
                relevant_objects.append(self.entities[entity_name])

        for dim_name in intent.get('dimensions', []):
            if dim_name in self.dimensions:
                relevant_objects.append(self.dimensions[dim_name])

        for fact_name in intent.get('facts', []):
            if fact_name in self.facts:
                relevant_objects.append(self.facts[fact_name])

        # If no specific objects found, include a sample of available objects
        # Prioritize hot/warm tables when falling back
        if not relevant_objects:
            # Sort facts and dimensions by production usage before selecting
            sorted_facts = self._sort_objects_by_priority(list(self.facts.values()))
            sorted_dims = self._sort_objects_by_priority(list(self.dimensions.values()))

            # Include up to 5 facts and their related dimensions (hot/warm first)
            for fact in sorted_facts[:5]:
                relevant_objects.append(fact)
            for dim in sorted_dims[:5]:
                relevant_objects.append(dim)
        else:
            # Sort selected objects by priority (hot first)
            relevant_objects = self._sort_objects_by_priority(relevant_objects)

        # Track cold tables - we may want to filter these out
        cold_objects = []
        hot_warm_objects = []

        for obj in relevant_objects:
            source = obj.get('source', '')
            access_info = self._get_table_access_info(source)

            if access_info['is_cold'] or access_info['is_history']:
                cold_objects.append(obj)
            else:
                hot_warm_objects.append(obj)

        # If we have hot/warm alternatives, filter out cold tables for facts
        # (entities/dimensions are always needed for joins)
        objects_to_include = relevant_objects
        if hot_warm_objects and cold_objects and self._has_audit_data:
            # Only include cold tables if they're required (entities/dimensions for joins)
            objects_to_include = hot_warm_objects
            for cold_obj in cold_objects:
                # Check if this cold table is referenced in a required relationship
                obj_name = cold_obj.get('name', '')
                is_required = any(
                    obj_name in rel.get('from', '') or obj_name in rel.get('to', '')
                    for rel in self._find_required_relationships(intent)
                )
                if is_required:
                    objects_to_include.append(cold_obj)
                    logger.debug(f"Including cold table {obj_name} - required for joins")
                else:
                    logger.debug(f"Filtering out cold table {obj_name} - not required")

        # Format each object with access pattern info
        for obj in objects_to_include:
            obj_type = obj.get('type', 'unknown')
            obj_name = obj.get('name', 'Unknown')
            source = obj.get('source', 'unknown')

            # Get access pattern for annotation
            access_info = self._get_table_access_info(source)
            access_tag = ""
            if self._has_audit_data:
                if access_info['is_hot']:
                    access_tag = " [HOT - PREFERRED]"
                elif access_info['is_warm']:
                    access_tag = " [WARM]"
                elif access_info['is_cold']:
                    access_tag = " [COLD - USE IF NO ALTERNATIVE]"

            lines.append(f"\n-- {obj_type.upper()}: {obj_name}{access_tag}")
            lines.append(f"-- Source: {source}")

            # List columns
            columns = obj.get('columns', [])
            if columns:
                col_names = [c.get('name', '') for c in columns if c.get('name')]
                lines.append(f"-- Columns: {', '.join(col_names[:20])}")  # Limit to 20 columns

            # List measures (for facts)
            measures = obj.get('measures', [])
            if measures:
                measure_strs = []
                for m in measures:
                    m_name = m.get('name', '')
                    m_expr = m.get('expression', m.get('column', ''))
                    measure_strs.append(f"{m_name} ({m_expr})")
                lines.append(f"-- Measures: {', '.join(measure_strs)}")

        # Add relationships with confidence from audit
        required_rels = self._find_required_relationships(intent)
        if required_rels:
            lines.append("\n-- RELATIONSHIPS (Join Paths):")
            for rel in required_rels:
                # Mark high-confidence relationships from audit
                confidence = rel.get('confidence', 'medium')
                confidence_tag = ""
                if confidence in ('very_high', 'high'):
                    confidence_tag = " [VERIFIED]"
                lines.append(f"--   {rel['from']} -> {rel['to']} ({confidence}){confidence_tag}")

        return "\n".join(lines)

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
