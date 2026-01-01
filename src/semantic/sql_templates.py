"""
SQL Template Library

Extracts verified SQL patterns from:
1. RDL files (production reports)
2. Views (database-embedded queries)
3. Stored procedures (optional)

These templates enable constrained SQL generation:
- LLM adapts templates instead of inventing SQL
- Tables/JOINs are verified from production
- Reduces hallucination significantly

Also includes column_index extraction for business term -> column mapping.
(Consolidated from column_fingerprints.py per REFACTORING_PLAN.md simplification)

Per REFACTORING_PLAN.md Phase 2: SQL Templates (HIGH IMPACT)
"""

import json
import re
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# Business term patterns for Greek/English column name parsing
BUSINESS_TERM_PATTERNS = {
    # Financial terms
    "amount": ["amount", "αξία", "ποσό", "τιμή", "price", "cost", "value"],
    "revenue": ["revenue", "έσοδα", "sales", "πωλήσεις"],
    "discount": ["discount", "έκπτωση"],
    "tax": ["tax", "φπα", "vat", "φόρος"],

    # Entity terms
    "customer": ["customer", "πελάτης", "client", "κλιεντ"],
    "contract": ["contract", "συμβόλαιο", "σύμβαση"],
    "campaign": ["campaign", "καμπάνια"],
    "salesman": ["salesman", "πωλητής", "seller", "sales_rep"],
    "user": ["user", "χρήστης"],
    "product": ["product", "προϊόν"],

    # Temporal terms
    "date": ["date", "ημερομηνία", "ημ", "dt"],
    "created": ["created", "δημιουργία", "createdon", "createddate"],
    "modified": ["modified", "τροποποίηση", "updatedon", "modifieddate"],
    "signed": ["signed", "υπογραφή", "signdate"],

    # Status terms
    "status": ["status", "κατάσταση", "state"],
    "active": ["active", "ενεργό", "enabled"],
    "type": ["type", "τύπος", "kind"],

    # Identifiers
    "id": ["id", "κωδικός", "code", "key"],
    "name": ["name", "όνομα", "επωνυμία", "title"],
}

try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    logger.warning("sqlglot not available - SQL parsing will be limited")
    SQLGLOT_AVAILABLE = False


class SQLTemplateExtractor:
    """
    Extracts SQL templates from RDL files and Views.

    A template includes:
    - id: unique identifier
    - name: human-readable name
    - source: where it came from (rdl:filename, view:viewname)
    - keywords: terms for matching user questions
    - tables: tables referenced
    - columns: selectable/filterable/groupable columns
    - sql: the actual SQL
    - joins: join conditions
    """

    def __init__(
        self,
        discovery_path: Optional[Path] = None,
        rdl_path: Optional[Path] = None
    ):
        """
        Args:
            discovery_path: Path to discovery.json
            rdl_path: Path to RDL files directory
        """
        from config.settings import get_path_config
        path_config = get_path_config()

        if discovery_path is None:
            discovery_path = path_config.cache_dir / 'discovery.json'
        if rdl_path is None:
            rdl_path = path_config.rdl_path

        self.discovery_path = discovery_path
        self.rdl_path = Path(rdl_path) if rdl_path else None
        self._discovery_data: Optional[Dict] = None

    def load_discovery(self) -> Dict[str, Any]:
        """Load discovery.json"""
        if self._discovery_data is None:
            logger.info(f"Loading discovery data from {self.discovery_path}")
            with open(self.discovery_path, 'r', encoding='utf-8') as f:
                self._discovery_data = json.load(f)
        return self._discovery_data

    def extract_templates(self, include_column_index: bool = True) -> Dict[str, Any]:
        """
        Extract all SQL templates from Views and RDL files.

        Args:
            include_column_index: If True, also extract column_index for term->column mapping

        Returns:
            Dict with 'templates' list, 'index' for keyword lookup, and optionally 'column_index'
        """
        templates = []
        keyword_index = defaultdict(list)

        # Extract from Views
        view_templates = self._extract_view_templates()
        templates.extend(view_templates)

        # Extract from RDL files
        rdl_templates = self._extract_rdl_templates()
        templates.extend(rdl_templates)

        # Build keyword index
        for template in templates:
            template_id = template['id']
            for keyword in template.get('keywords', []):
                if template_id not in keyword_index[keyword]:
                    keyword_index[keyword].append(template_id)

        logger.info(f"Extracted {len(templates)} SQL templates "
                   f"({len(view_templates)} from views, {len(rdl_templates)} from RDL)")

        result = {
            'templates': templates,
            'index': dict(keyword_index),
            'metadata': {
                'total_templates': len(templates),
                'view_templates': len(view_templates),
                'rdl_templates': len(rdl_templates),
                'total_keywords': len(keyword_index)
            }
        }

        # Extract column_index for business term -> column mapping
        if include_column_index:
            column_index = self._extract_column_index()
            result['column_index'] = column_index
            result['metadata']['total_column_terms'] = len(column_index)
            logger.info(f"Extracted column_index with {len(column_index)} business terms")

        return result

    def _extract_column_index(self) -> Dict[str, List[str]]:
        """
        Extract column_index mapping business terms to columns.

        This enables quick lookup: "campaign" -> [dbo.Campaign.ID, dbo.Contract.CampaignID, ...]

        Returns:
            Dict mapping business terms to list of full column names (schema.table.column)
        """
        discovery = self.load_discovery()
        column_index: Dict[str, List[str]] = defaultdict(list)

        for schema in discovery.get('schemas', []):
            schema_name = schema.get('name', '')

            for table in schema.get('tables', []):
                table_name = table.get('name', '')
                full_table = f"{schema_name}.{table_name}"

                for col in table.get('columns', []):
                    col_name = col.get('name', '')
                    full_col = f"{full_table}.{col_name}"

                    # Infer business terms from column name
                    terms = self._infer_business_terms_from_column(col_name)

                    for term in terms:
                        if full_col not in column_index[term]:
                            column_index[term].append(full_col)

        return dict(column_index)

    def _infer_business_terms_from_column(self, col_name: str) -> List[str]:
        """Infer business terms from column name"""
        terms = set()
        col_lower = col_name.lower()

        # Split column name into parts (handle CamelCase and snake_case)
        parts = re.split(r'[_\s]+', col_lower)
        # Also split CamelCase
        camel_parts = re.findall(r'[a-z]+|[A-Z][a-z]*', col_name)
        all_parts = set(parts + [p.lower() for p in camel_parts])

        # Match against business term patterns
        for term, patterns in BUSINESS_TERM_PATTERNS.items():
            for pattern in patterns:
                if pattern in col_lower or pattern in all_parts:
                    terms.add(term)
                    break

        # Add the column name itself as a term (normalized)
        terms.add(col_lower.replace('_', ' ').strip())

        return list(terms)

    def _extract_view_templates(self) -> List[Dict[str, Any]]:
        """Extract templates from Views in discovery.json"""
        discovery = self.load_discovery()
        templates = []

        for asset in discovery.get('named_assets', []):
            if asset.get('kind') != 'view':
                continue

            view_name = asset.get('name', '')
            sql = asset.get('sql_normalized') or asset.get('sql_raw', '')

            if not sql:
                continue

            # Skip simple attribute/lookup views
            if self._is_simple_lookup_view(sql, view_name):
                continue

            template = self._parse_sql_to_template(
                sql=sql,
                source_type='view',
                source_name=view_name
            )

            if template:
                templates.append(template)

        return templates

    def _extract_rdl_templates(self) -> List[Dict[str, Any]]:
        """Extract templates from RDL files"""
        discovery = self.load_discovery()
        templates = []

        for asset in discovery.get('named_assets', []):
            if asset.get('kind') != 'rdl':
                continue

            rdl_name = asset.get('name', '')
            datasets = asset.get('datasets', [])

            for dataset in datasets:
                sql = dataset.get('sql_normalized') or dataset.get('query', '')
                dataset_name = dataset.get('name', 'Unknown')

                if not sql:
                    continue

                template = self._parse_sql_to_template(
                    sql=sql,
                    source_type='rdl',
                    source_name=f"{rdl_name}:{dataset_name}",
                    fields=dataset.get('fields', [])
                )

                if template:
                    templates.append(template)

        return templates

    def _is_simple_lookup_view(self, sql: str, view_name: str) -> bool:
        """Detect if a view is a simple lookup (e.g., Attr* views)"""
        # Skip Attr* views - they're just attribute lookups
        if 'Attr' in view_name and 'AttrRepository' in sql:
            return True

        # Skip views with no JOINs and very simple structure
        sql_upper = sql.upper()
        if 'JOIN' not in sql_upper:
            # Count SELECT columns
            if sql_upper.count(',') < 3:
                return True

        return False

    def _parse_sql_to_template(
        self,
        sql: str,
        source_type: str,
        source_name: str,
        fields: Optional[List[Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse SQL and extract template information.

        Returns:
            Template dict or None if parsing fails
        """
        # Generate unique ID
        template_id = self._generate_template_id(source_type, source_name)

        # Clean SQL
        sql_clean = self._clean_sql(sql)
        if not sql_clean:
            return None

        # Parse with sqlglot if available
        if SQLGLOT_AVAILABLE:
            try:
                parsed = sqlglot.parse_one(sql_clean, dialect='tsql')
                tables = self._extract_tables(parsed)
                columns = self._extract_columns(parsed)
                joins = self._extract_joins(parsed)
                aggregations = self._detect_aggregations(parsed)
            except Exception as e:
                logger.debug(f"SQL parsing failed for {source_name}: {e}")
                # Fall back to basic extraction
                tables = self._extract_tables_basic(sql_clean)
                columns = {'selectable': [], 'filterable': [], 'groupable': []}
                joins = []
                aggregations = []
        else:
            tables = self._extract_tables_basic(sql_clean)
            columns = {'selectable': [], 'filterable': [], 'groupable': []}
            joins = []
            aggregations = []

        # Skip templates with no meaningful tables
        if not tables:
            return None

        # Generate keywords
        keywords = self._generate_keywords(source_name, tables, columns, fields)

        # Build human-readable name
        display_name = self._generate_display_name(source_type, source_name)

        return {
            'id': template_id,
            'name': display_name,
            'source_type': source_type,
            'source': f"{source_type}:{source_name}",
            'keywords': keywords,
            'tables': tables,
            'columns': columns,
            'joins': joins,
            'aggregations': aggregations,
            'sql': sql_clean
        }

    def _generate_template_id(self, source_type: str, source_name: str) -> str:
        """Generate unique template ID"""
        content = f"{source_type}:{source_name}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        # Clean name for ID
        name_clean = re.sub(r'[^a-zA-Z0-9]+', '_', source_name.lower())[:40]
        return f"{source_type}_{name_clean}_{hash_suffix}"

    def _clean_sql(self, sql: str) -> str:
        """Clean SQL for processing"""
        if not sql:
            return ''

        # Remove CREATE VIEW wrapper
        sql = re.sub(r'CREATE\s+VIEW\s+\[?\w+\]?\.\[?\w+\]?\s+AS\s*', '', sql, flags=re.IGNORECASE)

        # Remove XML entities
        sql = sql.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')

        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()

        return sql

    def _extract_tables(self, parsed: Any) -> List[str]:
        """Extract table names from parsed SQL"""
        tables = []
        try:
            for table in parsed.find_all(exp.Table):
                # Build full table name
                parts = []
                if table.catalog:
                    parts.append(table.catalog)
                if table.db:
                    parts.append(table.db)
                if table.name:
                    parts.append(table.name)

                if parts:
                    full_name = '.'.join(parts)
                    # Clean brackets
                    full_name = full_name.replace('[', '').replace(']', '')
                    if full_name and full_name not in tables:
                        tables.append(full_name)
        except Exception as e:
            logger.debug(f"Table extraction error: {e}")

        return tables

    def _extract_tables_basic(self, sql: str) -> List[str]:
        """Basic table extraction without sqlglot"""
        tables = []

        # Match FROM and JOIN clauses
        patterns = [
            r'FROM\s+\[?(\w+)\]?\.\[?(\w+)\]?',  # FROM schema.table
            r'FROM\s+\[?(\w+)\]?(?:\s|$)',        # FROM table
            r'JOIN\s+\[?(\w+)\]?\.\[?(\w+)\]?',   # JOIN schema.table
            r'JOIN\s+\[?(\w+)\]?(?:\s|$)',        # JOIN table
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    table = '.'.join(m for m in match if m)
                else:
                    table = match
                table = table.replace('[', '').replace(']', '')
                if table and table not in tables:
                    tables.append(table)

        return tables

    def _extract_columns(self, parsed: Any) -> Dict[str, List[str]]:
        """Extract columns categorized by usage"""
        columns = {
            'selectable': [],
            'filterable': [],
            'groupable': [],
            'aggregatable': []
        }

        try:
            # SELECT columns
            for select in parsed.find_all(exp.Select):
                for expr in select.expressions:
                    col_name = self._get_column_name(expr)
                    if col_name and col_name not in columns['selectable']:
                        columns['selectable'].append(col_name)

            # WHERE columns (filterable)
            for where in parsed.find_all(exp.Where):
                for col in where.find_all(exp.Column):
                    col_name = col.name
                    if col_name and col_name not in columns['filterable']:
                        columns['filterable'].append(col_name)

            # GROUP BY columns
            for group in parsed.find_all(exp.Group):
                for expr in group.expressions:
                    col_name = self._get_column_name(expr)
                    if col_name and col_name not in columns['groupable']:
                        columns['groupable'].append(col_name)

            # Aggregated columns
            for agg in parsed.find_all(exp.AggFunc):
                for col in agg.find_all(exp.Column):
                    col_name = col.name
                    if col_name and col_name not in columns['aggregatable']:
                        columns['aggregatable'].append(col_name)

        except Exception as e:
            logger.debug(f"Column extraction error: {e}")

        return columns

    def _get_column_name(self, expr: Any) -> Optional[str]:
        """Get column name from expression"""
        try:
            if isinstance(expr, exp.Column):
                return expr.name
            elif isinstance(expr, exp.Alias):
                return expr.alias
            elif hasattr(expr, 'name'):
                return expr.name
            elif hasattr(expr, 'alias'):
                return expr.alias
        except:
            pass
        return None

    def _extract_joins(self, parsed: Any) -> List[Dict[str, str]]:
        """Extract JOIN conditions"""
        joins = []

        try:
            for join in parsed.find_all(exp.Join):
                on_clause = join.args.get('on')
                if on_clause:
                    # Extract equality conditions
                    for eq in on_clause.find_all(exp.EQ):
                        left = str(eq.left).replace('[', '').replace(']', '')
                        right = str(eq.right).replace('[', '').replace(']', '')
                        joins.append({
                            'left': left,
                            'right': right,
                            'type': join.args.get('kind', 'JOIN')
                        })
        except Exception as e:
            logger.debug(f"Join extraction error: {e}")

        return joins

    def _detect_aggregations(self, parsed: Any) -> List[str]:
        """Detect aggregation functions used"""
        aggs = []

        try:
            for agg in parsed.find_all(exp.AggFunc):
                agg_str = str(agg)
                if agg_str and agg_str not in aggs:
                    aggs.append(agg_str)
        except Exception as e:
            logger.debug(f"Aggregation detection error: {e}")

        return aggs

    def _generate_keywords(
        self,
        source_name: str,
        tables: List[str],
        columns: Dict[str, List[str]],
        fields: Optional[List[Dict]] = None
    ) -> List[str]:
        """Generate keywords for template matching"""
        keywords = set()

        # From source name
        name_parts = re.split(r'[_\-\s:.]+', source_name.lower())
        for part in name_parts:
            if len(part) > 2:
                keywords.add(part)

        # From table names
        for table in tables:
            table_parts = re.split(r'[_.\s]+', table.lower())
            for part in table_parts:
                if len(part) > 2 and part not in ('dbo', 'schema'):
                    keywords.add(part)

        # From columns
        all_cols = []
        for col_list in columns.values():
            all_cols.extend(col_list)

        for col in all_cols:
            col_parts = re.split(r'[_\s]+', col.lower())
            for part in col_parts:
                if len(part) > 2:
                    keywords.add(part)

        # From RDL fields (display names)
        if fields:
            for field in fields:
                field_name = field.get('name', '') or field.get('data_field', '')
                if field_name:
                    # Add the field name
                    keywords.add(field_name.lower())
                    # Also split and add parts
                    for part in re.split(r'[_\s]+', field_name.lower()):
                        if len(part) > 2:
                            keywords.add(part)

        # Add Greek equivalents for common terms
        greek_map = {
            'contract': ['συμβόλαιο', 'σύμβαση'],
            'customer': ['πελάτης'],
            'campaign': ['καμπάνια'],
            'salesman': ['πωλητής'],
            'product': ['προϊόν'],
            'price': ['τιμή', 'αξία'],
            'amount': ['ποσό'],
            'date': ['ημερομηνία'],
            'status': ['κατάσταση'],
        }

        keywords_copy = list(keywords)
        for kw in keywords_copy:
            if kw in greek_map:
                keywords.update(greek_map[kw])

        return sorted(list(keywords))

    def _generate_display_name(self, source_type: str, source_name: str) -> str:
        """Generate human-readable display name"""
        # Remove schema prefix
        name = re.sub(r'^dbo\.', '', source_name)
        # Remove file extension
        name = re.sub(r'\.\w+$', '', name)
        # Split and title case
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # CamelCase
        name = name.replace('_', ' ').replace(':', ' - ')
        return name.strip()


class TemplateIndex:
    """
    Index for fast template lookup by keywords.

    Supports:
    - Keyword matching
    - Fuzzy matching
    - Table-based filtering
    """

    def __init__(self, templates_data: Dict[str, Any]):
        """
        Args:
            templates_data: Output from SQLTemplateExtractor.extract_templates()
        """
        self.templates = {t['id']: t for t in templates_data.get('templates', [])}
        self.keyword_index = templates_data.get('index', {})

    def _normalize_keyword(self, keyword: str) -> List[str]:
        """
        Normalize keyword to handle plurals and common variations.
        Returns list of variants to check.
        """
        kw = keyword.lower().strip()
        variants = [kw]

        # Handle common English plural forms
        if kw.endswith('ies'):
            # companies -> company
            variants.append(kw[:-3] + 'y')
        elif kw.endswith('es'):
            # matches -> match
            variants.append(kw[:-2])
        elif kw.endswith('s') and not kw.endswith('ss'):
            # contracts -> contract
            variants.append(kw[:-1])

        # Also try adding 's' for singular -> plural check
        if not kw.endswith('s'):
            variants.append(kw + 's')

        return variants

    def find_by_keywords(
        self,
        keywords: List[str],
        limit: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find templates matching keywords.

        Enhanced scoring:
        - Direct keyword match: +1.0
        - Singular/plural variant match: +0.9 (almost as good as direct)
        - Partial match (substring): +0.1 (reduced from 0.5 to avoid noise)
        - Keyword in template name: +0.5 (bonus for relevance)
        - Score normalized by number of keywords matched (coverage)

        Args:
            keywords: List of search terms
            limit: Max templates to return

        Returns:
            List of (template, score) tuples sorted by score desc
        """
        template_scores = defaultdict(float)
        template_direct_matches = defaultdict(int)  # Track direct matches separately

        for keyword in keywords:
            kw_lower = keyword.lower()
            kw_variants = self._normalize_keyword(keyword)

            # Direct match - highest priority
            if kw_lower in self.keyword_index:
                for template_id in self.keyword_index[kw_lower]:
                    template_scores[template_id] += 1.0
                    template_direct_matches[template_id] += 1

            # Check for singular/plural variants (nearly as good as direct match)
            for variant in kw_variants:
                if variant != kw_lower and variant in self.keyword_index:
                    for template_id in self.keyword_index[variant]:
                        # Only add if not already counted as direct
                        if template_id not in [tid for tid in self.keyword_index.get(kw_lower, [])]:
                            template_scores[template_id] += 0.9
                            template_direct_matches[template_id] += 1

            # Partial match - track per template to cap contribution
            # This prevents templates with many keywords (like y01...contracts, y02...contracts)
            # from accumulating huge partial match scores
            templates_with_partial = set()
            for indexed_kw, template_ids in self.keyword_index.items():
                # Skip if already counted as direct match or variant
                if indexed_kw == kw_lower or indexed_kw in kw_variants:
                    continue
                # Only count partial if meaningful overlap (at least 4 chars matching)
                if (kw_lower in indexed_kw or indexed_kw in kw_lower) and len(kw_lower) >= 4:
                    for template_id in template_ids:
                        # Only count first partial match per template per keyword
                        if template_id not in templates_with_partial:
                            template_scores[template_id] += 0.1
                            templates_with_partial.add(template_id)

        # Add bonus for templates where keywords appear in the template name
        # This helps "show contracts" prefer templates with "contract" in name
        # Also checks singular/plural variants
        for template_id, template in self.templates.items():
            template_name_lower = template.get('name', '').lower()
            name_words = template_name_lower.split()

            for keyword in keywords:
                kw_variants = self._normalize_keyword(keyword)

                for variant in kw_variants:
                    if variant in template_name_lower:
                        # Full bonus for original keyword, slightly less for variant
                        bonus = 0.5 if variant == keyword.lower() else 0.4
                        template_scores[template_id] += bonus
                        # Extra bonus if keyword is a primary word in name
                        if variant in name_words:
                            template_scores[template_id] += 0.3
                        break  # Only count once per keyword

        # Adjust scores based on direct match coverage
        # Templates with more direct matches should score higher
        for template_id in template_scores:
            direct_count = template_direct_matches.get(template_id, 0)
            if direct_count > 0:
                # Boost based on what % of keywords had direct matches
                coverage = direct_count / max(len(keywords), 1)
                template_scores[template_id] *= (1.0 + coverage * 0.5)

        # Sort by score DESC, then template_id ASC (for deterministic tie-breaking)
        sorted_templates = sorted(
            template_scores.items(),
            key=lambda x: (-x[1], x[0])  # score DESC, template_id ASC
        )[:limit]

        # Build result
        results = []
        for template_id, score in sorted_templates:
            template = self.templates.get(template_id)
            if template:
                # Normalize score - use a reasonable max score
                # With new scoring: max could be ~2.5 per keyword (1.0 direct + 0.5 name + 0.3 word + 0.5 coverage)
                max_possible = len(keywords) * 2.5
                normalized_score = min(1.0, score / max(max_possible, 1))
                results.append((template, normalized_score))

        return results

    def find_by_tables(
        self,
        table_names: List[str],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find templates that use specific tables.

        Args:
            table_names: Tables to search for
            limit: Max templates to return

        Returns:
            List of matching templates
        """
        table_set = {t.lower() for t in table_names}
        results = []

        for template in self.templates.values():
            template_tables = {t.lower() for t in template.get('tables', [])}
            overlap = table_set & template_tables

            if overlap:
                results.append((template, len(overlap)))

        # Sort by overlap count
        results.sort(key=lambda x: -x[1])

        return [t for t, _ in results[:limit]]

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template by ID"""
        return self.templates.get(template_id)


class ColumnIndex:
    """
    Index for fast column lookup by business terms.

    Supports:
    - Direct term matching
    - Fuzzy matching (substring)
    - Finding tables that match multiple terms
    """

    def __init__(self, column_index: Dict[str, List[str]]):
        """
        Args:
            column_index: Dict mapping term -> list of full column names
        """
        self._index = column_index

    def find_columns(self, term: str, limit: int = 10) -> List[str]:
        """
        Find columns matching a business term

        Args:
            term: Business term to search (e.g., "campaign", "amount")
            limit: Max results to return

        Returns:
            List of full column names (schema.table.column)
        """
        term_lower = term.lower()

        # Direct match
        if term_lower in self._index:
            return self._index[term_lower][:limit]

        # Fuzzy match - check if term is substring of any indexed term
        matches = []
        for indexed_term, columns in self._index.items():
            if term_lower in indexed_term or indexed_term in term_lower:
                matches.extend(columns)

        # Deduplicate and limit
        seen = set()
        result = []
        for col in matches:
            if col not in seen:
                seen.add(col)
                result.append(col)
                if len(result) >= limit:
                    break

        return result

    def find_tables_for_terms(self, terms: List[str]) -> List[Tuple[str, int]]:
        """
        Find tables that have columns matching given terms

        Args:
            terms: List of business terms

        Returns:
            List of (table_name, match_count) sorted by match count desc
        """
        table_scores: Dict[str, int] = defaultdict(int)

        for term in terms:
            columns = self.find_columns(term)
            for col in columns:
                # Extract table name from schema.table.column
                parts = col.split('.')
                if len(parts) >= 2:
                    table = '.'.join(parts[:-1])
                    table_scores[table] += 1

        # Sort by score descending
        sorted_tables = sorted(table_scores.items(), key=lambda x: -x[1])
        return sorted_tables


def extract_and_save_templates(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to extract templates and save to file.

    Args:
        output_path: Where to save (defaults to cache/sql_templates.json)

    Returns:
        The extracted templates
    """
    from config.settings import get_path_config

    if output_path is None:
        path_config = get_path_config()
        output_path = path_config.cache_dir / 'sql_templates.json'

    extractor = SQLTemplateExtractor()
    templates = extractor.extract_templates()

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved SQL templates to {output_path}")

    return templates


def load_templates(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load templates from file.

    Args:
        path: Path to templates file (defaults to cache/sql_templates.json)

    Returns:
        Templates dict
    """
    from config.settings import get_path_config

    if path is None:
        path_config = get_path_config()
        path = path_config.cache_dir / 'sql_templates.json'

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
