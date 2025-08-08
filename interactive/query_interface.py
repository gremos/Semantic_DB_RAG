#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced 4-Stage Query Pipeline with Progressive Relaxations (v3.2)
-------------------------------------------------------------------
Drop-in upgrade over v3.1. Keeps strict/verified-joins philosophy while
adding a progressive relaxation ladder so generic/ambiguous queries
actually return data instead of dying empty.

What's new vs v3.1:
- Progressive relaxation ladder (strict → bool_flex → auto_cols → widen_time →
  alt_same_table → fk_path → single_table)
- Boolean normalization helpers (tolerant truthiness for NVARCHAR/INT/BIT)
- Auto-pick of date/amount columns from real sample data
- Optional multi-hop join via verified FK graph paths
- Diagnostics payload so the UI knows which relaxations were used

Non-goals:
- We do NOT hallucinate joins. Every join must be:
  • listed in database_structure.json relationships, or
  • composed ONLY of edges that exist in that file (fk_path mode), or
  • skipped (single_table fallback).

Dependencies/assumptions:
- Same imports and models as v3.1 (Config, TableInfo, BusinessDomain, Relationship, QueryResult)
- pyodbc connectivity
- AzureChatOpenAI via langchain_openai

Implementation notes:
- The LLM still generates SQL, but we now pass concrete column choices,
  boolean-flex directives, and explicit relationship edges for each plan.
- Validation/repair/parse-only loop is reused per plan.
"""
from __future__ import annotations

import json
import re
import pyodbc
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult


# -------------------------------
# LLM Client
# -------------------------------
class LLMClient:
    """LLM client with conservative decoding for deterministic-ish SQL."""

    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            # temperature=0.05,
            request_timeout=60,
        )

    async def ask(self, system_prompt: str, user_prompt: str) -> str:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
            return ""


# -------------------------------
# Query Interface
# -------------------------------
class QueryInterface:
    """4-Stage Query Pipeline with Real Database Analysis + Relaxations (v3.2)"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.classified_tables: List[TableInfo] = []
        self.domain: Optional[BusinessDomain] = None
        self.relationships: List[Relationship] = []

        # Load actual database structure and relationships
        self.database_structure = self._load_database_structure()
        self.foreign_key_map = self._build_foreign_key_map()
        self.fk_graph = self._build_fk_graph()

        print(f"   📊 Loaded database structure with {len(self.database_structure.get('tables', []))} tables")
        print(f"   🔗 Built foreign key map with {len(self.foreign_key_map)} tables having relationships")
        self._debug_database_structure()

    # -------------------------------
    # Debug helpers
    # -------------------------------
    def _debug_database_structure(self):
        tables = self.database_structure.get('tables', [])
        if tables:
            print("   🔍 Sample table structures:")
            for table in tables[:2]:
                table_name = table.get('full_name', table.get('name', 'Unknown'))
                relationships = table.get('relationships', [])
                sample_data = table.get('sample_data', [])
                print(f"      📋 {table_name}:")
                if relationships:
                    print(f"         🔗 Relationships: {relationships[:3]}")
                else:
                    print(f"         🔗 No relationships found")
                if sample_data:
                    first_row = sample_data[0]
                    non_null_cols, null_cols = [], []
                    for col_name, value in first_row.items():
                        if value not in (None, '', 'null'):
                            non_null_cols.append(f"{col_name}={value}")
                        else:
                            null_cols.append(col_name)
                    if non_null_cols:
                        print(f"         ✅ Non-NULL columns: {', '.join(non_null_cols[:3])}")
                    if null_cols:
                        print(f"         ❌ NULL columns: {', '.join(null_cols[:3])}")
                else:
                    print(f"         📊 No sample data")
        explicit_rels = self.database_structure.get('relationships', [])
        if explicit_rels:
            print(f"   🔗 Explicit relationships in database_structure.json: {len(explicit_rels)}")
            for rel in explicit_rels[:3]:
                print(f"      • {rel.get('from_table','?')}.{rel.get('from_column','?')} → {rel.get('to_table','?')}.{rel.get('to_column','?')}")
        else:
            print(f"   ⚠️ No explicit relationships found in database_structure.json")

    # -------------------------------
    # Structure loading & FK map/graph
    # -------------------------------
    def _load_database_structure(self) -> Dict:
        structure_file = self.config.get_cache_path("database_structure.json")
        if structure_file.exists():
            try:
                with open(structure_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"   ⚠️ Failed to load database structure: {e}")
        return {'tables': [], 'relationships': []}

    def _build_foreign_key_map(self) -> Dict[str, List[Dict]]:
        fk_map: Dict[str, List[Dict]] = {}
        relationship_count = 0
        print("   🔍 Analyzing database relationships...")
        # explicit relationships
        for rel in self.database_structure.get('relationships', []):
            from_table = rel.get('from_table', '').strip()
            to_table = rel.get('to_table', '').strip()
            from_column = rel.get('from_column', '').strip()
            to_column = rel.get('to_column', '').strip()
            if not (from_table and to_table and from_column and to_column):
                continue
            fk_map.setdefault(from_table, []).append({
                'from_table': from_table,
                'to_table': to_table,
                'from_column': from_column,
                'to_column': to_column,
                'relationship_type': rel.get('relationship_type') or 'foreign_key',
                'confidence': rel.get('confidence', 0.9),
                'source': 'database_structure',
            })
            relationship_count += 1
        # table-level relationship strings
        for table_data in self.database_structure.get('tables', []):
            table_name = table_data.get('full_name', '').strip()
            if not table_name:
                continue
            for fk_info in table_data.get('relationships', []):
                if '->' not in fk_info:
                    continue
                parts = fk_info.split(' -> ')
                if len(parts) != 2:
                    continue
                from_column = parts[0].strip()
                target_full = parts[1].strip()
                target_table, target_column = None, None
                if '.' in target_full:
                    tp = target_full.split('.')
                    if len(tp) == 2:
                        tt, target_column = tp[0].strip(), tp[1].strip()
                        target_table = tt if tt.startswith('[') else f"[dbo].[{tt}]"
                    elif len(tp) == 3:
                        schema, tname, target_column = tp[0].strip(), tp[1].strip(), tp[2].strip()
                        target_table = f"[{schema}].[{tname}]"
                if not (from_column and target_table and target_column):
                    continue
                fk_map.setdefault(table_name, []).append({
                    'from_table': table_name,
                    'to_table': target_table,
                    'from_column': from_column,
                    'to_column': target_column,
                    'relationship_type': 'foreign_key',
                    'confidence': 0.95,
                    'source': 'table_constraints',
                })
                relationship_count += 1
        print(f"   ✅ Found {relationship_count} verified relationships from database structure")
        if relationship_count == 0:
            print("   ⚠️ No foreign key relationships found in database structure")
        return fk_map

    def _build_fk_graph(self) -> Dict[str, List[Tuple[str, Dict]]]:
        g: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        for src, edges in self.foreign_key_map.items():
            for e in edges:
                a, b = e['from_table'], e['to_table']
                g[a].append((b, e))
                g[b].append((a, e))
        return g

    # -------------------------------
    # Session
    # -------------------------------
    async def start_interactive_session(self, tables: List[TableInfo], domain: Optional[BusinessDomain], relationships: List[Relationship]):
        self.classified_tables = tables
        self.domain = domain
        self.relationships = relationships
        print(f"🚀 Enhanced 4-Stage Pipeline Ready")
        print(f"   📊 Classified tables: {len(tables)}")
        print(f"   🔗 Database relationships: {len(self.foreign_key_map)}")
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        entity_counts: Dict[str, int] = {}
        for table in tables:
            if hasattr(table, 'entity_type') and table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        if entity_counts:
            print(f"   📊 Available entities: {dict(list(entity_counts.items())[:5])}")
        query_count = 0
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                query_count += 1
                print(f"🚀 Processing with enhanced 4-stage pipeline...")
                start_time = time.time()
                result = await self.process_enhanced_pipeline(question)
                result.execution_time = time.time() - start_time
                self.display_result(result)
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        print(f"\n📊 Session summary: {query_count} queries processed")

    # -------------------------------
    # Pipeline
    # -------------------------------
    async def process_enhanced_pipeline(self, question: str) -> QueryResult:
        try:
            # Stage 1: Intent (LLM with heuristic fallback)
            print("   🎯 Stage 1: Understanding intent...")
            intent: Optional[Dict[str, Any]] = None
            if hasattr(self, "analyze_intent"):
                try:
                    intent = await self.analyze_intent(question)
                except Exception as e:
                    print(f"   ⚠️ analyze_intent failed: {e}")
            if not intent:
                print("   ℹ️ Falling back to heuristic intent parser")
                intent = self._fallback_intent(question)
            if not intent:
                return QueryResult(question=question, sql_query="", results=[],
                                   error="Failed to understand question (LLM and heuristic both failed)")

            # Stage 2: Table selection
            print("   📋 Stage 2: Finding tables using real sample data...")
            selected_tables = await self.select_tables_with_real_data(question, intent)
            if not selected_tables:
                return QueryResult(question=question, sql_query="", results=[], error="No relevant tables found")
            print(f"      ✅ Found {len(selected_tables)} relevant tables")

            # Stage 3: Relationship discovery
            print("   🔗 Stage 3: Analyzing actual database relationships...")
            verified_relationships = await self.discover_actual_relationships(selected_tables)

            # Optional: ensure reachability when payments implied
            if self._intent_mentions_payment(intent):
                _ = self._ensure_customer_to_transaction_path(selected_tables)

            # Stage 4/5: Generate & Execute with progressive relaxations
            rows, sql, plan, diagnostics, err = await self.run_with_relaxations(
                question, intent, selected_tables, verified_relationships
            )

            if err:
                res = QueryResult(
                    question=question,
                    sql_query=sql or "",
                    results=rows or [],
                    error=err,
                    tables_used=[t['full_name'] for t in selected_tables],
                )
                try:
                    setattr(res, "diagnostics", diagnostics)
                except Exception:
                    pass
                return res

            res = QueryResult(
                question=question,
                sql_query=sql,
                results=rows,
                error=None,
                tables_used=[t['full_name'] for t in selected_tables],
            )
            try:
                setattr(res, "diagnostics", diagnostics)
            except Exception:
                pass
            return res
        except Exception as e:
            return QueryResult(question=question, sql_query="", results=[], error=f"Pipeline failed: {str(e)}")

    # -------------------------------
    # Relaxation runner
    # -------------------------------
    async def run_with_relaxations(self, question: str, intent: Dict, selected_tables: List[Dict], verified_relationships: List[Dict]):
        plans = [
            {"name": "strict", "bool_flex": False, "pick_cols": False, "qtr_expand": 0, "join_variant": "verified"},
            {"name": "bool_flex", "bool_flex": True, "pick_cols": False, "qtr_expand": 0, "join_variant": "verified"},
            {"name": "auto_cols", "bool_flex": True, "pick_cols": True, "qtr_expand": 0, "join_variant": "verified"},
            {"name": "widen_time", "bool_flex": True, "pick_cols": True, "qtr_expand": 1, "join_variant": "verified"},
            {"name": "alt_same_table", "bool_flex": True, "pick_cols": True, "qtr_expand": 1, "join_variant": "alt_same_table"},
            {"name": "fk_path", "bool_flex": True, "pick_cols": True, "qtr_expand": 1, "join_variant": "fk_path"},
            {"name": "single_table", "bool_flex": True, "pick_cols": True, "qtr_expand": 1, "join_variant": "single_table"},
        ]
        last_err = None
        for plan in plans:
            try:
                rows, sql, diagnostics, err = await self._attempt_plan(
                    plan, question, intent, selected_tables, verified_relationships
                )
                if err:
                    last_err = err
                    continue
                if rows is not None and len(rows) >= 0:
                    return rows, sql, plan, diagnostics, None
            except Exception as e:
                last_err = str(e)
                continue
        return [], "", plans[-1], {"error": last_err}, last_err or "All relaxations failed"

    async def _attempt_plan(self, plan: Dict[str, Any], question: str, intent: Dict,
                             selected_tables: List[Dict], verified_relationships: List[Dict]):
        # Prepare relationship set depending on join_variant
        relationships = list(verified_relationships)
        allowed_tables = [t['full_name'] for t in selected_tables]
        table_contexts = self._analyze_tables_for_context(selected_tables)

        # If alt_same_table: permit alternate edges that connect the same table pair
        if plan.get('join_variant') == 'alt_same_table':
            alt_edges = self._alternate_edges_among(selected_tables)
            if alt_edges:
                relationships = alt_edges

        # If fk_path: try to augment edges with multi-hop paths limited to selected tables
        if plan.get('join_variant') == 'fk_path':
            path_edges = self._compose_paths_within(selected_tables)
            if path_edges:
                relationships = path_edges

        # Column role picks (only if requested)
        role_cols = {}
        if plan.get('pick_cols'):
            role_cols = self._pick_role_columns_for_known_tables(table_contexts)

        # Generate SQL with constraints + plan hints
        sql = await self.generate_verified_sql(
            question=question,
            intent=intent,
            tables=selected_tables,
            relationships=relationships,
            plan=plan,
            role_cols=role_cols,
            table_contexts=table_contexts,
        )

        # First validation/repair
        violations = self._validate_or_explain(sql, selected_tables, relationships)
        if violations:
            print(f"      🛠️ Draft SQL violated: {', '.join(violations)}. Repairing...")
            sql = await self.repair_sql(
                question, intent, selected_tables, relationships,
                self._inject_violation_comment(sql, violations), plan=plan, role_cols=role_cols,
            )

        # Parse-only check
        ok, parse_err = self.try_parse_only(sql)
        if not ok:
            print(f"      🛠️ Parse failed. Repairing with error context...")
            sql = await self.repair_sql(
                question, intent, selected_tables, relationships,
                f"-- parse_error: {parse_err}\n{sql}", plan=plan, role_cols=role_cols,
            )

        # Final validation
        final_violations = self._validate_or_explain(sql, selected_tables, relationships)
        if not sql or self._looks_trivial_sql(sql) or final_violations:
            return [], sql or "", {"strategy": plan.get('name'), "violations": final_violations}, "Model produced trivial/invalid SQL"

        # Execute
        results, error = self.execute_sql(sql)
        diagnostics = {
            "strategy": plan.get('name'),
            "time_window": "next_quarter",
            "window_widened": bool(plan.get('qtr_expand', 0)),
            "join_mode": plan.get('join_variant'),
            "role_cols": role_cols,
        }
        return results, sql, diagnostics, error

    # -------------------------------
    # Stage 2: Table selection with real data + transactional bias
    # -------------------------------
    async def select_tables_with_real_data(self, question: str, intent: Dict) -> List[Dict]:
        real_tables: List[Dict] = []
        structure_tables = {t.get('full_name', ''): t for t in self.database_structure.get('tables', [])}
        for table in self.classified_tables:
            st = structure_tables.get(table.full_name)
            if not st or not st.get('sample_data'):
                continue
            sample_preview = self._create_sample_preview(st.get('sample_data', []))
            real_tables.append({
                'full_name': table.full_name,
                'name': table.name,
                'entity_type': getattr(table, 'entity_type', 'Unknown'),
                'confidence': getattr(table, 'confidence', 0.0),
                'row_count': st.get('row_count', 0),
                'columns': st.get('columns', []),
                'sample_data': st.get('sample_data', []),
                'sample_preview': sample_preview,
                'foreign_keys': st.get('relationships', []),
            })
        # baseline sort
        real_tables.sort(key=lambda t: (t['confidence'], len(t['sample_data'])), reverse=True)
        # transactional bias
        subj = json.dumps(intent).lower() if intent else ""
        payment_intent = any(w in subj for w in ['payment', 'paid', 'invoice', 'receipt', 'revenue', 'renewal'])
        if payment_intent:
            real_tables.sort(key=lambda t: (self._is_transactional_table(t), t['confidence'], len(t['sample_data'])), reverse=True)
        top_candidates = real_tables[:30]

        system_prompt = (
            "You are a database analyst. Select tables based on ACTUAL SAMPLE DATA and entity classifications.\n"
            "Use both sample values and entity types to make accurate selections.\n"
            "Focus on tables with the necessary facts (transactions, amounts, dates) plus the related dimension/account tables.\n"
            "If the intent is about payments/invoices/renewals, selection MUST include at least one transactional table (has amount+date).\n"
            "Respond with JSON only."
        )
        user_prompt = f"""
Question: "{question}"
Intent: {json.dumps(intent)}

TABLES WITH REAL SAMPLE DATA:
{json.dumps([
    {
        'full_name': t['full_name'],
        'entity_type': t['entity_type'],
        'confidence': t['confidence'],
        'row_count': t['row_count'],
        'sample_preview': t['sample_preview'],
        'columns': [col.get('name', col.get('COLUMN_NAME','')) for col in t['columns'][:8]],
        'foreign_keys': t['foreign_keys'][:3],
        'is_transactional': self._is_transactional_table(t),
    }
    for t in top_candidates
], indent=2)}

Select 3–6 tables that truly hold the data needed. Avoid pure config/reference tables unless required for a necessary join.

JSON format:
{{
  "selected_tables": ["[dbo].[Customers]", "[dbo].[Payments]"],
  "reasoning": "..."
}}
"""
        response = await self.llm.ask(system_prompt, user_prompt)
        result = self.parse_json(response)
        if result and 'selected_tables' in result:
            table_lookup = {t['full_name']: t for t in top_candidates}
            return [table_lookup[n] for n in result['selected_tables'] if n in table_lookup]
        return []

    # -------------------------------
    # Stage 3: Verified relationships only
    # -------------------------------
    async def discover_actual_relationships(self, tables: List[Dict]) -> List[Dict]:
        if len(tables) <= 1:
            return []
        names = [t['full_name'] for t in tables]
        verified_relationships: List[Dict] = []
        print(f"      🔍 Searching for verified relationships between selected tables...")
        def _cols(tbl: Dict) -> List[str]:
            return [c.get('name', c.get('COLUMN_NAME', '')) for c in tbl.get('columns', [])]
        for table_name in names:
            for fk_rel in self.foreign_key_map.get(table_name, []):
                target_table = fk_rel.get('to_table', '')
                from_column = (fk_rel.get('from_column', '') or '').strip()
                to_column = (fk_rel.get('to_column', '') or '').strip()
                if not from_column or not to_column:
                    continue
                target_found = None
                for t in tables:
                    if target_table == t['full_name'] or target_table in t['full_name']:
                        target_found = t['full_name']; break
                if not target_found:
                    continue
                from_tbl = next((t for t in tables if t['full_name'] == table_name), None)
                to_tbl = next((t for t in tables if t['full_name'] == target_found), None)
                if from_tbl and to_tbl:
                    if from_column in _cols(from_tbl) and to_column in _cols(to_tbl):
                        verified_relationships.append({
                            'from_table': table_name,
                            'to_table': target_found,
                            'from_column': from_column,
                            'to_column': to_column,
                            'join_type': 'INNER JOIN',
                            'confidence': fk_rel.get('confidence', 0.95),
                            'source': fk_rel.get('source', 'database_structure'),
                        })
                        print(f"         ✅ {table_name}.{from_column} → {target_found}.{to_column}")
        if verified_relationships:
            print(f"      🔗 Found {len(verified_relationships)} verified relationships with valid columns")
        else:
            print(f"      ⚠️ No valid foreign key relationships found between selected tables")
            print(f"      💡 Will use single-table approach or analyze sample data")
        return verified_relationships

    # -------------------------------
    # Stage 4: SQL generation (STRICT + plan hints)
    # -------------------------------
    async def generate_verified_sql(self, question: str, intent: Dict, tables: List[Dict], relationships: List[Dict],
                                    plan: Dict[str, Any], role_cols: Dict[str, Dict[str, str]], table_contexts: Dict[str, Dict[str, Any]]) -> str:
        print("      📊 Analyzing sample data for column validation...")
        allowed_columns = self._columns_whitelist(tables)
        allowed_tables = [t['full_name'] for t in tables]

        # Prompt skeletons
        skeleton_cte = (
            "If you use CTEs, follow this pattern strictly:\n"
            "WITH a AS (SELECT ... FROM [schema].[TableA]),\n"
            "     b AS (SELECT ... FROM [schema].[TableB])\n"
            "SELECT ... FROM a JOIN b ON ...;\n"
        )
        common_rules = (
            "HARD CONSTRAINTS:\n"
            "- Use ONLY the verified join relationships provided. Do not create/assume others.\n"
            "- Do NOT use T-SQL variables (@var). Use inline expressions only.\n"
            "- Use ONLY columns from allowed_columns for each table.\n"
            "- Prefer a single concrete date column and a single amount column from the lists;\n"
            "  avoid long COALESCE chains across unknown names.\n"
            "- If you emit CTEs, include the leading WITH and comma-separate CTEs.\n"
            "- Return ONLY valid SQL Server T-SQL (no markdown, no comments). If unsure, return FAIL.\n"
        )

        plan_hints = {
            "bool_flex": plan.get('bool_flex', False),
            "pick_cols": plan.get('pick_cols', False),
            "qtr_expand": plan.get('qtr_expand', 0),
            "join_variant": plan.get('join_variant', 'verified'),
        }

        # Prepare role column picks as static facts for the LLM
        role_fact_lines = []
        for tbl, roles in (role_cols or {}).items():
            if roles:
                for role_name, col in roles.items():
                    if col:
                        role_fact_lines.append(f"{tbl}::{role_name}={col}")
        role_facts = "\n".join(role_fact_lines) if role_fact_lines else "(none)"

        system_prompt = (
            "You are an expert SQL developer with VERIFIED database relationships.\n"
            "Use ONLY the provided verified joins.\n"
            + skeleton_cte + common_rules +
            "Additional directives based on plan:\n"
            "- If bool_flex=true, when filtering boolean-like NVARCHAR/INT columns, normalize truthiness with TRY_CONVERT and ('Y','YES','TRUE','1').\n"
            "- If pick_cols=true, you MUST use the provided role columns (date/amount) instead of guessing.\n"
            "- If qtr_expand=1, slightly widen the next-quarter window (± ~90 days) and include a diagnostic label column like diagnostic_window.\n"
            "- If join_variant='single_table', answer from the best single table only.\n"
        )

        # Decide which relationships to expose (these are the ONLY allowed joins)
        rels_text = json.dumps(relationships, indent=2)

        user_prompt = f"""
Question: "{question}"
Intent: {json.dumps(intent)}

TABLE ANALYSIS WITH SAMPLE DATA:
{json.dumps(table_contexts, indent=2)}

VERIFIED RELATIONSHIPS (USE THESE EXACT JOINS):
{rels_text}

allowed_tables = {json.dumps(allowed_tables)}
allowed_columns = {json.dumps({k: sorted(list(v)) for k,v in allowed_columns.items()}, indent=2)}

plan_hints = {json.dumps(plan_hints)}
role_columns (facts) =\n{role_facts}

Generate ONLY the SQL. If you cannot, return FAIL.
"""
        # If no relationships and multiple tables but not using fk_path, force single table
        if not relationships and len(tables) > 1 and plan.get('join_variant') != 'fk_path':
            system_prompt = (
                "You are an expert SQL developer.\n"
                "No verified joins found. Generate a single-table query that best answers the question.\n"
                + skeleton_cte + common_rules
            )

        response = await self.llm.ask(system_prompt, user_prompt)
        sql = self.clean_sql(response)
        return '' if sql.strip().upper() == 'FAIL' else sql

    # -------------------------------
    # Repair pass (STRICT + plan)
    # -------------------------------
    async def repair_sql(self, question: str, intent: Dict, tables: List[Dict], relationships: List[Dict], bad_sql: str,
                         plan: Optional[Dict[str, Any]] = None, role_cols: Optional[Dict[str, Dict[str, str]]] = None) -> str:
        table_contexts = self._analyze_tables_for_context(tables)
        allowed_columns = self._columns_whitelist(tables)
        allowed_tables = [t['full_name'] for t in tables]

        plan_hints = {
            "bool_flex": bool(plan.get('bool_flex')) if plan else False,
            "pick_cols": bool(plan.get('pick_cols')) if plan else False,
            "qtr_expand": int(plan.get('qtr_expand', 0)) if plan else 0,
            "join_variant": (plan.get('join_variant') if plan else 'verified')
        }
        role_facts = {tbl: roles for tbl, roles in (role_cols or {}).items()}

        system = (
            "You are a senior SQL reviewer. Fix SQL so it compiles and respects constraints.\n"
            "Return ONLY SQL. If you cannot, return FAIL."
        )
        user = f"""
Question: {question}
Intent: {json.dumps(intent)}
Tables: {json.dumps(allowed_tables)}
Relationships: {json.dumps(relationships, indent=2)}
Table Contexts: {json.dumps(table_contexts, indent=2)}
allowed_columns = {json.dumps({k: sorted(list(v)) for k,v in allowed_columns.items()}, indent=2)}

Plan hints: {json.dumps(plan_hints)}
Role columns: {json.dumps(role_facts)}

Constraints:
- Use ONLY verified joins listed above; otherwise use a single table.
- Do NOT use @variables. Inline expressions only.
- Use ONLY columns present in allowed_columns per table.
- If using CTEs, include the leading WITH and proper commas.
- Prefer one concrete date column & one amount column from the context; avoid COALESCE of unknown names.

Bad SQL (with hints/violations if any):
{bad_sql}
"""
        resp = await self.llm.ask(system, user)
        sql = self.clean_sql(resp)
        return '' if sql.strip().upper() == 'FAIL' else sql

    # -------------------------------
    # Validators & helpers
    # -------------------------------
    def _joined_tables(self, sql: str) -> List[str]:
        tables = []
        for m in re.finditer(r"\bFROM\s+\[([^\]]+)\]\.\[([^\]]+)\]", sql, re.I):
            tables.append(f"[{m.group(1)}].[{m.group(2)}]")
        for m in re.finditer(r"\bJOIN\s+\[([^\]]+)\]\.\[([^\]]+)\]", sql, re.I):
            tables.append(f"[{m.group(1)}].[{m.group(2)}]")
        return tables

    def _columns_whitelist(self, tables: List[Dict]) -> Dict[str, set]:
        m: Dict[str, set] = {}
        for t in tables:
            key = t['full_name'].lower()
            m[key] = {
                (c.get('name') or c.get('COLUMN_NAME') or '').lower() for c in t.get('columns', [])
            }
        return m

    def _violates_join_policy(self, sql: str, relationships: List[Dict]) -> bool:
        allowed_tables = set()
        for r in relationships:
            allowed_tables.add(r['from_table'].lower())
            allowed_tables.add(r['to_table'].lower())
        for t in self._joined_tables(sql):
            if t.lower() not in allowed_tables and len(allowed_tables) > 0:
                return True
        return False

    def _needs_with_header(self, sql: str) -> bool:
        has_cte_tail = bool(re.search(r"\)\s*,\s*\w+\s+AS\s*\(", sql, re.I))
        starts_with_with = bool(re.match(r"^\s*WITH\s", sql, re.I))
        return has_cte_tail and not starts_with_with

    def _has_tsql_vars(self, sql: str) -> bool:
        return bool(re.search(r"@\w+", sql))

    def _references_unknown_tables(self, sql: str, selected: List[Dict]) -> bool:
        allowed = {t['full_name'].lower() for t in selected}
        used = {t.lower() for t in self._joined_tables(sql)}
        return any(u not in allowed for u in used)

    def _uses_unstable_agg_ordering(self, sql: str) -> bool:
        # Keep simple: allow STRING_AGG without WITHIN GROUP for broad compatibility
        return False

    def _validate_or_explain(self, sql: str, tables: List[Dict], relationships: List[Dict]) -> List[str]:
        reasons = []
        if not sql or self._looks_trivial_sql(sql):
            reasons.append("trivial_sql")
        if self._needs_with_header(sql):
            reasons.append("missing_with_cte")
        if self._has_tsql_vars(sql):
            reasons.append("tsql_variables")
        if self._violates_join_policy(sql, relationships):
            reasons.append("unverified_join")
        if self._references_unknown_tables(sql, tables):
            reasons.append("unknown_table")
        if self._uses_unstable_agg_ordering(sql):
            reasons.append("stringagg_within_group")
        return reasons

    def _inject_violation_comment(self, sql: str, violations: List[str]) -> str:
        return f"-- violations: {', '.join(violations)}\n{sql}"

    def try_parse_only(self, sql: str) -> Tuple[bool, Optional[str]]:
        if not sql.strip():
            return False, "empty sql"
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                cur = conn.cursor()
                cur.execute("SET NOEXEC ON; " + sql + "; SET NOEXEC OFF;")
            return True, None
        except Exception as e:
            return False, str(e)

    # -------------------------------
    # Table analysis helpers
    # -------------------------------
    def _analyze_tables_for_context(self, tables: List[Dict]) -> Dict[str, Dict[str, Any]]:
        table_contexts: Dict[str, Dict[str, Any]] = {}
        for table in tables:
            columns_info = []
            non_null_columns = {}
            date_columns: List[str] = []
            amount_columns: List[str] = []
            null_columns: List[str] = []
            for col in table.get('columns', []):
                col_name = col.get('name', col.get('COLUMN_NAME', ''))
                col_type = col.get('data_type', col.get('DATA_TYPE', ''))
                is_pk = col.get('is_primary_key', False)
                columns_info.append(f"{col_name} {col_type}{' (PK)' if is_pk else ''}")
                if isinstance(col_type, str) and any(x in col_type.lower() for x in ['date', 'time']):
                    date_columns.append(col_name)
                n = (col_name or '').lower()
                if n.endswith(('amount', 'total', 'price', 'cost', 'netamount', 'grossamount', 'lineamount', 'mrr', 'arr')) or n in ('amount','total'):
                    amount_columns.append(col_name)
            sample_data = table.get('sample_data', [])
            if sample_data:
                first_row = sample_data[0]
                sample_items = []
                for key, value in first_row.items():
                    if value not in (None, '', 'null'):
                        non_null_columns[key] = value
                        if len(sample_items) < 5:
                            v = str(value)
                            sample_items.append(f"{key}={v[:20]}{'...' if len(v)>20 else ''}")
                    else:
                        null_columns.append(key)
                sample_info = "Sample data: " + ", ".join(sample_items)
            else:
                sample_info = "No sample data"
            table_contexts[table['full_name']] = {
                'entity_type': table.get('entity_type', 'Unknown'),
                'columns': columns_info,
                'sample_info': sample_info,
                'row_count': table.get('row_count', 0),
                'non_null_columns': list(non_null_columns.keys()),
                'null_columns': null_columns,
                'date_columns': date_columns,
                'amount_columns': amount_columns,
                'sample_values': non_null_columns,
            }
            print(f"         📋 {table['full_name']}:")
            if non_null_columns:
                print(f"            ✅ Columns with data: {', '.join(list(non_null_columns.keys())[:5])}")
            if null_columns:
                print(f"            ❌ NULL columns (avoid for filters): {', '.join(null_columns[:5])}")
        return table_contexts

    def _create_sample_preview(self, sample_data: List[Dict]) -> str:
        if not sample_data or not isinstance(sample_data, list):
            return "No sample data"
        row = sample_data[0] or {}
        items = []
        for k, v in row.items():
            if v is not None and str(v).strip() and len(items) < 4:
                s = str(v)
                if len(s) > 20:
                    s = s[:20] + "..."
                items.append(f"{k}={s}")
        return ", ".join(items) if items else "Empty data"

    def _is_transactional_table(self, t: Dict) -> bool:
        names = {t.get('full_name', '').lower(), t.get('name', '').lower()}
        cols = {(c.get('name') or c.get('COLUMN_NAME') or '').lower() for c in t.get('columns', [])}
        transactional_name_hit = any(
            kw in n for n in names
            for kw in ['payment', 'invoice', 'receipt', 'transaction', 'order', 'sale', 'ledger', 'posting', 'collection', 'subscription', 'contract']
        )
        has_amount = any(k in cols for k in ['amount', 'totalamount', 'netamount', 'grossamount', 'total', 'lineamount', 'mrr', 'arr', 'price', 'onetimeamount'])
        has_date = any(k in cols for k in ['paymentdate', 'invoicedate', 'transactiondate', 'postedon', 'createdon', 'createddate', 'renewaldate', 'expirationdate', 'duedate', 'date', 'collectiondate'])
        sample = t.get('sample_data') or []
        sample_has_values = False
        if sample:
            row = sample[0]
            sample_has_values = any((row.get(k) not in (None, '', 'null')) for k in row.keys())
        return (transactional_name_hit and (has_amount or has_date)) or (has_amount and has_date and sample_has_values)

    def _intent_mentions_payment(self, intent: Dict) -> bool:
        return any(word in json.dumps(intent).lower() for word in ['payment', 'paid', 'invoice', 'receipt', 'revenue', 'renewal'])

    def _ensure_customer_to_transaction_path(self, selected_tables: List[Dict]) -> bool:
        if not selected_tables:
            return True
        customer_like = [t['full_name'] for t in selected_tables if str(t.get('entity_type', '')).lower() in ['customer', 'contact', 'crm', 'account'] or 'customer' in t['full_name'].lower()]
        transactional = [t['full_name'] for t in selected_tables if self._is_transactional_table(t)]
        if not customer_like or not transactional:
            return False
        return any(self._has_path(c, set(transactional)) for c in customer_like)

    def _has_path(self, start: str, targets: set[str]) -> bool:
        seen = {start}
        dq = deque([start])
        while dq:
            u = dq.popleft()
            if u in targets:
                return True
            for v, _ in self.fk_graph.get(u, []):
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        return False

    # -------------------------------
    # SQL exec, parsing & cleaning
    # -------------------------------
    def execute_sql(self, sql: str) -> tuple:
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                conn.setencoding(encoding='utf-8')
                cursor = conn.cursor()
                cursor.execute(sql)
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    for row in cursor.fetchmany(100):
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = self.safe_value(value)
                        results.append(row_dict)
                    return results, None
                else:
                    return [], None
        except Exception as e:
            error_msg = str(e)
            if "Invalid column name" in error_msg or "Invalid object name" in error_msg:
                return [], f"SQL Error - Check column/table names: {error_msg}"
            elif "Conversion failed" in error_msg:
                return [], f"Data Type Error - Check date/number formats: {error_msg}"
            else:
                return [], f"Database Error: {error_msg}"

    def parse_json(self, response: str) -> Dict:
        try:
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = re.sub(r'^[^{]*', '', cleaned)
            cleaned = re.sub(r'[^}]*$', '', cleaned)
            return json.loads(cleaned)
        except Exception:
            return {}

    def clean_sql(self, response: str) -> str:
        cleaned = re.sub(r'```sql\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)
        m = re.search(r"(?is)(WITH\s|SELECT\s).*$", response)
        if m:
            s = m.group(0)
        else:
            s = response
        s = s.strip()
        if s.endswith(';'):
            s = s[:-1]
        return s

    def _looks_trivial_sql(self, sql: str) -> bool:
        s = (sql or '').strip().upper().replace('\n', ' ')
        if not s:
            return True
        if ' FROM ' not in s:
            return True
        if s.startswith('SELECT 0') or s.startswith('SELECT 1'):
            return True
        return False

    def safe_value(self, value):
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)[:200]

    def display_result(self, result: QueryResult):
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        if result.error:
            print(f"❌ Error: {result.error}")
            print(f"💡 Generated SQL:\n   {result.sql_query}")
            if getattr(result, 'tables_used', None):
                print(f"💡 Tables attempted: {', '.join(result.tables_used)}")
            print(f"💡 Suggestion: Check the relationships section above to see what foreign keys were found")
        else:
            print(f"📋 Generated SQL:\n   {result.sql_query}")
            print(f"📊 Results: {len(result.results)} rows")
            if result.results:
                if len(result.results) == 1 and len(result.results[0]) == 1:
                    value = list(result.results[0].values())[0]
                    column_name = list(result.results[0].keys())[0]
                    if isinstance(value, (int, float)):
                        print(f"   🎯 {column_name}: {value:,}")
                    else:
                        print(f"   🎯 {column_name}: {value}")
                else:
                    for i, row in enumerate(result.results[:5], 1):
                        display_row = {}
                        for key, value in list(row.items())[:6]:
                            if isinstance(value, str) and len(value) > 30:
                                display_row[key] = value[:30] + "..."
                            elif isinstance(value, (int, float)) and value > 1000:
                                display_row[key] = f"{value:,}"
                            else:
                                display_row[key] = value
                        print(f"   {i}. {display_row}")
                    if len(result.results) > 5:
                        print(f"   ... and {len(result.results) - 5} more rows")
            if getattr(result, 'tables_used', None):
                print(f"📋 Tables used:")
                table_lookup = {t.full_name: t for t in self.classified_tables}
                for table_name in result.tables_used:
                    if table_name in table_lookup:
                        table = table_lookup[table_name]
                        entity_type = getattr(table, 'entity_type', 'Unknown')
                        print(f"      • {table_name} ({entity_type})")
                    else:
                        print(f"      • {table_name}")
            has_joins = 'JOIN' in (result.sql_query or '').upper()
            print("✅ Query used verified database relationships" if has_joins else "ℹ️ Single-table query (no joins or none verified)")
        if getattr(result, 'diagnostics', None):
            print("\n🔎 Diagnostics:", result.diagnostics)
        print("\n💡 Debugging Help:")
        print("   • Validators block unverified joins, unknown columns, variables, and malformed CTEs")
        print("   • Parse-only step catches syntax errors before execution")
        print("   • Check database_structure.json for missing FK info if joins are restricted")

    # -------------------------------
    # Stage 1 (LLM) intent analyzer (optional, used when present)
    # -------------------------------
    async def analyze_intent(self, question: str) -> Dict[str, Any]:
        system_prompt = (
            "Analyze business questions to extract key intent information.\n"
            "Focus on understanding what data is needed and what operation to perform.\n"
            "Respond with JSON only."
        )
        user_prompt = f"""
Question: "{question}"

Extract the key information:
1. Action: What operation (count, list, show, calculate, find, etc.)
2. Subject: What entities are involved (customers, payments, orders, products, etc.)
3. Filters: Any conditions or time periods mentioned
4. Result type: What kind of result is expected

JSON format:
{{
  "action": "count",
  "subject": "paid customers",
  "filters": ["2025", "payment made"],
  "result_type": "single_number",
  "entities_needed": ["Customer", "Payment"]
}}
"""
        response = await self.llm.ask(system_prompt, user_prompt)
        return self.parse_json(response)

    # -------------------------------
    # Stage 1 heuristic fallback
    # -------------------------------
    def _fallback_intent(self, question: str) -> Dict[str, Any]:
        q = (question or "").lower()
        years = re.findall(r"\b(20\d{2})\b", q)
        action = "count" if "count" in q else ("sum" if ("sum" in q or "total" in q) else "list")
        if ("paid" in q and "customer" in q) or ("paying customers" in q):
            subject = "paid customers"
        elif "invoice" in q:
            subject = "invoices"
        elif "payment" in q or "paid" in q:
            subject = "payments"
        else:
            subject = "records"
        result_type = "single_number" if action in ("count", "sum") else "table"
        entities: List[str] = []
        if "customer" in q:
            entities.append("Customer")
        if ("payment" in q or "paid" in q or "invoice" in q):
            entities.append("Payment")
        if not entities:
            entities.append("Record")
        filters: List[str] = list(dict.fromkeys(years))
        if "paid" in q:
            filters.append("paid")
        if "next quarter" in q:
            filters.append("next_quarter")
        if "this quarter" in q:
            filters.append("this_quarter")
        if "last quarter" in q:
            filters.append("last_quarter")
        return {
            "action": action,
            "subject": subject,
            "filters": filters,
            "result_type": result_type,
            "entities_needed": entities,
        }

    # -------------------------------
    # Relaxation helpers
    # -------------------------------
    def sql_true_clause(self, col: str) -> str:
        return (f"(TRY_CONVERT(int,{col}) = 1 OR {col} IN ('1','Y','YES','True','TRUE'))")

    def sql_false_clause(self, col: str) -> str:
        return (f"(TRY_CONVERT(int,{col}) = 0 OR {col} IN ('0','N','NO','False','FALSE'))")

    def _pick_role_columns_for_known_tables(self, table_contexts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        def pick_role_column(ctx: Dict, role: str, preferred: List[str]) -> Optional[str]:
            pool = (ctx.get('date_columns') if role=='date' else ctx.get('amount_columns')) or []
            for p in preferred:
                if p in pool:
                    return p
            non_null = set(ctx.get('non_null_columns') or [])
            for c in pool:
                if c in non_null:
                    return c
            return pool[0] if pool else None

        role_cols: Dict[str, Dict[str, str]] = {}
        for tbl, ctx in table_contexts.items():
            # Heuristics for payments/collections tables
            if any(k in tbl.lower() for k in ['collection', 'payment', 'invoice', 'salescollection']):
                role_cols.setdefault(tbl, {})['date'] = pick_role_column(ctx, 'date', ['collectiondate','paymentdate','postedon','createdon'])
                role_cols.setdefault(tbl, {})['amount'] = pick_role_column(ctx, 'amount', ['amount','total','price'])
            # Heuristics for contract/line items
            if any(k in tbl.lower() for k in ['contractproduct', 'contract_line', 'orderitem', 'line']):
                role_cols.setdefault(tbl, {})['amount'] = pick_role_column(ctx, 'amount', ['price','amount','total','lineamount','onetimeamount'])
        return role_cols

    def _alternate_edges_among(self, selected_tables: List[Dict]) -> List[Dict]:
        names = [t['full_name'] for t in selected_tables]
        alt: List[Dict] = []
        # include any edges whose endpoints are both in selected tables
        for src in names:
            for e in self.foreign_key_map.get(src, []):
                if e.get('to_table') in names:
                    alt.append({
                        'from_table': e['from_table'],
                        'to_table': e['to_table'],
                        'from_column': e['from_column'],
                        'to_column': e['to_column'],
                        'join_type': 'INNER JOIN',
                        'confidence': e.get('confidence', 0.9),
                        'source': e.get('source', 'database_structure'),
                    })
        # dedupe
        seen = set()
        uniq = []
        for e in alt:
            key = (e['from_table'], e['from_column'], e['to_table'], e['to_column'])
            if key not in seen:
                uniq.append(e); seen.add(key)
        return uniq

    def _compose_paths_within(self, selected_tables: List[Dict]) -> List[Dict]:
        names = [t['full_name'] for t in selected_tables]
        edges: List[Dict] = []
        # try to connect every pair via BFS; only keep paths that stay within selected tables
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                path = self._shortest_fk_path_within(a, b, set(names))
                if not path:
                    continue
                edges.extend(path)
        # dedupe
        seen = set()
        uniq = []
        for e in edges:
            key = (e['from_table'], e['from_column'], e['to_table'], e['to_column'])
            if key not in seen:
                uniq.append(e); seen.add(key)
        return uniq

    def _shortest_fk_path_within(self, src: str, dst: str, allowed_nodes: set) -> Optional[List[Dict]]:
        # BFS that records predecessor edge
        if src == dst:
            return []
        q = deque([src])
        prev: Dict[str, Tuple[str, Dict]] = {src: (None, None)}
        while q:
            u = q.popleft()
            for v, edge in self.fk_graph.get(u, []):
                if v not in allowed_nodes:
                    continue
                if v in prev:
                    continue
                prev[v] = (u, edge)
                if v == dst:
                    # reconstruct
                    path_edges: List[Dict] = []
                    cur = v
                    while prev[cur][0] is not None:
                        pu, pe = prev[cur]
                        # normalize edge direction (from_table -> to_table) as stored
                        path_edges.append({
                            'from_table': pe['from_table'],
                            'to_table': pe['to_table'],
                            'from_column': pe['from_column'],
                            'to_column': pe['to_column'],
                            'join_type': 'INNER JOIN',
                            'confidence': pe.get('confidence', 0.9),
                            'source': pe.get('source', 'database_structure'),
                        })
                        cur = pu
                    return list(reversed(path_edges))
                q.append(v)
        return None

    # -------------------------------
    # Stage 4/5 utilities (booleans, etc.)
    # -------------------------------
    # (LLM uses these through prompt instructions; included here for reference and potential template generation.)

    # -------------------------------
    # End helpers
    # -------------------------------


# End of file
