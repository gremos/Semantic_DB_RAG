Goal

Auto-build semantic model from DB; answer NL→SQL; export to BI.

Prioritize correctness on “status/active” logic.

Inputs

DB catalog (tables, cols, PK/FK, types, row counts, samples).

Optional domain hints.

Outputs

Semantic model JSON (facts/dims/entities, measures, status filters, relationships).

Answers: validated SQL + brief rationale.

Exports: .bim, SQL views, JSON, Markdown.

Hard Constraints

Use incremental small calls (<500 tokens each) with fallbacks.

Respect cache (TTL 168h).

Deterministic style (temperature 0.1).

Validate SQL cols/tables exist before return.

Never ignore status semantics.

Pipeline (Phases)

Discovery & Compress

Read catalog; sample up to 100 semantic columns/table (status/type/name/desc/amount/ids).

Store compressed summary (~60% smaller).

Table Classify (per table)

Output: {classification: FACT|DIMENSION|ENTITY, confidence}.

Fallback heuristics: measures+FK→FACT; descriptive no-FK→DIM; else ENTITY.

Measure Identify (per fact)

Propose SUM/COUNT/AVG; keep dependencies + format.

Fallback: SUM numeric columns.

Status Analyze (per status-like col)

Infer NULL/value meaning; emit active_filter.

Heuristics: CancelledOn→IS NULL; IsActive→=1; enum Status via samples.

Relationship Infer (per FK/matches)

Emit cardinality + business meaning.

Assemble Model

Try single call; if empty/timeout, do programmatic merge from summaries.

Persist semantic cache.

Q&A (NL→SQL)

Map NL terms → columns via alias map (e.g., “product name”→ProductName).

Apply status filters automatically on measures/views.

Validate schema, parse SQL; return JSON with SQL + notes.

SQL Rules

Always include active filters from status indicators.

Prefer explicit joins on FK.

Use guarded aggregations (e.g., SUM(ISNULL(...,0))).

Date filters: pushdown to WHERE; support YoY/QoQ bins.

Top-K: ORDER BY + LIMIT/TOP.

Caching

Keys: discovery SHA256(conn_str); semantic SHA256(conn_str+hints).

TTL: 168h; manual bypass flag; clear-able.

First run may be long; later runs hit cache.

Exports

SQL Views (recommended): pre-filtered (active_filter), one per fact/dim + KPI views.

.bim: Tabular model with measures/relationships.

JSON: full model + usage guide.

Markdown: dictionary + diagrams refs.

Error Handling

Assembly timeout → fallback assemble.

Wrong status token (“success”) → normalize to ok.

Missing FKs → allow manual hints; keep soft relationships.

Column typos → verify, correct or fail with exact message.

Performance

Parallelize phases 1–4 (10–20 workers); rate-limit sleeps.

Prefer nearby LLM region.

Short prompts; JSON-only responses.

Log tokens/time per phase.

Exemplar Minimal Schemas

Status column record

{"table":"ContractProduct","column":"CancelledOn","role":"status_indicator","active_filter":"CancelledOn IS NULL","desc":"NULL=active, value=cancelled"}


Measure

{"table":"ContractProduct","measure":"NetAmount","expr":"SUM(Price*Quantity-Discount)","type":"decimal","format":"currency","deps":["Price","Quantity","Discount"]}


Relationship

{"from":"ContractProduct.BusinessPointID","to":"BusinessPoint.ID","cardinality":"many-to-one","meaning":"line → location"}


Answer payload

{"status":"ok","sql":"SELECT SUM(Price*Quantity-Discount) FROM dbo.ContractProduct WHERE CancelledOn IS NULL;"}

Troubleshooting (Quick)

No cache: ensure .cache/ writeable; init CacheManager.

Slow: check cache hits; reduce concurrency; shorter prompts.

Bad SQL: re-check dependencies; refresh discovery; clear cache.

No FKs: infer via name patterns or accept hints.

Prompting Style (for all LLM calls)

“Return JSON only; no prose.”

“Be concise; omit null/empty fields.”

“If uncertain, emit confidence:"low" + fallback suggestion.”

Safety/Determinism

Temperature 0.1, top_p 0.95, max_tokens 4000 (assembly).

Never fabricate columns/tables; if missing, stop and report.

NL Mapping

Build alias map from name/description columns; use in Q&A term grounding.

Examples: “client”→Customer, “contract line”→ContractProduct.

Default Views (pattern)

semantic.<FactName>: base filtered rows.

semantic.Metric_<KPI>: aggregated KPIs referencing base views.