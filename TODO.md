# Implementation TODO

Roadmap for completing the GPT-5 Semantic Modeling & SQL Q&A System.

## Priority 1: Core LLM Integration

These are required for the system to work end-to-end.

### 1. LLM Client Enhancement (`src/llm/client.py`)

- [ ] Implement `invoke_with_json()` with retry logic
- [ ] Add JSON schema validation with exponential backoff
- [ ] Add error handling for API failures
- [ ] Add token counting and management
- [ ] Add prompt compression if needed

**Estimated Effort:** 4-6 hours

### 2. Prompt Engineering (`src/llm/prompts.py`)

Create prompt templates for:

- [ ] Entity identification
- [ ] Dimension identification  
- [ ] Fact identification with measures
- [ ] Column semantic type assignment
- [ ] Question parsing and intent extraction
- [ ] SQL generation with grounding

**Estimated Effort:** 8-10 hours

## Priority 2: Semantic Model Builder

Complete Phase 2 implementation.

### 3. Model Builder LLM Integration (`src/semantic/model_builder.py`)

- [ ] Implement `_identify_entities()` with LLM
  - Extract entity tables from discovery
  - Assign semantic types to columns
  - Define display properties
- [ ] Implement `_identify_dimensions()` with LLM
  - Extract dimension tables
  - Define hierarchies and attributes
  - Order attributes logically
- [ ] Implement `_identify_facts()` with LLM
  - Extract fact tables
  - Define measures with units/currency
  - Identify grain columns
  - Define default filters/breakdowns
- [ ] Implement `_build_relationships()`
  - Map discovery relationships to semantic objects
  - Validate cardinality
  - Calculate confidence
- [ ] Implement `_rank_tables()`
  - Prefer views > SPs > RDL > tables
  - Mark duplicates

**Estimated Effort:** 12-16 hours

### 4. Compression Strategies (`src/semantic/compression.py`)

- [ ] Implement `tldr` compression (default)
  - Extract essential info: columns, types, PK/FK, top 5 values
- [ ] Implement `map_reduce` compression
  - Summarize per-table
  - Combine summaries
- [ ] Implement `recap` compression
  - Dedupe synonyms and aliases
- [ ] Add token counting to stay within limits

**Estimated Effort:** 6-8 hours

### 5. Batch Processing (`src/semantic/batch_processor.py`)

- [ ] Implement entity batching (ENTITY_BATCH_SIZE)
- [ ] Implement dimension batching (DIMENSION_BATCH_SIZE)
- [ ] Implement fact batching (FACT_BATCH_SIZE)
- [ ] Handle batch assembly and merging
- [ ] Add progress tracking

**Estimated Effort:** 4-6 hours

### 6. Table Ranking (`src/semantic/ranking.py`)

- [ ] Implement ranking algorithm
- [ ] Mark duplicates based on column overlap
- [ ] Prefer curated sources (views, SPs, RDL)
- [ ] Handle synonyms and aliases

**Estimated Effort:** 4-6 hours

## Priority 3: Q&A Pipeline

Complete Phase 3 implementation.

### 7. Question Parser (`src/qa/question_parser.py`)

- [ ] Implement question intent extraction with LLM
- [ ] Implement confidence scoring
  - Entity match (30%)
  - Measure match (30%)
  - Relationship clarity (20%)
  - Temporal clarity (10%)
  - Aggregation clarity (10%)
- [ ] Handle ambiguous questions
- [ ] Generate clarifying questions

**Estimated Effort:** 8-10 hours

### 8. SQL Generator (`src/qa/sql_generator.py`)

- [ ] Generate SQL from question intent + semantic model
- [ ] Enforce grounding (whitelist to semantic model)
- [ ] Add TOP(n) limits automatically
- [ ] Add timeout hints
- [ ] Generate evidence chain (entities, measures, relationships used)
- [ ] Handle complex joins
- [ ] Handle aggregations and grouping
- [ ] Handle date filters

**Estimated Effort:** 12-16 hours

### 9. Query Executor (`src/qa/executor.py`)

- [ ] Implement safe query execution
- [ ] Enforce read-only (validate before execution)
- [ ] Apply row limits (default 10, max 1000)
- [ ] Apply timeouts (default 60s)
- [ ] Handle execution errors gracefully
- [ ] Return results as structured data

**Estimated Effort:** 6-8 hours

### 10. Response Formatter (`src/qa/response_formatter.py`)

- [ ] Format query results as JSON
- [ ] Generate first row interpretation (plain English)
- [ ] Extract top 10 rows for history
- [ ] Generate 2-6 suggested follow-up questions
- [ ] Add next steps/tips if relevant
- [ ] Format refusal with clarifying questions

**Estimated Effort:** 6-8 hours

### 11. Q&A History Logging

- [ ] Implement JSONL logger for Q&A history
- [ ] Log: question, SQL, evidence, top 10 rows, timestamp
- [ ] Add log rotation if needed
- [ ] Add analysis tools for mining patterns

**Estimated Effort:** 3-4 hours

## Priority 4: QuadRails (Hallucination Prevention)

### 12. Grounding Verification (`src/guardrails/grounding.py`)

- [ ] Verify all SQL tokens map to discovery objects
- [ ] Cross-check against semantic model
- [ ] Validate table names exist
- [ ] Validate column names exist
- [ ] Validate joins are defined
- [ ] Return violations if any

**Estimated Effort:** 6-8 hours

### 13. JSON Validator (`src/guardrails/validator.py`)

- [ ] Implement schema validation (jsonschema library)
- [ ] Add 3 retries with exponential backoff
- [ ] Log validation errors
- [ ] Return structured errors for LLM to fix

**Estimated Effort:** 4-6 hours

### 14. SQL Verifier (`src/guardrails/sql_verifier.py`)

- [ ] Use sqlglot to parse SQL
- [ ] Verify syntax is valid
- [ ] Verify tables/columns exist in semantic model
- [ ] Verify joins have defined relationships
- [ ] Check for forbidden operations (DML/DDL)
- [ ] Return lint warnings

**Estimated Effort:** 6-8 hours

### 15. Confidence Scorer (`src/guardrails/confidence.py`)

- [ ] Implement confidence calculation
- [ ] Use scoring weights from config
- [ ] Apply thresholds (high, medium, low)
- [ ] Auto-execute if high confidence
- [ ] Execute with disclaimer if medium
- [ ] Refuse with clarifiers if low

**Estimated Effort:** 4-6 hours

## Priority 5: Data Models

### 16. Pydantic Models (`src/models/`)

- [ ] Create `discovery_model.py`
  - Schema, Table, Column, Relationship, Asset models
- [ ] Create `semantic_model.py`
  - Entity, Dimension, Fact, Measure, Relationship models
- [ ] Create `qa_model.py`
  - Question, Answer, SQL, Evidence, ResultPreview models
- [ ] Add validation rules
- [ ] Add serialization helpers

**Estimated Effort:** 8-10 hours

## Priority 6: Testing

### 17. Unit Tests

- [ ] Test discovery introspection
- [ ] Test data sampling
- [ ] Test relationship detection
- [ ] Test RDL parsing
- [ ] Test semantic model building
- [ ] Test SQL generation
- [ ] Test guardrails validation
- [ ] Test caching
- [ ] Test configuration

**Estimated Effort:** 16-20 hours

### 18. Integration Tests

- [ ] Test full discovery pipeline
- [ ] Test full semantic model pipeline
- [ ] Test full Q&A pipeline
- [ ] Test cache invalidation
- [ ] Test error handling

**Estimated Effort:** 8-10 hours

## Priority 7: Database Support

### 19. Connection Management (`src/db/connection.py`)

- [ ] Implement connection pooling
- [ ] Add connection retry logic
- [ ] Add connection timeout handling
- [ ] Support multiple database vendors

**Estimated Effort:** 4-6 hours

### 20. Read-Only Guard (`src/db/readonly_guard.py`)

- [ ] Enforce read-only at connection level
- [ ] Block DML/DDL operations
- [ ] Add transaction rollback on violations

**Estimated Effort:** 3-4 hours

## Priority 8: Polish & Documentation

### 21. Error Handling

- [ ] Add custom exceptions
- [ ] Improve error messages
- [ ] Add error recovery strategies
- [ ] Add user-friendly error reporting

**Estimated Effort:** 6-8 hours

### 22. Logging Improvements

- [ ] Add structured logging
- [ ] Add log levels per module
- [ ] Add performance metrics
- [ ] Add debug mode

**Estimated Effort:** 4-6 hours

### 23. CLI Enhancements

- [ ] Add --verbose flag
- [ ] Add --force-refresh flag
- [ ] Add progress bars for long operations
- [ ] Add interactive mode
- [ ] Add question history browser

**Estimated Effort:** 6-8 hours

### 24. Documentation

- [ ] Add inline code documentation
- [ ] Add API documentation
- [ ] Add architecture diagrams
- [ ] Add tutorial notebooks
- [ ] Add FAQ

**Estimated Effort:** 8-10 hours

## Priority 9: Performance

### 25. Optimization

- [ ] Profile discovery performance
- [ ] Add parallel processing for sampling
- [ ] Optimize relationship detection
- [ ] Add database query caching
- [ ] Optimize LLM token usage

**Estimated Effort:** 8-12 hours

### 26. Scalability

- [ ] Handle very large databases (1000+ tables)
- [ ] Add incremental discovery
- [ ] Add diff detection for schema changes
- [ ] Add nightly refresh jobs

**Estimated Effort:** 12-16 hours

## Priority 10: Advanced Features

### 27. Multi-Database Support

- [ ] Support PostgreSQL
- [ ] Support MySQL
- [ ] Support SQLite
- [ ] Support Oracle
- [ ] Auto-detect dialect

**Estimated Effort:** 12-16 hours

### 28. Advanced Q&A

- [ ] Support multi-query questions
- [ ] Support subqueries
- [ ] Support CTEs
- [ ] Support window functions
- [ ] Support complex aggregations

**Estimated Effort:** 12-16 hours

### 29. Visualization

- [ ] Generate query result charts
- [ ] Create semantic model diagrams
- [ ] Add interactive dashboards
- [ ] Export to BI tools

**Estimated Effort:** 16-20 hours

### 30. Collaboration Features

- [ ] Share semantic models
- [ ] Share Q&A history
- [ ] Version control for models
- [ ] Model approval workflow

**Estimated Effort:** 16-20 hours

## Total Estimated Effort

**Priority 1-4 (MVP):** 120-160 hours
**Priority 5-6 (Production Ready):** 32-40 hours
**Priority 7-8 (Polish):** 27-36 hours
**Priority 9-10 (Advanced):** 76-104 hours

**Total:** 255-340 hours (6-8 weeks for 1 developer)

## Recommended Development Order

### Phase A: Get to Working MVP (Priority 1-4)

1. LLM Client & Prompts (12-16 hours)
2. Semantic Model Builder (38-50 hours)
3. Q&A Pipeline (41-56 hours)
4. QuadRails (20-28 hours)

**Result:** End-to-end working system

### Phase B: Make Production Ready (Priority 5-6)

5. Pydantic Models (8-10 hours)
6. Testing (24-30 hours)

**Result:** Stable, tested system

### Phase C: Polish (Priority 7-8)

7. Database Support (7-10 hours)
8. Error Handling & Logging (10-14 hours)
9. CLI & Documentation (14-18 hours)

**Result:** Professional-grade system

### Phase D: Optimize & Extend (Priority 9-10)

10. Performance & Scalability (20-28 hours)
11. Advanced Features (56-84 hours)

**Result:** Enterprise-grade system

## Quick Wins

Easy implementations that provide immediate value:

1. **JSON Validator** (4-6 hours) - Prevents bad LLM outputs
2. **Q&A History Logging** (3-4 hours) - Enables pattern analysis
3. **Confidence Scorer** (4-6 hours) - Improves answer quality
4. **Error Handling** (6-8 hours) - Better user experience
5. **CLI Enhancements** (6-8 hours) - Better usability

## Current Status

### ✅ Completed
- Project structure
- Configuration management
- Phase 1 Discovery (full implementation)
- Utilities (caching, logging, SQL utils)
- CLI framework
- Documentation (README, QUICKSTART, PROJECT_STRUCTURE)

### 🔨 Next Up
- LLM Client & Prompts
- Semantic Model Builder
- Q&A Pipeline
- QuadRails

### 📋 Future
- Testing
- Multi-database support
- Advanced features
- Visualization

## Notes

- Focus on getting MVP working first (Priority 1-4)
- Test each component thoroughly before moving on
- Use real database for testing
- Collect example questions for testing Q&A
- Monitor LLM token usage and costs
- Consider using Claude Code for implementation assistance
