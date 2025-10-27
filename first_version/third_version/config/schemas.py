"""
JSON Schemas for validation of Discovery, Semantic Model, and Q&A outputs.
Used to enforce constraint guardrails.
"""

DISCOVERY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["database", "dialect", "schemas"],
    "properties": {
        "database": {
            "type": "object",
            "required": ["vendor"],
            "properties": {
                "vendor": {"type": "string"},
                "version": {"type": "string"}
            }
        },
        "dialect": {"type": "string"},
        "schemas": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "tables"],
                "properties": {
                    "name": {"type": "string"},
                    "tables": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "type", "columns"],
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "enum": ["table", "view"]},
                                "columns": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["name", "type", "nullable"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                            "nullable": {"type": "boolean"},
                                            "stats": {"type": "object"}
                                        }
                                    }
                                },
                                "primary_key": {"type": "array", "items": {"type": "string"}},
                                "foreign_keys": {"type": "array"},
                                "rowcount_sample": {"type": "integer"},
                                "sample_rows": {"type": "array"}
                            }
                        }
                    }
                }
            }
        },
        "named_assets": {"type": "array"},
        "inferred_relationships": {"type": "array"}
    }
}


SEMANTIC_MODEL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["entities", "dimensions", "facts", "relationships"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "source", "columns"],
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string"},
                    "primary_key": {"type": "array", "items": {"type": "string"}},
                    "display": {"type": "object"},
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string"},
                                "role": {"type": "string"},
                                "semantic_type": {"type": "string"},
                                "description": {"type": "string"},
                                "aliases": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            }
        },
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "source", "keys", "attributes"],
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "attributes": {"type": "array"},
                    "display": {"type": "object"}
                }
            }
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "source", "grain"],
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string"},
                    "grain": {"type": "array", "items": {"type": "string"}},
                    "measures": {"type": "array"},
                    "foreign_keys": {"type": "array"},
                    "display": {"type": "object"}
                }
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to", "cardinality"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "cardinality": {"type": "string"},
                    "confidence": {"type": "string"},
                    "verification": {"type": "object"}
                }
            }
        },
        "table_rankings": {"type": "array"},
        "audit": {"type": "object"}
    }
}


QA_RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["status"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "refuse"]},
        "sql": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["dialect", "statement", "explanation", "evidence"],
                "properties": {
                    "dialect": {"type": "string"},
                    "statement": {"type": "string"},
                    "explanation": {"type": "string"},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "measures": {"type": "array", "items": {"type": "string"}},
                            "relationships": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "limits": {
                        "type": "object",
                        "properties": {
                            "row_limit": {"type": "integer"},
                            "timeout_sec": {"type": "integer"}
                        }
                    }
                }
            }
        },
        "result_preview": {
            "type": "object",
            "properties": {
                "first_row": {"type": "object"},
                "first_row_meaning": {"type": "string"},
                "rows_sampled": {"type": "integer"},
                "top_10_rows": {"type": "array"}
            }
        },
        "suggested_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"}
        },
        "refusal": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "clarifying_questions": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    "allOf": [
        {
            "if": {
                "properties": {"status": {"const": "ok"}}
            },
            "then": {
                "required": ["sql", "result_preview", "suggested_questions"]
            }
        },
        {
            "if": {
                "properties": {"status": {"const": "refuse"}}
            },
            "then": {
                "required": ["refusal"]
            }
        }
    ]
}


# Confidence scoring configuration
CONFIDENCE_SCORING = {
    "entity_match": 0.3,       # Found entities in question
    "measure_match": 0.3,      # Found measures
    "relationship_clear": 0.2, # Joins are unambiguous
    "temporal_clarity": 0.1,   # Date filters clear
    "aggregation_clarity": 0.1 # Grouping clear
}


# Compression strategies configuration
COMPRESSION_STRATEGIES = {
    "detailed": {
        "include_all": True,
        "sample_values": 10,
        "full_stats": True
    },
    "tldr": {
        "include_all": False,
        "sample_values": 5,
        "full_stats": False,
        "summary_only": True
    },
    "map_reduce": {
        "include_all": False,
        "sample_values": 5,
        "full_stats": False,
        "per_table_summary": True,
        "combine_phase": True
    },
    "recap": {
        "include_all": False,
        "sample_values": 3,
        "full_stats": False,
        "dedupe_synonyms": True
    }
}