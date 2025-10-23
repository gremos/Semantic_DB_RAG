from typing import Dict, Any, List, Tuple
from llm.azure_client import AzureLLMClient
from utils.json_extractor import JSONExtractor
import json
import logging

logger = logging.getLogger(__name__)

class ModelAssembler:
    """Phase 5: Assemble complete semantic model from phase results."""
    
    def __init__(self, llm_client: AzureLLMClient):
        self.llm = llm_client
        self.json_extractor = JSONExtractor()
        self._load_prompt()
    
    def _load_prompt(self):
        """Load assembly prompt."""
        with open("prompts/assembly_prompt.txt", 'r') as f:
            self.system_prompt = f.read()
    
    def assemble_model(
        self,
        classifications: Dict[str, Dict[str, Any]],
        measures: Dict[str, List[Dict[str, Any]]],
        status_columns: Dict[str, Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Assemble final semantic model from all phase results.
        
        Returns:
            (success, semantic_model, error)
        """
        try:
            # Build compressed summary
            summary = self._build_summary(
                classifications,
                measures,
                status_columns,
                relationships,
                compressed_discovery
            )
            
            logger.info(f"Assembly summary size: {len(json.dumps(summary))} characters")
            
            user_prompt = f"""# Phase Results Summary

{json.dumps(summary, indent=2)}

# Task
Generate complete semantic model JSON following semantic_model_schema.json structure.

Include:
1. entities: From ENTITY classifications, with columns
2. dimensions: From DIMENSION classifications, with columns
3. facts: From FACT classifications, with columns + measures from phase 2
4. relationships: From phase 4 results
5. metrics: Derive from measures (at least 3 metrics)
6. audit: dialect, source_assets_used, assumptions

For each fact, include:
- All columns with semantic_role
- Mark status_indicator columns from phase 3
- Add filters_applied to measures based on status columns

Return ONLY complete semantic model JSON."""
            
            response = self.llm.generate(self.system_prompt, user_prompt)
            
            logger.info(f"Assembly response size: {len(response)} characters")
            
            result, method = self.json_extractor.extract(response, log_failures=True)
            
            if not result:
                logger.error("Failed to assemble model, trying fallback")
                result = self._fallback_assembly(
                    classifications, measures, status_columns, 
                    relationships, compressed_discovery
                )
            
            return (True, result, "")
        
        except Exception as e:
            logger.error(f"Error assembling model: {e}", exc_info=True)
            return (False, {}, str(e))
    
    def _build_summary(
        self,
        classifications: Dict[str, Dict[str, Any]],
        measures: Dict[str, List[Dict[str, Any]]],
        status_columns: Dict[str, Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build ULTRA-COMPRESSED summary for final assembly."""
        # Group by classification
        entities = []
        dimensions = []
        facts = []
        
        for table_name, classification in classifications.items():
            table_data = compressed_discovery["tables"].get(table_name, {})
            
            # ULTRA-COMPRESS: Only include essential info
            entry = {
                "name": table_name,
                "pk": table_data.get("pk", []),
                # Don't include all columns - too verbose
                "col_count": len(table_data.get("columns", [])),
                # Only include column names, not full definitions
                "cols": [c["name"] for c in table_data.get("columns", [])[:10]],  # First 10 only
            }
            
            # Add status column keys only (not full analysis)
            table_status_keys = [k.split('.')[1] for k in status_columns.keys() if k.startswith(f"{table_name}.")]
            if table_status_keys:
                entry["status_cols"] = table_status_keys[:5]  # Limit to 5
            
            # Add measure names only (not full definitions) if fact
            if classification["classification"] == "fact" and table_name in measures:
                entry["measure_names"] = [m["name"] for m in measures[table_name]]
            
            # Categorize
            if classification["classification"] == "entity":
                entities.append(entry)
            elif classification["classification"] == "dimension":
                dimensions.append(entry)
            else:  # fact
                facts.append(entry)
        
        # Compress relationships - just from/to
        rel_summary = [f"{r['from']}→{r['to']}" for r in relationships[:50]]  # Limit to 50
        
        # Compress status columns - only keys and active_filter
        status_summary = {
            k: {
                "role": "status_indicator",
                "filter": v.get("active_filter", "")
            }
            for k, v in list(status_columns.items())[:30]  # Limit to 30
        }
        
        return {
            "dialect": compressed_discovery.get("dialect", "tsql"),
            "entity_names": [e["name"] for e in entities],
            "dimension_names": [d["name"] for d in dimensions],
            "fact_names": [f["name"] for f in facts],
            "facts_detail": facts,  # Only facts need detail
            "relationships": rel_summary,
            "status_columns": status_summary,
            "total_entities": len(entities),
            "total_dimensions": len(dimensions),
            "total_facts": len(facts)
        }
    
    def _fallback_assembly(
        self,
        classifications: Dict[str, Dict[str, Any]],
        measures: Dict[str, List[Dict[str, Any]]],
        status_columns: Dict[str, Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        compressed_discovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback: Manually construct model if LLM fails."""
        logger.info("Using fallback manual assembly")
        
        model = {
            "entities": [],
            "dimensions": [],
            "facts": [],
            "relationships": [],
            "metrics": [],
            "audit": {
                "dialect": compressed_discovery.get("dialect", "tsql"),
                "source_assets_used": [],
                "assumptions": ["Generated via fallback assembly"]
            }
        }
        
        # Build entities, dimensions, facts
        for table_name, classification in classifications.items():
            table_data = compressed_discovery["tables"].get(table_name, {})
            
            # Build columns with status info
            columns = []
            for col in table_data.get("columns", []):
                col_key = f"{table_name}.{col['name']}"
                status_info = status_columns.get(col_key)
                
                col_entry = {
                    "name": col["name"],
                    "type": col["type"],
                    "nullable": col.get("nullable", True),
                    "semantic_role": "metadata",
                    "description": col["name"]
                }
                
                if status_info:
                    col_entry["semantic_role"] = "status_indicator"
                    col_entry["description"] = status_info.get("description", "")
                
                columns.append(col_entry)
            
            if classification["classification"] == "entity":
                model["entities"].append({
                    "name": table_name.split('.')[-1],
                    "source": table_name,
                    "primary_key": table_data.get("pk", []),
                    "description": f"Entity: {table_name}",
                    "columns": columns
                })
                model["audit"]["source_assets_used"].append({
                    "kind": "table",
                    "name_or_path": table_name
                })
            
            elif classification["classification"] == "dimension":
                model["dimensions"].append({
                    "name": table_name.split('.')[-1],
                    "source": table_name,
                    "keys": table_data.get("pk", []),
                    "attributes": [c["name"] for c in columns],
                    "columns": columns
                })
                model["audit"]["source_assets_used"].append({
                    "kind": "table",
                    "name_or_path": table_name
                })
            
            else:  # fact
                table_measures = measures.get(table_name, [])
                
                # Add filters to measures based on status columns
                for measure in table_measures:
                    filters = []
                    for col_name in measure.get("depends_on", []):
                        col_key = f"{table_name}.{col_name}"
                        if col_key in status_columns:
                            filter_cond = status_columns[col_key].get("active_filter", "")
                            if filter_cond:
                                filters.append(filter_cond)
                    measure["filters_applied"] = filters
                
                model["facts"].append({
                    "name": table_name.split('.')[-1],
                    "source": table_name,
                    "grain": table_data.get("pk", []),
                    "columns": columns,
                    "measures": table_measures,
                    "foreign_keys": [
                        {"column": fk.split('→')[0], "references": fk.split('→')[1].split('.')[0]}
                        for fk in table_data.get("fks", [])
                    ]
                })
                model["audit"]["source_assets_used"].append({
                    "kind": "table",
                    "name_or_path": table_name
                })
        
        # Add relationships
        for rel in relationships:
            model["relationships"].append({
                "from": rel["from"],
                "to": rel["to"],
                "cardinality": rel.get("cardinality", "many-to-one"),
                "type": "foreign_key"
            })
        
        # Generate basic metrics from measures
        for fact in model["facts"]:
            for measure in fact.get("measures", [])[:3]:  # First 3 measures
                model["metrics"].append({
                    "name": f"{fact['name']} {measure['name']}",
                    "purpose": f"Track {measure['name']}",
                    "logic": measure["expression"],
                    "inputs": [f"{fact['name']}.{measure['name']}"],
                    "constraints": measure.get("filters_applied", []),
                    "explain": f"Aggregate {measure['name']} from {fact['name']}"
                })
        
        return model