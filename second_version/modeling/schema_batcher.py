from typing import Dict, Any, List
import json

class SchemaBatcher:
    """Split discovery data into batches for rate-limited API calls."""
    
    @staticmethod
    def estimate_tokens(data: Dict[str, Any]) -> int:
        """Rough estimate of tokens in JSON data."""
        json_str = json.dumps(data)
        # Rough estimate: 1 token ≈ 4 characters
        return len(json_str) // 4
    
    @staticmethod
    def create_batches(
        discovery_data: Dict[str, Any], 
        max_tokens_per_batch: int = 20000
    ) -> List[Dict[str, Any]]:
        """
        Split discovery data into batches by schema.
        
        Args:
            discovery_data: Full discovery JSON
            max_tokens_per_batch: Target tokens per batch (default: 20k)
        
        Returns:
            List of discovery data batches
        """
        schemas = discovery_data.get("schemas", [])
        
        batches = []
        current_batch_schemas = []
        current_batch_tokens = 0
        
        base_data = {
            "database": discovery_data.get("database"),
            "dialect": discovery_data.get("dialect"),
            "named_assets": []  # Exclude named assets from batches
        }
        base_tokens = SchemaBatcher.estimate_tokens(base_data)
        
        for schema in schemas:
            schema_data = {"schemas": [schema]}
            schema_tokens = SchemaBatcher.estimate_tokens(schema_data)
            
            # If single schema exceeds limit, split its tables
            if schema_tokens > max_tokens_per_batch:
                table_batches = SchemaBatcher._split_large_schema(
                    schema, 
                    max_tokens_per_batch - base_tokens
                )
                for table_batch in table_batches:
                    batch = base_data.copy()
                    batch["schemas"] = [table_batch]
                    batches.append(batch)
                continue
            
            # Check if adding this schema exceeds limit
            if current_batch_tokens + schema_tokens > max_tokens_per_batch:
                # Save current batch and start new one
                if current_batch_schemas:
                    batch = base_data.copy()
                    batch["schemas"] = current_batch_schemas
                    batches.append(batch)
                
                current_batch_schemas = [schema]
                current_batch_tokens = base_tokens + schema_tokens
            else:
                current_batch_schemas.append(schema)
                current_batch_tokens += schema_tokens
        
        # Add final batch
        if current_batch_schemas:
            batch = base_data.copy()
            batch["schemas"] = current_batch_schemas
            batches.append(batch)
        
        return batches
    
    @staticmethod
    def _split_large_schema(
        schema: Dict[str, Any], 
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Split a large schema into multiple batches by tables."""
        tables = schema.get("tables", [])
        batches = []
        current_tables = []
        current_tokens = 0
        
        for table in tables:
            table_tokens = SchemaBatcher.estimate_tokens({"tables": [table]})
            
            if current_tokens + table_tokens > max_tokens:
                if current_tables:
                    batches.append({
                        "name": schema["name"],
                        "tables": current_tables
                    })
                current_tables = [table]
                current_tokens = table_tokens
            else:
                current_tables.append(table)
                current_tokens += table_tokens
        
        if current_tables:
            batches.append({
                "name": schema["name"],
                "tables": current_tables
            })
        
        return batches
    
    @staticmethod
    def merge_semantic_models(models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple semantic models into one.
        
        Args:
            models: List of semantic model JSONs
        
        Returns:
            Merged semantic model
        """
        if not models:
            return {}
        
        if len(models) == 1:
            return models[0]
        
        merged = {
            "entities": [],
            "dimensions": [],
            "facts": [],
            "relationships": [],
            "metrics": [],
            "audit": {
                "dialect": models[0].get("audit", {}).get("dialect", "tsql"),
                "source_assets_used": [],
                "assumptions": []
            }
        }
        
        # Merge each section, avoiding duplicates
        seen_entities = set()
        seen_dimensions = set()
        seen_facts = set()
        seen_relationships = set()
        seen_metrics = set()
        
        for model in models:
            # Merge entities
            for entity in model.get("entities", []):
                entity_key = entity.get("name")
                if entity_key not in seen_entities:
                    merged["entities"].append(entity)
                    seen_entities.add(entity_key)
            
            # Merge dimensions
            for dim in model.get("dimensions", []):
                dim_key = dim.get("name")
                if dim_key not in seen_dimensions:
                    merged["dimensions"].append(dim)
                    seen_dimensions.add(dim_key)
            
            # Merge facts
            for fact in model.get("facts", []):
                fact_key = fact.get("name")
                if fact_key not in seen_facts:
                    merged["facts"].append(fact)
                    seen_facts.add(fact_key)
            
            # Merge relationships
            for rel in model.get("relationships", []):
                rel_key = (rel.get("from"), rel.get("to"))
                if rel_key not in seen_relationships:
                    merged["relationships"].append(rel)
                    seen_relationships.add(rel_key)
            
            # Merge metrics
            for metric in model.get("metrics", []):
                metric_key = metric.get("name")
                if metric_key not in seen_metrics:
                    merged["metrics"].append(metric)
                    seen_metrics.add(metric_key)
            
            # Merge audit info
            audit = model.get("audit", {})
            merged["audit"]["source_assets_used"].extend(
                audit.get("source_assets_used", [])
            )
            merged["audit"]["assumptions"].extend(
                audit.get("assumptions", [])
            )
        
        return merged