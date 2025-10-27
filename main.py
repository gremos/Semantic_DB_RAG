"""
Main entry point for GPT-5 Semantic Modeling & SQL Q&A System
Enhanced with SQL execution and result display
"""

import sys
import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline components
from orchestration.pipeline import SemanticPipeline


def load_config() -> dict:
    """Load configuration from .env file."""
    load_dotenv()
    
    config = {
        "DEPLOYMENT_NAME": os.getenv("DEPLOYMENT_NAME"),
        "API_VERSION": os.getenv("API_VERSION", "2025-01-01-preview"),
        "AZURE_ENDPOINT": os.getenv("AZURE_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "DATABASE_CONNECTION_STRING": os.getenv("DATABASE_CONNECTION_STRING"),
        "UTF8_ENCODING": os.getenv("UTF8_ENCODING", "true").lower() == "true",
        "SCHEMA_EXCLUSIONS": os.getenv("SCHEMA_EXCLUSIONS", "sys,information_schema").split(','),
        "TABLE_EXCLUSIONS": os.getenv("TABLE_EXCLUSIONS", "temp_,test_,backup_,old_").split(','),
        "DISCOVERY_CACHE_HOURS": int(os.getenv("DISCOVERY_CACHE_HOURS", "168")),
        "SEMANTIC_CACHE_HOURS": int(os.getenv("SEMANTIC_CACHE_HOURS", "168")),
        "CACHE_DIR": os.getenv("CACHE_DIR", "./cache"),
    }
    
    # Validate required config
    required = [
        "DEPLOYMENT_NAME",
        "AZURE_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "DATABASE_CONNECTION_STRING"
    ]
    
    missing = [key for key in required if not config.get(key)]
    if missing:
        logger.error(f"Missing required configuration: {', '.join(missing)}")
        logger.error("Please check your .env file")
        sys.exit(1)
    
    return config


def print_usage():
    """Print usage information."""
    print("""
Usage: python main.py <command> [arguments]

Commands:
    cache-clear              Clear all cached discovery and semantic models
    discover                 Run database discovery phase only
    model                    Build semantic model (requires discovery)
    query <text>             Ask a question and generate SQL
    explain-model            Show semantic model summary
    validate                 Validate semantic model completeness
    
Examples:
    python main.py discover
    python main.py model
    python main.py query "rank products by total sales"
    python main.py explain-model
    python main.py cache-clear
""")


def display_answer(answer: dict, verbose: bool = False):
    """
    Display answer in a readable format.
    
    Args:
        answer: Answer JSON from pipeline
        verbose: Show full evidence and metadata
    """
    print(f"\n{'='*80}")
    print(f"STATUS: {answer.get('status', 'unknown').upper()}")
    print(f"{'='*80}\n")
    
    if answer.get("status") == "refuse":
        # Show refusal reason and guidance
        refusal = answer.get("refusal", {})
        print(f"❌ REFUSAL REASON:")
        print(f"   {refusal.get('reason', 'Unknown reason')}\n")
        
        if refusal.get("missing"):
            print(f"🔍 MISSING FROM SEMANTIC MODEL:")
            for item in refusal["missing"]:
                print(f"   • {item}")
            print()
        
        if refusal.get("clarifying_questions"):
            print(f"💡 CLARIFYING QUESTIONS:")
            for i, q in enumerate(refusal["clarifying_questions"], 1):
                print(f"   {i}. {q}")
            print()
        
        if refusal.get("suggestions"):
            print(f"💭 SUGGESTIONS:")
            for suggestion in refusal["suggestions"]:
                print(f"   • {suggestion}")
            print()
    
    elif answer.get("status") == "ok":
        # Show SQL and evidence
        sql_statements = answer.get("sql", [])
        
        for idx, sql_obj in enumerate(sql_statements, 1):
            sql = sql_obj.get("statement", "")
            dialect = sql_obj.get("dialect", "unknown")
            explanation = sql_obj.get("explanation", "")
            evidence = sql_obj.get("evidence", {})
            limits = sql_obj.get("limits", {})
            
            print(f"📝 SQL QUERY {idx} ({dialect}):")
            print("-" * 80)
            print(sql)
            print("-" * 80)
            print()
            
            if explanation:
                print(f"💡 EXPLANATION:")
                print(f"   {explanation}\n")
            
            if evidence and verbose:
                print(f"🔍 EVIDENCE:")
                if evidence.get("entities"):
                    print(f"   Entities: {', '.join(evidence['entities'])}")
                if evidence.get("dimensions"):
                    print(f"   Dimensions: {', '.join(evidence['dimensions'])}")
                if evidence.get("facts"):
                    print(f"   Facts: {', '.join(evidence['facts'])}")
                if evidence.get("measures"):
                    print(f"   Measures: {', '.join(evidence['measures'])}")
                if evidence.get("filters"):
                    print(f"   Filters: {', '.join(evidence['filters'])}")
                print()
            
            if limits:
                print(f"⚙️  LIMITS:")
                if limits.get("row_limit"):
                    print(f"   Row Limit: {limits['row_limit']}")
                if limits.get("timeout_sec"):
                    print(f"   Timeout: {limits['timeout_sec']}s")
                print()
        
        if answer.get("next_steps"):
            print(f"➡️  NEXT STEPS:")
            for step in answer["next_steps"]:
                print(f"   • {step}")
            print()
    
    else:
        print(f"⚠️  Unknown status: {answer.get('status')}")
        print(json.dumps(answer, indent=2))
    
    print(f"{'='*80}\n")


def main():
    """Main CLI entry point."""
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Load configuration
    config = load_config()
    
    # Initialize LLM client
    from langchain_openai import AzureChatOpenAI
    
    logger.info("Initializing Azure OpenAI client...")
    llm_client = AzureChatOpenAI(
        azure_endpoint=config["AZURE_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"],
        deployment_name=config["DEPLOYMENT_NAME"],
        api_version=config["API_VERSION"],
        # temperature=0.1,  # Low temperature for deterministic outputs
        max_tokens=4000
    )
    
    # Initialize pipeline with proper parameters
    logger.info("Initializing semantic pipeline...")
    pipeline = SemanticPipeline(
        db_connection_string=config["DATABASE_CONNECTION_STRING"],
        llm_client=llm_client,
        cache_dir=Path(config["CACHE_DIR"]),
        discovery_cache_hours=config["DISCOVERY_CACHE_HOURS"],
        semantic_cache_hours=config["SEMANTIC_CACHE_HOURS"]
    )
    
    # Execute command
    try:
        if command == "cache-clear":
            logger.info("Clearing all caches...")
            pipeline.invalidate_caches()
            logger.info("✅ Cache cleared successfully")
        
        elif command == "discover":
            logger.info("Running discovery phase...")
            discovery = pipeline._get_or_create_discovery()
            
            # Pretty print summary
            print("\n=== DISCOVERY SUMMARY ===")
            print(f"Database: {discovery.get('database', {}).get('vendor', 'unknown')} "
                  f"({discovery.get('database', {}).get('version', 'unknown')})")
            print(f"Dialect: {discovery.get('dialect', 'unknown')}")
            print(f"Schemas: {len(discovery.get('schemas', []))}")
            
            total_tables = sum(len(s.get('tables', [])) for s in discovery.get('schemas', []))
            print(f"Tables: {total_tables}")
            
            views = len([a for a in discovery.get('named_assets', []) if a.get('kind') == 'view'])
            sps = len([a for a in discovery.get('named_assets', []) if a.get('kind') == 'stored_procedure'])
            rdls = len([a for a in discovery.get('named_assets', []) if a.get('kind') == 'rdl'])
            
            print(f"Views: {views}")
            print(f"Stored Procedures: {sps}")
            print(f"RDL Files: {rdls}")
            print(f"\n✅ Discovery saved to cache")
        
        elif command == "model":
            logger.info("Building semantic model...")
            
            # Get discovery first
            discovery = pipeline._get_or_create_discovery()
            
            # Build semantic model
            model = pipeline._get_or_create_semantic_model(discovery)
            
            # Validate completeness
            validation = pipeline._validate_model_completeness(model)
            
            # Pretty print summary
            print("\n=== SEMANTIC MODEL SUMMARY ===")
            print(f"Entities: {len(model.get('entities', []))}")
            print(f"Dimensions: {len(model.get('dimensions', []))}")
            print(f"Facts: {len(model.get('facts', []))}")
            print(f"Relationships: {len(model.get('relationships', []))}")
            print(f"Metrics: {len(model.get('metrics', []))}")
            
            if not validation["is_complete"]:
                print(f"\n⚠️  WARNINGS:")
                for warning in validation["missing"]:
                    print(f"   • {warning}")
            
            print(f"\n✅ Semantic model saved to cache")
        
        elif command == "query":
            if len(sys.argv) < 3:
                logger.error("Please provide a question")
                print_usage()
                sys.exit(1)
            
            question = " ".join(sys.argv[2:])
            logger.info(f"Processing question: {question}")
            
            # Run full pipeline
            answer = pipeline.process_question(question)
            
            # Display answer
            verbose = "--verbose" in sys.argv or "-v" in sys.argv
            display_answer(answer, verbose=verbose)
        
        elif command == "explain-model":
            logger.info("Getting semantic model summary...")
            summary = pipeline.get_model_summary()
            
            if "error" in summary:
                print(f"\n❌ {summary['error']}")
                print("Run 'python main.py model' first")
            else:
                print("\n=== SEMANTIC MODEL DETAILS ===")
                print(f"Cache File: {summary['cache_file']}")
                print(f"Cache Age: {summary['cache_age_hours']:.1f} hours")
                print(f"\nCounts:")
                print(f"  Entities: {summary['entity_count']}")
                print(f"  Dimensions: {summary['dimension_count']}")
                print(f"  Facts: {summary['fact_count']}")
                print(f"  Relationships: {summary['relationship_count']}")
                print(f"  Metrics: {summary['metric_count']}")
                
                if summary.get("facts"):
                    print(f"\nFacts:")
                    for fact in summary["facts"]:
                        print(f"  • {fact['name']} (source: {fact['source']})")
                        print(f"    - Measures: {fact['measure_count']}")
                        print(f"    - Filter Columns: {fact['filter_column_count']}")
                        if fact.get("measures"):
                            for measure in fact["measures"]:
                                print(f"      → {measure}")
        
        elif command == "validate":
            logger.info("Validating semantic model...")
            
            # Load cached model
            import glob
            cache_files = glob.glob(f"{config['CACHE_DIR']}/semantic_model_*.json")
            
            if not cache_files:
                print("\n❌ No semantic model found in cache")
                print("Run 'python main.py model' first")
                sys.exit(1)
            
            latest_cache = max(cache_files, key=os.path.getmtime)
            with open(latest_cache, 'r') as f:
                model = json.load(f)
            
            validation = pipeline._validate_model_completeness(model)
            
            print("\n=== MODEL VALIDATION ===")
            if validation["is_complete"]:
                print("✅ Model is complete and ready for Q&A")
            else:
                print("⚠️  Model has issues:")
                for issue in validation["missing"]:
                    print(f"   • {issue}")
        
        else:
            logger.error(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error executing command '{command}': {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()