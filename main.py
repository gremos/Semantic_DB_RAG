"""
Main entry point for GPT-5 Semantic Modeling & SQL Q&A System
Enhanced with SQL execution and result display
"""

import sys
import os
import logging
import json
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline and executor
from orchestration.pipeline import SemanticPipeline
from qa.sql_executor import SQLExecutor


def load_config() -> dict:
    """Load configuration from .env file."""
    load_dotenv()
    
    config = {
        "DEPLOYMENT_NAME": os.getenv("DEPLOYMENT_NAME"),
        "API_VERSION": os.getenv("API_VERSION"),
        "AZURE_ENDPOINT": os.getenv("AZURE_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "DATABASE_CONNECTION_STRING": os.getenv("DATABASE_CONNECTION_STRING"),
        "UTF8_ENCODING": os.getenv("UTF8_ENCODING", "true"),
        "SCHEMA_EXCLUSIONS": os.getenv("SCHEMA_EXCLUSIONS", "sys,information_schema"),
        "TABLE_EXCLUSIONS": os.getenv("TABLE_EXCLUSIONS", "temp_,test_,backup_,old_"),
        "DISCOVERY_CACHE_HOURS": os.getenv("DISCOVERY_CACHE_HOURS", "168"),
        "SEMANTIC_CACHE_HOURS": os.getenv("SEMANTIC_CACHE_HOURS", "168"),
    }
    
    # Validate required config
    required = [
        "DEPLOYMENT_NAME",
        "API_VERSION",
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
    cache-clear              Clear all cached discovery and models
    discover                 Run database discovery only
    model                    Run semantic modeling only (requires discovery)
    question <text>          Ask a question (generates SQL only)
    full <text>              Run full pipeline with question (generates SQL only)
    execute <text>           Ask question AND execute SQL with results (NEW)
    
Examples:
    python main.py cache-clear
    python main.py discover
    python main.py model
    python main.py question "What are the total sales by customer?"
    python main.py full "Show me revenue trends by month"
    python main.py execute "Show me top 10 customers by revenue"
""")


def execute_and_display_results(
    answer: dict,
    executor: SQLExecutor,
    row_limit: int = 10
):
    """
    Execute SQL from answer and display results in terminal.
    
    Args:
        answer: Answer JSON from pipeline
        executor: SQL executor instance
        row_limit: Number of rows to display
    """
    if answer.get("status") != "ok":
        print("\n❌ Cannot execute: Query was refused")
        print(f"Reason: {answer.get('refusal', {}).get('reason', 'Unknown')}")
        return
    
    sql_statements = answer.get("sql", [])
    
    if not sql_statements:
        print("\n❌ No SQL statements to execute")
        return
    
    # Execute each SQL statement
    for idx, sql_obj in enumerate(sql_statements, 1):
        sql = sql_obj.get("statement", "")
        dialect = sql_obj.get("dialect", "unknown")
        explanation = sql_obj.get("explanation", "")
        limits = sql_obj.get("limits", {})
        
        print(f"\n{'='*80}")
        print(f"QUERY {idx} of {len(sql_statements)}")
        print(f"{'='*80}")
        
        # Show SQL
        print(f"\n📝 SQL ({dialect}):")
        print("-" * 80)
        print(sql)
        print("-" * 80)
        
        # Show explanation
        if explanation:
            print(f"\n💡 Explanation:")
            print(f"   {explanation}")
        
        # Show evidence
        evidence = sql_obj.get("evidence", {})
        if evidence:
            print(f"\n🔍 Evidence:")
            if evidence.get("entities"):
                print(f"   Entities: {', '.join(evidence['entities'])}")
            if evidence.get("measures"):
                print(f"   Measures: {', '.join(evidence['measures'])}")
        
        # Execute query
        print(f"\n⚡ Executing query (limit: {row_limit} rows)...")
        
        timeout_sec = limits.get("timeout_sec", 60)
        success, results, error_msg, exec_time = executor.execute_query(
            sql=sql,
            row_limit=row_limit,
            timeout_sec=timeout_sec
        )
        
        if not success:
            print(f"\n❌ Query failed: {error_msg}")
            continue
        
        # Display results
        print(f"\n✅ Query successful! ({len(results)} rows in {exec_time:.2f}s)")
        print(f"\n📊 Results:\n")
        
        if results:
            # Format as table
            table = executor.format_results_table(results)
            print(table)
            
            # Show row count info
            if len(results) == row_limit:
                print(f"\n⚠️  Showing first {row_limit} rows. More results may exist.")
        else:
            print("   No rows returned.")
    
    print(f"\n{'='*80}\n")


def main():
    """Main CLI entry point."""
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Load configuration
    config = load_config()
    
    # Initialize pipeline
    pipeline = SemanticPipeline(config)
    
    # Execute command
    if command == "cache-clear":
        logger.info("Clearing cache...")
        pipeline.invalidate_cache()
        logger.info("Cache cleared successfully")
    
    elif command == "discover":
        logger.info("Running discovery...")
        discovery = pipeline.run_discovery(force_refresh=True)
        
        # Pretty print summary
        print("\n=== DISCOVERY SUMMARY ===")
        print(f"Database: {discovery['database']['vendor']} ({discovery['database']['version']})")
        print(f"Schemas: {len(discovery['schemas'])}")
        print(f"Tables: {sum(len(s['tables']) for s in discovery['schemas'])}")
        print(f"Views: {len([a for a in discovery['named_assets'] if a['kind'] == 'view'])}")
        print(f"Stored Procedures: {len([a for a in discovery['named_assets'] if a['kind'] == 'stored_procedure'])}")
        print(f"RDL Files: {len([a for a in discovery['named_assets'] if a['kind'] == 'rdl'])}")
        print("\nDiscovery saved to cache/discovery.json")
    
    elif command == "model":
        logger.info("Running semantic modeling...")
        
        # Ensure discovery exists
        if not os.path.exists("cache/discovery.json"):
            logger.error("Discovery cache not found. Run 'python main.py discover' first.")
            sys.exit(1)
        
        pipeline.run_discovery(force_refresh=False)
        pipeline.run_semantic_relationship_extraction()
        model = pipeline.run_semantic_modeling(force_refresh=True)
        
        # Pretty print summary
        print("\n=== SEMANTIC MODEL SUMMARY ===")
        print(f"Entities: {len(model['entities'])}")
        print(f"Dimensions: {len(model['dimensions'])}")
        print(f"Facts: {len(model['facts'])}")
        print(f"Relationships: {len(model['relationships'])}")
        print(f"Metrics: {len(model['metrics'])}")
        print("\nModel saved to cache/semantic_model.json")
    
    elif command in ["question", "full"]:
        if len(sys.argv) < 3:
            logger.error("Please provide a question")
            print_usage()
            sys.exit(1)
        
        question = " ".join(sys.argv[2:])
        logger.info(f"Processing question: {question}")
        
        # Run full pipeline
        answer = pipeline.run_full_pipeline(question, force_refresh=(command == "full"))
        
        # Pretty print answer (SQL only, no execution)
        print("\n=== ANSWER ===")
        print(json.dumps(answer, indent=2))
    
    elif command == "execute":
        # NEW COMMAND: Execute SQL and show results
        if len(sys.argv) < 3:
            logger.error("Please provide a question")
            print_usage()
            sys.exit(1)
        
        question = " ".join(sys.argv[2:])
        logger.info(f"Processing question: {question}")
        
        # Run full pipeline
        answer = pipeline.run_full_pipeline(question, force_refresh=False)
        
        # Initialize SQL executor
        executor = SQLExecutor(
            connection_string=config["DATABASE_CONNECTION_STRING"],
            default_row_limit=10,  # Show 10 rows by default
            default_timeout=60
        )
        
        try:
            # Execute and display results
            execute_and_display_results(answer, executor, row_limit=10)
        finally:
            executor.close()
    
    else:
        logger.error(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()