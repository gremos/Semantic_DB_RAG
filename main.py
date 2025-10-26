"""
Main entry point for GPT-5 Semantic Modeling & SQL Q&A System
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

# Import pipeline
from orchestration.pipeline import SemanticPipeline


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
    question <text>          Ask a question (runs full pipeline)
    full <text>              Run full pipeline with question
    
Examples:
    python main.py cache-clear
    python main.py discover
    python main.py model
    python main.py question "What are the total sales by customer?"
    python main.py full "Show me revenue trends by month"
""")


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
        
        # Pretty print answer
        print("\n=== ANSWER ===")
        print(json.dumps(answer, indent=2))
    
    else:
        logger.error(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()