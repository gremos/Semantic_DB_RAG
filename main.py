#!/usr/bin/env python3
"""
GPT-5 Semantic Modeling & SQL Q&A System
Main CLI Entry Point

Commands:
    python main.py discovery              # Phase 1: discover & cache
    python main.py model                  # Phase 2: build semantic model
    python main.py question "..."         # Phase 3: NL → SQL
    python main.py cache-clear            # Clear all caches
    python main.py config                 # Show configuration
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from config.settings import get_settings
from src.discovery.discovery_engine import run_discovery, clear_discovery_cache
from src.semantic.model_builder import SemanticModelBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def cmd_discovery(args):
    """
    Phase 1: Database Discovery
    Introspects database schema, samples data, detects relationships
    """
    logger.info("Starting Phase 1: Discovery")
    
    try:
        discovery_data = run_discovery(
            use_cache=not args.no_cache,
            skip_relationships=args.skip_relationships
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("DISCOVERY SUMMARY")
        print("=" * 80)
        print(f"Schemas:       {len(discovery_data['schemas'])}")
        print(f"Tables:        {discovery_data['metadata']['total_tables']}")
        print(f"Columns:       {discovery_data['metadata']['total_columns']}")
        print(f"Relationships: {len(discovery_data.get('inferred_relationships', []))}")
        print(f"Duration:      {discovery_data['metadata']['discovery_duration_seconds']:.1f}s")
        print("=" * 80)
        
        # Save to output
        settings = get_settings()
        output_path = settings.paths.cache_dir / 'discovery.json'
        print(f"\n✅ Discovery complete! Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        print(f"\n❌ Discovery failed: {e}")
        return 1


def cmd_model(args):
    """
    Phase 2: Semantic Model Generation
    Builds business-friendly semantic model from discovery data
    """
    logger.info("Starting Phase 2: Semantic Model Generation")

    # Paths & prechecks
    settings = get_settings()
    discovery_path = settings.paths.cache_dir / 'discovery.json'
    output_path = settings.paths.cache_dir / 'semantic_model.json'

    console = Console()

    if not discovery_path.exists():
        print(f"\n❌ Discovery data not found at {discovery_path}")
        print("Please run discovery first: python main.py discovery")
        return 1

    try:
        # Load discovery data (optional, in case your builder uses it)
        with open(discovery_path, 'r', encoding='utf-8') as f:
            discovery_data = json.load(f)

        # Build model
        console.print("[yellow]Loading discovery cache...[/yellow]")
        builder = SemanticModelBuilder(settings)

        # Prefer passing discovery data if your builder supports it.
        # If not, keep the no-arg call.
        try:
            semantic_model = builder.build(discovery_data=discovery_data)
        except TypeError:
            # Fallback for builders that don't accept args
            semantic_model = builder.build()

        # Persist to cache
        settings.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(semantic_model, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✓[/green] Semantic model built successfully!")
        console.print(f"[green]✓[/green] Cached at: {output_path}")

        # Summary counts (robust to missing keys)
        entities = len(semantic_model.get('entities', []))
        dimensions = len(semantic_model.get('dimensions', []))
        facts = len(semantic_model.get('facts', []))
        relationships = len(semantic_model.get('relationships', []))

        console.print(Panel(
            f"[bold]Semantic Model Summary[/bold]\n\n"
            f"• Entities: {entities}\n"
            f"• Dimensions: {dimensions}\n"
            f"• Facts: {facts}\n"
            f"• Relationships: {relationships}",
            border_style="green"
        ))

        return 0

    except Exception as e:
        logger.error(f"Semantic model generation failed: {e}", exc_info=True)
        print(f"\n❌ Semantic model generation failed: {e}")
        return 1



def cmd_question(args):
    """
    Phase 3: Question Answering
    Converts natural language question to SQL and executes
    """
    logger.info(f"Starting Phase 3: Question Answering")
    logger.info(f"Question: {args.question}")
    
    # Check if semantic model exists
    settings = get_settings()
    semantic_model_path = settings.paths.cache_dir / 'semantic_model.json'
    
    if not semantic_model_path.exists():
        print(f"\n❌ Semantic model not found at {semantic_model_path}")
        print("Please run model generation first: python main.py model")
        return 1
    
    try:
        # Load semantic model
        with open(semantic_model_path, 'r', encoding='utf-8') as f:
            semantic_model = json.load(f)
        
        print(f"\n✅ Loaded semantic model")
        
        # TODO: Implement question answering
        # This would:
        # 1. Load semantic model
        # 2. Parse question
        # 3. Generate SQL using LLM with grounding
        # 4. Validate SQL
        # 5. Execute and return results
        # 6. Log to Q&A history
        
        print("\n⚠️  Question answering not yet implemented")
        print("This will be implemented in Phase 3")
        
        return 0
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}", exc_info=True)
        print(f"\n❌ Question answering failed: {e}")
        return 1


def cmd_cache_clear(args):
    """Clear all caches (discovery and semantic model)"""
    logger.info("Clearing caches...")
    
    try:
        settings = get_settings()
        cache_dir = settings.paths.cache_dir
        
        # Clear discovery cache
        clear_discovery_cache()
        
        # Clear semantic model cache
        semantic_model_path = cache_dir / 'semantic_model.json'
        if semantic_model_path.exists():
            semantic_model_path.unlink()
            logger.info(f"Removed semantic model cache: {semantic_model_path}")
        
        # Clear relationship cache
        relationships_path = cache_dir / 'relationships.json'
        if relationships_path.exists():
            relationships_path.unlink()
            logger.info(f"Removed relationships cache: {relationships_path}")
        
        print("\n✅ All caches cleared")
        return 0
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}", exc_info=True)
        print(f"\n❌ Cache clear failed: {e}")
        return 1


def cmd_config(args):
    """Show current configuration"""
    try:
        settings = get_settings()
        
        if args.json:
            # Output as JSON
            config_dict = {
                'azure_openai': {
                    'deployment_name': settings.azure_openai.deployment_name,
                    'api_version': settings.azure_openai.api_version,
                    'endpoint': settings.azure_openai.endpoint,
                },
                'database': {
                    'connection_string': settings.database.connection_string[:50] + '...',
                },
                'paths': {
                    'rdl_path': str(settings.paths.rdl_path),
                    'cache_dir': str(settings.paths.cache_dir),
                    'log_dir': str(settings.paths.log_dir),
                },
                'discovery': {
                    'timeout': settings.discovery.timeout,
                    'cache_hours': settings.discovery.cache_hours,
                    'max_workers': settings.discovery.max_workers,
                },
                'relationships': {
                    'enabled': settings.relationships.enabled,
                    'strategy': settings.relationships.strategy,
                    'sample_size': settings.relationships.sample_size,
                    'max_workers': settings.relationships.max_workers,
                    'max_comparisons': settings.relationships.max_comparisons,
                },
            }
            print(json.dumps(config_dict, indent=2, ensure_ascii=False))
        else:
            # Output as formatted text
            print(settings.summary())
        
        return 0
        
    except Exception as e:
        logger.error(f"Config display failed: {e}", exc_info=True)
        print(f"\n❌ Config display failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='GPT-5 Semantic Modeling & SQL Q&A System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discovery phase
  python main.py discovery
  python main.py discovery --no-cache
  python main.py discovery --skip-relationships
  
  # Semantic model generation
  python main.py model
  
  # Question answering
  python main.py question "What were our top 5 customers by revenue last quarter?"
  
  # Utilities
  python main.py cache-clear
  python main.py config
  python main.py config --json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Discovery command
    discovery_parser = subparsers.add_parser(
        'discovery',
        help='Phase 1: Discover database schema and relationships'
    )
    discovery_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Skip cache and run fresh discovery'
    )
    discovery_parser.add_argument(
        '--skip-relationships',
        action='store_true',
        help='Skip relationship detection (faster, for testing)'
    )
    discovery_parser.set_defaults(func=cmd_discovery)
    
    # Model command
    model_parser = subparsers.add_parser(
        'model',
        help='Phase 2: Build semantic model from discovery data'
    )
    model_parser.set_defaults(func=cmd_model)
    
    # Question command
    question_parser = subparsers.add_parser(
        'question',
        help='Phase 3: Answer natural language question with SQL'
    )
    question_parser.add_argument(
        'question',
        help='Natural language question to answer'
    )
    question_parser.set_defaults(func=cmd_question)
    
    # Cache clear command
    cache_clear_parser = subparsers.add_parser(
        'cache-clear',
        help='Clear all caches'
    )
    cache_clear_parser.set_defaults(func=cmd_cache_clear)
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Show current configuration'
    )
    config_parser.add_argument(
        '--json',
        action='store_true',
        help='Output configuration as JSON'
    )
    config_parser.set_defaults(func=cmd_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        print(f"\n❌ Command failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())