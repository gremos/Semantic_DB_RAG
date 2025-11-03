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
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from config.settings import get_settings, get_discovery_config
from src.discovery.discovery_engine import run_discovery, clear_discovery_cache
from src.semantic.model_builder import SemanticModelBuilder

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
logger = logging.getLogger(__name__)


def cmd_discovery(args):
    """
    Phase 1: Database Discovery
    Introspects database schema, samples data, detects relationships
    """
    # Ensure logging is configured
    get_settings()
    # Handle sample mode CLI arguments (override env vars)
    if hasattr(args, 'sample_mode') and args.sample_mode:
        logger.info("=" * 80)
        logger.info("🔬 SAMPLE DISCOVERY MODE ENABLED (via CLI)")
        logger.info("=" * 80)
        
        # Set environment variables from CLI args
        os.environ['SAMPLE_MODE_ENABLED'] = 'true'
        
        if args.sample_tables:
            os.environ['SAMPLE_MAX_TABLES_PER_SCHEMA'] = str(args.sample_tables)
            logger.info(f"  • Max tables per schema: {args.sample_tables}")
        
        if args.sample_views:
            os.environ['SAMPLE_MAX_VIEWS_PER_SCHEMA'] = str(args.sample_views)
            logger.info(f"  • Max views per schema: {args.sample_views}")
        
        if args.sample_sps:
            os.environ['SAMPLE_MAX_STORED_PROCEDURES'] = str(args.sample_sps)
            logger.info(f"  • Max stored procedures: {args.sample_sps}")
        
        if args.sample_rdls:
            os.environ['SAMPLE_MAX_RDL_FILES'] = str(args.sample_rdls)
            logger.info(f"  • Max RDL files: {args.sample_rdls}")
        
        # Force reload settings to pick up new env vars
        from config.settings import get_settings
        get_settings(reload=True)
    
    logger.info("Starting Phase 1: Discovery")
    
    try:
        discovery_data = run_discovery(
            use_cache=not args.no_cache,
            skip_relationships=args.skip_relationships
        )
        
        # Get discovery config to check sample mode
        discovery_config = get_discovery_config()
        
        # Print summary
        print("\n" + "=" * 80)
        if discovery_config.sample_mode_enabled:
            print("🔬 SAMPLE DISCOVERY SUMMARY")
        else:
            print("DISCOVERY SUMMARY")
        print("=" * 80)
        print(f"Schemas:       {len(discovery_data['schemas'])}")
        print(f"Tables:        {discovery_data['metadata']['total_tables']}")
        print(f"Columns:       {discovery_data['metadata']['total_columns']}")
        print(f"Relationships: {len(discovery_data.get('inferred_relationships', []))}")
        print(f"Named Assets:  {len(discovery_data.get('named_assets', []))}")
        print(f"Duration:      {discovery_data['metadata']['discovery_duration_seconds']:.1f}s")
        print("=" * 80)
        
        # Save to output
        settings = get_settings()
        
        # Determine output filename based on sample mode
        if discovery_config.sample_mode_enabled:
            filename = discovery_config.sample_output_filename
        else:
            filename = 'discovery.json'
        
        output_path = settings.paths.cache_dir / filename
        
        # Display sample mode info if enabled
        if discovery_config.sample_mode_enabled:
            print(f"\n✅ SAMPLE Discovery complete! (saved to {filename})")
            print("\n📊 Sample Limits Applied:")
            
            if discovery_config.sample_max_tables_per_schema:
                print(f"  • Tables per schema:   {discovery_config.sample_max_tables_per_schema}")
            else:
                print(f"  • Tables per schema:   ALL")
            
            if discovery_config.sample_max_views_per_schema:
                print(f"  • Views per schema:    {discovery_config.sample_max_views_per_schema}")
            else:
                print(f"  • Views per schema:    ALL")
            
            if discovery_config.sample_max_stored_procedures:
                print(f"  • Stored procedures:   {discovery_config.sample_max_stored_procedures}")
            else:
                print(f"  • Stored procedures:   ALL")
            
            if discovery_config.sample_max_rdl_files:
                print(f"  • RDL files:           {discovery_config.sample_max_rdl_files}")
            else:
                print(f"  • RDL files:           ALL")
            
            print(f"\n💡 Tip: Use full discovery for production: python main.py discovery")
        else:
            print(f"\n✅ Discovery complete! (saved to {filename})")
        
        print(f"Results saved to: {output_path}")
        
        # Show expensive objects if any
        if discovery_data['metadata'].get('expensive_objects'):
            print("\n⚠️  Slow objects detected (>30s):")
            for obj in discovery_data['metadata']['expensive_objects'][:5]:
                print(f"  • {obj['name']}: {obj['duration']:.1f}s")
        
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
    discovery_config = get_discovery_config()
    
    # Determine which discovery file to use
    if discovery_config.sample_mode_enabled:
        discovery_filename = discovery_config.sample_output_filename
        logger.info(f"📊 Using SAMPLE discovery: {discovery_filename}")
    else:
        discovery_filename = 'discovery.json'
    
    discovery_path = settings.paths.cache_dir / discovery_filename
    output_path = settings.paths.cache_dir / 'semantic_model.json'

    console = Console()

    if not discovery_path.exists():
        print(f"\n❌ Discovery data not found at {discovery_path}")
        print(f"Please run discovery first: python main.py discovery")
        if discovery_config.sample_mode_enabled:
            print(f"Or run full discovery: python main.py discovery (without --sample-mode)")
        return 1

    try:
        # Load discovery data
        with open(discovery_path, 'r', encoding='utf-8') as f:
            discovery_data = json.load(f)

        # Build model
        console.print("[yellow]Loading discovery cache...[/yellow]")
        builder = SemanticModelBuilder()

        # Build semantic model
        semantic_model = builder.build(
            discovery_data=discovery_data, 
            use_cache=not getattr(args, 'no_cache', False)
        )
        
        console.print("[green]✓[/green] Semantic model built successfully!")

        # Persist to cache
        settings.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(semantic_model, f, indent=2, ensure_ascii=False)

        # Print summary
        print("\n" + "=" * 80)
        print("SEMANTIC MODEL SUMMARY")
        print("=" * 80)
        print(f"Entities:      {len(semantic_model.get('entities', []))}")
        print(f"Dimensions:    {len(semantic_model.get('dimensions', []))}")
        print(f"Facts:         {len(semantic_model.get('facts', []))}")
        print(f"Relationships: {len(semantic_model.get('relationships', []))}")
        print(f"Table Rankings: {len(semantic_model.get('table_rankings', []))}")
        print("=" * 80)
        
        print(f"\n✅ Semantic model saved to: {output_path}")
        
        if discovery_config.sample_mode_enabled:
            print(f"\n📊 Note: This model is based on SAMPLE discovery data")
            print(f"   For production use, run full discovery first")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Discovery file not found: {e}")
        print(f"\n❌ Discovery data not found at {discovery_path}")
        print("Please run discovery first: python main.py discovery")
        return 1
        
    except Exception as e:
        logger.error(f"Semantic model generation failed: {e}", exc_info=True)
        print(f"\n❌ Semantic model generation failed: {e}")
        return 1


def cmd_question(args):
    """
    Phase 3: Question Answering
    Answer natural language questions with SQL
    """
    logger.info(f"Processing question: {args.question}")
    
    try:
        # TODO: Implement Q&A pipeline
        # Steps:
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
        
        cleared = []
        
        # Clear discovery cache (both regular and sample)
        clear_discovery_cache()
        cleared.append("discovery.json")
        
        # Clear sample discovery if it exists
        sample_discovery = cache_dir / 'discovery-sample.json'
        if sample_discovery.exists():
            sample_discovery.unlink()
            cleared.append("discovery-sample.json")
            logger.info(f"Removed sample discovery cache: {sample_discovery}")
        
        # Clear semantic model cache
        semantic_model_path = cache_dir / 'semantic_model.json'
        if semantic_model_path.exists():
            semantic_model_path.unlink()
            cleared.append("semantic_model.json")
            logger.info(f"Removed semantic model cache: {semantic_model_path}")
        
        # Clear fingerprints
        fingerprint_path = cache_dir / 'discovery_fingerprint.json'
        if fingerprint_path.exists():
            fingerprint_path.unlink()
            cleared.append("discovery_fingerprint.json")
        
        print("\n✅ Caches cleared:")
        for item in cleared:
            print(f"  • {item}")
        
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
                    'sample_mode_enabled': settings.discovery.sample_mode_enabled,
                    'sample_max_tables_per_schema': settings.discovery.sample_max_tables_per_schema,
                    'sample_max_views_per_schema': settings.discovery.sample_max_views_per_schema,
                    'sample_max_stored_procedures': settings.discovery.sample_max_stored_procedures,
                    'sample_max_rdl_files': settings.discovery.sample_max_rdl_files,
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
            
            # Add sample mode info
            print("\nSample Mode:")
            print(f"  Enabled:           {settings.discovery.sample_mode_enabled}")
            if settings.discovery.sample_mode_enabled:
                print(f"  Output File:       {settings.discovery.sample_output_filename}")
                print(f"  Tables/Schema:     {settings.discovery.sample_max_tables_per_schema or 'ALL'}")
                print(f"  Views/Schema:      {settings.discovery.sample_max_views_per_schema or 'ALL'}")
                print(f"  Stored Procedures: {settings.discovery.sample_max_stored_procedures or 'ALL'}")
                print(f"  RDL Files:         {settings.discovery.sample_max_rdl_files or 'ALL'}")
        
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
  
  # Sample mode for testing
  python main.py discovery --sample-mode --sample-tables 5
  python main.py discovery --sample-mode --sample-tables 3 --sample-views 2
  
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
    
    # ============================================================================
    # DISCOVERY COMMAND
    # ============================================================================
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
    
    # Sample mode arguments
    discovery_parser.add_argument(
        '--sample-mode',
        action='store_true',
        help='Enable sample mode (discover limited number of objects for testing)'
    )
    discovery_parser.add_argument(
        '--sample-tables',
        type=int,
        metavar='N',
        help='Max tables per schema to discover (sample mode)'
    )
    discovery_parser.add_argument(
        '--sample-views',
        type=int,
        metavar='N',
        help='Max views per schema to discover (sample mode)'
    )
    discovery_parser.add_argument(
        '--sample-sps',
        type=int,
        metavar='N',
        help='Max stored procedures to discover (sample mode)'
    )
    discovery_parser.add_argument(
        '--sample-rdls',
        type=int,
        metavar='N',
        help='Max RDL files to discover (sample mode)'
    )
    
    discovery_parser.set_defaults(func=cmd_discovery)
    
    # ============================================================================
    # MODEL COMMAND
    # ============================================================================
    model_parser = subparsers.add_parser(
        'model',
        help='Phase 2: Build semantic model from discovery data'
    )
    model_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force rebuild semantic model (ignore cache)'
    )
    model_parser.set_defaults(func=cmd_model)
    
    # ============================================================================
    # QUESTION COMMAND
    # ============================================================================
    question_parser = subparsers.add_parser(
        'question',
        help='Phase 3: Answer natural language question with SQL'
    )
    question_parser.add_argument(
        'question',
        help='Natural language question to answer'
    )
    question_parser.set_defaults(func=cmd_question)
    
    # ============================================================================
    # CACHE-CLEAR COMMAND
    # ============================================================================
    cache_clear_parser = subparsers.add_parser(
        'cache-clear',
        help='Clear all caches'
    )
    cache_clear_parser.set_defaults(func=cmd_cache_clear)
    
    # ============================================================================
    # CONFIG COMMAND
    # ============================================================================
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