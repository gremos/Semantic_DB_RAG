#!/usr/bin/env python3
"""
Semantic Engine - Main Entry Point

Usage:
    python main.py discover [--bypass-cache]               # Run discovery only
    python main.py model [--hints "..."] [--bypass-cache]  # Create semantic model
    python main.py query "question" [--bypass-cache]       # Answer question (uses cache)
    python main.py full "question" [--hints "..."]         # Full pipeline
    python main.py cache-info                              # Show cache status
    python main.py cache-clear                             # Clear all caches
"""

import argparse
import json
import sys
import os
from orchestration.pipeline import SemanticPipeline
from utils.logging_config import logger
from caching.cache_manager import CacheManager
from config.settings import settings
from export.powerbi_exporter import PowerBIExporter
from export.sql_exporter import SQLExporter
from export.json_exporter import JSONExporter
from export.markdown_exporter import MarkdownExporter

def show_cache_info():
    """Display cache status information."""
    cache_dir = ".cache"
    
    if not os.path.exists(cache_dir):
        print("❌ No cache directory found")
        return
    
    print("=" * 60)
    print("CACHE STATUS")
    print("=" * 60)
    
    # Check discovery cache
    discovery_cache_file = os.path.join(cache_dir, "discovery_cache.pkl")
    if os.path.exists(discovery_cache_file):
        size = os.path.getsize(discovery_cache_file)
        print(f"✅ Discovery Cache: {size:,} bytes")
        print(f"   TTL: {settings.discovery_cache_hours} hours")
    else:
        print("❌ Discovery Cache: Not found")
    
    # Check semantic model cache
    semantic_cache_file = os.path.join(cache_dir, "semantic_cache.pkl")
    if os.path.exists(semantic_cache_file):
        size = os.path.getsize(semantic_cache_file)
        print(f"✅ Semantic Model Cache: {size:,} bytes")
        print(f"   TTL: {settings.semantic_cache_hours} hours")
    else:
        print("❌ Semantic Model Cache: Not found")
    
    print("=" * 60)
    print("\nTo clear cache: python main.py cache-clear")
    print("To force regeneration: python main.py query '...' --bypass-cache")
    print("=" * 60)

def clear_cache():
    """Clear all caches."""
    cache = CacheManager(
        settings.discovery_cache_hours,
        settings.semantic_cache_hours
    )
    cache.clear_all()
    print("✅ All caches cleared")

def main():
    parser = argparse.ArgumentParser(description="Semantic Engine CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Discovery command
    parser_discover = subparsers.add_parser('discover', help='Run discovery phase')
    parser_discover.add_argument('--bypass-cache', action='store_true', help='Bypass cache')
    
    # Model command
    parser_model = subparsers.add_parser('model', help='Create semantic model')
    parser_model.add_argument('--hints', type=str, default='', help='Domain hints')
    parser_model.add_argument('--bypass-cache', action='store_true', help='Bypass cache')
    
    # Query command
    parser_query = subparsers.add_parser('query', help='Answer question')
    parser_query.add_argument('question', type=str, help='Natural language question')
    parser_query.add_argument('--bypass-cache', action='store_true', help='Force regenerate model')
    
    # Full pipeline command
    parser_full = subparsers.add_parser('full', help='Run full pipeline')
    parser_full.add_argument('question', type=str, help='Natural language question')
    parser_full.add_argument('--hints', type=str, default='', help='Domain hints')
    parser_full.add_argument('--bypass-cache', action='store_true', help='Bypass cache')

    parser_export = subparsers.add_parser('export', help='Export semantic model')
    parser_export.add_argument('format', choices=['powerbi', 'sql', 'json', 'markdown', 'all'], 
                            help='Export format')
    parser_export.add_argument('--output', type=str, help='Output filename')
    
    # Cache info command
    parser_cache_info = subparsers.add_parser('cache-info', help='Show cache status')
    
    # Cache clear command
    parser_cache_clear = subparsers.add_parser('cache-clear', help='Clear all caches')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle cache commands
    if args.command == 'cache-info':
        show_cache_info()
        return 0
    
    if args.command == 'cache-clear':
        clear_cache()
        return 0
    
    # Regular pipeline commands
    pipeline = SemanticPipeline()
    
    try:
        if args.command == 'discover':
            success, error = pipeline.initialize(args.bypass_cache)
            if not success:
                logger.error(f"Discovery failed: {error}")
                return 1
            
            print(json.dumps(pipeline.get_discovery_data(), indent=2))
            return 0
        
        elif args.command == 'model':
            # Run discovery first
            success, error = pipeline.initialize()
            if not success:
                logger.error(f"Discovery failed: {error}")
                return 1
            
            # Create model
            success, error = pipeline.create_semantic_model(args.hints, args.bypass_cache)
            if not success:
                logger.error(f"Modeling failed: {error}")
                return 1
            
            print(json.dumps(pipeline.get_semantic_model(), indent=2))
            return 0
        
        elif args.command == 'query':
            # Use cached model if available
            bypass = args.bypass_cache if hasattr(args, 'bypass_cache') else False
            
            logger.info("Loading discovery and model (will use cache if available)...")
            success, error = pipeline.initialize(bypass_cache=False)  # Use cache
            if not success:
                logger.error(f"Discovery failed: {error}")
                return 1
            
            success, error = pipeline.create_semantic_model(bypass_cache=bypass)
            if not success:
                logger.error(f"Modeling failed: {error}")
                return 1
            
            success, answer, error = pipeline.answer_question(args.question)
            if not success:
                logger.error(f"Q&A failed: {error}")
                return 1
            
            print(json.dumps(answer, indent=2))
            return 0
        
        elif args.command == 'full':
            # Full pipeline
            logger.info("=== Phase 1: Discovery ===")
            success, error = pipeline.initialize(args.bypass_cache)
            if not success:
                logger.error(f"Discovery failed: {error}")
                return 1
            
            logger.info("=== Phase 2: Semantic Modeling ===")
            success, error = pipeline.create_semantic_model(args.hints, args.bypass_cache)
            if not success:
                logger.error(f"Modeling failed: {error}")
                return 1
            
            logger.info("=== Phase 3: Question Answering ===")
            success, answer, error = pipeline.answer_question(args.question)
            if not success:
                logger.error(f"Q&A failed: {error}")
                return 1
            
            # Pretty output
            print("\n" + "="*80)
            print("ANSWER")
            print("="*80)
            print(json.dumps(answer, indent=2))
            
            if answer.get("status") == "ok":
                print("\n" + "="*80)
                print("EXECUTABLE SQL")
                print("="*80)
                for i, sql_obj in enumerate(answer.get("sql", []), 1):
                    print(f"\n-- Query {i}: {sql_obj.get('explanation', '')}")
                    print(sql_obj.get("statement"))

                    
            return 0
            
        elif args.command == 'export':
                # Load cached model
                success, error = pipeline.initialize(bypass_cache=False)
                if not success:
                    logger.error(f"Discovery failed: {error}")
                    return 1
                
                success, error = pipeline.create_semantic_model(bypass_cache=False)
                if not success:
                    logger.error(f"Modeling failed: {error}")
                    return 1
                
                model = pipeline.get_semantic_model()
                
                # Export in requested format(s)
                if args.format == 'powerbi' or args.format == 'all':
                    output = args.output or 'semantic_model.bim'
                    PowerBIExporter.export(model, output)
                    print(f"✅ Power BI model: {output}")
                
                if args.format == 'sql' or args.format == 'all':
                    output = args.output or 'semantic_views.sql'
                    SQLExporter.export(model, output)
                    print(f"✅ SQL views: {output}")
                
                if args.format == 'json' or args.format == 'all':
                    output = args.output or 'semantic_model_export.json'
                    JSONExporter.export(model, output)
                    print(f"✅ JSON export: {output}")
                
                if args.format == 'markdown' or args.format == 'all':
                    output = args.output or 'SEMANTIC_MODEL.md'
                    MarkdownExporter.export(model, output)
                    print(f"✅ Markdown docs: {output}")
                
                return 0


    
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    sys.exit(main())