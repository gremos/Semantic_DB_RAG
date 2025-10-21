#!/usr/bin/env python3
"""
Semantic Engine - Main Entry Point

Usage:
    python main.py discover                          # Run discovery only
    python main.py model --hints "sales, customers"  # Create semantic model
    python main.py query "find upsell opportunities" # Answer question
    python main.py full "find revenue by region"     # Full pipeline
"""

import argparse
import json
import sys
from orchestration.pipeline import SemanticPipeline
from utils.logging_config import logger

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
    
    # Full pipeline command
    parser_full = subparsers.add_parser('full', help='Run full pipeline')
    parser_full.add_argument('question', type=str, help='Natural language question')
    parser_full.add_argument('--hints', type=str, default='', help='Domain hints')
    parser_full.add_argument('--bypass-cache', action='store_true', help='Bypass cache')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
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
            # Assumes model already exists (from cache or prior run)
            success, error = pipeline.initialize()
            if not success:
                logger.error(f"Discovery failed: {error}")
                return 1
            
            success, error = pipeline.create_semantic_model()
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
    
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    sys.exit(main())