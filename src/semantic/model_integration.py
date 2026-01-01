"""
Integration Guide for Semantic Model Builder

This module demonstrates how to integrate the semantic model builder
with the discovery engine and provides usage examples.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN INTEGRATION FUNCTION
# ============================================================================

def run_semantic_model_pipeline(
    connection_string: Optional[str] = None,
    use_discovery_cache: bool = True,
    use_semantic_cache: bool = True,
    skip_relationships: bool = False,
    use_audit: bool = True
) -> Dict[str, Any]:
    """
    Run complete pipeline: Discovery -> Semantic Model with Audit Integration

    Args:
        connection_string: Database connection string (uses config if not provided)
        use_discovery_cache: Use cached discovery if valid
        use_semantic_cache: Use cached semantic model if valid
        skip_relationships: Skip relationship detection in discovery
        use_audit: Load and integrate audit data for enhanced ranking/relationships

    Returns:
        Semantic model dictionary
    """
    from src.discovery.engine import run_discovery
    from model_builder import build_semantic_model

    logger.info("=" * 80)
    logger.info("RUNNING FULL SEMANTIC MODEL PIPELINE")
    if use_audit:
        logger.info("(with audit integration enabled)")
    logger.info("=" * 80)

    # Phase 0: Load Audit (optional)
    audit_report = None
    if use_audit:
        logger.info("\n[PHASE 0] Loading Audit Data...")
        try:
            from src.discovery.audit_integration import AuditEnhancedDiscovery
            enhanced = AuditEnhancedDiscovery()
            audit_report = enhanced.load_audit_with_mapping()

            if audit_report:
                logger.info(f"✓ Audit loaded: {audit_report.source_server}/{audit_report.database_name}")
                logger.info(f"  - Hot tables: {audit_report.hot_tables_count}")
                logger.info(f"  - Warm tables: {audit_report.warm_tables_count}")
                logger.info(f"  - Join patterns: {len(audit_report.join_frequency)}")
            else:
                logger.info("  No audit data available - continuing without")
        except Exception as e:
            logger.warning(f"  Could not load audit data: {e}")
            audit_report = None

    # Phase 1: Discovery
    logger.info("\n[PHASE 1] Running Discovery...")
    discovery_data = run_discovery(
        use_cache=use_discovery_cache,
        skip_relationships=skip_relationships,
        connection_string=connection_string
    )

    logger.info(f"✓ Discovery complete: {discovery_data['metadata']['total_tables']} tables")

    # Phase 2: Semantic Model (with audit integration)
    logger.info("\n[PHASE 2] Building Semantic Model...")
    semantic_model = build_semantic_model(
        discovery_data=discovery_data,
        audit_report=audit_report,
        use_cache=use_semantic_cache,
        use_audit=False  # Already loaded above
    )

    logger.info(f"✓ Semantic model complete:")
    logger.info(f"  - Entities: {len(semantic_model['entities'])}")
    logger.info(f"  - Dimensions: {len(semantic_model['dimensions'])}")
    logger.info(f"  - Facts: {len(semantic_model['facts'])}")
    logger.info(f"  - Relationships: {len(semantic_model['relationships'])}")

    # Log audit integration results
    if audit_report and semantic_model.get('audit', {}).get('production_audit'):
        audit_info = semantic_model['audit']['production_audit']
        logger.info(f"  - Audit integrated: {audit_info.get('table_metrics_count', 0)} metrics, "
                    f"{audit_info.get('join_patterns_count', 0)} join patterns")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)

    return semantic_model


# ============================================================================
# CLI INTEGRATION
# ============================================================================

def add_semantic_model_commands(cli_parser):
    """
    Add semantic model commands to CLI parser
    
    Usage in main.py:
    
    ```python
    import argparse
    from integration import add_semantic_model_commands
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # Add existing commands (discovery, question, etc.)
    discovery_parser = subparsers.add_parser('discovery')
    # ...
    
    # Add semantic model commands
    add_semantic_model_commands(subparsers)
    ```
    """
    
    # model command
    model_parser = cli_parser.add_parser(
        'model',
        help='Build semantic model from discovery cache'
    )
    model_parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force rebuild even if cache is valid'
    )
    model_parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: cache/semantic_model.json)'
    )
    
    # full-pipeline command
    pipeline_parser = cli_parser.add_parser(
        'full-pipeline',
        help='Run complete pipeline: discovery -> semantic model'
    )
    pipeline_parser.add_argument(
        '--skip-discovery-cache',
        action='store_true',
        help='Force fresh discovery'
    )
    pipeline_parser.add_argument(
        '--skip-semantic-cache',
        action='store_true',
        help='Force fresh semantic model'
    )
    pipeline_parser.add_argument(
        '--skip-relationships',
        action='store_true',
        help='Skip relationship detection'
    )
    pipeline_parser.add_argument(
        '--no-audit',
        action='store_true',
        help='Skip audit data integration'
    )


def handle_semantic_model_command(args):
    """
    Handle semantic model CLI commands

    Usage in main.py:

    ```python
    args = parser.parse_args()

    if args.command == 'model':
        handle_semantic_model_command(args)
    elif args.command == 'full-pipeline':
        handle_semantic_model_command(args)
    ```
    """
    if args.command == 'model':
        from model_builder import build_semantic_model

        # Check for --no-audit flag
        use_audit = not getattr(args, 'no_audit', False)

        logger.info("Building semantic model...")
        if use_audit:
            logger.info("(with audit integration)")

        semantic_model = build_semantic_model(
            use_cache=not args.force_refresh,
            use_audit=use_audit
        )

        # Save to output file if specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(semantic_model, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved semantic model to {output_path}")

        print("\n Semantic model built successfully")
        print(f"  Entities: {len(semantic_model['entities'])}")
        print(f"  Dimensions: {len(semantic_model['dimensions'])}")
        print(f"  Facts: {len(semantic_model['facts'])}")

        # Show audit integration status
        if semantic_model.get('audit', {}).get('production_audit'):
            audit_info = semantic_model['audit']['production_audit']
            print(f"  Audit: {audit_info.get('table_metrics_count', 0)} metrics integrated")
        else:
            print("  Audit: Not integrated")

    elif args.command == 'full-pipeline':
        use_audit = not getattr(args, 'no_audit', False)

        semantic_model = run_semantic_model_pipeline(
            use_discovery_cache=not args.skip_discovery_cache,
            use_semantic_cache=not args.skip_semantic_cache,
            skip_relationships=args.skip_relationships,
            use_audit=use_audit
        )

        print("\n Full pipeline completed successfully")
        print(f"  Entities: {len(semantic_model['entities'])}")
        print(f"  Dimensions: {len(semantic_model['dimensions'])}")
        print(f"  Facts: {len(semantic_model['facts'])}")

        # Show audit integration status
        if semantic_model.get('audit', {}).get('production_audit'):
            audit_info = semantic_model['audit']['production_audit']
            print(f"  Audit: {audit_info.get('table_metrics_count', 0)} metrics, "
                  f"{audit_info.get('join_patterns_count', 0)} join patterns")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """
    Example: Basic usage with defaults
    """
    from model_builder import build_semantic_model
    
    # Build semantic model (uses cached discovery)
    semantic_model = build_semantic_model()
    
    # Access entities
    for entity in semantic_model['entities']:
        print(f"Entity: {entity['name']}")
        print(f"  Source: {entity['source']}")
        print(f"  Columns: {len(entity['columns'])}")
    
    # Access facts
    for fact in semantic_model['facts']:
        print(f"Fact: {fact['name']}")
        print(f"  Measures: {[m['name'] for m in fact['measures']]}")


def example_full_pipeline():
    """
    Example: Run full pipeline from scratch
    """
    # Run complete pipeline
    semantic_model = run_semantic_model_pipeline(
        use_discovery_cache=False,  # Force fresh discovery
        use_semantic_cache=False     # Force fresh semantic model
    )
    
    # Save to custom location
    output_path = Path('./output/semantic_model.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(semantic_model, f, indent=2)
    
    print(f"Semantic model saved to {output_path}")


def example_with_enrichment():
    """
    Example: Using semantic enrichment utilities
    """
    from model_builder import build_semantic_model
    from model_enrichment import (
        detect_duplicate_tables,
        infer_measures_from_columns,
        calculate_relationship_confidence
    )
    
    # Build model
    semantic_model = build_semantic_model()
    
    # Detect duplicates
    all_tables = []
    for entity in semantic_model['entities']:
        all_tables.append(entity)
    for dim in semantic_model['dimensions']:
        all_tables.append(dim)
    for fact in semantic_model['facts']:
        all_tables.append(fact)
    
    duplicates = detect_duplicate_tables(all_tables, threshold=0.8)
    
    if duplicates:
        print("Duplicate tables found:")
        for table1, table2, similarity in duplicates:
            print(f"  {table1} ≈ {table2} ({similarity:.2%})")
    
    # Calculate relationship confidence
    for rel in semantic_model['relationships']:
        confidence = calculate_relationship_confidence(rel, {})
        print(f"Relationship: {rel['from']} -> {rel['to']}")
        print(f"  Confidence: {confidence}")
        print(f"  Cardinality: {rel['cardinality']}")


def example_custom_llm_config():
    """
    Example: Using custom LLM configuration
    """
    import os
    from model_builder import SemanticModelBuilder
    
    # Set custom environment variables
    os.environ['DEPLOYMENT_NAME'] = 'gpt-4'
    os.environ['ENTITY_BATCH_SIZE'] = '5'
    os.environ['DIMENSION_BATCH_SIZE'] = '5'
    os.environ['FACT_BATCH_SIZE'] = '2'
    
    # Build with custom config
    builder = SemanticModelBuilder()
    
    # Load discovery data
    from config.settings import get_path_config
    path_config = get_path_config()
    
    with open(path_config.cache_dir / 'discovery.json', 'r') as f:
        discovery_data = json.load(f)
    
    # Build
    semantic_model = builder.build(discovery_data, use_cache=False)
    
    print(f"Built with batch sizes: E={builder.entity_batch_size}, "
          f"D={builder.dimension_batch_size}, F={builder.fact_batch_size}")


# ============================================================================
# VALIDATION AND DEBUGGING
# ============================================================================

def validate_semantic_model(semantic_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate semantic model for common issues
    
    Returns:
        Validation report with warnings and errors
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check for empty collections
    if not semantic_model.get('entities') and not semantic_model.get('dimensions'):
        report['warnings'].append("No entities or dimensions defined")
    
    if not semantic_model.get('facts'):
        report['warnings'].append("No fact tables defined")
    
    # Check relationships
    entity_names = {e['name'] for e in semantic_model.get('entities', [])}
    dimension_names = {d['name'] for d in semantic_model.get('dimensions', [])}
    fact_names = {f['name'] for f in semantic_model.get('facts', [])}
    all_names = entity_names | dimension_names | fact_names
    
    for rel in semantic_model.get('relationships', []):
        from_obj = rel['from'].split('.')[0]
        to_obj = rel['to'].split('.')[0]
        
        if from_obj not in all_names:
            report['errors'].append(f"Relationship references unknown object: {from_obj}")
            report['valid'] = False
        
        if to_obj not in all_names:
            report['errors'].append(f"Relationship references unknown object: {to_obj}")
            report['valid'] = False
    
    # Check for missing primary keys
    for entity in semantic_model.get('entities', []):
        if not entity.get('primary_key'):
            report['warnings'].append(f"Entity {entity['name']} has no primary key")
    
    # Check for facts without measures
    for fact in semantic_model.get('facts', []):
        if not fact.get('measures'):
            report['warnings'].append(f"Fact {fact['name']} has no measures defined")
    
    return report


def debug_semantic_model(semantic_model: Dict[str, Any]):
    """
    Print detailed debug information about semantic model
    """
    print("\n" + "=" * 80)
    print("SEMANTIC MODEL DEBUG INFO")
    print("=" * 80)
    
    # Summary
    print("\nSummary:")
    print(f"  Entities: {len(semantic_model.get('entities', []))}")
    print(f"  Dimensions: {len(semantic_model.get('dimensions', []))}")
    print(f"  Facts: {len(semantic_model.get('facts', []))}")
    print(f"  Relationships: {len(semantic_model.get('relationships', []))}")
    print(f"  Table Rankings: {len(semantic_model.get('table_rankings', []))}")
    
    # Entities detail
    print("\nEntities:")
    for entity in semantic_model.get('entities', []):
        print(f"  • {entity['name']} ({entity['source']})")
        print(f"    PK: {entity.get('primary_key', [])}")
        print(f"    Columns: {len(entity.get('columns', []))}")
    
    # Dimensions detail
    print("\nDimensions:")
    for dim in semantic_model.get('dimensions', []):
        print(f"  • {dim['name']} ({dim['source']})")
        print(f"    Keys: {dim.get('keys', [])}")
        print(f"    Attributes: {len(dim.get('attributes', []))}")
    
    # Facts detail
    print("\nFacts:")
    for fact in semantic_model.get('facts', []):
        print(f"  • {fact['name']} ({fact['source']})")
        print(f"    Grain: {fact.get('grain', [])}")
        print(f"    Measures: {[m['name'] for m in fact.get('measures', [])]}")
        print(f"    FKs: {len(fact.get('foreign_keys', []))}")
    
    # Relationships detail
    print("\nRelationships:")
    for rel in semantic_model.get('relationships', []):
        print(f"  • {rel['from']} -> {rel['to']}")
        print(f"    Cardinality: {rel.get('cardinality')}")
        print(f"    Confidence: {rel.get('confidence')}")
    
    # Validation
    print("\nValidation:")
    validation = validate_semantic_model(semantic_model)
    print(f"  Valid: {validation['valid']}")
    if validation['warnings']:
        print(f"  Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings'][:5]:
            print(f"    - {warning}")
    if validation['errors']:
        print(f"  Errors: {len(validation['errors'])}")
        for error in validation['errors'][:5]:
            print(f"    - {error}")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN ENTRY POINT FOR TESTING
# ============================================================================

if __name__ == '__main__':
    """
    Test integration module
    
    Usage:
        python integration.py
    """
    import sys
    from pathlib import Path
    
    # Setup logging
    from config.settings import get_settings
    get_settings() 
    
    print("Semantic Model Builder - Integration Module")
    print("=" * 80)
    
    # Check if discovery cache exists
    from config.settings import get_path_config
    path_config = get_path_config()
    discovery_file = path_config.cache_dir / 'discovery.json'
    
    if not discovery_file.exists():
        print("\n❌ Discovery cache not found!")
        print("   Run discovery first: python main.py discovery")
        sys.exit(1)
    
    print("\n✓ Discovery cache found")
    
    # Run semantic model
    print("\n[1] Building semantic model...")
    try:
        from model_builder import build_semantic_model
        semantic_model = build_semantic_model(use_cache=False)
        
        print("✓ Semantic model built")
        
        # Debug output
        debug_semantic_model(semantic_model)
        
        # Validation
        print("\n[2] Validating model...")
        validation = validate_semantic_model(semantic_model)
        
        if validation['valid']:
            print("✓ Model is valid")
        else:
            print("❌ Model has errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        if validation['warnings']:
            print(f"\n⚠️  {len(validation['warnings'])} warnings:")
            for warning in validation['warnings'][:10]:
                print(f"  - {warning}")
        
        print("\n" + "=" * 80)
        print("Integration test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)