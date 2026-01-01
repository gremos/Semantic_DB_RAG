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
from src.discovery.audit_collector import AuditCollector, AuditConfig, AuditTimeRange
from src.discovery.audit_integration import AuditEnhancedDiscovery

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
    settings = get_settings()
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

        # Try to load audit report for enhanced model building
        audit_report = None
        try:
            enhanced = AuditEnhancedDiscovery()  # Uses default cache_dir from settings
            audit_report = enhanced.load_audit()
            if audit_report:
                console.print(f"[cyan]OK[/cyan] Loaded audit from: {audit_report.source_server}/{audit_report.database_name}")
                console.print(f"  Tables with metrics: {len(audit_report.table_metrics)}")
        except Exception as e:
            console.print(f"[yellow]![/yellow] No audit data available: {e}")
            console.print("  Run 'python main.py audit' to collect production metrics")

        # Build model
        console.print("[yellow]Loading discovery cache...[/yellow]")
        builder = SemanticModelBuilder(audit_report=audit_report)

        # Build semantic model
        semantic_model = builder.build(
            discovery_data=discovery_data,
            use_cache=not getattr(args, 'no_cache', False)
        )
        
        console.print("[green]OK[/green] Semantic model built successfully!")

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
        
        print(f"\n[OK] Semantic model saved to: {output_path}")

        if discovery_config.sample_mode_enabled:
            print(f"\n[Note] This model is based on SAMPLE discovery data")
            print(f"   For production use, run full discovery first")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Discovery file not found: {e}")
        print(f"\n[ERROR] Discovery data not found at {discovery_path}")
        print("Please run discovery first: python main.py discovery")
        return 1

    except Exception as e:
        logger.error(f"Semantic model generation failed: {e}", exc_info=True)
        print(f"\n[ERROR] Semantic model generation failed: {e}")
        return 1


def cmd_question(args):
    """
    Phase 3: Question Answering
    Answer natural language questions with SQL
    """
    from src.qa import QuestionParser

    logger.info(f"Processing question: {args.question}")
    console = Console()

    try:
        # Initialize parser (loads semantic model)
        parser = QuestionParser()

        # Get options from args
        execute = not getattr(args, 'no_execute', False)
        row_limit = getattr(args, 'limit', 100)
        timeout = getattr(args, 'timeout', 60)
        output_format = getattr(args, 'format', 'text')

        # Answer the question
        response = parser.answer(
            args.question,
            execute=execute,
            row_limit=row_limit,
            timeout_sec=timeout
        )

        # Format and display response
        if output_format == 'json':
            print(json.dumps(response, indent=2, default=str))
        else:
            # Rich formatted output
            status = response.get('status', 'unknown')
            confidence = response.get('confidence', 0)

            if status == 'ok':
                console.print(Panel(
                    f"[bold green]Q:[/bold green] {args.question}\n\n"
                    f"[bold]Confidence:[/bold] {confidence:.0%}",
                    title="Question Answered",
                    border_style="green"
                ))

                # Show SQL
                sql_info = response.get('sql', {})
                if sql_info.get('statement'):
                    console.print("\n[bold cyan]Generated SQL:[/bold cyan]")
                    console.print(sql_info['statement'])

                    if sql_info.get('explanation'):
                        console.print(f"\n[dim]{sql_info['explanation']}[/dim]")

                # Show results
                results = response.get('results', {})
                if results and results.get('status') == 'ok':
                    console.print(f"\n[bold]Results:[/bold] {results.get('row_count', 0)} rows "
                                 f"({results.get('execution_time_ms', 0):.0f}ms)")

                    if results.get('data'):
                        formatted = parser.sql_executor.format_results(results, 'table')
                        console.print(formatted)

            elif status == 'refuse':
                refusal = response.get('refusal', {})
                console.print(Panel(
                    f"[bold yellow]Q:[/bold yellow] {args.question}\n\n"
                    f"[bold]Confidence:[/bold] {confidence:.0%}\n"
                    f"[bold]Reason:[/bold] {refusal.get('reason', 'Unknown')}",
                    title="Unable to Answer",
                    border_style="yellow"
                ))

                clarifications = refusal.get('clarifying_questions', [])
                if clarifications:
                    console.print("\n[bold]Clarifying questions:[/bold]")
                    for q in clarifications:
                        console.print(f"  - {q}")

            else:  # error
                console.print(Panel(
                    f"[bold red]Q:[/bold red] {args.question}\n\n"
                    f"[bold]Error:[/bold] {response.get('error', 'Unknown error')}",
                    title="Error",
                    border_style="red"
                ))

        return 0 if response.get('status') == 'ok' else 1

    except FileNotFoundError as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("Run: python main.py model")
        return 1

    except Exception as e:
        logger.error(f"Question answering failed: {e}", exc_info=True)
        console.print(f"\n[red]Question answering failed:[/red] {e}")
        return 1


def cmd_qa_history(args):
    """View or clear Q&A history"""
    from src.qa import QuestionParser

    console = Console()

    try:
        parser = QuestionParser()

        if args.clear:
            if parser.clear_history():
                console.print("[green]Q&A history cleared[/green]")
            else:
                console.print("[red]Failed to clear history[/red]")
            return 0

        # Get history
        history = parser.get_history(limit=args.limit)

        if not history:
            console.print("[yellow]No Q&A history found[/yellow]")
            return 0

        console.print(f"\n[bold]Q&A History[/bold] (last {len(history)} entries)\n")
        console.print("-" * 80)

        for entry in history:
            timestamp = entry.get('timestamp', '')[:19]  # Trim to datetime
            status = entry.get('status', 'unknown')
            confidence = entry.get('confidence', 0)
            question = entry.get('question', '')[:60]

            # Status indicator
            if status == 'ok':
                status_icon = "[green]OK[/green]"
            elif status == 'refuse':
                status_icon = "[yellow]REFUSE[/yellow]"
            else:
                status_icon = "[red]ERROR[/red]"

            console.print(f"{timestamp} | {status_icon} | {confidence:.0%} | {question}")

            # Show error/refusal reason if applicable
            if entry.get('error'):
                console.print(f"  [red]Error: {entry['error'][:50]}[/red]")
            elif entry.get('refusal_reason'):
                console.print(f"  [yellow]Reason: {entry['refusal_reason'][:50]}[/yellow]")

        console.print("-" * 80)
        return 0

    except FileNotFoundError:
        console.print("[yellow]Semantic model not found - no history available[/yellow]")
        return 0

    except Exception as e:
        logger.error(f"Q&A history failed: {e}", exc_info=True)
        console.print(f"\n[red]Q&A history failed:[/red] {e}")
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


def cmd_audit(args):
    """
    Collect access audit from production database

    Collects table access patterns and query statistics
    to enhance discovery with usage-based prioritization.
    """
    from datetime import datetime

    logger.info("Starting Production Database Audit Collection")

    try:
        settings = get_settings()

        # Build time range from arguments
        time_range = None
        if args.start_date or args.end_date or args.lookback_days:
            time_range = AuditTimeRange(
                start_date=datetime.fromisoformat(args.start_date) if args.start_date else None,
                end_date=datetime.fromisoformat(args.end_date) if args.end_date else None,
                lookback_days=args.lookback_days or 30
            )

        # Initialize collector
        config = AuditConfig.from_env()
        collector = AuditCollector(config, settings.paths.cache_dir)

        # Check for cached report if not forcing
        if not args.no_cache and not args.force:
            cached = collector.load_cached_report()
            if cached:
                print("\n✓ Using cached audit report")
                print(f"  Source:      {cached.source_server}")
                print(f"  Collected:   {cached.collected_at}")
                print(f"  Period:      {cached.audit_start_date} to {cached.audit_end_date}")
                print(f"  Tables:      {cached.total_tables_analyzed}")
                print(f"  Hot:         {cached.hot_tables_count}")
                print(f"  To skip:     {len(cached.tables_to_skip)}")
                print(f"\nUse --no-cache to collect fresh audit")
                return 0

        # Collect audit
        report = collector.collect(force=args.force, time_range=time_range)

        # Print summary
        print("\n" + "=" * 60)
        print("AUDIT COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Source:        {report.source_server} / {report.database_name}")
        print(f"Period:        {report.audit_start_date} to {report.audit_end_date}")
        print(f"Duration:      {report.collection_duration_seconds:.1f}s")
        print()
        print("TABLE ACCESS PATTERNS:")
        print(f"  Hot:         {report.hot_tables_count:>5} tables")
        print(f"  Warm:        {report.warm_tables_count:>5} tables")
        print(f"  Cold:        {report.cold_tables_count:>5} tables")
        print(f"  Unused:      {report.unused_tables_count:>5} tables")
        print()
        print("HISTORY DETECTION:")
        print(f"  History:     {report.likely_history_tables_count:>5} tables detected")
        print()
        print("RECOMMENDATIONS:")
        print(f"  Prioritize:  {len(report.tables_to_prioritize):>5} tables")
        print(f"  Skip:        {len(report.tables_to_skip):>5} tables")
        print()

        if report.join_frequency:
            print("TOP 5 JOIN PATTERNS:")
            for join, count in list(report.join_frequency.items())[:5]:
                print(f"  {count:>8}x  {join}")
            print()

        print(f"Report saved to: {collector.audit_file}")
        print("=" * 60)

        return 0

    except PermissionError as e:
        print(f"\n❌ Permission denied: {e}")
        print("\nRequired permissions on production database:")
        print("  GRANT VIEW SERVER STATE TO [audit_user];")
        print("  GRANT VIEW DATABASE STATE TO [audit_user];")
        return 1

    except RuntimeError as e:
        print(f"\n❌ {e}")
        return 1

    except Exception as e:
        logger.error(f"Audit collection failed: {e}", exc_info=True)
        print(f"\n❌ Audit collection failed: {e}")
        return 1


def cmd_audit_status(args):
    """Show current audit report status"""
    try:
        settings = get_settings()
        config = AuditConfig.from_env()
        collector = AuditCollector(config, settings.paths.cache_dir)

        if not collector.audit_file.exists():
            print("\n⚠️  No audit report found")
            print("Run: python main.py audit")
            return 1

        # Use the integration class with mapping support
        enhanced = AuditEnhancedDiscovery()

        # Load with mapping if --use-mapping is set
        if args.use_mapping:
            report = enhanced.load_audit_with_mapping()
        else:
            report = enhanced.load_audit()

        if not report:
            print("\n⚠️  Audit report expired or invalid")
            print("Run: python main.py audit --no-cache")
            return 1

        # Print summary (includes mapping info)
        enhanced.print_summary()

        # Show history mappings if requested
        if args.show_history:
            print("\nHISTORY TABLE MAPPINGS:")
            if report.history_table_mappings:
                for history, base in report.history_table_mappings.items():
                    print(f"  {history} -> {base}")
            else:
                print("  (none detected)")

        # Show tables to skip if requested
        if args.show_skip:
            print("\nTABLES TO SKIP:")
            for table in report.tables_to_skip[:20]:
                print(f"  {table}")
            if len(report.tables_to_skip) > 20:
                print(f"  ... and {len(report.tables_to_skip) - 20} more")

        return 0

    except Exception as e:
        logger.error(f"Audit status failed: {e}", exc_info=True)
        print(f"\n❌ Audit status failed: {e}")
        return 1


def cmd_db_mappings(args):
    """Show or manage database mappings between production and development"""
    try:
        from config.settings import get_database_mappings

        mappings_config = get_database_mappings()
        mappings = mappings_config.list_mappings()

        print("\n" + "=" * 60)
        print("DATABASE MAPPINGS (Production -> Development)")
        print("=" * 60)

        if not mappings:
            print("\nNo database mappings configured.")
            print("\nTo add mappings, set DATABASE_MAPPINGS in .env:")
            print("  DATABASE_MAPPINGS=prod_server/prod_db:dev_server/dev_db")
            print("\nExample:")
            print("  DATABASE_MAPPINGS=cardinal01/YPS:cardinal03/YPS2907")
            return 0

        print("\nConfigured mappings:")
        for i, m in enumerate(mappings, 1):
            print(f"  {i}. {m['production']} -> {m['development']}")

        # Show current audit info if available
        enhanced = AuditEnhancedDiscovery()
        report = enhanced.load_audit()

        if report:
            print(f"\nCurrent audit source: {report.source_server} / {report.database_name}")

            # Check if audit matches any production database
            for m in mappings_config.mappings:
                if m.matches_prod(report.source_server, report.database_name):
                    print(f"  -> Maps to development: {m.dev_server} / {m.dev_database}")
                    break
            else:
                print("  -> No mapping found for this audit source")

        print("\n" + "=" * 60)

        # Show usage hints
        print("\nUsage:")
        print("  1. Configure mapping in .env:")
        print("     DATABASE_MAPPINGS=cardinal01/YPS:cardinal03/YPS2907")
        print()
        print("  2. Collect audit from production:")
        print("     python main.py audit --lookback-days 30")
        print()
        print("  3. View audit status with mapping validation:")
        print("     python main.py audit-status --use-mapping")
        print()
        print("  4. The discovery command will automatically use audit data")
        print("     when running on the mapped development database.")

        return 0

    except Exception as e:
        logger.error(f"Database mappings failed: {e}", exc_info=True)
        print(f"\n❌ Database mappings failed: {e}")
        return 1


def cmd_extract_templates(args):
    """
    Extract SQL templates from Views and RDL files.

    Creates sql_templates.json with:
    - Verified SQL from production views
    - SQL from RDL report datasets
    - Keyword index for template matching
    - Column index for business term -> column mapping
    """
    logger.info("Extracting SQL templates from Views and RDL")
    console = Console()

    try:
        from src.semantic.sql_templates import extract_and_save_templates

        templates = extract_and_save_templates()

        console.print("\n[bold green]✓[/bold green] SQL templates extracted!")
        console.print(f"  View templates:  {templates['metadata']['view_templates']}")
        console.print(f"  RDL templates:   {templates['metadata']['rdl_templates']}")
        console.print(f"  Total templates: {templates['metadata']['total_templates']}")
        console.print(f"  Keywords:        {templates['metadata']['total_keywords']}")
        console.print(f"  Column terms:    {templates['metadata'].get('total_column_terms', 0)}")

        settings = get_settings()
        output_path = settings.paths.cache_dir / 'sql_templates.json'
        console.print(f"\nSaved to: {output_path}")

        return 0

    except FileNotFoundError as e:
        console.print(f"\n[red]Error:[/red] Discovery data not found")
        console.print("Run: python main.py discovery")
        return 1

    except Exception as e:
        logger.error(f"Template extraction failed: {e}", exc_info=True)
        console.print(f"\n[red]Template extraction failed:[/red] {e}")
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

  # Production database audit
  python main.py audit                              # Collect with default 30-day lookback
  python main.py audit --lookback-days 60           # Last 60 days
  python main.py audit --start-date 2025-01-01      # From specific date
  python main.py audit --start-date 2025-01-01 --end-date 2025-06-30  # Date range
  python main.py audit --force                      # Bypass execution window
  python main.py audit-status                       # Show audit summary
  python main.py audit-status --show-history        # Show history table mappings
  python main.py audit-status --use-mapping         # Validate with database mapping

  # Database mapping (production audit -> development discovery)
  python main.py db-mappings                        # Show configured mappings

  # Extract SQL templates (for accurate Q&A)
  python main.py extract-templates                  # Extract SQL templates + column index

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
    question_parser.add_argument(
        '--no-execute',
        action='store_true',
        help='Generate SQL but do not execute (explain mode)'
    )
    question_parser.add_argument(
        '--limit',
        type=int,
        default=100,
        metavar='N',
        help='Maximum rows to return (default: 100)'
    )
    question_parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        metavar='SEC',
        help='Query timeout in seconds (default: 60)'
    )
    question_parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    question_parser.set_defaults(func=cmd_question)

    # ============================================================================
    # QA-HISTORY COMMAND
    # ============================================================================
    qa_history_parser = subparsers.add_parser(
        'qa-history',
        help='View Q&A history'
    )
    qa_history_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        metavar='N',
        help='Number of entries to show (default: 20)'
    )
    qa_history_parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear Q&A history'
    )
    qa_history_parser.set_defaults(func=cmd_qa_history)
    
    # ============================================================================
    # CACHE-CLEAR COMMAND
    # ============================================================================
    cache_clear_parser = subparsers.add_parser(
        'cache-clear',
        help='Clear all caches'
    )
    cache_clear_parser.set_defaults(func=cmd_cache_clear)
    
    # ============================================================================
    # AUDIT COMMAND
    # ============================================================================
    audit_parser = subparsers.add_parser(
        'audit',
        help='Collect access audit from production database'
    )
    audit_parser.add_argument(
        '--force',
        action='store_true',
        help='Bypass execution window check'
    )
    audit_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force fresh audit collection (ignore cache)'
    )
    audit_parser.add_argument(
        '--start-date',
        type=str,
        metavar='YYYY-MM-DD',
        help='Start date for audit period (e.g., 2025-01-01)'
    )
    audit_parser.add_argument(
        '--end-date',
        type=str,
        metavar='YYYY-MM-DD',
        help='End date for audit period (e.g., 2025-12-31)'
    )
    audit_parser.add_argument(
        '--lookback-days',
        type=int,
        metavar='N',
        default=30,
        help='Days to look back (default: 30, ignored if --start-date set)'
    )
    audit_parser.set_defaults(func=cmd_audit)

    # ============================================================================
    # AUDIT-STATUS COMMAND
    # ============================================================================
    audit_status_parser = subparsers.add_parser(
        'audit-status',
        help='Show current audit report status and recommendations'
    )
    audit_status_parser.add_argument(
        '--show-history',
        action='store_true',
        help='Show history table mappings'
    )
    audit_status_parser.add_argument(
        '--show-skip',
        action='store_true',
        help='Show tables recommended to skip'
    )
    audit_status_parser.add_argument(
        '--use-mapping',
        action='store_true',
        help='Validate audit source against configured database mappings'
    )
    audit_status_parser.set_defaults(func=cmd_audit_status)

    # ============================================================================
    # DB-MAPPINGS COMMAND
    # ============================================================================
    db_mappings_parser = subparsers.add_parser(
        'db-mappings',
        help='Show database mappings between production and development'
    )
    db_mappings_parser.set_defaults(func=cmd_db_mappings)

    # ============================================================================
    # EXTRACT-TEMPLATES COMMAND
    # ============================================================================
    templates_parser = subparsers.add_parser(
        'extract-templates',
        help='Extract SQL templates from Views and RDL files'
    )
    templates_parser.set_defaults(func=cmd_extract_templates)

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