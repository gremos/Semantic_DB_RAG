#!/usr/bin/env python3
"""
GPT-5 Semantic Modeling & SQL Q&A System
Main CLI entry point
"""

import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

from config.settings import Settings
from src.utils.logging_config import setup_logging
from src.discovery.discovery_engine import DiscoveryEngine
from src.semantic.model_builder import SemanticModelBuilder
from src.qa.question_parser import QuestionParser
from src.utils.cache import CacheManager

console = Console()
logger = None


def init_app():
    """Initialize application settings and logging."""
    global logger
    settings = Settings()
    logger = setup_logging(settings)
    return settings


@click.group()
def cli():
    """GPT-5 Semantic Modeling & SQL Q&A System"""
    pass


@cli.command()
def discovery():
    """
    Phase 1: Discover database schema, samples, and relationships.
    Results are cached for use in semantic model building.
    """
    console.print(Panel.fit(
        "[bold cyan]Phase 1: Discovery[/bold cyan]\n"
        "Discovering database schema, samples, and relationships...",
        border_style="cyan"
    ))
    
    try:
        settings = init_app()
        engine = DiscoveryEngine(settings)
        
        console.print("[yellow]Starting discovery process...[/yellow]")
        discovery_result = engine.discover()
        
        console.print(f"[green]✓[/green] Discovery completed successfully!")
        console.print(f"[green]✓[/green] Found {len(discovery_result.get('schemas', []))} schemas")
        console.print(f"[green]✓[/green] Cached at: {settings.CACHE_DIR}/discovery.json")
        
        # Display summary
        total_tables = sum(len(schema.get('tables', [])) for schema in discovery_result.get('schemas', []))
        total_relationships = len(discovery_result.get('inferred_relationships', []))
        
        console.print(Panel(
            f"[bold]Discovery Summary[/bold]\n\n"
            f"• Total Tables: {total_tables}\n"
            f"• Inferred Relationships: {total_relationships}\n"
            f"• Named Assets: {len(discovery_result.get('named_assets', []))}",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]✗[/red] Discovery failed: {str(e)}")
        logger.exception("Discovery failed")
        sys.exit(1)


@cli.command()
def model():
    """
    Phase 2: Build semantic model from discovery cache.
    Creates business-friendly entities, dimensions, and facts.
    """
    console.print(Panel.fit(
        "[bold magenta]Phase 2: Semantic Model Building[/bold magenta]\n"
        "Building semantic model from discovery cache...",
        border_style="magenta"
    ))
    
    try:
        settings = init_app()
        builder = SemanticModelBuilder(settings)
        
        console.print("[yellow]Loading discovery cache...[/yellow]")
        semantic_model = builder.build()
        
        console.print(f"[green]✓[/green] Semantic model built successfully!")
        console.print(f"[green]✓[/green] Cached at: {settings.CACHE_DIR}/semantic_model.json")
        
        # Display summary
        console.print(Panel(
            f"[bold]Semantic Model Summary[/bold]\n\n"
            f"• Entities: {len(semantic_model.get('entities', []))}\n"
            f"• Dimensions: {len(semantic_model.get('dimensions', []))}\n"
            f"• Facts: {len(semantic_model.get('facts', []))}\n"
            f"• Relationships: {len(semantic_model.get('relationships', []))}",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]✗[/red] Model building failed: {str(e)}")
        logger.exception("Model building failed")
        sys.exit(1)


@cli.command()
@click.argument('question', type=str)
def question(question: str):
    """
    Phase 3: Answer a natural language question with SQL.
    Returns SQL query, results, and evidence chain.
    
    Example: python main.py question "What were top 10 customers by revenue last month?"
    """
    console.print(Panel.fit(
        f"[bold blue]Phase 3: Question Answering[/bold blue]\n"
        f"Question: {question}",
        border_style="blue"
    ))
    
    try:
        settings = init_app()
        parser = QuestionParser(settings)
        
        console.print("[yellow]Analyzing question...[/yellow]")
        answer = parser.answer(question)
        
        # Display status
        if answer['status'] == 'refuse':
            console.print(Panel(
                f"[bold red]Unable to Answer[/bold red]\n\n"
                f"Reason: {answer['refusal']['reason']}\n\n"
                f"[bold]Please clarify:[/bold]\n" +
                "\n".join(f"• {q}" for q in answer.get('clarifying_questions', [])),
                border_style="red"
            ))
            return
        
        # Display SQL
        console.print("\n[bold]Generated SQL:[/bold]")
        for i, sql_obj in enumerate(answer.get('sql', []), 1):
            console.print(Panel(
                sql_obj['statement'],
                title=f"Query {i} ({sql_obj['dialect']})",
                border_style="cyan"
            ))
            console.print(f"[dim]{sql_obj['explanation']}[/dim]\n")
        
        # Display first row interpretation
        if 'result_preview' in answer:
            preview = answer['result_preview']
            console.print(Panel(
                preview.get('first_row_meaning', 'No results'),
                title="First Row Interpretation",
                border_style="green"
            ))
            
            # Display results table
            if 'top_10_rows' in preview and preview['top_10_rows']:
                console.print(f"\n[bold]Results ({preview['rows_sampled']} rows):[/bold]")
                console.print(JSON.from_data(preview['top_10_rows'][:5]))  # Show first 5
        
        # Display suggested questions
        if 'suggested_questions' in answer and answer['suggested_questions']:
            console.print("\n[bold]Suggested follow-up questions:[/bold]")
            for i, q in enumerate(answer['suggested_questions'], 1):
                console.print(f"  {i}. {q}")
        
        console.print(f"\n[green]✓[/green] Answer logged to: {settings.LOG_DIR}/qa_history.log.jsonl")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Question answering failed: {str(e)}")
        logger.exception("Question answering failed")
        sys.exit(1)


@cli.command()
def cache_clear():
    """
    Clear discovery and semantic model caches.
    Forces fresh discovery and model building on next run.
    """
    console.print(Panel.fit(
        "[bold yellow]Cache Management[/bold yellow]\n"
        "Clearing discovery and semantic caches...",
        border_style="yellow"
    ))
    
    try:
        settings = init_app()
        cache_mgr = CacheManager(settings)
        
        cleared = cache_mgr.clear_all()
        
        if cleared:
            console.print("[green]✓[/green] All caches cleared successfully!")
            console.print("Next discovery and model build will be fresh.")
        else:
            console.print("[yellow]⚠[/yellow] No caches found to clear.")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Cache clearing failed: {str(e)}")
        sys.exit(1)


@cli.command()
def version():
    """Display version information."""
    console.print(Panel.fit(
        "[bold]GPT-5 Semantic Modeling & SQL Q&A System[/bold]\n"
        "Version: 1.0.0\n"
        "Python: 3.11+",
        border_style="blue"
    ))


if __name__ == '__main__':
    cli()