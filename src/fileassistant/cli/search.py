"""Search CLI command for finding files using natural language."""

import json
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import get_config_manager
from ..search import SearchEngine, SearchResult
from ..utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def parse_extensions(value: str) -> list[str]:
    """Parse comma-separated extensions into a list."""
    if not value:
        return []
    # Split on comma, strip whitespace, ensure dot prefix
    extensions = []
    for ext in value.split(","):
        ext = ext.strip().lower()
        if ext:
            if not ext.startswith("."):
                ext = f".{ext}"
            extensions.append(ext)
    return extensions


def parse_date(value: str) -> datetime | None:
    """Parse a date string in YYYY-MM-DD format."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter(f"Invalid date format: {value}. Use YYYY-MM-DD")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_relevance_color(score: float) -> str:
    """Get color based on relevance score."""
    if score >= 0.8:
        return "green"
    elif score >= 0.5:
        return "yellow"
    else:
        return "red"


def format_result_rich(result: SearchResult, index: int) -> Panel:
    """Format a single search result as a Rich panel."""
    # Build header with relevance score
    score_color = get_relevance_color(result.relevance_score)
    score_text = f"[{score_color}]{result.relevance_score:.1%}[/{score_color}]"

    # Filename (bold)
    header = Text()
    header.append(f"{index}. ", style="dim")
    header.append(result.filename, style="bold")
    header.append(f"  {score_text}")

    # Build content
    content = Text()

    # File path (dimmed)
    content.append(result.file_path, style="dim")
    content.append("\n")

    # File info line
    info_parts = []
    if result.extension:
        info_parts.append(f"Type: {result.extension}")
    if result.size_bytes:
        info_parts.append(f"Size: {format_file_size(result.size_bytes)}")
    if result.modified_at:
        info_parts.append(f"Modified: {result.modified_at.strftime('%Y-%m-%d')}")

    if info_parts:
        content.append(" | ".join(info_parts), style="dim")
        content.append("\n")

    # Tags as badges
    if result.tags:
        for tag in result.tags:
            content.append(f"[{tag}]", style="cyan")
            content.append(" ")
        content.append("\n")

    # Content snippet
    if result.content_snippet:
        content.append("\n")
        content.append(result.content_snippet, style="dim italic")

    return Panel(
        content,
        title=header,
        title_align="left",
        border_style=score_color,
        padding=(0, 1),
    )


def format_results_table(results: list[SearchResult]) -> Table:
    """Format results as a compact Rich table."""
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=7)
    table.add_column("Filename", style="bold")
    table.add_column("Path", style="dim", no_wrap=True, overflow="ellipsis")
    table.add_column("Type", width=6)
    table.add_column("Size", width=8)

    for i, result in enumerate(results, 1):
        score_color = get_relevance_color(result.relevance_score)
        score_text = f"[{score_color}]{result.relevance_score:.1%}[/{score_color}]"

        table.add_row(
            str(i),
            score_text,
            result.filename,
            result.file_path,
            result.extension.lstrip(".") if result.extension else "",
            format_file_size(result.size_bytes) if result.size_bytes else "",
        )

    return table


def format_results_json(results: list[SearchResult]) -> str:
    """Format results as JSON for programmatic use."""
    output = []
    for result in results:
        output.append({
            "file_path": result.file_path,
            "filename": result.filename,
            "relevance_score": result.relevance_score,
            "content_snippet": result.content_snippet,
            "tags": result.tags,
            "file_type": result.file_type,
            "extension": result.extension,
            "modified_at": result.modified_at.isoformat() if result.modified_at else None,
            "size_bytes": result.size_bytes,
        })
    return json.dumps(output, indent=2)


@click.command(name="search")
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--type", "-t",
    "file_type",
    default=None,
    help="Filter by file extension(s). Examples: --type pdf, --type py,js,ts",
)
@click.option(
    "--after",
    default=None,
    help="Only files modified after this date (YYYY-MM-DD)",
)
@click.option(
    "--before",
    default=None,
    help="Only files modified before this date (YYYY-MM-DD)",
)
@click.option(
    "--tag",
    default=None,
    help="Filter by tag",
)
@click.option(
    "--limit", "-n",
    default=10,
    show_default=True,
    help="Maximum number of results",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.option(
    "--compact",
    is_flag=True,
    help="Show compact table view instead of detailed panels",
)
@click.pass_context
def search_command(
    ctx,
    query: tuple,
    file_type: str | None,
    after: str | None,
    before: str | None,
    tag: str | None,
    limit: int,
    output_json: bool,
    compact: bool,
):
    """
    Search indexed files using natural language.

    QUERY is the search query (can be multiple words).

    Examples:

        fileassistant search machine learning papers

        fileassistant search "quarterly report" --type pdf

        fileassistant search python code --type py,js --after 2024-01-01

        fileassistant search meeting notes --tag work --limit 5
    """
    # Join query parts into single string
    query_str = " ".join(query)

    if len(query_str) < 2:
        console.print("[bold red]Error:[/bold red] Query must be at least 2 characters")
        sys.exit(1)

    # Load config
    try:
        config_manager = get_config_manager()
        config = config_manager.load()
        vector_store_path = config.database.vector_store_path
    except FileNotFoundError:
        # Use default path if no config
        vector_store_path = None
    except Exception as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] Could not load config: {e}")
        vector_store_path = None

    # Build filters
    filters = {}

    if file_type:
        extensions = parse_extensions(file_type)
        if extensions:
            filters["extension"] = extensions if len(extensions) > 1 else extensions[0]

    if after:
        try:
            filters["after"] = parse_date(after)
        except click.BadParameter as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)

    if before:
        try:
            filters["before"] = parse_date(before)
        except click.BadParameter as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)

    if tag:
        filters["tag"] = tag

    # Initialize search engine
    try:
        engine = SearchEngine(persist_directory=vector_store_path)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to initialize search: {e}")
        logger.exception("Search engine initialization failed")
        sys.exit(1)

    try:
        # Check if index is empty
        if engine.is_index_empty():
            console.print(
                Panel(
                    "[yellow]No files have been indexed yet.[/yellow]\n\n"
                    "Run [bold]fileassistant index <path>[/bold] to index your files first.\n\n"
                    "Example:\n"
                    "  fileassistant index ~/Documents --recursive",
                    title="Empty Index",
                    border_style="yellow",
                )
            )
            sys.exit(0)

        # Perform search
        if not output_json:
            console.print(f"[dim]Searching for:[/dim] [bold]{query_str}[/bold]")
            if filters:
                filter_parts = []
                if "extension" in filters:
                    ext = filters["extension"]
                    if isinstance(ext, list):
                        filter_parts.append(f"types: {', '.join(ext)}")
                    else:
                        filter_parts.append(f"type: {ext}")
                if "after" in filters:
                    filter_parts.append(f"after: {filters['after'].strftime('%Y-%m-%d')}")
                if "before" in filters:
                    filter_parts.append(f"before: {filters['before'].strftime('%Y-%m-%d')}")
                if "tag" in filters:
                    filter_parts.append(f"tag: {filters['tag']}")
                console.print(f"[dim]Filters: {', '.join(filter_parts)}[/dim]")
            console.print()

        results = engine.search(query_str, filters=filters, limit=limit)

        # Output results
        if output_json:
            print(format_results_json(results))
        elif not results:
            console.print("[yellow]No matching files found.[/yellow]")
            console.print("\n[dim]Tips:[/dim]")
            console.print("  - Try different or simpler search terms")
            console.print("  - Remove filters to broaden the search")
            console.print("  - Check that the files you're looking for are indexed")
        elif compact:
            console.print(format_results_table(results))
            console.print(f"\n[dim]Found {len(results)} result(s)[/dim]")
        else:
            for i, result in enumerate(results, 1):
                console.print(format_result_rich(result, i))
                console.print()
            console.print(f"[dim]Found {len(results)} result(s)[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Search failed: {e}")
        logger.exception("Search failed")
        sys.exit(1)
    finally:
        engine.close()
