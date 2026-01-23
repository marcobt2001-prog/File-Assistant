"""Main CLI interface for FileAssistant using Click."""

import os
import platform
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..config import get_config_manager
from ..database import get_database, initialize_migrations
from ..utils.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="FileAssistant")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx, config: Path | None):
    """
    FileAssistant - Your local, privacy-first AI file organizer.

    A smart assistant that learns how you organize files and helps automate the process.
    """
    # Store config path in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    help="Custom database path (overrides config)",
)
@click.pass_context
def init(ctx, db_path: Path | None):
    """
    Initialize FileAssistant database and create default configuration.

    This command sets up the database schema and creates a default configuration
    file if one doesn't exist.
    """
    console.print("\n[bold cyan]FileAssistant Initialization[/bold cyan]\n")

    try:
        # Load or create config
        config_manager = get_config_manager(ctx.obj.get("config_path"))
        config = config_manager.load(create_if_missing=True)

        # Save config if newly created
        if config_manager.config_path is None:
            default_config_path = Path("config/default_config.yaml")
            config_manager.save(config, default_config_path)
            console.print(f"✓ Created default configuration: [green]{default_config_path}[/green]")
        else:
            console.print(
                f"✓ Loaded configuration from: [green]{config_manager.config_path}[/green]"
            )

        # Setup logging
        setup_logging(
            level=config.logging.level,
            log_dir=config.logging.log_dir,
            max_bytes=config.logging.max_bytes,
            backup_count=config.logging.backup_count,
            console_enabled=config.logging.console_enabled,
            file_enabled=config.logging.file_enabled,
        )

        # Initialize database
        final_db_path = db_path or config.database.path
        console.print(f"\n[cyan]Initializing database:[/cyan] {final_db_path}")

        db = get_database(final_db_path)
        db.create_all_tables()
        console.print("✓ Created database tables")

        # Run migrations
        migration_manager = initialize_migrations(db)
        migration_manager.apply_migrations()
        console.print("✓ Applied database migrations")

        # Create necessary directories
        config.database.path.parent.mkdir(parents=True, exist_ok=True)
        config.database.vector_store_path.mkdir(parents=True, exist_ok=True)
        config.logging.log_dir.mkdir(parents=True, exist_ok=True)
        console.print("✓ Created data directories")

        console.print("\n[bold green]✓ Initialization complete![/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Review configuration: [yellow]fileassistant config show[/yellow]")
        console.print("  2. Edit if needed: [yellow]fileassistant config edit[/yellow]")
        console.print("  3. Check status: [yellow]fileassistant status[/yellow]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Initialization failed:[/bold red] {e}")
        logger.exception("Initialization error")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """
    Show FileAssistant status and database statistics.

    Displays information about processed files, pending items, and system state.
    """
    console.print("\n[bold cyan]FileAssistant Status[/bold cyan]\n")

    try:
        # Load config
        config_manager = get_config_manager(ctx.obj.get("config_path"))
        config = config_manager.load()

        # Connect to database
        db = get_database(config.database.path)
        session = db.get_session()

        # Import models for queries
        from ..database import (
            Action,
            Classification,
            ClassificationStatus,
            File,
            FileStatus,
            Tag,
        )

        # Gather statistics
        total_files = session.query(File).count()
        pending_files = session.query(File).filter(File.status == FileStatus.PENDING).count()
        processed_files = session.query(File).filter(File.status == FileStatus.PROCESSED).count()
        error_files = session.query(File).filter(File.status == FileStatus.ERROR).count()

        total_tags = session.query(Tag).count()
        total_classifications = session.query(Classification).count()
        pending_classifications = (
            session.query(Classification)
            .filter(Classification.status == ClassificationStatus.PENDING)
            .count()
        )
        total_actions = session.query(Action).count()

        session.close()

        # Create status table
        table = Table(title="Database Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Total Files", str(total_files))
        table.add_row("  ├─ Pending", str(pending_files))
        table.add_row("  ├─ Processed", str(processed_files))
        table.add_row("  └─ Errors", str(error_files))
        table.add_row("", "")
        table.add_row("Total Tags", str(total_tags))
        table.add_row("Total Classifications", str(total_classifications))
        table.add_row("  └─ Pending Review", str(pending_classifications))
        table.add_row("Total Actions", str(total_actions))

        console.print(table)

        # Configuration info
        console.print(f"\n[cyan]Database:[/cyan] {config.database.path}")
        console.print(f"[cyan]Config:[/cyan] {config_manager.config_path}")
        console.print(
            f"[cyan]Auto-processing:[/cyan] {'[green]Enabled[/green]' if config.auto_process_enabled else '[yellow]Disabled[/yellow]'}"
        )

        # Inbox folders
        console.print("\n[cyan]Monitored Folders:[/cyan]")
        for folder in config.inbox_folders:
            exists = "✓" if folder.exists() else "✗"
            color = "green" if folder.exists() else "red"
            console.print(f"  [{color}]{exists}[/{color}] {folder}")

    except FileNotFoundError:
        console.print(
            "[yellow]⚠ No configuration found. Run:[/yellow] [cyan]fileassistant init[/cyan]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Status command error")
        sys.exit(1)


@cli.group(name="config")
def config_group():
    """Manage FileAssistant configuration."""
    pass


@config_group.command(name="show")
@click.pass_context
def config_show(ctx):
    """Display current configuration."""
    console.print("\n[bold cyan]FileAssistant Configuration[/bold cyan]\n")

    try:
        config_manager = get_config_manager(ctx.obj.get("config_path"))
        config = config_manager.load()

        # Display config sections
        console.print("[bold]Inbox Folders:[/bold]")
        for folder in config.inbox_folders:
            console.print(f"  • {folder}")

        console.print("\n[bold]Organized Files:[/bold]")
        console.print(f"  Base Path: {config.organized_base_path}")

        console.print("\n[bold]Confidence Thresholds:[/bold]")
        console.print(f"  High:   {config.confidence_thresholds.high}")
        console.print(f"  Medium: {config.confidence_thresholds.medium}")
        console.print(f"  Low:    {config.confidence_thresholds.low}")

        console.print("\n[bold]Processing Settings:[/bold]")
        console.print(f"  Idle Only: {config.processing.idle_only}")
        console.print(f"  Debounce: {config.processing.debounce_seconds}s")
        console.print(f"  Max File Size: {config.processing.max_file_size_mb}MB")
        console.print(f"  Batch Size: {config.processing.batch_size}")

        console.print("\n[bold]AI Settings:[/bold]")
        console.print(f"  Model: {config.ai_settings.model_name}")
        console.print(f"  Embedding Model: {config.ai_settings.embedding_model}")
        console.print(f"  Temperature: {config.ai_settings.temperature}")
        console.print(f"  Ollama URL: {config.ai_settings.ollama_base_url}")

        console.print("\n[bold]Database:[/bold]")
        console.print(f"  Path: {config.database.path}")
        console.print(f"  Vector Store: {config.database.vector_store_path}")

        console.print("\n[bold]Feature Flags:[/bold]")
        console.print(
            f"  Auto-processing: {'[green]Enabled[/green]' if config.auto_process_enabled else '[yellow]Disabled[/yellow]'}"
        )
        console.print(
            f"  Learning: {'[green]Enabled[/green]' if config.learning_enabled else '[yellow]Disabled[/yellow]'}"
        )

        console.print(f"\n[dim]Config file: {config_manager.config_path}[/dim]")

    except FileNotFoundError:
        console.print(
            "[yellow]⚠ No configuration found. Run:[/yellow] [cyan]fileassistant init[/cyan]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Config show error")
        sys.exit(1)


@config_group.command(name="edit")
@click.pass_context
def config_edit(ctx):
    """Open configuration file in default editor."""
    try:
        config_manager = get_config_manager(ctx.obj.get("config_path"))

        # Try to load config to find its location
        try:
            config_manager.load()
            config_path = config_manager.config_path
        except FileNotFoundError:
            # Create default config
            console.print("[yellow]No config found. Creating default...[/yellow]")
            default_path = Path("config/default_config.yaml")
            config = config_manager.load(create_if_missing=True)
            config_manager.save(config, default_path)
            config_path = default_path
            console.print(f"[green]✓ Created:[/green] {config_path}")

        # Open in editor
        console.print(f"[cyan]Opening config file:[/cyan] {config_path}")

        # Determine editor based on platform
        if platform.system() == "Windows":
            os.startfile(config_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", config_path])
        else:  # Linux
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run([editor, config_path])

        console.print("[green]✓ Config file opened[/green]")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Config edit error")
        sys.exit(1)


# =============================================================================
# Phase 1: Watch and Analyze Commands
# =============================================================================


@cli.command()
@click.option(
    "--folder",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Specific folder(s) to watch (defaults to config inbox_folders)",
)
@click.pass_context
def watch(ctx, folder: tuple[Path, ...]):
    """
    Watch inbox folders for new files.

    Monitors configured inbox folders (or specified folders) and reports
    when new supported files are detected. Press Ctrl+C to stop.
    """
    from ..watcher import FileWatcher, SUPPORTED_EXTENSIONS

    console.print("\n[bold cyan]FileAssistant File Watcher[/bold cyan]\n")

    try:
        config_manager = get_config_manager(ctx.obj.get("config_path"))
        config = config_manager.load()

        # Override inbox folders if specified
        if folder:
            config.inbox_folders = list(folder)

        console.print(f"[cyan]Supported extensions:[/cyan] {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        console.print(f"[cyan]Debounce delay:[/cyan] {config.processing.debounce_seconds}s\n")

        console.print("[cyan]Watching folders:[/cyan]")
        for f in config.inbox_folders:
            exists = "✓" if f.exists() else "✗ (will create)"
            console.print(f"  • {f} {exists}")

        console.print("\n[yellow]Press Ctrl+C to stop watching[/yellow]\n")

        # Counter for files detected
        file_count = [0]

        def on_file_ready(file_path: Path):
            """Callback when a file is ready for processing."""
            file_count[0] += 1
            console.print(
                f"[green]File ready:[/green] {file_path.name} "
                f"[dim]({file_path.stat().st_size / 1024:.1f} KB)[/dim]"
            )

        watcher = FileWatcher(config=config, on_file_ready=on_file_ready)

        # Scan for existing files first
        existing = watcher.scan_existing()
        if existing:
            console.print(f"[cyan]Found {len(existing)} existing file(s):[/cyan]")
            for f in existing[:10]:  # Show first 10
                console.print(f"  • {f.name}")
            if len(existing) > 10:
                console.print(f"  ... and {len(existing) - 10} more")
            console.print()

        with watcher:
            console.print("[green]Watcher started. Waiting for files...[/green]\n")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping watcher...[/yellow]")

        console.print(f"\n[green]✓ Watcher stopped. Detected {file_count[0]} new file(s).[/green]")

    except FileNotFoundError:
        console.print(
            "[yellow]⚠ No configuration found. Run:[/yellow] [cyan]fileassistant init[/cyan]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Watch command error")
        sys.exit(1)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--show-content", "-c", is_flag=True, help="Show extracted content")
@click.option("--preview-length", "-p", default=500, help="Content preview length (chars)")
def analyze(file_path: Path, show_content: bool, preview_length: int):
    """
    Analyze a single file and show extracted content.

    FILE_PATH is the path to the file to analyze.
    """
    from ..analyzer import FileAnalyzer, get_supported_extensions

    console.print("\n[bold cyan]FileAssistant File Analyzer[/bold cyan]\n")

    supported = get_supported_extensions()
    if file_path.suffix.lower() not in supported:
        console.print(
            f"[yellow]⚠ Unsupported file type:[/yellow] {file_path.suffix}\n"
            f"[dim]Supported: {', '.join(sorted(supported))}[/dim]"
        )
        sys.exit(1)

    analyzer = FileAnalyzer()
    result = analyzer.analyze(file_path)

    if not result.success:
        console.print(f"[bold red]✗ Analysis failed:[/bold red] {result.error_message}")
        sys.exit(1)

    # Display results
    console.print(f"[bold]File:[/bold] {result.file_path.name}")
    console.print(f"[bold]Path:[/bold] {result.file_path}")

    # Metadata table
    table = Table(title="Metadata", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Extension", result.metadata.extension)
    table.add_row("Size", f"{result.metadata.size_bytes:,} bytes ({result.metadata.size_bytes / 1024:.1f} KB)")
    table.add_row("Created", result.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("Modified", result.metadata.modified_at.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("MD5 Hash", result.metadata.hash_md5)

    console.print()
    console.print(table)

    # Content stats
    console.print(f"\n[bold]Content Stats:[/bold]")
    console.print(f"  Words: {result.word_count:,}")
    console.print(f"  Lines: {result.line_count:,}")
    console.print(f"  Characters: {len(result.content):,}")

    # Content preview
    if show_content or preview_length > 0:
        console.print(f"\n[bold]Content Preview:[/bold]")
        preview = result.content[:preview_length]
        if len(result.content) > preview_length:
            preview += "\n[dim]... (truncated)[/dim]"
        console.print(f"[dim]{'-' * 60}[/dim]")
        console.print(preview)
        console.print(f"[dim]{'-' * 60}[/dim]")

    console.print("\n[green]✓ Analysis complete[/green]")


@cli.command()
@click.argument(
    "folder",
    type=click.Path(exists=True, path_type=Path),
    default=".",
)
@click.option("--recursive", "-r", is_flag=True, help="Scan subfolders recursively")
def scan(folder: Path, recursive: bool):
    """
    Scan a folder and analyze all supported files.

    FOLDER is the path to scan (defaults to current directory).
    """
    from ..analyzer import FileAnalyzer, get_supported_extensions

    console.print("\n[bold cyan]FileAssistant Folder Scanner[/bold cyan]\n")

    supported = get_supported_extensions()
    console.print(f"[cyan]Scanning:[/cyan] {folder}")
    console.print(f"[cyan]Supported:[/cyan] {', '.join(sorted(supported))}")
    console.print(f"[cyan]Recursive:[/cyan] {'Yes' if recursive else 'No'}\n")

    # Find all supported files
    files: list[Path] = []
    if recursive:
        for ext in supported:
            files.extend(folder.rglob(f"*{ext}"))
    else:
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in supported:
                files.append(f)

    # Filter hidden files
    files = [f for f in files if not f.name.startswith(".")]

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        return

    console.print(f"[green]Found {len(files)} file(s)[/green]\n")

    # Analyze each file
    analyzer = FileAnalyzer()

    table = Table(title="Scan Results", show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan", max_width=40)
    table.add_column("Type", style="blue")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Words", style="yellow", justify="right")
    table.add_column("Status", style="green")

    success_count = 0
    error_count = 0

    for file_path in sorted(files):
        result = analyzer.analyze(file_path)

        if result.success:
            success_count += 1
            size_str = f"{result.metadata.size_bytes / 1024:.1f} KB"
            status = "[green]✓[/green]"
            table.add_row(
                file_path.name,
                result.metadata.extension,
                size_str,
                str(result.word_count),
                status,
            )
        else:
            error_count += 1
            table.add_row(
                file_path.name,
                file_path.suffix.lower(),
                "-",
                "-",
                f"[red]✗[/red] {result.error_message[:20]}...",
            )

    console.print(table)
    console.print(f"\n[green]✓ Scan complete:[/green] {success_count} succeeded, {error_count} failed")


if __name__ == "__main__":
    cli()
