"""CLI command for bulk indexing files for search."""

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..analyzer import FileAnalyzer, get_supported_extensions
from ..config import get_config_manager
from ..database import File, FileStatus, get_database
from ..embeddings import EmbeddingGenerator
from ..search import IndexManager
from ..utils.logging import get_logger

console = Console()
logger = get_logger(__name__)

# Additional extensions for code/config files that can be indexed
# These are plain text files that the PlainTextExtractor could handle
INDEXABLE_EXTENSIONS = {
    # Documents (from existing extractors)
    ".pdf", ".docx", ".txt", ".md",
    # Code files (plain text)
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php",
    ".css", ".scss", ".less",
    ".sh", ".bash", ".zsh", ".ps1",
    # Config/data files (plain text)
    ".json", ".yaml", ".yml", ".toml",
    ".xml", ".html", ".htm",
    ".csv", ".ini", ".cfg", ".conf",
    # Other text files
    ".rst", ".tex", ".log",
}


def get_indexable_extensions() -> set[str]:
    """Get all extensions that can be indexed."""
    # Combine analyzer-supported extensions with additional indexable ones
    return get_supported_extensions() | INDEXABLE_EXTENSIONS


def should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped (hidden files/folders)."""
    # Skip hidden files and folders
    for part in path.parts:
        if part.startswith(".") and part not in {".", ".."}:
            return True
    return False


def collect_files(
    root_path: Path,
    recursive: bool,
    max_size_mb: int,
    extensions: set[str],
) -> list[Path]:
    """
    Collect files to index from a directory.

    Args:
        root_path: Root directory to scan
        recursive: Whether to scan recursively
        max_size_mb: Maximum file size in MB
        extensions: Set of extensions to include

    Returns:
        List of file paths to index
    """
    files: list[Path] = []
    max_size_bytes = max_size_mb * 1024 * 1024

    if recursive:
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if should_skip_path(path):
                continue
            if path.suffix.lower() not in extensions:
                continue
            try:
                if path.stat().st_size > max_size_bytes:
                    continue
            except OSError:
                continue
            files.append(path)
    else:
        for path in root_path.iterdir():
            if not path.is_file():
                continue
            if should_skip_path(path):
                continue
            if path.suffix.lower() not in extensions:
                continue
            try:
                if path.stat().st_size > max_size_bytes:
                    continue
            except OSError:
                continue
            files.append(path)

    return sorted(files)


def extract_text_for_indexing(file_path: Path, analyzer: FileAnalyzer) -> tuple[str | None, str | None]:
    """
    Extract text from a file for indexing.

    Falls back to plain text reading for code/config files.

    Returns:
        Tuple of (text, error_message)
    """
    # Try the analyzer first for supported types
    if file_path.suffix.lower() in analyzer.supported_extensions:
        result = analyzer.analyze(file_path)
        if result.success:
            return result.content, None
        return None, result.error_message

    # For code/config files, try plain text extraction
    if file_path.suffix.lower() in INDEXABLE_EXTENSIONS:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                return content, None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return None, str(e)
        return None, "Could not decode file with any supported encoding"

    return None, "Unsupported file type"


@click.command(name="index")
@click.argument(
    "path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--recursive/--no-recursive",
    "-r/-R",
    default=True,
    help="Scan subdirectories recursively (default: recursive)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-index files even if already indexed",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Show what would be indexed without actually doing it",
)
@click.option(
    "--max-size",
    type=int,
    default=50,
    help="Maximum file size in MB (default: 50)",
)
@click.pass_context
def index_command(
    ctx,
    path: Path,
    recursive: bool,
    force: bool,
    dry_run: bool,
    max_size: int,
):
    """
    Index files for search.

    Scans PATH for supported files, extracts text, generates embeddings,
    and stores them in the search index for fast retrieval.

    This command is essential for making your existing files searchable.
    After indexing, use 'fileassistant search' to find files by content.

    \b
    Examples:
        fileassistant index ~/Documents
        fileassistant index ~/Projects --force
        fileassistant index ./folder --no-recursive --dry-run
    """
    console.print("\n[bold cyan]FileAssistant File Indexer[/bold cyan]\n")

    start_time = time.time()

    try:
        # Load config
        config_manager = get_config_manager(ctx.obj.get("config_path"))
        config = config_manager.load()

        # Get supported extensions
        extensions = get_indexable_extensions()
        console.print(f"[cyan]Scanning:[/cyan] {path}")
        console.print(f"[cyan]Recursive:[/cyan] {'Yes' if recursive else 'No'}")
        console.print(f"[cyan]Max file size:[/cyan] {max_size} MB")
        console.print(f"[cyan]Force re-index:[/cyan] {'Yes' if force else 'No'}")
        if dry_run:
            console.print("[yellow]DRY RUN - no files will be indexed[/yellow]")
        console.print()

        # Collect files
        console.print("[cyan]Scanning for files...[/cyan]")
        files = collect_files(path, recursive, max_size, extensions)

        if not files:
            console.print("[yellow]No supported files found.[/yellow]")
            console.print(f"\n[dim]Supported extensions: {', '.join(sorted(extensions))}[/dim]")
            return

        console.print(f"[green]Found {len(files)} file(s)[/green]\n")

        if dry_run:
            # Show what would be indexed
            console.print("[bold]Files that would be indexed:[/bold]")
            for f in files[:50]:  # Show first 50
                console.print(f"  • {f.relative_to(path) if f.is_relative_to(path) else f}")
            if len(files) > 50:
                console.print(f"  ... and {len(files) - 50} more files")

            elapsed = time.time() - start_time
            console.print(f"\n[green]Dry run complete.[/green] {len(files)} files would be indexed.")
            console.print(f"[dim]Elapsed: {elapsed:.1f}s[/dim]")
            return

        # Initialize components
        console.print("[cyan]Initializing...[/cyan]")

        # Initialize database
        db = get_database(config.database.path)
        db.create_all_tables()
        session = db.get_session()

        # Initialize analyzer
        analyzer = FileAnalyzer(max_file_size_mb=max_size)

        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(
            model_name=config.ai_settings.embedding_model,
            chunk_size=config.search.chunk_size,
            chunk_overlap=config.search.chunk_overlap,
        )

        # Initialize index manager
        index_manager = IndexManager(persist_directory=config.database.vector_store_path)

        console.print("[green]✓ Components initialized[/green]\n")

        # Stats
        stats = {
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
            "already_indexed": 0,
        }
        errors: list[tuple[Path, str]] = []

        # Process files with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Indexing files...", total=len(files))

            for file_path in files:
                try:
                    # Generate file ID
                    file_id = f"file_{hash(str(file_path.resolve())) & 0xffffffff:08x}"

                    # Check if already indexed (unless force)
                    if not force:
                        # Read file to compute hash for change detection
                        text, error = extract_text_for_indexing(file_path, analyzer)
                        if text:
                            content_hash = IndexManager.compute_content_hash(text)
                            if index_manager.is_indexed(file_id, content_hash):
                                stats["already_indexed"] += 1
                                progress.advance(task)
                                continue
                    else:
                        text, error = extract_text_for_indexing(file_path, analyzer)

                    if error or not text:
                        stats["errors"] += 1
                        errors.append((file_path, error or "Empty content"))
                        progress.advance(task)
                        continue

                    if not text.strip():
                        stats["skipped"] += 1
                        progress.advance(task)
                        continue

                    # Generate embedding
                    embedding_result = embedding_generator.generate(text)
                    if not embedding_result.success:
                        stats["errors"] += 1
                        errors.append((file_path, f"Embedding failed: {embedding_result.error_message}"))
                        progress.advance(task)
                        continue

                    # Get file stats
                    try:
                        file_stat = file_path.stat()
                        size_bytes = file_stat.st_size
                        from datetime import datetime
                        created_at = datetime.fromtimestamp(file_stat.st_ctime)
                        modified_at = datetime.fromtimestamp(file_stat.st_mtime)
                    except OSError:
                        size_bytes = 0
                        created_at = None
                        modified_at = None

                    # Index the file
                    success = index_manager.index_file(
                        file_id=file_id,
                        file_path=file_path,
                        text=text,
                        embedding=embedding_result.embedding,
                        content_summary=text[:500],
                        file_type="document",
                        created_at=created_at,
                        modified_at=modified_at,
                        size_bytes=size_bytes,
                    )

                    if not success:
                        stats["errors"] += 1
                        errors.append((file_path, "Failed to store in index"))
                        progress.advance(task)
                        continue

                    # Register in SQLite files table if not exists
                    existing_file = (
                        session.query(File)
                        .filter(File.path == str(file_path.resolve()))
                        .first()
                    )
                    if not existing_file:
                        new_file = File(
                            path=str(file_path.resolve()),
                            filename=file_path.name,
                            extension=file_path.suffix.lower(),
                            status=FileStatus.PROCESSED.value,
                            embedding_id=file_id,
                        )
                        session.add(new_file)

                    stats["indexed"] += 1
                    progress.advance(task)

                except Exception as e:
                    stats["errors"] += 1
                    errors.append((file_path, str(e)))
                    logger.exception(f"Error indexing {file_path}")
                    progress.advance(task)

            # Commit database changes
            session.commit()

        # Summary
        elapsed = time.time() - start_time
        console.print()
        console.print("[bold cyan]Indexing Summary[/bold cyan]")
        console.print(f"  [green]Indexed:[/green]        {stats['indexed']}")
        console.print(f"  [yellow]Already indexed:[/yellow] {stats['already_indexed']}")
        console.print(f"  [dim]Skipped (empty):[/dim] {stats['skipped']}")
        console.print(f"  [red]Errors:[/red]          {stats['errors']}")
        console.print(f"  [dim]Time elapsed:[/dim]    {elapsed:.1f}s")

        # Show errors if any
        if errors and len(errors) <= 10:
            console.print("\n[red]Errors:[/red]")
            for file_path, error in errors:
                console.print(f"  • {file_path.name}: {error[:60]}")
        elif errors:
            console.print(f"\n[red]{len(errors)} errors occurred. Check logs for details.[/red]")

        # Get total indexed count
        total_indexed = index_manager.get_indexed_count()
        console.print(f"\n[bold]Total files in index:[/bold] {total_indexed}")

        # Suggest search if files were indexed
        if stats["indexed"] > 0 or total_indexed > 0:
            console.print(
                "\n[green]✓ Indexing complete![/green] "
                'Try: [cyan]fileassistant search "your query"[/cyan]'
            )
        else:
            console.print("\n[yellow]No files were indexed.[/yellow]")

        # Cleanup
        index_manager.close()
        session.close()

    except FileNotFoundError:
        console.print(
            "[yellow]⚠ No configuration found. Run:[/yellow] [cyan]fileassistant init[/cyan]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Index command error")
        sys.exit(1)
