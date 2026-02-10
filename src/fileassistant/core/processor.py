"""Processing loop orchestrator for the full file pipeline."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..analyzer import FileAnalyzer, AnalysisResult
from ..classifier import FileClassifier, ClassificationResult
from ..config.models import FileAssistantConfig
from ..database import Action, Classification, ClassificationStatus, File, FileStatus
from ..mover import FileMover, MoveResult
from ..utils.logging import get_logger
from ..utils.folder_scanner import FolderScanner, FolderScanResult

logger = get_logger(__name__)
console = Console()


class UserDecision(str, Enum):
    """User decision options for file processing."""

    ACCEPT = "accept"
    EDIT = "edit"
    SKIP = "skip"


@dataclass
class ProcessingResult:
    """Result of processing a single file through the pipeline."""

    file_path: Path
    filename: str

    # Pipeline results
    analysis: AnalysisResult | None = None
    classification: ClassificationResult | None = None
    move_result: MoveResult | None = None

    # User interaction
    user_decision: UserDecision | None = None
    edited_destination: str | None = None

    # Status
    success: bool = False
    skipped: bool = False
    error_message: str | None = None

    @property
    def final_destination(self) -> str | None:
        """Get the final destination (edited or original)."""
        if self.edited_destination:
            return self.edited_destination
        if self.classification:
            return self.classification.destination_folder
        return None


class FileProcessor:
    """
    Orchestrates the full file processing pipeline.

    Pipeline: file arrives -> analyze -> classify -> user confirmation -> move
    """

    def __init__(self, config: FileAssistantConfig, db_session=None):
        """
        Initialize the file processor.

        Args:
            config: FileAssistant configuration
            db_session: Optional database session for persistence
        """
        self.config = config
        self.db_session = db_session

        # Initialize components
        self.analyzer = FileAnalyzer(
            max_file_size_mb=config.processing.max_file_size_mb
        )
        self.classifier = FileClassifier(
            ai_settings=config.ai_settings,
            confidence_thresholds=config.confidence_thresholds,
        )
        self.mover = FileMover(
            organized_base_path=config.organized_base_path,
            db_session=db_session,
        )

        # Folder scanner for providing context to classifier
        self.folder_scanner = FolderScanner(max_depth=config.folder_scan_depth)
        self._folder_context: FolderScanResult | None = None

    def check_system_ready(self) -> tuple[bool, list[str]]:
        """
        Check if the system is ready to process files.

        Returns:
            Tuple of (is_ready, list of issues)
        """
        issues = []

        # Check Ollama
        ollama_ready, ollama_msg = self.classifier.check_ollama_status()
        if not ollama_ready:
            issues.append(f"Ollama: {ollama_msg}")

        # Check base path is writable
        try:
            self.config.organized_base_path.mkdir(parents=True, exist_ok=True)
            test_file = self.config.organized_base_path / ".fileassistant_test"
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            issues.append(f"Cannot write to {self.config.organized_base_path}: {e}")

        return len(issues) == 0, issues

    def _scan_folder_context(self) -> FolderScanResult | None:
        """
        Scan folders to provide context for classification.

        Returns:
            FolderScanResult with existing folder structure
        """
        context_folders = self.config.get_context_folders()
        if not context_folders:
            return None

        # Filter to only existing folders
        existing_folders = [f for f in context_folders if f and f.exists()]
        if not existing_folders:
            return None

        try:
            result = self.folder_scanner.scan(existing_folders)
            if result.total_folders > 0:
                logger.info(
                    f"Scanned {result.total_folders} folders for context "
                    f"(max depth: {result.max_depth_reached})"
                )
            return result
        except Exception as e:
            logger.warning(f"Failed to scan folders for context: {e}")
            return None

    def _display_classification(
        self, classification: ClassificationResult, analysis: AnalysisResult
    ):
        """Display classification result in a formatted panel."""
        # Build info table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("File", classification.filename)
        table.add_row("Size", f"{analysis.metadata.size_bytes / 1024:.1f} KB")
        table.add_row("", "")

        # Show destination with folder status indicator
        folder_status = "[yellow][new folder][/yellow]" if classification.is_new_folder else "[green][existing][/green]"
        table.add_row("Destination", f"{classification.destination_folder} {folder_status}")

        table.add_row("Tags", ", ".join(classification.tags) if classification.tags else "(none)")
        table.add_row("Confidence", f"{classification.confidence:.0%} ({classification.confidence_level})")

        # Build reasoning panel
        reasoning_text = classification.reasoning or "No reasoning provided"

        console.print()
        console.print(Panel(
            table,
            title="[bold cyan]Classification Result[/bold cyan]",
            border_style="cyan",
        ))
        console.print(f"[dim]Reasoning: {reasoning_text}[/dim]")
        console.print()

    def _get_user_decision(self, classification: ClassificationResult) -> tuple[UserDecision, str | None]:
        """
        Prompt user for decision on file classification.

        Returns:
            Tuple of (decision, edited_destination or None)
        """
        console.print("[yellow]Options:[/yellow]")
        console.print("  [green][Y]es[/green] - Accept and move to suggested destination")
        console.print("  [blue][E]dit[/blue] - Edit destination folder")
        console.print("  [red][S]kip[/red] - Skip this file")
        console.print()

        while True:
            choice = Prompt.ask(
                "Your choice",
                choices=["y", "yes", "e", "edit", "s", "skip"],
                default="y",
            ).lower()

            if choice in ("y", "yes"):
                return UserDecision.ACCEPT, None

            elif choice in ("e", "edit"):
                new_dest = Prompt.ask(
                    "Enter new destination folder",
                    default=classification.destination_folder,
                )
                new_dest = new_dest.strip("/\\").replace("\\", "/")
                if new_dest:
                    return UserDecision.EDIT, new_dest
                console.print("[red]Invalid destination, please try again.[/red]")

            elif choice in ("s", "skip"):
                return UserDecision.SKIP, None

    def _record_classification(
        self,
        file_path: Path,
        classification: ClassificationResult,
        decision: UserDecision,
        final_destination: str | None,
    ):
        """Record classification in the database."""
        if self.db_session is None:
            return

        try:
            # Create or get file record
            file_record = (
                self.db_session.query(File)
                .filter(File.path == str(file_path))
                .first()
            )

            if file_record is None:
                file_record = File(
                    path=str(file_path),
                    filename=file_path.name,
                    extension=file_path.suffix.lower(),
                )
                self.db_session.add(file_record)
                self.db_session.flush()

            # Determine status
            if decision == UserDecision.ACCEPT:
                status = ClassificationStatus.ACCEPTED
            elif decision == UserDecision.EDIT:
                status = ClassificationStatus.MODIFIED
            else:
                status = ClassificationStatus.REJECTED

            # Create classification record
            classification_record = Classification(
                file_id=file_record.id,
                suggested_destination=classification.destination_folder,
                suggested_tags=classification.tags,
                confidence=classification.confidence,
                reasoning=classification.reasoning,
                status=status.value,
                final_destination=final_destination,
                final_tags=classification.tags,
            )
            self.db_session.add(classification_record)

            # Update file status
            if decision == UserDecision.SKIP:
                file_record.status = FileStatus.SKIPPED.value
            else:
                file_record.status = FileStatus.PROCESSED.value

            self.db_session.commit()

        except Exception as e:
            logger.error(f"Failed to record classification: {e}")
            self.db_session.rollback()

    def process_file(
        self,
        file_path: Path,
        interactive: bool = True,
    ) -> ProcessingResult:
        """
        Process a single file through the full pipeline.

        Args:
            file_path: Path to the file to process
            interactive: Whether to prompt user for confirmation

        Returns:
            ProcessingResult with pipeline results
        """
        file_path = Path(file_path).resolve()
        result = ProcessingResult(
            file_path=file_path,
            filename=file_path.name,
        )

        # Step 1: Analyze
        console.print(f"[cyan]Analyzing[/cyan] {file_path.name}...")
        analysis = self.analyzer.analyze(file_path)
        result.analysis = analysis

        if not analysis.success:
            result.error_message = f"Analysis failed: {analysis.error_message}"
            console.print(f"[red]Analysis failed:[/red] {analysis.error_message}")
            return result

        console.print(f"[green]✓[/green] Analyzed: {analysis.word_count} words, {analysis.metadata.size_bytes / 1024:.1f} KB")

        # Step 2: Scan folders for context (if not already done)
        if self._folder_context is None:
            console.print("[cyan]Scanning[/cyan] existing folder structure...")
            self._folder_context = self._scan_folder_context()
            if self._folder_context and self._folder_context.total_folders > 0:
                console.print(f"[green]✓[/green] Found {self._folder_context.total_folders} existing folders")

        # Step 3: Classify
        console.print(f"[cyan]Classifying[/cyan] with {self.config.ai_settings.model_name}...")
        classification = self.classifier.classify(analysis, folder_context=self._folder_context)
        result.classification = classification

        if not classification.success:
            result.error_message = f"Classification failed: {classification.error_message}"
            console.print(f"[red]Classification failed:[/red] {classification.error_message}")
            return result

        console.print("[green]✓[/green] Classification complete")

        # Step 4: Display and get user decision
        self._display_classification(classification, analysis)

        if interactive:
            decision, edited_dest = self._get_user_decision(classification)
        else:
            # Auto-accept for non-interactive mode (future feature)
            decision = UserDecision.ACCEPT
            edited_dest = None

        result.user_decision = decision
        result.edited_destination = edited_dest

        # Handle skip
        if decision == UserDecision.SKIP:
            result.skipped = True
            self._record_classification(file_path, classification, decision, None)
            console.print("[yellow]Skipped[/yellow]")
            return result

        # Step 5: Move file
        final_destination = edited_dest or classification.destination_folder
        console.print(f"[cyan]Moving[/cyan] to {final_destination}...")

        move_result = self.mover.move(file_path, final_destination)
        result.move_result = move_result

        if not move_result.success:
            result.error_message = f"Move failed: {move_result.error_message}"
            console.print(f"[red]Move failed:[/red] {move_result.error_message}")
            return result

        # Record to database
        self._record_classification(file_path, classification, decision, final_destination)

        result.success = True
        console.print(f"[green]✓[/green] Moved to: [bold]{move_result.destination_path}[/bold]")

        return result

    def process_multiple(
        self,
        file_paths: list[Path],
        interactive: bool = True,
    ) -> list[ProcessingResult]:
        """
        Process multiple files through the pipeline.

        Args:
            file_paths: List of file paths to process
            interactive: Whether to prompt user for confirmation

        Returns:
            List of ProcessingResult objects
        """
        results: list[ProcessingResult] = []

        total = len(file_paths)
        for i, file_path in enumerate(file_paths, 1):
            console.print()
            console.rule(f"[bold]File {i}/{total}[/bold]")
            result = self.process_file(file_path, interactive=interactive)
            results.append(result)

        return results
