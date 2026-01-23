"""Mover component for safe file movement with undo capability."""

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from ..classifier.classifier import ClassificationResult
from ..database.schema import Action, ActionType
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MoveResult:
    """Result of a file move operation."""

    # File info
    source_path: Path
    destination_path: Path
    filename: str

    # Status
    success: bool
    error_message: str | None = None

    # Database tracking
    action_id: int | None = None

    @property
    def destination_folder(self) -> Path:
        """Get the destination folder."""
        return self.destination_path.parent


class FileMover:
    """
    Moves files safely with conflict handling and undo capability.

    All moves are recorded in the database for potential undo operations.
    """

    def __init__(self, organized_base_path: Path, db_session: Session | None = None):
        """
        Initialize the file mover.

        Args:
            organized_base_path: Base path where organized files are stored
            db_session: Optional database session for action logging
        """
        self.organized_base_path = Path(organized_base_path)
        self.db_session = db_session

    def _resolve_conflict(self, destination: Path) -> Path:
        """
        Resolve naming conflicts by adding (1), (2), etc.

        Args:
            destination: Original destination path

        Returns:
            Conflict-free destination path
        """
        if not destination.exists():
            return destination

        # Split into stem and suffix(es)
        stem = destination.stem
        suffixes = "".join(destination.suffixes)
        parent = destination.parent

        counter = 1
        while True:
            new_name = f"{stem} ({counter}){suffixes}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

            # Safety limit
            if counter > 1000:
                raise RuntimeError(f"Too many conflicts for {destination}")

    def _record_action(
        self,
        source: Path,
        destination: Path,
        action_type: ActionType = ActionType.MOVE,
    ) -> int | None:
        """
        Record an action in the database for undo capability.

        Args:
            source: Original file path
            destination: New file path
            action_type: Type of action

        Returns:
            Action ID if recorded, None otherwise
        """
        if self.db_session is None:
            return None

        try:
            action = Action(
                action_type=action_type.value,
                before_state={
                    "path": str(source),
                    "filename": source.name,
                    "existed": source.exists(),
                },
                after_state={
                    "path": str(destination),
                    "filename": destination.name,
                },
            )
            self.db_session.add(action)
            self.db_session.commit()

            logger.debug(f"Recorded action {action.id}: {action_type.value}")
            return action.id

        except Exception as e:
            logger.error(f"Failed to record action: {e}")
            self.db_session.rollback()
            return None

    def move(
        self,
        source: Path,
        destination_folder: str,
        create_folders: bool = True,
    ) -> MoveResult:
        """
        Move a file to the organized folder structure.

        Args:
            source: Source file path
            destination_folder: Relative folder path within organized_base_path
            create_folders: Whether to create destination folders if they don't exist

        Returns:
            MoveResult with operation status
        """
        source = Path(source).resolve()

        # Validate source exists
        if not source.exists():
            return MoveResult(
                source_path=source,
                destination_path=source,
                filename=source.name,
                success=False,
                error_message=f"Source file not found: {source}",
            )

        if not source.is_file():
            return MoveResult(
                source_path=source,
                destination_path=source,
                filename=source.name,
                success=False,
                error_message=f"Source is not a file: {source}",
            )

        # Build destination path
        dest_folder = self.organized_base_path / destination_folder
        destination = dest_folder / source.name

        # Create folders if needed
        if create_folders and not dest_folder.exists():
            try:
                dest_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created folder: {dest_folder}")

                # Record folder creation
                self._record_action(
                    dest_folder, dest_folder, ActionType.CREATE_FOLDER
                )

            except OSError as e:
                return MoveResult(
                    source_path=source,
                    destination_path=destination,
                    filename=source.name,
                    success=False,
                    error_message=f"Failed to create folder: {e}",
                )

        # Resolve naming conflicts
        try:
            destination = self._resolve_conflict(destination)
        except RuntimeError as e:
            return MoveResult(
                source_path=source,
                destination_path=destination,
                filename=source.name,
                success=False,
                error_message=str(e),
            )

        # Perform the move
        try:
            shutil.move(str(source), str(destination))
            logger.info(f"Moved: {source.name} -> {destination}")

            # Record the action
            action_id = self._record_action(source, destination, ActionType.MOVE)

            return MoveResult(
                source_path=source,
                destination_path=destination,
                filename=destination.name,
                success=True,
                action_id=action_id,
            )

        except PermissionError as e:
            logger.error(f"Permission denied moving {source}: {e}")
            return MoveResult(
                source_path=source,
                destination_path=destination,
                filename=source.name,
                success=False,
                error_message=f"Permission denied: {e}",
            )

        except OSError as e:
            logger.error(f"Error moving {source}: {e}")
            return MoveResult(
                source_path=source,
                destination_path=destination,
                filename=source.name,
                success=False,
                error_message=f"Move failed: {e}",
            )

    def move_from_classification(
        self,
        classification: ClassificationResult,
    ) -> MoveResult:
        """
        Move a file based on classification result.

        Args:
            classification: ClassificationResult with destination folder

        Returns:
            MoveResult with operation status
        """
        return self.move(
            source=classification.file_path,
            destination_folder=classification.destination_folder,
        )

    def undo_move(self, action_id: int) -> MoveResult:
        """
        Undo a previous move action.

        Args:
            action_id: ID of the action to undo

        Returns:
            MoveResult with operation status
        """
        if self.db_session is None:
            return MoveResult(
                source_path=Path("."),
                destination_path=Path("."),
                filename="",
                success=False,
                error_message="Database session not available",
            )

        # Get the action
        action = self.db_session.query(Action).filter(Action.id == action_id).first()

        if action is None:
            return MoveResult(
                source_path=Path("."),
                destination_path=Path("."),
                filename="",
                success=False,
                error_message=f"Action {action_id} not found",
            )

        if action.undone:
            return MoveResult(
                source_path=Path("."),
                destination_path=Path("."),
                filename="",
                success=False,
                error_message=f"Action {action_id} was already undone",
            )

        if action.action_type != ActionType.MOVE.value:
            return MoveResult(
                source_path=Path("."),
                destination_path=Path("."),
                filename="",
                success=False,
                error_message=f"Cannot undo action type: {action.action_type}",
            )

        # Get paths from action state
        current_path = Path(action.after_state["path"])
        original_path = Path(action.before_state["path"])

        # Validate current file exists
        if not current_path.exists():
            return MoveResult(
                source_path=current_path,
                destination_path=original_path,
                filename=current_path.name,
                success=False,
                error_message=f"Current file not found: {current_path}",
            )

        # Move back to original location
        try:
            # Ensure original directory exists
            original_path.parent.mkdir(parents=True, exist_ok=True)

            # Resolve conflicts for original location
            final_destination = self._resolve_conflict(original_path)

            shutil.move(str(current_path), str(final_destination))

            # Mark action as undone
            action.undone = True
            action.undone_at = datetime.utcnow()
            self.db_session.commit()

            logger.info(f"Undone move: {current_path.name} -> {final_destination}")

            return MoveResult(
                source_path=current_path,
                destination_path=final_destination,
                filename=final_destination.name,
                success=True,
                action_id=action_id,
            )

        except Exception as e:
            logger.error(f"Failed to undo action {action_id}: {e}")
            return MoveResult(
                source_path=current_path,
                destination_path=original_path,
                filename=original_path.name,
                success=False,
                error_message=f"Undo failed: {e}",
            )

    def get_recent_actions(self, limit: int = 10) -> list[Action]:
        """
        Get recent move actions for display.

        Args:
            limit: Maximum number of actions to return

        Returns:
            List of Action objects
        """
        if self.db_session is None:
            return []

        return (
            self.db_session.query(Action)
            .filter(Action.action_type == ActionType.MOVE.value)
            .order_by(Action.timestamp.desc())
            .limit(limit)
            .all()
        )
