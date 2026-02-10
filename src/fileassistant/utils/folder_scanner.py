"""Folder scanner utility for discovering existing folder structures."""

from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)

# Folders to always exclude from scanning
EXCLUDED_FOLDERS = {
    # Hidden folders
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    ".vs",
    "__pycache__",
    ".cache",
    ".npm",
    ".yarn",
    # System folders
    "$RECYCLE.BIN",
    "System Volume Information",
    "node_modules",
    ".Trash",
    ".Spotlight-V100",
    ".fseventsd",
    # Application data
    "AppData",
    "Application Data",
    "Library",
    # FileAssistant internal
    ".fileassistant",
}


@dataclass
class FolderNode:
    """Represents a folder in the tree structure."""

    name: str
    path: Path
    children: list["FolderNode"] = field(default_factory=list)
    depth: int = 0

    def to_tree_string(self, prefix: str = "", is_last: bool = True) -> str:
        """Convert this node and children to a tree string representation."""
        lines = []

        # Add this node
        if self.depth == 0:
            lines.append(f"{self.name}/")
        else:
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{self.name}/")

        # Add children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            lines.append(child.to_tree_string(child_prefix, is_last_child))

        return "\n".join(lines)

    def get_all_paths(self, relative_to: Path | None = None) -> list[str]:
        """Get all folder paths as a flat list of relative path strings."""
        paths = []

        if relative_to:
            try:
                rel_path = self.path.relative_to(relative_to)
                paths.append(str(rel_path).replace("\\", "/"))
            except ValueError:
                paths.append(str(self.path).replace("\\", "/"))
        else:
            paths.append(str(self.path).replace("\\", "/"))

        for child in self.children:
            paths.extend(child.get_all_paths(relative_to))

        return paths


@dataclass
class FolderScanResult:
    """Result of scanning folders for context."""

    roots: list[FolderNode] = field(default_factory=list)
    total_folders: int = 0
    max_depth_reached: int = 0

    def to_tree_string(self) -> str:
        """Get a combined tree string of all roots."""
        trees = []
        for root in self.roots:
            trees.append(root.to_tree_string())
        return "\n\n".join(trees)

    def get_all_paths(self) -> list[str]:
        """Get all folder paths as a flat list."""
        paths = []
        for root in self.roots:
            # Use the root's parent as the relative base
            paths.extend(root.get_all_paths(root.path.parent))
        return paths

    def to_prompt_context(self, max_folders: int = 100) -> str:
        """
        Generate a concise representation for LLM prompt context.

        Args:
            max_folders: Maximum number of folders to include

        Returns:
            A string representation suitable for LLM context
        """
        all_paths = self.get_all_paths()

        # Sort by path for readability
        all_paths.sort()

        # Truncate if needed
        if len(all_paths) > max_folders:
            all_paths = all_paths[:max_folders]
            truncated_note = f"\n... and {self.total_folders - max_folders} more folders"
        else:
            truncated_note = ""

        return "\n".join(f"- {p}" for p in all_paths) + truncated_note


class FolderScanner:
    """
    Scans directories to discover existing folder structures.

    Used to provide context to the classifier about where files
    could potentially be organized.
    """

    def __init__(
        self,
        max_depth: int = 4,
        excluded_folders: set[str] | None = None,
    ):
        """
        Initialize the folder scanner.

        Args:
            max_depth: Maximum depth to scan (0 = root only)
            excluded_folders: Additional folder names to exclude
        """
        self.max_depth = max_depth
        self.excluded_folders = EXCLUDED_FOLDERS.copy()
        if excluded_folders:
            self.excluded_folders.update(excluded_folders)

    def _should_exclude(self, folder: Path) -> bool:
        """Check if a folder should be excluded from scanning."""
        name = folder.name

        # Check against excluded names
        if name in self.excluded_folders:
            return True

        # Exclude hidden folders (starting with .)
        if name.startswith("."):
            return True

        # Exclude folders starting with ~ (temp files)
        if name.startswith("~"):
            return True

        return False

    def _scan_folder(self, folder: Path, current_depth: int) -> FolderNode | None:
        """
        Recursively scan a folder and build a tree.

        Args:
            folder: Folder to scan
            current_depth: Current depth in the tree

        Returns:
            FolderNode or None if folder should be excluded
        """
        if self._should_exclude(folder):
            return None

        node = FolderNode(
            name=folder.name,
            path=folder,
            depth=current_depth,
        )

        # Stop at max depth
        if current_depth >= self.max_depth:
            return node

        # Scan children
        try:
            for child in sorted(folder.iterdir()):
                if child.is_dir():
                    child_node = self._scan_folder(child, current_depth + 1)
                    if child_node:
                        node.children.append(child_node)
        except PermissionError:
            logger.debug(f"Permission denied accessing {folder}")
        except OSError as e:
            logger.debug(f"Error accessing {folder}: {e}")

        return node

    def scan(self, paths: list[Path]) -> FolderScanResult:
        """
        Scan multiple root paths and return combined results.

        Args:
            paths: List of root paths to scan

        Returns:
            FolderScanResult with all discovered folders
        """
        result = FolderScanResult()

        for path in paths:
            path = Path(path).resolve()

            if not path.exists():
                logger.warning(f"Scan path does not exist: {path}")
                continue

            if not path.is_dir():
                logger.warning(f"Scan path is not a directory: {path}")
                continue

            logger.debug(f"Scanning folder structure: {path}")

            root_node = self._scan_folder(path, 0)
            if root_node:
                result.roots.append(root_node)

        # Calculate statistics
        result.total_folders = self._count_folders(result)
        result.max_depth_reached = self._find_max_depth(result)

        logger.info(
            f"Scanned {len(paths)} root(s), found {result.total_folders} folders "
            f"(max depth: {result.max_depth_reached})"
        )

        return result

    def _count_folders(self, result: FolderScanResult) -> int:
        """Count total folders in the result."""

        def count_node(node: FolderNode) -> int:
            return 1 + sum(count_node(child) for child in node.children)

        return sum(count_node(root) for root in result.roots)

    def _find_max_depth(self, result: FolderScanResult) -> int:
        """Find the maximum depth in the result."""

        def max_depth_node(node: FolderNode) -> int:
            if not node.children:
                return node.depth
            return max(max_depth_node(child) for child in node.children)

        if not result.roots:
            return 0
        return max(max_depth_node(root) for root in result.roots)


def scan_folders_for_context(
    paths: list[Path],
    max_depth: int = 4,
) -> FolderScanResult:
    """
    Convenience function to scan folders for classifier context.

    Args:
        paths: List of paths to scan
        max_depth: Maximum depth to scan

    Returns:
        FolderScanResult ready for use in classifier
    """
    scanner = FolderScanner(max_depth=max_depth)
    return scanner.scan(paths)
