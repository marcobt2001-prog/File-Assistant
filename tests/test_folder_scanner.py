"""Tests for the folder scanner utility."""

from pathlib import Path

import pytest

from fileassistant.utils.folder_scanner import (
    EXCLUDED_FOLDERS,
    FolderNode,
    FolderScanResult,
    FolderScanner,
    scan_folders_for_context,
)


class TestFolderNode:
    """Tests for FolderNode."""

    def test_creation(self, tmp_path):
        """Test basic node creation."""
        node = FolderNode(name="test", path=tmp_path / "test", depth=0)
        assert node.name == "test"
        assert node.depth == 0
        assert node.children == []

    def test_to_tree_string_single_node(self, tmp_path):
        """Test tree string for a single node."""
        node = FolderNode(name="root", path=tmp_path / "root", depth=0)
        result = node.to_tree_string()
        assert result == "root/"

    def test_to_tree_string_with_children(self, tmp_path):
        """Test tree string with children."""
        root = FolderNode(name="root", path=tmp_path / "root", depth=0)
        child1 = FolderNode(name="child1", path=tmp_path / "root" / "child1", depth=1)
        child2 = FolderNode(name="child2", path=tmp_path / "root" / "child2", depth=1)
        root.children = [child1, child2]

        result = root.to_tree_string()

        assert "root/" in result
        assert "child1/" in result
        assert "child2/" in result
        assert "├──" in result or "└──" in result

    def test_get_all_paths(self, tmp_path):
        """Test getting all paths from a node."""
        root = FolderNode(name="Documents", path=tmp_path / "Documents", depth=0)
        work = FolderNode(name="Work", path=tmp_path / "Documents" / "Work", depth=1)
        projects = FolderNode(
            name="Projects", path=tmp_path / "Documents" / "Work" / "Projects", depth=2
        )
        work.children = [projects]
        root.children = [work]

        paths = root.get_all_paths(relative_to=tmp_path)

        assert "Documents" in paths
        assert "Documents/Work" in paths
        assert "Documents/Work/Projects" in paths

    def test_get_all_paths_without_relative(self, tmp_path):
        """Test getting all paths without relative_to."""
        root = FolderNode(name="test", path=tmp_path / "test", depth=0)
        paths = root.get_all_paths()
        assert len(paths) == 1
        assert "test" in paths[0]


class TestFolderScanResult:
    """Tests for FolderScanResult."""

    def test_empty_result(self):
        """Test empty scan result."""
        result = FolderScanResult()
        assert result.roots == []
        assert result.total_folders == 0
        assert result.max_depth_reached == 0

    def test_to_tree_string(self, tmp_path):
        """Test combined tree string."""
        root1 = FolderNode(name="Documents", path=tmp_path / "Documents", depth=0)
        root2 = FolderNode(name="Downloads", path=tmp_path / "Downloads", depth=0)

        result = FolderScanResult(roots=[root1, root2])
        tree = result.to_tree_string()

        assert "Documents/" in tree
        assert "Downloads/" in tree

    def test_get_all_paths(self, tmp_path):
        """Test getting all paths from result."""
        root = FolderNode(name="Documents", path=tmp_path / "Documents", depth=0)
        work = FolderNode(name="Work", path=tmp_path / "Documents" / "Work", depth=1)
        root.children = [work]

        result = FolderScanResult(roots=[root])
        paths = result.get_all_paths()

        assert "Documents" in paths
        assert "Documents/Work" in paths

    def test_to_prompt_context(self, tmp_path):
        """Test prompt context generation."""
        root = FolderNode(name="Documents", path=tmp_path / "Documents", depth=0)
        work = FolderNode(name="Work", path=tmp_path / "Documents" / "Work", depth=1)
        personal = FolderNode(
            name="Personal", path=tmp_path / "Documents" / "Personal", depth=1
        )
        root.children = [work, personal]

        result = FolderScanResult(roots=[root], total_folders=3)
        context = result.to_prompt_context()

        assert "- Documents" in context
        assert "- Documents/Work" in context
        assert "- Documents/Personal" in context

    def test_to_prompt_context_truncation(self, tmp_path):
        """Test prompt context truncation."""
        # Create many folders
        root = FolderNode(name="root", path=tmp_path / "root", depth=0)
        for i in range(20):
            child = FolderNode(
                name=f"folder{i}", path=tmp_path / "root" / f"folder{i}", depth=1
            )
            root.children.append(child)

        result = FolderScanResult(roots=[root], total_folders=21)
        context = result.to_prompt_context(max_folders=10)

        # Should truncate and show note
        assert "... and" in context
        assert "more folders" in context


class TestFolderScanner:
    """Tests for FolderScanner."""

    @pytest.fixture
    def scanner(self):
        """Create a scanner instance."""
        return FolderScanner(max_depth=3)

    @pytest.fixture
    def folder_structure(self, tmp_path):
        """Create a test folder structure."""
        # Documents/
        #   Work/
        #     Projects/
        #       Python/
        #   Personal/
        #     Finance/
        docs = tmp_path / "Documents"
        (docs / "Work" / "Projects" / "Python").mkdir(parents=True)
        (docs / "Personal" / "Finance").mkdir(parents=True)
        return tmp_path

    def test_initialization_defaults(self):
        """Test scanner initialization with defaults."""
        scanner = FolderScanner()
        assert scanner.max_depth == 4
        assert EXCLUDED_FOLDERS.issubset(scanner.excluded_folders)

    def test_initialization_custom(self):
        """Test scanner with custom settings."""
        scanner = FolderScanner(max_depth=2, excluded_folders={"custom_exclude"})
        assert scanner.max_depth == 2
        assert "custom_exclude" in scanner.excluded_folders

    def test_scan_basic_structure(self, scanner, folder_structure):
        """Test scanning a basic folder structure."""
        result = scanner.scan([folder_structure / "Documents"])

        assert result.total_folders > 0
        paths = result.get_all_paths()
        assert any("Work" in p for p in paths)
        assert any("Personal" in p for p in paths)

    def test_scan_respects_max_depth(self, folder_structure):
        """Test that scanning respects max_depth."""
        scanner = FolderScanner(max_depth=1)
        result = scanner.scan([folder_structure / "Documents"])

        paths = result.get_all_paths()
        # Should have Documents, Work, Personal but NOT Projects (depth 2)
        assert any("Work" in p for p in paths)
        # Check no deep paths
        assert not any("Projects" in p for p in paths)

    def test_scan_excludes_hidden_folders(self, scanner, tmp_path):
        """Test that hidden folders are excluded."""
        visible = tmp_path / "visible"
        hidden = tmp_path / ".hidden"
        visible.mkdir()
        hidden.mkdir()

        result = scanner.scan([tmp_path])
        paths = result.get_all_paths()

        assert any("visible" in p for p in paths)
        assert not any(".hidden" in p for p in paths)

    def test_scan_excludes_known_folders(self, scanner, tmp_path):
        """Test that known excluded folders are skipped."""
        normal = tmp_path / "normal"
        node_modules = tmp_path / "node_modules"
        git = tmp_path / ".git"

        normal.mkdir()
        node_modules.mkdir()
        git.mkdir()

        result = scanner.scan([tmp_path])
        paths = result.get_all_paths()

        assert any("normal" in p for p in paths)
        assert not any("node_modules" in p for p in paths)
        assert not any(".git" in p for p in paths)

    def test_scan_nonexistent_path(self, scanner, tmp_path):
        """Test scanning a nonexistent path."""
        result = scanner.scan([tmp_path / "nonexistent"])
        assert result.total_folders == 0
        assert result.roots == []

    def test_scan_file_instead_of_directory(self, scanner, tmp_path):
        """Test scanning a file path (not directory)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = scanner.scan([test_file])
        assert result.total_folders == 0

    def test_scan_multiple_roots(self, scanner, tmp_path):
        """Test scanning multiple root paths."""
        docs = tmp_path / "Documents"
        downloads = tmp_path / "Downloads"
        docs.mkdir()
        downloads.mkdir()

        result = scanner.scan([docs, downloads])

        assert len(result.roots) == 2
        paths = result.get_all_paths()
        assert any("Documents" in p for p in paths)
        assert any("Downloads" in p for p in paths)

    def test_scan_empty_directory(self, scanner, tmp_path):
        """Test scanning an empty directory."""
        empty = tmp_path / "empty"
        empty.mkdir()

        result = scanner.scan([empty])

        assert result.total_folders == 1
        assert len(result.roots) == 1

    def test_max_depth_calculation(self, folder_structure):
        """Test that max depth is calculated correctly."""
        scanner = FolderScanner(max_depth=10)  # High limit
        result = scanner.scan([folder_structure / "Documents"])

        # Should find the deepest path: Documents/Work/Projects/Python (depth 3)
        assert result.max_depth_reached >= 3

    def test_should_exclude_tilde_folders(self, scanner, tmp_path):
        """Test that folders starting with ~ are excluded."""
        normal = tmp_path / "normal"
        tilde = tmp_path / "~temp"
        normal.mkdir()
        tilde.mkdir()

        result = scanner.scan([tmp_path])
        paths = result.get_all_paths()

        assert any("normal" in p for p in paths)
        assert not any("~temp" in p for p in paths)


class TestScanFoldersForContext:
    """Tests for the convenience function."""

    def test_scan_folders_for_context(self, tmp_path):
        """Test the convenience function."""
        folder = tmp_path / "test"
        subfolder = folder / "subfolder"
        subfolder.mkdir(parents=True)

        result = scan_folders_for_context([folder])

        assert result.total_folders == 2
        paths = result.get_all_paths()
        assert any("test" in p for p in paths)
        assert any("subfolder" in p for p in paths)

    def test_scan_folders_with_custom_depth(self, tmp_path):
        """Test convenience function with custom depth."""
        # Create a deeper structure: a/b/c/d/e (5 levels deep from tmp_path's child)
        deep = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)

        result = scan_folders_for_context([tmp_path], max_depth=2)

        # The scan starts at tmp_path (depth 0), so:
        # - tmp_path is scanned at depth 0
        # - a is at depth 1
        # - b is at depth 2 (max_depth reached, children not scanned)
        # So c, d, e should NOT appear in the paths
        paths = result.get_all_paths()

        # Get just the folder names from paths for easier checking
        folder_names_in_paths = set()
        for p in paths:
            for part in p.split("/"):
                folder_names_in_paths.add(part)

        # a and b should be found
        assert "a" in folder_names_in_paths
        assert "b" in folder_names_in_paths
        # c should NOT be found (it's at depth 3, beyond max_depth=2)
        assert "c" not in folder_names_in_paths
        assert "d" not in folder_names_in_paths
        assert "e" not in folder_names_in_paths
