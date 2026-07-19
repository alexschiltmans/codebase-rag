"""Unit tests for local folder ingestion feature."""

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.data_ingestion.git_loader import GitLoader


class TestGitLoaderLocalDirectory:
    """Tests for GitLoader with non-git local directories."""

    def test_clone_or_pull_returns_none_for_non_git_dir(self, tmp_path: Path) -> None:
        """Non-git local directory should return None, not raise."""
        loader = GitLoader(repo_url=None, local_path=tmp_path)
        result = loader.clone_or_pull()
        assert result is None

    def test_get_file_paths_works_for_non_git_dir(self, tmp_path: Path) -> None:
        """get_file_paths should find files in a plain local directory."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# readme")

        loader = GitLoader(repo_url=None, local_path=tmp_path)
        paths = loader.get_file_paths(included_dirs=["src"], included_files=["README.md"])

        filenames = {p.name for p in paths}
        assert "main.py" in filenames
        assert "README.md" in filenames

    def test_clone_new_repo_raises_without_url(self) -> None:
        """_clone_new_repo should raise ValueError when repo_url is None."""
        loader = GitLoader(repo_url=None, local_path=Path("/nonexistent"))
        with pytest.raises(ValueError, match="no repo_url provided"):
            loader._clone_new_repo()


class TestPipelineResolveRepoSource:
    """Tests for IngestPipeline._resolve_repo_source."""

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_resolve_local_folder(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant: MagicMock, tmp_path: Path
    ) -> None:
        from codebase_rag.data_ingestion.pipeline import IngestPipeline

        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = MagicMock()

        pipeline = IngestPipeline(repo_url=str(tmp_path))

        repo_name, local_path, git_loader, is_local = pipeline._resolve_repo_source(str(tmp_path))

        assert repo_name == tmp_path.name
        assert local_path == tmp_path
        assert git_loader.repo_url is None
        assert is_local is True

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_resolve_github_url(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant: MagicMock
    ) -> None:
        from codebase_rag.data_ingestion.pipeline import IngestPipeline

        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = MagicMock()

        pipeline = IngestPipeline(repo_url="https://github.com/owner/my-repo")

        repo_name, local_path, git_loader, is_local = pipeline._resolve_repo_source("https://github.com/owner/my-repo")

        assert repo_name == "my-repo"
        assert local_path == Path("/tmp/repos/my-repo")
        assert git_loader.repo_url == "https://github.com/owner/my-repo"
        assert is_local is False


class TestOpenFolderDialog:
    """Tests for _pick_folder_path and _open_folder_dialog."""

    @patch("codebase_rag.app.components.subprocess.run")
    @patch("codebase_rag.app.components.sys")
    def test_macos_dialog(self, mock_sys: MagicMock, mock_run: MagicMock) -> None:
        from codebase_rag.app.components import _pick_folder_path

        mock_sys.platform = "darwin"
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="/Users/test/project/\n", stderr="")

        result = _pick_folder_path()

        assert result == ("/Users/test/project", None)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "osascript" in cmd[0]
        # The dialog must run via an activated System Events, otherwise it
        # opens unfocused behind the browser (or on another Space entirely).
        script = cmd[2]
        assert "System Events" in script
        assert "activate" in script

    @patch("codebase_rag.app.components.subprocess.run")
    @patch("codebase_rag.app.components.sys")
    def test_windows_dialog(self, mock_sys: MagicMock, mock_run: MagicMock) -> None:
        from codebase_rag.app.components import _pick_folder_path

        mock_sys.platform = "win32"
        mock_run.return_value = CompletedProcess(
            args=[], returncode=0, stdout="C:\\Users\\test\\project\\\n", stderr=""
        )

        result = _pick_folder_path()

        assert result == ("C:\\Users\\test\\project", None)
        mock_run.assert_called_once()
        assert "powershell" in mock_run.call_args[0][0][0]

    @patch("codebase_rag.app.components.shutil.which", return_value="/usr/bin/zenity")
    @patch("codebase_rag.app.components.subprocess.run")
    @patch("codebase_rag.app.components.sys")
    def test_linux_zenity_dialog(self, mock_sys: MagicMock, mock_run: MagicMock, mock_which: MagicMock) -> None:
        from codebase_rag.app.components import _pick_folder_path

        mock_sys.platform = "linux"
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="/home/test/project\n", stderr="")

        result = _pick_folder_path()

        assert result == ("/home/test/project", None)

    @patch("codebase_rag.app.components.subprocess.run")
    @patch("codebase_rag.app.components.sys")
    def test_cancel_returns_no_path_and_no_error(self, mock_sys: MagicMock, mock_run: MagicMock) -> None:
        from codebase_rag.app.components import _pick_folder_path

        mock_sys.platform = "darwin"
        mock_run.return_value = CompletedProcess(
            args=[], returncode=1, stdout="", stderr="execution error: User cancelled. (-128)"
        )

        assert _pick_folder_path() == (None, None)

    @patch("codebase_rag.app.components.subprocess.run")
    @patch("codebase_rag.app.components.sys")
    def test_automation_denial_surfaces_permission_hint(self, mock_sys: MagicMock, mock_run: MagicMock) -> None:
        from codebase_rag.app.components import _pick_folder_path

        mock_sys.platform = "darwin"
        mock_run.return_value = CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="execution error: Not authorized to send Apple events to System Events. (-1743)",
        )

        path, error = _pick_folder_path()

        assert path is None
        assert error is not None
        assert "Automation" in error

    @patch("codebase_rag.app.components.subprocess.run", side_effect=OSError("no such command"))
    @patch("codebase_rag.app.components.sys")
    def test_oserror_surfaces_error_message(self, mock_sys: MagicMock, mock_run: MagicMock) -> None:
        from codebase_rag.app.components import _pick_folder_path

        mock_sys.platform = "darwin"

        path, error = _pick_folder_path()

        assert path is None
        assert error is not None
        assert "no such command" in error

    @patch("codebase_rag.app.components.shutil.which", return_value=None)
    @patch("codebase_rag.app.components.sys")
    def test_linux_no_dialog_tool(self, mock_sys: MagicMock, mock_which: MagicMock) -> None:
        from codebase_rag.app.components import _pick_folder_path

        mock_sys.platform = "linux"

        path, error = _pick_folder_path()

        assert path is None
        assert error is not None
        assert "zenity" in error

    def test_open_folder_dialog_does_not_stack_dialogs(self) -> None:
        """A second Browse click while a dialog is open must not spawn a
        second dialog whose result would overwrite the first."""
        import threading

        import codebase_rag.app.components as comp

        release = threading.Event()
        calls: list[int] = []

        def slow_pick() -> tuple[str | None, str | None]:
            calls.append(1)
            release.wait(timeout=5)
            return None, None

        original_thread = comp._folder_dialog_thread
        comp._folder_dialog_thread = None
        try:
            with patch.object(comp, "_pick_folder_path", side_effect=slow_pick):
                comp._open_folder_dialog()
                first_thread = comp._folder_dialog_thread
                comp._open_folder_dialog()
                assert comp._folder_dialog_thread is first_thread
                release.set()
                assert first_thread is not None
                first_thread.join(timeout=5)
            assert calls == [1]
        finally:
            comp._folder_dialog_thread = original_thread

    def test_dialog_result_delivers_path(self) -> None:
        """The background thread's selection must land in the shared result
        dict that the sidebar fragment polls."""
        import codebase_rag.app.components as comp

        original_thread = comp._folder_dialog_thread
        comp._folder_dialog_thread = None
        try:
            with comp._folder_dialog_lock:
                comp._folder_dialog_result.clear()
            with patch.object(comp, "_pick_folder_path", return_value=("/Users/test/project", None)):
                comp._open_folder_dialog()
                assert comp._folder_dialog_thread is not None
                comp._folder_dialog_thread.join(timeout=5)
            with comp._folder_dialog_lock:
                assert comp._folder_dialog_result.get("path") == "/Users/test/project"
        finally:
            with comp._folder_dialog_lock:
                comp._folder_dialog_result.clear()
            comp._folder_dialog_thread = original_thread
