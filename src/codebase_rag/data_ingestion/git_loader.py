"""Module for retrieving and loading Git repositories."""

import logging
import shutil
from pathlib import Path

import git

from ..config import Config

logger = logging.getLogger(__name__)


class GitLoader:
    """Class for retrieving and managing Git repositories.

    This class handles cloning or updating a Git repository to a local path
    and provides methods to access specific files or directories.
    """

    def __init__(self, repo_url: str | None = None, local_path: Path | None = None) -> None:  # noqa: FBT001
        """Initialize the GitLoader.

        Args:
            repo_url: URL of the Git repository to clone.
            local_path: Local path where the repository should be cloned.
        """
        config = Config.get_instance()
        self.repo_url: str | None = repo_url
        self.local_path = local_path or config.repo_local_path
        self.repo: git.Repo | None = None

    def clone_or_pull(self) -> git.Repo | None:
        """Clone the repository or pull the latest changes if it already exists.

        Returns:
            The Git repository object, or None for non-git local directories.
        """
        # Local directory without git, used for local folder ingestion
        if self.repo_url is None and self.local_path.exists() and not (self.local_path / ".git").exists():
            logger.info("Using local directory (non-git) at %s", self.local_path)
            return None

        # For testing purposes, we may have a local repo without a remote
        if self.repo_url is None and self.local_path.exists() and (self.local_path / ".git").exists():
            logger.info("Using existing local repository without remote at %s", self.local_path)
            self.repo = git.Repo(self.local_path)
            return self.repo

        # Normal case, repo with remote
        if self.local_path.exists() and (self.local_path / ".git").exists():
            self._pull_existing_repo()
        else:
            self._clone_new_repo()

        return self.repo

    def _pull_existing_repo(self) -> None:
        """Pull latest changes for an existing local repository."""
        logger.info("Repository exists at %s, attempting to pull latest changes", self.local_path)
        self.repo = git.Repo(self.local_path)

        if "origin" not in [remote.name for remote in self.repo.remotes]:
            logger.warning("Repository does not have an 'origin' remote, skipping pull")
            return

        try:
            self.repo.remotes.origin.pull()
            logger.info("Successfully pulled latest changes")
        except Exception as e:
            logger.warning("Failed to pull from origin: %s", e)

    def _clone_new_repo(self) -> None:
        """Clone a repository from the remote URL."""
        if self.repo_url is None:
            raise ValueError("Cannot clone repository: no repo_url provided")

        if self.local_path.exists():
            logger.info("Cleaning existing directory at %s", self.local_path)
            shutil.rmtree(self.local_path)

        logger.info("Cloning repository from %s to %s", self.repo_url, self.local_path)
        self.repo = git.Repo.clone_from(self.repo_url, self.local_path)

    def get_file_paths(
        self,
        included_dirs: list[str] | None = None,
        included_files: list[str] | None = None,
        excluded_extensions: list[str] | None = None,
    ) -> list[Path]:
        """Get the paths of files to process based on inclusion/exclusion criteria.

        Args:
            included_dirs: List of directory paths to include (relative to repo root).
            included_files: List of specific files to include (relative to repo root).
            excluded_extensions: List of file extensions to exclude.

        Returns:
            List[Path]: List of file paths to process.
        """
        if self.repo is None and not self.local_path.exists():
            self.clone_or_pull()

        if included_dirs is None:
            included_dirs = ["docs", "src", "tests"]

        if included_files is None:
            included_files = ["README.md", "pyproject.toml"]

        if excluded_extensions is None:
            excluded_extensions = [".pyc", ".git", ".png", ".jpg", ".jpeg", ".gif"]

        file_paths = self._collect_root_files(included_files)
        file_paths.extend(self._collect_dir_files(included_dirs, excluded_extensions))

        logger.info("Found %d files to process", len(file_paths))
        return file_paths

    def _collect_root_files(self, included_files: list[str]) -> list[Path]:
        """Collect specific files from the repository root."""
        paths: list[Path] = []
        for file_name in included_files:
            file_path = self.local_path / file_name
            if file_path.exists() and file_path.is_file():
                paths.append(file_path)
        return paths

    def _collect_dir_files(self, included_dirs: list[str], excluded_extensions: list[str]) -> list[Path]:
        """Collect files from included directories, excluding certain extensions."""
        paths: list[Path] = []
        for dir_name in included_dirs:
            dir_path = self.local_path / dir_name
            if not (dir_path.exists() and dir_path.is_dir()):
                continue
            for file_path in dir_path.glob("**/*"):
                if self._should_include_file(file_path, excluded_extensions):
                    paths.append(file_path)
        return paths

    @staticmethod
    def _should_include_file(file_path: Path, excluded_extensions: list[str]) -> bool:
        """Check whether a file should be included based on extension and path."""
        if not file_path.is_file():
            return False
        if any(file_path.name.endswith(ext) for ext in excluded_extensions):
            return False
        return not any(part.startswith(".") for part in file_path.parts)
