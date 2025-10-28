"""Utilities to traverse the assets directory tree."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class AssetTraversalService:
    """Traverse an ``assets`` directory locally or via the GitHub API."""

    def __init__(
        self,
        assets_root: str | Path | None = None,
        *,
        remote_repo_url: str | None = None,
        branch: str = "main",
        github_token: str | None = None,
        assets_subpath: str = "assets",
    ) -> None:
        """Create a traversal service.

        Parameters
        ----------
        assets_root:
            Optional custom path to the local assets directory. When omitted,
            the service assumes an ``assets`` folder at the project root.
        remote_repo_url:
            GitHub repository URL to fetch from (e.g.
            ``https://github.com/owner/repo.git``). When provided, the service
            traverses the remote repository instead of local files.
        branch:
            Git reference (branch, tag, or commit SHA) to inspect when using a
            remote repository.
        github_token:
            Optional personal access token to increase GitHub API rate limits.
            When omitted, the service will consult the ``GITHUB_TOKEN``
            environment variable.
        assets_subpath:
            Relative path to the assets directory within the repository.
        """

        if assets_root and remote_repo_url:
            raise ValueError(
                "Specify either 'assets_root' for local traversal or "
                "'remote_repo_url' for remote traversal, not both."
            )

        self._mode = "remote" if remote_repo_url else "local"

        if self._mode == "local":
            resolved_root = (
                Path(assets_root)
                if assets_root is not None
                else Path(__file__).resolve().parents[2] / "assets"
            )

            if not resolved_root.exists():
                raise FileNotFoundError(
                    f"Assets directory not found at '{resolved_root}'. Provide a valid path."
                )

            if not resolved_root.is_dir():
                raise NotADirectoryError(
                    f"Assets root must be a directory. Got '{resolved_root}'."
                )

            self._assets_root: Path | None = resolved_root
            self._repo_owner = None
            self._repo_name = None
            self._branch = None
            self._assets_subpath = None
            self._github_token = None
        else:
            owner, name = self._parse_github_repo_url(remote_repo_url)  # type: ignore[arg-type]
            self._assets_root = None
            self._repo_owner = owner
            self._repo_name = name
            self._branch = branch
            self._assets_subpath = assets_subpath.strip("/")
            self._github_token = github_token or os.getenv("GITHUB_TOKEN")

    @property
    def assets_root(self) -> Path:
        """Absolute path to the assets root directory (local mode only)."""

        if self._mode != "local" or self._assets_root is None:
            raise RuntimeError("assets_root is only available in local traversal mode.")

        return self._assets_root

    def iter_directory_names(self) -> Iterable[str]:
        """Yield relative directory names found within the assets tree.

        The generator excludes the root directory itself and returns names
        relative to the root. Results are yielded in lexicographical order to
        guarantee deterministic traversal.
        """

        if self._mode == "remote":
            yield from self._iter_remote_directory_names()
            return

        assert self._assets_root is not None  # Narrowing for type checkers

        directories = sorted(
            path
            for path in self._assets_root.rglob("*")
            if path.is_dir() and path != self._assets_root
        )

        for directory in directories:
            yield str(directory.relative_to(self._assets_root))

    def print_directory_names(self) -> None:
        """Print each directory found in the assets tree to stdout."""

        for directory_name in self.iter_directory_names():
            print(directory_name)

    def _iter_remote_directory_names(self) -> Iterator[str]:
        tree = self._fetch_remote_tree()

        if not self._assets_subpath:
            prefix = ""
        else:
            prefix = f"{self._assets_subpath}/"

        directories = set()

        for node in tree:
            if node.get("type") != "tree":
                continue

            path = node.get("path", "")

            if self._assets_subpath:
                if path == self._assets_subpath:
                    continue

                if not path.startswith(prefix):
                    continue

                relative = path[len(prefix) :]
            else:
                relative = path

            if relative:
                directories.add(relative)

        if not directories and self._assets_subpath:
            raise FileNotFoundError(
                f"Assets directory '{self._assets_subpath}' not found in "
                f"{self._repo_owner}/{self._repo_name}@{self._branch}."
            )

        for directory in sorted(directories):
            yield directory

    def _fetch_remote_tree(self) -> list[dict[str, str]]:
        if not all([self._repo_owner, self._repo_name, self._branch is not None]):
            raise RuntimeError("Remote repository details are not configured.")

        api_url = (
            "https://api.github.com/repos/"
            f"{self._repo_owner}/{self._repo_name}/git/trees/"
            f"{self._branch}?recursive=1"
        )

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "asset-generator/0.1",
        }

        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"

        request = Request(api_url, headers=headers)

        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"GitHub API request failed with status {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Unable to reach GitHub: {exc.reason}") from exc

        tree = payload.get("tree")

        if tree is None:
            message = payload.get("message", "Unexpected GitHub API response.")
            raise RuntimeError(f"GitHub API response missing tree: {message}")

        return tree

    @staticmethod
    def _parse_github_repo_url(url: str) -> tuple[str, str]:
        """Return the owner and repository name extracted from a GitHub URL."""

        if url.startswith("git@github.com:"):
            path = url.split(":", 1)[1]
        else:
            parsed = urlparse(url)
            if parsed.netloc not in {"github.com", "www.github.com"}:
                raise ValueError(f"Unsupported GitHub URL: {url}")
            path = parsed.path.lstrip("/")

        if path.endswith(".git"):
            path = path[:-4]

        parts = [segment for segment in path.split("/") if segment]

        if len(parts) < 2:
            raise ValueError(f"Unable to parse repository owner and name from '{url}'.")

        return parts[0], parts[1]
