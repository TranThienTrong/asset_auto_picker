from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EmojiAssetMatch:
    """Represents a directory returned from the emoji Qdrant search."""
    # Static remote repository configuration
    REMOTE_REPO_URL = "https://github.com/TranThienTrong/vibe-habit-fluentui-emoji.git"
    REMOTE_BRANCH = "main"
    REMOTE_ASSETS_SUBPATH = "assets"

    directory: str

    def to_json(self) -> str:
        return self.directory

    def to_raw_url(self) -> str:
        remote_path = f"{self.REMOTE_ASSETS_SUBPATH}/{self.directory}"
        return (
            "https://raw.githubusercontent.com/"
            "TranThienTrong/vibe-habit-fluentui-emoji/"
            f"{self.REMOTE_BRANCH}/{remote_path}"
        )
