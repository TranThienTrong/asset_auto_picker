import argparse
import functools
import math
import os
from typing import Any

from colorama import Fore, Style
import httpx
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import logging
import urllib.request

import numpy as np
import uvicorn
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from asset_traversal_service import AssetTraversalService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Initialize FastMCP server
mcp = FastMCP("assetAutoPicker", stateless_http=True)

# Static remote repository configuration
REMOTE_REPO_URL = "https://github.com/TranThienTrong/vibe-habit-fluentui-emoji.git"
REMOTE_BRANCH = "main"
REMOTE_ASSETS_SUBPATH = "assets"

# Local cache of directory names
CACHED_DIRECTORIES_PATH = Path(__file__).resolve().parent / "cached_file_name_short.json"
CACHED_3D_PNG_PATH = Path(__file__).resolve().parent / "cached_3d_png_paths.json"


@dataclass(slots=True)
class Match:
    directory: str
    score: float


@functools.lru_cache(maxsize=1)
def _load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=1)
def _load_directory_embeddings() -> dict[str, np.ndarray]:
    directories = _load_cached_directories()
    if not directories:
        return {}

    model = _load_embedding_model()
    embeddings = model.encode(directories, convert_to_numpy=True, normalize_embeddings=True)
    return {directory: embedding for directory, embedding in zip(directories, embeddings)}


def _encode_query(query: str) -> np.ndarray:
    model = _load_embedding_model()
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return embedding


def _semantic_similarity(query_embedding: np.ndarray, directory_embedding: np.ndarray) -> float:
    return float(np.dot(query_embedding, directory_embedding))


def _load_cached_directories() -> list[str]:
    payload = json.loads(CACHED_DIRECTORIES_PATH.read_text(encoding="utf-8"))

    directories = payload.get("directories")

    if not isinstance(directories, list):
        raise ValueError("Cached directory payload is malformed.")

    return [directory for directory in directories if isinstance(directory, str)]


def _load_cached_3d_pngs() -> list[str]:
    payload = json.loads(CACHED_3D_PNG_PATH.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError("Cached 3D PNG list is malformed.")

    return [path for path in payload if isinstance(path, str)]


async def _gather_directories(service: AssetTraversalService) -> list[str]:
    """Return directory names from cache with service fallback."""

    try:
        directories = await asyncio.to_thread(_load_cached_directories)
        if directories:
            return directories
    except FileNotFoundError:
        logger.warning("Cached directory file not found; falling back to traversal.")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse cached directories: %s", exc)

    logger.info("Falling back to directory traversal.")
    return await asyncio.to_thread(lambda: list(service.iter_directory_names()))


def _rank_directories(
        query: str,
        directories: Iterable[str],
        *,
        limit: int,
        min_score: float,
) -> list[Match]:
    """Return ranked directory matches filtered by score threshold."""

    query_embedding = _encode_query(query)

    directory_embeddings = _load_directory_embeddings()

    scored = []
    for directory in directories:
        embedding = directory_embeddings[directory]
        score = _semantic_similarity(query_embedding, embedding) * 100
        scored.append(Match(directory=directory, score=score))

    filtered = [match for match in scored if match.score >= min_score]
    filtered.sort(key=lambda match: match.score, reverse=True)

    return filtered[:limit]


def _relative_png_path(directory: str, filename: str) -> str:
    directory = directory.strip("/")

    if directory:
        return f"{directory}/3D/{filename}"

    return f"3D/{filename}"


def _find_first_png_local(service: AssetTraversalService, directory: str) -> str | None:
    root = service.assets_root
    target_dir = root / directory / "3D"

    if not target_dir.exists() or not target_dir.is_dir():
        return None

    pngs = sorted(
        path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )

    if not pngs:
        return None

    return _relative_png_path(directory, pngs[0].name)


def _find_first_png_remote(
        directory: str,
        tree: list[dict[str, str]],
        assets_subpath: str,
) -> str | None:
    prefix = f"{assets_subpath}/{directory}/3D/"

    candidates = sorted(
        entry.get("path", "")
        for entry in tree
        if entry.get("type") == "blob"
        and entry.get("path", "").startswith(prefix)
        and entry.get("path", "").lower().endswith(".png")
    )

    if not candidates:
        return None

    first = candidates[0]
    if assets_subpath and first.startswith(f"{assets_subpath}/"):
        return first[len(assets_subpath) + 1:]

    return first


def _filter_cached_pngs_by_directories(
        *,
        cached_paths: Iterable[str],
        directories: Iterable[str],
) -> list[str]:
    normalized_dirs = {directory.strip("/") for directory in directories}

    matched = [
        path
        for path in cached_paths
        if any(path.startswith(f"{directory}/3D/") for directory in normalized_dirs)
    ]

    return matched


def _fetch_github_blob(url: str, github_token: str | None) -> bytes:
    headers = {
        "Accept": "application/vnd.github.raw",
        "User-Agent": "asset_auto_generator/0.1",
    }

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request) as response:
        return response.read()


@mcp.tool("list_directories")
async def list_directories(
        *,
        assets_root: str | None = None,
        github_token: str | None = None,
) -> str:
    """Return newline-separated directories discovered under the assets path."""

    try:
        if assets_root is not None:
            service = AssetTraversalService(assets_root=assets_root)
        else:
            service = AssetTraversalService(
                remote_repo_url=REMOTE_REPO_URL,
                branch=REMOTE_BRANCH,
                github_token=github_token,
                assets_subpath=REMOTE_ASSETS_SUBPATH,
            )

        directories = await _gather_directories(service)
    except Exception as exc:  # noqa: BLE001 - propagate with context
        logger.exception("Unable to list directories")
        raise RuntimeError("Failed to list directories") from exc

    if not directories:
        return "No directories found."

    return "\n".join(directories)


@mcp.tool("find_directory")
async def find_directory(
        query: str,
        *,
        limit: int = 10,
        min_score: float = 60.0,
        assets_root: str | None = None,
        github_token: str | None = None,
) -> str:
    """Return newline-separated directory matches with similarity scores."""

    if limit <= 0:
        raise ValueError("'limit' must be a positive integer.")

    try:
        if assets_root is not None:
            service = AssetTraversalService(assets_root=assets_root)
        else:
            service = AssetTraversalService(
                remote_repo_url=REMOTE_REPO_URL,
                branch=REMOTE_BRANCH,
                github_token=github_token,
                assets_subpath=REMOTE_ASSETS_SUBPATH,
            )

        directories = await _gather_directories(service)
    except Exception as exc:  # noqa: BLE001 - propagate with context
        logger.exception("Unable to search directories")
        raise RuntimeError("Failed to search directories") from exc

    matches = _rank_directories(
        query,
        directories,
        limit=limit,
        min_score=min_score,
    )

    if not matches:
        return f"No directories matching '{query}' were found."

    formatted = [f"{match.directory} (score={match.score:.1f})" for match in matches]
    return "\n".join(formatted)




def _to_raw_url(path: str) -> str:
    remote_path = f"{REMOTE_ASSETS_SUBPATH}/{path}" if REMOTE_ASSETS_SUBPATH else path
    return (
        "https://raw.githubusercontent.com/"
        "TranThienTrong/vibe-habit-fluentui-emoji/"
        f"{REMOTE_BRANCH}/{remote_path}"
    )


@mcp.tool("get_3d_png_asset")
async def get_3d_png_asset(
        query: str,
        *,
        limit: int = 99999,
        min_score: float = 70.0,

) -> list[str]:
    """Return raw GitHub URLs for 3D assets matching the query."""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("'query' must be a non-empty string.")

    if limit <= 0:
        raise ValueError("'limit' must be a positive integer.")

    directory_embeddings = _load_directory_embeddings()

    directories = list(directory_embeddings.keys())

    if not directories:
        return []

    try:
        cached_pngs = await asyncio.to_thread(_load_cached_3d_pngs)
    except Exception as exc:  # noqa: BLE001 - propagate with context
        logger.exception("Unable to load cached 3D PNG paths")
        raise RuntimeError("Failed to load cached 3D PNG paths") from exc

    matches = _rank_directories(
        normalized_query,
        directories,
        limit=limit,
        min_score=min_score,
    )

    if not matches:
        return []

    candidate_dirs = [match.directory for match in matches]
    candidate_pngs = _filter_cached_pngs_by_directories(
        cached_paths=cached_pngs,
        directories=candidate_dirs,
    )

    if not candidate_pngs:
        return []

    seen: set[str] = set()
    urls: list[str] = []

    for path in candidate_pngs:
        url = _to_raw_url(path)
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


# if __name__ == "__main__":
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Run MCP server')
#     parser.add_argument('--host', type=str,
#                         help='Host to bind the server to')
#     parser.add_argument('--port',
#                         help='Port to bind the server to')
#     parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
#     parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
#
#     args = parser.parse_args()
#
#     # Run the server
#     uvicorn.run(
#         "main:main",
#         host=args.host,
#         port=args.port,
#         reload=args.reload,
#         workers=args.workers,
#     )
# MCP SERVER STARTUP
# This section configures and starts the MCP server
if __name__ == "__main__":
    # Get port from environment variable (used by deployment platforms like DigitalOcean)
    port = int(os.environ.get("PORT", 8080))

    # Start the MCP server with HTTP transport
    # - transport="streamable-http": Uses HTTP for communication with MCP clients
    # - host="0.0.0.0": Accepts connections from any IP (needed for remote deployment)
    # - port: The port to listen on
    # - log_level="debug": Enables detailed logging for development
    logger.info(f"Embedding Directories")
    _load_directory_embeddings()
    logger.info(f"Running MCP Server")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, log_level="debug")
