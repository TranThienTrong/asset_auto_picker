"""Utilities for syncing cached asset directory names into a Qdrant collection."""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from main import _load_cached_directories, _load_embedding_model
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _resolve_qdrant_credentials() -> tuple[str, str | None]:
    """Return the Qdrant URL and API key from environment variables."""

    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("Environment variable 'QDRANT_URL' is required for Qdrant ingestion.")

    api_key = os.getenv("QDRANT_API_KEY") or os.getenv("QDRANT_KEY")
    return url, api_key


def ingest_cached_directories_to_qdrant(
    *,
    collection_name: str = "emoji_asset_directories",
    payload_key: str = "directory",
    batch_size: int = 64,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> None:
    """Embed cached directory names and upsert them into a Qdrant collection.

    Parameters
    ----------
    collection_name:
        Target Qdrant collection name. Created automatically if it does not exist.
    payload_key:
        Payload field name that will store the raw directory string for each point.
    batch_size:
        Number of vectors to upsert per batch. Tune for large datasets.
    """

    directories = _load_cached_directories()
    if not directories:
        raise RuntimeError("No cached directories found to ingest into Qdrant.")

    model = _load_embedding_model()
    embeddings = model.encode(
        directories,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    url, api_key = _resolve_qdrant_credentials()
    client = QdrantClient(url=url, api_key=api_key, timeout=timeout)

    vector_size = embeddings.shape[1]

    if not client.collection_exists(collection_name=collection_name):
        logger.info("Creating Qdrant collection '%s' (vector size=%s)", collection_name, vector_size)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    logger.info("Ingesting %s directories into Qdrant collection '%s'", len(directories), collection_name)

    def batched_iterable(sequence: Iterable[tuple[str, list[float]]]) -> Iterable[list[models.PointStruct]]:
        batch: list[models.PointStruct] = []
        for directory, vector in sequence:
            point_id = _compute_point_id(directory)
            payload = {
                payload_key: directory,
                "slug": _slugify(directory),
                "category": _infer_category(directory),
            }
            batch.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    vector_pairs = (
        (directory, embedding.tolist())
        for directory, embedding in zip(directories, embeddings, strict=True)
    )

    for batch in batched_iterable(vector_pairs):
        _upsert_with_retries(
            client,
            collection_name=collection_name,
            points=batch,
            max_retries=max_retries,
        )

    logger.info("Finished Qdrant ingestion for collection '%s'", collection_name)


def _compute_point_id(directory: str) -> int:
    """Return a stable integer identifier for a directory string."""

    # Use a deterministic hash to keep point IDs stable across ingestions.
    return abs(hash(directory))


def _slugify(value: str) -> str:
    """Return a lowercase slug suitable for filtering."""

    return value.strip().lower().replace(" ", "_")


def _infer_category(directory: str) -> str:
    """Return a basic category based on path structure."""

    parts = directory.split("/")
    return parts[0] if parts else "unknown"


def _upsert_with_retries(
    client: QdrantClient,
    *,
    collection_name: str,
    points: list[models.PointStruct],
    max_retries: int,
) -> None:
    """Upsert a batch with retry handling for transient HTTP timeouts."""

    attempt = 1
    while True:
        try:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            return
        except (ResponseHandlingException, httpx.TimeoutException) as exc:
            if attempt >= max_retries:
                logger.error(
                    "Upsert failed after %s attempts; last error: %s", attempt, exc
                )
                raise

            delay = min(2 ** attempt, 10)
            logger.warning(
                "Upsert attempt %s/%s failed (%s). Retrying in %ss...",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
            attempt += 1


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    logging.basicConfig(level=logging.INFO)
    ingest_cached_directories_to_qdrant()
