"""FastAPI service exposing emoji directory search backed by Qdrant and OpenAI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Iterable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, Runner, function_tool
from openai import OpenAI
from openai.types import Reasoning, ReasoningEffort
from qdrant_client import QdrantClient

from main import _encode_query
from model.emoji_asset import EmojiAssetMatch

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMOJI_AGENT_MODEL", "gpt-5-mini")
DEFAULT_COLLECTION = os.getenv("EMOJI_QDRANT_COLLECTION", "emoji_asset_directories")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.critical("❌ OpenAI API key not found. Make sure it's in your .env file. Shutting down.")
    exit(1)
RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-5-mini")
DEFAULT_TOP_K = 1

_openai_client = OpenAI()
app = FastAPI(title="Emoji Agent API", version="1.0.0")


def _build_qdrant_client(timeout: float = 60.0) -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("Environment variable 'QDRANT_URL' is required for emoji agent.")

    api_key = os.getenv("QDRANT_API_KEY") or os.getenv("QDRANT_KEY")
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def _format_results(points: Iterable) -> str:
    ordered_points = sorted(
        points,
        key=lambda point: getattr(point, "score", 0.0),
        reverse=True,
    )

    matches: list[EmojiAssetMatch] = []

    for point in ordered_points:
        payload = point.payload or {}
        directory = payload.get("directory")
        if isinstance(directory, str) and directory:
            matches.append(EmojiAssetMatch(directory=directory))

    return json.dumps([match.to_raw_url() for match in matches])


def _resolve_limit(requested_limit: int | None) -> int:
    if requested_limit is None:
        return DEFAULT_TOP_K

    if requested_limit <= 0:
        raise ValueError("'limit' must be a positive integer.")

    return min(requested_limit, 50)


@function_tool
def search_emoji_directories(query: str, limit: int | None = None) -> str:
    """Search the emoji Qdrant collection and return a formatted list of matches."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("'query' must be a non-empty string.")

    top_k = _resolve_limit(limit)
    collection = DEFAULT_COLLECTION

    query_vector = _encode_query(normalized_query)
    qclient = _build_qdrant_client()
    points = qclient.search(
        collection_name=collection,
        query_vector=query_vector.tolist(),
        limit=top_k,
    )

    reranked = list(points)
    logger.info(
        "search_emoji_directories | query='%s' | collection='%s' | limit=%s | results=%s",
        normalized_query,
        collection,
        top_k,
        len(reranked),
    )

    return _format_results(reranked)


def _rerank_points(points: Iterable, *, query: str, top_k: int) -> list:
    original_points: list = list(points)
    if not original_points:
        return []

    candidates: list[dict[str, str | int]] = []
    for idx, point in enumerate(original_points):
        payload = point.payload or {}
        directory = payload.get("directory")
        if isinstance(directory, str) and directory:
            candidates.append({"id": idx, "text": directory})

    if not candidates:
        return original_points

    request_payload = {
        "query": query,
        "candidates": candidates,
        "return_top": min(top_k, len(candidates)),
    }

    try:
        response = _openai_client.responses.create(
            model=RERANK_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You rerank emoji asset directory candidates."
                        " Return the indices of the best matches in descending order of relevance as a JSON object"
                        " with the shape {\"ranking\": [index,...]}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False),
                },
            ],
        )
        content = (response.output_text or "").strip()
        if not content:
            raise ValueError("empty response")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as json_exc:
            logger.debug("Raw rerank response: %s", content)
            raise json_exc

        ranking = parsed.get("ranking", [])
    except Exception as exc:  # noqa: BLE001 - fall back to vector order
        logger.warning("Rerank failed, using vector order. Error: %s", exc)
        return original_points

    reordered: list = []
    seen: set[int] = set()
    for idx in ranking:
        if isinstance(idx, int) and 0 <= idx < len(original_points) and idx not in seen:
            reordered.append(original_points[idx])
            seen.add(idx)
        if len(reordered) >= top_k:
            break

    if len(reordered) < len(original_points):
        for idx, point in enumerate(original_points):
            if idx not in seen:
                reordered.append(point)
            if len(reordered) >= len(original_points):
                break

    return reordered


emoji_agent = Agent(
    name="EmojiDirectoryAssistant",
    instructions=(
        f"""
        You help users find emoji asset directories stored in Qdrant.
        Analyze the input sentences to identify the most relevant keywords, 
        then use those keywords to query the `search_emoji_directories` tool to retrieve the most relevant directories.
        """
    ),
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="medium"),
    ),
    tools=[search_emoji_directories],
)


async def run_emoji_agent(
        query: str,
        *,
        limit: int = DEFAULT_TOP_K,
        max_attempts: int = 1,
        initial_retry_delay: float = 2.0,
) -> str:
    prompt = (
        f"Find emoji asset directories for the query: '{query}'."
        f" Return at most {limit} results."
    )

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = await Runner.run(
                starting_agent=emoji_agent,
                input=prompt,
            )

            return result.final_output.strip()
        except Exception as exc:  # noqa: BLE001 - surface after retries
            last_error = exc
            if attempt >= max_attempts:
                logger.error(
                    "run_emoji_agent failed after %s attempts: %s", attempt, exc
                )
                raise

            delay = initial_retry_delay * attempt
            logger.warning(
                "run_emoji_agent attempt %s/%s failed (%s); retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None  # Defensive - should never reach.
    raise last_error


class EmojiSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language emoji description")
    limit: int | None = Field(None, ge=1, le=50, description="Maximum number of results to return")


class EmojiSearchResponse(BaseModel):
    query: str
    limit: int
    results: list[str]
    agent_output: str | None = Field(
        default=None,
        description="Raw agent response when available",
    )


@app.post("/search", response_model=EmojiSearchResponse)
async def search_emoji(request: EmojiSearchRequest) -> EmojiSearchResponse:
    """Entry point for querying the emoji agent via HTTP."""

    limit = _resolve_limit(request.limit)

    try:
        agent_output = await run_emoji_agent(request.query, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent run failed for query '%s'", request.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        results = json.loads(agent_output)
        if not isinstance(results, list):
            raise ValueError("Agent output is not a list")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to parse agent output as JSON: %s", exc)
        results = [agent_output]

    return EmojiSearchResponse(
        query=request.query,
        limit=limit,
        results=results,
        agent_output=None if results == [agent_output] else agent_output,
    )
