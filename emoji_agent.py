"""OpenAI Agents SDK workflow for querying the emoji Qdrant collection."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Iterable

from dotenv import load_dotenv
from agents import Agent, ModelSettings, Runner, function_tool
from fastmcp import FastMCP
from openai import OpenAI
from qdrant_client import QdrantClient

from main import _encode_query
from model.emoji_asset import EmojiAssetMatch

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMOJI_AGENT_MODEL", "gpt-4o-mini")
DEFAULT_COLLECTION = os.getenv("EMOJI_QDRANT_COLLECTION", "emoji_asset_directories")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.critical("❌ OpenAI API key not found. Make sure it's in your .env file. Shutting down.")
    exit(1)
RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = 5

_openai_client = OpenAI()
mcp = FastMCP("assetAutoPicker", stateless_http=True)

def _build_qdrant_client(timeout: float = 60.0) -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("Environment variable 'QDRANT_URL' is required for emoji agent.")

    api_key = os.getenv("QDRANT_API_KEY") or os.getenv("QDRANT_KEY")
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def _format_results(points: Iterable) -> str:
    matches: list[EmojiAssetMatch] = []

    for point in points:
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

    reranked = _rerank_points(points, query=normalized_query, top_k=top_k)

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

    print("After Reorder: "+repr(reordered))
    return reordered


emoji_agent = Agent(
    name="EmojiDirectoryAssistant",
    instructions=(
        "You help users find emoji asset directories stored in Qdrant."
        " When a user requests emoji assets, call the `search_emoji_directories`"
        " tool to retrieve the most relevant directories"
        " clearly. Include scores only when helpful."
    ),
    model=DEFAULT_MODEL,
    model_settings=ModelSettings(temperature=0.6),
    tools=[search_emoji_directories],
)

@mcp.tool("find_emoji")
async def run_emoji_agent(
        query: str,
        *,
        limit: int = DEFAULT_TOP_K,
        max_attempts: int = 3,
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
                emoji_agent,
                prompt,
                context=(
                    "Always invoke search_emoji_directories with the provided query."
                    f" Use limit={limit}."
                ),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the emoji directory agent via the OpenAI Agents SDK.",
    )
    parser.add_argument("--query", help="User said they won a prize")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum number of results to retrieve (default: 10).",
    )
    parser.add_argument(
        "--model",
        help="Optional override for the OpenAI model used by the agent.",
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()

    if args.model:
        emoji_agent.model = args.model

    output = await run_emoji_agent("Saying hello", limit=args.limit)
    print(output)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main_async())


if __name__ == "__main__":
    # Get port from environment variable (used by deployment platforms like DigitalOcean)
    port = int(os.environ.get("PORT", 8080))

    # Start the MCP server with HTTP transport
    # - transport="streamable-http": Uses HTTP for communication with MCP clients
    # - host="0.0.0.0": Accepts connections from any IP (needed for remote deployment)
    # - port: The port to listen on
    # - log_level="debug": Enables detailed logging for development

    logger.info(f"Running MCP Server")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, log_level="debug")

