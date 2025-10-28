import asyncio

from main import mcp


async def main():
    # Initialize and run the server
    await mcp.run_streamable_http_async()

if __name__ == "__main__":
    asyncio.run(main())