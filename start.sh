#!/bin/bash

# Print the environment for debugging
echo "Current environment variables:"
printenv

# Always use port 8000 regardless of PORT environment variable
echo "Starting MCP Server on port 8000"
exec uv run main.py --reload --workers 1
