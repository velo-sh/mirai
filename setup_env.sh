#!/bin/bash

# Mirai Environment Setup Script
# This script ensures uv is used to manage the environment and dependencies.

set -e

# Add ~/.local/bin to PATH if not already present (for uv)
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment and install dependencies
echo "Synchronizing project environment..."
uv sync

# Activate the virtual environment
echo "To activate the environment, run:"
echo "source .venv/bin/activate"
