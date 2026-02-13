#!/usr/bin/env bash
# restart.sh — Clean restart of the Mirai service.
# Kills existing processes, clears bytecode cache, and starts fresh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Stopping existing Mirai processes..."
pkill -9 -f 'python.*main.py' 2>/dev/null || true
sleep 2

echo "==> Clearing __pycache__..."
find "$PROJECT_DIR" -path "$PROJECT_DIR/.venv" -prune -o \
     -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null || true

echo "==> Starting Mirai..."
cd "$PROJECT_DIR"
PYTHONPATH=. exec python -B main.py
