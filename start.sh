#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${OUT_DIR:-/data/outputs}"
PORT="${PORT:-8000}"
# Single worker so the GPU is used safely
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
