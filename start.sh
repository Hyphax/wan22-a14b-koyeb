#!/usr/bin/env bash
set -e
mkdir -p "$OUT_DIR"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
