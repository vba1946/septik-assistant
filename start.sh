#!/bin/bash
printenv | grep -E '^(OPENAI_API_KEY|FLASK_ENV)=' > /app/.env 2>/dev/null || true
if [ "$1" = "simple" ]; then
  touch /app/.simple-mode
fi
if [ -n "$DATA_DIR" ]; then
  mkdir -p "$DATA_DIR"
fi
exec python web_app.py
