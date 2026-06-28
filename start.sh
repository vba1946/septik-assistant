#!/bin/bash
printenv | grep -E '^(OPENAI_API_KEY|FLASK_ENV|MODE)=' > /app/.env 2>/dev/null || true
if [ "$1" = "simple" ]; then
  echo 'MODE=simple' >> /app/.env
fi
if [ -n "$DATA_DIR" ]; then
  mkdir -p "$DATA_DIR"
fi
exec python web_app.py
