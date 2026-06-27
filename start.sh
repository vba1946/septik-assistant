#!/bin/bash
# Создаём .env из переменных окружения (резерв для Railway)
printenv | grep -E '^(OPENAI_API_KEY|FLASK_ENV)=' > /app/.env 2>/dev/null || true
# Запуск приложения
exec python web_app.py
