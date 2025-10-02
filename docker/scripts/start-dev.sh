#!/bin/bash
# Development startup script for the trading platform

set -e

echo "Starting Automated Trading Platform (Development Mode)..."

# Wait for database to be ready
echo "Waiting for database..."
python -c "
import time
import sys
import asyncio
from backend.models.database import DatabaseManager
from backend.config.settings import get_settings

async def wait_for_db():
    settings = get_settings()
    db_manager = DatabaseManager(settings.database_url)
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            await db_manager.init_db()
            print('Database connection successful!')
            await db_manager.close()
            return
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f'Database not ready (attempt {attempt + 1}/{max_attempts}): {e}')
                time.sleep(2)
            else:
                print('Failed to connect to database after all attempts')
                sys.exit(1)

asyncio.run(wait_for_db())
"

# Run database migrations
echo "Running database migrations..."
python -m backend.migrations.migrate

# Initialize database with test data
echo "Initializing database with test data..."
python -m backend.scripts.init_db --test-data

# Run tests
echo "Running tests..."
pytest backend/tests/ -v --tb=short

# Start the application with hot reload
echo "Starting API server with hot reload..."
exec uvicorn backend.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir backend \
    --access-log \
    --log-level debug