#!/bin/bash
# Production startup script for the trading platform

set -e

echo "Starting Automated Trading Platform..."

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

# Initialize database with default data
echo "Initializing database..."
python -m backend.scripts.init_db

# Start the application
echo "Starting API server..."
exec uvicorn backend.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --access-log \
    --log-level info