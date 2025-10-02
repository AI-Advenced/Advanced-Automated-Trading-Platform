# Advanced Trading Platform Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading -s /bin/bash trading

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/results /app/config && \
    chown -R trading:trading /app

# Copy application code
COPY --chown=trading:trading backend/ ./backend/
COPY --chown=trading:trading migrations/ ./migrations/
COPY --chown=trading:trading pyproject.toml ./
COPY --chown=trading:trading README.md ./
COPY --chown=trading:trading .env.example ./

# Create startup script
COPY --chown=trading:trading <<EOF /app/start.sh
#!/bin/bash
set -e

# Wait for dependencies (Redis, InfluxDB) to be ready
if [ "\$WAIT_FOR_REDIS" = "true" ]; then
    echo "Waiting for Redis..."
    while ! redis-cli -h redis ping > /dev/null 2>&1; do
        sleep 1
    done
    echo "Redis is ready!"
fi

if [ "\$WAIT_FOR_INFLUXDB" = "true" ]; then
    echo "Waiting for InfluxDB..."
    while ! curl -f http://influxdb:8086/ping > /dev/null 2>&1; do
        sleep 1
    done
    echo "InfluxDB is ready!"
fi

# Initialize database if needed
if [ "\$INIT_DATABASE" = "true" ]; then
    echo "Initializing database..."
    python -c "
from backend.core.trading_watcher import DatabaseManager
db = DatabaseManager('/app/data/trading_data.db')
print('Database initialized successfully!')
"
fi

# Start the trading platform
echo "Starting Advanced Trading Platform..."
exec python -m backend.main
EOF

RUN chmod +x /app/start.sh

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "
import sys
try:
    from backend.main import TradingPlatform
    print('Health check passed')
    sys.exit(0)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"

# Expose ports
EXPOSE 8000

# Set default environment variables
ENV WAIT_FOR_REDIS=true \
    WAIT_FOR_INFLUXDB=true \
    INIT_DATABASE=true \
    LOG_LEVEL=INFO

# Default command
CMD ["/app/start.sh"]

# Labels for metadata
LABEL maintainer="Trading Bot Developer <dev@tradingbot.com>" \
      version="1.0.0" \
      description="Advanced Automated Trading Platform with AI/ML Integration" \
      org.opencontainers.image.title="Advanced Trading Platform" \
      org.opencontainers.image.description="AI-powered automated cryptocurrency trading platform" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Trading Bot Solutions" \
      org.opencontainers.image.licenses="MIT"