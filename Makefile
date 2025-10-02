# Makefile for Automated Trading Platform

.PHONY: help build up down logs test clean install dev prod status health

# Default target
help: ## Show this help message
	@echo "Automated Trading Platform - Make Commands"
	@echo "=========================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development commands
install: ## Install dependencies locally
	pip install -r requirements.txt
	pip install -e .

dev: ## Start development environment
	docker-compose --profile dev up -d
	@echo "Development environment started!"
	@echo "API: http://localhost:8001"
	@echo "Jupyter: http://localhost:8888"
	@echo "Grafana: http://localhost:3000"

dev-logs: ## Show development logs
	docker-compose --profile dev logs -f

# Production commands
prod: ## Start production environment
	docker-compose up -d
	@echo "Production environment started!"
	@echo "API: http://localhost:8000"
	@echo "Grafana: http://localhost:3000"

build: ## Build Docker images
	docker-compose build --no-cache

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## Show logs for all services
	docker-compose logs -f

logs-app: ## Show application logs only
	docker-compose logs -f trading-app

# Database commands
db-migrate: ## Run database migrations
	docker-compose exec trading-app python -m backend.migrations.migrate

db-init: ## Initialize database with default data
	docker-compose exec trading-app python -m backend.scripts.init_db

db-init-test: ## Initialize database with test data
	docker-compose exec trading-app python -m backend.scripts.init_db --test-data

db-reset: ## Reset database completely
	docker-compose down -v
	docker-compose up -d postgres redis
	sleep 10
	make db-migrate
	make db-init-test

# Testing commands
test: ## Run all tests
	docker-compose exec trading-app pytest backend/tests/ -v

test-cov: ## Run tests with coverage
	docker-compose exec trading-app pytest backend/tests/ --cov=backend --cov-report=html

test-unit: ## Run unit tests only
	docker-compose exec trading-app pytest backend/tests/test_*.py -v

test-api: ## Run API tests only
	docker-compose exec trading-app pytest backend/tests/test_api.py -v

test-ml: ## Run ML model tests
	docker-compose exec trading-app pytest backend/tests/test_ml_models.py -v

# Code quality commands
lint: ## Run code linting
	docker-compose exec trading-app flake8 backend/
	docker-compose exec trading-app mypy backend/

format: ## Format code
	docker-compose exec trading-app black backend/
	docker-compose exec trading-app isort backend/

# Monitoring commands
status: ## Check status of all services
	docker-compose ps
	@echo "\nHealth checks:"
	@curl -s http://localhost:8000/health | jq . || echo "API not available"

health: ## Detailed health check
	@echo "=== System Health Check ==="
	@echo "Docker containers:"
	@docker-compose ps
	@echo "\nAPI Health:"
	@curl -s http://localhost:8000/health | jq . || echo "❌ API not responding"
	@echo "\nDatabase connectivity:"
	@docker-compose exec -T postgres pg_isready -U trading_user || echo "❌ Database not ready"
	@echo "\nRedis connectivity:"
	@docker-compose exec -T redis redis-cli ping || echo "❌ Redis not responding"

# Trading commands
start-trading: ## Start automated trading
	curl -X POST "http://localhost:8000/api/v1/start"

stop-trading: ## Stop automated trading
	curl -X POST "http://localhost:8000/api/v1/stop"

trading-status: ## Get trading status
	curl -s "http://localhost:8000/api/v1/status" | jq .

portfolio: ## Show current portfolio
	curl -s "http://localhost:8000/api/v1/portfolio" | jq .

signals: ## Show recent signals
	curl -s "http://localhost:8000/api/v1/signals?limit=10" | jq .

performance: ## Show performance metrics
	curl -s "http://localhost:8000/api/v1/performance" | jq .

# Maintenance commands
clean: ## Clean up Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

backup: ## Backup database
	mkdir -p backups
	docker-compose exec -T postgres pg_dump -U trading_user trading_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backed up to backups/ directory"

restore: ## Restore database from backup (usage: make restore BACKUP=filename)
	@if [ -z "$(BACKUP)" ]; then echo "Usage: make restore BACKUP=backup_file.sql"; exit 1; fi
	docker-compose exec -T postgres psql -U trading_user trading_db < $(BACKUP)

# Security commands
security-scan: ## Run security scan
	docker-compose exec trading-app safety check
	docker-compose exec trading-app bandit -r backend/

# Environment commands
env-check: ## Check environment configuration
	@echo "=== Environment Configuration Check ==="
	@echo "Required environment variables:"
	@docker-compose exec -T trading-app python -c "
from backend.config.settings import get_settings
import os
settings = get_settings()
print(f'Environment: {settings.environment}')
print(f'Debug mode: {settings.debug}')
print(f'Database URL: {settings.database_url[:50]}...')
print(f'Redis URL: {settings.redis_url}')
print(f'Binance testnet: {settings.binance_testnet}')
print(f'Coinbase sandbox: {settings.coinbase_sandbox}')
print(f'API keys configured: {bool(settings.binance_api_key and settings.coinbase_api_key)}')
"

# Update commands
update: ## Update dependencies and rebuild
	git pull
	docker-compose build --no-cache
	docker-compose up -d

# Documentation commands
docs: ## Open API documentation
	@echo "Opening API documentation..."
	@echo "Swagger UI: http://localhost:8000/docs"
	@echo "ReDoc: http://localhost:8000/redoc"

jupyter: ## Start Jupyter notebook for analysis
	docker-compose --profile dev up -d jupyter
	@echo "Jupyter started: http://localhost:8888"
	@echo "Token: trading-jupyter-token"

# Quick start commands
quick-start: ## Quick start for new users
	@echo "🚀 Automated Trading Platform Quick Start"
	@echo "========================================"
	@echo "1. Copying environment template..."
	cp .env.example .env
	@echo "2. ⚠️  IMPORTANT: Edit .env file with your API keys!"
	@echo "3. Starting services..."
	make prod
	@echo "4. Waiting for services to start..."
	sleep 30
	@echo "5. Initializing database..."
	make db-migrate
	make db-init-test
	@echo "6. ✅ Setup complete!"
	@echo ""
	@echo "🔗 Access points:"
	@echo "   API Documentation: http://localhost:8000/docs"
	@echo "   Grafana Dashboard: http://localhost:3000 (admin/admin_password)"
	@echo "   Health Check: http://localhost:8000/health"
	@echo ""
	@echo "📚 Next steps:"
	@echo "   make health          # Check system health"
	@echo "   make trading-status  # Check trading status"
	@echo "   make start-trading   # Start automated trading"

# Demo commands
demo: ## Run demo with test data
	make down
	make prod
	sleep 30
	make db-migrate
	make db-init-test
	@echo "🎭 Demo environment ready!"
	@echo "   Grafana: http://localhost:3000 (admin/admin_password)"
	@echo "   API: http://localhost:8000/docs"

# Monitoring shortcuts
monitor: ## Open monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "API Health: http://localhost:8000/health"

# Development helpers
shell: ## Open shell in trading container
	docker-compose exec trading-app /bin/bash

db-shell: ## Open database shell
	docker-compose exec postgres psql -U trading_user trading_db

redis-shell: ## Open Redis shell
	docker-compose exec redis redis-cli

# Performance commands
benchmark: ## Run performance benchmarks
	docker-compose exec trading-app python -m backend.tests.benchmark

load-test: ## Run load tests against API
	@echo "Running load tests..."
	@echo "Note: Requires 'ab' (Apache Bench) to be installed"
	ab -n 100 -c 10 http://localhost:8000/api/v1/status

# Utility commands
reset: ## Complete reset (careful: deletes all data!)
	@echo "⚠️  This will delete all data. Are you sure? [y/N]"
	@read confirm; if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "Resetting everything..."; \
		make clean; \
		make quick-start; \
	else \
		echo "Reset cancelled."; \
	fi