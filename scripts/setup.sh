#!/bin/bash
# Setup script for Automated Trading Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available memory
    total_mem=$(free -m | awk 'NR==2{print $2}')
    if [ "$total_mem" -lt 4096 ]; then
        print_warning "System has less than 4GB RAM. Performance may be affected."
    fi
    
    # Check disk space
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 10 ]; then
        print_warning "Less than 10GB disk space available. Consider freeing up space."
    fi
    
    print_success "System requirements check completed"
}

# Setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        cp .env.example .env
        print_success "Environment file created from template"
        print_warning "Please edit .env file with your API keys and configuration"
        
        if command_exists nano; then
            read -p "Do you want to edit the .env file now? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                nano .env
            fi
        fi
    else
        print_status "Environment file already exists"
    fi
}

# Create required directories
create_directories() {
    print_status "Creating required directories..."
    
    directories=(
        "data"
        "logs" 
        "models"
        "backups"
        "notebooks"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        fi
    done
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    if command_exists docker-compose; then
        docker-compose build
    else
        docker compose build
    fi
    
    print_success "Docker images built successfully"
}

# Start services
start_services() {
    print_status "Starting services..."
    
    if command_exists docker-compose; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    
    print_status "Waiting for services to start..."
    sleep 30
    
    print_success "Services started successfully"
}

# Initialize database
initialize_database() {
    print_status "Initializing database..."
    
    # Wait for database to be ready
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose exec -T postgres pg_isready -U trading_user >/dev/null 2>&1; then
            break
        fi
        attempt=$((attempt + 1))
        print_status "Waiting for database... ($attempt/$max_attempts)"
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Database failed to start within timeout"
        exit 1
    fi
    
    # Run migrations
    if command_exists docker-compose; then
        docker-compose exec -T trading-app python -m backend.migrations.migrate
        docker-compose exec -T trading-app python -m backend.scripts.init_db --test-data
    else
        docker compose exec -T trading-app python -m backend.migrations.migrate  
        docker compose exec -T trading-app python -m backend.scripts.init_db --test-data
    fi
    
    print_success "Database initialized successfully"
}

# Run health check
health_check() {
    print_status "Running health check..."
    
    # Check API health
    max_attempts=10
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            break
        fi
        attempt=$((attempt + 1))
        print_status "Waiting for API... ($attempt/$max_attempts)"
        sleep 3
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "API health check failed"
        exit 1
    fi
    
    # Get health status
    health_response=$(curl -s http://localhost:8000/health)
    print_success "API is healthy: $health_response"
}

# Display access information
show_access_info() {
    echo
    print_success "🚀 Automated Trading Platform Setup Complete!"
    echo
    echo "🔗 Access Points:"
    echo "   API Documentation: http://localhost:8000/docs"
    echo "   Health Check:      http://localhost:8000/health"
    echo "   Grafana Dashboard: http://localhost:3000 (admin/admin_password)"
    echo "   Prometheus:        http://localhost:9090"
    echo
    echo "📚 Quick Commands:"
    echo "   make status        # Check system status"
    echo "   make logs          # View logs"
    echo "   make start-trading # Start automated trading"
    echo "   make help          # View all commands"
    echo
    print_warning "⚠️  Remember to configure your API keys in the .env file!"
    print_warning "⚠️  Always test in sandbox/testnet mode first!"
}

# Main setup function
main() {
    echo "🚀 Automated Trading Platform Setup"
    echo "=================================="
    echo
    
    # Parse command line arguments
    SKIP_BUILD=false
    DEVELOPMENT_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --dev)
                DEVELOPMENT_MODE=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-build    Skip Docker image building"
                echo "  --dev          Setup for development mode"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_requirements
    setup_environment
    create_directories
    
    if [ "$SKIP_BUILD" != true ]; then
        build_images
    fi
    
    start_services
    initialize_database
    health_check
    show_access_info
    
    print_success "Setup completed successfully! 🎉"
}

# Run main function with all arguments
main "$@"