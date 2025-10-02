#!/bin/bash

# Advanced Automated Trading Platform - Setup Script
# This script helps you set up the trading platform quickly

set -e

echo "🤖 Advanced Automated Trading Platform Setup"
echo "=============================================="

# Check requirements
echo "📋 Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "✅ Python $PYTHON_VERSION found"
    
    if [[ $(python3 -c 'import sys; print(sys.version_info >= (3, 9))') != "True" ]]; then
        echo "❌ Python 3.9+ required. Please upgrade Python."
        exit 1
    fi
else
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker found"
else
    echo "⚠️  Docker not found. Docker is recommended for easy setup."
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose found"
else
    echo "⚠️  Docker Compose not found. Will use local installation."
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data logs models results config

# Setup environment file
echo "⚙️  Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template"
    echo "⚠️  IMPORTANT: Please edit .env file with your API keys before running!"
else
    echo "✅ .env file already exists"
fi

# Choose installation method
echo ""
echo "🚀 Choose installation method:"
echo "1) Docker (Recommended) - Full stack with monitoring"
echo "2) Local Python - Development setup"
echo "3) Paper Trading Only - Safe testing mode"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🐳 Setting up Docker environment..."
        
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker is required for this option. Please install Docker first."
            exit 1
        fi
        
        echo "📦 Building Docker images..."
        docker-compose build
        
        echo "🔧 Starting services..."
        docker-compose up -d
        
        echo "⏳ Waiting for services to start..."
        sleep 30
        
        echo "✅ Docker setup complete!"
        echo "🌐 Access the platform at:"
        echo "   • Main Platform: http://localhost:8000"
        echo "   • Grafana Dashboard: http://localhost:3000 (admin/grafana_admin_123)"
        echo "   • Trading Dashboard: http://localhost:3001"
        
        echo ""
        echo "📊 Check status with: docker-compose logs -f trading-bot"
        ;;
        
    2)
        echo "🐍 Setting up local Python environment..."
        
        # Create virtual environment
        if [ ! -d "venv" ]; then
            echo "📦 Creating virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate virtual environment
        echo "🔌 Activating virtual environment..."
        source venv/bin/activate
        
        # Install dependencies
        echo "📦 Installing Python dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        
        echo "✅ Local setup complete!"
        echo "🚀 To start the platform:"
        echo "   source venv/bin/activate"
        echo "   python -m backend.main"
        ;;
        
    3)
        echo "📝 Setting up paper trading mode..."
        
        # Configure for paper trading
        sed -i 's/USE_SANDBOX=false/USE_SANDBOX=true/' .env 2>/dev/null || true
        sed -i 's/PAPER_TRADING=false/PAPER_TRADING=true/' .env 2>/dev/null || true
        sed -i 's/BINANCE_TESTNET=false/BINANCE_TESTNET=true/' .env 2>/dev/null || true
        
        # Create virtual environment
        if [ ! -d "venv" ]; then
            echo "📦 Creating virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate and install
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        
        echo "✅ Paper trading setup complete!"
        echo "💰 You can now safely test with virtual money"
        echo "🚀 To start: source venv/bin/activate && python -m backend.main"
        ;;
        
    *)
        echo "❌ Invalid choice. Please run setup again."
        exit 1
        ;;
esac

# Configuration reminder
echo ""
echo "📝 IMPORTANT CONFIGURATION STEPS:"
echo "1. Edit .env file with your exchange API keys"
echo "2. Set your initial balance and risk parameters"
echo "3. Choose your watchlist symbols"
echo "4. Configure notification settings (optional)"

echo ""
echo "🔐 SECURITY REMINDERS:"
echo "• Use API keys with READ-ONLY permissions for safety"
echo "• Start with paper trading mode to test strategies"
echo "• Never share your .env file or API credentials"
echo "• Monitor your trades regularly"

echo ""
echo "📚 USEFUL COMMANDS:"
echo "• View logs: tail -f logs/trading_platform.log"
echo "• Check status: curl http://localhost:8000/health"
echo "• Stop platform: Ctrl+C (local) or docker-compose down (Docker)"

echo ""
echo "🎉 Setup complete! Happy trading! 🚀"
echo "💡 For support, check the README.md or visit our documentation"