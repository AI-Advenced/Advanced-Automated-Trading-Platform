"""
Pytest configuration and fixtures for the trading platform tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from backend.models.database import DatabaseManager, Base
from backend.core.data_collector import DataCollector
from backend.core.ml_models import TradingMLModels
from backend.core.signal_generator import TradingSignalGenerator
from backend.core.portfolio_manager import PortfolioManager
from backend.core.trading_watcher import TradingWatcher
from backend.config.settings import TradingSettings
from backend.utils.logger import TradingLogger

# Test settings
@pytest.fixture(scope="session")
def test_settings():
    """Test configuration settings"""
    return TradingSettings(
        environment="testing",
        debug=True,
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/1",  # Use different DB for tests
        log_level="DEBUG",
        binance_testnet=True,
        coinbase_sandbox=True,
        max_portfolio_risk=0.01,  # Reduced for testing
        signal_confidence_threshold=0.5,  # Lower threshold for testing
    )

# Event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Database fixtures
@pytest.fixture
async def test_db() -> AsyncGenerator[DatabaseManager, None]:
    """Create a test database manager with in-memory SQLite"""
    db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db_manager.init_db()
    yield db_manager
    await db_manager.close()

@pytest.fixture
async def db_session(test_db: DatabaseManager) -> AsyncSession:
    """Create a test database session"""
    async with test_db.get_session() as session:
        yield session

# Mock market data
@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Generate realistic price data
    prices = []
    current_price = 50000.0  # Starting BTC price
    
    for _ in range(len(dates)):
        # Random walk with slight upward bias
        change = np.random.normal(0, 0.02) * current_price
        current_price = max(current_price + change, current_price * 0.95)  # Min 5% drop
        prices.append(current_price)
    
    # Generate OHLCV from prices
    ohlcv_data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(100, 10000)
        
        ohlcv_data.append({
            'timestamp': dates[i],
            'open': prices[i-1] if i > 0 else price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'symbol': 'BTCUSDT'
        })
    
    return pd.DataFrame(ohlcv_data)

@pytest.fixture
def sample_trading_signals():
    """Generate sample trading signals for testing"""
    return [
        {
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'confidence': 0.8,
            'price': 50000.0,
            'timestamp': pd.Timestamp.now(),
            'indicators': {
                'rsi': 65.0,
                'macd': 0.05,
                'sma_20': 49800.0,
                'sma_50': 49500.0
            }
        },
        {
            'symbol': 'ETHUSDT',
            'action': 'SELL',
            'confidence': 0.75,
            'price': 3000.0,
            'timestamp': pd.Timestamp.now(),
            'indicators': {
                'rsi': 75.0,
                'macd': -0.02,
                'sma_20': 3050.0,
                'sma_50': 3100.0
            }
        }
    ]

# Mock exchange data
@pytest.fixture
def mock_exchange_data():
    """Mock exchange data for testing"""
    return {
        'binance': {
            'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            'ticker': {
                'BTCUSDT': {'price': 50000.0, 'volume': 1000000},
                'ETHUSDT': {'price': 3000.0, 'volume': 500000},
                'ADAUSDT': {'price': 0.5, 'volume': 10000000}
            }
        }
    }

# Component fixtures with mocks
@pytest.fixture
async def mock_data_collector(test_settings, sample_ohlcv_data):
    """Create a mocked data collector"""
    collector = DataCollector(test_settings)
    
    # Mock the async methods
    collector.get_historical_data = AsyncMock(return_value=sample_ohlcv_data)
    collector.get_real_time_data = AsyncMock(return_value=sample_ohlcv_data.iloc[-1].to_dict())
    collector.connect_websocket = AsyncMock()
    collector.disconnect_websocket = AsyncMock()
    
    return collector

@pytest.fixture
async def mock_ml_models(test_settings, sample_ohlcv_data):
    """Create a mocked ML models component"""
    ml_models = TradingMLModels(test_settings)
    
    # Mock the methods
    ml_models.prepare_features = MagicMock(return_value=(
        sample_ohlcv_data.iloc[:-1],  # Features
        sample_ohlcv_data['close'].iloc[1:].values  # Target
    ))
    ml_models.train_models = AsyncMock(return_value={'accuracy': 0.85, 'loss': 0.15})
    ml_models.predict = AsyncMock(return_value={
        'prediction': 1,  # Buy signal
        'confidence': 0.8,
        'probabilities': [0.2, 0.8]
    })
    
    return ml_models

@pytest.fixture
async def mock_signal_generator(test_settings, sample_trading_signals):
    """Create a mocked signal generator"""
    signal_gen = TradingSignalGenerator(test_settings)
    
    # Mock the methods
    signal_gen.generate_signals = AsyncMock(return_value=sample_trading_signals)
    signal_gen.calculate_technical_indicators = MagicMock(return_value={
        'rsi': 65.0,
        'macd': 0.05,
        'sma_20': 49800.0,
        'sma_50': 49500.0
    })
    
    return signal_gen

@pytest.fixture
async def mock_portfolio_manager(test_settings):
    """Create a mocked portfolio manager"""
    portfolio_mgr = PortfolioManager(test_settings)
    
    # Mock the methods
    portfolio_mgr.get_portfolio_value = AsyncMock(return_value=10000.0)
    portfolio_mgr.get_positions = AsyncMock(return_value=[
        {'symbol': 'BTCUSDT', 'quantity': 0.1, 'avg_price': 50000.0}
    ])
    portfolio_mgr.calculate_position_size = AsyncMock(return_value=0.02)  # 2% of portfolio
    portfolio_mgr.execute_trade = AsyncMock(return_value={
        'order_id': 'test_order_123',
        'status': 'filled',
        'executed_price': 50000.0,
        'executed_quantity': 0.02
    })
    
    return portfolio_mgr

@pytest.fixture
async def mock_trading_watcher(test_settings, test_db):
    """Create a mocked trading watcher"""
    watcher = TradingWatcher(test_settings, test_db)
    
    # Mock the methods
    watcher.log_trade = AsyncMock()
    watcher.calculate_performance = AsyncMock(return_value={
        'total_return': 0.15,  # 15%
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.05,
        'win_rate': 0.65
    })
    watcher.get_performance_metrics = AsyncMock(return_value={
        'daily_return': 0.01,
        'volatility': 0.02,
        'var_95': 0.025
    })
    
    return watcher

# Test data helpers
@pytest.fixture
def create_test_trades():
    """Helper to create test trade data"""
    def _create_trades(count=10):
        trades = []
        for i in range(count):
            trade = {
                'id': f'trade_{i}',
                'symbol': 'BTCUSDT',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 0.01,
                'price': 50000 + (i * 100),
                'timestamp': pd.Timestamp.now() - pd.Timedelta(hours=i),
                'status': 'FILLED',
                'pnl': np.random.normal(10, 50)  # Random P&L
            }
            trades.append(trade)
        return trades
    return _create_trades

# Mock external APIs
@pytest.fixture
def mock_binance_api():
    """Mock Binance API responses"""
    class MockBinanceAPI:
        @staticmethod
        async def get_symbol_ticker(symbol):
            prices = {
                'BTCUSDT': 50000.0,
                'ETHUSDT': 3000.0,
                'ADAUSDT': 0.5
            }
            return {'symbol': symbol, 'price': prices.get(symbol, 100.0)}
        
        @staticmethod
        async def get_klines(symbol, interval, limit=100):
            # Return mock kline data
            klines = []
            for i in range(limit):
                timestamp = int((pd.Timestamp.now() - pd.Timedelta(hours=i)).timestamp() * 1000)
                klines.append([
                    timestamp,  # Open time
                    "50000.00",  # Open
                    "50100.00",  # High
                    "49900.00",  # Low
                    "50050.00",  # Close
                    "100.00",    # Volume
                    timestamp + 3600000,  # Close time
                    "5000000.00",  # Quote volume
                    1000,  # Count
                    "50.00",  # Taker buy base
                    "2500000.00",  # Taker buy quote
                    "0"  # Ignore
                ])
            return klines
    
    return MockBinanceAPI()

@pytest.fixture
def mock_coinbase_api():
    """Mock Coinbase Pro API responses"""
    class MockCoinbaseAPI:
        @staticmethod
        async def get_product_ticker(product_id):
            prices = {
                'BTC-USD': 50000.0,
                'ETH-USD': 3000.0,
                'ADA-USD': 0.5
            }
            return {
                'trade_id': 12345,
                'price': prices.get(product_id, 100.0),
                'size': '0.01',
                'time': pd.Timestamp.now().isoformat(),
                'bid': prices.get(product_id, 100.0) - 1,
                'ask': prices.get(product_id, 100.0) + 1,
                'volume': '1000'
            }
    
    return MockCoinbaseAPI()

# Test utilities
@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def logger():
    """Create a test logger"""
    return TradingLogger(log_level="DEBUG")

# Async test helpers
def pytest_collection_modifyitems(config, items):
    """Automatically mark async tests"""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

# Custom assertions
class TradingAssertions:
    @staticmethod
    def assert_valid_signal(signal):
        """Assert that a trading signal has valid structure"""
        required_fields = ['symbol', 'action', 'confidence', 'price', 'timestamp']
        for field in required_fields:
            assert field in signal, f"Signal missing required field: {field}"
        
        assert signal['action'] in ['BUY', 'SELL', 'HOLD'], f"Invalid action: {signal['action']}"
        assert 0 <= signal['confidence'] <= 1, f"Invalid confidence: {signal['confidence']}"
        assert signal['price'] > 0, f"Invalid price: {signal['price']}"
    
    @staticmethod
    def assert_valid_trade(trade):
        """Assert that a trade has valid structure"""
        required_fields = ['symbol', 'side', 'quantity', 'price', 'timestamp', 'status']
        for field in required_fields:
            assert field in trade, f"Trade missing required field: {field}"
        
        assert trade['side'] in ['BUY', 'SELL'], f"Invalid side: {trade['side']}"
        assert trade['quantity'] > 0, f"Invalid quantity: {trade['quantity']}"
        assert trade['price'] > 0, f"Invalid price: {trade['price']}"
        assert trade['status'] in ['PENDING', 'FILLED', 'CANCELED', 'REJECTED'], f"Invalid status: {trade['status']}"

@pytest.fixture
def trading_assertions():
    """Provide trading-specific assertions"""
    return TradingAssertions()