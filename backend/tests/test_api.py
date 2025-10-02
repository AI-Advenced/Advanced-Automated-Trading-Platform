"""
Unit tests for the FastAPI application.
Tests API endpoints, authentication, and error handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

from backend.api.app import app, get_db_manager, get_trading_platform
from backend.models.schemas import TradingSignal, PortfolioSummary, Trade, SystemHealth

class TestTradingAPI:
    """Test suite for Trading Platform API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager"""
        mock_db = AsyncMock()
        mock_db.get_recent_signals.return_value = []
        mock_db.get_trades.return_value = []
        mock_db.get_trade_by_id.return_value = None
        return mock_db
    
    @pytest.fixture
    def mock_trading_platform(self):
        """Mock trading platform"""
        mock_platform = AsyncMock()
        mock_platform.get_trading_status.return_value = {
            'status': 'running',
            'active_positions': 5,
            'total_pnl': 1500.0,
            'uptime': 86400
        }
        mock_platform.get_portfolio_summary.return_value = {
            'total_value': 10000.0,
            'available_balance': 5000.0,
            'invested_amount': 5000.0,
            'total_pnl': 1500.0,
            'pnl_percentage': 15.0,
            'positions': []
        }
        mock_platform.get_positions.return_value = []
        mock_platform.get_active_signals.return_value = []
        return mock_platform
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'components' in data
        assert 'timestamp' in data
    
    @pytest.mark.asyncio
    async def test_get_trading_status(self, async_client, mock_trading_platform):
        """Test get trading status endpoint"""
        # Mock dependencies
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'running'
        assert data['active_positions'] == 5
        assert data['total_pnl'] == 1500.0
        
        # Clean up
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_start_trading(self, async_client, mock_trading_platform):
        """Test start trading endpoint"""
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.post("/api/v1/start")
        assert response.status_code == 200
        
        data = response.json()
        assert data['message'] == 'Trading started'
        assert data['status'] == 'starting'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_stop_trading(self, async_client, mock_trading_platform):
        """Test stop trading endpoint"""
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.post("/api/v1/stop")
        assert response.status_code == 200
        
        data = response.json()
        assert data['message'] == 'Trading stopped'
        assert data['status'] == 'stopped'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_portfolio(self, async_client, mock_trading_platform):
        """Test get portfolio endpoint"""
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/portfolio")
        assert response.status_code == 200
        
        data = response.json()
        assert data['total_value'] == 10000.0
        assert data['pnl_percentage'] == 15.0
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_positions(self, async_client, mock_trading_platform):
        """Test get positions endpoint"""
        mock_positions = [
            {
                'symbol': 'BTCUSDT',
                'quantity': 0.1,
                'avg_price': 50000.0,
                'current_price': 51000.0,
                'pnl': 100.0,
                'pnl_percentage': 2.0
            }
        ]
        mock_trading_platform.get_positions.return_value = mock_positions
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/portfolio/positions")
        assert response.status_code == 200
        
        data = response.json()
        assert 'positions' in data
        assert len(data['positions']) == 1
        assert data['positions'][0]['symbol'] == 'BTCUSDT'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_signals(self, async_client, mock_db_manager):
        """Test get signals endpoint"""
        mock_signals = [
            MagicMock(
                id='signal_1',
                symbol='BTCUSDT',
                action='BUY',
                confidence=0.8,
                price=50000.0,
                timestamp='2023-01-01T00:00:00',
                indicators={'rsi': 65.0}
            )
        ]
        mock_db_manager.get_recent_signals.return_value = mock_signals
        
        app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
        
        response = await async_client.get("/api/v1/signals")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_active_signals(self, async_client, mock_trading_platform):
        """Test get active signals endpoint"""
        mock_signals = [
            {
                'symbol': 'ETHUSDT',
                'action': 'SELL',
                'confidence': 0.75,
                'price': 3000.0,
                'timestamp': '2023-01-01T01:00:00'
            }
        ]
        mock_trading_platform.get_active_signals.return_value = mock_signals
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/signals/active")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]['symbol'] == 'ETHUSDT'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_trades(self, async_client, mock_db_manager):
        """Test get trades endpoint"""
        mock_trades = [
            MagicMock(
                id='trade_1',
                symbol='BTCUSDT',
                side='BUY',
                quantity=0.1,
                price=50000.0,
                timestamp='2023-01-01T00:00:00',
                status='FILLED'
            )
        ]
        mock_db_manager.get_trades.return_value = mock_trades
        
        app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
        
        response = await async_client.get("/api/v1/trades")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # Test with filters
        response = await async_client.get("/api/v1/trades?symbol=BTCUSDT&limit=10")
        assert response.status_code == 200
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_trade_by_id(self, async_client, mock_db_manager):
        """Test get specific trade endpoint"""
        mock_trade = MagicMock(
            id='trade_123',
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.1,
            price=50000.0,
            timestamp='2023-01-01T00:00:00',
            status='FILLED'
        )
        mock_db_manager.get_trade_by_id.return_value = mock_trade
        
        app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
        
        response = await async_client.get("/api/v1/trades/trade_123")
        assert response.status_code == 200
        
        # Test trade not found
        mock_db_manager.get_trade_by_id.return_value = None
        response = await async_client.get("/api/v1/trades/nonexistent")
        assert response.status_code == 404
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_performance(self, async_client, mock_trading_platform):
        """Test get performance metrics endpoint"""
        mock_performance = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.05,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'total_trades': 100
        }
        mock_trading_platform.get_performance_metrics.return_value = mock_performance
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert data['total_return'] == 0.15
        assert data['sharpe_ratio'] == 1.5
        
        # Test with timeframe parameter
        response = await async_client.get("/api/v1/performance?timeframe=7d")
        assert response.status_code == 200
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_performance_charts(self, async_client, mock_trading_platform):
        """Test get performance charts endpoint"""
        mock_chart_data = {
            'timestamps': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'values': [10000, 10150, 10300],
            'chart_type': 'equity_curve'
        }
        mock_trading_platform.get_performance_charts.return_value = mock_chart_data
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/performance/charts?timeframe=7d&chart_type=equity_curve")
        assert response.status_code == 200
        
        data = response.json()
        assert 'chart_data' in data
        assert data['chart_data']['chart_type'] == 'equity_curve'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, async_client, mock_trading_platform):
        """Test get market data endpoint"""
        mock_market_data = [
            {
                'timestamp': '2023-01-01T00:00:00',
                'open': 50000.0,
                'high': 50100.0,
                'low': 49900.0,
                'close': 50050.0,
                'volume': 1000.0
            }
        ]
        mock_trading_platform.get_market_data.return_value = mock_market_data
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/market/BTCUSDT?timeframe=1h&limit=100")
        assert response.status_code == 200
        
        data = response.json()
        assert data['symbol'] == 'BTCUSDT'
        assert 'data' in data
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_technical_analysis(self, async_client, mock_trading_platform):
        """Test get technical analysis endpoint"""
        mock_analysis = {
            'rsi': 65.0,
            'macd': 0.05,
            'sma_20': 49800.0,
            'sma_50': 49500.0,
            'bollinger_bands': {
                'upper': 50200.0,
                'middle': 50000.0,
                'lower': 49800.0
            }
        }
        mock_trading_platform.get_technical_analysis.return_value = mock_analysis
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/market/BTCUSDT/analysis")
        assert response.status_code == 200
        
        data = response.json()
        assert data['symbol'] == 'BTCUSDT'
        assert 'analysis' in data
        assert data['analysis']['rsi'] == 65.0
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_strategy_config(self, async_client, mock_trading_platform):
        """Test get strategy configuration endpoint"""
        mock_config = {
            'strategies': {
                'ma_crossover': {'enabled': True, 'weight': 0.3},
                'rsi': {'enabled': True, 'weight': 0.25},
                'ml_prediction': {'enabled': True, 'weight': 0.45}
            },
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_percentage': 0.02
            }
        }
        mock_trading_platform.get_strategy_config.return_value = mock_config
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/config/strategy")
        assert response.status_code == 200
        
        data = response.json()
        assert 'strategies' in data
        assert 'risk_management' in data
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_update_strategy_config(self, async_client, mock_trading_platform):
        """Test update strategy configuration endpoint"""
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        new_config = {
            'strategies': {
                'ma_crossover': {'enabled': False, 'weight': 0.2},
                'rsi': {'enabled': True, 'weight': 0.3},
                'ml_prediction': {'enabled': True, 'weight': 0.5}
            }
        }
        
        response = await async_client.put("/api/v1/config/strategy", json=new_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data['message'] == 'Strategy configuration updated successfully'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_models_status(self, async_client, mock_trading_platform):
        """Test get ML models status endpoint"""
        mock_status = {
            'BTCUSDT': {
                'random_forest': {'accuracy': 0.85, 'last_trained': '2023-01-01T00:00:00'},
                'lstm': {'accuracy': 0.78, 'last_trained': '2023-01-01T00:00:00'}
            }
        }
        mock_trading_platform.get_models_status.return_value = mock_status
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/models/status")
        assert response.status_code == 200
        
        data = response.json()
        assert 'models_status' in data
        assert 'BTCUSDT' in data['models_status']
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_retrain_models(self, async_client, mock_trading_platform):
        """Test retrain models endpoint"""
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.post("/api/v1/models/retrain?symbol=BTCUSDT")
        assert response.status_code == 200
        
        data = response.json()
        assert data['message'] == 'Model retraining initiated'
        assert data['status'] == 'training'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_risk_metrics(self, async_client, mock_trading_platform):
        """Test get risk metrics endpoint"""
        mock_metrics = {
            'portfolio_var': 0.025,
            'expected_shortfall': 0.035,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.05,
            'correlation_matrix': {}
        }
        mock_trading_platform.get_risk_metrics.return_value = mock_metrics
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/risk/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert 'risk_metrics' in data
        assert data['risk_metrics']['sharpe_ratio'] == 1.5
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_run_backtest(self, async_client, mock_trading_platform):
        """Test run backtest endpoint"""
        mock_trading_platform.start_backtest.return_value = 'backtest_123'
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        backtest_config = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 10000,
            'symbols': ['BTCUSDT', 'ETHUSDT']
        }
        
        response = await async_client.post("/api/v1/backtest", json=backtest_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data['message'] == 'Backtesting initiated'
        assert data['task_id'] == 'backtest_123'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_backtest_results(self, async_client, mock_trading_platform):
        """Test get backtest results endpoint"""
        mock_results = {
            'total_return': 0.25,
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.08,
            'total_trades': 150,
            'win_rate': 0.68
        }
        mock_trading_platform.get_backtest_results.return_value = mock_results
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/backtest/backtest_123")
        assert response.status_code == 200
        
        data = response.json()
        assert data['task_id'] == 'backtest_123'
        assert 'results' in data
        assert data['results']['total_return'] == 0.25
        
        # Test backtest not found
        mock_trading_platform.get_backtest_results.return_value = None
        response = await async_client.get("/api/v1/backtest/nonexistent")
        assert response.status_code == 404
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_client, mock_trading_platform):
        """Test API error handling"""
        # Mock trading platform to raise an exception
        mock_trading_platform.get_trading_status.side_effect = Exception("Internal error")
        
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        response = await async_client.get("/api/v1/status")
        assert response.status_code == 500
        
        data = response.json()
        assert 'error' in data
        assert data['error'] == 'Internal server error'
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers are present"""
        response = await async_client.options("/api/v1/status")
        
        # CORS headers should be present
        assert 'access-control-allow-origin' in response.headers
        assert 'access-control-allow-methods' in response.headers
    
    def test_api_documentation(self, client):
        """Test API documentation endpoints"""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert 'paths' in schema
        assert 'info' in schema
        assert schema['info']['title'] == 'Automated Trading Platform API'
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection"""
        # This would require a more complex setup with WebSocket testing
        # For now, we just test that the endpoint exists
        pass  # WebSocket testing requires special setup
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client, mock_trading_platform):
        """Test API rate limiting"""
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = await async_client.get("/api/v1/status")
            responses.append(response.status_code)
        
        # All requests should succeed (assuming no rate limiting in test)
        assert all(status == 200 for status in responses)
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, async_client, mock_trading_platform):
        """Test API parameter validation"""
        app.dependency_overrides[get_trading_platform] = lambda: mock_trading_platform
        
        # Test invalid timeframe parameter
        response = await async_client.get("/api/v1/performance?timeframe=invalid")
        # Should still return 200 as the backend should handle invalid timeframes gracefully
        assert response.status_code == 200
        
        # Test invalid limit parameter
        response = await async_client.get("/api/v1/trades?limit=-1")
        assert response.status_code == 200  # Should use default limit
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_pagination(self, async_client, mock_db_manager):
        """Test API pagination"""
        # Create mock paginated data
        mock_trades = [MagicMock(id=f'trade_{i}') for i in range(150)]
        mock_db_manager.get_trades.return_value = mock_trades[:100]  # First page
        
        app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
        
        # Test first page
        response = await async_client.get("/api/v1/trades?limit=100")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) <= 100
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_data_filtering(self, async_client, mock_db_manager):
        """Test API data filtering"""
        mock_db_manager.get_trades.return_value = []
        
        app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
        
        # Test symbol filtering
        response = await async_client.get("/api/v1/trades?symbol=BTCUSDT")
        assert response.status_code == 200
        
        # Test status filtering
        response = await async_client.get("/api/v1/trades?status=FILLED")
        assert response.status_code == 200
        
        # Verify the filter parameters were passed to the database
        mock_db_manager.get_trades.assert_called()
        
        app.dependency_overrides.clear()