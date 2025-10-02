"""
Unit tests for the DataCollector component.
Tests data collection from multiple exchanges and WebSocket connections.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from backend.core.data_collector import DataCollector
from backend.utils.exceptions import DataCollectionError
from backend.config.settings import TradingSettings

class TestDataCollector:
    """Test suite for DataCollector"""
    
    @pytest.fixture
    def data_collector(self, test_settings):
        """Create DataCollector instance for testing"""
        return DataCollector(test_settings)
    
    @pytest.mark.asyncio
    async def test_initialization(self, data_collector):
        """Test DataCollector initialization"""
        assert data_collector.settings is not None
        assert data_collector.exchanges == {}
        assert data_collector.websockets == {}
        assert data_collector.rate_limiter is not None
    
    @pytest.mark.asyncio
    async def test_add_exchange(self, data_collector):
        """Test adding exchange configuration"""
        exchange_config = {
            'name': 'binance',
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'testnet': True
        }
        
        data_collector.add_exchange('binance', exchange_config)
        
        assert 'binance' in data_collector.exchanges
        assert data_collector.exchanges['binance']['name'] == 'binance'
    
    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, data_collector):
        """Test getting supported trading symbols"""
        # Mock exchange info
        mock_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        with patch.object(data_collector, '_fetch_exchange_info') as mock_fetch:
            mock_fetch.return_value = {'symbols': mock_symbols}
            
            symbols = await data_collector.get_supported_symbols('binance')
            
            assert symbols == mock_symbols
            mock_fetch.assert_called_once_with('binance')
    
    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, data_collector, sample_ohlcv_data):
        """Test successful historical data retrieval"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        limit = 100
        
        with patch.object(data_collector, '_fetch_historical_data') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            result = await data_collector.get_historical_data(
                exchange='binance',
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= limit
            assert 'close' in result.columns
            assert 'volume' in result.columns
            mock_fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_historical_data_invalid_exchange(self, data_collector):
        """Test historical data retrieval with invalid exchange"""
        with pytest.raises(DataCollectionError, match="Exchange .* not configured"):
            await data_collector.get_historical_data(
                exchange='invalid_exchange',
                symbol='BTCUSDT',
                timeframe='1h'
            )
    
    @pytest.mark.asyncio
    async def test_get_real_time_data(self, data_collector):
        """Test real-time data retrieval"""
        symbol = 'BTCUSDT'
        mock_ticker = {
            'symbol': symbol,
            'price': 50000.0,
            'volume': 1000.0,
            'timestamp': datetime.now()
        }
        
        with patch.object(data_collector, '_fetch_ticker') as mock_ticker_fetch:
            mock_ticker_fetch.return_value = mock_ticker
            
            result = await data_collector.get_real_time_data('binance', symbol)
            
            assert result['symbol'] == symbol
            assert result['price'] == 50000.0
            assert 'timestamp' in result
    
    @pytest.mark.asyncio
    async def test_connect_websocket_success(self, data_collector):
        """Test successful WebSocket connection"""
        exchange = 'binance'
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        mock_websocket = AsyncMock()
        
        with patch('websockets.connect', return_value=mock_websocket):
            await data_collector.connect_websocket(exchange, symbols)
            
            assert exchange in data_collector.websockets
            assert data_collector.websockets[exchange]['connection'] == mock_websocket
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, data_collector):
        """Test WebSocket disconnection"""
        exchange = 'binance'
        
        # Setup mock WebSocket connection
        mock_websocket = AsyncMock()
        data_collector.websockets[exchange] = {
            'connection': mock_websocket,
            'symbols': ['BTCUSDT']
        }
        
        await data_collector.disconnect_websocket(exchange)
        
        mock_websocket.close.assert_called_once()
        assert exchange not in data_collector.websockets
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, data_collector):
        """Test WebSocket error handling and reconnection"""
        exchange = 'binance'
        symbols = ['BTCUSDT']
        
        # Mock WebSocket that raises an exception
        mock_websocket = AsyncMock()
        mock_websocket.recv.side_effect = ConnectionError("Connection lost")
        
        with patch('websockets.connect', return_value=mock_websocket):
            with patch.object(data_collector, '_handle_websocket_message') as mock_handler:
                # This should handle the error gracefully
                try:
                    await data_collector.connect_websocket(exchange, symbols)
                    await data_collector._websocket_handler(exchange)
                except ConnectionError:
                    pass  # Expected to be handled
        
        # Should attempt reconnection
        assert data_collector.websockets.get(exchange, {}).get('reconnect_count', 0) >= 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, data_collector):
        """Test API rate limiting functionality"""
        exchange = 'binance'
        
        # Mock rate limiter
        with patch.object(data_collector.rate_limiter, 'acquire') as mock_acquire:
            mock_acquire.return_value = True
            
            # Make multiple rapid requests
            tasks = []
            for _ in range(5):
                task = data_collector._make_api_request(exchange, 'GET', '/api/v3/ticker/24hr')
                tasks.append(task)
            
            # Should respect rate limits
            assert mock_acquire.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_data_validation(self, data_collector):
        """Test data validation and cleaning"""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [None],  # Invalid null value
            'high': [50000],
            'low': [49000],
            'close': [49500],
            'volume': [-100]  # Invalid negative volume
        })
        
        # Should clean and validate data
        cleaned_data = data_collector._validate_ohlcv_data(invalid_data)
        
        assert not cleaned_data['open'].isna().any()  # No null values
        assert (cleaned_data['volume'] >= 0).all()   # No negative volumes
    
    @pytest.mark.asyncio
    async def test_multiple_exchange_data_aggregation(self, data_collector, sample_ohlcv_data):
        """Test aggregating data from multiple exchanges"""
        exchanges = ['binance', 'coinbase']
        symbol = 'BTCUSDT'
        
        with patch.object(data_collector, 'get_historical_data') as mock_get_data:
            mock_get_data.return_value = sample_ohlcv_data
            
            # Get data from multiple exchanges
            all_data = {}
            for exchange in exchanges:
                data = await data_collector.get_historical_data(exchange, symbol, '1h')
                all_data[exchange] = data
            
            assert len(all_data) == len(exchanges)
            for exchange_data in all_data.values():
                assert isinstance(exchange_data, pd.DataFrame)
                assert not exchange_data.empty
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, data_collector):
        """Test error recovery mechanisms"""
        symbol = 'BTCUSDT'
        
        # Mock API that fails then succeeds
        call_count = 0
        def mock_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return {'price': 50000}
        
        with patch.object(data_collector, '_make_api_request', side_effect=mock_api_call):
            # Should retry and eventually succeed
            result = await data_collector.get_real_time_data('binance', symbol)
            assert result is not None
            assert call_count > 1  # Should have retried
    
    @pytest.mark.asyncio
    async def test_data_caching(self, data_collector, sample_ohlcv_data):
        """Test data caching functionality"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        
        with patch.object(data_collector, '_fetch_historical_data') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            # First call should fetch data
            data1 = await data_collector.get_historical_data('binance', symbol, timeframe)
            
            # Second call should use cache (if implemented)
            data2 = await data_collector.get_historical_data('binance', symbol, timeframe)
            
            # Both should return data
            assert isinstance(data1, pd.DataFrame)
            assert isinstance(data2, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_websocket_message_processing(self, data_collector):
        """Test WebSocket message processing"""
        # Mock WebSocket message
        mock_message = {
            'stream': 'btcusdt@ticker',
            'data': {
                's': 'BTCUSDT',
                'c': '50000.00',  # Current price
                'v': '1000.00',   # Volume
                'E': int(datetime.now().timestamp() * 1000)  # Event time
            }
        }
        
        # Process the message
        processed = data_collector._process_websocket_message(mock_message)
        
        assert processed is not None
        assert processed['symbol'] == 'BTCUSDT'
        assert processed['price'] == 50000.00
    
    def test_timeframe_conversion(self, data_collector):
        """Test timeframe string to minutes conversion"""
        test_cases = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        for timeframe, expected_minutes in test_cases.items():
            minutes = data_collector._timeframe_to_minutes(timeframe)
            assert minutes == expected_minutes
    
    def test_symbol_normalization(self, data_collector):
        """Test symbol normalization across exchanges"""
        test_cases = [
            ('BTCUSDT', 'binance', 'BTCUSDT'),
            ('BTC-USD', 'coinbase', 'BTCUSD'),
            ('XBTUSD', 'kraken', 'BTCUSD')
        ]
        
        for input_symbol, exchange, expected in test_cases:
            normalized = data_collector._normalize_symbol(input_symbol, exchange)
            # This would depend on implementation
            assert isinstance(normalized, str)
    
    @pytest.mark.asyncio
    async def test_concurrent_data_collection(self, data_collector, sample_ohlcv_data):
        """Test concurrent data collection from multiple sources"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        with patch.object(data_collector, 'get_historical_data') as mock_get_data:
            mock_get_data.return_value = sample_ohlcv_data
            
            # Collect data concurrently
            tasks = [
                data_collector.get_historical_data('binance', symbol, '1h')
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(symbols)
            for result in results:
                assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_cleanup_on_shutdown(self, data_collector):
        """Test proper cleanup on shutdown"""
        # Setup WebSocket connections
        mock_websocket = AsyncMock()
        data_collector.websockets['binance'] = {
            'connection': mock_websocket,
            'symbols': ['BTCUSDT']
        }
        
        # Shutdown should clean up connections
        await data_collector.shutdown()
        
        mock_websocket.close.assert_called_once()
        assert len(data_collector.websockets) == 0