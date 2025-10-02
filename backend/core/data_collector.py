"""
Advanced Data Collector for Multiple Asset Classes
Supports crypto, forex, stocks, and commodities
"""
import asyncio
import aiohttp
import pandas as pd
import ccxt
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json
import yfinance as yf

class DataCollector:
    """
    Comprehensive data collector supporting multiple exchanges and asset types
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = self._initialize_exchanges()
        self.session = None
        self.logger = logging.getLogger(__name__)
        
    def _initialize_exchanges(self) -> Dict:
        """Initialize trading exchanges"""
        exchanges = {}
        
        # Crypto exchanges
        if 'binance_api_key' in self.config:
            exchanges['binance'] = ccxt.binance({
                'apiKey': self.config.get('binance_api_key', ''),
                'secret': self.config.get('binance_secret', ''),
                'sandbox': self.config.get('sandbox', True),
                'enableRateLimit': True,
            })
        
        if 'coinbase_api_key' in self.config:
            exchanges['coinbase'] = ccxt.coinbasepro({
                'apiKey': self.config.get('coinbase_api_key', ''),
                'secret': self.config.get('coinbase_secret', ''),
                'passphrase': self.config.get('coinbase_passphrase', ''),
                'enableRateLimit': True,
            })
            
        return exchanges
    
    async def get_real_time_data(self, symbol: str, exchange: str = 'binance') -> Optional[Dict]:
        """
        Fetch real-time market data
        """
        try:
            if exchange not in self.exchanges:
                self.logger.error(f"Exchange {exchange} not configured")
                return None
                
            exchange_obj = self.exchanges[exchange]
            ticker = await self._safe_api_call(exchange_obj.fetch_ticker, symbol)
            
            if ticker:
                return {
                    'symbol': symbol,
                    'price': ticker.get('last', 0),
                    'volume': ticker.get('baseVolume', 0),
                    'change_24h': ticker.get('change', 0),
                    'percentage_change': ticker.get('percentage', 0),
                    'high_24h': ticker.get('high', 0),
                    'low_24h': ticker.get('low', 0),
                    'timestamp': datetime.now(),
                    'source': exchange
                }
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                                 limit: int = 1000, exchange: str = 'binance') -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        """
        try:
            if exchange not in self.exchanges:
                self.logger.error(f"Exchange {exchange} not configured")
                return pd.DataFrame()
                
            exchange_obj = self.exchanges[exchange]
            ohlcv = await self._safe_api_call(
                exchange_obj.fetch_ohlcv, symbol, timeframe, limit=limit
            )
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            
        return pd.DataFrame()
    
    async def get_crypto_market_overview(self) -> List[Dict]:
        """
        Get overview of top cryptocurrencies including meme coins
        """
        crypto_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT'
        ]
        
        tasks = []
        for symbol in crypto_symbols:
            task = self.get_real_time_data(symbol, 'binance')
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]
    
    async def get_forex_data(self, pair: str) -> Optional[Dict]:
        """
        Fetch forex data from Alpha Vantage API
        """
        if 'alpha_vantage_key' not in self.config:
            self.logger.error("Alpha Vantage API key not configured")
            return None
            
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': pair.split('/')[0],
            'to_symbol': pair.split('/')[1],
            'interval': '5min',
            'apikey': self.config['alpha_vantage_key']
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'Time Series (5min)' in data:
                            latest_time = list(data['Time Series (5min)'].keys())[0]
                            latest_data = data['Time Series (5min)'][latest_time]
                            
                            return {
                                'symbol': pair,
                                'price': float(latest_data['4. close']),
                                'volume': float(latest_data['5. volume']),
                                'high': float(latest_data['2. high']),
                                'low': float(latest_data['3. low']),
                                'timestamp': datetime.now(),
                                'source': 'alphavantage'
                            }
        except Exception as e:
            self.logger.error(f"Error fetching forex data for {pair}: {e}")
            
        return None
    
    async def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch stock data using yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                
                return {
                    'symbol': symbol,
                    'price': latest['Close'],
                    'volume': latest['Volume'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'open': latest['Open'],
                    'change_24h': info.get('regularMarketChange', 0),
                    'percentage_change': info.get('regularMarketChangePercent', 0),
                    'timestamp': datetime.now(),
                    'source': 'yahoo_finance'
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {e}")
            
        return None
    
    async def get_market_sentiment_data(self) -> Dict:
        """
        Fetch market sentiment indicators
        """
        sentiment_data = {
            'fear_greed_index': await self._get_fear_greed_index(),
            'bitcoin_dominance': await self._get_bitcoin_dominance(),
            'total_market_cap': await self._get_total_market_cap(),
        }
        
        return sentiment_data
    
    async def _get_fear_greed_index(self) -> Optional[Dict]:
        """Fetch Fear & Greed Index from API"""
        try:
            url = "https://api.alternative.me/fng/"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['data']:
                            return {
                                'value': int(data['data'][0]['value']),
                                'classification': data['data'][0]['value_classification'],
                                'timestamp': datetime.now()
                            }
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {e}")
        return None
    
    async def _get_bitcoin_dominance(self) -> Optional[float]:
        """Get Bitcoin market dominance"""
        try:
            # This would typically come from CoinGecko or similar API
            # For demo purposes, returning a placeholder
            return 42.5  # Placeholder value
        except Exception as e:
            self.logger.error(f"Error fetching Bitcoin dominance: {e}")
        return None
    
    async def _get_total_market_cap(self) -> Optional[float]:
        """Get total cryptocurrency market cap"""
        try:
            # This would typically come from CoinGecko API
            # For demo purposes, returning a placeholder
            return 2500000000000.0  # Placeholder value
        except Exception as e:
            self.logger.error(f"Error fetching total market cap: {e}")
        return None
    
    async def start_websocket_stream(self, symbols: List[str], callback):
        """
        Start real-time data stream via WebSocket
        """
        if not symbols:
            return
            
        # Binance WebSocket stream
        stream_url = "wss://stream.binance.com:9443/ws/stream"
        
        # Format symbols for Binance stream
        streams = [f"{symbol.lower().replace('/', '')}@ticker" for symbol in symbols]
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        try:
            import websockets
            
            async with websockets.connect(stream_url) as websocket:
                await websocket.send(json.dumps(subscribe_message))
                self.logger.info(f"Subscribed to WebSocket streams for {len(symbols)} symbols")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'data' in data:
                            processed_data = self._process_websocket_data(data['data'])
                            if processed_data:
                                await callback(processed_data)
                    except Exception as e:
                        self.logger.error(f"Error processing WebSocket message: {e}")
                        
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
    
    def _process_websocket_data(self, raw_data: Dict) -> Optional[Dict]:
        """
        Process raw WebSocket data into standardized format
        """
        try:
            return {
                'symbol': raw_data.get('s', ''),
                'price': float(raw_data.get('c', 0)),
                'volume': float(raw_data.get('v', 0)),
                'change_24h': float(raw_data.get('P', 0)),
                'high_24h': float(raw_data.get('h', 0)),
                'low_24h': float(raw_data.get('l', 0)),
                'timestamp': datetime.now(),
                'source': 'binance_ws'
            }
        except Exception as e:
            self.logger.error(f"Error processing WebSocket data: {e}")
            return None
    
    async def _safe_api_call(self, func, *args, **kwargs):
        """
        Safely execute API calls with error handling and rate limiting
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add small delay to respect rate limits
                if attempt > 0:
                    await asyncio.sleep(2 ** attempt)
                
                result = func(*args, **kwargs)
                
                # Handle both sync and async functions
                if asyncio.iscoroutine(result):
                    return await result
                else:
                    return result
                    
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
        
        return None
    
    async def get_multiple_timeframes(self, symbol: str, timeframes: List[str], 
                                    exchange: str = 'binance') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes simultaneously
        """
        tasks = []
        for tf in timeframes:
            task = self.get_historical_data(symbol, tf, exchange=exchange)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            tf: result for tf, result in zip(timeframes, results)
            if isinstance(result, pd.DataFrame) and not result.empty
        }
    
    def close(self):
        """Close all connections and cleanup resources"""
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                exchange.close()
        
        self.logger.info("Data collector connections closed")