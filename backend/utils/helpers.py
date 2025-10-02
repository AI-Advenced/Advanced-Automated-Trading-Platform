"""
Helper functions and utilities
"""
import asyncio
import functools
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from pathlib import Path

def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    
    # Create file handler
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Async retry decorator with exponential backoff
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff ** attempt)
                        await asyncio.sleep(sleep_time)
                    
            raise last_exception
        return wrapper
    return decorator

def rate_limit(calls_per_second: float):
    """
    Rate limiting decorator
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
                
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_symbol_format(symbol: str) -> bool:
    """Validate trading pair symbol format"""
    if not symbol or '/' not in symbol:
        return False
    
    parts = symbol.split('/')
    if len(parts) != 2:
        return False
        
    base, quote = parts
    return len(base) >= 2 and len(quote) >= 2

def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to standard format"""
    return symbol.upper().replace('-', '/').replace('_', '/')

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    Calculate position size based on risk management
    
    Args:
        account_balance: Total account balance
        risk_percent: Risk percentage (0.02 = 2%)
        entry_price: Entry price for the trade
        stop_loss_price: Stop loss price
        
    Returns:
        Position size in base currency
    """
    if stop_loss_price <= 0 or entry_price <= 0:
        return 0.0
        
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0.0
        
    risk_amount = account_balance * risk_percent
    position_size = risk_amount / risk_per_unit
    
    return position_size

def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float
) -> float:
    """Calculate risk/reward ratio"""
    if not all([entry_price > 0, stop_loss_price > 0, take_profit_price > 0]):
        return 0.0
        
    risk = abs(entry_price - stop_loss_price)
    reward = abs(take_profit_price - entry_price)
    
    if risk == 0:
        return 0.0
        
    return reward / risk

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount for display"""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency in ["BTC", "ETH"]:
        return f"{amount:.6f} {currency}"
    else:
        return f"{amount:.4f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage for display"""
    return f"{value:.{decimals}f}%"

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0
        
    excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
    return excess_returns / returns.std() * np.sqrt(252)  # Annualized

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve
    
    Args:
        equity_curve: Series of equity values over time
        
    Returns:
        Maximum drawdown as negative percentage
    """
    if equity_curve.empty:
        return 0.0
        
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Series of returns
        confidence: Confidence level (0.05 = 95% VaR)
        
    Returns:
        VaR value
    """
    if returns.empty:
        return 0.0
        
    return np.percentile(returns, confidence * 100)

def validate_api_credentials(api_key: str, secret: str, passphrase: str = None) -> bool:
    """Validate API credentials format"""
    if not api_key or not secret:
        return False
        
    # Basic validation - check if they look like valid keys
    if len(api_key) < 10 or len(secret) < 10:
        return False
        
    return True

def create_signature(
    timestamp: str,
    method: str,
    request_path: str,
    body: str,
    secret: str
) -> str:
    """Create HMAC signature for API requests"""
    message = timestamp + method + request_path + body
    signature = hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def round_to_precision(value: float, precision: int) -> float:
    """Round value to specified precision"""
    if precision <= 0:
        return round(value)
    return round(value, precision)

def format_timeframe(minutes: int) -> str:
    """Convert minutes to human readable format"""
    if minutes < 60:
        return f"{minutes}m"
    elif minutes < 1440:  # Less than 1 day
        hours = minutes // 60
        return f"{hours}h"
    else:
        days = minutes // 1440
        return f"{days}d"

def parse_timeframe(timeframe: str) -> int:
    """Parse timeframe string to minutes"""
    timeframe = timeframe.lower()
    
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 1440
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 10080  # 7 * 24 * 60
    else:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

def get_market_session(timestamp: datetime = None) -> str:
    """Get current market session"""
    if timestamp is None:
        timestamp = datetime.utcnow()
        
    hour = timestamp.hour
    
    # Market sessions (UTC)
    if 0 <= hour < 8:
        return "ASIA"
    elif 8 <= hour < 16:
        return "EUROPE" 
    else:
        return "AMERICA"

def is_market_open(symbol: str, timestamp: datetime = None) -> bool:
    """Check if market is open for given symbol"""
    if timestamp is None:
        timestamp = datetime.utcnow()
        
    # Crypto markets are always open
    if '/' in symbol and any(crypto in symbol for crypto in ['BTC', 'ETH', 'USDT', 'USDC']):
        return True
        
    # For forex, check weekdays
    weekday = timestamp.weekday()
    return weekday < 5  # Monday = 0, Friday = 4

def calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """Calculate correlation between two price series"""
    try:
        return series1.corr(series2)
    except:
        return 0.0

def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 2.0) -> pd.Series:
    """
    Detect outliers in data
    
    Args:
        data: Data series
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
        
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def clean_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """Clean and prepare data for analysis"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove outliers if requested
    if remove_outliers:
        for column in ['open', 'high', 'low', 'close', 'volume']:
            if column in df.columns:
                outliers = detect_outliers(df[column])
                df = df[~outliers]
    
    return df

def create_config_backup(config: Dict, backup_dir: str = "backups") -> str:
    """Create backup of configuration"""
    Path(backup_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_backup_{timestamp}.json"
    filepath = Path(backup_dir) / filename
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)
        
    return str(filepath)

def load_config_backup(filepath: str) -> Dict:
    """Load configuration from backup"""
    with open(filepath, 'r') as f:
        return json.load(f)

class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.description} took {elapsed:.2f} seconds")

class DataBuffer:
    """Thread-safe data buffer for real-time data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        self._lock = asyncio.Lock()
        
    async def add(self, item: Any):
        async with self._lock:
            self.buffer.append(item)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)
                
    async def get_all(self) -> List[Any]:
        async with self._lock:
            return self.buffer.copy()
            
    async def get_latest(self, n: int = 1) -> List[Any]:
        async with self._lock:
            return self.buffer[-n:] if self.buffer else []
            
    async def clear(self):
        async with self._lock:
            self.buffer.clear()

def ensure_directory_exists(path: Union[str, Path]):
    """Ensure directory exists, create if necessary"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_age_hours(filepath: Union[str, Path]) -> float:
    """Get file age in hours"""
    path = Path(filepath)
    if not path.exists():
        return float('inf')
        
    modified_time = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - modified_time
    return age.total_seconds() / 3600