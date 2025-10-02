"""
Settings and configuration management for the trading platform.
Handles environment variables, API keys, and system configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

class TradingSettings(BaseSettings):
    """Main settings class for the trading platform"""
    
    # Application settings
    app_name: str = Field(default="Automated Trading Platform", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="production", description="Environment (development/production)")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Database settings
    database_url: str = Field(
        default="sqlite:///./trading_platform.db",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_echo: bool = Field(default=False, description="Enable SQL query logging")
    
    # Redis settings (for caching and messaging)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # Exchange API settings
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_secret_key: Optional[str] = Field(default=None, description="Binance secret key")
    binance_testnet: bool = Field(default=True, description="Use Binance testnet")
    
    coinbase_api_key: Optional[str] = Field(default=None, description="Coinbase Pro API key")
    coinbase_secret_key: Optional[str] = Field(default=None, description="Coinbase Pro secret key")
    coinbase_passphrase: Optional[str] = Field(default=None, description="Coinbase Pro passphrase")
    coinbase_sandbox: bool = Field(default=True, description="Use Coinbase sandbox")
    
    kraken_api_key: Optional[str] = Field(default=None, description="Kraken API key")
    kraken_secret_key: Optional[str] = Field(default=None, description="Kraken secret key")
    
    # Trading settings
    default_base_currency: str = Field(default="USDT", description="Default base currency")
    default_quote_currencies: List[str] = Field(
        default=["BTC", "ETH", "ADA", "DOT", "LINK"],
        description="Default quote currencies to trade"
    )
    max_positions: int = Field(default=10, description="Maximum number of open positions")
    default_leverage: float = Field(default=1.0, description="Default leverage")
    
    # Risk management settings
    max_portfolio_risk: float = Field(default=0.02, description="Maximum portfolio risk per trade (2%)")
    max_daily_drawdown: float = Field(default=0.05, description="Maximum daily drawdown (5%)")
    max_total_drawdown: float = Field(default=0.20, description="Maximum total drawdown (20%)")
    stop_loss_percentage: float = Field(default=0.02, description="Default stop loss (2%)")
    take_profit_percentage: float = Field(default=0.06, description="Default take profit (6%)")
    
    # ML Model settings
    model_retrain_interval: int = Field(default=24, description="Model retraining interval (hours)")
    model_data_lookback: int = Field(default=1000, description="Historical data lookback for training")
    min_prediction_confidence: float = Field(default=0.6, description="Minimum prediction confidence")
    
    # Technical analysis settings
    default_timeframes: List[str] = Field(
        default=["1m", "5m", "15m", "1h", "4h", "1d"],
        description="Default timeframes for analysis"
    )
    sma_periods: List[int] = Field(
        default=[20, 50, 100, 200],
        description="Simple Moving Average periods"
    )
    ema_periods: List[int] = Field(
        default=[12, 26, 50, 100],
        description="Exponential Moving Average periods"
    )
    
    # Signal generation settings
    signal_cooldown: int = Field(default=300, description="Signal cooldown period (seconds)")
    max_signals_per_hour: int = Field(default=10, description="Maximum signals per hour")
    signal_confidence_threshold: float = Field(default=0.7, description="Minimum signal confidence")
    
    # Data collection settings
    data_update_interval: int = Field(default=60, description="Data update interval (seconds)")
    max_websocket_retries: int = Field(default=5, description="Maximum WebSocket connection retries")
    rate_limit_requests_per_minute: int = Field(default=1200, description="Rate limit (requests/minute)")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    log_file: Optional[str] = Field(default="trading_platform.log", description="Log file path")
    log_rotation: str = Field(default="1 day", description="Log rotation interval")
    log_retention: str = Field(default="30 days", description="Log retention period")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")
    health_check_interval: int = Field(default=30, description="Health check interval (seconds)")
    
    # Security settings
    jwt_secret_key: str = Field(
        default="your-super-secret-jwt-key-change-in-production",
        description="JWT secret key for authentication"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration (hours)")
    
    # Email notifications (optional)
    smtp_server: Optional[str] = Field(default=None, description="SMTP server for email notifications")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    notification_email: Optional[str] = Field(default=None, description="Email for notifications")
    
    # Webhook settings
    webhook_urls: List[str] = Field(default=[], description="Webhook URLs for notifications")
    webhook_timeout: int = Field(default=10, description="Webhook timeout (seconds)")
    
    # Backtesting settings
    backtest_initial_capital: float = Field(default=10000.0, description="Initial capital for backtesting")
    backtest_commission: float = Field(default=0.001, description="Commission rate for backtesting")
    backtest_slippage: float = Field(default=0.0005, description="Slippage for backtesting")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be 'development', 'staging', or 'production'")
        return v
        
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
        
    @validator("max_portfolio_risk")
    def validate_portfolio_risk(cls, v):
        if not 0 < v <= 0.1:  # 0.1% to 10%
            raise ValueError("Portfolio risk must be between 0.1% and 10%")
        return v
        
    @validator("signal_confidence_threshold")
    def validate_confidence_threshold(cls, v):
        if not 0.5 <= v <= 1.0:
            raise ValueError("Signal confidence threshold must be between 0.5 and 1.0")
        return v
        
    def get_exchange_config(self, exchange: str) -> Dict[str, Any]:
        """Get configuration for a specific exchange"""
        if exchange.lower() == "binance":
            return {
                "api_key": self.binance_api_key,
                "secret_key": self.binance_secret_key,
                "testnet": self.binance_testnet,
                "base_url": "https://testapi.binance.com" if self.binance_testnet else "https://api.binance.com"
            }
        elif exchange.lower() == "coinbase":
            return {
                "api_key": self.coinbase_api_key,
                "secret_key": self.coinbase_secret_key,
                "passphrase": self.coinbase_passphrase,
                "sandbox": self.coinbase_sandbox,
                "base_url": "https://api-public.sandbox.pro.coinbase.com" if self.coinbase_sandbox else "https://api.pro.coinbase.com"
            }
        elif exchange.lower() == "kraken":
            return {
                "api_key": self.kraken_api_key,
                "secret_key": self.kraken_secret_key,
                "base_url": "https://api.kraken.com"
            }
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
            
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "echo": self.database_echo
        }
        
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": self.redis_url,
            "password": self.redis_password
        }
        
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
        
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"

# Global settings instance
_settings: Optional[TradingSettings] = None

def get_settings() -> TradingSettings:
    """Get global settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = TradingSettings()
    return _settings

def reload_settings() -> TradingSettings:
    """Reload settings from environment"""
    global _settings
    _settings = TradingSettings()
    return _settings

# Trading strategy configurations
class StrategySettings:
    """Strategy-specific settings"""
    
    # Moving Average Crossover Strategy
    MA_CROSSOVER = {
        "name": "MA Crossover",
        "short_period": 20,
        "long_period": 50,
        "signal_threshold": 0.001,  # 0.1%
        "weight": 0.3
    }
    
    # RSI Strategy
    RSI_STRATEGY = {
        "name": "RSI Strategy",
        "period": 14,
        "oversold": 30,
        "overbought": 70,
        "weight": 0.25
    }
    
    # MACD Strategy
    MACD_STRATEGY = {
        "name": "MACD Strategy",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "weight": 0.25
    }
    
    # ML Prediction Strategy
    ML_STRATEGY = {
        "name": "ML Prediction",
        "models": ["random_forest", "lstm", "gradient_boosting"],
        "ensemble_method": "weighted_average",
        "confidence_threshold": 0.7,
        "weight": 0.2
    }
    
    @classmethod
    def get_all_strategies(cls) -> Dict[str, Dict]:
        """Get all available strategy configurations"""
        return {
            "ma_crossover": cls.MA_CROSSOVER,
            "rsi": cls.RSI_STRATEGY,
            "macd": cls.MACD_STRATEGY,
            "ml_prediction": cls.ML_STRATEGY
        }

# Risk management configurations
class RiskSettings:
    """Risk management settings"""
    
    # Position sizing methods
    POSITION_SIZING_METHODS = {
        "fixed_percentage": 0.02,  # 2% of portfolio
        "kelly_criterion": True,
        "volatility_adjusted": True,
        "correlation_adjusted": True
    }
    
    # Risk limits
    RISK_LIMITS = {
        "max_position_size": 0.1,  # 10% of portfolio
        "max_sector_exposure": 0.3,  # 30% in one sector
        "max_correlation": 0.7,  # Maximum correlation between positions
        "var_limit": 0.05,  # 5% Value at Risk
        "expected_shortfall_limit": 0.08  # 8% Expected Shortfall
    }
    
    # Stop loss configurations
    STOP_LOSS_CONFIGS = {
        "trailing_stop": True,
        "trailing_percentage": 0.02,  # 2%
        "time_based_stop": True,
        "time_limit_hours": 72,  # 3 days
        "volatility_stop": True,
        "atr_multiplier": 2.0
    }

# Market data configurations
class MarketDataSettings:
    """Market data and exchange settings"""
    
    # Supported exchanges
    EXCHANGES = {
        "binance": {
            "name": "Binance",
            "api_version": "v3",
            "rate_limit": 1200,  # requests per minute
            "supported_assets": ["BTC", "ETH", "ADA", "DOT", "LINK", "BNB"],
            "min_notional": 10.0  # USDT
        },
        "coinbase": {
            "name": "Coinbase Pro",
            "api_version": "v1",
            "rate_limit": 100,
            "supported_assets": ["BTC", "ETH", "ADA", "DOT", "LINK"],
            "min_notional": 5.0  # USD
        },
        "kraken": {
            "name": "Kraken",
            "api_version": "v0",
            "rate_limit": 60,
            "supported_assets": ["BTC", "ETH", "ADA", "DOT", "LINK"],
            "min_notional": 5.0  # USD
        }
    }
    
    # Default trading pairs
    TRADING_PAIRS = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "BNBUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT", "UNIUSDT"
    ]
    
    # Data collection intervals
    COLLECTION_INTERVALS = {
        "real_time": "1s",
        "minute": "1m",
        "five_minute": "5m",
        "fifteen_minute": "15m",
        "hourly": "1h",
        "four_hourly": "4h",
        "daily": "1d"
    }

# Monitoring and alerting configurations
class MonitoringSettings:
    """Monitoring and alerting settings"""
    
    # Alert thresholds
    ALERT_THRESHOLDS = {
        "daily_pnl_loss": -0.05,  # -5%
        "drawdown": -0.10,  # -10%
        "win_rate_drop": 0.4,  # Below 40%
        "api_error_rate": 0.05,  # Above 5%
        "latency_threshold": 1000  # 1 second
    }
    
    # Health check endpoints
    HEALTH_CHECKS = {
        "database": "/health/db",
        "exchanges": "/health/exchanges",
        "ml_models": "/health/models",
        "websockets": "/health/websockets",
        "redis": "/health/redis"
    }
    
    # Metrics to collect
    METRICS = [
        "trades_per_hour",
        "win_rate",
        "profit_loss",
        "drawdown",
        "sharpe_ratio",
        "api_latency",
        "error_rate",
        "active_positions"
    ]