"""
Comprehensive Trading Platform Configuration
"""
import os
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging

class TradingConfig:
    """
    Centralized configuration management for the trading platform
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or environment variables"""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from file if provided
        if config_path and Path(config_path).exists():
            self._load_from_file(config_path)
        else:
            self._load_default_config()
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge file config with default config
            for section, values in file_config.items():
                if hasattr(self, section):
                    getattr(self, section).update(values)
                    
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration"""
        
        # API Configuration
        self.api_config = {
            # Binance API
            'binance_api_key': os.getenv('BINANCE_API_KEY', ''),
            'binance_secret': os.getenv('BINANCE_SECRET', ''),
            'binance_testnet': os.getenv('BINANCE_TESTNET', 'true').lower() == 'true',
            
            # Coinbase Pro API
            'coinbase_api_key': os.getenv('COINBASE_API_KEY', ''),
            'coinbase_secret': os.getenv('COINBASE_SECRET', ''),
            'coinbase_passphrase': os.getenv('COINBASE_PASSPHRASE', ''),
            
            # Alpha Vantage for Forex/Stocks
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', ''),
            
            # General API settings
            'request_timeout': 30,
            'retry_attempts': 3,
            'rate_limit_delay': 1.0,
            'sandbox': os.getenv('USE_SANDBOX', 'true').lower() == 'true'
        }
        
        # Trading Configuration
        self.trading_config = {
            'initial_balance': float(os.getenv('INITIAL_BALANCE', '10000')),
            'min_signal_confidence': float(os.getenv('MIN_SIGNAL_CONFIDENCE', '0.65')),
            'cycle_interval_minutes': int(os.getenv('CYCLE_INTERVAL_MINUTES', '5')),
            'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', '8')),
            'max_holding_days': int(os.getenv('MAX_HOLDING_DAYS', '7')),
            
            # Watchlist - Major cryptocurrencies and popular meme coins
            'watchlist': [
                # Major cryptocurrencies
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT', 'UNI/USDT',
                
                # Popular meme/trending coins
                'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT',
                'WIF/USDT', 'MEME/USDT', 'BABYDOGE/USDT'
            ],
            
            # High-potential altcoins
            'high_potential_watchlist': [
                'ATOM/USDT', 'ICP/USDT', 'NEAR/USDT', 'FTM/USDT', 'ALGO/USDT',
                'VET/USDT', 'THETA/USDT', 'HBAR/USDT', 'EOS/USDT', 'TRON/USDT'
            ],
            
            # New/trending coins (higher risk)
            'trending_watchlist': [
                'ARB/USDT', 'OP/USDT', 'APT/USDT', 'SUI/USDT', 'SEI/USDT'
            ]
        }
        
        # Risk Management Configuration
        self.risk_config = {
            'max_risk_per_trade': float(os.getenv('MAX_RISK_PER_TRADE', '0.02')),  # 2%
            'max_portfolio_risk': float(os.getenv('MAX_PORTFOLIO_RISK', '0.10')),  # 10%
            'max_correlation_exposure': float(os.getenv('MAX_CORRELATION_EXPOSURE', '0.60')),  # 60%
            'max_sector_exposure': float(os.getenv('MAX_SECTOR_EXPOSURE', '0.40')),  # 40%
            'max_single_position': float(os.getenv('MAX_SINGLE_POSITION', '0.15')),  # 15%
            'stop_loss_buffer': float(os.getenv('STOP_LOSS_BUFFER', '1.1')),  # 10% buffer
            'emergency_stop_loss': float(os.getenv('EMERGENCY_STOP_LOSS', '0.15')),  # 15% emergency exit
            'profit_target_multiplier': float(os.getenv('PROFIT_TARGET_MULTIPLIER', '2.5')),  # 2.5x risk/reward
        }
        
        # Signal Generation Configuration
        self.signal_config = {
            # Technical Analysis Weights
            'indicator_weights': {
                'rsi': 1.5,
                'macd': 1.3,
                'bollinger': 1.4,
                'moving_average': 1.0,
                'stochastic': 0.9,
                'williams_r': 0.8,
                'volume': 1.2,
                'support_resistance': 1.6,
                'momentum': 1.1,
                'volatility': 0.7
            },
            
            # Machine Learning Weights (higher priority)
            'ml_weights': {
                'ml_direction': 2.5,
                'ml_price': 2.2,
                'ml_multiclass': 2.0
            },
            
            # Signal thresholds
            'min_signal_strength': 1.5,
            'min_confidence': 0.60,
            'min_risk_reward_ratio': 1.5,
            'max_signal_age_minutes': 30,
            
            # Technical indicator parameters
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_squeeze_threshold': 0.8,
            'volume_spike_threshold': 2.0,
            'momentum_threshold': 0.05
        }
        
        # Machine Learning Configuration
        self.ml_config = {
            'model_retrain_frequency_days': int(os.getenv('MODEL_RETRAIN_DAYS', '7')),
            'min_training_samples': int(os.getenv('MIN_TRAINING_SAMPLES', '1000')),
            'feature_selection_threshold': float(os.getenv('FEATURE_SELECTION_THRESHOLD', '0.01')),
            'cross_validation_folds': int(os.getenv('CV_FOLDS', '5')),
            
            # Model parameters
            'lstm_sequence_length': 60,
            'lstm_forecast_horizon': 5,
            'random_forest_estimators': 200,
            'gradient_boosting_estimators': 100,
            
            # Training parameters
            'test_size': 0.2,
            'validation_size': 0.1,
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 100
        }
        
        # Data Collection Configuration
        self.data_config = {
            'default_timeframe': '1h',
            'max_historical_periods': 2000,
            'real_time_update_interval': 60,  # seconds
            'data_cache_ttl': 300,  # 5 minutes
            'websocket_reconnect_delay': 5,
            'price_precision': 8,
            'volume_precision': 4,
            
            # Data sources priority
            'exchange_priority': ['binance', 'coinbase', 'kraken'],
            'backup_data_sources': ['yfinance', 'alpha_vantage'],
            
            # Market data validation
            'price_change_threshold': 0.20,  # 20% sudden price change alert
            'volume_spike_alert': 5.0,  # 5x normal volume alert
            'data_quality_min_points': 50
        }
        
        # Performance Monitoring Configuration
        self.monitoring_config = {
            'enable_learning': True,
            'retrain_threshold': 50,  # trades before considering retrain
            'performance_window_days': 30,
            'alert_win_rate_threshold': 0.40,  # Alert if win rate drops below 40%
            'alert_drawdown_threshold': 0.10,  # Alert if drawdown exceeds 10%
            'alert_consecutive_losses': 5,
            
            # Database settings
            'database_path': 'data/trading_data.db',
            'backup_frequency_hours': 6,
            'cleanup_old_data_days': 90,
            
            # Reporting
            'daily_report_enabled': True,
            'weekly_report_enabled': True,
            'email_alerts_enabled': False,
            'telegram_alerts_enabled': False
        }
        
        # Logging Configuration
        self.logging_config = {
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': 'logs/trading_platform.log',
            'max_log_size_mb': 100,
            'backup_count': 5,
            'console_output': True,
            
            # Log rotation
            'rotate_logs': True,
            'rotation_interval': 'daily',
            
            # Performance logging
            'log_trades': True,
            'log_signals': True,
            'log_market_data': False,
            'log_performance_metrics': True
        }
        
        # Notification Configuration
        self.notification_config = {
            # Telegram notifications
            'telegram_enabled': False,
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            
            # Email notifications
            'email_enabled': False,
            'smtp_server': os.getenv('SMTP_SERVER', ''),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email_user': os.getenv('EMAIL_USER', ''),
            'email_password': os.getenv('EMAIL_PASSWORD', ''),
            'email_recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
            
            # Webhook notifications
            'webhook_enabled': False,
            'webhook_url': os.getenv('WEBHOOK_URL', ''),
            'webhook_secret': os.getenv('WEBHOOK_SECRET', ''),
            
            # Notification triggers
            'notify_on_trade_open': True,
            'notify_on_trade_close': True,
            'notify_on_profit_target': True,
            'notify_on_stop_loss': True,
            'notify_on_daily_summary': True,
            'notify_on_errors': True
        }
        
        # Development/Testing Configuration
        self.dev_config = {
            'simulation_mode': os.getenv('SIMULATION_MODE', 'false').lower() == 'true',
            'paper_trading': os.getenv('PAPER_TRADING', 'true').lower() == 'true',
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'mock_market_data': False,
            'accelerated_testing': False,
            'test_data_path': 'test_data/',
            'save_test_results': True
        }
    
    def _load_from_environment(self):
        """Override configuration with environment variables"""
        
        # Override specific values from environment
        env_overrides = {
            'INITIAL_BALANCE': ('trading_config', 'initial_balance', float),
            'MAX_RISK_PER_TRADE': ('risk_config', 'max_risk_per_trade', float),
            'MIN_SIGNAL_CONFIDENCE': ('trading_config', 'min_signal_confidence', float),
            'MAX_OPEN_POSITIONS': ('trading_config', 'max_open_positions', int),
            'CYCLE_INTERVAL_MINUTES': ('trading_config', 'cycle_interval_minutes', int),
            'USE_SANDBOX': ('api_config', 'sandbox', lambda x: x.lower() == 'true'),
        }
        
        for env_var, (section, key, converter) in env_overrides.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    getattr(self, section)[key] = converted_value
                    self.logger.info(f"Override {section}.{key} = {converted_value} from env {env_var}")
                except ValueError as e:
                    self.logger.warning(f"Invalid value for {env_var}: {value}, error: {e}")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate risk parameters
        if self.risk_config['max_risk_per_trade'] <= 0 or self.risk_config['max_risk_per_trade'] > 0.1:
            errors.append("max_risk_per_trade must be between 0 and 0.1 (10%)")
        
        if self.risk_config['max_portfolio_risk'] <= 0 or self.risk_config['max_portfolio_risk'] > 0.5:
            errors.append("max_portfolio_risk must be between 0 and 0.5 (50%)")
        
        if self.trading_config['initial_balance'] <= 0:
            errors.append("initial_balance must be positive")
        
        if self.trading_config['min_signal_confidence'] < 0 or self.trading_config['min_signal_confidence'] > 1:
            errors.append("min_signal_confidence must be between 0 and 1")
        
        # Validate watchlist
        if not self.trading_config['watchlist']:
            errors.append("watchlist cannot be empty")
        
        # Validate API configuration (warn if missing)
        if not self.api_config.get('binance_api_key') and not self.dev_config['simulation_mode']:
            self.logger.warning("Binance API key not configured - using sandbox mode")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("✅ Configuration validation passed")
    
    def get_watchlist_by_category(self, category: str = 'default') -> List[str]:
        """Get watchlist by category"""
        if category == 'high_potential':
            return self.trading_config.get('high_potential_watchlist', [])
        elif category == 'trending':
            return self.trading_config.get('trending_watchlist', [])
        elif category == 'all':
            return (self.trading_config['watchlist'] + 
                   self.trading_config.get('high_potential_watchlist', []) +
                   self.trading_config.get('trending_watchlist', []))
        else:
            return self.trading_config['watchlist']
    
    def update_watchlist(self, new_symbols: List[str], category: str = 'default'):
        """Update watchlist with new symbols"""
        if category == 'high_potential':
            self.trading_config['high_potential_watchlist'] = new_symbols
        elif category == 'trending':
            self.trading_config['trending_watchlist'] = new_symbols
        else:
            self.trading_config['watchlist'] = new_symbols
        
        self.logger.info(f"Updated {category} watchlist with {len(new_symbols)} symbols")
    
    def get_indicator_weight(self, indicator: str) -> float:
        """Get weight for a specific indicator"""
        return (self.signal_config['indicator_weights'].get(indicator) or 
                self.signal_config['ml_weights'].get(indicator, 1.0))
    
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode"""
        return (self.dev_config['paper_trading'] or 
                self.dev_config['simulation_mode'] or
                self.api_config['sandbox'])
    
    def save_config(self, file_path: str):
        """Save current configuration to file"""
        config_data = {
            'api_config': self.api_config,
            'trading_config': self.trading_config,
            'risk_config': self.risk_config,
            'signal_config': self.signal_config,
            'ml_config': self.ml_config,
            'data_config': self.data_config,
            'monitoring_config': self.monitoring_config,
            'notification_config': self.notification_config,
            'dev_config': self.dev_config
        }
        
        # Remove sensitive data
        safe_config = self._remove_sensitive_data(config_data)
        
        with open(file_path, 'w') as f:
            json.dump(safe_config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {file_path}")
    
    def _remove_sensitive_data(self, config_data: Dict) -> Dict:
        """Remove sensitive information from config before saving"""
        safe_config = config_data.copy()
        
        # Remove API keys and secrets
        sensitive_keys = [
            'binance_api_key', 'binance_secret', 'coinbase_api_key', 
            'coinbase_secret', 'coinbase_passphrase', 'alpha_vantage_key',
            'telegram_bot_token', 'email_password', 'webhook_secret'
        ]
        
        for section in safe_config.values():
            if isinstance(section, dict):
                for key in sensitive_keys:
                    if key in section:
                        section[key] = '***REDACTED***'
        
        return safe_config
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""Trading Platform Configuration:
- Initial Balance: ${self.trading_config['initial_balance']:,.2f}
- Max Risk per Trade: {self.risk_config['max_risk_per_trade']:.1%}
- Watchlist: {len(self.trading_config['watchlist'])} symbols
- Paper Trading: {self.is_paper_trading()}
- Cycle Interval: {self.trading_config['cycle_interval_minutes']} minutes
"""

# Default configuration instance
DEFAULT_CONFIG = TradingConfig()