"""
Advanced logging system for the trading platform
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from pythonjsonlogger import jsonlogger

class TradingFormatter(logging.Formatter):
    """Custom formatter for trading platform logs"""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record: logging.LogRecord) -> str:
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Add trading-specific context
        if hasattr(record, 'symbol'):
            record.symbol = getattr(record, 'symbol', 'N/A')
        if hasattr(record, 'trade_id'):
            record.trade_id = getattr(record, 'trade_id', 'N/A')
        if hasattr(record, 'signal_confidence'):
            record.signal_confidence = getattr(record, 'signal_confidence', 'N/A')
            
        # Format message
        formatted = (
            f"[{record.timestamp}] "
            f"{record.levelname:8} | "
            f"{record.name:20} | "
            f"{record.getMessage()}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
            
        return formatted

class StructuredLogger:
    """Structured logging with JSON output for monitoring systems"""
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        self.name = name
        self.log_file = log_file
        self.level = level
        self._setup_logger()
        
    def _setup_logger(self):
        """Setup structured logger with multiple handlers"""
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Create logger
        self.logger = structlog.get_logger(self.name)
        
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(message, **kwargs)
        
    def trade_opened(self, symbol: str, side: str, quantity: float, price: float, **kwargs):
        """Log trade opening"""
        self.logger.info(
            "Trade opened",
            event_type="trade_opened",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            **kwargs
        )
        
    def trade_closed(self, symbol: str, pnl: float, pnl_percent: float, reason: str, **kwargs):
        """Log trade closing"""
        self.logger.info(
            "Trade closed",
            event_type="trade_closed",
            symbol=symbol,
            pnl=pnl,
            pnl_percent=pnl_percent,
            reason=reason,
            **kwargs
        )
        
    def signal_generated(self, symbol: str, signal_type: str, confidence: float, **kwargs):
        """Log signal generation"""
        self.logger.info(
            "Signal generated",
            event_type="signal_generated",
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            **kwargs
        )
        
    def performance_update(self, total_return: float, win_rate: float, drawdown: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            "Performance update",
            event_type="performance_update",
            total_return=total_return,
            win_rate=win_rate,
            drawdown=drawdown,
            **kwargs
        )

class TradingLogger:
    """Main logging class for the trading platform"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loggers = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup all logging handlers and formatters"""
        
        # Create logs directory
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Main platform logger
        self._setup_platform_logger(log_dir)
        
        # Trading-specific loggers
        self._setup_trading_logger(log_dir)
        self._setup_signal_logger(log_dir)
        self._setup_performance_logger(log_dir)
        self._setup_error_logger(log_dir)
        
    def _setup_platform_logger(self, log_dir: Path):
        """Setup main platform logger"""
        logger = logging.getLogger('trading_platform')
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(TradingFormatter())
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'trading_platform.log',
            maxBytes=self.config.get('max_log_size_mb', 100) * 1024 * 1024,
            backupCount=self.config.get('backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(TradingFormatter())
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        self.loggers['platform'] = logger
        
    def _setup_trading_logger(self, log_dir: Path):
        """Setup trading-specific logger"""
        logger = logging.getLogger('trading')
        logger.setLevel(logging.INFO)
        
        # JSON formatter for structured logs
        json_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(name)s %(levelname)s %(message)s'
        )
        
        # Trading file handler
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / 'trades.log',
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(json_formatter)
        
        logger.addHandler(trade_handler)
        self.loggers['trading'] = logger
        
    def _setup_signal_logger(self, log_dir: Path):
        """Setup signal generation logger"""
        logger = logging.getLogger('signals')
        logger.setLevel(logging.INFO)
        
        # Signal file handler
        signal_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / 'signals.log',
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        signal_handler.setLevel(logging.INFO)
        signal_handler.setFormatter(jsonlogger.JsonFormatter())
        
        logger.addHandler(signal_handler)
        self.loggers['signals'] = logger
        
    def _setup_performance_logger(self, log_dir: Path):
        """Setup performance metrics logger"""
        logger = logging.getLogger('performance')
        logger.setLevel(logging.INFO)
        
        # Performance file handler
        perf_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / 'performance.log',
            when='midnight',
            interval=1,
            backupCount=90,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(jsonlogger.JsonFormatter())
        
        logger.addHandler(perf_handler)
        self.loggers['performance'] = logger
        
    def _setup_error_logger(self, log_dir: Path):
        """Setup error logger with email alerts"""
        logger = logging.getLogger('errors')
        logger.setLevel(logging.ERROR)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'errors.log',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(TradingFormatter())
        
        logger.addHandler(error_handler)
        
        # Email handler for critical errors (if configured)
        if self.config.get('email_alerts_enabled', False):
            self._setup_email_handler(logger)
            
        self.loggers['errors'] = logger
        
    def _setup_email_handler(self, logger: logging.Logger):
        """Setup email handler for critical alerts"""
        try:
            email_handler = logging.handlers.SMTPHandler(
                mailhost=(self.config['smtp_server'], self.config.get('smtp_port', 587)),
                fromaddr=self.config['email_from'],
                toaddrs=self.config['email_recipients'],
                subject='Trading Platform Critical Error',
                credentials=(self.config['email_user'], self.config['email_password']),
                secure=()
            )
            email_handler.setLevel(logging.CRITICAL)
            logger.addHandler(email_handler)
        except KeyError as e:
            print(f"Email handler setup failed: missing config {e}")
            
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger by name"""
        return self.loggers.get(name, self.loggers['platform'])
        
    def log_trade_event(self, event_type: str, data: Dict[str, Any]):
        """Log trading event with structured data"""
        logger = self.loggers['trading']
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            **data
        }
        
        logger.info(json.dumps(log_data))
        
    def log_signal_event(self, signal_data: Dict[str, Any]):
        """Log signal generation event"""
        logger = self.loggers['signals']
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'signal_generated',
            **signal_data
        }
        
        logger.info(json.dumps(log_data))
        
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        logger = self.loggers['performance']
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'performance_metrics',
            **metrics
        }
        
        logger.info(json.dumps(log_data))
        
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        logger = self.loggers['errors']
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        logger.error(json.dumps(log_data), exc_info=True)

# Singleton instance
_trading_logger: Optional[TradingLogger] = None

def get_trading_logger(config: Dict[str, Any] = None) -> TradingLogger:
    """Get or create trading logger singleton"""
    global _trading_logger
    
    if _trading_logger is None:
        if config is None:
            config = {
                'log_level': 'INFO',
                'log_dir': 'logs',
                'max_log_size_mb': 100,
                'backup_count': 5
            }
        _trading_logger = TradingLogger(config)
        
    return _trading_logger

def log_trade_opened(symbol: str, side: str, quantity: float, price: float, **kwargs):
    """Convenience function to log trade opening"""
    logger = get_trading_logger()
    logger.log_trade_event('trade_opened', {
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'price': price,
        **kwargs
    })

def log_trade_closed(symbol: str, pnl: float, pnl_percent: float, reason: str, **kwargs):
    """Convenience function to log trade closing"""
    logger = get_trading_logger()
    logger.log_trade_event('trade_closed', {
        'symbol': symbol,
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'reason': reason,
        **kwargs
    })

def log_signal_generated(symbol: str, signal_type: str, confidence: float, **kwargs):
    """Convenience function to log signal generation"""
    logger = get_trading_logger()
    logger.log_signal_event({
        'symbol': symbol,
        'signal_type': signal_type,
        'confidence': confidence,
        **kwargs
    })

def log_performance_update(metrics: Dict[str, Any]):
    """Convenience function to log performance update"""
    logger = get_trading_logger()
    logger.log_performance_metrics(metrics)

def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
    """Convenience function to log error with context"""
    logger = get_trading_logger()
    logger.log_error(error, context)