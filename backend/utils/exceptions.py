"""
Custom exceptions for the trading platform
"""

class TradingPlatformError(Exception):
    """Base exception for all trading platform errors"""
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        
    def __str__(self):
        return f"[{self.error_code}] {self.message}" if self.error_code else self.message

# Data Collection Exceptions
class DataCollectionError(TradingPlatformError):
    """Base exception for data collection errors"""
    pass

class ExchangeConnectionError(DataCollectionError):
    """Exchange connection failed"""
    def __init__(self, exchange: str, message: str = None):
        message = message or f"Failed to connect to exchange: {exchange}"
        super().__init__(message, "EXCHANGE_CONNECTION_ERROR", {"exchange": exchange})

class InvalidSymbolError(DataCollectionError):
    """Invalid trading symbol"""
    def __init__(self, symbol: str):
        super().__init__(
            f"Invalid trading symbol: {symbol}",
            "INVALID_SYMBOL",
            {"symbol": symbol}
        )

class DataNotAvailableError(DataCollectionError):
    """Requested data is not available"""
    def __init__(self, symbol: str, timeframe: str = None, exchange: str = None):
        message = f"Data not available for {symbol}"
        if timeframe:
            message += f" ({timeframe})"
        if exchange:
            message += f" on {exchange}"
            
        super().__init__(message, "DATA_NOT_AVAILABLE", {
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange
        })

class RateLimitExceededError(DataCollectionError):
    """API rate limit exceeded"""
    def __init__(self, exchange: str, retry_after: int = None):
        message = f"Rate limit exceeded for {exchange}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            
        super().__init__(message, "RATE_LIMIT_EXCEEDED", {
            "exchange": exchange,
            "retry_after": retry_after
        })

class WebSocketConnectionError(DataCollectionError):
    """WebSocket connection error"""
    def __init__(self, exchange: str, reason: str = None):
        message = f"WebSocket connection failed for {exchange}"
        if reason:
            message += f": {reason}"
            
        super().__init__(message, "WEBSOCKET_ERROR", {
            "exchange": exchange,
            "reason": reason
        })

# Machine Learning Exceptions
class MLModelError(TradingPlatformError):
    """Base exception for machine learning errors"""
    pass

class ModelNotTrainedError(MLModelError):
    """Model has not been trained yet"""
    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' has not been trained yet",
            "MODEL_NOT_TRAINED",
            {"model_name": model_name}
        )

class InsufficientTrainingDataError(MLModelError):
    """Insufficient data for model training"""
    def __init__(self, required: int, available: int):
        super().__init__(
            f"Insufficient training data. Required: {required}, Available: {available}",
            "INSUFFICIENT_TRAINING_DATA",
            {"required": required, "available": available}
        )

class ModelTrainingError(MLModelError):
    """Error during model training"""
    def __init__(self, model_name: str, details: str = None):
        message = f"Failed to train model '{model_name}'"
        if details:
            message += f": {details}"
            
        super().__init__(message, "MODEL_TRAINING_ERROR", {
            "model_name": model_name,
            "details": details
        })

class PredictionError(MLModelError):
    """Error during prediction"""
    def __init__(self, model_name: str, details: str = None):
        message = f"Prediction failed for model '{model_name}'"
        if details:
            message += f": {details}"
            
        super().__init__(message, "PREDICTION_ERROR", {
            "model_name": model_name,
            "details": details
        })

class FeatureCalculationError(MLModelError):
    """Error calculating features"""
    def __init__(self, feature_name: str, details: str = None):
        message = f"Failed to calculate feature '{feature_name}'"
        if details:
            message += f": {details}"
            
        super().__init__(message, "FEATURE_CALCULATION_ERROR", {
            "feature_name": feature_name,
            "details": details
        })

# Signal Generation Exceptions
class SignalGenerationError(TradingPlatformError):
    """Base exception for signal generation errors"""
    pass

class InvalidSignalError(SignalGenerationError):
    """Invalid trading signal"""
    def __init__(self, symbol: str, reason: str):
        super().__init__(
            f"Invalid signal for {symbol}: {reason}",
            "INVALID_SIGNAL",
            {"symbol": symbol, "reason": reason}
        )

class SignalFilterError(SignalGenerationError):
    """Signal filtered out by risk management"""
    def __init__(self, symbol: str, filter_reason: str):
        super().__init__(
            f"Signal for {symbol} filtered out: {filter_reason}",
            "SIGNAL_FILTERED",
            {"symbol": symbol, "filter_reason": filter_reason}
        )

class IndicatorCalculationError(SignalGenerationError):
    """Error calculating technical indicator"""
    def __init__(self, indicator: str, symbol: str, details: str = None):
        message = f"Failed to calculate {indicator} for {symbol}"
        if details:
            message += f": {details}"
            
        super().__init__(message, "INDICATOR_CALCULATION_ERROR", {
            "indicator": indicator,
            "symbol": symbol,
            "details": details
        })

# Portfolio Management Exceptions
class PortfolioError(TradingPlatformError):
    """Base exception for portfolio management errors"""
    pass

class InsufficientFundsError(PortfolioError):
    """Insufficient funds for trade"""
    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient funds. Required: ${required:.2f}, Available: ${available:.2f}",
            "INSUFFICIENT_FUNDS",
            {"required": required, "available": available}
        )

class PositionNotFoundError(PortfolioError):
    """Position not found"""
    def __init__(self, symbol: str):
        super().__init__(
            f"Position not found for symbol: {symbol}",
            "POSITION_NOT_FOUND",
            {"symbol": symbol}
        )

class RiskLimitExceededError(PortfolioError):
    """Risk limit exceeded"""
    def __init__(self, limit_type: str, current: float, maximum: float):
        super().__init__(
            f"{limit_type} limit exceeded. Current: {current:.2%}, Maximum: {maximum:.2%}",
            "RISK_LIMIT_EXCEEDED",
            {"limit_type": limit_type, "current": current, "maximum": maximum}
        )

class MaxPositionsExceededError(PortfolioError):
    """Maximum number of positions exceeded"""
    def __init__(self, current: int, maximum: int):
        super().__init__(
            f"Maximum positions exceeded. Current: {current}, Maximum: {maximum}",
            "MAX_POSITIONS_EXCEEDED",
            {"current": current, "maximum": maximum}
        )

class PositionSizeError(PortfolioError):
    """Invalid position size"""
    def __init__(self, size: float, reason: str):
        super().__init__(
            f"Invalid position size {size}: {reason}",
            "POSITION_SIZE_ERROR",
            {"size": size, "reason": reason}
        )

# Trading Execution Exceptions
class TradingError(TradingPlatformError):
    """Base exception for trading errors"""
    pass

class OrderExecutionError(TradingError):
    """Order execution failed"""
    def __init__(self, symbol: str, side: str, quantity: float, reason: str = None):
        message = f"Failed to execute {side} order for {quantity} {symbol}"
        if reason:
            message += f": {reason}"
            
        super().__init__(message, "ORDER_EXECUTION_ERROR", {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "reason": reason
        })

class InvalidOrderError(TradingError):
    """Invalid order parameters"""
    def __init__(self, symbol: str, details: str):
        super().__init__(
            f"Invalid order for {symbol}: {details}",
            "INVALID_ORDER",
            {"symbol": symbol, "details": details}
        )

class MarketClosedError(TradingError):
    """Market is closed"""
    def __init__(self, symbol: str):
        super().__init__(
            f"Market is closed for {symbol}",
            "MARKET_CLOSED",
            {"symbol": symbol}
        )

class ExchangeMaintenanceError(TradingError):
    """Exchange is under maintenance"""
    def __init__(self, exchange: str):
        super().__init__(
            f"Exchange {exchange} is under maintenance",
            "EXCHANGE_MAINTENANCE",
            {"exchange": exchange}
        )

# Configuration Exceptions
class ConfigurationError(TradingPlatformError):
    """Base exception for configuration errors"""
    pass

class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing"""
    def __init__(self, config_name: str):
        super().__init__(
            f"Missing required configuration: {config_name}",
            "MISSING_CONFIGURATION",
            {"config_name": config_name}
        )

class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration value"""
    def __init__(self, config_name: str, value: str, reason: str):
        super().__init__(
            f"Invalid configuration {config_name}='{value}': {reason}",
            "INVALID_CONFIGURATION",
            {"config_name": config_name, "value": value, "reason": reason}
        )

class ConfigurationValidationError(ConfigurationError):
    """Configuration validation failed"""
    def __init__(self, errors: list):
        message = "Configuration validation failed:\n" + "\n".join(errors)
        super().__init__(message, "CONFIGURATION_VALIDATION_ERROR", {"errors": errors})

# Database Exceptions
class DatabaseError(TradingPlatformError):
    """Base exception for database errors"""
    pass

class ConnectionError(DatabaseError):
    """Database connection error"""
    def __init__(self, database_url: str, details: str = None):
        message = f"Failed to connect to database: {database_url}"
        if details:
            message += f" ({details})"
            
        super().__init__(message, "DATABASE_CONNECTION_ERROR", {
            "database_url": database_url,
            "details": details
        })

class MigrationError(DatabaseError):
    """Database migration error"""
    def __init__(self, migration: str, details: str = None):
        message = f"Migration failed: {migration}"
        if details:
            message += f" ({details})"
            
        super().__init__(message, "MIGRATION_ERROR", {
            "migration": migration,
            "details": details
        })

class DataIntegrityError(DatabaseError):
    """Data integrity violation"""
    def __init__(self, table: str, constraint: str, details: str = None):
        message = f"Data integrity violation in {table}: {constraint}"
        if details:
            message += f" ({details})"
            
        super().__init__(message, "DATA_INTEGRITY_ERROR", {
            "table": table,
            "constraint": constraint,
            "details": details
        })

# System Exceptions
class SystemError(TradingPlatformError):
    """Base exception for system errors"""
    pass

class ServiceUnavailableError(SystemError):
    """Service is unavailable"""
    def __init__(self, service: str, reason: str = None):
        message = f"Service unavailable: {service}"
        if reason:
            message += f" ({reason})"
            
        super().__init__(message, "SERVICE_UNAVAILABLE", {
            "service": service,
            "reason": reason
        })

class ResourceExhaustedError(SystemError):
    """System resources exhausted"""
    def __init__(self, resource: str, usage: float, limit: float):
        super().__init__(
            f"Resource exhausted: {resource} usage {usage:.1%} exceeds limit {limit:.1%}",
            "RESOURCE_EXHAUSTED",
            {"resource": resource, "usage": usage, "limit": limit}
        )

class TimeoutError(SystemError):
    """Operation timeout"""
    def __init__(self, operation: str, timeout_seconds: int):
        super().__init__(
            f"Operation timeout: {operation} exceeded {timeout_seconds}s",
            "TIMEOUT",
            {"operation": operation, "timeout_seconds": timeout_seconds}
        )

# Notification Exceptions  
class NotificationError(TradingPlatformError):
    """Base exception for notification errors"""
    pass

class NotificationDeliveryError(NotificationError):
    """Failed to deliver notification"""
    def __init__(self, channel: str, recipient: str, reason: str = None):
        message = f"Failed to deliver notification via {channel} to {recipient}"
        if reason:
            message += f": {reason}"
            
        super().__init__(message, "NOTIFICATION_DELIVERY_ERROR", {
            "channel": channel,
            "recipient": recipient,
            "reason": reason
        })

class InvalidNotificationConfigError(NotificationError):
    """Invalid notification configuration"""
    def __init__(self, channel: str, details: str):
        super().__init__(
            f"Invalid {channel} notification config: {details}",
            "INVALID_NOTIFICATION_CONFIG",
            {"channel": channel, "details": details}
        )

# Authentication/Authorization Exceptions
class AuthenticationError(TradingPlatformError):
    """Authentication failed"""
    def __init__(self, service: str, reason: str = None):
        message = f"Authentication failed for {service}"
        if reason:
            message += f": {reason}"
            
        super().__init__(message, "AUTHENTICATION_ERROR", {
            "service": service,
            "reason": reason
        })

class AuthorizationError(TradingPlatformError):
    """Authorization failed"""  
    def __init__(self, action: str, resource: str):
        super().__init__(
            f"Not authorized to {action} on {resource}",
            "AUTHORIZATION_ERROR",
            {"action": action, "resource": resource}
        )

class InvalidCredentialsError(TradingPlatformError):
    """Invalid credentials"""
    def __init__(self, service: str):
        super().__init__(
            f"Invalid credentials for {service}",
            "INVALID_CREDENTIALS",
            {"service": service}
        )

# Utility Functions
def handle_exception(exc: Exception, context: dict = None) -> TradingPlatformError:
    """Convert generic exceptions to trading platform exceptions"""
    
    if isinstance(exc, TradingPlatformError):
        return exc
        
    # Map common exceptions
    exception_mapping = {
        ConnectionRefusedError: lambda e: ExchangeConnectionError("unknown", str(e)),
        TimeoutError: lambda e: TimeoutError("unknown", 30),
        ValueError: lambda e: InvalidConfigurationError("unknown", str(e), "Invalid value"),
        KeyError: lambda e: MissingConfigurationError(str(e).strip("'")),
        FileNotFoundError: lambda e: ConfigurationError(f"File not found: {e.filename}"),
        PermissionError: lambda e: AuthorizationError("access", str(e)),
    }
    
    # Try to map the exception
    exc_type = type(exc)
    if exc_type in exception_mapping:
        return exception_mapping[exc_type](exc)
    
    # Default to generic trading platform error
    return TradingPlatformError(
        f"Unexpected error: {str(exc)}",
        "UNEXPECTED_ERROR",
        {"original_exception": exc_type.__name__, "context": context or {}}
    )