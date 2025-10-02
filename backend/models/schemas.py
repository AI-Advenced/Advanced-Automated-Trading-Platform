"""
Pydantic schemas for data validation and serialization
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

# Market Data Schemas
class MarketDataPoint(BaseModel):
    symbol: str
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)  
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high must be >= low')
        return v
        
    @validator('close')
    def close_within_range(cls, v, values):
        if 'high' in values and 'low' in values:
            if not (values['low'] <= v <= values['high']):
                raise ValueError('close must be between low and high')
        return v

class RealtimeData(BaseModel):
    symbol: str
    price: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    change_24h: Optional[float] = None
    percentage_change: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    timestamp: datetime
    source: str

# Signal Schemas
class TradingSignalBase(BaseModel):
    symbol: str
    signal_type: SignalType
    confidence: float = Field(..., ge=0, le=1)
    entry_price: float = Field(..., gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    risk_reward_ratio: Optional[float] = Field(None, gt=0)
    reasoning: List[str] = []
    
    @validator('stop_loss')
    def validate_stop_loss(cls, v, values):
        if v and 'entry_price' in values:
            entry = values['entry_price']
            signal = values.get('signal_type')
            if signal == SignalType.BUY and v >= entry:
                raise ValueError('Stop loss must be below entry price for BUY signals')
            elif signal == SignalType.SELL and v <= entry:
                raise ValueError('Stop loss must be above entry price for SELL signals')
        return v

class TradingSignal(TradingSignalBase):
    id: Optional[int] = None
    timestamp: datetime
    indicators: Dict[str, Any] = {}
    executed: bool = False
    
    class Config:
        orm_mode = True

# Portfolio Schemas  
class Position(BaseModel):
    symbol: str
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)
    entry_time: datetime
    position_type: OrderSide
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float
    unrealized_pnl_percent: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

class PortfolioSummary(BaseModel):
    initial_balance: float = Field(..., gt=0)
    current_balance: float = Field(..., ge=0)
    available_balance: float = Field(..., ge=0)
    reserved_balance: float = Field(..., ge=0)
    total_equity: float = Field(..., ge=0)
    unrealized_pnl: float
    realized_pnl: float
    total_return_percent: float
    open_positions_count: int = Field(..., ge=0)
    total_trades: int = Field(..., ge=0)
    win_rate_percent: float = Field(..., ge=0, le=100)
    positions: Dict[str, Position] = {}

# Trade Schemas
class TradeBase(BaseModel):
    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    signal_confidence: Optional[float] = Field(None, ge=0, le=1)

class TradeCreate(TradeBase):
    trade_id: str

class TradeUpdate(BaseModel):
    exit_price: Optional[float] = Field(None, gt=0)
    exit_time: Optional[datetime] = None
    status: Optional[TradeStatus] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    commission: Optional[float] = Field(None, ge=0)

class Trade(TradeBase):
    id: int
    trade_id: str
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    commission: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    exit_reason: Optional[str] = None
    created_at: datetime
    
    class Config:
        orm_mode = True

# Performance Schemas
class PerformanceMetrics(BaseModel):
    period_days: int = Field(..., gt=0)
    total_trades: int = Field(..., ge=0)
    profitable_trades: int = Field(..., ge=0)
    losing_trades: int = Field(..., ge=0)
    win_rate_percent: float = Field(..., ge=0, le=100)
    total_pnl: float
    average_win_percent: float
    average_loss_percent: float
    largest_win_percent: float
    largest_loss_percent: float
    profit_factor: float = Field(..., ge=0)
    sharpe_ratio: float
    max_drawdown_percent: float = Field(..., le=0)
    recovery_factor: float
    
class RiskMetrics(BaseModel):
    max_risk_per_trade: float = Field(..., gt=0, le=0.1)
    max_portfolio_risk: float = Field(..., gt=0, le=0.5)
    current_portfolio_risk: float = Field(..., ge=0)
    max_correlation_exposure: float = Field(..., gt=0, le=1)
    max_single_position: float = Field(..., gt=0, le=1)
    max_drawdown_percent: float = Field(..., le=0)
    var_95_percent: float  # Value at Risk 95%
    max_consecutive_losses: int = Field(..., ge=0)

# Configuration Schemas
class TradingConfig(BaseModel):
    initial_balance: float = Field(..., gt=0)
    max_risk_per_trade: float = Field(..., gt=0, le=0.1)
    min_signal_confidence: float = Field(..., ge=0, le=1)
    max_open_positions: int = Field(..., gt=0, le=50)
    cycle_interval_minutes: int = Field(..., gt=0, le=1440)
    watchlist: List[str] = Field(..., min_items=1)
    
class ExchangeConfig(BaseModel):
    name: str
    api_key: str
    secret: str
    passphrase: Optional[str] = None
    sandbox: bool = True
    testnet: bool = True
    
class RiskConfig(BaseModel):
    max_risk_per_trade: float = Field(0.02, gt=0, le=0.1)
    max_portfolio_risk: float = Field(0.10, gt=0, le=0.5)
    max_correlation_exposure: float = Field(0.60, gt=0, le=1)
    max_single_position: float = Field(0.15, gt=0, le=1)
    stop_loss_buffer: float = Field(1.1, gt=1, le=2)
    
# API Response Schemas
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime
    
class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    uptime_seconds: int
    active_positions: int
    last_signal_time: Optional[datetime] = None
    
class SystemStatus(BaseModel):
    platform_status: str
    trading_active: bool
    data_collection_active: bool
    signal_generation_active: bool
    portfolio_monitoring_active: bool
    last_cycle_time: Optional[datetime] = None
    next_cycle_time: Optional[datetime] = None
    
# WebSocket Schemas
class WSMessage(BaseModel):
    type: str
    data: Any
    timestamp: datetime
    
class PriceUpdate(BaseModel):
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    
class SignalUpdate(BaseModel):
    signal: TradingSignal
    
class PortfolioUpdate(BaseModel):
    portfolio: PortfolioSummary
    
# Notification Schemas
class NotificationConfig(BaseModel):
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    email_enabled: bool = False
    email_recipients: List[str] = []
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    
class Alert(BaseModel):
    type: str  # trade_open, trade_close, profit_target, stop_loss, error
    title: str
    message: str
    symbol: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical
    timestamp: datetime
    
# Validation helpers
def validate_symbol(symbol: str) -> str:
    """Validate trading symbol format"""
    if not symbol or '/' not in symbol:
        raise ValueError("Symbol must be in format BASE/QUOTE (e.g., BTC/USDT)")
    
    base, quote = symbol.split('/')
    if not base or not quote:
        raise ValueError("Both base and quote assets must be specified")
    
    return symbol.upper()

def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe format"""
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
    if timeframe not in valid_timeframes:
        raise ValueError(f"Timeframe must be one of: {valid_timeframes}")
    return timeframe