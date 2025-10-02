"""
Database models and ORM definitions
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class TradingPair(Base):
    """Trading pair information"""
    __tablename__ = 'trading_pairs'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)
    exchange = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    min_order_size = Column(Float)
    max_order_size = Column(Float)
    price_precision = Column(Integer, default=8)
    quantity_precision = Column(Integer, default=6)
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketData(Base):
    """Market data storage"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timeframe = Column(String(5), nullable=False)  # 1m, 5m, 1h, etc.
    
class Trade(Base):
    """Individual trade records"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)  # BUY or SELL
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float)
    pnl_percent = Column(Float)
    commission = Column(Float)
    status = Column(String(10), default='OPEN')  # OPEN, CLOSED, CANCELLED
    exit_reason = Column(String(20))
    signal_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Signal(Base):
    """Trading signals"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(4), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_reward_ratio = Column(Float)
    indicators = Column(Text)  # JSON string of indicator values
    reasoning = Column(Text)  # JSON string of reasoning
    timestamp = Column(DateTime, nullable=False)
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Portfolio(Base):
    """Portfolio snapshots"""
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    total_equity = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    reserved_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    total_return_percent = Column(Float, default=0.0)
    open_positions = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)

class PerformanceMetrics(Base):
    """Performance metrics storage"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    recovery_factor = Column(Float, default=0.0)
    
class ModelPerformance(Base):
    """ML model performance tracking"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime, nullable=False)
    test_samples = Column(Integer)
    feature_count = Column(Integer)
    model_params = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemLog(Base):
    """System logging"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text)  # JSON string for additional data

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str = "sqlite:///data/trading_platform.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
        
    def close_session(self, session):
        """Close database session"""
        session.close()
        
    def save_market_data(self, symbol: str, data: dict, timeframe: str = '1h'):
        """Save market data to database"""
        session = self.get_session()
        try:
            market_data = MarketData(
                symbol=symbol,
                timestamp=data['timestamp'],
                open_price=data['open'],
                high_price=data['high'],
                low_price=data['low'],
                close_price=data['close'],
                volume=data['volume'],
                timeframe=timeframe
            )
            session.add(market_data)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
            
    def save_trade(self, trade_data: dict):
        """Save trade to database"""
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            return trade.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
            
    def save_signal(self, signal_data: dict):
        """Save trading signal to database"""
        session = self.get_session()
        try:
            signal = Signal(
                symbol=signal_data['symbol'],
                signal_type=signal_data['signal'],
                confidence=signal_data['confidence'],
                entry_price=signal_data['entry_price'],
                stop_loss=signal_data.get('stop_loss'),
                take_profit=signal_data.get('take_profit'),
                risk_reward_ratio=signal_data.get('risk_reward_ratio'),
                indicators=json.dumps(signal_data.get('indicators', {})),
                reasoning=json.dumps(signal_data.get('reasoning', [])),
                timestamp=signal_data['timestamp']
            )
            session.add(signal)
            session.commit()
            return signal.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
            
    def get_recent_trades(self, days: int = 30, limit: int = 100):
        """Get recent trades"""
        session = self.get_session()
        try:
            return session.query(Trade)\
                .filter(Trade.entry_time >= datetime.utcnow() - timedelta(days=days))\
                .order_by(Trade.entry_time.desc())\
                .limit(limit)\
                .all()
        finally:
            self.close_session(session)
            
    def get_performance_summary(self, days: int = 30):
        """Get performance summary"""
        session = self.get_session()
        try:
            return session.query(PerformanceMetrics)\
                .filter(PerformanceMetrics.date >= datetime.utcnow() - timedelta(days=days))\
                .order_by(PerformanceMetrics.date.desc())\
                .first()
        finally:
            self.close_session(session)