"""
Database initialization script for the trading platform.
Creates initial data, default configurations, and test data.
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from decimal import Decimal

# Add the parent directory to the path so we can import backend modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.models.database import DatabaseManager
from backend.models.schemas import TradingSignal, Trade, PerformanceMetrics
from backend.config.settings import get_settings, StrategySettings, RiskSettings
from backend.utils.logger import TradingLogger

# Initialize logger
logger = TradingLogger()

class DatabaseInitializer:
    """Handles database initialization with default and test data"""
    
    def __init__(self, database_url: str):
        self.db_manager = DatabaseManager(database_url)
    
    async def init_db(self):
        """Initialize database connection"""
        await self.db_manager.init_db()
    
    async def close(self):
        """Close database connection"""
        await self.db_manager.close()
    
    async def create_default_configurations(self):
        """Create default system configurations"""
        try:
            logger.info("Creating default configurations...")
            
            async with self.db_manager.get_session() as session:
                # Create default strategy configurations
                from backend.models.database import Configuration
                
                # Strategy configurations
                strategy_configs = [
                    Configuration(
                        key="strategy_ma_crossover",
                        value=json.dumps(StrategySettings.MA_CROSSOVER),
                        description="Moving Average Crossover Strategy Configuration"
                    ),
                    Configuration(
                        key="strategy_rsi",
                        value=json.dumps(StrategySettings.RSI_STRATEGY),
                        description="RSI Strategy Configuration"
                    ),
                    Configuration(
                        key="strategy_macd",
                        value=json.dumps(StrategySettings.MACD_STRATEGY),
                        description="MACD Strategy Configuration"
                    ),
                    Configuration(
                        key="strategy_ml",
                        value=json.dumps(StrategySettings.ML_STRATEGY),
                        description="ML Prediction Strategy Configuration"
                    ),
                ]
                
                # Risk management configurations
                risk_configs = [
                    Configuration(
                        key="risk_position_sizing",
                        value=json.dumps(RiskSettings.POSITION_SIZING_METHODS),
                        description="Position Sizing Methods Configuration"
                    ),
                    Configuration(
                        key="risk_limits",
                        value=json.dumps(RiskSettings.RISK_LIMITS),
                        description="Risk Limits Configuration"
                    ),
                    Configuration(
                        key="risk_stop_loss",
                        value=json.dumps(RiskSettings.STOP_LOSS_CONFIGS),
                        description="Stop Loss Configuration"
                    ),
                ]
                
                # Add all configurations
                for config in strategy_configs + risk_configs:
                    # Check if configuration already exists
                    existing = await self.db_manager.get_configuration(config.key)
                    if not existing:
                        session.add(config)
                
                await session.commit()
                logger.info("Default configurations created successfully")
                
        except Exception as e:
            logger.error(f"Error creating default configurations: {str(e)}")
            raise
    
    async def create_default_users(self):
        """Create default system users"""
        try:
            logger.info("Creating default users...")
            
            async with self.db_manager.get_session() as session:
                from backend.models.database import User
                
                # Create default admin user
                admin_user = User(
                    username="admin",
                    email="admin@tradingplatform.com",
                    full_name="System Administrator",
                    hashed_password="$2b$12$dummy_hash_for_admin",  # Change in production
                    is_active=True,
                    is_superuser=True
                )
                
                # Create default trading bot user
                bot_user = User(
                    username="trading_bot",
                    email="bot@tradingplatform.com",
                    full_name="Trading Bot",
                    hashed_password="$2b$12$dummy_hash_for_bot",
                    is_active=True,
                    is_superuser=False
                )
                
                # Check if users already exist
                existing_admin = await session.execute(
                    session.query(User).filter(User.username == "admin")
                )
                if not existing_admin.scalar():
                    session.add(admin_user)
                
                existing_bot = await session.execute(
                    session.query(User).filter(User.username == "trading_bot")
                )
                if not existing_bot.scalar():
                    session.add(bot_user)
                
                await session.commit()
                logger.info("Default users created successfully")
                
        except Exception as e:
            logger.error(f"Error creating default users: {str(e)}")
            raise
    
    async def create_test_data(self):
        """Create test data for development and testing"""
        try:
            logger.info("Creating test data...")
            
            await self._create_test_trades()
            await self._create_test_signals()
            await self._create_test_portfolio()
            await self._create_test_performance_metrics()
            
            logger.info("Test data created successfully")
            
        except Exception as e:
            logger.error(f"Error creating test data: {str(e)}")
            raise
    
    async def _create_test_trades(self):
        """Create sample trade data"""
        async with self.db_manager.get_session() as session:
            from backend.models.database import Trade, User
            
            # Get bot user
            bot_user = await session.execute(
                session.query(User).filter(User.username == "trading_bot")
            )
            bot_user = bot_user.scalar_one()
            
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            
            trades = []
            base_time = datetime.utcnow() - timedelta(days=30)
            
            for i in range(100):
                symbol = np.random.choice(symbols)
                side = np.random.choice(["BUY", "SELL"])
                
                # Generate realistic prices
                price_ranges = {
                    "BTCUSDT": (45000, 55000),
                    "ETHUSDT": (2800, 3200),
                    "ADAUSDT": (0.4, 0.6),
                    "DOTUSDT": (5, 8),
                    "LINKUSDT": (12, 18)
                }
                
                min_price, max_price = price_ranges[symbol]
                price = np.random.uniform(min_price, max_price)
                quantity = np.random.uniform(0.001, 0.1)
                
                # Calculate P&L (simulate some wins and losses)
                pnl = np.random.normal(0, price * quantity * 0.02)
                
                trade = Trade(
                    id=f"test_trade_{i:03d}",
                    user_id=bot_user.id,
                    symbol=symbol,
                    side=side,
                    quantity=Decimal(str(round(quantity, 6))),
                    price=Decimal(str(round(price, 2))),
                    timestamp=base_time + timedelta(hours=i * 2),
                    status="FILLED",
                    pnl=Decimal(str(round(pnl, 2))),
                    commission=Decimal(str(round(price * quantity * 0.001, 6))),
                    order_type="MARKET"
                )
                
                trades.append(trade)
            
            # Add trades to session
            for trade in trades:
                session.add(trade)
            
            await session.commit()
            logger.info(f"Created {len(trades)} test trades")
    
    async def _create_test_signals(self):
        """Create sample trading signals"""
        async with self.db_manager.get_session() as session:
            from backend.models.database import Signal
            
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            actions = ["BUY", "SELL", "HOLD"]
            
            signals = []
            base_time = datetime.utcnow() - timedelta(days=7)
            
            for i in range(50):
                symbol = np.random.choice(symbols)
                action = np.random.choice(actions)
                confidence = np.random.uniform(0.6, 0.95)
                
                # Generate realistic prices
                price_ranges = {
                    "BTCUSDT": (45000, 55000),
                    "ETHUSDT": (2800, 3200),
                    "ADAUSDT": (0.4, 0.6),
                    "DOTUSDT": (5, 8),
                    "LINKUSDT": (12, 18)
                }
                
                min_price, max_price = price_ranges[symbol]
                price = np.random.uniform(min_price, max_price)
                
                # Generate mock indicators
                indicators = {
                    "rsi": np.random.uniform(20, 80),
                    "macd": np.random.uniform(-0.1, 0.1),
                    "sma_20": price * np.random.uniform(0.98, 1.02),
                    "sma_50": price * np.random.uniform(0.95, 1.05),
                    "bb_upper": price * 1.02,
                    "bb_lower": price * 0.98,
                    "volume_ratio": np.random.uniform(0.5, 2.0)
                }
                
                signal = Signal(
                    id=f"test_signal_{i:03d}",
                    symbol=symbol,
                    action=action,
                    confidence=Decimal(str(round(confidence, 3))),
                    price=Decimal(str(round(price, 2))),
                    timestamp=base_time + timedelta(hours=i * 3),
                    indicators=indicators,
                    strategy="test_strategy"
                )
                
                signals.append(signal)
            
            # Add signals to session
            for signal in signals:
                session.add(signal)
            
            await session.commit()
            logger.info(f"Created {len(signals)} test signals")
    
    async def _create_test_portfolio(self):
        """Create sample portfolio data"""
        async with self.db_manager.get_session() as session:
            from backend.models.database import Portfolio, User
            
            # Get bot user
            bot_user = await session.execute(
                session.query(User).filter(User.username == "trading_bot")
            )
            bot_user = bot_user.scalar_one()
            
            # Create portfolio positions
            positions = [
                {
                    "symbol": "BTCUSDT",
                    "quantity": Decimal("0.05"),
                    "avg_price": Decimal("50000.00"),
                    "current_price": Decimal("51000.00")
                },
                {
                    "symbol": "ETHUSDT",
                    "quantity": Decimal("1.2"),
                    "avg_price": Decimal("3000.00"),
                    "current_price": Decimal("3100.00")
                },
                {
                    "symbol": "ADAUSDT",
                    "quantity": Decimal("5000"),
                    "avg_price": Decimal("0.50"),
                    "current_price": Decimal("0.52")
                }
            ]
            
            for pos in positions:
                portfolio_item = Portfolio(
                    user_id=bot_user.id,
                    symbol=pos["symbol"],
                    quantity=pos["quantity"],
                    avg_price=pos["avg_price"],
                    current_price=pos["current_price"],
                    last_updated=datetime.utcnow()
                )
                session.add(portfolio_item)
            
            await session.commit()
            logger.info(f"Created {len(positions)} test portfolio positions")
    
    async def _create_test_performance_metrics(self):
        """Create sample performance metrics"""
        async with self.db_manager.get_session() as session:
            from backend.models.database import PerformanceMetrics
            
            timeframes = ["1d", "7d", "30d"]
            base_time = datetime.utcnow()
            
            metrics = []
            
            for timeframe in timeframes:
                # Generate realistic performance metrics
                total_return = np.random.uniform(0.05, 0.25)  # 5-25% return
                sharpe_ratio = np.random.uniform(1.2, 2.5)
                max_drawdown = np.random.uniform(0.02, 0.08)  # 2-8% drawdown
                win_rate = np.random.uniform(0.55, 0.75)  # 55-75% win rate
                
                metric = PerformanceMetrics(
                    timeframe=timeframe,
                    total_return=Decimal(str(round(total_return, 4))),
                    sharpe_ratio=Decimal(str(round(sharpe_ratio, 3))),
                    max_drawdown=Decimal(str(round(max_drawdown, 4))),
                    win_rate=Decimal(str(round(win_rate, 3))),
                    total_trades=np.random.randint(50, 200),
                    profit_factor=Decimal(str(round(np.random.uniform(1.5, 2.8), 2))),
                    timestamp=base_time
                )
                
                metrics.append(metric)
            
            # Add metrics to session
            for metric in metrics:
                session.add(metric)
            
            await session.commit()
            logger.info(f"Created {len(metrics)} test performance metrics")

async def main():
    """Main initialization function"""
    parser = argparse.ArgumentParser(description="Initialize trading platform database")
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Include test data for development"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force initialization even if data exists"
    )
    
    args = parser.parse_args()
    
    try:
        # Get settings
        settings = get_settings()
        
        # Create initializer
        initializer = DatabaseInitializer(settings.database_url)
        
        # Initialize database connection
        await initializer.init_db()
        
        logger.info("Starting database initialization...")
        
        # Create default configurations
        await initializer.create_default_configurations()
        
        # Create default users
        await initializer.create_default_users()
        
        # Create test data if requested
        if args.test_data:
            await initializer.create_test_data()
        
        # Close connection
        await initializer.close()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())