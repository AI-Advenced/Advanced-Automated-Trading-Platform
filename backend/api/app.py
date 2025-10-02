"""
FastAPI application for the automated trading platform.
Provides REST API endpoints for monitoring and controlling the trading system.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
from typing import List, Optional, Dict, Any
import uvicorn

from ..models.database import DatabaseManager
from ..models.schemas import (
    TradingSignal, PortfolioSummary, Trade, PerformanceMetrics,
    StrategyConfig, TradingStatus, SystemHealth
)
from ..core.data_collector import DataCollector
from ..core.ml_models import TradingMLModels
from ..core.signal_generator import TradingSignalGenerator
from ..core.portfolio_manager import PortfolioManager
from ..core.trading_watcher import TradingWatcher
from ..main import TradingPlatform
from ..utils.logger import TradingLogger
from ..utils.exceptions import TradingPlatformError
from ..config.settings import get_settings

# Initialize logger
logger = TradingLogger()

# Global trading platform instance
trading_platform: Optional[TradingPlatform] = None
db_manager: Optional[DatabaseManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global trading_platform, db_manager
    
    # Startup
    try:
        logger.info("Starting trading platform API...")
        settings = get_settings()
        
        # Initialize database
        db_manager = DatabaseManager(settings.database_url)
        await db_manager.init_db()
        
        # Initialize trading platform
        trading_platform = TradingPlatform()
        
        logger.info("Trading platform API started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start trading platform API: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down trading platform API...")
        if trading_platform:
            await trading_platform.shutdown()
        if db_manager:
            await db_manager.close()
        logger.info("Trading platform API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Automated Trading Platform API",
    description="Advanced AI-powered cryptocurrency trading platform with ML integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database manager
async def get_db_manager():
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db_manager

# Dependency to get trading platform
async def get_trading_platform():
    if not trading_platform:
        raise HTTPException(status_code=503, detail="Trading platform not initialized")
    return trading_platform

# Health check endpoint
@app.get("/health", response_model=SystemHealth)
async def health_check():
    """System health check"""
    try:
        status = "healthy"
        components = {
            "database": "healthy" if db_manager else "unavailable",
            "trading_platform": "healthy" if trading_platform else "unavailable",
        }
        
        if trading_platform:
            platform_status = await trading_platform.get_system_status()
            components.update(platform_status.get("components", {}))
        
        overall_healthy = all(status == "healthy" for status in components.values())
        
        return SystemHealth(
            status="healthy" if overall_healthy else "degraded",
            components=components,
            timestamp=asyncio.get_event_loop().time()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return SystemHealth(
            status="unhealthy",
            components={"error": str(e)},
            timestamp=asyncio.get_event_loop().time()
        )

# Trading status endpoints
@app.get("/api/v1/status", response_model=TradingStatus)
async def get_trading_status(platform: TradingPlatform = Depends(get_trading_platform)):
    """Get current trading system status"""
    try:
        status = await platform.get_trading_status()
        return TradingStatus(**status)
    except Exception as e:
        logger.error(f"Error getting trading status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/start")
async def start_trading(
    background_tasks: BackgroundTasks,
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Start the trading system"""
    try:
        background_tasks.add_task(platform.start_trading)
        return {"message": "Trading started", "status": "starting"}
    except Exception as e:
        logger.error(f"Error starting trading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/stop")
async def stop_trading(platform: TradingPlatform = Depends(get_trading_platform)):
    """Stop the trading system"""
    try:
        await platform.stop_trading()
        return {"message": "Trading stopped", "status": "stopped"}
    except Exception as e:
        logger.error(f"Error stopping trading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio endpoints
@app.get("/api/v1/portfolio", response_model=PortfolioSummary)
async def get_portfolio(
    db: DatabaseManager = Depends(get_db_manager),
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get current portfolio summary"""
    try:
        portfolio_data = await platform.get_portfolio_summary()
        return PortfolioSummary(**portfolio_data)
    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/portfolio/positions")
async def get_positions(
    symbol: Optional[str] = None,
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get current positions"""
    try:
        positions = await platform.get_positions(symbol)
        return {"positions": positions}
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Signal endpoints
@app.get("/api/v1/signals", response_model=List[TradingSignal])
async def get_signals(
    symbol: Optional[str] = None,
    limit: int = 50,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get recent trading signals"""
    try:
        signals = await db.get_recent_signals(symbol=symbol, limit=limit)
        return [TradingSignal.from_orm(signal) for signal in signals]
    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/signals/active", response_model=List[TradingSignal])
async def get_active_signals(
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get currently active trading signals"""
    try:
        signals = await platform.get_active_signals()
        return [TradingSignal(**signal) for signal in signals]
    except Exception as e:
        logger.error(f"Error getting active signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Trade endpoints
@app.get("/api/v1/trades", response_model=List[Trade])
async def get_trades(
    symbol: Optional[str] = None,
    limit: int = 100,
    status: Optional[str] = None,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get trade history"""
    try:
        trades = await db.get_trades(
            symbol=symbol,
            limit=limit,
            status=status
        )
        return [Trade.from_orm(trade) for trade in trades]
    except Exception as e:
        logger.error(f"Error getting trades: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trades/{trade_id}", response_model=Trade)
async def get_trade(
    trade_id: str,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get specific trade details"""
    try:
        trade = await db.get_trade_by_id(trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        return Trade.from_orm(trade)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trade {trade_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance endpoints
@app.get("/api/v1/performance", response_model=PerformanceMetrics)
async def get_performance(
    timeframe: str = "1d",
    db: DatabaseManager = Depends(get_db_manager),
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get performance metrics"""
    try:
        performance = await platform.get_performance_metrics(timeframe)
        return PerformanceMetrics(**performance)
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/performance/charts")
async def get_performance_charts(
    timeframe: str = "7d",
    chart_type: str = "equity_curve",
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get performance chart data"""
    try:
        chart_data = await platform.get_performance_charts(timeframe, chart_type)
        return {"chart_data": chart_data}
    except Exception as e:
        logger.error(f"Error getting performance charts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Market data endpoints
@app.get("/api/v1/market/{symbol}")
async def get_market_data(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get market data for a symbol"""
    try:
        market_data = await platform.get_market_data(symbol, timeframe, limit)
        return {"symbol": symbol, "data": market_data}
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market/{symbol}/analysis")
async def get_technical_analysis(
    symbol: str,
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get technical analysis for a symbol"""
    try:
        analysis = await platform.get_technical_analysis(symbol)
        return {"symbol": symbol, "analysis": analysis}
    except Exception as e:
        logger.error(f"Error getting technical analysis for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/api/v1/config/strategy", response_model=StrategyConfig)
async def get_strategy_config(
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get current strategy configuration"""
    try:
        config = await platform.get_strategy_config()
        return StrategyConfig(**config)
    except Exception as e:
        logger.error(f"Error getting strategy config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/config/strategy")
async def update_strategy_config(
    config: StrategyConfig,
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Update strategy configuration"""
    try:
        await platform.update_strategy_config(config.dict())
        return {"message": "Strategy configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating strategy config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ML Model endpoints
@app.get("/api/v1/models/status")
async def get_models_status(
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get ML models training status"""
    try:
        status = await platform.get_models_status()
        return {"models_status": status}
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/retrain")
async def retrain_models(
    background_tasks: BackgroundTasks,
    symbol: Optional[str] = None,
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Trigger ML models retraining"""
    try:
        background_tasks.add_task(platform.retrain_models, symbol)
        return {"message": "Model retraining initiated", "status": "training"}
    except Exception as e:
        logger.error(f"Error initiating model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk management endpoints
@app.get("/api/v1/risk/metrics")
async def get_risk_metrics(
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get current risk metrics"""
    try:
        metrics = await platform.get_risk_metrics()
        return {"risk_metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/limits")
async def get_risk_limits(
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get current risk limits"""
    try:
        limits = await platform.get_risk_limits()
        return {"risk_limits": limits}
    except Exception as e:
        logger.error(f"Error getting risk limits: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/risk/limits")
async def update_risk_limits(
    limits: Dict[str, Any],
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Update risk limits"""
    try:
        await platform.update_risk_limits(limits)
        return {"message": "Risk limits updated successfully"}
    except Exception as e:
        logger.error(f"Error updating risk limits: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting endpoints
@app.post("/api/v1/backtest")
async def run_backtest(
    background_tasks: BackgroundTasks,
    config: Dict[str, Any],
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Run backtesting with specified configuration"""
    try:
        task_id = await platform.start_backtest(config)
        return {"message": "Backtesting initiated", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error starting backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/backtest/{task_id}")
async def get_backtest_results(
    task_id: str,
    platform: TradingPlatform = Depends(get_trading_platform)
):
    """Get backtesting results"""
    try:
        results = await platform.get_backtest_results(task_id)
        if not results:
            raise HTTPException(status_code=404, detail="Backtest results not found")
        return {"task_id": task_id, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.active_connections.discard(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle ping/pong
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Error handlers
@app.exception_handler(TradingPlatformError)
async def trading_error_handler(request, exc: TradingPlatformError):
    logger.error(f"Trading platform error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": "Trading platform error", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_error_handler(request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Start the server
if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )