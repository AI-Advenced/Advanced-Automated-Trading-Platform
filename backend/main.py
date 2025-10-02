"""
Advanced Automated Trading Platform
Main orchestrator that coordinates all trading components
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
from pathlib import Path

# Core components
from core.data_collector import DataCollector
from core.ml_models import TradingMLModels
from core.signal_generator import TradingSignalGenerator
from core.portfolio_manager import PortfolioManager
from core.trading_watcher import TradingWatcher

# Configuration
from config.settings import TradingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_platform.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class TradingPlatform:
    """
    Main automated trading platform orchestrator
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the trading platform"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Advanced Trading Platform...")
        
        # Load configuration
        self.config = TradingConfig(config_path)
        
        # Platform state
        self.running = False
        self.cycle_count = 0
        self.start_time = None
        
        # Initialize core components
        self._initialize_components()
        
        # Performance tracking
        self.performance_history = []
        self.last_retrain_time = datetime.now()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            # Data collector
            self.data_collector = DataCollector(self.config.api_config)
            self.logger.info("✅ Data collector initialized")
            
            # ML models
            self.ml_models = TradingMLModels()
            self.logger.info("✅ ML models initialized")
            
            # Signal generator
            self.signal_generator = TradingSignalGenerator(
                self.ml_models, 
                self.config.signal_config
            )
            self.logger.info("✅ Signal generator initialized")
            
            # Portfolio manager
            self.portfolio_manager = PortfolioManager(
                initial_balance=self.config.trading_config['initial_balance'],
                config=self.config.risk_config
            )
            self.logger.info("✅ Portfolio manager initialized")
            
            # Trading watcher
            self.trading_watcher = TradingWatcher(
                self.ml_models,
                self.config.monitoring_config
            )
            self.logger.info("✅ Trading watcher initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    async def initialize_models(self):
        """Initialize and train ML models with historical data"""
        self.logger.info("🧠 Initializing ML models with historical data...")
        
        watchlist = self.config.trading_config['watchlist']
        training_results = {}
        
        for symbol in watchlist:
            try:
                self.logger.info(f"📊 Fetching historical data for {symbol}...")
                
                # Get historical data for training
                historical_data = await self.data_collector.get_historical_data(
                    symbol=symbol,
                    timeframe='1h',
                    limit=2000  # ~83 days of hourly data
                )
                
                if historical_data.empty:
                    self.logger.warning(f"⚠️ No historical data available for {symbol}")
                    continue
                
                self.logger.info(f"📈 Training models for {symbol} with {len(historical_data)} data points...")
                
                # Train direction prediction model
                direction_results = self.ml_models.train_direction_model(historical_data)
                self.logger.info(f"✅ Direction model trained for {symbol}: "
                               f"Accuracy={direction_results['test_accuracy']:.2%}, "
                               f"F1={direction_results['f1_score']:.3f}")
                
                # Train LSTM price prediction model
                lstm_results = self.ml_models.train_lstm_model(historical_data)
                self.logger.info(f"✅ LSTM model trained for {symbol}: "
                               f"Val Loss={lstm_results['final_val_loss']:.6f}")
                
                training_results[symbol] = {
                    'direction_accuracy': direction_results['test_accuracy'],
                    'direction_f1': direction_results['f1_score'],
                    'lstm_val_loss': lstm_results['final_val_loss'],
                    'feature_count': len(direction_results['feature_columns'])
                }
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train models for {symbol}: {e}")
        
        # Save trained models
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        self.ml_models.save_models(str(models_dir))
        self.logger.info(f"💾 Models saved to {models_dir}")
        
        # Log training summary
        if training_results:
            avg_accuracy = sum(r['direction_accuracy'] for r in training_results.values()) / len(training_results)
            self.logger.info(f"🎯 Model training completed. Average accuracy: {avg_accuracy:.2%}")
            self.logger.info(f"📊 Trained models for {len(training_results)} symbols: {list(training_results.keys())}")
        
        return training_results
    
    async def collect_market_data(self) -> Dict[str, Dict]:
        """Collect current market data for all watchlist symbols"""
        self.logger.debug("📡 Collecting market data...")
        
        watchlist = self.config.trading_config['watchlist']
        market_data = {}
        
        # Collect data for all symbols
        tasks = []
        for symbol in watchlist:
            # Real-time data
            real_time_task = self.data_collector.get_real_time_data(symbol)
            tasks.append((symbol, 'real_time', real_time_task))
            
            # Historical data for analysis
            historical_task = self.data_collector.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                limit=200  # Last 200 hours for analysis
            )
            tasks.append((symbol, 'historical', historical_task))
        
        # Execute all data collection tasks
        for symbol, data_type, task in tasks:
            try:
                result = await task
                
                if symbol not in market_data:
                    market_data[symbol] = {}
                
                market_data[symbol][data_type] = result
                
            except Exception as e:
                self.logger.error(f"❌ Failed to collect {data_type} data for {symbol}: {e}")
        
        # Filter out symbols with incomplete data
        complete_data = {
            symbol: data for symbol, data in market_data.items()
            if 'real_time' in data and 'historical' in data 
            and data['real_time'] is not None 
            and not data['historical'].empty
        }
        
        self.logger.debug(f"📊 Collected data for {len(complete_data)} symbols")
        return complete_data
    
    async def generate_trading_signals(self, market_data: Dict) -> Dict[str, Dict]:
        """Generate trading signals for all symbols"""
        self.logger.debug("🎯 Generating trading signals...")
        
        # Prepare historical data for signal generation
        historical_data = {
            symbol: data['historical'] 
            for symbol, data in market_data.items()
            if 'historical' in data
        }
        
        # Generate signals
        signals = self.signal_generator.generate_signals_for_multiple_assets(historical_data)
        
        # Filter signals by risk criteria
        filtered_signals = self.signal_generator.filter_signals_by_risk(
            signals,
            max_risk_per_trade=self.config.risk_config['max_risk_per_trade']
        )
        
        self.logger.info(f"📈 Generated {len(signals)} signals, {len(filtered_signals)} passed risk filter")
        
        # Log signal details
        for symbol, signal in filtered_signals.items():
            self.logger.info(f"🎯 {symbol}: {signal['signal']} "
                           f"(confidence: {signal['confidence']:.1%}, "
                           f"R/R: {signal.get('risk_reward_ratio', 'N/A'):.1f})")
        
        return filtered_signals
    
    async def execute_trades(self, signals: Dict[str, Dict]):
        """Execute trades based on generated signals"""
        self.logger.debug("💼 Executing trades...")
        
        executed_trades = []
        
        for symbol, signal in signals.items():
            try:
                signal_type = signal['signal']
                confidence = signal['confidence']
                
                # Check minimum confidence threshold
                min_confidence = self.config.trading_config.get('min_signal_confidence', 0.65)
                if confidence < min_confidence:
                    self.logger.debug(f"⏭️ Skipping {symbol}: confidence {confidence:.1%} < {min_confidence:.1%}")
                    continue
                
                # Check if we can open a position
                can_open, reason = self.portfolio_manager.can_open_position(symbol, signal)
                
                if not can_open:
                    self.logger.info(f"🚫 Cannot open position for {symbol}: {reason}")
                    continue
                
                # Open position
                success, message, position = self.portfolio_manager.open_position(symbol, signal)
                
                if success:
                    # Start monitoring the trade
                    trade_id = self.trading_watcher.start_monitoring_trade(
                        position.position_id, symbol, signal
                    )
                    
                    executed_trades.append({
                        'symbol': symbol,
                        'trade_id': trade_id,
                        'signal_type': signal_type,
                        'entry_price': signal['entry_price'],
                        'confidence': confidence,
                        'position_size': position.quantity
                    })
                    
                    self.logger.info(f"✅ Opened {signal_type} position for {symbol}: "
                                   f"${signal['entry_price']:.4f} x {position.quantity:.6f}")
                else:
                    self.logger.warning(f"⚠️ Failed to open position for {symbol}: {message}")
                
            except Exception as e:
                self.logger.error(f"❌ Error executing trade for {symbol}: {e}")
        
        if executed_trades:
            self.logger.info(f"🎉 Executed {len(executed_trades)} trades this cycle")
        
        return executed_trades
    
    async def update_positions(self, market_data: Dict):
        """Update existing positions with current market data"""
        if not self.portfolio_manager.positions:
            return
        
        # Extract current prices
        current_prices = {}
        for symbol, data in market_data.items():
            if 'real_time' in data and data['real_time']:
                current_prices[symbol] = data['real_time']['price']
        
        # Update portfolio positions
        self.portfolio_manager.update_positions(current_prices)
        
        # Update trading watcher
        self.trading_watcher.update_trade_prices(current_prices)
        
        # Check for completed trades
        completed_trades = []
        for symbol in list(self.portfolio_manager.positions.keys()):
            position = self.portfolio_manager.positions.get(symbol)
            if not position:  # Position might have been closed
                continue
            
            # Check if position was closed by portfolio manager
            if symbol not in self.portfolio_manager.positions:
                # Find the corresponding trade in watcher and complete it
                for trade_id, trade_info in self.trading_watcher.active_trades.items():
                    if trade_info['symbol'] == symbol:
                        try:
                            current_price = current_prices.get(symbol, position.current_price)
                            trade_result = self.trading_watcher.complete_trade(
                                trade_id, current_price, 'auto_exit', 
                                position.quantity, commission=0.001 * position.quantity * current_price
                            )
                            completed_trades.append(trade_result)
                        except Exception as e:
                            self.logger.error(f"Error completing trade monitoring for {symbol}: {e}")
                        break
        
        if completed_trades:
            self.logger.info(f"📊 Completed monitoring for {len(completed_trades)} trades")
    
    async def trading_cycle(self):
        """Execute one complete trading cycle"""
        cycle_start = datetime.now()
        self.cycle_count += 1
        
        self.logger.info(f"🔄 Starting trading cycle #{self.cycle_count}")
        
        try:
            # 1. Collect market data
            market_data = await self.collect_market_data()
            
            if not market_data:
                self.logger.warning("⚠️ No market data available, skipping cycle")
                return
            
            # 2. Update existing positions
            await self.update_positions(market_data)
            
            # 3. Generate trading signals
            signals = await self.generate_trading_signals(market_data)
            
            # 4. Execute new trades
            if signals:
                executed_trades = await self.execute_trades(signals)
            
            # 5. Log portfolio status
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            self.logger.info(f"💰 Portfolio: ${portfolio_summary['total_equity']:.2f} "
                           f"({portfolio_summary['total_return_percent']:.2f}%), "
                           f"{portfolio_summary['open_positions_count']} positions")
            
            # 6. Performance monitoring
            if self.cycle_count % 10 == 0:  # Every 10 cycles
                performance_report = self.trading_watcher.get_performance_report(days=7)
                if 'summary' in performance_report:
                    summary = performance_report['summary']
                    self.logger.info(f"📊 7-day performance: {summary['total_trades']} trades, "
                                   f"{summary.get('win_rate_percent', 0):.1f}% win rate")
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"✅ Cycle #{self.cycle_count} completed in {cycle_duration:.1f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle #{self.cycle_count}: {e}")
            raise
    
    async def run_trading_loop(self):
        """Main trading loop"""
        self.logger.info("🎯 Starting main trading loop...")
        
        cycle_interval = self.config.trading_config.get('cycle_interval_minutes', 5)
        
        while self.running:
            try:
                # Execute trading cycle
                await self.trading_cycle()
                
                # Wait for next cycle
                self.logger.debug(f"😴 Waiting {cycle_interval} minutes until next cycle...")
                await asyncio.sleep(cycle_interval * 60)
                
            except asyncio.CancelledError:
                self.logger.info("🛑 Trading loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in trading loop: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
    
    async def start(self):
        """Start the trading platform"""
        try:
            self.start_time = datetime.now()
            self.running = True
            
            self.logger.info("🚀 Starting Automated Trading Platform...")
            self.logger.info(f"📅 Start time: {self.start_time}")
            self.logger.info(f"💰 Initial balance: ${self.config.trading_config['initial_balance']:,.2f}")
            self.logger.info(f"👀 Watchlist: {self.config.trading_config['watchlist']}")
            
            # Initialize ML models
            training_results = await self.initialize_models()
            
            # Start trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in trading platform: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the platform"""
        self.logger.info("🛑 Shutting down trading platform...")
        
        self.running = False
        
        try:
            # Close all open positions
            if self.portfolio_manager.positions:
                self.logger.info("💼 Closing all open positions...")
                
                # Get current prices for position closure
                current_prices = {}
                for symbol in self.portfolio_manager.positions.keys():
                    try:
                        price_data = await self.data_collector.get_real_time_data(symbol)
                        if price_data:
                            current_prices[symbol] = price_data['price']
                    except Exception as e:
                        self.logger.error(f"Error getting closing price for {symbol}: {e}")
                
                self.portfolio_manager.close_all_positions(current_prices, 'shutdown')
            
            # Save final portfolio state
            final_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Export final results
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            portfolio_file = results_dir / f"portfolio_final_{timestamp}.json"
            
            self.portfolio_manager.export_positions_to_json(str(portfolio_file))
            
            # Generate final performance report
            final_report = self.trading_watcher.get_performance_report(days=30)
            
            report_file = results_dir / f"performance_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            # Log final statistics
            runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            self.logger.info("📊 Final Performance Summary:")
            self.logger.info(f"⏱️  Runtime: {runtime}")
            self.logger.info(f"🔄 Cycles completed: {self.cycle_count}")
            self.logger.info(f"💰 Final equity: ${final_summary['total_equity']:.2f}")
            self.logger.info(f"📈 Total return: {final_summary['total_return_percent']:.2f}%")
            self.logger.info(f"🎯 Total trades: {final_summary['total_trades']}")
            
            if final_summary['total_trades'] > 0:
                self.logger.info(f"🏆 Win rate: {final_summary['win_rate_percent']:.1f}%")
            
            # Close data collector connections
            if hasattr(self.data_collector, 'close'):
                self.data_collector.close()
            
            self.logger.info("✅ Platform shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, initiating shutdown...")
        self.running = False

async def main():
    """Main entry point"""
    platform = None
    
    try:
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize platform
        platform = TradingPlatform()
        
        # Start trading
        await platform.start()
        
    except KeyboardInterrupt:
        logging.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        raise
    finally:
        if platform:
            await platform.shutdown()

if __name__ == "__main__":
    print("🤖 Advanced Automated Trading Platform")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)