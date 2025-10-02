"""
Advanced Trading Watcher and Performance Monitoring System
Tracks trades, analyzes performance, and provides learning feedback
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import json
import sqlite3
from pathlib import Path

@dataclass
class TradeResult:
    """
    Comprehensive trade result structure
    """
    trade_id: str
    symbol: str
    signal_type: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    profit_loss: float
    profit_loss_percent: float
    commission: float
    net_profit_loss: float
    duration_minutes: int
    exit_reason: str
    entry_confidence: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    signal_quality_score: Optional[float] = None

@dataclass
class SignalPerformance:
    """
    Performance metrics for signal quality
    """
    indicator_name: str
    total_signals: int
    correct_signals: int
    accuracy_rate: float
    avg_profit_on_correct: float
    avg_loss_on_incorrect: float
    signal_strength_distribution: Dict[str, int]
    best_timeframes: List[str]
    worst_timeframes: List[str]

class DatabaseManager:
    """
    Database manager for trade and performance data
    """
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Trade results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_results (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    profit_loss REAL NOT NULL,
                    profit_loss_percent REAL NOT NULL,
                    commission REAL NOT NULL,
                    net_profit_loss REAL NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    exit_reason TEXT NOT NULL,
                    entry_confidence REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    signal_quality_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Signal performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    indicator_name TEXT NOT NULL,
                    signal_strength INTEGER NOT NULL,
                    was_correct INTEGER NOT NULL,
                    contribution_score REAL,
                    timeframe TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trade_results (trade_id)
                )
            """)
            
            # Daily performance summary table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    total_commission REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_win REAL DEFAULT 0,
                    avg_loss REAL DEFAULT 0,
                    largest_win REAL DEFAULT 0,
                    largest_loss REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    portfolio_value REAL DEFAULT 0,
                    drawdown_percent REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_trade_result(self, trade: TradeResult):
        """Save trade result to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trade_results VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                trade.trade_id, trade.symbol, trade.signal_type,
                trade.entry_price, trade.exit_price, trade.quantity,
                trade.entry_time.isoformat(), trade.exit_time.isoformat(),
                trade.profit_loss, trade.profit_loss_percent, trade.commission,
                trade.net_profit_loss, trade.duration_minutes, trade.exit_reason,
                trade.entry_confidence, trade.stop_loss, trade.take_profit,
                trade.max_favorable_excursion, trade.max_adverse_excursion,
                trade.signal_quality_score
            ))
            conn.commit()
    
    def get_trade_history(self, days: int = 30) -> List[TradeResult]:
        """Get trade history for specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM trade_results 
                WHERE date(entry_time) >= date('now', '-{} days')
                ORDER BY entry_time DESC
            """.format(days))
            
            trades = []
            for row in cursor.fetchall():
                trade = TradeResult(
                    trade_id=row[0], symbol=row[1], signal_type=row[2],
                    entry_price=row[3], exit_price=row[4], quantity=row[5],
                    entry_time=datetime.fromisoformat(row[6]),
                    exit_time=datetime.fromisoformat(row[7]),
                    profit_loss=row[8], profit_loss_percent=row[9],
                    commission=row[10], net_profit_loss=row[11],
                    duration_minutes=row[12], exit_reason=row[13],
                    entry_confidence=row[14], stop_loss=row[15],
                    take_profit=row[16], max_favorable_excursion=row[17],
                    max_adverse_excursion=row[18], signal_quality_score=row[19]
                )
                trades.append(trade)
            
            return trades
    
    def save_signal_performance(self, trade_id: str, indicator_name: str, 
                              signal_strength: int, was_correct: bool, 
                              contribution_score: float = None, timeframe: str = None):
        """Save signal performance data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO signal_performance 
                (trade_id, indicator_name, signal_strength, was_correct, contribution_score, timeframe)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (trade_id, indicator_name, signal_strength, int(was_correct), 
                  contribution_score, timeframe))
            conn.commit()

class TradingWatcher:
    """
    Comprehensive trading performance watcher and analyzer
    """
    
    def __init__(self, ml_models, config: Dict):
        self.ml_models = ml_models
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database manager
        self.db_manager = DatabaseManager(config.get('database_path', 'trading_data.db'))
        
        # Active monitoring
        self.active_trades: Dict[str, Dict] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        
        # Performance metrics
        self.signal_accuracy: Dict[str, Dict] = {}
        self.indicator_performance: Dict[str, SignalPerformance] = {}
        
        # Learning parameters
        self.learning_enabled = config.get('enable_learning', True)
        self.retrain_threshold = config.get('retrain_threshold', 50)  # Retrain after 50 trades
        
    def start_monitoring_trade(self, trade_id: str, symbol: str, signal_data: Dict) -> str:
        """
        Start monitoring a new trade
        """
        self.active_trades[trade_id] = {
            'symbol': symbol,
            'signal_data': signal_data,
            'entry_time': datetime.now(),
            'price_history': [],
            'max_favorable_excursion': 0.0,
            'max_adverse_excursion': 0.0,
            'status': 'ACTIVE'
        }
        
        # Initialize price history for this symbol if not exists
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.logger.info(f"Started monitoring trade {trade_id} for {symbol}")
        return trade_id
    
    def update_trade_prices(self, price_updates: Dict[str, float]):
        """
        Update prices for all active trades and track excursions
        """
        for trade_id, trade_info in self.active_trades.items():
            symbol = trade_info['symbol']
            
            if symbol in price_updates:
                current_price = price_updates[symbol]
                signal_data = trade_info['signal_data']
                entry_price = signal_data['entry_price']
                signal_type = signal_data['signal']
                
                # Calculate current P&L
                if signal_type == 'BUY':
                    current_pnl = (current_price - entry_price) / entry_price
                else:  # SELL
                    current_pnl = (entry_price - current_price) / entry_price
                
                # Update max favorable/adverse excursion
                if current_pnl > trade_info['max_favorable_excursion']:
                    trade_info['max_favorable_excursion'] = current_pnl
                
                if current_pnl < 0 and abs(current_pnl) > trade_info['max_adverse_excursion']:
                    trade_info['max_adverse_excursion'] = abs(current_pnl)
                
                # Store price point
                price_point = {
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'pnl_percent': current_pnl * 100
                }
                trade_info['price_history'].append(price_point)
                
                # Limit price history to last 1000 points per trade
                if len(trade_info['price_history']) > 1000:
                    trade_info['price_history'] = trade_info['price_history'][-1000:]
    
    def complete_trade(self, trade_id: str, exit_price: float, exit_reason: str, 
                      quantity: float, commission: float = 0.0) -> TradeResult:
        """
        Complete a trade and analyze performance
        """
        if trade_id not in self.active_trades:
            raise ValueError(f"Trade {trade_id} not found in active trades")
        
        trade_info = self.active_trades[trade_id]
        signal_data = trade_info['signal_data']
        
        # Calculate trade results
        entry_price = signal_data['entry_price']
        entry_time = trade_info['entry_time']
        exit_time = datetime.now()
        
        # P&L calculation
        if signal_data['signal'] == 'BUY':
            profit_loss = (exit_price - entry_price) * quantity
        else:  # SELL
            profit_loss = (entry_price - exit_price) * quantity
        
        profit_loss_percent = (profit_loss / (entry_price * quantity)) * 100
        net_profit_loss = profit_loss - commission
        duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
        
        # Create trade result
        trade_result = TradeResult(
            trade_id=trade_id,
            symbol=trade_info['symbol'],
            signal_type=signal_data['signal'],
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_time=entry_time,
            exit_time=exit_time,
            profit_loss=profit_loss,
            profit_loss_percent=profit_loss_percent,
            commission=commission,
            net_profit_loss=net_profit_loss,
            duration_minutes=duration_minutes,
            exit_reason=exit_reason,
            entry_confidence=signal_data.get('confidence'),
            stop_loss=signal_data.get('stop_loss'),
            take_profit=signal_data.get('take_profit'),
            max_favorable_excursion=trade_info['max_favorable_excursion'] * 100,
            max_adverse_excursion=trade_info['max_adverse_excursion'] * 100,
            signal_quality_score=self._calculate_signal_quality_score(signal_data, trade_result)
        )
        
        # Save to database
        self.db_manager.save_trade_result(trade_result)
        
        # Analyze signal performance
        self._analyze_signal_performance(trade_result, signal_data)
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        # Check if we should retrain models
        if self.learning_enabled:
            self._check_retrain_trigger(trade_result)
        
        self.logger.info(f"Completed trade analysis for {trade_id}: "
                        f"{profit_loss_percent:.2f}% P&L, {exit_reason}")
        
        return trade_result
    
    def _calculate_signal_quality_score(self, signal_data: Dict, trade_result: TradeResult) -> float:
        """
        Calculate overall signal quality score (0-10 scale)
        """
        score = 0.0
        
        # Confidence score (0-3 points)
        confidence = signal_data.get('confidence', 0.5)
        score += confidence * 3
        
        # Risk/Reward ratio score (0-2 points)
        rr_ratio = signal_data.get('risk_reward_ratio', 1.0)
        if rr_ratio >= 2.0:
            score += 2.0
        elif rr_ratio >= 1.5:
            score += 1.5
        elif rr_ratio >= 1.0:
            score += 1.0
        
        # Outcome score (0-3 points)
        if trade_result.profit_loss_percent > 0:
            score += min(3.0, trade_result.profit_loss_percent / 2)  # Cap at 3 points
        
        # Duration efficiency score (0-2 points)
        if trade_result.duration_minutes <= 60:  # Quick profit
            score += 2.0
        elif trade_result.duration_minutes <= 240:  # 4 hours
            score += 1.5
        elif trade_result.duration_minutes <= 1440:  # 1 day
            score += 1.0
        
        return min(10.0, score)  # Cap at 10
    
    def _analyze_signal_performance(self, trade_result: TradeResult, signal_data: Dict):
        """
        Analyze performance of individual indicators
        """
        individual_signals = signal_data.get('individual_signals', {})
        was_profitable = trade_result.profit_loss_percent > 0
        
        for indicator, signal_info in individual_signals.items():
            # Determine if the indicator was correct
            indicator_signal = signal_info['signal']
            was_correct = (
                (indicator_signal == 'BUY' and was_profitable) or
                (indicator_signal == 'SELL' and was_profitable) or
                (indicator_signal == 'HOLD' and abs(trade_result.profit_loss_percent) < 1)
            )
            
            # Calculate contribution score
            signal_strength = signal_info['strength']
            contribution_score = self._calculate_contribution_score(
                was_correct, signal_strength, trade_result.profit_loss_percent
            )
            
            # Save to database
            self.db_manager.save_signal_performance(
                trade_result.trade_id, indicator, signal_strength, 
                was_correct, contribution_score
            )
            
            # Update in-memory performance tracking
            self._update_indicator_performance(indicator, was_correct, 
                                             trade_result.profit_loss_percent, signal_strength)
    
    def _calculate_contribution_score(self, was_correct: bool, signal_strength: int, 
                                    pnl_percent: float) -> float:
        """
        Calculate how much an indicator contributed to the trade outcome
        """
        base_score = 1.0 if was_correct else -1.0
        strength_multiplier = signal_strength / 4.0  # Normalize strength (1-4 scale)
        outcome_multiplier = min(abs(pnl_percent) / 5.0, 2.0)  # Cap at 2x for 5%+ moves
        
        return base_score * strength_multiplier * outcome_multiplier
    
    def _update_indicator_performance(self, indicator: str, was_correct: bool, 
                                    pnl_percent: float, signal_strength: int):
        """
        Update indicator performance metrics
        """
        if indicator not in self.signal_accuracy:
            self.signal_accuracy[indicator] = {
                'total_signals': 0,
                'correct_signals': 0,
                'total_pnl': 0.0,
                'strength_distribution': {1: 0, 2: 0, 3: 0, 4: 0}
            }
        
        stats = self.signal_accuracy[indicator]
        stats['total_signals'] += 1
        stats['total_pnl'] += pnl_percent
        stats['strength_distribution'][signal_strength] += 1
        
        if was_correct:
            stats['correct_signals'] += 1
    
    def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        trades = self.db_manager.get_trade_history(days)
        
        if not trades:
            return {'message': 'No completed trades in the specified period'}
        
        # Basic statistics
        total_trades = len(trades)
        profitable_trades = [t for t in trades if t.profit_loss_percent > 0]
        losing_trades = [t for t in trades if t.profit_loss_percent <= 0]
        
        win_rate = len(profitable_trades) / total_trades * 100
        
        # P&L statistics
        total_pnl = sum(t.net_profit_loss for t in trades)
        total_gross_pnl = sum(t.profit_loss for t in trades)
        total_commission = sum(t.commission for t in trades)
        
        avg_win = np.mean([t.profit_loss_percent for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t.profit_loss_percent for t in losing_trades]) if losing_trades else 0
        
        largest_win = max((t.profit_loss_percent for t in trades), default=0)
        largest_loss = min((t.profit_loss_percent for t in trades), default=0)
        
        # Profit factor
        gross_profit = sum(t.profit_loss for t in profitable_trades)
        gross_loss = abs(sum(t.profit_loss for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Duration analysis
        avg_trade_duration = np.mean([t.duration_minutes for t in trades])
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'avg_pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['avg_pnl'] += trade.profit_loss_percent
        
        for reason in exit_reasons:
            exit_reasons[reason]['avg_pnl'] /= exit_reasons[reason]['count']
        
        # Symbol performance
        symbol_performance = {}
        for trade in trades:
            symbol = trade.symbol
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {
                    'trades': 0, 'wins': 0, 'total_pnl': 0, 'avg_duration': 0
                }
            
            stats = symbol_performance[symbol]
            stats['trades'] += 1
            stats['total_pnl'] += trade.profit_loss_percent
            stats['avg_duration'] += trade.duration_minutes
            
            if trade.profit_loss_percent > 0:
                stats['wins'] += 1
        
        for symbol in symbol_performance:
            stats = symbol_performance[symbol]
            stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
            stats['avg_duration'] /= stats['trades']
        
        # Risk analysis
        consecutive_losses = self._calculate_consecutive_losses(trades)
        max_drawdown = self._calculate_max_drawdown(trades)
        
        return {
            'period_days': days,
            'summary': {
                'total_trades': total_trades,
                'profitable_trades': len(profitable_trades),
                'losing_trades': len(losing_trades),
                'win_rate_percent': win_rate,
                'total_pnl': total_pnl,
                'total_gross_pnl': total_gross_pnl,
                'total_commission': total_commission,
                'avg_trade_duration_minutes': avg_trade_duration
            },
            'profitability': {
                'average_win_percent': avg_win,
                'average_loss_percent': avg_loss,
                'largest_win_percent': largest_win,
                'largest_loss_percent': largest_loss,
                'profit_factor': profit_factor,
                'expectancy': (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
            },
            'risk_metrics': {
                'max_consecutive_losses': consecutive_losses,
                'max_drawdown_percent': max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(trades),
                'recovery_factor': total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0
            },
            'exit_analysis': exit_reasons,
            'symbol_performance': symbol_performance,
            'indicator_performance': self._get_indicator_performance_summary(),
            'recent_trades': [
                {
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'signal_type': t.signal_type,
                    'pnl_percent': t.profit_loss_percent,
                    'duration_minutes': t.duration_minutes,
                    'exit_reason': t.exit_reason,
                    'entry_time': t.entry_time.isoformat(),
                    'signal_quality_score': t.signal_quality_score
                }
                for t in trades[:10]  # Last 10 trades
            ]
        }
    
    def _calculate_consecutive_losses(self, trades: List[TradeResult]) -> int:
        """Calculate maximum consecutive losing trades"""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        # Sort by entry time
        sorted_trades = sorted(trades, key=lambda t: t.entry_time)
        
        for trade in sorted_trades:
            if trade.profit_loss_percent <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_drawdown(self, trades: List[TradeResult]) -> float:
        """Calculate maximum drawdown percentage"""
        if not trades:
            return 0.0
        
        # Sort by entry time
        sorted_trades = sorted(trades, key=lambda t: t.entry_time)
        
        # Calculate cumulative P&L
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        
        for trade in sorted_trades:
            cumulative_pnl += trade.profit_loss_percent
            
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, trades: List[TradeResult]) -> float:
        """Calculate Sharpe ratio for the trading period"""
        if len(trades) < 2:
            return 0.0
        
        returns = [t.profit_loss_percent for t in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Assume risk-free rate of 3% annually, adjust for trade frequency
        risk_free_rate = 0.03 / 252  # Daily risk-free rate
        
        return (mean_return - risk_free_rate) / std_return if std_return > 0 else 0.0
    
    def _get_indicator_performance_summary(self) -> Dict[str, Dict]:
        """Get summary of indicator performance"""
        summary = {}
        
        for indicator, stats in self.signal_accuracy.items():
            accuracy = (stats['correct_signals'] / stats['total_signals']) * 100 if stats['total_signals'] > 0 else 0
            avg_pnl = stats['total_pnl'] / stats['total_signals'] if stats['total_signals'] > 0 else 0
            
            summary[indicator] = {
                'accuracy_percent': accuracy,
                'total_signals': stats['total_signals'],
                'average_pnl_percent': avg_pnl,
                'strength_distribution': stats['strength_distribution']
            }
        
        return summary
    
    def _check_retrain_trigger(self, trade_result: TradeResult):
        """Check if we should trigger model retraining"""
        recent_trades = self.db_manager.get_trade_history(days=30)
        
        if len(recent_trades) >= self.retrain_threshold:
            # Check if performance is declining
            recent_10 = recent_trades[:10]
            older_10 = recent_trades[10:20] if len(recent_trades) >= 20 else []
            
            if older_10:
                recent_win_rate = len([t for t in recent_10 if t.profit_loss_percent > 0]) / len(recent_10)
                older_win_rate = len([t for t in older_10 if t.profit_loss_percent > 0]) / len(older_10)
                
                # Trigger retraining if win rate declined by more than 10%
                if recent_win_rate < older_win_rate - 0.1:
                    self.logger.info("Performance decline detected, triggering model retraining")
                    self._retrain_models(recent_trades)
    
    def _retrain_models(self, recent_trades: List[TradeResult]):
        """Retrain ML models based on recent performance"""
        try:
            if self.ml_models and hasattr(self.ml_models, 'retrain_with_new_data'):
                # This would require implementing the retraining logic
                # For now, log the intention
                self.logger.info(f"Retraining models with data from {len(recent_trades)} recent trades")
                
                # Analyze which indicators performed best/worst
                best_indicators = self._identify_best_performing_indicators()
                worst_indicators = self._identify_worst_performing_indicators()
                
                self.logger.info(f"Best performing indicators: {best_indicators}")
                self.logger.info(f"Worst performing indicators: {worst_indicators}")
                
                # Could implement dynamic indicator weighting here
                
        except Exception as e:
            self.logger.error(f"Error during model retraining: {e}")
    
    def _identify_best_performing_indicators(self) -> List[str]:
        """Identify the best performing indicators"""
        performance_scores = []
        
        for indicator, stats in self.signal_accuracy.items():
            if stats['total_signals'] >= 5:  # Minimum sample size
                accuracy = stats['correct_signals'] / stats['total_signals']
                avg_pnl = stats['total_pnl'] / stats['total_signals']
                
                # Combined score: accuracy weighted by average P&L
                score = accuracy * (1 + max(0, avg_pnl / 100))
                performance_scores.append((indicator, score))
        
        # Sort by score and return top performers
        performance_scores.sort(key=lambda x: x[1], reverse=True)
        return [indicator for indicator, score in performance_scores[:5]]
    
    def _identify_worst_performing_indicators(self) -> List[str]:
        """Identify the worst performing indicators"""
        performance_scores = []
        
        for indicator, stats in self.signal_accuracy.items():
            if stats['total_signals'] >= 5:  # Minimum sample size
                accuracy = stats['correct_signals'] / stats['total_signals']
                avg_pnl = stats['total_pnl'] / stats['total_signals']
                
                # Combined score: accuracy weighted by average P&L
                score = accuracy * (1 + max(0, avg_pnl / 100))
                performance_scores.append((indicator, score))
        
        # Sort by score and return worst performers
        performance_scores.sort(key=lambda x: x[1])
        return [indicator for indicator, score in performance_scores[:3]]
    
    def get_real_time_monitoring_data(self) -> Dict[str, Any]:
        """Get real-time monitoring data for active trades"""
        return {
            'active_trades_count': len(self.active_trades),
            'active_trades': {
                trade_id: {
                    'symbol': info['symbol'],
                    'signal_type': info['signal_data']['signal'],
                    'entry_price': info['signal_data']['entry_price'],
                    'entry_time': info['entry_time'].isoformat(),
                    'max_favorable_excursion': info['max_favorable_excursion'],
                    'max_adverse_excursion': info['max_adverse_excursion'],
                    'price_points': len(info['price_history'])
                }
                for trade_id, info in self.active_trades.items()
            },
            'monitoring_status': 'active' if self.active_trades else 'idle'
        }