"""
Advanced Portfolio Manager and Risk Management System
Handles position sizing, risk control, and portfolio optimization
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json

class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"

@dataclass
class Position:
    """
    Comprehensive position structure
    """
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: PositionType
    status: PositionStatus
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_id: Optional[str] = None
    entry_signal_confidence: Optional[float] = None
    max_risk_amount: Optional[float] = None
    
    def __post_init__(self):
        if self.position_id is None:
            self.position_id = f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        entry_value = self.entry_price * self.quantity
        return (self.unrealized_pnl / entry_value) * 100
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable"""
        return self.unrealized_pnl > 0

class RiskManager:
    """
    Advanced risk management system
    """
    
    def __init__(self, config: Dict):
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.08)  # 8%
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.6)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)
        self.max_single_position = config.get('max_single_position', 0.15)  # 15%
        self.stop_loss_buffer = config.get('stop_loss_buffer', 1.1)  # 10% buffer
        
        # Correlation matrix for major assets (simplified)
        self.correlation_matrix = {
            'BTC': {'ETH': 0.8, 'DOGE': 0.6, 'SHIB': 0.5},
            'ETH': {'BTC': 0.8, 'ADA': 0.7, 'SOL': 0.75},
            'DOGE': {'SHIB': 0.9, 'PEPE': 0.8, 'FLOKI': 0.7},
            # Add more correlations as needed
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, signal_data: Dict, account_balance: float) -> Dict[str, Any]:
        """
        Calculate optimal position size using multiple methods
        """
        entry_price = signal_data['entry_price']
        stop_loss = signal_data.get('stop_loss')
        confidence = signal_data.get('confidence', 0.5)
        
        # Method 1: Fixed fractional method
        fixed_risk_size = self._calculate_fixed_risk_size(account_balance, entry_price, stop_loss)
        
        # Method 2: Kelly Criterion (simplified)
        kelly_size = self._calculate_kelly_size(confidence, signal_data.get('risk_reward_ratio', 2.0), account_balance, entry_price)
        
        # Method 3: Volatility-based sizing
        volatility_size = self._calculate_volatility_size(signal_data, account_balance, entry_price)
        
        # Use the most conservative size
        conservative_size = min(fixed_risk_size, kelly_size, volatility_size)
        
        # Apply maximum position size limit
        max_position_value = account_balance * self.max_single_position
        max_position_size = max_position_value / entry_price
        
        final_size = min(conservative_size, max_position_size)
        
        return {
            'position_size': final_size,
            'position_value': final_size * entry_price,
            'risk_amount': abs(entry_price - stop_loss) * final_size if stop_loss else final_size * entry_price * self.max_risk_per_trade,
            'methods': {
                'fixed_risk': fixed_risk_size,
                'kelly': kelly_size,
                'volatility': volatility_size,
                'final': final_size
            }
        }
    
    def _calculate_fixed_risk_size(self, balance: float, entry_price: float, stop_loss: Optional[float]) -> float:
        """Fixed risk percentage method"""
        if not stop_loss:
            return (balance * self.max_risk_per_trade) / entry_price
        
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = balance * self.max_risk_per_trade
        
        return max_risk_amount / risk_per_share if risk_per_share > 0 else 0
    
    def _calculate_kelly_size(self, confidence: float, risk_reward_ratio: float, balance: float, entry_price: float) -> float:
        """Simplified Kelly Criterion"""
        # Kelly formula: f = (bp - q) / b
        # where b = odds (risk_reward_ratio), p = probability of win (confidence), q = 1-p
        
        win_prob = confidence
        loss_prob = 1 - confidence
        
        if risk_reward_ratio <= 0:
            return 0
        
        kelly_fraction = (win_prob * risk_reward_ratio - loss_prob) / risk_reward_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% and floor at 0
        
        # Scale down Kelly for safety (typically use 25-50% of Kelly)
        safe_kelly_fraction = kelly_fraction * 0.25
        
        return (balance * safe_kelly_fraction) / entry_price
    
    def _calculate_volatility_size(self, signal_data: Dict, balance: float, entry_price: float) -> float:
        """Volatility-based position sizing"""
        atr = signal_data.get('atr', entry_price * 0.02)  # Default 2% ATR
        
        # Inverse volatility sizing - lower volatility allows larger positions
        volatility_factor = min(0.02 / (atr / entry_price), 2.0)  # Cap at 2x
        base_size = (balance * self.max_risk_per_trade) / entry_price
        
        return base_size * volatility_factor
    
    def check_correlation_risk(self, new_symbol: str, existing_positions: Dict[str, Position]) -> bool:
        """
        Check if adding new position would create excessive correlation risk
        """
        if not existing_positions:
            return True
        
        # Get base symbol (remove trading pairs)
        new_base = new_symbol.split('/')[0] if '/' in new_symbol else new_symbol
        
        total_correlated_exposure = 0
        
        for pos in existing_positions.values():
            existing_base = pos.symbol.split('/')[0] if '/' in pos.symbol else pos.symbol
            
            # Check correlation
            correlation = self._get_correlation(new_base, existing_base)
            if correlation > 0.5:  # High correlation threshold
                total_correlated_exposure += pos.market_value
        
        # Check if total correlated exposure would exceed limit
        return total_correlated_exposure <= self.max_correlation_exposure
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        if symbol1 == symbol2:
            return 1.0
        
        if symbol1 in self.correlation_matrix and symbol2 in self.correlation_matrix[symbol1]:
            return self.correlation_matrix[symbol1][symbol2]
        elif symbol2 in self.correlation_matrix and symbol1 in self.correlation_matrix[symbol2]:
            return self.correlation_matrix[symbol2][symbol1]
        
        return 0.0  # No correlation data available

class PortfolioManager:
    """
    Comprehensive portfolio management system
    """
    
    def __init__(self, initial_balance: float, config: Dict):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.reserved_balance = 0.0  # Reserved for open positions
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        self.risk_manager = RiskManager(config)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.daily_returns = []
        self.equity_curve = [initial_balance]
        self.max_equity = initial_balance
        self.max_drawdown = 0.0
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
    @property
    def available_balance(self) -> float:
        """Available balance for new trades"""
        return max(0, self.current_balance - self.reserved_balance)
    
    @property
    def total_equity(self) -> float:
        """Total portfolio equity"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.current_balance + unrealized_pnl
    
    @property
    def total_return_percent(self) -> float:
        """Total return percentage"""
        return ((self.total_equity - self.initial_balance) / self.initial_balance) * 100
    
    def can_open_position(self, symbol: str, signal_data: Dict) -> Tuple[bool, str]:
        """
        Check if we can open a new position
        """
        # Check if position already exists
        if symbol in self.positions:
            return False, f"Position already exists for {symbol}"
        
        # Check available balance
        position_calc = self.risk_manager.calculate_position_size(signal_data, self.available_balance)
        required_capital = position_calc['position_value']
        
        if required_capital > self.available_balance * 0.95:  # Keep 5% buffer
            return False, "Insufficient available balance"
        
        # Check correlation risk
        if not self.risk_manager.check_correlation_risk(symbol, self.positions):
            return False, "Would exceed correlation risk limits"
        
        # Check maximum number of positions
        max_positions = self.config.get('max_open_positions', 10)
        if len(self.positions) >= max_positions:
            return False, f"Maximum positions limit reached ({max_positions})"
        
        # Check portfolio risk
        if self._calculate_portfolio_risk() > self.risk_manager.max_portfolio_risk * 0.8:
            return False, "Portfolio risk too high"
        
        return True, "All checks passed"
    
    def open_position(self, symbol: str, signal_data: Dict) -> Tuple[bool, str, Optional[Position]]:
        """
        Open a new trading position
        """
        try:
            # Validate if we can open position
            can_open, reason = self.can_open_position(symbol, signal_data)
            if not can_open:
                return False, reason, None
            
            # Calculate position size
            position_calc = self.risk_manager.calculate_position_size(signal_data, self.available_balance)
            position_size = position_calc['position_size']
            
            if position_size <= 0:
                return False, "Invalid position size calculated", None
            
            # Create position
            position = Position(
                symbol=symbol,
                quantity=position_size,
                entry_price=signal_data['entry_price'],
                current_price=signal_data['current_price'],
                entry_time=datetime.now(),
                position_type=PositionType.LONG if signal_data['signal'] == 'BUY' else PositionType.SHORT,
                status=PositionStatus.OPEN,
                stop_loss=signal_data.get('stop_loss'),
                take_profit=signal_data.get('take_profit'),
                entry_signal_confidence=signal_data.get('confidence'),
                max_risk_amount=position_calc['risk_amount']
            )
            
            # Reserve capital
            required_capital = position.quantity * position.entry_price
            self.reserved_balance += required_capital
            
            # Add to positions
            self.positions[symbol] = position
            
            self.logger.info(f"Opened {position.position_type.value} position: {symbol} "
                           f"Qty: {position.quantity:.6f} @ ${position.entry_price:.6f}")
            
            return True, f"Position opened successfully", position
            
        except Exception as e:
            self.logger.error(f"Error opening position for {symbol}: {e}")
            return False, f"Error: {str(e)}", None
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'manual') -> Tuple[bool, str, Optional[Dict]]:
        """
        Close an existing position
        """
        if symbol not in self.positions:
            return False, f"No open position found for {symbol}", None
        
        try:
            position = self.positions[symbol]
            
            # Calculate final P&L
            if position.position_type == PositionType.LONG:
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # SHORT
                pnl = (position.entry_price - exit_price) * position.quantity
            
            pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100
            
            # Calculate commission (simplified - 0.1% per trade)
            commission = (position.entry_price * position.quantity * 0.001) + (exit_price * position.quantity * 0.001)
            net_pnl = pnl - commission
            
            # Update balance
            position_value = exit_price * position.quantity
            self.current_balance += position_value
            self.reserved_balance -= position.entry_price * position.quantity
            self.total_commission += commission
            
            # Update position
            position.current_price = exit_price
            position.status = PositionStatus.CLOSED
            
            # Record trade statistics
            self.total_trades += 1
            if net_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # Update equity curve
            self.equity_curve.append(self.total_equity)
            self._update_drawdown()
            
            trade_record = {
                'symbol': symbol,
                'position_type': position.position_type.value,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
                'pnl': net_pnl,
                'pnl_percent': pnl_percent,
                'commission': commission,
                'entry_time': position.entry_time.isoformat(),
                'exit_time': datetime.now().isoformat(),
                'duration_minutes': int((datetime.now() - position.entry_time).total_seconds() / 60),
                'exit_reason': reason,
                'entry_confidence': position.entry_signal_confidence
            }
            
            self.logger.info(f"Closed {position.position_type.value} position: {symbol} "
                           f"P&L: ${net_pnl:.2f} ({pnl_percent:.2f}%) - {reason}")
            
            return True, f"Position closed successfully", trade_record
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return False, f"Error: {str(e)}", None
    
    def update_positions(self, price_data: Dict[str, float]):
        """
        Update all positions with current prices and check exit conditions
        """
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]
                position.current_price = current_price
                
                # Check exit conditions
                exit_reason = self._check_exit_conditions(position, current_price)
                if exit_reason:
                    positions_to_close.append((symbol, current_price, exit_reason))
        
        # Close positions that meet exit criteria
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, reason)
    
    def _check_exit_conditions(self, position: Position, current_price: float) -> Optional[str]:
        """
        Check if position should be closed based on exit conditions
        """
        # Stop loss check
        if position.stop_loss:
            if position.position_type == PositionType.LONG and current_price <= position.stop_loss:
                return 'stop_loss'
            elif position.position_type == PositionType.SHORT and current_price >= position.stop_loss:
                return 'stop_loss'
        
        # Take profit check
        if position.take_profit:
            if position.position_type == PositionType.LONG and current_price >= position.take_profit:
                return 'take_profit'
            elif position.position_type == PositionType.SHORT and current_price <= position.take_profit:
                return 'take_profit'
        
        # Time-based exit (holding too long)
        holding_time = datetime.now() - position.entry_time
        max_holding_days = self.config.get('max_holding_days', 30)
        if holding_time.days >= max_holding_days:
            return 'max_holding_time'
        
        # Emergency exit for large losses
        if position.unrealized_pnl_percent < -10:  # 10% loss threshold
            return 'emergency_exit'
        
        return None
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk exposure"""
        if not self.positions:
            return 0.0
        
        total_risk = 0
        for position in self.positions.values():
            if position.max_risk_amount:
                total_risk += position.max_risk_amount
        
        return total_risk / self.total_equity if self.total_equity > 0 else 0.0
    
    def _update_drawdown(self):
        """Update maximum drawdown calculation"""
        current_equity = self.total_equity
        
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_drawdown = (self.max_equity - current_equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio summary
        """
        # Calculate unrealized P&L
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Calculate realized P&L
        total_realized_pnl = sum(
            (pos.current_price - pos.entry_price) * pos.quantity 
            for pos in self.closed_positions
        )
        
        # Win rate calculation
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Average trade metrics
        profitable_trades = [pos for pos in self.closed_positions if pos.unrealized_pnl > 0]
        losing_trades = [pos for pos in self.closed_positions if pos.unrealized_pnl <= 0]
        
        avg_win = np.mean([pos.unrealized_pnl for pos in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([pos.unrealized_pnl for pos in losing_trades]) if losing_trades else 0
        
        return {
            # Balance and equity
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'available_balance': self.available_balance,
            'reserved_balance': self.reserved_balance,
            'total_equity': self.total_equity,
            'total_return_percent': self.total_return_percent,
            
            # P&L
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_commission': self.total_commission,
            
            # Risk metrics
            'max_drawdown_percent': self.max_drawdown * 100,
            'current_portfolio_risk': self._calculate_portfolio_risk(),
            
            # Position metrics
            'open_positions_count': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_percent': win_rate,
            
            # Trade analysis
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            
            # Current positions
            'positions': {
                symbol: {
                    'position_type': pos.position_type.value,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_percent': pos.unrealized_pnl_percent,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'entry_time': pos.entry_time.isoformat(),
                    'days_held': (datetime.now() - pos.entry_time).days
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics
        """
        if len(self.equity_curve) < 2:
            return {'error': 'Insufficient data for risk calculations'}
        
        # Convert equity curve to returns
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) == 0:
            return {'error': 'No return data available'}
        
        # Basic statistics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        annualized_return = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (365 / len(equity_series)) - 1) * 100
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Sharpe ratio (assuming 3% risk-free rate)
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return / 100 - risk_free_rate) / (volatility / 100) if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns * 100, 5)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Consecutive losses
        consecutive_losses = self._calculate_consecutive_losses()
        
        return {
            'total_return_percent': total_return,
            'annualized_return_percent': annualized_return,
            'volatility_percent': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_percent': max_drawdown,
            'var_95_percent': var_95,
            'win_rate_percent': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'profit_factor': self._calculate_profit_factor(),
            'max_consecutive_losses': consecutive_losses,
            'current_drawdown_percent': (self.max_equity - self.total_equity) / self.max_equity * 100
        }
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        if not self.closed_positions:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for position in self.closed_positions:
            if position.unrealized_pnl <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.closed_positions:
            return 0.0
        
        gross_profit = sum(pos.unrealized_pnl for pos in self.closed_positions if pos.unrealized_pnl > 0)
        gross_loss = abs(sum(pos.unrealized_pnl for pos in self.closed_positions if pos.unrealized_pnl <= 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def export_positions_to_json(self, file_path: str):
        """Export current positions to JSON file"""
        positions_data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': self.get_portfolio_summary(),
            'positions': {
                symbol: asdict(position) for symbol, position in self.positions.items()
            }
        }
        
        # Convert datetime objects to strings for JSON serialization
        for symbol, pos_data in positions_data['positions'].items():
            pos_data['entry_time'] = pos_data['entry_time'].isoformat()
            pos_data['position_type'] = pos_data['position_type'].value
            pos_data['status'] = pos_data['status'].value
        
        with open(file_path, 'w') as f:
            json.dump(positions_data, f, indent=2, default=str)
        
        self.logger.info(f"Portfolio data exported to {file_path}")
    
    def close_all_positions(self, current_prices: Dict[str, float], reason: str = 'shutdown'):
        """Close all open positions"""
        positions_to_close = list(self.positions.keys())
        
        for symbol in positions_to_close:
            if symbol in current_prices:
                self.close_position(symbol, current_prices[symbol], reason)
            else:
                self.logger.warning(f"No current price available for {symbol}, using last known price")
                position = self.positions[symbol]
                self.close_position(symbol, position.current_price, f"{reason}_no_price")