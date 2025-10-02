"""
Advanced Trading Signal Generator
Combines technical analysis, machine learning, and market sentiment
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class TradingSignal:
    """
    Comprehensive trading signal structure
    """
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    reasoning: List[str]
    timestamp: datetime

class TradingSignalGenerator:
    """
    Advanced signal generator combining multiple analysis methods
    """
    
    def __init__(self, ml_models, config: Dict):
        self.ml_models = ml_models
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Signal weights for different indicators
        self.indicator_weights = {
            'rsi': 1.5,
            'macd': 1.3,
            'bollinger': 1.4,
            'moving_average': 1.0,
            'stochastic': 0.9,
            'williams_r': 0.8,
            'volume': 1.2,
            'ml_direction': 2.5,  # Higher weight for ML predictions
            'ml_price': 2.2,
            'support_resistance': 1.6,
            'momentum': 1.1,
            'volatility': 0.7
        }
    
    def calculate_rsi_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate RSI-based signals with multiple timeframes
        """
        signals = {}
        latest = df.iloc[-1]
        
        rsi = latest.get('rsi', 50)
        rsi_30 = latest.get('rsi_30', 50)  # Longer period RSI
        
        # Multi-timeframe RSI analysis
        if rsi < 25:  # Oversold
            if rsi_30 < 30:  # Confirmed by longer timeframe
                signals['rsi'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.VERY_STRONG,
                    'reasoning': f'RSI oversold: {rsi:.1f}, confirmed by RSI(30): {rsi_30:.1f}'
                }
            else:
                signals['rsi'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'RSI oversold: {rsi:.1f}'
                }
        elif rsi < 35:
            signals['rsi'] = {
                'signal': SignalType.BUY,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'RSI approaching oversold: {rsi:.1f}'
            }
        elif rsi > 75:  # Overbought
            if rsi_30 > 70:  # Confirmed by longer timeframe
                signals['rsi'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.VERY_STRONG,
                    'reasoning': f'RSI overbought: {rsi:.1f}, confirmed by RSI(30): {rsi_30:.1f}'
                }
            else:
                signals['rsi'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'RSI overbought: {rsi:.1f}'
                }
        elif rsi > 65:
            signals['rsi'] = {
                'signal': SignalType.SELL,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'RSI approaching overbought: {rsi:.1f}'
            }
        else:
            signals['rsi'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': f'RSI neutral: {rsi:.1f}'
            }
        
        return signals
    
    def calculate_macd_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Advanced MACD signal analysis
        """
        signals = {}
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_diff = latest.get('macd_diff', 0)
        
        prev_macd_diff = previous.get('macd_diff', 0)
        
        # MACD crossover analysis
        if macd_diff > 0 and prev_macd_diff <= 0:  # Bullish crossover
            if macd < 0:  # Crossover below zero line (stronger signal)
                signals['macd'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.STRONG,
                    'reasoning': 'MACD bullish crossover below zero line'
                }
            else:
                signals['macd'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': 'MACD bullish crossover above zero line'
                }
        elif macd_diff < 0 and prev_macd_diff >= 0:  # Bearish crossover
            if macd > 0:  # Crossover above zero line (stronger signal)
                signals['macd'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.STRONG,
                    'reasoning': 'MACD bearish crossover above zero line'
                }
            else:
                signals['macd'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': 'MACD bearish crossover below zero line'
                }
        elif macd > macd_signal and macd_diff > 0:  # Bullish momentum
            signals['macd'] = {
                'signal': SignalType.BUY,
                'strength': SignalStrength.WEAK,
                'reasoning': 'MACD showing bullish momentum'
            }
        elif macd < macd_signal and macd_diff < 0:  # Bearish momentum
            signals['macd'] = {
                'signal': SignalType.SELL,
                'strength': SignalStrength.WEAK,
                'reasoning': 'MACD showing bearish momentum'
            }
        else:
            signals['macd'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': 'MACD neutral'
            }
        
        return signals
    
    def calculate_bollinger_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Bollinger Bands analysis with squeeze detection
        """
        signals = {}
        latest = df.iloc[-1]
        
        price = latest.get('close', 0)
        bb_upper = latest.get('bb_high', 0)
        bb_lower = latest.get('bb_low', 0)
        bb_middle = latest.get('bb_mid', 0)
        bb_width = latest.get('bb_width', 0)
        bb_position = latest.get('bb_position', 0.5)
        
        # Bollinger Band squeeze detection (low volatility)
        bb_width_sma = df['bb_width'].rolling(window=20).mean().iloc[-1] if 'bb_width' in df.columns else bb_width
        is_squeeze = bb_width < bb_width_sma * 0.8
        
        if price <= bb_lower:  # Price at lower band
            if is_squeeze:
                signals['bollinger'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.VERY_STRONG,
                    'reasoning': f'Price at lower BB with squeeze (low volatility): BB%={bb_position:.2f}'
                }
            else:
                signals['bollinger'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'Price at lower Bollinger Band: BB%={bb_position:.2f}'
                }
        elif price >= bb_upper:  # Price at upper band
            if is_squeeze:
                signals['bollinger'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.VERY_STRONG,
                    'reasoning': f'Price at upper BB with squeeze: BB%={bb_position:.2f}'
                }
            else:
                signals['bollinger'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'Price at upper Bollinger Band: BB%={bb_position:.2f}'
                }
        elif bb_position < 0.2:  # Near lower band
            signals['bollinger'] = {
                'signal': SignalType.BUY,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'Price near lower BB: BB%={bb_position:.2f}'
            }
        elif bb_position > 0.8:  # Near upper band
            signals['bollinger'] = {
                'signal': SignalType.SELL,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'Price near upper BB: BB%={bb_position:.2f}'
            }
        else:
            signals['bollinger'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': f'Price in middle of BB: BB%={bb_position:.2f}'
            }
        
        return signals
    
    def calculate_moving_average_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Multiple timeframe moving average analysis
        """
        signals = {}
        latest = df.iloc[-1]
        
        price = latest.get('close', 0)
        sma_5 = latest.get('sma_5', 0)
        sma_20 = latest.get('sma_20', 0)
        sma_50 = latest.get('sma_50', 0)
        ema_12 = latest.get('ema_12', 0)
        ema_26 = latest.get('ema_26', 0)
        
        # Golden Cross / Death Cross detection
        if sma_20 > sma_50 and price > sma_20 > sma_50:
            if ema_12 > ema_26:  # Confirmed by EMA
                signals['moving_average'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.STRONG,
                    'reasoning': 'Golden cross confirmed: SMA20 > SMA50, price above MAs'
                }
            else:
                signals['moving_average'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': 'Bullish MA alignment: price > SMA20 > SMA50'
                }
        elif sma_20 < sma_50 and price < sma_20 < sma_50:
            if ema_12 < ema_26:  # Confirmed by EMA
                signals['moving_average'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.STRONG,
                    'reasoning': 'Death cross confirmed: SMA20 < SMA50, price below MAs'
                }
            else:
                signals['moving_average'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': 'Bearish MA alignment: price < SMA20 < SMA50'
                }
        elif price > sma_5 > sma_20:  # Short-term bullish
            signals['moving_average'] = {
                'signal': SignalType.BUY,
                'strength': SignalStrength.WEAK,
                'reasoning': 'Short-term bullish: price > SMA5 > SMA20'
            }
        elif price < sma_5 < sma_20:  # Short-term bearish
            signals['moving_average'] = {
                'signal': SignalType.SELL,
                'strength': SignalStrength.WEAK,
                'reasoning': 'Short-term bearish: price < SMA5 < SMA20'
            }
        else:
            signals['moving_average'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': 'Mixed moving average signals'
            }
        
        return signals
    
    def calculate_volume_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Volume analysis with accumulation/distribution
        """
        signals = {}
        latest = df.iloc[-1]
        
        volume = latest.get('volume', 0)
        volume_sma = latest.get('volume_sma', volume)
        obv = latest.get('obv', 0)
        ad = latest.get('ad', 0)
        cmf = latest.get('cmf', 0)
        
        volume_ratio = volume / volume_sma if volume_sma > 0 else 1
        
        # Volume analysis
        if volume_ratio > 2.5:  # Very high volume
            if cmf > 0.1:  # Money flowing in
                signals['volume'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'High volume with money inflow: Vol ratio={volume_ratio:.1f}, CMF={cmf:.2f}'
                }
            elif cmf < -0.1:  # Money flowing out
                signals['volume'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'High volume with money outflow: Vol ratio={volume_ratio:.1f}, CMF={cmf:.2f}'
                }
            else:
                signals['volume'] = {
                    'signal': SignalType.HOLD,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': f'High volume but mixed money flow: Vol ratio={volume_ratio:.1f}'
                }
        elif volume_ratio > 1.5:  # Moderate high volume
            if cmf > 0.05:
                signals['volume'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': f'Above average volume with buying pressure: Vol ratio={volume_ratio:.1f}'
                }
            elif cmf < -0.05:
                signals['volume'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': f'Above average volume with selling pressure: Vol ratio={volume_ratio:.1f}'
                }
            else:
                signals['volume'] = {
                    'signal': SignalType.HOLD,
                    'strength': SignalStrength.WEAK,
                    'reasoning': f'Above average volume: Vol ratio={volume_ratio:.1f}'
                }
        elif volume_ratio < 0.5:  # Low volume
            signals['volume'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': f'Low volume - wait for confirmation: Vol ratio={volume_ratio:.1f}'
            }
        else:
            signals['volume'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': f'Normal volume: Vol ratio={volume_ratio:.1f}'
            }
        
        return signals
    
    def calculate_support_resistance_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Support and resistance level analysis
        """
        signals = {}
        latest = df.iloc[-1]
        
        price = latest.get('close', 0)
        support_20 = latest.get('support_20', 0)
        resistance_20 = latest.get('resistance_20', 0)
        dist_from_support = latest.get('dist_from_support', 0)
        dist_from_resistance = latest.get('dist_from_resistance', 0)
        
        # Support/Resistance analysis
        if dist_from_support <= 0.01:  # Very close to support (within 1%)
            signals['support_resistance'] = {
                'signal': SignalType.BUY,
                'strength': SignalStrength.STRONG,
                'reasoning': f'Price near strong support: {dist_from_support*100:.1f}% from support'
            }
        elif dist_from_support <= 0.02:  # Close to support
            signals['support_resistance'] = {
                'signal': SignalType.BUY,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'Price approaching support: {dist_from_support*100:.1f}% from support'
            }
        elif dist_from_resistance <= 0.01:  # Very close to resistance
            signals['support_resistance'] = {
                'signal': SignalType.SELL,
                'strength': SignalStrength.STRONG,
                'reasoning': f'Price near strong resistance: {dist_from_resistance*100:.1f}% from resistance'
            }
        elif dist_from_resistance <= 0.02:  # Close to resistance
            signals['support_resistance'] = {
                'signal': SignalType.SELL,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'Price approaching resistance: {dist_from_resistance*100:.1f}% from resistance'
            }
        else:
            signals['support_resistance'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': 'Price in middle range between support and resistance'
            }
        
        return signals
    
    def calculate_ml_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Machine learning based signals
        """
        signals = {}
        
        try:
            # Direction prediction
            direction_pred = self.ml_models.predict_direction(df)
            
            if direction_pred['direction'] == 'UP':
                strength = self._confidence_to_strength(direction_pred['confidence'])
                signals['ml_direction'] = {
                    'signal': SignalType.BUY,
                    'strength': strength,
                    'reasoning': f'ML predicts UP with {direction_pred["confidence"]:.1%} confidence'
                }
            else:
                strength = self._confidence_to_strength(direction_pred['confidence'])
                signals['ml_direction'] = {
                    'signal': SignalType.SELL,
                    'strength': strength,
                    'reasoning': f'ML predicts DOWN with {direction_pred["confidence"]:.1%} confidence'
                }
            
            # Price prediction
            if 'lstm' in self.ml_models.models:
                price_pred = self.ml_models.predict_price_lstm(df)
                price_change = price_pred['price_change_percent']
                
                if price_change > 3:  # Expected rise > 3%
                    signals['ml_price'] = {
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.STRONG,
                        'reasoning': f'LSTM predicts {price_change:.1f}% price increase'
                    }
                elif price_change > 1:  # Expected rise > 1%
                    signals['ml_price'] = {
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.MODERATE,
                        'reasoning': f'LSTM predicts {price_change:.1f}% price increase'
                    }
                elif price_change < -3:  # Expected drop > 3%
                    signals['ml_price'] = {
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.STRONG,
                        'reasoning': f'LSTM predicts {price_change:.1f}% price decrease'
                    }
                elif price_change < -1:  # Expected drop > 1%
                    signals['ml_price'] = {
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.MODERATE,
                        'reasoning': f'LSTM predicts {price_change:.1f}% price decrease'
                    }
                else:
                    signals['ml_price'] = {
                        'signal': SignalType.HOLD,
                        'strength': SignalStrength.WEAK,
                        'reasoning': f'LSTM predicts small change: {price_change:.1f}%'
                    }
            
        except Exception as e:
            self.logger.error(f"Error in ML signal calculation: {e}")
        
        return signals
    
    def _confidence_to_strength(self, confidence: float) -> SignalStrength:
        """Convert ML confidence to signal strength"""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.65:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def calculate_momentum_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Momentum indicators analysis
        """
        signals = {}
        latest = df.iloc[-1]
        
        stoch_k = latest.get('stoch_k', 50)
        stoch_d = latest.get('stoch_d', 50)
        williams_r = latest.get('williams_r', -50)
        
        # Stochastic analysis
        if stoch_k < 20 and stoch_d < 20:  # Oversold
            if stoch_k > stoch_d:  # K crossing above D
                signals['stochastic'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'Stochastic oversold with bullish crossover: K={stoch_k:.1f}, D={stoch_d:.1f}'
                }
            else:
                signals['stochastic'] = {
                    'signal': SignalType.BUY,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': f'Stochastic oversold: K={stoch_k:.1f}, D={stoch_d:.1f}'
                }
        elif stoch_k > 80 and stoch_d > 80:  # Overbought
            if stoch_k < stoch_d:  # K crossing below D
                signals['stochastic'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.STRONG,
                    'reasoning': f'Stochastic overbought with bearish crossover: K={stoch_k:.1f}, D={stoch_d:.1f}'
                }
            else:
                signals['stochastic'] = {
                    'signal': SignalType.SELL,
                    'strength': SignalStrength.MODERATE,
                    'reasoning': f'Stochastic overbought: K={stoch_k:.1f}, D={stoch_d:.1f}'
                }
        else:
            signals['stochastic'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': f'Stochastic neutral: K={stoch_k:.1f}, D={stoch_d:.1f}'
            }
        
        # Williams %R
        if williams_r < -80:  # Oversold
            signals['williams_r'] = {
                'signal': SignalType.BUY,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'Williams %R oversold: {williams_r:.1f}'
            }
        elif williams_r > -20:  # Overbought
            signals['williams_r'] = {
                'signal': SignalType.SELL,
                'strength': SignalStrength.MODERATE,
                'reasoning': f'Williams %R overbought: {williams_r:.1f}'
            }
        else:
            signals['williams_r'] = {
                'signal': SignalType.HOLD,
                'strength': SignalStrength.WEAK,
                'reasoning': f'Williams %R neutral: {williams_r:.1f}'
            }
        
        return signals
    
    def generate_composite_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive composite signal
        """
        if df.empty or len(df) < 50:  # Need sufficient data
            return {
                'signal': SignalType.HOLD.value,
                'confidence': 0.5,
                'reasoning': ['Insufficient data for analysis'],
                'timestamp': datetime.now()
            }
        
        # Collect all individual signals
        all_signals = {}
        
        # Technical indicators
        all_signals.update(self.calculate_rsi_signals(df))
        all_signals.update(self.calculate_macd_signals(df))
        all_signals.update(self.calculate_bollinger_signals(df))
        all_signals.update(self.calculate_moving_average_signals(df))
        all_signals.update(self.calculate_volume_signals(df))
        all_signals.update(self.calculate_support_resistance_signals(df))
        all_signals.update(self.calculate_momentum_signals(df))
        
        # Machine learning signals
        if self.ml_models and hasattr(self.ml_models, 'models') and self.ml_models.models:
            all_signals.update(self.calculate_ml_signals(df))
        
        # Calculate weighted scores
        buy_score = 0
        sell_score = 0
        total_weight = 0
        reasoning = []
        
        for indicator, signal_data in all_signals.items():
            weight = self.indicator_weights.get(indicator, 1.0)
            strength_multiplier = signal_data['strength'].value
            
            if signal_data['signal'] == SignalType.BUY:
                buy_score += weight * strength_multiplier
                reasoning.append(f"✓ {signal_data['reasoning']}")
            elif signal_data['signal'] == SignalType.SELL:
                sell_score += weight * strength_multiplier
                reasoning.append(f"✗ {signal_data['reasoning']}")
            else:
                reasoning.append(f"→ {signal_data['reasoning']}")
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final signal
        min_threshold = 1.2  # Minimum threshold for signal generation
        
        if buy_score > sell_score and buy_score > min_threshold:
            final_signal = SignalType.BUY
            confidence = min(buy_score / 3.0, 0.95)  # Cap confidence at 95%
        elif sell_score > buy_score and sell_score > min_threshold:
            final_signal = SignalType.SELL
            confidence = min(sell_score / 3.0, 0.95)
        else:
            final_signal = SignalType.HOLD
            confidence = 0.5
        
        # Calculate entry levels and risk management
        current_price = df['close'].iloc[-1]
        atr = df.get('atr', pd.Series([current_price * 0.02])).iloc[-1]  # Default 2% ATR
        
        if final_signal == SignalType.BUY:
            entry_price = current_price * 1.001  # Slight premium for market orders
            stop_loss = current_price - (2.5 * atr)
            take_profit = current_price + (4 * atr)  # 1.6:1 risk/reward
        elif final_signal == SignalType.SELL:
            entry_price = current_price * 0.999  # Slight discount for market orders
            stop_loss = current_price + (2.5 * atr)
            take_profit = current_price - (4 * atr)
        else:
            entry_price = current_price
            stop_loss = None
            take_profit = None
        
        # Calculate risk/reward ratio
        if stop_loss and take_profit:
            risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        else:
            risk_reward_ratio = None
        
        return {
            'symbol': symbol,
            'signal': final_signal.value,
            'confidence': confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': current_price,
            'risk_reward_ratio': risk_reward_ratio,
            'reasoning': reasoning,
            'individual_signals': {
                indicator: {
                    'signal': data['signal'].value,
                    'strength': data['strength'].value,
                    'reasoning': data['reasoning']
                }
                for indicator, data in all_signals.items()
            },
            'timestamp': datetime.now(),
            'atr': atr
        }
    
    def generate_signals_for_multiple_assets(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate signals for multiple assets
        """
        signals = {}
        
        for symbol, df in data_dict.items():
            try:
                signal = self.generate_composite_signal(df, symbol)
                signals[symbol] = signal
                self.logger.info(f"Generated signal for {symbol}: {signal['signal']} (confidence: {signal['confidence']:.2%})")
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def filter_signals_by_risk(self, signals: Dict[str, Dict], 
                              max_risk_per_trade: float = 0.025) -> Dict[str, Dict]:
        """
        Filter signals based on risk management criteria
        """
        filtered_signals = {}
        
        for symbol, signal in signals.items():
            try:
                # Check risk/reward ratio
                if signal.get('risk_reward_ratio') and signal['risk_reward_ratio'] >= 1.5:
                    # Check risk percentage
                    if signal.get('stop_loss'):
                        risk_percent = abs(signal['entry_price'] - signal['stop_loss']) / signal['entry_price']
                        
                        if risk_percent <= max_risk_per_trade:
                            # Check minimum confidence
                            if signal['confidence'] >= 0.6:
                                filtered_signals[symbol] = signal
                            else:
                                self.logger.info(f"Signal for {symbol} filtered: low confidence ({signal['confidence']:.2%})")
                        else:
                            self.logger.info(f"Signal for {symbol} filtered: high risk ({risk_percent:.2%})")
                    else:
                        # No stop loss defined, use confidence threshold
                        if signal['confidence'] >= 0.75:
                            filtered_signals[symbol] = signal
                        else:
                            self.logger.info(f"Signal for {symbol} filtered: no stop loss and low confidence")
                else:
                    self.logger.info(f"Signal for {symbol} filtered: poor risk/reward ratio")
                    
            except Exception as e:
                self.logger.error(f"Error filtering signal for {symbol}: {e}")
        
        return filtered_signals