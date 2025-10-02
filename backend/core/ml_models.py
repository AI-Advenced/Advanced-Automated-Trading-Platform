"""
Advanced Machine Learning Models for Trading
Includes technical analysis, predictive models, and pattern recognition
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import ta  # Technical Analysis library
import joblib
import logging
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysisEngine:
    """
    Comprehensive technical analysis engine
    """
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        """
        data = df.copy()
        
        # Trend Indicators
        data['sma_5'] = ta.trend.sma_indicator(data['close'], window=5)
        data['sma_10'] = ta.trend.sma_indicator(data['close'], window=10)
        data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['sma_100'] = ta.trend.sma_indicator(data['close'], window=100)
        
        data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
        data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
        data['ema_50'] = ta.trend.ema_indicator(data['close'], window=50)
        
        # MACD
        data['macd'] = ta.trend.macd(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        data['macd_diff'] = ta.trend.macd_diff(data['close'])
        
        # Momentum Indicators
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        data['rsi_30'] = ta.momentum.rsi(data['close'], window=30)
        
        # Stochastic
        data['stoch_k'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
        data['stoch_d'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
        
        # Williams %R
        data['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])
        
        # Volatility Indicators
        data['bb_high'] = ta.volatility.bollinger_hband(data['close'])
        data['bb_low'] = ta.volatility.bollinger_lband(data['close'])
        data['bb_mid'] = ta.volatility.bollinger_mavg(data['close'])
        data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['bb_mid']
        data['bb_position'] = (data['close'] - data['bb_low']) / (data['bb_high'] - data['bb_low'])
        
        # Average True Range
        data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
        
        # Volume Indicators
        data['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume'], window=20)
        data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
        data['ad'] = ta.volume.acc_dist_index(data['high'], data['low'], data['close'], data['volume'])
        data['cmf'] = ta.volume.chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'])
        
        # Price Action Features
        data['price_change'] = data['close'].pct_change()
        data['price_change_2'] = data['close'].pct_change(2)
        data['price_change_5'] = data['close'].pct_change(5)
        
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        data['body_size'] = np.abs(data['close'] - data['open'])
        
        # Volatility measures
        data['volatility_5'] = data['price_change'].rolling(window=5).std()
        data['volatility_20'] = data['price_change'].rolling(window=20).std()
        
        # Support and Resistance
        data['support_5'] = data['low'].rolling(window=5).min()
        data['resistance_5'] = data['high'].rolling(window=5).max()
        data['support_20'] = data['low'].rolling(window=20).min()
        data['resistance_20'] = data['high'].rolling(window=20).max()
        
        # Distance from support/resistance
        data['dist_from_support'] = (data['close'] - data['support_20']) / data['support_20']
        data['dist_from_resistance'] = (data['resistance_20'] - data['close']) / data['resistance_20']
        
        # Fibonacci retracements
        data['fib_23.6'] = data['support_20'] + 0.236 * (data['resistance_20'] - data['support_20'])
        data['fib_38.2'] = data['support_20'] + 0.382 * (data['resistance_20'] - data['support_20'])
        data['fib_61.8'] = data['support_20'] + 0.618 * (data['resistance_20'] - data['support_20'])
        
        # Additional features
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
        data['price_position'] = (data['close'] - data['close'].rolling(window=20).min()) / \
                                (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
        
        return data

class TradingMLModels:
    """
    Comprehensive machine learning models for trading
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features using technical analysis
        """
        return TechnicalAnalysisEngine.calculate_indicators(df)
    
    def create_target_variables(self, df: pd.DataFrame, forecast_horizon: int = 5) -> pd.DataFrame:
        """
        Create target variables for prediction
        """
        data = df.copy()
        
        # Direction prediction (up/down)
        data['future_return'] = data['close'].shift(-forecast_horizon) / data['close'] - 1
        data['direction'] = (data['future_return'] > 0).astype(int)
        
        # Multi-class direction (strong up, up, sideways, down, strong down)
        conditions = [
            data['future_return'] > 0.03,  # Strong up (>3%)
            (data['future_return'] > 0.01) & (data['future_return'] <= 0.03),  # Up (1-3%)
            (data['future_return'] >= -0.01) & (data['future_return'] <= 0.01),  # Sideways (-1% to 1%)
            (data['future_return'] >= -0.03) & (data['future_return'] < -0.01),  # Down (-3% to -1%)
            data['future_return'] < -0.03  # Strong down (<-3%)
        ]
        choices = [4, 3, 2, 1, 0]  # Strong up, Up, Sideways, Down, Strong down
        data['direction_multiclass'] = np.select(conditions, choices, default=2)
        
        # Price target
        data['price_target'] = data['close'].shift(-forecast_horizon)
        
        # Volatility target
        data['volatility_target'] = data['close'].rolling(window=forecast_horizon).std().shift(-forecast_horizon)
        
        # Risk level (based on expected volatility)
        data['risk_level'] = pd.qcut(data['volatility_20'], q=3, labels=['low', 'medium', 'high'])
        
        return data.dropna()
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for modeling
        """
        # Exclude target variables and timestamp columns
        exclude_cols = [
            'direction', 'future_return', 'price_target', 'volatility_target',
            'direction_multiclass', 'risk_level', 'open', 'high', 'low', 'close', 'volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('Unnamed')]
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols if df[col].isna().sum() / len(df) < 0.2]
        
        return feature_cols
    
    def train_direction_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train ensemble model for direction prediction
        """
        self.logger.info("Training direction prediction model...")
        
        # Prepare data
        features_df = self.prepare_features(df)
        target_df = self.create_target_variables(features_df)
        
        # Select features
        self.feature_columns = self.select_features(target_df)
        
        X = target_df[self.feature_columns].fillna(0)
        y = target_df['direction']
        
        # Time series split for financial data
        tscv = TimeSeriesSplit(n_splits=5)
        split_idx = list(tscv.split(X))[-1]  # Use the last split
        train_idx, test_idx = split_idx
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create ensemble model
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Voting classifier
        ensemble = VotingClassifier(
            estimators=[('rf', rf)],
            voting='soft'
        )
        
        # Train model
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = ensemble.score(X_train_scaled, y_train)
        test_score = ensemble.score(X_test_scaled, y_test)
        
        y_pred = ensemble.predict(X_test_scaled)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        # Feature importance (from Random Forest)
        feature_importance = dict(zip(self.feature_columns, rf.feature_importances_))
        
        # Save model and scaler
        self.models['direction'] = ensemble
        self.scalers['direction'] = scaler
        
        return {
            'model': ensemble,
            'scaler': scaler,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'feature_columns': self.feature_columns
        }
    
    def train_multiclass_direction_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model for multi-class direction prediction
        """
        self.logger.info("Training multi-class direction model...")
        
        # Prepare data
        features_df = self.prepare_features(df)
        target_df = self.create_target_variables(features_df)
        
        feature_columns = self.select_features(target_df)
        
        X = target_df[feature_columns].fillna(0)
        y = target_df['direction_multiclass']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        split_idx = list(tscv.split(X))[-1]
        train_idx, test_idx = split_idx
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Multi-class Random Forest
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Save model
        self.models['direction_multiclass'] = model
        self.scalers['direction_multiclass'] = scaler
        
        return {
            'model': model,
            'scaler': scaler,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
    
    def train_lstm_model(self, df: pd.DataFrame, sequence_length: int = 60, 
                        forecast_horizon: int = 5) -> Dict[str, Any]:
        """
        Train LSTM model for price prediction
        """
        self.logger.info("Training LSTM model...")
        
        # Prepare data with technical indicators
        data = self.prepare_features(df)
        
        # Select features for LSTM
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'atr']
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if not available_cols:
            available_cols = ['close']
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[available_cols].fillna(method='ffill'))
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - forecast_horizon):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i+forecast_horizon, 0])  # Predict close price
        
        X, y = np.array(X), np.array(y)
        
        # Split data (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build advanced LSTM model
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(units=100, return_sequences=True),
            Dropout(0.3),
            LSTM(units=50, return_sequences=False),
            Dropout(0.3),
            Dense(units=50, activation='relu'),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        # Compile with advanced optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Save model
        self.models['lstm'] = model
        self.scalers['lstm'] = scaler
        
        return {
            'model': model,
            'scaler': scaler,
            'history': history.history,
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'feature_columns': available_cols,
            'final_val_loss': min(history.history['val_loss'])
        }
    
    def predict_direction(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict market direction
        """
        if 'direction' not in self.models:
            raise ValueError("Direction model not trained")
        
        # Prepare features
        features_df = self.prepare_features(current_data)
        
        # Get latest features
        latest_features = features_df[self.feature_columns].iloc[-1:].fillna(0).values
        
        # Scale features
        scaled_features = self.scalers['direction'].transform(latest_features)
        
        # Make prediction
        prediction = self.models['direction'].predict(scaled_features)[0]
        probabilities = self.models['direction'].predict_proba(scaled_features)[0]
        
        return {
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': max(probabilities),
            'probabilities': {
                'down': probabilities[0],
                'up': probabilities[1]
            },
            'prediction_time': datetime.now()
        }
    
    def predict_price_lstm(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict future price using LSTM
        """
        if 'lstm' not in self.models:
            raise ValueError("LSTM model not trained")
        
        model_info = self.models.get('lstm_info', {})
        sequence_length = model_info.get('sequence_length', 60)
        feature_cols = model_info.get('feature_columns', ['close'])
        
        # Prepare data
        data = self.prepare_features(current_data)
        
        # Get recent data for sequence
        available_cols = [col for col in feature_cols if col in data.columns]
        recent_data = data[available_cols].tail(sequence_length).fillna(method='ffill')
        
        if len(recent_data) < sequence_length:
            raise ValueError(f"Insufficient data: need {sequence_length} periods, got {len(recent_data)}")
        
        # Scale data
        scaled_data = self.scalers['lstm'].transform(recent_data)
        
        # Prepare for prediction
        X = np.array([scaled_data])
        
        # Make prediction
        prediction_scaled = self.models['lstm'].predict(X, verbose=0)
        
        # Inverse transform (only for price column)
        price_scaler = MinMaxScaler()
        price_scaler.fit(current_data[['close']].fillna(method='ffill'))
        prediction = price_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        current_price = current_data['close'].iloc[-1]
        price_change = (prediction - current_price) / current_price * 100
        
        return {
            'predicted_price': prediction,
            'current_price': current_price,
            'price_change_percent': price_change,
            'prediction_time': datetime.now()
        }
    
    def get_feature_importance(self, model_name: str = 'direction') -> Dict[str, float]:
        """
        Get feature importance from trained model
        """
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.feature_columns, model.feature_importances_))
        elif hasattr(model, 'estimators_'):
            # For ensemble methods
            importances = model.estimators_[0].feature_importances_
            return dict(zip(self.feature_columns, importances))
        
        return {}
    
    def save_models(self, path: str):
        """
        Save all trained models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'lstm':
                model.save(f"{path}/{name}_model.h5")
            else:
                joblib.dump(model, f"{path}/{name}_model.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{path}/{name}_scaler.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_columns, f"{path}/feature_columns.pkl")
        
        self.logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """
        Load trained models
        """
        import os
        
        if not os.path.exists(path):
            self.logger.error(f"Model path {path} does not exist")
            return
        
        # Load feature columns
        feature_file = f"{path}/feature_columns.pkl"
        if os.path.exists(feature_file):
            self.feature_columns = joblib.load(feature_file)
        
        # Load models and scalers
        for file in os.listdir(path):
            if file.endswith('_model.h5'):
                name = file.replace('_model.h5', '')
                self.models[name] = tf.keras.models.load_model(f"{path}/{file}")
            elif file.endswith('_model.pkl'):
                name = file.replace('_model.pkl', '')
                self.models[name] = joblib.load(f"{path}/{file}")
            elif file.endswith('_scaler.pkl'):
                name = file.replace('_scaler.pkl', '')
                self.scalers[name] = joblib.load(f"{path}/{file}")
        
        self.logger.info(f"Models loaded from {path}")
    
    def retrain_with_new_data(self, new_data: pd.DataFrame):
        """
        Retrain models with new market data
        """
        self.logger.info("Retraining models with new data...")
        
        # Retrain direction model
        if 'direction' in self.models:
            direction_results = self.train_direction_model(new_data)
            self.logger.info(f"Direction model retrained - Accuracy: {direction_results['test_accuracy']:.3f}")
        
        # Retrain LSTM model
        if 'lstm' in self.models:
            lstm_results = self.train_lstm_model(new_data)
            self.logger.info(f"LSTM model retrained - Val Loss: {lstm_results['final_val_loss']:.6f}")
        
        return True