"""
Unit tests for the ML Models component.
Tests machine learning model training, prediction, and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tempfile
import os

from backend.core.ml_models import TradingMLModels
from backend.utils.exceptions import MLModelError
from backend.config.settings import TradingSettings

class TestTradingMLModels:
    """Test suite for TradingMLModels"""
    
    @pytest.fixture
    def ml_models(self, test_settings):
        """Create TradingMLModels instance for testing"""
        return TradingMLModels(test_settings)
    
    @pytest.fixture
    def sample_features_data(self, sample_ohlcv_data):
        """Create sample features data with technical indicators"""
        df = sample_ohlcv_data.copy()
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = 50 + np.random.randn(len(df)) * 10  # Mock RSI
        df['macd'] = np.random.randn(len(df)) * 0.1      # Mock MACD
        df['bb_upper'] = df['close'] * 1.02
        df['bb_lower'] = df['close'] * 0.98
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Create target (next period return > 0)
        df['target'] = (df['close'].shift(-1) / df['close'] - 1 > 0).astype(int)
        
        return df.dropna()
    
    def test_initialization(self, ml_models):
        """Test ML models initialization"""
        assert ml_models.settings is not None
        assert ml_models.models == {}
        assert ml_models.model_performance == {}
        assert ml_models.feature_importance == {}
    
    def test_prepare_features(self, ml_models, sample_ohlcv_data):
        """Test feature preparation from OHLCV data"""
        features_df = ml_models.prepare_features(sample_ohlcv_data)
        
        # Check that features are created
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        
        # Check for expected feature columns
        expected_features = [
            'returns', 'volatility', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns, f"Missing feature: {feature}"
    
    def test_calculate_technical_indicators(self, ml_models, sample_ohlcv_data):
        """Test calculation of individual technical indicators"""
        df = sample_ohlcv_data.copy()
        
        # Test SMA calculation
        sma_20 = ml_models._calculate_sma(df['close'], 20)
        assert len(sma_20) == len(df)
        assert not np.isnan(sma_20.iloc[-1])  # Last value should be valid
        
        # Test EMA calculation
        ema_12 = ml_models._calculate_ema(df['close'], 12)
        assert len(ema_12) == len(df)
        
        # Test RSI calculation
        rsi = ml_models._calculate_rsi(df['close'], 14)
        assert len(rsi) == len(df)
        assert 0 <= rsi.iloc[-10:].max() <= 100  # RSI should be 0-100
        
        # Test MACD calculation
        macd, macd_signal = ml_models._calculate_macd(df['close'])
        assert len(macd) == len(df)
        assert len(macd_signal) == len(df)
        
        # Test Bollinger Bands
        bb_upper, bb_middle, bb_lower = ml_models._calculate_bollinger_bands(df['close'])
        assert len(bb_upper) == len(df)
        assert (bb_upper >= bb_middle).all()  # Upper should be >= middle
        assert (bb_middle >= bb_lower).all()  # Middle should be >= lower
    
    def test_create_target_variable(self, ml_models, sample_ohlcv_data):
        """Test target variable creation"""
        df = sample_ohlcv_data.copy()
        
        # Test binary classification target
        target = ml_models._create_target_variable(df, 'binary', periods=1)
        assert len(target) == len(df) - 1  # One less due to forward looking
        assert set(target.unique()).issubset({0, 1})  # Only 0s and 1s
        
        # Test regression target
        target_reg = ml_models._create_target_variable(df, 'regression', periods=1)
        assert len(target_reg) == len(df) - 1
        assert isinstance(target_reg.iloc[0], (int, float))
    
    @pytest.mark.asyncio
    async def test_train_random_forest_model(self, ml_models, sample_features_data):
        """Test Random Forest model training"""
        # Prepare data
        features = sample_features_data.drop(['target'], axis=1)
        target = sample_features_data['target']
        
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Train model
        model_info = await ml_models._train_random_forest(
            numeric_features.iloc[:-50],  # Training set
            target.iloc[:-50],
            numeric_features.iloc[-50:],  # Test set
            target.iloc[-50:]
        )
        
        assert 'model' in model_info
        assert 'accuracy' in model_info
        assert 'feature_importance' in model_info
        assert isinstance(model_info['model'], RandomForestClassifier)
        assert 0 <= model_info['accuracy'] <= 1
    
    @pytest.mark.asyncio
    async def test_train_lstm_model(self, ml_models, sample_features_data):
        """Test LSTM model training"""
        # Prepare sequential data
        features = sample_features_data.select_dtypes(include=[np.number]).drop(['target'], axis=1)
        target = sample_features_data['target']
        
        # Mock TensorFlow/Keras components
        with patch('tensorflow.keras.models.Sequential') as mock_sequential:
            with patch('tensorflow.keras.layers.LSTM') as mock_lstm:
                with patch('tensorflow.keras.layers.Dense') as mock_dense:
                    mock_model = MagicMock()
                    mock_model.fit.return_value = MagicMock()
                    mock_model.evaluate.return_value = [0.5, 0.75]  # [loss, accuracy]
                    mock_sequential.return_value = mock_model
                    
                    model_info = await ml_models._train_lstm_model(
                        features.iloc[:-50],
                        target.iloc[:-50],
                        features.iloc[-50:],
                        target.iloc[-50:]
                    )
                    
                    assert 'model' in model_info
                    assert 'accuracy' in model_info
                    assert model_info['accuracy'] == 0.75
    
    @pytest.mark.asyncio
    async def test_train_gradient_boosting_model(self, ml_models, sample_features_data):
        """Test Gradient Boosting model training"""
        features = sample_features_data.select_dtypes(include=[np.number]).drop(['target'], axis=1)
        target = sample_features_data['target']
        
        model_info = await ml_models._train_gradient_boosting(
            features.iloc[:-50],
            target.iloc[:-50],
            features.iloc[-50:],
            target.iloc[-50:]
        )
        
        assert 'model' in model_info
        assert 'accuracy' in model_info
        assert 'feature_importance' in model_info
        assert 0 <= model_info['accuracy'] <= 1
    
    @pytest.mark.asyncio
    async def test_full_model_training(self, ml_models, sample_ohlcv_data):
        """Test complete model training pipeline"""
        symbol = 'BTCUSDT'
        
        # Mock the individual training methods
        mock_rf_result = {
            'model': MagicMock(),
            'accuracy': 0.75,
            'feature_importance': {'feature1': 0.5}
        }
        
        with patch.object(ml_models, '_train_random_forest', return_value=mock_rf_result):
            with patch.object(ml_models, '_train_lstm_model', return_value=mock_rf_result):
                with patch.object(ml_models, '_train_gradient_boosting', return_value=mock_rf_result):
                    
                    results = await ml_models.train_models(symbol, sample_ohlcv_data)
                    
                    assert 'random_forest' in results
                    assert 'lstm' in results
                    assert 'gradient_boosting' in results
                    
                    # Check that models are stored
                    assert symbol in ml_models.models
                    assert 'random_forest' in ml_models.models[symbol]
    
    @pytest.mark.asyncio
    async def test_model_prediction(self, ml_models, sample_features_data):
        """Test model prediction"""
        symbol = 'BTCUSDT'
        
        # Create and store a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])  # Buy signal
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # Probabilities
        
        ml_models.models[symbol] = {
            'random_forest': {'model': mock_model, 'accuracy': 0.75}
        }
        
        # Prepare current data
        current_data = sample_features_data.iloc[-1:].select_dtypes(include=[np.number])
        current_data = current_data.drop(['target'], axis=1, errors='ignore')
        
        prediction = await ml_models.predict(symbol, current_data)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'probabilities' in prediction
        assert prediction['prediction'] in [0, 1]
        assert 0 <= prediction['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, ml_models, sample_features_data):
        """Test ensemble prediction from multiple models"""
        symbol = 'BTCUSDT'
        
        # Create multiple mock models with different predictions
        mock_rf = MagicMock()
        mock_rf.predict.return_value = np.array([1])
        mock_rf.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        mock_gb = MagicMock()
        mock_gb.predict.return_value = np.array([0])
        mock_gb.predict_proba.return_value = np.array([[0.6, 0.4]])
        
        ml_models.models[symbol] = {
            'random_forest': {'model': mock_rf, 'accuracy': 0.8},
            'gradient_boosting': {'model': mock_gb, 'accuracy': 0.7}
        }
        
        current_data = sample_features_data.iloc[-1:].select_dtypes(include=[np.number])
        current_data = current_data.drop(['target'], axis=1, errors='ignore')
        
        prediction = await ml_models.predict(symbol, current_data, ensemble=True)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'model_predictions' in prediction
        assert len(prediction['model_predictions']) == 2
    
    def test_model_serialization(self, ml_models, temp_directory):
        """Test model saving and loading"""
        symbol = 'BTCUSDT'
        
        # Create a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Fit with dummy data
        X_dummy = np.random.randn(100, 5)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        
        ml_models.models[symbol] = {
            'random_forest': {
                'model': model,
                'accuracy': 0.75,
                'feature_importance': {'feature1': 0.5}
            }
        }
        
        # Test saving
        model_path = os.path.join(temp_directory, f"{symbol}_random_forest.pkl")
        ml_models.save_model(symbol, 'random_forest', model_path)
        
        assert os.path.exists(model_path)
        
        # Test loading
        loaded_model_info = ml_models.load_model(model_path)
        
        assert 'model' in loaded_model_info
        assert 'accuracy' in loaded_model_info
        assert loaded_model_info['accuracy'] == 0.75
    
    def test_feature_importance_analysis(self, ml_models, sample_features_data):
        """Test feature importance analysis"""
        # Create mock model with feature importance
        from sklearn.ensemble import RandomForestClassifier
        
        features = sample_features_data.select_dtypes(include=[np.number]).drop(['target'], axis=1)
        target = sample_features_data['target']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(features, target)
        
        # Get feature importance
        importance = ml_models._get_feature_importance(model, features.columns)
        
        assert isinstance(importance, dict)
        assert len(importance) == len(features.columns)
        assert all(0 <= score <= 1 for score in importance.values())
        
        # Test top features
        top_features = ml_models._get_top_features(importance, n=5)
        assert len(top_features) <= 5
        assert isinstance(top_features, list)
    
    def test_model_performance_evaluation(self, ml_models, sample_features_data):
        """Test model performance evaluation"""
        features = sample_features_data.select_dtypes(include=[np.number]).drop(['target'], axis=1)
        target = sample_features_data['target']
        
        # Create and train a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(features[:-20], target[:-20])
        
        # Evaluate performance
        performance = ml_models._evaluate_model_performance(
            model, 
            features[-20:], 
            target[-20:]
        )
        
        assert 'accuracy' in performance
        assert 'precision' in performance
        assert 'recall' in performance
        assert 'f1_score' in performance
        assert all(0 <= score <= 1 for score in performance.values())
    
    @pytest.mark.asyncio
    async def test_model_retraining_trigger(self, ml_models, sample_ohlcv_data):
        """Test automatic model retraining trigger"""
        symbol = 'BTCUSDT'
        
        # Mock performance degradation
        ml_models.model_performance[symbol] = {
            'random_forest': {'accuracy': 0.45}  # Below threshold
        }
        
        should_retrain = ml_models._should_retrain_model(symbol, 'random_forest')
        assert should_retrain is True
        
        # Mock good performance
        ml_models.model_performance[symbol] = {
            'random_forest': {'accuracy': 0.85}  # Above threshold
        }
        
        should_retrain = ml_models._should_retrain_model(symbol, 'random_forest')
        assert should_retrain is False
    
    def test_data_preprocessing(self, ml_models, sample_ohlcv_data):
        """Test data preprocessing and cleaning"""
        # Add some problematic data
        df = sample_ohlcv_data.copy()
        df.loc[10, 'close'] = np.inf  # Infinite value
        df.loc[15, 'volume'] = np.nan  # NaN value
        df.loc[20, 'high'] = -100     # Invalid negative price
        
        # Preprocess the data
        cleaned_df = ml_models._preprocess_data(df)
        
        # Check that problematic data is handled
        assert not np.isinf(cleaned_df.values).any()
        assert not np.isnan(cleaned_df.values).any()
        assert (cleaned_df['high'] > 0).all()
    
    @pytest.mark.asyncio
    async def test_cross_validation(self, ml_models, sample_features_data):
        """Test cross-validation for model evaluation"""
        features = sample_features_data.select_dtypes(include=[np.number]).drop(['target'], axis=1)
        target = sample_features_data['target']
        
        # Mock cross-validation
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.7, 0.75, 0.8, 0.72, 0.78])
            
            cv_scores = await ml_models._perform_cross_validation(
                RandomForestClassifier(), 
                features, 
                target
            )
            
            assert 'mean_score' in cv_scores
            assert 'std_score' in cv_scores
            assert 'scores' in cv_scores
            assert cv_scores['mean_score'] == 0.75
    
    def test_hyperparameter_tuning(self, ml_models, sample_features_data):
        """Test hyperparameter optimization"""
        features = sample_features_data.select_dtypes(include=[np.number]).drop(['target'], axis=1)
        target = sample_features_data['target']
        
        # Mock GridSearchCV
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
            mock_result = MagicMock()
            mock_result.best_params_ = {'n_estimators': 100, 'max_depth': 10}
            mock_result.best_score_ = 0.82
            mock_result.best_estimator_ = RandomForestClassifier()
            
            mock_grid.return_value = mock_result
            mock_grid.return_value.fit.return_value = mock_result
            
            best_params = ml_models._tune_hyperparameters(
                RandomForestClassifier(),
                features.iloc[:100],  # Smaller dataset for speed
                target.iloc[:100],
                {'n_estimators': [50, 100], 'max_depth': [5, 10]}
            )
            
            assert 'n_estimators' in best_params
            assert 'max_depth' in best_params
    
    @pytest.mark.asyncio
    async def test_prediction_confidence_calculation(self, ml_models, sample_features_data):
        """Test prediction confidence calculation"""
        # Create mock model with probability predictions
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # High confidence
        
        confidence = ml_models._calculate_prediction_confidence(mock_model, np.array([[1, 2, 3, 4, 5]]))
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be high confidence
        
        # Test low confidence
        mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])  # Low confidence
        confidence_low = ml_models._calculate_prediction_confidence(mock_model, np.array([[1, 2, 3, 4, 5]]))
        
        assert confidence_low < confidence  # Should be lower confidence
    
    def test_model_versioning(self, ml_models, temp_directory):
        """Test model versioning and management"""
        symbol = 'BTCUSDT'
        model_type = 'random_forest'
        
        # Create multiple model versions
        for version in [1, 2, 3]:
            model_path = os.path.join(temp_directory, f"{symbol}_{model_type}_v{version}.pkl")
            
            # Create simple model
            model = RandomForestClassifier(n_estimators=10 * version)
            model_info = {
                'model': model,
                'version': version,
                'accuracy': 0.7 + (version * 0.05)
            }
            
            # Save model
            joblib.dump(model_info, model_path)
        
        # Test getting latest version
        latest_version = ml_models._get_latest_model_version(temp_directory, symbol, model_type)
        assert latest_version == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_in_training(self, ml_models, sample_ohlcv_data):
        """Test error handling during model training"""
        symbol = 'BTCUSDT'
        
        # Test with insufficient data
        small_data = sample_ohlcv_data.head(10)  # Too small for training
        
        with pytest.raises(MLModelError, match="Insufficient data"):
            await ml_models.train_models(symbol, small_data)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        
        with pytest.raises(MLModelError):
            await ml_models.train_models(symbol, invalid_data)
    
    @pytest.mark.asyncio
    async def test_prediction_without_trained_model(self, ml_models, sample_features_data):
        """Test prediction when no model is trained"""
        symbol = 'BTCUSDT'
        current_data = sample_features_data.iloc[-1:]
        
        with pytest.raises(MLModelError, match="No trained models"):
            await ml_models.predict(symbol, current_data)