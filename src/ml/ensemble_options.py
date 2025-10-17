"""
Ensemble Models for Options Trading
Stacked ensemble with regime-dependent weights and meta-learning
XGBoost, LightGBM, Random Forest, LSTM with regime-dependent meta-learner
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import os

# ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from src.portfolio.account_manager import AccountProfile, AccountTier
from src.ml.options_deep_learning.volatility_lstm import VolatilityLSTMForecaster
from src.ml.options_deep_learning.price_transformer import PriceTransformer


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    prediction: float
    confidence: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    regime: str
    feature_importance: Dict[str, float]
    model_version: str


@dataclass
class EnsembleMetrics:
    """Ensemble model performance metrics"""
    mse: float
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    sharpe_ratio: float
    model_correlations: Dict[str, float]
    ensemble_diversity: float


class OptionsEnsemble:
    """
    Stacked ensemble for options trading with regime-dependent weights
    
    Architecture:
    - Level 0: XGBoost (win probability), LightGBM (expected return), Random Forest (risk score), LSTM (volatility forecast)
    - Level 1: Meta-learner with regime-dependent weights and recent performance weighting
    - Account-size-specific model training
    """
    
    def __init__(self, account_profile: AccountProfile, config: Dict = None):
        self.profile = account_profile
        
        # Configuration
        self.config = config or self._default_config()
        
        # Level 0 models (base learners)
        self.base_models = {
            'xgboost_win': None,
            'lightgbm_return': None,
            'random_forest_risk': None,
            'lstm_volatility': None
        }
        
        # Level 1 meta-learner
        self.meta_learner = None
        
        # Data preprocessing
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature names
        self.feature_names = []
        
        # Training data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Regime detection
        self.regime_history = []
        self.regime_weights = {}
        
        # Performance tracking
        self.model_performance = {}
        self.ensemble_metrics = []
        
        # Deep learning models
        self.lstm_model = None
        self.transformer_model = None
        
        logger.info(f"OptionsEnsemble initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default configuration for ensemble"""
        return {
            'validation_split': 0.2,
            'time_series_cv_folds': 5,
            'feature_engineering': True,
            'regime_detection': True,
            'meta_learning': True,
            'model_retraining_frequency': 30,  # days
            'performance_window': 100,  # trades
            'regime_adaptation_rate': 0.1,
            'diversity_weight': 0.2,
            'base_models': {
                'xgboost_win': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'lightgbm_return': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_byfeature': 0.8,
                    'random_state': 42
                },
                'random_forest_risk': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'meta_learner': {
                'model_type': 'linear',  # 'linear', 'tree', 'neural'
                'regularization': 0.01,
                'learning_rate': 0.001
            }
        }
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ensemble training
        
        Args:
            data: Historical options and market data
        
        Returns:
            X, y arrays for training
        """
        try:
            # Ensure data is sorted by date
            data = data.sort_values('date').reset_index(drop=True)
            
            # Feature engineering
            features_df = self._engineer_features(data)
            
            # Select features
            feature_cols = self._get_feature_columns(features_df)
            self.feature_names = feature_cols
            
            # Create target variables
            targets = self._create_targets(features_df)
            
            # Align features and targets
            min_length = min(len(features_df), len(targets))
            features = features_df[feature_cols].values[:min_length]
            targets = targets[:min_length]
            
            # Split into train/validation
            split_idx = int(len(features) * (1 - self.config['validation_split']))
            self.X_train = features[:split_idx]
            self.y_train = targets[:split_idx]
            self.X_val = features[split_idx:]
            self.y_val = targets[split_idx:]
            
            # Scale features
            self.X_train = self.scaler_X.fit_transform(self.X_train)
            self.X_val = self.scaler_X.transform(self.X_val)
            
            logger.info(f"Prepared data: {len(self.X_train)} train, {len(self.X_val)} val samples")
            
            return np.vstack([self.X_train, self.X_val]), np.concatenate([self.y_train, self.y_val])
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return np.array([]), np.array([])
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ensemble training"""
        try:
            features_df = data.copy()
            
            # Price-based features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            features_df['volatility'] = features_df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(features_df['close'])
            features_df['macd'], features_df['macd_signal'] = self._calculate_macd(features_df['close'])
            features_df['bb_upper'], features_df['bb_middle'], features_df['bb_lower'] = self._calculate_bollinger_bands(features_df['close'])
            
            # Options-specific features
            if 'iv' not in features_df.columns:
                features_df['iv'] = 0.2
            if 'delta' not in features_df.columns:
                features_df['delta'] = 0.5
            if 'gamma' not in features_df.columns:
                features_df['gamma'] = 0.01
            if 'theta' not in features_df.columns:
                features_df['theta'] = -0.01
            if 'vega' not in features_df.columns:
                features_df['vega'] = 0.1
            
            if 'put_call_ratio' not in features_df.columns:
                features_df['put_call_ratio'] = 1.0
            if 'iv_rank' not in features_df.columns:
                features_df['iv_rank'] = 0.5
            if 'skew' not in features_df.columns:
                features_df['skew'] = 0.0
            if 'term_structure' not in features_df.columns:
                features_df['term_structure'] = 0.0
            
            # Volume features
            features_df['volume_ma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma']
            
            # Price momentum
            features_df['momentum_5'] = features_df['close'] / features_df['close'].shift(5) - 1
            features_df['momentum_20'] = features_df['close'] / features_df['close'].shift(20) - 1
            
            # Volatility features
            features_df['vol_of_vol'] = features_df['volatility'].rolling(window=20).std()
            features_df['iv_realized_ratio'] = features_df['iv'] / features_df['volatility']
            
            # Market regime features
            features_df['vix_level'] = features_df.get('vix', 20)
            features_df['regime'] = self._detect_regime(features_df)
            
            # Temporal features
            features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
            features_df['month'] = pd.to_datetime(features_df['date']).dt.month
            features_df['quarter'] = pd.to_datetime(features_df['date']).dt.quarter
            
            # Interaction features
            features_df['delta_vol_interaction'] = features_df['delta'] * features_df['volatility']
            features_df['theta_time_interaction'] = features_df['theta'] * features_df.get('days_to_expiry', 30)
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return data
    
    def _get_feature_columns(self, features_df: pd.DataFrame) -> List[str]:
        """Get feature column names"""
        feature_cols = [
            'close', 'volume', 'high', 'low', 'open',
            'returns', 'volatility', 'rsi', 'macd',
            'iv', 'delta', 'gamma', 'theta', 'vega',
            'put_call_ratio', 'iv_rank', 'skew', 'term_structure',
            'volume_ratio', 'momentum_5', 'momentum_20',
            'vol_of_vol', 'iv_realized_ratio', 'vix_level',
            'day_of_week', 'month', 'quarter',
            'delta_vol_interaction', 'theta_time_interaction'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        return available_cols
    
    def _create_targets(self, features_df: pd.DataFrame) -> np.ndarray:
        """Create target variables for ensemble"""
        try:
            # Multi-target: [win_probability, expected_return, risk_score, volatility_forecast]
            targets = []
            
            for i in range(len(features_df)):
                # Win probability (binary classification target)
                if i + 1 < len(features_df):
                    future_return = features_df['returns'].iloc[i + 1]
                    win_prob = 1.0 if future_return > 0 else 0.0
                else:
                    win_prob = 0.5
                
                # Expected return (regression target)
                expected_return = features_df['returns'].iloc[i] if i < len(features_df) else 0.0
                
                # Risk score (volatility-based)
                risk_score = features_df['volatility'].iloc[i] if i < len(features_df) else 0.0
                
                # Volatility forecast (next period volatility)
                if i + 1 < len(features_df):
                    vol_forecast = features_df['volatility'].iloc[i + 1]
                else:
                    vol_forecast = features_df['volatility'].iloc[i]
                
                targets.append([win_prob, expected_return, risk_score, vol_forecast])
            
            return np.array(targets)
            
        except Exception as e:
            logger.error(f"Error creating targets: {e}")
            return np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(50, index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            return macd, macd_signal
        except:
            return pd.Series(0, index=prices.index), pd.Series(0, index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, rolling_mean, lower_band
        except:
            return prices, prices, prices
    
    def _detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regime"""
        try:
            # Simple regime detection based on volatility
            volatility = data['volatility']
            vix = data.get('vix', 20)
            
            regime = pd.Series('NORMAL_VOL', index=data.index)
            
            # High volatility regime
            high_vol_mask = (volatility > volatility.quantile(0.8)) | (vix > 30)
            regime[high_vol_mask] = 'HIGH_VOL'
            
            # Low volatility regime
            low_vol_mask = (volatility < volatility.quantile(0.2)) & (vix < 15)
            regime[low_vol_mask] = 'LOW_VOL'
            
            # Crisis regime
            crisis_mask = (volatility > volatility.quantile(0.95)) & (vix > 40)
            regime[crisis_mask] = 'CRISIS'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return pd.Series('NORMAL_VOL', index=data.index)
    
    def train_base_models(self, data: pd.DataFrame) -> bool:
        """
        Train base models (Level 0)
        
        Args:
            data: Training data
        
        Returns:
            Success status
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            if len(X) == 0:
                logger.error("No data available for training")
                return False
            
            # Split targets
            y_win = y[:, 0]  # Win probability
            y_return = y[:, 1]  # Expected return
            y_risk = y[:, 2]  # Risk score
            y_vol = y[:, 3]  # Volatility forecast
            
            # Train XGBoost for win probability
            logger.info("Training XGBoost win probability model...")
            xgb_config = self.config['base_models']['xgboost_win']
            self.base_models['xgboost_win'] = xgb.XGBClassifier(**xgb_config)
            self.base_models['xgboost_win'].fit(self.X_train, y_win)
            
            # Train LightGBM for expected return
            logger.info("Training LightGBM expected return model...")
            lgb_config = self.config['base_models']['lightgbm_return']
            self.base_models['lightgbm_return'] = lgb.LGBMRegressor(**lgb_config)
            self.base_models['lightgbm_return'].fit(self.X_train, y_return)
            
            # Train Random Forest for risk score
            logger.info("Training Random Forest risk model...")
            rf_config = self.config['base_models']['random_forest_risk']
            self.base_models['random_forest_risk'] = RandomForestRegressor(**rf_config)
            self.base_models['random_forest_risk'].fit(self.X_train, y_risk)
            
            # Train LSTM for volatility forecast (if data is sufficient)
            if len(self.X_train) > 100:
                logger.info("Training LSTM volatility model...")
                self.lstm_model = VolatilityLSTMForecaster(self.profile)
                # Note: LSTM training would require sequence data preparation
                # This is simplified for demonstration
                self.base_models['lstm_volatility'] = 'lstm_placeholder'
            
            # Evaluate base models
            self._evaluate_base_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training base models: {e}")
            return False
    
    def _evaluate_base_models(self):
        """Evaluate base models performance"""
        try:
            if self.X_val is None or self.y_val is None:
                return
            
            # Split validation targets
            y_win_val = self.y_val[:, 0]
            y_return_val = self.y_val[:, 1]
            y_risk_val = self.y_val[:, 2]
            y_vol_val = self.y_val[:, 3]
            
            # Evaluate XGBoost
            if self.base_models['xgboost_win'] is not None:
                y_pred_win = self.base_models['xgboost_win'].predict(self.X_val)
                win_accuracy = accuracy_score(y_win_val, y_pred_win)
                self.model_performance['xgboost_win'] = {'accuracy': win_accuracy}
                logger.info(f"XGBoost win accuracy: {win_accuracy:.3f}")
            
            # Evaluate LightGBM
            if self.base_models['lightgbm_return'] is not None:
                y_pred_return = self.base_models['lightgbm_return'].predict(self.X_val)
                return_mse = mean_squared_error(y_return_val, y_pred_return)
                self.model_performance['lightgbm_return'] = {'mse': return_mse}
                logger.info(f"LightGBM return MSE: {return_mse:.6f}")
            
            # Evaluate Random Forest
            if self.base_models['random_forest_risk'] is not None:
                y_pred_risk = self.base_models['random_forest_risk'].predict(self.X_val)
                risk_mse = mean_squared_error(y_risk_val, y_pred_risk)
                self.model_performance['random_forest_risk'] = {'mse': risk_mse}
                logger.info(f"Random Forest risk MSE: {risk_mse:.6f}")
            
        except Exception as e:
            logger.error(f"Error evaluating base models: {e}")
    
    def train_meta_learner(self, data: pd.DataFrame) -> bool:
        """
        Train meta-learner (Level 1)
        
        Args:
            data: Training data
        
        Returns:
            Success status
        """
        try:
            if not all(model is not None for model in self.base_models.values()):
                logger.error("Base models not trained")
                return False
            
            # Generate meta-features from base models
            meta_features = self._generate_meta_features(self.X_train, self.y_train)
            
            if len(meta_features) == 0:
                logger.error("No meta-features generated")
                return False
            
            # Create meta-target (ensemble prediction target)
            meta_target = self._create_meta_target(self.y_train)
            
            # Train meta-learner
            meta_config = self.config['meta_learner']
            
            if meta_config['model_type'] == 'linear':
                from sklearn.linear_model import Ridge
                self.meta_learner = Ridge(alpha=meta_config['regularization'])
            elif meta_config['model_type'] == 'tree':
                self.meta_learner = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )
            else:  # neural
                from sklearn.neural_network import MLPRegressor
                self.meta_learner = MLPRegressor(
                    hidden_layer_sizes=(50, 25),
                    learning_rate_init=meta_config['learning_rate'],
                    max_iter=1000,
                    random_state=42
                )
            
            # Fit meta-learner
            self.meta_learner.fit(meta_features, meta_target)
            
            # Evaluate meta-learner
            meta_features_val = self._generate_meta_features(self.X_val, self.y_val)
            meta_target_val = self._create_meta_target(self.y_val)
            
            if len(meta_features_val) > 0:
                meta_pred = self.meta_learner.predict(meta_features_val)
                meta_mse = mean_squared_error(meta_target_val, meta_pred)
                logger.info(f"Meta-learner MSE: {meta_mse:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training meta-learner: {e}")
            return False
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features from base models"""
        try:
            meta_features_list = []
            
            for i in range(len(X)):
                # Get base model predictions
                x_sample = X[i:i+1]
                
                # XGBoost win prediction
                if self.base_models['xgboost_win'] is not None:
                    win_pred = self.base_models['xgboost_win'].predict_proba(x_sample)[0]
                    win_prob = win_pred[1] if len(win_pred) > 1 else 0.5
                else:
                    win_prob = 0.5
                
                # LightGBM return prediction
                if self.base_models['lightgbm_return'] is not None:
                    return_pred = self.base_models['lightgbm_return'].predict(x_sample)[0]
                else:
                    return_pred = 0.0
                
                # Random Forest risk prediction
                if self.base_models['random_forest_risk'] is not None:
                    risk_pred = self.base_models['random_forest_risk'].predict(x_sample)[0]
                else:
                    risk_pred = 0.0
                
                # LSTM volatility prediction (simplified)
                if self.base_models['lstm_volatility'] is not None:
                    vol_pred = 0.2  # Placeholder
                else:
                    vol_pred = 0.2
                
                # Create meta-features
                meta_features = [
                    win_prob,
                    return_pred,
                    risk_pred,
                    vol_pred,
                    win_prob * return_pred,  # Interaction
                    return_pred / (risk_pred + 1e-8),  # Risk-adjusted return
                    vol_pred * risk_pred  # Volatility-risk interaction
                ]
                
                meta_features_list.append(meta_features)
            
            return np.array(meta_features_list)
            
        except Exception as e:
            logger.error(f"Error generating meta-features: {e}")
            return np.array([])
    
    def _create_meta_target(self, y: np.ndarray) -> np.ndarray:
        """Create meta-target for ensemble"""
        try:
            # Combine targets into single ensemble target
            # Weighted combination of win probability, expected return, and risk
            win_prob = y[:, 0]
            expected_return = y[:, 1]
            risk_score = y[:, 2]
            
            # Risk-adjusted return as meta-target
            meta_target = expected_return / (risk_score + 1e-8)
            
            return meta_target
            
        except Exception as e:
            logger.error(f"Error creating meta-target: {e}")
            return np.array([])
    
    def predict(self, X: np.ndarray, context: Dict[str, Any] = None) -> EnsemblePrediction:
        """
        Make ensemble prediction
        
        Args:
            X: Input features
            context: Optional context (market regime, etc.)
        
        Returns:
            EnsemblePrediction object
        """
        try:
            if self.meta_learner is None:
                logger.error("Meta-learner not trained")
                return self._empty_prediction()
            
            # Scale input features
            X_scaled = self.scaler_X.transform(X.reshape(1, -1))
            
            # Generate meta-features
            meta_features = self._generate_meta_features(X_scaled, np.zeros((1, 4)))
            
            if len(meta_features) == 0:
                return self._empty_prediction()
            
            # Get base model predictions
            individual_predictions = {}
            
            if self.base_models['xgboost_win'] is not None:
                win_pred = self.base_models['xgboost_win'].predict_proba(X_scaled)[0]
                individual_predictions['xgboost_win'] = win_pred[1] if len(win_pred) > 1 else 0.5
            
            if self.base_models['lightgbm_return'] is not None:
                individual_predictions['lightgbm_return'] = self.base_models['lightgbm_return'].predict(X_scaled)[0]
            
            if self.base_models['random_forest_risk'] is not None:
                individual_predictions['random_forest_risk'] = self.base_models['random_forest_risk'].predict(X_scaled)[0]
            
            if self.base_models['lstm_volatility'] is not None:
                individual_predictions['lstm_volatility'] = 0.2  # Placeholder
            
            # Get meta-learner prediction
            ensemble_prediction = self.meta_learner.predict(meta_features)[0]
            
            # Calculate model weights based on performance
            model_weights = self._calculate_model_weights(context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(individual_predictions, model_weights)
            
            # Detect regime
            regime = context.get('regime', 'NORMAL_VOL') if context else 'NORMAL_VOL'
            
            # Feature importance (simplified)
            feature_importance = self._calculate_feature_importance(X_scaled)
            
            return EnsemblePrediction(
                prediction=ensemble_prediction,
                confidence=confidence,
                individual_predictions=individual_predictions,
                model_weights=model_weights,
                regime=regime,
                feature_importance=feature_importance,
                model_version=f"ensemble_v1_{self.profile.tier.value}"
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._empty_prediction()
    
    def _calculate_model_weights(self, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate model weights based on performance and regime"""
        try:
            # Base weights
            base_weights = {
                'xgboost_win': 0.25,
                'lightgbm_return': 0.25,
                'random_forest_risk': 0.25,
                'lstm_volatility': 0.25
            }
            
            # Adjust weights based on model performance
            for model_name, performance in self.model_performance.items():
                if 'accuracy' in performance:
                    # For classification models, higher accuracy = higher weight
                    base_weights[model_name] *= (1 + performance['accuracy'])
                elif 'mse' in performance:
                    # For regression models, lower MSE = higher weight
                    base_weights[model_name] *= (1 - performance['mse'])
            
            # Adjust weights based on regime
            if context and 'regime' in context:
                regime = context['regime']
                
                if regime == 'HIGH_VOL':
                    # Favor volatility and risk models in high vol
                    base_weights['random_forest_risk'] *= 1.2
                    base_weights['lstm_volatility'] *= 1.2
                elif regime == 'LOW_VOL':
                    # Favor directional models in low vol
                    base_weights['xgboost_win'] *= 1.2
                    base_weights['lightgbm_return'] *= 1.2
                elif regime == 'CRISIS':
                    # Equal weights in crisis
                    pass
            
            # Normalize weights
            total_weight = sum(base_weights.values())
            for model_name in base_weights:
                base_weights[model_name] /= total_weight
            
            return base_weights
            
        except Exception as e:
            logger.error(f"Error calculating model weights: {e}")
            return {'xgboost_win': 0.25, 'lightgbm_return': 0.25, 'random_forest_risk': 0.25, 'lstm_volatility': 0.25}
    
    def _calculate_confidence(self, individual_predictions: Dict[str, float], model_weights: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        try:
            # Calculate weighted variance of predictions
            weighted_predictions = []
            for model_name, prediction in individual_predictions.items():
                weight = model_weights.get(model_name, 0.25)
                weighted_predictions.extend([prediction] * int(weight * 100))
            
            if len(weighted_predictions) > 1:
                variance = np.var(weighted_predictions)
                confidence = max(0.1, 1.0 - variance)
            else:
                confidence = 0.5
            
            return min(0.99, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            feature_importance = {}
            
            # Get feature importance from tree-based models
            if self.base_models['xgboost_win'] is not None:
                xgb_importance = self.base_models['xgboost_win'].feature_importances_
                for i, feature in enumerate(self.feature_names):
                    if i < len(xgb_importance):
                        feature_importance[f'xgboost_{feature}'] = float(xgb_importance[i])
            
            if self.base_models['lightgbm_return'] is not None:
                lgb_importance = self.base_models['lightgbm_return'].feature_importances_
                for i, feature in enumerate(self.feature_names):
                    if i < len(lgb_importance):
                        feature_importance[f'lightgbm_{feature}'] = float(lgb_importance[i])
            
            if self.base_models['random_forest_risk'] is not None:
                rf_importance = self.base_models['random_forest_risk'].feature_importances_
                for i, feature in enumerate(self.feature_names):
                    if i < len(rf_importance):
                        feature_importance[f'randomforest_{feature}'] = float(rf_importance[i])
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def evaluate_ensemble(self) -> EnsembleMetrics:
        """Evaluate ensemble performance"""
        try:
            if self.X_val is None or self.y_val is None or self.meta_learner is None:
                return self._empty_metrics()
            
            # Generate predictions
            predictions = []
            for i in range(len(self.X_val)):
                pred = self.predict(self.X_val[i])
                predictions.append(pred.prediction)
            
            predictions = np.array(predictions)
            
            # Create ensemble target
            meta_target = self._create_meta_target(self.y_val)
            
            # Calculate metrics
            mse = mean_squared_error(meta_target, predictions)
            mae = np.mean(np.abs(meta_target - predictions))
            rmse = np.sqrt(mse)
            
            # R² score
            ss_res = np.sum((meta_target - predictions) ** 2)
            ss_tot = np.sum((meta_target - np.mean(meta_target)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Directional accuracy
            direction_true = np.diff(meta_target) > 0
            direction_pred = np.diff(predictions) > 0
            directional_accuracy = np.mean(direction_true == direction_pred) if len(direction_true) > 0 else 0
            
            # Sharpe ratio (simplified)
            returns = predictions - meta_target
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Model correlations
            model_correlations = {}
            if len(predictions) > 1:
                for i, model_name in enumerate(['ensemble']):
                    model_correlations[model_name] = 1.0  # Self-correlation
            
            # Ensemble diversity (correlation between base models)
            ensemble_diversity = 0.5  # Placeholder
            
            return EnsembleMetrics(
                mse=mse,
                mae=mae,
                rmse=rmse,
                r2=r2,
                directional_accuracy=directional_accuracy,
                sharpe_ratio=sharpe_ratio,
                model_correlations=model_correlations,
                ensemble_diversity=ensemble_diversity
            )
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return self._empty_metrics()
    
    def _empty_prediction(self) -> EnsemblePrediction:
        """Return empty prediction"""
        return EnsemblePrediction(
            prediction=0.0,
            confidence=0.0,
            individual_predictions={},
            model_weights={},
            regime='NORMAL_VOL',
            feature_importance={},
            model_version="empty"
        )
    
    def _empty_metrics(self) -> EnsembleMetrics:
        """Return empty metrics"""
        return EnsembleMetrics(0, 0, 0, 0, 0, 0, {}, 0)
    
    def save_model(self, model_path: str = None) -> bool:
        """Save ensemble model"""
        try:
            if model_path is None:
                model_path = f"models/ensemble_{self.profile.tier.value}.pkl"
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'base_models': self.base_models,
                'meta_learner': self.meta_learner,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'model_performance': self.model_performance,
                'config': self.config,
                'account_profile': self.profile
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Ensemble model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str = None) -> bool:
        """Load ensemble model"""
        try:
            if model_path is None:
                model_path = f"models/ensemble_{self.profile.tier.value}.pkl"
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore components
            self.base_models = model_data['base_models']
            self.meta_learner = model_data['meta_learner']
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.feature_names = model_data['feature_names']
            self.model_performance = model_data['model_performance']
            self.config = model_data['config']
            
            logger.info(f"Ensemble model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create ensemble
    ensemble = OptionsEnsemble(profile)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(n_days) * 0.01),
        'volume': np.random.randint(1000, 10000, n_days),
        'high': 100 + np.cumsum(np.random.randn(n_days) * 0.01) + np.random.rand(n_days) * 2,
        'low': 100 + np.cumsum(np.random.randn(n_days) * 0.01) - np.random.rand(n_days) * 2,
        'open': 100 + np.cumsum(np.random.randn(n_days) * 0.01),
        'iv': np.random.uniform(0.15, 0.35, n_days),
        'delta': np.random.uniform(0.3, 0.7, n_days),
        'gamma': np.random.uniform(0.005, 0.02, n_days),
        'theta': np.random.uniform(-0.05, -0.01, n_days),
        'vega': np.random.uniform(0.05, 0.2, n_days),
        'put_call_ratio': np.random.uniform(0.8, 1.2, n_days),
        'iv_rank': np.random.uniform(0.2, 0.8, n_days),
        'skew': np.random.uniform(-0.5, 0.5, n_days),
        'term_structure': np.random.uniform(-0.2, 0.2, n_days),
        'vix': np.random.uniform(15, 35, n_days)
    })
    
    print("Training Ensemble Models...")
    
    # Train base models
    success = ensemble.train_base_models(sample_data)
    if success:
        print("Base models trained successfully")
    
    # Train meta-learner
    success = ensemble.train_meta_learner(sample_data)
    if success:
        print("Meta-learner trained successfully")
    
    # Evaluate ensemble
    metrics = ensemble.evaluate_ensemble()
    
    print(f"\nEnsemble Performance:")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    print(f"R²: {metrics.r2:.4f}")
    print(f"Directional Accuracy: {metrics.directional_accuracy:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Ensemble Diversity: {metrics.ensemble_diversity:.3f}")
    
    # Test prediction
    test_sample = sample_data.iloc[-1][ensemble.feature_names].values
    context = {'regime': 'NORMAL_VOL', 'iv_rank': 0.6}
    
    prediction = ensemble.predict(test_sample, context)
    
    print(f"\nEnsemble Prediction:")
    print(f"Prediction: {prediction.prediction:.4f}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print(f"Regime: {prediction.regime}")
    print(f"Model Version: {prediction.model_version}")
    
    print(f"\nIndividual Model Predictions:")
    for model, pred in prediction.individual_predictions.items():
        print(f"  {model}: {pred:.4f}")
    
    print(f"\nModel Weights:")
    for model, weight in prediction.model_weights.items():
        print(f"  {model}: {weight:.3f}")
    
    print(f"\nTop Feature Importance:")
    sorted_features = sorted(prediction.feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.4f}")
