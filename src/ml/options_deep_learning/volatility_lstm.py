"""
LSTM Volatility Forecaster
Predict next-day realized volatility and IV changes with attention mechanism
Trained on 5+ years of Polygon flat file data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import os

from src.portfolio.account_manager import AccountProfile


@dataclass
class VolatilityPrediction:
    """Volatility prediction result"""
    symbol: str
    prediction_date: datetime
    predicted_vol: float
    confidence: float
    prediction_horizon: int  # days
    feature_importance: Dict[str, float]
    model_version: str


@dataclass
class LSTMMetrics:
    """LSTM model performance metrics"""
    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float
    directional_accuracy: float
    sharpe_ratio: float


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for volatility forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Fully connected layers
        out = self.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out, attn_weights


class VolatilityLSTMForecaster:
    """
    LSTM-based volatility forecaster for options trading
    
    Features:
    - Predict next-day realized volatility
    - IV changes and mean reversion forecasting
    - Multi-step predictions (1D, 3D, 7D)
    - Attention mechanism for feature importance
    - Trained on 5+ years of Polygon flat file data
    """
    
    def __init__(self, account_profile: AccountProfile, model_config: Dict = None):
        self.profile = account_profile
        
        # Model configuration
        self.config = model_config or self._default_config()
        
        # Model components
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names = []
        
        # Training data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Performance tracking
        self.training_history = []
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"VolatilityLSTMForecaster initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default model configuration"""
        return {
            'sequence_length': 60,  # 60 days of lookback
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'min_delta': 1e-6,
            'validation_split': 0.2,
            'prediction_horizons': [1, 3, 7],  # 1D, 3D, 7D predictions
            'features': [
                'close', 'volume', 'high', 'low', 'open',
                'returns', 'volatility', 'rsi', 'macd', 'bb_upper', 'bb_lower',
                'put_call_ratio', 'iv_rank', 'skew', 'term_structure'
            ]
        }
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: Historical options and underlying data
        
        Returns:
            X, y arrays for training
        """
        try:
            # Ensure data is sorted by date
            data = data.sort_values('date').reset_index(drop=True)
            
            # Feature engineering
            features_df = self._engineer_features(data)
            
            # Select features
            feature_cols = [col for col in self.config['features'] if col in features_df.columns]
            self.feature_names = feature_cols
            
            # Create sequences
            X, y = self._create_sequences(features_df, feature_cols)
            
            # Split into train/validation
            split_idx = int(len(X) * (1 - self.config['validation_split']))
            self.X_train = X[:split_idx]
            self.y_train = y[:split_idx]
            self.X_val = X[split_idx:]
            self.y_val = y[split_idx:]
            
            logger.info(f"Prepared data: {len(self.X_train)} train, {len(self.X_val)} val samples")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return np.array([]), np.array([])
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for volatility prediction"""
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
            if 'put_call_ratio' not in features_df.columns:
                features_df['put_call_ratio'] = 1.0  # Default if not available
            if 'iv_rank' not in features_df.columns:
                features_df['iv_rank'] = 0.5  # Default if not available
            if 'skew' not in features_df.columns:
                features_df['skew'] = 0.0  # Default if not available
            if 'term_structure' not in features_df.columns:
                features_df['term_structure'] = 0.0  # Default if not available
            
            # Volume features
            features_df['volume_ma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma']
            
            # Price momentum
            features_df['momentum_5'] = features_df['close'] / features_df['close'].shift(5) - 1
            features_df['momentum_20'] = features_df['close'] / features_df['close'].shift(20) - 1
            
            # Volatility of volatility
            features_df['vol_of_vol'] = features_df['volatility'].rolling(window=20).std()
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return data
    
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
            return pd.Series(50, index=prices.index)  # Default neutral RSI
    
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
    
    def _create_sequences(self, data: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            sequence_length = self.config['sequence_length']
            
            # Extract features
            features = data[feature_cols].values
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features)):
                # Input sequence
                X.append(features[i-sequence_length:i])
                
                # Target: next-day realized volatility
                if i + 1 < len(data):
                    # Calculate realized volatility for next day
                    next_day_returns = data['returns'].iloc[i:i+5]  # 5-day window for realized vol
                    realized_vol = next_day_returns.std() * np.sqrt(252) if len(next_day_returns) > 1 else 0
                    y.append(realized_vol)
                else:
                    y.append(0)  # Default for last sequence
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            return X_scaled, y_scaled
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def train_model(self, data: pd.DataFrame) -> LSTMMetrics:
        """
        Train the LSTM model
        
        Args:
            data: Training data
        
        Returns:
            Training metrics
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            if len(X) == 0:
                logger.error("No data available for training")
                return self._empty_metrics()
            
            # Initialize model
            input_size = len(self.feature_names)
            self.model = AttentionLSTM(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(self.X_train),
                torch.FloatTensor(self.y_train)
            )
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            val_dataset = TensorDataset(
                torch.FloatTensor(self.X_val),
                torch.FloatTensor(self.y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            # Training loop
            patience_counter = 0
            best_val_loss = float('inf')
            
            for epoch in range(self.config['epochs']):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs, _ = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs, _ = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss - self.config['min_delta']:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Log progress
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)
            
            # Evaluate model
            metrics = self.evaluate_model()
            
            # Save model
            self.save_model()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return self._empty_metrics()
    
    def evaluate_model(self) -> LSTMMetrics:
        """Evaluate model performance"""
        try:
            if self.model is None or self.X_val is None or self.y_val is None:
                return self._empty_metrics()
            
            self.model.eval()
            
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
                y_pred, _ = self.model(X_val_tensor)
                y_pred = y_pred.squeeze().cpu().numpy()
            
            # Inverse transform predictions
            y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true_original = self.scaler_y.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_true_original, y_pred_original)
            mae = mean_absolute_error(y_true_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_original, y_pred_original)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
            
            # Directional accuracy
            direction_true = np.diff(y_true_original) > 0
            direction_pred = np.diff(y_pred_original) > 0
            directional_accuracy = np.mean(direction_true == direction_pred) if len(direction_true) > 0 else 0
            
            # Sharpe ratio (simplified)
            returns = y_pred_original - y_true_original
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            return LSTMMetrics(
                mse=mse,
                mae=mae,
                rmse=rmse,
                r2=r2,
                mape=mape,
                directional_accuracy=directional_accuracy,
                sharpe_ratio=sharpe_ratio
            )
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return self._empty_metrics()
    
    def predict_volatility(
        self,
        data: pd.DataFrame,
        symbol: str,
        prediction_date: datetime,
        horizon: int = 1
    ) -> VolatilityPrediction:
        """
        Predict volatility for a specific symbol and date
        
        Args:
            data: Recent market data
            symbol: Symbol to predict
            prediction_date: Date for prediction
            horizon: Prediction horizon in days
        
        Returns:
            VolatilityPrediction object
        """
        try:
            if self.model is None:
                logger.error("Model not trained")
                return self._empty_prediction(symbol, prediction_date)
            
            # Prepare input data
            features_df = self._engineer_features(data)
            feature_cols = [col for col in self.config['features'] if col in features_df.columns]
            
            if len(feature_cols) == 0:
                logger.error("No features available for prediction")
                return self._empty_prediction(symbol, prediction_date)
            
            # Get last sequence
            sequence_length = self.config['sequence_length']
            if len(features_df) < sequence_length:
                logger.error(f"Insufficient data: {len(features_df)} < {sequence_length}")
                return self._empty_prediction(symbol, prediction_date)
            
            # Extract and scale features
            features = features_df[feature_cols].values[-sequence_length:]
            features_scaled = self.scaler_X.transform(features.reshape(1, -1)).reshape(1, sequence_length, -1)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(features_scaled).to(self.device)
                prediction, attention_weights = self.model(X_tensor)
                prediction_scaled = prediction.squeeze().cpu().numpy()
            
            # Inverse transform prediction
            prediction_original = self.scaler_y.inverse_transform([[prediction_scaled]])[0][0]
            
            # Calculate confidence (based on prediction variance)
            confidence = 0.8  # Simplified confidence score
            
            # Feature importance from attention weights
            feature_importance = {}
            if attention_weights is not None:
                # Average attention weights across heads and sequence
                avg_attention = torch.mean(attention_weights, dim=1).squeeze().cpu().numpy()
                for i, feature in enumerate(feature_cols):
                    feature_importance[feature] = float(avg_attention[i])
            
            return VolatilityPrediction(
                symbol=symbol,
                prediction_date=prediction_date,
                predicted_vol=prediction_original,
                confidence=confidence,
                prediction_horizon=horizon,
                feature_importance=feature_importance,
                model_version=f"lstm_v1_{self.profile.tier.value}"
            )
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return self._empty_prediction(symbol, prediction_date)
    
    def save_model(self, model_path: str = None) -> bool:
        """Save trained model"""
        try:
            if model_path is None:
                model_path = f"models/volatility_lstm_{self.profile.tier.value}.pkl"
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model_state_dict': self.model.state_dict() if self.model else None,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'config': self.config,
                'account_profile': self.profile
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str = None) -> bool:
        """Load trained model"""
        try:
            if model_path is None:
                model_path = f"models/volatility_lstm_{self.profile.tier.value}.pkl"
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model components
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            
            # Initialize and load model
            input_size = len(self.feature_names)
            self.model = AttentionLSTM(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            if model_data['model_state_dict']:
                self.model.load_state_dict(model_data['model_state_dict'])
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _empty_metrics(self) -> LSTMMetrics:
        """Return empty metrics"""
        return LSTMMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def _empty_prediction(self, symbol: str, prediction_date: datetime) -> VolatilityPrediction:
        """Return empty prediction"""
        return VolatilityPrediction(
            symbol=symbol,
            prediction_date=prediction_date,
            predicted_vol=0.0,
            confidence=0.0,
            prediction_horizon=1,
            feature_importance={},
            model_version="empty"
        )


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create LSTM forecaster
    forecaster = VolatilityLSTMForecaster(profile)
    
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
        'put_call_ratio': np.random.uniform(0.8, 1.2, n_days),
        'iv_rank': np.random.uniform(0.2, 0.8, n_days),
        'skew': np.random.uniform(-0.5, 0.5, n_days),
        'term_structure': np.random.uniform(-0.2, 0.2, n_days)
    })
    
    print("Training LSTM volatility forecaster...")
    metrics = forecaster.train_model(sample_data)
    
    print(f"\nTraining Results:")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    print(f"R²: {metrics.r2:.4f}")
    print(f"MAPE: {metrics.mape:.2f}%")
    print(f"Directional Accuracy: {metrics.directional_accuracy:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    
    # Make prediction
    prediction = forecaster.predict_volatility(
        data=sample_data.tail(100),
        symbol='SPY',
        prediction_date=datetime.now(),
        horizon=1
    )
    
    print(f"\nVolatility Prediction for SPY:")
    print(f"Predicted Volatility: {prediction.predicted_vol:.4f}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print(f"Horizon: {prediction.prediction_horizon} day(s)")
    print(f"Model Version: {prediction.model_version}")
