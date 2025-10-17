"""
Transformer Price Predictor
Multi-variate time series prediction with cross-asset attention
Cross-asset attention (SPY, VIX, QQQ correlations) and temporal encoding
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
import math

from src.portfolio.account_manager import AccountProfile


@dataclass
class PricePrediction:
    """Price prediction result"""
    symbol: str
    prediction_date: datetime
    predicted_price: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: int  # days
    cross_asset_correlations: Dict[str, float]
    temporal_features: Dict[str, float]
    model_version: str


@dataclass
class TransformerMetrics:
    """Transformer model performance metrics"""
    mse: float
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal features"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class CrossAssetAttention(nn.Module):
    """Cross-asset attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super(CrossAssetAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.out_linear(context)
        
        return output, attention_weights


class OptionsTransformer(nn.Module):
    """Transformer model for options price prediction"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super(OptionsTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Cross-asset attention
        self.cross_asset_attention = CrossAssetAttention(d_model, num_heads)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cross_asset_features=None):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Cross-asset attention if provided
        if cross_asset_features is not None:
            cross_features = self.input_projection(cross_asset_features)
            x, attention_weights = self.cross_asset_attention(x, cross_features, cross_features)
        else:
            attention_weights = None
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output, attention_weights


class PriceTransformer:
    """
    Transformer-based price predictor for options trading
    
    Features:
    - Multi-variate time series (price, volume, IV, Greeks)
    - Cross-asset attention (SPY, VIX, QQQ correlations)
    - Temporal encoding for cyclical patterns
    - Works across all account sizes
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
        self.cross_asset_features = []
        
        # Training data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.cross_asset_train = None
        self.cross_asset_val = None
        
        # Performance tracking
        self.training_history = []
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"PriceTransformer initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default model configuration"""
        return {
            'sequence_length': 60,  # 60 days of lookback
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 15,
            'min_delta': 1e-6,
            'validation_split': 0.2,
            'prediction_horizons': [1, 3, 7],  # 1D, 3D, 7D predictions
            'features': [
                'close', 'volume', 'high', 'low', 'open',
                'returns', 'volatility', 'rsi', 'macd', 'bb_upper', 'bb_lower',
                'iv', 'delta', 'gamma', 'theta', 'vega',
                'put_call_ratio', 'iv_rank', 'skew', 'term_structure'
            ],
            'cross_asset_symbols': ['SPY', 'QQQ', 'VIX', 'TLT', 'GLD'],
            'cross_asset_features': ['close', 'volume', 'returns', 'volatility']
        }
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        cross_asset_data: Dict[str, pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Transformer training
        
        Args:
            data: Main options/underlying data
            cross_asset_data: Cross-asset data for attention
        
        Returns:
            X, y arrays for training
        """
        try:
            # Ensure data is sorted by date
            data = data.sort_values('date').reset_index(drop=True)
            
            # Feature engineering
            features_df = self._engineer_features(data)
            
            # Cross-asset feature engineering
            if cross_asset_data:
                cross_features_df = self._engineer_cross_asset_features(cross_asset_data)
            else:
                cross_features_df = None
            
            # Select features
            feature_cols = [col for col in self.config['features'] if col in features_df.columns]
            self.feature_names = feature_cols
            
            # Create sequences
            X, y = self._create_sequences(features_df, feature_cols)
            
            # Create cross-asset sequences
            if cross_features_df is not None:
                cross_X = self._create_cross_asset_sequences(cross_features_df)
            else:
                cross_X = None
            
            # Split into train/validation
            split_idx = int(len(X) * (1 - self.config['validation_split']))
            self.X_train = X[:split_idx]
            self.y_train = y[:split_idx]
            self.X_val = X[split_idx:]
            self.y_val = y[split_idx:]
            
            if cross_X is not None:
                self.cross_asset_train = cross_X[:split_idx]
                self.cross_asset_val = cross_X[split_idx:]
            else:
                self.cross_asset_train = None
                self.cross_asset_val = None
            
            logger.info(f"Prepared data: {len(self.X_train)} train, {len(self.X_val)} val samples")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return np.array([]), np.array([])
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for price prediction"""
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
                features_df['iv'] = 0.2  # Default IV
            if 'delta' not in features_df.columns:
                features_df['delta'] = 0.5  # Default delta
            if 'gamma' not in features_df.columns:
                features_df['gamma'] = 0.01  # Default gamma
            if 'theta' not in features_df.columns:
                features_df['theta'] = -0.01  # Default theta
            if 'vega' not in features_df.columns:
                features_df['vega'] = 0.1  # Default vega
            
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
            
            # Volatility of volatility
            features_df['vol_of_vol'] = features_df['volatility'].rolling(window=20).std()
            
            # Temporal features
            features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
            features_df['month'] = pd.to_datetime(features_df['date']).dt.month
            features_df['quarter'] = pd.to_datetime(features_df['date']).dt.quarter
            
            # Cyclical encoding for temporal features
            features_df['day_of_week_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_of_week_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return data
    
    def _engineer_cross_asset_features(self, cross_asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Engineer cross-asset features"""
        try:
            cross_features_list = []
            
            for symbol, asset_data in cross_asset_data.items():
                if symbol in self.config['cross_asset_symbols']:
                    asset_features = asset_data.copy()
                    
                    # Calculate returns and volatility
                    asset_features['returns'] = asset_features['close'].pct_change()
                    asset_features['volatility'] = asset_features['returns'].rolling(window=20).std() * np.sqrt(252)
                    
                    # Select relevant features
                    feature_cols = [col for col in self.config['cross_asset_features'] 
                                  if col in asset_features.columns]
                    
                    # Rename columns with symbol prefix
                    for col in feature_cols:
                        asset_features[f'{symbol}_{col}'] = asset_features[col]
                    
                    # Select renamed columns
                    symbol_features = asset_features[[f'{symbol}_{col}' for col in feature_cols]]
                    cross_features_list.append(symbol_features)
            
            if cross_features_list:
                # Combine all cross-asset features
                cross_features_df = pd.concat(cross_features_list, axis=1)
                cross_features_df = cross_features_df.fillna(method='ffill').fillna(0)
                return cross_features_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error engineering cross-asset features: {e}")
            return pd.DataFrame()
    
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
    
    def _create_sequences(self, data: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for Transformer training"""
        try:
            sequence_length = self.config['sequence_length']
            
            # Extract features
            features = data[feature_cols].values
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features)):
                # Input sequence
                X.append(features[i-sequence_length:i])
                
                # Target: next-day price
                if i + 1 < len(data):
                    target_price = data['close'].iloc[i + 1]
                    y.append(target_price)
                else:
                    y.append(data['close'].iloc[i])  # Use current price if no future data
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            return X_scaled, y_scaled
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def _create_cross_asset_sequences(self, cross_features_df: pd.DataFrame) -> np.ndarray:
        """Create cross-asset sequences"""
        try:
            sequence_length = self.config['sequence_length']
            
            # Extract cross-asset features
            cross_features = cross_features_df.values
            
            # Create sequences
            cross_X = []
            for i in range(sequence_length, len(cross_features)):
                cross_X.append(cross_features[i-sequence_length:i])
            
            cross_X = np.array(cross_X)
            
            # Scale cross-asset features
            cross_X_scaled = self.scaler_X.transform(cross_X.reshape(-1, cross_X.shape[-1])).reshape(cross_X.shape)
            
            return cross_X_scaled
            
        except Exception as e:
            logger.error(f"Error creating cross-asset sequences: {e}")
            return np.array([])
    
    def train_model(
        self,
        data: pd.DataFrame,
        cross_asset_data: Dict[str, pd.DataFrame] = None
    ) -> TransformerMetrics:
        """
        Train the Transformer model
        
        Args:
            data: Training data
            cross_asset_data: Cross-asset data for attention
        
        Returns:
            Training metrics
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data, cross_asset_data)
            
            if len(X) == 0:
                logger.error("No data available for training")
                return self._empty_metrics()
            
            # Initialize model
            input_size = len(self.feature_names)
            self.model = OptionsTransformer(
                input_size=input_size,
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=0.01
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['epochs'], eta_min=1e-7
            )
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(self.X_train),
                torch.FloatTensor(self.y_train)
            )
            if self.cross_asset_train is not None:
                train_dataset = TensorDataset(
                    torch.FloatTensor(self.X_train),
                    torch.FloatTensor(self.y_train),
                    torch.FloatTensor(self.cross_asset_train)
                )
            
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            val_dataset = TensorDataset(
                torch.FloatTensor(self.X_val),
                torch.FloatTensor(self.y_val)
            )
            if self.cross_asset_val is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(self.X_val),
                    torch.FloatTensor(self.y_val),
                    torch.FloatTensor(self.cross_asset_val)
                )
            
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            # Training loop
            patience_counter = 0
            best_val_loss = float('inf')
            
            for epoch in range(self.config['epochs']):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    if len(batch) == 3:  # With cross-asset features
                        batch_X, batch_y, batch_cross = batch
                        batch_X, batch_y, batch_cross = batch_X.to(self.device), batch_y.to(self.device), batch_cross.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs, _ = self.model(batch_X, batch_cross)
                        loss = criterion(outputs.squeeze(), batch_y)
                    else:  # Without cross-asset features
                        batch_X, batch_y = batch
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs, _ = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 3:  # With cross-asset features
                            batch_X, batch_y, batch_cross = batch
                            batch_X, batch_y, batch_cross = batch_X.to(self.device), batch_y.to(self.device), batch_cross.to(self.device)
                            
                            outputs, _ = self.model(batch_X, batch_cross)
                            loss = criterion(outputs.squeeze(), batch_y)
                        else:  # Without cross-asset features
                            batch_X, batch_y = batch
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            
                            outputs, _ = self.model(batch_X)
                            loss = criterion(outputs.squeeze(), batch_y)
                        
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # Learning rate scheduling
                scheduler.step()
                
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
    
    def evaluate_model(self) -> TransformerMetrics:
        """Evaluate model performance"""
        try:
            if self.model is None or self.X_val is None or self.y_val is None:
                return self._empty_metrics()
            
            self.model.eval()
            
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
                cross_val_tensor = torch.FloatTensor(self.cross_asset_val).to(self.device) if self.cross_asset_val is not None else None
                
                if cross_val_tensor is not None:
                    y_pred, _ = self.model(X_val_tensor, cross_val_tensor)
                else:
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
            
            # Directional accuracy
            direction_true = np.diff(y_true_original) > 0
            direction_pred = np.diff(y_pred_original) > 0
            directional_accuracy = np.mean(direction_true == direction_pred) if len(direction_true) > 0 else 0
            
            # Sharpe ratio (simplified)
            returns = y_pred_original - y_true_original
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Max drawdown
            cumulative_returns = (1 + returns / y_true_original).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return TransformerMetrics(
                mse=mse,
                mae=mae,
                rmse=rmse,
                r2=r2,
                directional_accuracy=directional_accuracy,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return self._empty_metrics()
    
    def predict_price(
        self,
        data: pd.DataFrame,
        symbol: str,
        prediction_date: datetime,
        horizon: int = 1,
        cross_asset_data: Dict[str, pd.DataFrame] = None
    ) -> PricePrediction:
        """
        Predict price for a specific symbol and date
        
        Args:
            data: Recent market data
            symbol: Symbol to predict
            prediction_date: Date for prediction
            horizon: Prediction horizon in days
            cross_asset_data: Cross-asset data for attention
        
        Returns:
            PricePrediction object
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
            
            # Prepare cross-asset features if available
            cross_features_scaled = None
            cross_asset_correlations = {}
            
            if cross_asset_data:
                cross_features_df = self._engineer_cross_asset_features(cross_asset_data)
                if not cross_features_df.empty:
                    cross_features = cross_features_df.values[-sequence_length:]
                    cross_features_scaled = self.scaler_X.transform(cross_features.reshape(1, -1)).reshape(1, sequence_length, -1)
                    
                    # Calculate cross-asset correlations
                    for cross_symbol in self.config['cross_asset_symbols']:
                        if f'{cross_symbol}_close' in cross_features_df.columns:
                            correlation = np.corrcoef(
                                features_df['close'].values[-sequence_length:],
                                cross_features_df[f'{cross_symbol}_close'].values[-sequence_length:]
                            )[0, 1]
                            cross_asset_correlations[cross_symbol] = correlation if not np.isnan(correlation) else 0
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(features_scaled).to(self.device)
                cross_tensor = torch.FloatTensor(cross_features_scaled).to(self.device) if cross_features_scaled is not None else None
                
                if cross_tensor is not None:
                    prediction, attention_weights = self.model(X_tensor, cross_tensor)
                else:
                    prediction, attention_weights = self.model(X_tensor)
                
                prediction_scaled = prediction.squeeze().cpu().numpy()
            
            # Inverse transform prediction
            prediction_original = self.scaler_y.inverse_transform([[prediction_scaled]])[0][0]
            
            # Calculate confidence interval (simplified)
            confidence_interval = (
                prediction_original * 0.95,  # Lower bound
                prediction_original * 1.05   # Upper bound
            )
            
            # Temporal features
            temporal_features = {
                'day_of_week': features_df['day_of_week'].iloc[-1] if 'day_of_week' in features_df.columns else 0,
                'month': features_df['month'].iloc[-1] if 'month' in features_df.columns else 0,
                'quarter': features_df['quarter'].iloc[-1] if 'quarter' in features_df.columns else 0
            }
            
            return PricePrediction(
                symbol=symbol,
                prediction_date=prediction_date,
                predicted_price=prediction_original,
                confidence_interval=confidence_interval,
                prediction_horizon=horizon,
                cross_asset_correlations=cross_asset_correlations,
                temporal_features=temporal_features,
                model_version=f"transformer_v1_{self.profile.tier.value}"
            )
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return self._empty_prediction(symbol, prediction_date)
    
    def save_model(self, model_path: str = None) -> bool:
        """Save trained model"""
        try:
            if model_path is None:
                model_path = f"models/price_transformer_{self.profile.tier.value}.pkl"
            
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
                model_path = f"models/price_transformer_{self.profile.tier.value}.pkl"
            
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
            self.model = OptionsTransformer(
                input_size=input_size,
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
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
    
    def _empty_metrics(self) -> TransformerMetrics:
        """Return empty metrics"""
        return TransformerMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def _empty_prediction(self, symbol: str, prediction_date: datetime) -> PricePrediction:
        """Return empty prediction"""
        return PricePrediction(
            symbol=symbol,
            prediction_date=prediction_date,
            predicted_price=0.0,
            confidence_interval=(0.0, 0.0),
            prediction_horizon=1,
            cross_asset_correlations={},
            temporal_features={},
            model_version="empty"
        )


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create Transformer predictor
    predictor = PriceTransformer(profile)
    
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
        'term_structure': np.random.uniform(-0.2, 0.2, n_days)
    })
    
    # Sample cross-asset data
    cross_asset_data = {
        'SPY': sample_data.copy(),
        'QQQ': sample_data.copy(),
        'VIX': sample_data.copy(),
        'TLT': sample_data.copy(),
        'GLD': sample_data.copy()
    }
    
    print("Training Transformer price predictor...")
    metrics = predictor.train_model(sample_data, cross_asset_data)
    
    print(f"\nTraining Results:")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    print(f"R²: {metrics.r2:.4f}")
    print(f"Directional Accuracy: {metrics.directional_accuracy:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    
    # Make prediction
    prediction = predictor.predict_price(
        data=sample_data.tail(100),
        symbol='SPY',
        prediction_date=datetime.now(),
        horizon=1,
        cross_asset_data={k: v.tail(100) for k, v in cross_asset_data.items()}
    )
    
    print(f"\nPrice Prediction for SPY:")
    print(f"Predicted Price: ${prediction.predicted_price:.2f}")
    print(f"Confidence Interval: ${prediction.confidence_interval[0]:.2f} - ${prediction.confidence_interval[1]:.2f}")
    print(f"Horizon: {prediction.prediction_horizon} day(s)")
    print(f"Model Version: {prediction.model_version}")
    print(f"Cross-Asset Correlations: {prediction.cross_asset_correlations}")
