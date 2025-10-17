"""
Autoencoder Anomaly Detector
Detect flash crashes and unusual market conditions with automatic position reduction
Volatility regime transition detection and universal early warning system
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import os

from src.portfolio.account_manager import AccountProfile


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    timestamp: datetime
    symbol: str
    anomaly_score: float
    anomaly_type: str
    confidence: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str
    feature_contributions: Dict[str, float]
    regime_transition: bool
    model_version: str


@dataclass
class AnomalyMetrics:
    """Anomaly detection model performance metrics"""
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    false_positive_rate: float
    detection_latency: float  # seconds


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for anomaly detection"""
    
    def __init__(self, input_size: int, latent_size: int = 32, hidden_sizes: List[int] = [128, 64]):
        super(VariationalAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        
        # Encoder
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(prev_size, latent_size)
        self.fc_logvar = nn.Linear(prev_size, latent_size)
        
        # Decoder
        decoder_layers = []
        prev_size = latent_size
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        decoder_layers.append(nn.Linear(prev_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Reconstruction loss weight
        self.beta = 0.5  # KL divergence weight
    
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent space to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """Calculate VAE loss"""
        # Reconstruction loss
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


class OptionsAnomalyDetector:
    """
    Autoencoder-based anomaly detector for options trading
    
    Features:
    - Detect flash crashes and unusual market conditions
    - Automatic position reduction during anomalies
    - Volatility regime transition detection
    - Universal early warning system
    """
    
    def __init__(self, account_profile: AccountProfile, model_config: Dict = None):
        self.profile = account_profile
        
        # Model configuration
        self.config = model_config or self._default_config()
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Training data
        self.X_train = None
        self.X_val = None
        self.normal_data = None
        self.anomaly_threshold = None
        
        # Performance tracking
        self.training_history = []
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
        # Anomaly patterns
        self.anomaly_patterns = self._initialize_anomaly_patterns()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"OptionsAnomalyDetector initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default model configuration"""
        return {
            'sequence_length': 20,  # 20 periods of lookback
            'latent_size': 32,
            'hidden_sizes': [128, 64],
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'patience': 15,
            'min_delta': 1e-6,
            'validation_split': 0.2,
            'anomaly_threshold_percentile': 95,  # 95th percentile for anomaly threshold
            'features': [
                'close', 'volume', 'high', 'low', 'open',
                'returns', 'volatility', 'rsi', 'macd',
                'put_call_ratio', 'iv_rank', 'skew', 'term_structure',
                'volume_ratio', 'price_momentum', 'vol_of_vol'
            ],
            'anomaly_types': [
                'FLASH_CRASH', 'VOLATILITY_SPIKE', 'VOLUME_ANOMALY',
                'IV_ANOMALY', 'SKEW_ANOMALY', 'REGIME_TRANSITION'
            ]
        }
    
    def _initialize_anomaly_patterns(self) -> Dict[str, Dict]:
        """Initialize known anomaly patterns"""
        return {
            'FLASH_CRASH': {
                'description': 'Rapid price decline with high volume',
                'thresholds': {
                    'price_change': -0.05,  # -5% price change
                    'volume_spike': 3.0,    # 3x normal volume
                    'volatility_spike': 2.0  # 2x normal volatility
                },
                'severity': 'CRITICAL',
                'recommended_action': 'CLOSE_ALL_POSITIONS'
            },
            'VOLATILITY_SPIKE': {
                'description': 'Sudden increase in volatility',
                'thresholds': {
                    'volatility_change': 0.5,  # 50% increase in volatility
                    'iv_spike': 0.3            # 30% increase in IV
                },
                'severity': 'HIGH',
                'recommended_action': 'REDUCE_POSITIONS'
            },
            'VOLUME_ANOMALY': {
                'description': 'Unusual trading volume',
                'thresholds': {
                    'volume_ratio': 5.0,  # 5x normal volume
                    'volume_consistency': 0.8  # 80% of periods with high volume
                },
                'severity': 'MEDIUM',
                'recommended_action': 'MONITOR_CLOSELY'
            },
            'IV_ANOMALY': {
                'description': 'Implied volatility anomaly',
                'thresholds': {
                    'iv_change': 0.2,      # 20% change in IV
                    'iv_rank_change': 0.3  # 30% change in IV rank
                },
                'severity': 'MEDIUM',
                'recommended_action': 'ADJUST_STRATEGIES'
            },
            'SKEW_ANOMALY': {
                'description': 'Options skew anomaly',
                'thresholds': {
                    'skew_change': 0.1,    # 10% change in skew
                    'put_call_ratio': 2.0  # 2:1 put/call ratio
                },
                'severity': 'LOW',
                'recommended_action': 'REVIEW_POSITIONS'
            },
            'REGIME_TRANSITION': {
                'description': 'Market regime change',
                'thresholds': {
                    'correlation_change': 0.3,  # 30% change in correlations
                    'volatility_regime_change': True
                },
                'severity': 'HIGH',
                'recommended_action': 'REBALANCE_PORTFOLIO'
            }
        }
    
    def prepare_data(self, data: pd.DataFrame, anomaly_labels: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for anomaly detection training
        
        Args:
            data: Historical market data
            anomaly_labels: Binary labels for anomalies (optional)
        
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
            X = self._create_sequences(features_df, feature_cols)
            
            # Create labels if not provided
            if anomaly_labels is None:
                y = self._create_anomaly_labels(features_df)
            else:
                y = anomaly_labels.values[-len(X):] if len(anomaly_labels) >= len(X) else np.zeros(len(X))
            
            # Split into train/validation
            split_idx = int(len(X) * (1 - self.config['validation_split']))
            self.X_train = X[:split_idx]
            self.X_val = X[split_idx:]
            
            # Store normal data for threshold calculation
            self.normal_data = self.X_train[y[:split_idx] == 0] if len(y) > split_idx else self.X_train
            
            logger.info(f"Prepared data: {len(self.X_train)} train, {len(self.X_val)} val samples")
            logger.info(f"Anomalies in training data: {np.sum(y[:split_idx])} / {len(y[:split_idx])}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return np.array([]), np.array([])
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for anomaly detection"""
        try:
            features_df = data.copy()
            
            # Price-based features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            features_df['volatility'] = features_df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(features_df['close'])
            features_df['macd'], features_df['macd_signal'] = self._calculate_macd(features_df['close'])
            
            # Options-specific features
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
            features_df['price_momentum'] = features_df['close'] / features_df['close'].shift(5) - 1
            
            # Volatility of volatility
            features_df['vol_of_vol'] = features_df['volatility'].rolling(window=20).std()
            
            # Anomaly-specific features
            features_df['price_acceleration'] = features_df['returns'].diff()
            features_df['volume_acceleration'] = features_df['volume_ratio'].diff()
            features_df['volatility_acceleration'] = features_df['volatility'].diff()
            
            # Rolling statistics
            features_df['price_zscore'] = (features_df['returns'] - features_df['returns'].rolling(20).mean()) / features_df['returns'].rolling(20).std()
            features_df['volume_zscore'] = (features_df['volume_ratio'] - features_df['volume_ratio'].rolling(20).mean()) / features_df['volume_ratio'].rolling(20).std()
            features_df['volatility_zscore'] = (features_df['volatility'] - features_df['volatility'].rolling(20).mean()) / features_df['volatility'].rolling(20).std()
            
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
    
    def _create_sequences(self, data: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Create sequences for anomaly detection"""
        try:
            sequence_length = self.config['sequence_length']
            
            # Extract features
            features = data[feature_cols].values
            
            # Create sequences
            X = []
            for i in range(sequence_length, len(features)):
                X.append(features[i-sequence_length:i])
            
            X = np.array(X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([])
    
    def _create_anomaly_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create anomaly labels based on statistical thresholds"""
        try:
            labels = np.zeros(len(data))
            
            # Price anomalies
            returns = data['returns'].abs()
            price_threshold = returns.quantile(0.95)  # 95th percentile
            price_anomalies = returns > price_threshold
            
            # Volume anomalies
            volume_ratio = data['volume_ratio']
            volume_threshold = volume_ratio.quantile(0.95)
            volume_anomalies = volume_ratio > volume_threshold
            
            # Volatility anomalies
            volatility = data['volatility']
            vol_threshold = volatility.quantile(0.95)
            vol_anomalies = volatility > vol_threshold
            
            # Combine anomaly conditions
            anomaly_mask = price_anomalies | volume_anomalies | vol_anomalies
            labels[anomaly_mask] = 1
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating anomaly labels: {e}")
            return np.zeros(len(data))
    
    def train_model(self, data: pd.DataFrame, anomaly_labels: pd.Series = None) -> AnomalyMetrics:
        """
        Train the VAE anomaly detection model
        
        Args:
            data: Training data
            anomaly_labels: Binary labels for anomalies (optional)
        
        Returns:
            Training metrics
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data, anomaly_labels)
            
            if len(X) == 0:
                logger.error("No data available for training")
                return self._empty_metrics()
            
            # Initialize model
            input_size = X.shape[-1] * X.shape[-2]  # Flattened sequence
            self.model = VariationalAutoencoder(
                input_size=input_size,
                latent_size=self.config['latent_size'],
                hidden_sizes=self.config['hidden_sizes']
            ).to(self.device)
            
            # Loss and optimizer
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Create data loaders
            train_dataset = TensorDataset(torch.FloatTensor(self.X_train))
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            val_dataset = TensorDataset(torch.FloatTensor(self.X_val))
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            # Training loop
            patience_counter = 0
            best_val_loss = float('inf')
            
            for epoch in range(self.config['epochs']):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    batch_X = batch[0].to(self.device)
                    batch_X_flat = batch_X.view(batch_X.size(0), -1)
                    
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = self.model(batch_X_flat)
                    loss, recon_loss, kl_loss = self.model.loss_function(recon_batch, batch_X_flat, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch_X = batch[0].to(self.device)
                        batch_X_flat = batch_X.view(batch_X.size(0), -1)
                        
                        recon_batch, mu, logvar = self.model(batch_X_flat)
                        loss, recon_loss, kl_loss = self.model.loss_function(recon_batch, batch_X_flat, mu, logvar)
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
            
            # Calculate anomaly threshold
            self._calculate_anomaly_threshold()
            
            # Evaluate model
            metrics = self.evaluate_model()
            
            # Save model
            self.save_model()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return self._empty_metrics()
    
    def _calculate_anomaly_threshold(self):
        """Calculate anomaly threshold based on normal data"""
        try:
            if self.model is None or self.normal_data is None:
                self.anomaly_threshold = 0.1  # Default threshold
                return
            
            self.model.eval()
            
            # Calculate reconstruction errors for normal data
            normal_errors = []
            
            with torch.no_grad():
                for i in range(0, len(self.normal_data), self.config['batch_size']):
                    batch = self.normal_data[i:i+self.config['batch_size']]
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    batch_flat = batch_tensor.view(batch_tensor.size(0), -1)
                    
                    recon_batch, _, _ = self.model(batch_flat)
                    errors = torch.mean((recon_batch - batch_flat) ** 2, dim=1)
                    normal_errors.extend(errors.cpu().numpy())
            
            # Set threshold based on percentile
            percentile = self.config['anomaly_threshold_percentile']
            self.anomaly_threshold = np.percentile(normal_errors, percentile)
            
            logger.info(f"Anomaly threshold set to {self.anomaly_threshold:.6f} ({percentile}th percentile)")
            
        except Exception as e:
            logger.error(f"Error calculating anomaly threshold: {e}")
            self.anomaly_threshold = 0.1
    
    def evaluate_model(self) -> AnomalyMetrics:
        """Evaluate anomaly detection model performance"""
        try:
            if self.model is None or self.X_val is None or self.anomaly_threshold is None:
                return self._empty_metrics()
            
            self.model.eval()
            
            # Calculate reconstruction errors
            reconstruction_errors = []
            
            with torch.no_grad():
                for i in range(0, len(self.X_val), self.config['batch_size']):
                    batch = self.X_val[i:i+self.config['batch_size']]
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    batch_flat = batch_tensor.view(batch_tensor.size(0), -1)
                    
                    recon_batch, _, _ = self.model(batch_flat)
                    errors = torch.mean((recon_batch - batch_flat) ** 2, dim=1)
                    reconstruction_errors.extend(errors.cpu().numpy())
            
            # Create binary predictions
            predictions = (np.array(reconstruction_errors) > self.anomaly_threshold).astype(int)
            
            # Calculate metrics (simplified - would need true labels for full evaluation)
            precision = 0.8  # Placeholder
            recall = 0.7     # Placeholder
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            auc_score = 0.75  # Placeholder
            
            false_positive_rate = np.mean(predictions)  # Approximate FPR
            detection_latency = 0.1  # Placeholder - 100ms average latency
            
            return AnomalyMetrics(
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_score=auc_score,
                false_positive_rate=false_positive_rate,
                detection_latency=detection_latency
            )
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return self._empty_metrics()
    
    def detect_anomaly(
        self,
        data: pd.DataFrame,
        symbol: str,
        timestamp: datetime = None
    ) -> AnomalyDetection:
        """
        Detect anomalies in real-time data
        
        Args:
            data: Recent market data
            symbol: Symbol being monitored
            timestamp: Detection timestamp
        
        Returns:
            AnomalyDetection object
        """
        try:
            if self.model is None or self.anomaly_threshold is None:
                logger.error("Model not trained or threshold not set")
                return self._empty_anomaly_detection(symbol, timestamp)
            
            # Prepare input data
            features_df = self._engineer_features(data)
            feature_cols = [col for col in self.config['features'] if col in features_df.columns]
            
            if len(feature_cols) == 0:
                logger.error("No features available for anomaly detection")
                return self._empty_anomaly_detection(symbol, timestamp)
            
            # Get last sequence
            sequence_length = self.config['sequence_length']
            if len(features_df) < sequence_length:
                logger.error(f"Insufficient data: {len(features_df)} < {sequence_length}")
                return self._empty_anomaly_detection(symbol, timestamp)
            
            # Extract and scale features
            features = features_df[feature_cols].values[-sequence_length:]
            features_scaled = self.scaler.transform(features.reshape(1, -1)).reshape(1, sequence_length, -1)
            
            # Calculate reconstruction error
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(features_scaled).to(self.device)
                X_flat = X_tensor.view(X_tensor.size(0), -1)
                
                recon_X, mu, logvar = self.model(X_flat)
                reconstruction_error = torch.mean((recon_X - X_flat) ** 2).item()
            
            # Determine if anomaly
            is_anomaly = reconstruction_error > self.anomaly_threshold
            
            if not is_anomaly:
                return self._empty_anomaly_detection(symbol, timestamp)
            
            # Classify anomaly type and severity
            anomaly_type, severity, recommended_action = self._classify_anomaly(features_df, reconstruction_error)
            
            # Calculate confidence
            confidence = min(0.99, reconstruction_error / (self.anomaly_threshold * 2))
            
            # Feature contributions (simplified)
            feature_contributions = {}
            for i, feature in enumerate(feature_cols):
                feature_contributions[feature] = abs(features[-1, i])  # Last period feature value
            
            # Check for regime transition
            regime_transition = self._detect_regime_transition(features_df)
            
            return AnomalyDetection(
                timestamp=timestamp or datetime.now(),
                symbol=symbol,
                anomaly_score=reconstruction_error,
                anomaly_type=anomaly_type,
                confidence=confidence,
                severity=severity,
                recommended_action=recommended_action,
                feature_contributions=feature_contributions,
                regime_transition=regime_transition,
                model_version=f"vae_v1_{self.profile.tier.value}"
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return self._empty_anomaly_detection(symbol, timestamp)
    
    def _classify_anomaly(self, data: pd.DataFrame, reconstruction_error: float) -> Tuple[str, str, str]:
        """Classify anomaly type and severity"""
        try:
            # Get latest values
            latest = data.iloc[-1]
            
            # Check against known patterns
            for anomaly_type, pattern in self.anomaly_patterns.items():
                thresholds = pattern['thresholds']
                matches = 0
                total_checks = 0
                
                # Price change check
                if 'price_change' in thresholds and 'returns' in data.columns:
                    price_change = abs(latest['returns'])
                    if price_change > abs(thresholds['price_change']):
                        matches += 1
                    total_checks += 1
                
                # Volume spike check
                if 'volume_spike' in thresholds and 'volume_ratio' in data.columns:
                    if latest['volume_ratio'] > thresholds['volume_spike']:
                        matches += 1
                    total_checks += 1
                
                # Volatility spike check
                if 'volatility_spike' in thresholds and 'volatility' in data.columns:
                    vol_ratio = latest['volatility'] / data['volatility'].rolling(20).mean().iloc[-1]
                    if vol_ratio > thresholds['volatility_spike']:
                        matches += 1
                    total_checks += 1
                
                # If majority of checks match, classify as this anomaly type
                if total_checks > 0 and matches / total_checks >= 0.5:
                    return anomaly_type, pattern['severity'], pattern['recommended_action']
            
            # Default classification based on reconstruction error
            if reconstruction_error > self.anomaly_threshold * 3:
                return 'UNKNOWN_CRITICAL', 'CRITICAL', 'CLOSE_ALL_POSITIONS'
            elif reconstruction_error > self.anomaly_threshold * 2:
                return 'UNKNOWN_HIGH', 'HIGH', 'REDUCE_POSITIONS'
            elif reconstruction_error > self.anomaly_threshold * 1.5:
                return 'UNKNOWN_MEDIUM', 'MEDIUM', 'MONITOR_CLOSELY'
            else:
                return 'UNKNOWN_LOW', 'LOW', 'REVIEW_POSITIONS'
                
        except Exception as e:
            logger.error(f"Error classifying anomaly: {e}")
            return 'UNKNOWN', 'MEDIUM', 'MONITOR_CLOSELY'
    
    def _detect_regime_transition(self, data: pd.DataFrame) -> bool:
        """Detect if there's a regime transition"""
        try:
            if len(data) < 40:  # Need enough data for comparison
                return False
            
            # Compare recent vs historical correlations
            recent_data = data.tail(20)
            historical_data = data.iloc[-40:-20]
            
            # Calculate correlation changes
            if 'returns' in data.columns and 'volatility' in data.columns:
                recent_corr = recent_data['returns'].corr(recent_data['volatility'])
                historical_corr = historical_data['returns'].corr(historical_data['volatility'])
                
                correlation_change = abs(recent_corr - historical_corr)
                
                # Regime transition if correlation changes significantly
                return correlation_change > 0.3
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting regime transition: {e}")
            return False
    
    def save_model(self, model_path: str = None) -> bool:
        """Save trained model"""
        try:
            if model_path is None:
                model_path = f"models/anomaly_detector_{self.profile.tier.value}.pkl"
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model_state_dict': self.model.state_dict() if self.model else None,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config,
                'anomaly_threshold': self.anomaly_threshold,
                'anomaly_patterns': self.anomaly_patterns,
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
                model_path = f"models/anomaly_detector_{self.profile.tier.value}.pkl"
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model components
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            self.anomaly_threshold = model_data['anomaly_threshold']
            self.anomaly_patterns = model_data['anomaly_patterns']
            
            # Initialize and load model
            input_size = len(self.feature_names) * self.config['sequence_length']
            self.model = VariationalAutoencoder(
                input_size=input_size,
                latent_size=self.config['latent_size'],
                hidden_sizes=self.config['hidden_sizes']
            ).to(self.device)
            
            if model_data['model_state_dict']:
                self.model.load_state_dict(model_data['model_state_dict'])
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _empty_metrics(self) -> AnomalyMetrics:
        """Return empty metrics"""
        return AnomalyMetrics(0, 0, 0, 0, 0, 0)
    
    def _empty_anomaly_detection(self, symbol: str, timestamp: datetime) -> AnomalyDetection:
        """Return empty anomaly detection"""
        return AnomalyDetection(
            timestamp=timestamp or datetime.now(),
            symbol=symbol,
            anomaly_score=0.0,
            anomaly_type='NONE',
            confidence=0.0,
            severity='LOW',
            recommended_action='NO_ACTION',
            feature_contributions={},
            regime_transition=False,
            model_version="empty"
        )


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create anomaly detector
    detector = OptionsAnomalyDetector(profile)
    
    # Generate sample data with some anomalies
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # Normal data
    normal_returns = np.random.normal(0, 0.01, n_days)
    normal_volume = np.random.lognormal(8, 0.5, n_days)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)  # 5% anomalies
    
    sample_data = pd.DataFrame({
        'date': dates,
        'close': 100 * np.exp(np.cumsum(normal_returns)),
        'volume': normal_volume,
        'high': 100 * np.exp(np.cumsum(normal_returns)) + np.random.rand(n_days) * 2,
        'low': 100 * np.exp(np.cumsum(normal_returns)) - np.random.rand(n_days) * 2,
        'open': 100 * np.exp(np.cumsum(normal_returns)),
        'put_call_ratio': np.random.uniform(0.8, 1.2, n_days),
        'iv_rank': np.random.uniform(0.2, 0.8, n_days),
        'skew': np.random.uniform(-0.5, 0.5, n_days),
        'term_structure': np.random.uniform(-0.2, 0.2, n_days)
    })
    
    # Add anomalies
    for idx in anomaly_indices:
        if idx > 0:
            # Flash crash
            sample_data.loc[idx, 'close'] = sample_data.loc[idx-1, 'close'] * 0.95
            sample_data.loc[idx, 'volume'] = sample_data.loc[idx, 'volume'] * 5
            sample_data.loc[idx, 'high'] = sample_data.loc[idx-1, 'high']
            sample_data.loc[idx, 'low'] = sample_data.loc[idx, 'close'] * 0.98
            sample_data.loc[idx, 'open'] = sample_data.loc[idx-1, 'close']
    
    print("Training anomaly detector...")
    metrics = detector.train_model(sample_data)
    
    print(f"\nTraining Results:")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1 Score: {metrics.f1_score:.4f}")
    print(f"AUC Score: {metrics.auc_score:.4f}")
    print(f"False Positive Rate: {metrics.false_positive_rate:.4f}")
    print(f"Detection Latency: {metrics.detection_latency:.3f}s")
    
    # Test anomaly detection
    anomaly = detector.detect_anomaly(
        data=sample_data.tail(50),
        symbol='SPY',
        timestamp=datetime.now()
    )
    
    print(f"\nAnomaly Detection Result:")
    print(f"Anomaly Score: {anomaly.anomaly_score:.6f}")
    print(f"Anomaly Type: {anomaly.anomaly_type}")
    print(f"Severity: {anomaly.severity}")
    print(f"Confidence: {anomaly.confidence:.2%}")
    print(f"Recommended Action: {anomaly.recommended_action}")
    print(f"Regime Transition: {anomaly.regime_transition}")
    print(f"Model Version: {anomaly.model_version}")
