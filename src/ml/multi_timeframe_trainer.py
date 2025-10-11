"""
Multi-timeframe ML model trainer
Trains models for different timeframes and combines predictions
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from loguru import logger

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .polygon_data_collector import PolygonDataCollector
from .options_feature_engineer import OptionsFeatureEngineer
from .model_loader import MLModelLoader


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    name: str
    timeframe: str  # '1Min', '5Min', '15Min', '1Hour'
    lookback_days: int
    min_samples: int
    features_to_include: List[str]


class MultiTimeframeTrainer:
    """Train ML models for multiple timeframes"""
    
    def __init__(self):
        # Initialize with None - will be set when needed
        self.data_collector = None
        self.feature_engineer = None
        self.model_loader = MLModelLoader()
        
        # Define timeframe configurations
        self.timeframe_configs = [
            # Short-term timeframes (scalping/intraday)
            TimeframeConfig(
                name="1min_scalping",
                timeframe="1Min",
                lookback_days=30,  # Shorter lookback for scalping
                min_samples=200,
                features_to_include=["price", "technical", "options", "time"]
            ),
            TimeframeConfig(
                name="5min_intraday", 
                timeframe="5Min",
                lookback_days=60,
                min_samples=150,
                features_to_include=["price", "technical", "options", "time"]
            ),
            TimeframeConfig(
                name="15min_swing",
                timeframe="15Min", 
                lookback_days=120,
                min_samples=100,
                features_to_include=["price", "technical", "options", "time", "regime"]
            ),
            TimeframeConfig(
                name="1hour_position",
                timeframe="1Hour",
                lookback_days=365,  # Longer lookback for position trading
                min_samples=50,
                features_to_include=["price", "technical", "options", "time", "regime"]
            ),
            
            # Medium-term timeframes (swing trading)
            TimeframeConfig(
                name="1day_swing",
                timeframe="1Day",
                lookback_days=730,  # 2 years of data
                min_samples=100,
                features_to_include=["price", "technical", "options", "time", "regime", "seasonal"]
            ),
            TimeframeConfig(
                name="1week_swing",
                timeframe="1Week",
                lookback_days=1095,  # 3 years of data
                min_samples=50,
                features_to_include=["price", "technical", "options", "time", "regime", "seasonal"]
            ),
            
            # Long-term timeframes (position trading/investment)
            TimeframeConfig(
                name="1month_position",
                timeframe="1Month",
                lookback_days=1825,  # 5 years of data
                min_samples=30,
                features_to_include=["price", "technical", "options", "regime", "seasonal", "macro"]
            ),
            TimeframeConfig(
                name="3month_investment",
                timeframe="3Month",
                lookback_days=2555,  # 7 years of data
                min_samples=20,
                features_to_include=["price", "technical", "regime", "seasonal", "macro"]
            ),
            TimeframeConfig(
                name="6month_investment",
                timeframe="6Month",
                lookback_days=3285,  # 9 years of data
                min_samples=15,
                features_to_include=["price", "technical", "regime", "seasonal", "macro"]
            ),
            TimeframeConfig(
                name="1year_investment",
                timeframe="1Year",
                lookback_days=5475,  # 15 years of data
                min_samples=10,
                features_to_include=["price", "technical", "regime", "seasonal", "macro"]
            )
        ]
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def collect_multi_timeframe_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect data for all timeframes"""
        logger.info(f"Collecting multi-timeframe data for {len(symbols)} symbols")
        
        # Initialize data collector and feature engineer if needed
        if self.data_collector is None or self.feature_engineer is None:
            try:
                from src.database.session import get_db
                db_session = get_db()
                if self.data_collector is None:
                    self.data_collector = PolygonDataCollector(db_session)
                if self.feature_engineer is None:
                    self.feature_engineer = OptionsFeatureEngineer(db_session)
            except Exception as e:
                logger.warning(f"Could not initialize ML components: {e}")
                logger.info("Using fallback data collection method")
                # For testing, return empty data
                return {config.name: {} for config in self.timeframe_configs}
        
        all_data = {}
        
        for config in self.timeframe_configs:
            logger.info(f"Collecting {config.name} data...")
            
            timeframe_data = {}
            
            try:
                logger.info(f"Fetching {config.timeframe} data for {len(symbols)} symbols...")
                
                # Get historical data for all symbols at once
                data = self.data_collector.collect_training_data(
                    symbols=symbols,
                    timeframe=config.timeframe,
                    lookback_days=config.lookback_days
                )
                
                if data is not None and not data.empty:
                    # Split data by symbol if necessary
                    for symbol in symbols:
                        try:
                            # Filter data for this symbol
                            if 'symbol' in data.columns:
                                symbol_data = data[data['symbol'] == symbol].copy()
                            else:
                                # If no symbol column, assume all data is for this symbol
                                symbol_data = data.copy()
                            
                            if len(symbol_data) > config.min_samples:
                                # Add options features if available
                                if self.feature_engineer is not None:
                                    data_with_options = self.feature_engineer.add_options_features(
                                        symbol_data, symbol, config.timeframe
                                    )
                                else:
                                    data_with_options = symbol_data  # Use basic data if feature engineer not available
                                
                                timeframe_data[symbol] = data_with_options
                                logger.info(f"✅ {symbol}: {len(data_with_options)} samples")
                            else:
                                logger.warning(f"❌ {symbol}: Insufficient data ({len(symbol_data)} samples)")
                        except Exception as e:
                            logger.error(f"Error processing {symbol} data: {e}")
                            continue
                else:
                    logger.warning(f"No data collected for {config.timeframe}")
                    
            except Exception as e:
                logger.error(f"Error collecting {config.timeframe} data: {e}")
            
            all_data[config.name] = timeframe_data
            logger.info(f"✅ {config.name}: {len(timeframe_data)} symbols collected")
        
        return all_data
    
    def prepare_features_and_labels(self, data: Dict[str, pd.DataFrame], config: TimeframeConfig) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prepare features and labels for a specific timeframe"""
        logger.info(f"Preparing features for {config.name}")
        
        all_features = []
        entry_labels = []
        win_prob_labels = []
        volatility_labels = []
        
        for symbol, df in data.items():
            if df is None or len(df) < config.min_samples:
                continue
                
            try:
                # Prepare features
                features = self._prepare_timeframe_features(df, config)
                
                if features is None or len(features) < config.min_samples:
                    continue
                
                # Prepare labels
                entry_labels_sym = self._prepare_entry_labels(features, config.timeframe)
                win_prob_labels_sym = self._prepare_win_probability_labels(features, config.timeframe)
                volatility_labels_sym = self._prepare_volatility_labels(features, config.timeframe)
                
                # Ensure all arrays have the same length
                min_len = min(len(entry_labels_sym), len(win_prob_labels_sym), len(volatility_labels_sym))
                
                # Trim features to match labels (labels are shorter due to forward-looking calculation)
                features_trimmed = features.iloc[:min_len].copy()
                
                # Combine data
                all_features.append(features_trimmed)
                entry_labels.extend(entry_labels_sym[:min_len])
                win_prob_labels.extend(win_prob_labels_sym[:min_len])
                volatility_labels.extend(volatility_labels_sym[:min_len])
                
                logger.info(f"✅ {symbol}: {len(features_trimmed)} samples prepared")
                
            except Exception as e:
                logger.error(f"Error preparing {symbol} features: {e}")
                continue
        
        if not all_features:
            logger.error(f"No features prepared for {config.name}")
            return None, None, None, None
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Clean data
        combined_features = self._clean_features(combined_features)
        
        # Ensure we have enough samples
        if len(combined_features) < config.min_samples:
            logger.error(f"Insufficient samples for {config.name}: {len(combined_features)}")
            return None, None, None, None
        
        logger.info(f"✅ {config.name}: {len(combined_features)} total samples prepared")
        
        return (
            combined_features,
            pd.Series(entry_labels[:len(combined_features)]),
            pd.Series(win_prob_labels[:len(combined_features)]),
            pd.Series(volatility_labels[:len(combined_features)])
        )
    
    def _prepare_timeframe_features(self, df: pd.DataFrame, config: TimeframeConfig) -> Optional[pd.DataFrame]:
        """Prepare features for a specific timeframe"""
        try:
            features = df.copy()
            
            # Add timeframe-specific features
            if "price" in config.features_to_include:
                features = self._add_price_features(features, config.timeframe)
            
            if "technical" in config.features_to_include:
                features = self._add_technical_features(features, config.timeframe)
            
            if "options" in config.features_to_include:
                features = self._add_options_features(features, config.timeframe)
            
            if "time" in config.features_to_include:
                features = self._add_time_features(features, config.timeframe)
            
            if "regime" in config.features_to_include:
                features = self._add_regime_features(features, config.timeframe)
            
            if "seasonal" in config.features_to_include:
                features = self._add_seasonal_features(features, config.timeframe)
            
            if "macro" in config.features_to_include:
                features = self._add_macro_features(features, config.timeframe)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing timeframe features: {e}")
            return None
    
    def _add_price_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add price-based features"""
        features = df.copy()
        
        # Returns
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Price momentum (timeframe-specific)
        if timeframe == "1Min":
            windows = [5, 10, 15, 30]  # Short-term for scalping
        elif timeframe == "5Min":
            windows = [3, 6, 12, 24]  # Medium-term for intraday
        elif timeframe == "15Min":
            windows = [2, 4, 8, 16]  # Longer-term for swing
        elif timeframe == "1Hour":
            windows = [1, 2, 4, 8]  # Long-term for position trading
        elif timeframe == "1Day":
            windows = [5, 10, 20, 50]  # Daily momentum
        elif timeframe == "1Week":
            windows = [2, 4, 8, 16]  # Weekly momentum
        elif timeframe == "1Month":
            windows = [1, 3, 6, 12]  # Monthly momentum
        elif timeframe in ["3Month", "6Month"]:
            windows = [1, 2, 4, 8]  # Quarterly/semi-annual momentum
        else:  # 1Year
            windows = [1, 2, 3, 5]  # Annual momentum
        
        for window in windows:
            features[f'momentum_{window}'] = features['close'].pct_change(window)
            features[f'sma_ratio_{window}'] = features['close'] / features['close'].rolling(window).mean()
        
        # Add volatility features for longer timeframes
        if timeframe in ["1Day", "1Week", "1Month", "3Month", "6Month", "1Year"]:
            features['volatility_20'] = features['returns'].rolling(20).std()
            features['volatility_50'] = features['returns'].rolling(50).std()
            features['vol_ratio'] = features['volatility_20'] / features['volatility_50']
        
        return features
    
    def _add_technical_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add technical indicators"""
        features = df.copy()
        
        # Timeframe-specific technical indicators
        if timeframe in ["1Min", "5Min"]:
            # Short-term indicators for scalping/intraday
            features['rsi_14'] = self._calculate_rsi(features['close'], 14)
            features['rsi_7'] = self._calculate_rsi(features['close'], 7)
            features['bb_upper'] = self._calculate_bollinger_bands(features['close'], 20, 2)[0]
            features['bb_lower'] = self._calculate_bollinger_bands(features['close'], 20, 2)[1]
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['close']
            
        elif timeframe in ["15Min", "1Hour"]:
            # Medium-term indicators for swing trading
            features['rsi_21'] = self._calculate_rsi(features['close'], 21)
            features['macd'] = self._calculate_macd(features['close'])
            features['atr_14'] = self._calculate_atr(features, 14)
            features['adx_14'] = self._calculate_adx(features, 14)
            
        elif timeframe in ["1Day", "1Week"]:
            # Daily/weekly indicators
            features['rsi_14'] = self._calculate_rsi(features['close'], 14)
            features['rsi_21'] = self._calculate_rsi(features['close'], 21)
            features['macd'] = self._calculate_macd(features['close'])
            features['macd_signal'] = self._calculate_macd_signal(features['close'])
            features['bb_upper'] = self._calculate_bollinger_bands(features['close'], 20, 2)[0]
            features['bb_lower'] = self._calculate_bollinger_bands(features['close'], 20, 2)[1]
            features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
        else:
            # Long-term indicators for position trading/investment
            features['rsi_21'] = self._calculate_rsi(features['close'], 21)
            features['rsi_50'] = self._calculate_rsi(features['close'], 50)
            features['macd'] = self._calculate_macd(features['close'])
            features['macd_signal'] = self._calculate_macd_signal(features['close'])
            features['sma_200'] = features['close'].rolling(200).mean()
            features['sma_50'] = features['close'].rolling(50).mean()
            features['sma_ratio'] = features['close'] / features['sma_200']
            features['trend_strength'] = (features['sma_50'] - features['sma_200']) / features['sma_200']
        
        return features
    
    def _add_options_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add options-specific features"""
        features = df.copy()
        
        # Timeframe-specific options features
        if timeframe in ["1Min", "5Min"]:
            # Short-term options features
            if 'atm_iv' in features.columns:
                features['iv_change_1'] = features['atm_iv'].pct_change()
                features['iv_momentum_5'] = features['atm_iv'].pct_change(5)
        
        else:
            # Longer-term options features
            if 'atm_iv' in features.columns:
                features['iv_change_5'] = features['atm_iv'].pct_change(5)
                features['iv_momentum_20'] = features['atm_iv'].pct_change(20)
                features['iv_rank'] = features['atm_iv'].rolling(252).rank(pct=True)
        
        return features
    
    def _add_time_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add time-based features"""
        features = df.copy()
        
        if 'timestamp' in features.columns:
            timestamp = pd.to_datetime(features['timestamp'])
            features['hour'] = timestamp.dt.hour
            features['minute'] = timestamp.dt.minute
            features['day_of_week'] = timestamp.dt.dayofweek
            features['month'] = timestamp.dt.month
            features['quarter'] = timestamp.dt.quarter
            features['year'] = timestamp.dt.year
            
            # Timeframe-specific time features
            if timeframe == "1Min":
                features['market_session'] = features['hour'].apply(self._get_market_session)
            elif timeframe == "5Min":
                features['quarter_hour'] = (features['minute'] // 15) * 15
            elif timeframe in ["15Min", "1Hour"]:
                features['is_opening'] = (features['hour'] == 9) & (features['minute'] < 30)
                features['is_closing'] = (features['hour'] == 15) & (features['minute'] >= 30)
            elif timeframe in ["1Day", "1Week"]:
                features['is_monday'] = (features['day_of_week'] == 0).astype(int)
                features['is_friday'] = (features['day_of_week'] == 4).astype(int)
                features['month_end'] = timestamp.dt.is_month_end.astype(int)
            elif timeframe in ["1Month", "3Month", "6Month", "1Year"]:
                features['is_q1'] = (features['quarter'] == 1).astype(int)
                features['is_q4'] = (features['quarter'] == 4).astype(int)
                features['is_january'] = (features['month'] == 1).astype(int)
                features['is_december'] = (features['month'] == 12).astype(int)
        
        return features
    
    def _add_seasonal_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add seasonal features for longer timeframes"""
        features = df.copy()
        
        if 'timestamp' in features.columns and timeframe in ["1Day", "1Week", "1Month", "3Month", "6Month", "1Year"]:
            timestamp = pd.to_datetime(features['timestamp'])
            
            # Seasonal patterns
            features['season'] = timestamp.dt.month.map({
                12: 0, 1: 0, 2: 0,  # Winter
                3: 1, 4: 1, 5: 1,   # Spring
                6: 2, 7: 2, 8: 2,   # Summer
                9: 3, 10: 3, 11: 3  # Fall
            })
            
            # Holiday effects (approximate)
            features['is_holiday_season'] = ((features['month'] == 12) | (features['month'] == 1)).astype(int)
            features['is_earnings_season'] = ((features['month'] == 1) | (features['month'] == 4) | 
                                            (features['month'] == 7) | (features['month'] == 10)).astype(int)
            
            # Year-end effects
            features['is_year_end'] = (features['month'] == 12).astype(int)
            features['is_year_start'] = (features['month'] == 1).astype(int)
        
        return features
    
    def _add_macro_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add macroeconomic features for longer timeframes"""
        features = df.copy()
        
        if timeframe in ["1Month", "3Month", "6Month", "1Year"]:
            # Add economic cycle indicators (simplified)
            if 'timestamp' in features.columns:
                timestamp = pd.to_datetime(features['timestamp'])
                year = timestamp.dt.year
                
                # Economic cycle phases (simplified model)
                # This would ideally use real economic data
                features['economic_cycle'] = ((year - 2000) % 8) / 8  # 8-year cycle
                
                # Market regime indicators
                features['is_bull_market'] = (features['economic_cycle'] > 0.5).astype(int)
                features['is_bear_market'] = (features['economic_cycle'] < 0.3).astype(int)
                
                # Crisis periods (simplified)
                features['is_crisis_period'] = ((year == 2008) | (year == 2020) | (year == 2022)).astype(int)
        
        return features
    
    def _add_regime_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add market regime features"""
        features = df.copy()
        
        # Add volatility regime
        if 'close' in features.columns:
            returns = features['close'].pct_change()
            volatility = returns.rolling(20).std()
            features['volatility_regime'] = pd.cut(
                volatility, 
                bins=3, 
                labels=['low', 'medium', 'high']
            ).astype('category').cat.codes
        
        return features
    
    def _prepare_entry_labels(self, features: pd.DataFrame, timeframe: str) -> List[int]:
        """Prepare entry signal labels"""
        labels = []
        
        if 'close' in features.columns:
            # Timeframe-specific forward periods and thresholds
            if timeframe == "1Min":
                forward_periods = 5  # 5 minutes ahead
                threshold = 0.002  # 0.2%
            elif timeframe == "5Min":
                forward_periods = 3  # 15 minutes ahead
                threshold = 0.003  # 0.3%
            elif timeframe == "15Min":
                forward_periods = 2  # 30 minutes ahead
                threshold = 0.005  # 0.5%
            elif timeframe == "1Hour":
                forward_periods = 1  # 1 hour ahead
                threshold = 0.008  # 0.8%
            elif timeframe == "1Day":
                forward_periods = 1  # 1 day ahead
                threshold = 0.015  # 1.5%
            elif timeframe == "1Week":
                forward_periods = 1  # 1 week ahead
                threshold = 0.025  # 2.5%
            elif timeframe == "1Month":
                forward_periods = 1  # 1 month ahead
                threshold = 0.05   # 5%
            elif timeframe == "3Month":
                forward_periods = 1  # 3 months ahead
                threshold = 0.10   # 10%
            elif timeframe == "6Month":
                forward_periods = 1  # 6 months ahead
                threshold = 0.15   # 15%
            else:  # 1Year
                forward_periods = 1  # 1 year ahead
                threshold = 0.20   # 20%
            
            for i in range(len(features) - forward_periods):
                current_price = features['close'].iloc[i]
                future_price = features['close'].iloc[i + forward_periods]
                
                # Entry signal: 1 if price increases by threshold, 0 otherwise
                labels.append(1 if (future_price - current_price) / current_price > threshold else 0)
        
        return labels
    
    def _prepare_win_probability_labels(self, features: pd.DataFrame, timeframe: str) -> List[float]:
        """Prepare win probability labels"""
        labels = []
        
        if 'close' in features.columns:
            # Timeframe-specific forward periods and normalization ranges
            if timeframe == "1Min":
                forward_periods = 5
                norm_range = 0.04  # ±2%
            elif timeframe == "5Min":
                forward_periods = 3
                norm_range = 0.06  # ±3%
            elif timeframe == "15Min":
                forward_periods = 2
                norm_range = 0.08  # ±4%
            elif timeframe == "1Hour":
                forward_periods = 1
                norm_range = 0.12  # ±6%
            elif timeframe == "1Day":
                forward_periods = 1
                norm_range = 0.20  # ±10%
            elif timeframe == "1Week":
                forward_periods = 1
                norm_range = 0.30  # ±15%
            elif timeframe == "1Month":
                forward_periods = 1
                norm_range = 0.50  # ±25%
            elif timeframe == "3Month":
                forward_periods = 1
                norm_range = 0.80  # ±40%
            elif timeframe == "6Month":
                forward_periods = 1
                norm_range = 1.00  # ±50%
            else:  # 1Year
                forward_periods = 1
                norm_range = 1.50  # ±75%
            
            for i in range(len(features) - forward_periods):
                current_price = features['close'].iloc[i]
                future_price = features['close'].iloc[i + forward_periods]
                
                # Win probability: normalized return (0-1 scale)
                return_pct = (future_price - current_price) / current_price
                win_prob = max(0, min(1, (return_pct + norm_range/2) / norm_range))
                labels.append(win_prob)
        
        return labels
    
    def _prepare_volatility_labels(self, features: pd.DataFrame, timeframe: str) -> List[int]:
        """Prepare volatility forecasting labels"""
        labels = []
        
        if 'close' in features.columns:
            # Timeframe-specific forward periods and volatility thresholds
            if timeframe == "1Min":
                forward_periods = 5
                low_thresh, high_thresh = 0.005, 0.015
            elif timeframe == "5Min":
                forward_periods = 3
                low_thresh, high_thresh = 0.008, 0.025
            elif timeframe == "15Min":
                forward_periods = 2
                low_thresh, high_thresh = 0.012, 0.035
            elif timeframe == "1Hour":
                forward_periods = 1
                low_thresh, high_thresh = 0.020, 0.060
            elif timeframe == "1Day":
                forward_periods = 1
                low_thresh, high_thresh = 0.015, 0.045
            elif timeframe == "1Week":
                forward_periods = 1
                low_thresh, high_thresh = 0.025, 0.075
            elif timeframe == "1Month":
                forward_periods = 1
                low_thresh, high_thresh = 0.050, 0.150
            elif timeframe == "3Month":
                forward_periods = 1
                low_thresh, high_thresh = 0.100, 0.300
            elif timeframe == "6Month":
                forward_periods = 1
                low_thresh, high_thresh = 0.150, 0.450
            else:  # 1Year
                forward_periods = 1
                low_thresh, high_thresh = 0.200, 0.600
            
            for i in range(len(features) - forward_periods):
                # Calculate future volatility
                future_returns = features['close'].iloc[i:i+forward_periods+1].pct_change().dropna()
                future_vol = future_returns.std()
                
                # Classify volatility: 0=low, 1=medium, 2=high
                if future_vol < low_thresh:
                    labels.append(0)
                elif future_vol < high_thresh:
                    labels.append(1)
                else:
                    labels.append(2)
        
        return labels
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for training"""
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Remove columns with too many NaN values
        numeric_features = numeric_features.dropna(axis=1, thresh=len(numeric_features) * 0.7)
        
        # Fill remaining NaN values
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        return numeric_features
    
    def train_timeframe_models(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict]:
        """Train models for each timeframe"""
        logger.info("Training multi-timeframe models...")
        
        trained_models = {}
        
        for config in self.timeframe_configs:
            logger.info(f"Training {config.name} models...")
            
            if config.name not in data:
                logger.warning(f"No data for {config.name}")
                continue
            
            # Prepare features and labels
            features, entry_labels, win_prob_labels, volatility_labels = self.prepare_features_and_labels(
                data[config.name], config
            )
            
            if features is None:
                logger.error(f"Failed to prepare features for {config.name}")
                continue
            
            # Check if we have enough class diversity
            n_entry_classes = len(set(entry_labels))
            if n_entry_classes < 2:
                logger.warning(f"⚠️ Skipping {config.name} - only {n_entry_classes} class(es) in entry labels")
                continue
            
            # Split data
            X_train, X_test, y_entry_train, y_entry_test = train_test_split(
                features, entry_labels, test_size=0.2, random_state=42, stratify=entry_labels
            )
            
            _, _, y_win_train, y_win_test = train_test_split(
                features, win_prob_labels, test_size=0.2, random_state=42
            )
            
            _, _, y_vol_train, y_vol_test = train_test_split(
                features, volatility_labels, test_size=0.2, random_state=42, stratify=volatility_labels
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {}
            
            # Entry Signal Model
            logger.info(f"Training entry signal model for {config.name}...")
            entry_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            entry_model.fit(X_train_scaled, y_entry_train)
            entry_pred = entry_model.predict(X_test_scaled)
            entry_accuracy = accuracy_score(y_entry_test, entry_pred)
            
            models['entry_signal'] = {
                'model': entry_model,
                'accuracy': entry_accuracy,
                'predictions': entry_pred,
                'test_labels': y_entry_test
            }
            
            # Win Probability Model
            logger.info(f"Training win probability model for {config.name}...")
            win_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            win_model.fit(X_train_scaled, y_win_train)
            win_pred = win_model.predict(X_test_scaled)
            win_r2 = r2_score(y_win_test, win_pred)
            
            models['win_probability'] = {
                'model': win_model,
                'r2_score': win_r2,
                'predictions': win_pred,
                'test_labels': y_win_test
            }
            
            # Volatility Model (skip if only one class)
            n_vol_classes = len(set(y_vol_train))
            vol_accuracy = None  # Initialize to None
            
            if n_vol_classes < 2:
                logger.warning(f"⚠️ Skipping volatility model for {config.name} - only {n_vol_classes} class(es) in data")
                models['volatility'] = None
            else:
                logger.info(f"Training volatility model for {config.name}...")
                vol_model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
                vol_model.fit(X_train_scaled, y_vol_train)
                vol_pred = vol_model.predict(X_test_scaled)
                vol_accuracy = accuracy_score(y_vol_test, vol_pred)
                
                models['volatility'] = {
                    'model': vol_model,
                    'accuracy': vol_accuracy,
                    'predictions': vol_pred,
                    'test_labels': y_vol_test
                }
            
            # Store models and scaler
            trained_models[config.name] = {
                'models': models,
                'scaler': scaler,
                'feature_names': features.columns.tolist(),
                'config': config
            }
            
            # Log results
            logger.info(f"✅ {config.name} models trained:")
            logger.info(f"   Entry Signal Accuracy: {entry_accuracy:.3f}")
            logger.info(f"   Win Probability R²: {win_r2:.3f}")
            if models['volatility'] is not None:
                logger.info(f"   Volatility Accuracy: {vol_accuracy:.3f}")
            else:
                logger.info(f"   Volatility Model: Skipped (insufficient class diversity)")
        
        return trained_models
    
    def save_models(self, models: Dict[str, Dict], model_dir: str = "models/multi_timeframe"):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        for timeframe, model_data in models.items():
            timeframe_dir = os.path.join(model_dir, timeframe)
            os.makedirs(timeframe_dir, exist_ok=True)
            
            # Save models
            for model_name, model_info in model_data['models'].items():
                model_path = os.path.join(timeframe_dir, f"{model_name}.joblib")
                joblib.dump(model_info['model'], model_path)
                logger.info(f"Saved {timeframe} {model_name} model")
            
            # Save scaler
            scaler_path = os.path.join(timeframe_dir, "scaler.joblib")
            joblib.dump(model_data['scaler'], scaler_path)
            
            # Save metadata
            metadata = {
                'feature_names': model_data['feature_names'],
                'config': {
                    'name': model_data['config'].name,
                    'timeframe': model_data['config'].timeframe,
                    'lookback_days': model_data['config'].lookback_days,
                    'min_samples': model_data['config'].min_samples
                }
            }
            
            import json
            metadata_path = os.path.join(timeframe_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"All multi-timeframe models saved to {model_dir}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2
    
    def _calculate_macd_signal(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD Signal Line"""
        macd = self._calculate_macd(prices)
        return macd.ewm(span=9).mean()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX (simplified version)"""
        # Simplified ADX calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return (atr / df['close'] * 100).rolling(window=period).mean()
    
    def _get_market_session(self, hour: int) -> int:
        """Get market session based on hour"""
        if 9 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 16:
            return 1  # Afternoon
        else:
            return 2  # After hours
    
    def train_all_models(self, symbols: List[str]) -> Dict[str, Dict]:
        """Train all multi-timeframe models"""
        logger.info("Starting multi-timeframe ML training...")
        
        # Collect data for all timeframes
        all_data = self.collect_multi_timeframe_data(symbols)
        
        # Train models for each timeframe
        trained_models = self.train_timeframe_models(all_data)
        
        # Save models
        self.save_models(trained_models)
        
        logger.info("Multi-timeframe ML training complete!")
        return trained_models


if __name__ == "__main__":
    # Test the multi-timeframe trainer
    trainer = MultiTimeframeTrainer()
    
    # Train models for SPY and QQQ
    symbols = ["SPY", "QQQ"]
    models = trainer.train_all_models(symbols)
    
    print(f"Trained {len(models)} timeframe models")
