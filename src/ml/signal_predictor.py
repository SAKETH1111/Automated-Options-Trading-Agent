"""
Signal Predictor Module
ML models for predicting entry/exit signals and win probability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy.orm import Session
from loguru import logger

from .feature_engineer import FeatureEngineer


class SignalPredictor:
    """
    Predict trading signals using machine learning
    Predicts: Entry signals, Exit signals, Win probability
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize signal predictor
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.feature_engineer = FeatureEngineer(db_session)
        
        # Models
        self.entry_model = None
        self.exit_model = None
        self.win_prob_model = None
        
        # Model metadata
        self.model_trained = False
        self.training_date = None
        self.model_accuracy = {}
        
        logger.info("Signal Predictor initialized")
    
    def train_entry_model(
        self,
        symbol: str,
        lookback_days: int = 30
    ) -> Dict:
        """
        Train model to predict entry signals
        
        Args:
            symbol: Symbol to train on
            lookback_days: Days of historical data
            
        Returns:
            Training results
        """
        try:
            logger.info(f"Training entry model for {symbol}")
            
            # Create features
            df = self.feature_engineer.create_features(symbol, lookback_hours=lookback_days*24)
            
            if df.empty or len(df) < 100:
                logger.error("Insufficient data for training")
                return {'success': False, 'error': 'Insufficient data'}
            
            # Create target (price will go up in next 10 periods)
            df = self.feature_engineer.create_target_variable(df, 'direction', forward_periods=10)
            df = df.dropna()
            
            # Prepare features and target
            feature_cols = [col for col in df.columns 
                          if col not in ['timestamp', 'target', 'future_price']]
            
            X = df[feature_cols]
            y = df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Train Random Forest
            self.entry_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
            
            self.entry_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.entry_model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            self.model_accuracy['entry'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            self.model_trained = True
            self.training_date = datetime.utcnow()
            
            logger.info(f"Entry model trained: Accuracy={accuracy:.2%}, F1={f1:.2%}")
            
            return {
                'success': True,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'samples': len(X),
                'features': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Error training entry model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_entry_signal(
        self,
        symbol: str
    ) -> Dict:
        """
        Predict if should enter a trade
        
        Args:
            symbol: Symbol to predict for
            
        Returns:
            Prediction result
        """
        if not self.model_trained or self.entry_model is None:
            return {
                'should_enter': False,
                'confidence': 0.0,
                'reason': 'Model not trained'
            }
        
        try:
            # Create features for current data
            df = self.feature_engineer.create_features(symbol, lookback_hours=24)
            
            if df.empty:
                return {
                    'should_enter': False,
                    'confidence': 0.0,
                    'reason': 'No data available'
                }
            
            # Get latest features
            latest = df.iloc[-1:]
            
            # Remove non-feature columns
            feature_cols = [col for col in latest.columns 
                          if col not in ['timestamp', 'target', 'future_price']]
            
            X = latest[feature_cols]
            
            # Predict
            prediction = self.entry_model.predict(X)[0]
            probabilities = self.entry_model.predict_proba(X)[0]
            
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            return {
                'should_enter': bool(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'down': float(probabilities[0]),
                    'up': float(probabilities[1])
                },
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error predicting entry signal: {e}")
            return {
                'should_enter': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_win_probability(
        self,
        symbol: str,
        strategy: str,
        strikes: List[float],
        dte: int
    ) -> float:
        """
        Predict probability of winning trade
        
        Args:
            symbol: Symbol
            strategy: Strategy type
            strikes: Strike prices
            dte: Days to expiration
            
        Returns:
            Win probability (0-1)
        """
        try:
            # Get current features
            df = self.feature_engineer.create_features(symbol, lookback_hours=24)
            
            if df.empty:
                return 0.50  # Default 50% if no data
            
            latest = df.iloc[-1]
            
            # Simple heuristic model (can be replaced with trained ML model)
            win_prob = 0.50  # Base probability
            
            # Adjust based on RSI
            rsi = latest.get('rsi', 50)
            if 40 <= rsi <= 60:
                win_prob += 0.10  # Neutral RSI is good
            
            # Adjust based on trend
            if latest.get('sma_10', 0) > latest.get('sma_20', 0):
                win_prob += 0.05  # Uptrend
            
            # Adjust based on IV rank
            iv_rank = latest.get('iv_rank', 50)
            if strategy in ['bull_put_spread', 'iron_condor']:
                if iv_rank > 60:
                    win_prob += 0.10  # High IV good for selling
            
            # Clamp to 0-1
            win_prob = max(0.0, min(1.0, win_prob))
            
            return win_prob
            
        except Exception as e:
            logger.error(f"Error predicting win probability: {e}")
            return 0.50
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.model_trained or self.entry_model is None:
            return {}
        
        try:
            feature_importance = self.entry_model.feature_importances_
            feature_names = self.entry_model.feature_names_in_
            
            # Sort by importance
            importance_dict = dict(zip(feature_names, feature_importance))
            sorted_importance = sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            return {
                'top_10': sorted_importance[:10],
                'all': dict(sorted_importance)
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

