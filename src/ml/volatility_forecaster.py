"""
Volatility Forecaster Module
Predict future volatility using ML
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy.orm import Session
from loguru import logger

from .feature_engineer import FeatureEngineer


class VolatilityForecaster:
    """
    Forecast future volatility using machine learning
    Helps optimize entry timing and strategy selection
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize volatility forecaster
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.feature_engineer = FeatureEngineer(db_session)
        self.model = None
        self.model_trained = False
        
        logger.info("Volatility Forecaster initialized")
    
    def train_model(
        self,
        symbol: str,
        lookback_days: int = 60
    ) -> Dict:
        """
        Train volatility forecasting model
        
        Args:
            symbol: Symbol to train on
            lookback_days: Days of historical data
            
        Returns:
            Training results
        """
        try:
            logger.info(f"Training volatility model for {symbol}")
            
            # Create features
            df = self.feature_engineer.create_features(symbol, lookback_hours=lookback_days*24)
            
            if df.empty or len(df) < 200:
                return {'success': False, 'error': 'Insufficient data'}
            
            # Create target (future volatility)
            df['future_vol'] = df['returns'].rolling(20).std().shift(-20)
            df = df.dropna()
            
            # Prepare features
            feature_cols = [col for col in df.columns 
                          if col not in ['timestamp', 'future_vol', 'target']]
            
            X = df[feature_cols]
            y = df['future_vol']
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X, y)
            
            # Evaluate
            predictions = self.model.predict(X)
            mse = np.mean((predictions - y) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y))
            
            self.model_trained = True
            
            logger.info(f"Volatility model trained: RMSE={rmse:.6f}")
            
            return {
                'success': True,
                'rmse': rmse,
                'mae': mae,
                'samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training volatility model: {e}")
            return {'success': False, 'error': str(e)}
    
    def forecast_volatility(
        self,
        symbol: str,
        periods_ahead: int = 20
    ) -> Dict:
        """
        Forecast future volatility
        
        Args:
            symbol: Symbol to forecast
            periods_ahead: Periods to forecast ahead
            
        Returns:
            Volatility forecast
        """
        if not self.model_trained:
            return {
                'forecast': None,
                'error': 'Model not trained'
            }
        
        try:
            # Get current features
            df = self.feature_engineer.create_features(symbol, lookback_hours=24)
            
            if df.empty:
                return {'forecast': None, 'error': 'No data'}
            
            latest = df.iloc[-1:]
            feature_cols = [col for col in latest.columns 
                          if col not in ['timestamp', 'target', 'future_vol']]
            
            X = latest[feature_cols]
            
            # Predict
            forecast = self.model.predict(X)[0]
            
            # Get current volatility for comparison
            current_vol = df['returns'].tail(20).std()
            
            # Determine regime
            if forecast > current_vol * 1.3:
                regime = 'INCREASING'
            elif forecast < current_vol * 0.7:
                regime = 'DECREASING'
            else:
                regime = 'STABLE'
            
            return {
                'forecast': float(forecast),
                'current': float(current_vol),
                'change': float(forecast - current_vol),
                'change_pct': float((forecast / current_vol - 1) * 100) if current_vol > 0 else 0,
                'regime': regime,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return {'forecast': None, 'error': str(e)}

