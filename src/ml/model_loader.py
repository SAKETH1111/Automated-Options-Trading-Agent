"""
Model Loader
Load and use trained ML models in production
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from loguru import logger


class MLModelLoader:
    """
    Load and use trained ML models
    Provides predictions for trading signals
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model loader
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.loaded = False
        
        logger.info(f"ML Model Loader initialized - looking for models in {self.models_dir}")
    
    def load_models(self) -> bool:
        """
        Load all available models
        
        Returns:
            True if at least one model loaded successfully
        """
        try:
            models_loaded = 0
            
            for model_name in ['entry_signal', 'win_probability', 'volatility']:
                model_path = self.models_dir / f"{model_name}_latest.pkl"
                scaler_path = self.models_dir / f"{model_name}_scaler_latest.pkl"
                features_path = self.models_dir / f"{model_name}_features_latest.json"
                
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"✅ Loaded {model_name} model")
                        models_loaded += 1
                        
                        if scaler_path.exists():
                            self.scalers[model_name] = joblib.load(scaler_path)
                        
                        if features_path.exists():
                            with open(features_path, 'r') as f:
                                self.feature_names[model_name] = json.load(f)
                    
                    except Exception as e:
                        logger.error(f"Error loading {model_name}: {e}")
                else:
                    logger.warning(f"Model not found: {model_path}")
            
            if models_loaded > 0:
                self.loaded = True
                logger.info(f"Successfully loaded {models_loaded} models")
                return True
            else:
                logger.warning("No models found. Run training script first!")
                return False
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_entry_signal(self, features: pd.DataFrame) -> Dict:
        """
        Predict if should enter a trade
        
        Args:
            features: DataFrame with features
            
        Returns:
            Prediction dict with signal and confidence
        """
        if not self.loaded or 'entry_signal' not in self.models:
            return {
                'should_enter': False,
                'confidence': 0.0,
                'reason': 'Model not loaded'
            }
        
        try:
            model = self.models['entry_signal']
            scaler = self.scalers.get('entry_signal')
            required_features = self.feature_names.get('entry_signal', [])
            
            # Ensure we have all required features
            missing = set(required_features) - set(features.columns)
            if missing:
                logger.warning(f"Missing features: {missing}")
                # Fill with defaults
                for feat in missing:
                    features[feat] = 0
            
            # Select only required features
            X = features[required_features]
            
            # Scale if scaler available
            if scaler:
                X = scaler.transform(X)
            
            # Predict
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            return {
                'should_enter': bool(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'bearish': float(probabilities[0]),
                    'bullish': float(probabilities[1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting entry signal: {e}")
            return {
                'should_enter': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_win_probability(self, features: pd.DataFrame) -> float:
        """
        Predict win probability for a trade
        
        Args:
            features: DataFrame with features
            
        Returns:
            Win probability (0-1)
        """
        if not self.loaded or 'win_probability' not in self.models:
            return 0.50  # Default 50%
        
        try:
            model = self.models['win_probability']
            scaler = self.scalers.get('win_probability')
            required_features = self.feature_names.get('win_probability', [])
            
            # Ensure we have all required features
            missing = set(required_features) - set(features.columns)
            if missing:
                for feat in missing:
                    features[feat] = 0
            
            # Select only required features
            X = features[required_features]
            
            # Scale if scaler available
            if scaler:
                X = scaler.transform(X)
            
            # Predict
            probability = model.predict(X)[0]
            
            # Clamp to 0-1
            probability = max(0.0, min(1.0, probability))
            
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error predicting win probability: {e}")
            return 0.50
    
    def predict_volatility(self, features: pd.DataFrame) -> str:
        """
        Predict future volatility regime
        
        Args:
            features: DataFrame with features
            
        Returns:
            Volatility regime ('low', 'medium', 'high')
        """
        if not self.loaded or 'volatility' not in self.models:
            return 'medium'
        
        try:
            model = self.models['volatility']
            scaler = self.scalers.get('volatility')
            required_features = self.feature_names.get('volatility', [])
            
            # Ensure we have all required features
            missing = set(required_features) - set(features.columns)
            if missing:
                for feat in missing:
                    features[feat] = 0
            
            # Select only required features
            X = features[required_features]
            
            # Scale if scaler available
            if scaler:
                X = scaler.transform(X)
            
            # Predict
            vol_class = model.predict(X)[0]
            
            vol_map = {0: 'low', 1: 'medium', 2: 'high'}
            return vol_map.get(vol_class, 'medium')
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return 'medium'
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.loaded
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'loaded': self.loaded,
            'models': list(self.models.keys()),
            'models_dir': str(self.models_dir)
        }

