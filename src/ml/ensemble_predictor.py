"""
Ensemble prediction system that combines multiple timeframe models
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib
from loguru import logger

from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from .multi_timeframe_trainer import MultiTimeframeTrainer
from .model_loader import MLModelLoader


@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictions"""
    name: str
    timeframes: List[str]  # Which timeframes to include
    weights: Optional[Dict[str, float]] = None  # Optional weights for each timeframe
    method: str = "voting"  # "voting", "weighted", "stacking"
    meta_model: str = "logistic"  # For stacking: "logistic", "linear"


class EnsemblePredictor:
    """Ensemble prediction system combining multiple timeframe models"""
    
    def __init__(self):
        self.model_loader = MLModelLoader()
        # Initialize multi_trainer without data collector dependency
        self.multi_trainer = MultiTimeframeTrainer()
        
        # Ensemble configurations
        self.ensemble_configs = [
            # Short-term ensemble (scalping/intraday)
            EnsembleConfig(
                name="short_term_ensemble",
                timeframes=["1min_scalping", "5min_intraday", "15min_swing"],
                method="voting"
            ),
            
            # Medium-term ensemble (swing trading)
            EnsembleConfig(
                name="medium_term_ensemble",
                timeframes=["15min_swing", "1hour_position", "1day_swing"],
                method="weighted",
                weights={"15min_swing": 0.3, "1hour_position": 0.4, "1day_swing": 0.3}
            ),
            
            # Long-term ensemble (position trading)
            EnsembleConfig(
                name="long_term_ensemble",
                timeframes=["1day_swing", "1week_swing", "1month_position"],
                method="stacking",
                meta_model="linear"
            ),
            
            # Investment ensemble (long-term investing)
            EnsembleConfig(
                name="investment_ensemble",
                timeframes=["1month_position", "3month_investment", "6month_investment", "1year_investment"],
                method="weighted",
                weights={
                    "1month_position": 0.4,
                    "3month_investment": 0.3,
                    "6month_investment": 0.2,
                    "1year_investment": 0.1
                }
            ),
            
            # All-timeframe ensemble (comprehensive)
            EnsembleConfig(
                name="comprehensive_ensemble",
                timeframes=[
                    "5min_intraday", "15min_swing", "1hour_position", 
                    "1day_swing", "1week_swing", "1month_position"
                ],
                method="stacking",
                meta_model="logistic"
            )
        ]
        
        self.ensemble_models = {}
        self.ensemble_scalers = {}
        
    def load_timeframe_models(self, model_dir: str = "models/multi_timeframe") -> Dict[str, Dict]:
        """Load all timeframe models"""
        logger.info("Loading timeframe models for ensemble...")
        
        loaded_models = {}
        
        for config in self.multi_trainer.timeframe_configs:
            timeframe_dir = os.path.join(model_dir, config.name)
            
            if not os.path.exists(timeframe_dir):
                logger.warning(f"No models found for {config.name}")
                continue
            
            try:
                models = {}
                
                # Load individual models
                for model_type in ['entry_signal', 'win_probability', 'volatility']:
                    model_path = os.path.join(timeframe_dir, f"{model_type}.joblib")
                    if os.path.exists(model_path):
                        models[model_type] = joblib.load(model_path)
                        logger.info(f"Loaded {config.name} {model_type} model")
                
                # Load scaler
                scaler_path = os.path.join(timeframe_dir, "scaler.joblib")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    models['scaler'] = scaler
                
                # Load metadata
                metadata_path = os.path.join(timeframe_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models['metadata'] = metadata
                
                loaded_models[config.name] = models
                logger.info(f"✅ Loaded {config.name} models")
                
            except Exception as e:
                logger.error(f"Error loading {config.name} models: {e}")
                continue
        
        logger.info(f"Loaded {len(loaded_models)} timeframe models")
        return loaded_models
    
    def create_ensemble_models(self, timeframe_models: Dict[str, Dict]) -> Dict[str, Dict]:
        """Create ensemble models from timeframe models"""
        logger.info("Creating ensemble models...")
        
        ensemble_models = {}
        
        for config in self.ensemble_configs:
            logger.info(f"Creating {config.name} ensemble...")
            
            try:
                ensemble_data = {}
                
                # Collect models for this ensemble
                for timeframe in config.timeframes:
                    if timeframe in timeframe_models:
                        ensemble_data[timeframe] = timeframe_models[timeframe]
                    else:
                        logger.warning(f"Timeframe {timeframe} not found for {config.name}")
                
                if not ensemble_data:
                    logger.error(f"No models found for {config.name}")
                    continue
                
                # Create ensemble models
                ensemble_models[config.name] = {
                    'config': config,
                    'timeframe_models': ensemble_data,
                    'ensemble_models': {}
                }
                
                # Create ensemble for each model type
                for model_type in ['entry_signal', 'win_probability', 'volatility']:
                    ensemble_model = self._create_ensemble_model(
                        ensemble_data, model_type, config
                    )
                    
                    if ensemble_model is not None:
                        ensemble_models[config.name]['ensemble_models'][model_type] = ensemble_model
                        logger.info(f"✅ Created {config.name} {model_type} ensemble")
                
                logger.info(f"✅ Created {config.name} ensemble with {len(ensemble_data)} timeframes")
                
            except Exception as e:
                logger.error(f"Error creating {config.name} ensemble: {e}")
                continue
        
        return ensemble_models
    
    def _create_ensemble_model(self, timeframe_models: Dict[str, Dict], model_type: str, config: EnsembleConfig) -> Optional[Any]:
        """Create ensemble model for a specific model type"""
        try:
            base_models = []
            
            for timeframe, models in timeframe_models.items():
                if model_type in models and 'model' in models[model_type]:
                    base_models.append((timeframe, models[model_type]['model']))
            
            if not base_models:
                logger.warning(f"No base models found for {model_type}")
                return None
            
            # Create ensemble based on method
            if config.method == "voting":
                if model_type == "entry_signal":
                    ensemble = VotingClassifier(
                        estimators=base_models,
                        voting='soft'
                    )
                else:
                    ensemble = VotingRegressor(
                        estimators=base_models
                    )
            
            elif config.method == "weighted":
                if config.weights:
                    weights = [config.weights.get(timeframe, 1.0) for timeframe, _ in base_models]
                    if model_type == "entry_signal":
                        ensemble = VotingClassifier(
                            estimators=base_models,
                            voting='soft',
                            weights=weights
                        )
                    else:
                        ensemble = VotingRegressor(
                            estimators=base_models,
                            weights=weights
                        )
                else:
                    # Equal weights
                    if model_type == "entry_signal":
                        ensemble = VotingClassifier(
                            estimators=base_models,
                            voting='soft'
                        )
                    else:
                        ensemble = VotingRegressor(
                            estimators=base_models
                        )
            
            elif config.method == "stacking":
                # Choose meta-model
                if config.meta_model == "logistic":
                    meta_model = LogisticRegression(random_state=42)
                else:
                    meta_model = LinearRegression()
                
                if model_type == "entry_signal":
                    ensemble = StackingClassifier(
                        estimators=base_models,
                        final_estimator=meta_model,
                        cv=5,
                        stack_method='predict_proba'
                    )
                else:
                    ensemble = StackingRegressor(
                        estimators=base_models,
                        final_estimator=meta_model,
                        cv=5
                    )
            
            else:
                logger.error(f"Unknown ensemble method: {config.method}")
                return None
            
            return {
                'ensemble': ensemble,
                'base_models': base_models,
                'method': config.method
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble model for {model_type}: {e}")
            return None
    
    def train_ensemble_models(self, ensemble_models: Dict[str, Dict], training_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Train ensemble models"""
        logger.info("Training ensemble models...")
        
        for ensemble_name, ensemble_data in ensemble_models.items():
            logger.info(f"Training {ensemble_name} ensemble...")
            
            config = ensemble_data['config']
            timeframe_models = ensemble_data['timeframe_models']
            
            try:
                # Prepare training data for ensemble
                ensemble_features, ensemble_labels = self._prepare_ensemble_training_data(
                    timeframe_models, training_data, config.timeframes
                )
                
                if ensemble_features is None:
                    logger.error(f"No training data for {ensemble_name}")
                    continue
                
                # Train each ensemble model
                for model_type, ensemble_model_data in ensemble_data['ensemble_models'].items():
                    if ensemble_model_data is None:
                        continue
                    
                    logger.info(f"Training {ensemble_name} {model_type} ensemble...")
                    
                    # Get labels for this model type
                    if model_type == "entry_signal":
                        labels = ensemble_labels['entry_signal']
                    elif model_type == "win_probability":
                        labels = ensemble_labels['win_probability']
                    else:  # volatility
                        labels = ensemble_labels['volatility']
                    
                    # Train ensemble
                    ensemble_model_data['ensemble'].fit(ensemble_features, labels)
                    
                    # Evaluate
                    predictions = ensemble_model_data['ensemble'].predict(ensemble_features)
                    
                    if model_type == "entry_signal":
                        accuracy = accuracy_score(labels, predictions)
                        logger.info(f"✅ {ensemble_name} {model_type} accuracy: {accuracy:.3f}")
                        ensemble_model_data['accuracy'] = accuracy
                    else:
                        r2 = r2_score(labels, predictions)
                        logger.info(f"✅ {ensemble_name} {model_type} R²: {r2:.3f}")
                        ensemble_model_data['r2_score'] = r2
                
                logger.info(f"✅ Trained {ensemble_name} ensemble")
                
            except Exception as e:
                logger.error(f"Error training {ensemble_name} ensemble: {e}")
                continue
        
        return ensemble_models
    
    def _prepare_ensemble_training_data(self, timeframe_models: Dict[str, Dict], training_data: Dict[str, pd.DataFrame], timeframes: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Prepare training data for ensemble models"""
        try:
            # Collect predictions from base models
            ensemble_predictions = []
            labels = {'entry_signal': [], 'win_probability': [], 'volatility': []}
            
            for timeframe in timeframes:
                if timeframe not in timeframe_models or timeframe not in training_data:
                    continue
                
                models = timeframe_models[timeframe]
                data = training_data[timeframe]
                
                if 'scaler' not in models:
                    continue
                
                # Scale features
                scaled_features = models['scaler'].transform(data)
                
                # Get predictions from each model type
                timeframe_predictions = []
                
                for model_type in ['entry_signal', 'win_probability', 'volatility']:
                    if model_type in models and 'model' in models[model_type]:
                        model = models[model_type]['model']
                        
                        if model_type == "entry_signal":
                            pred = model.predict_proba(scaled_features)[:, 1]  # Probability of positive class
                        else:
                            pred = model.predict(scaled_features)
                        
                        timeframe_predictions.extend(pred)
                
                if timeframe_predictions:
                    ensemble_predictions.append(timeframe_predictions)
                    
                    # Get labels (simplified - using first timeframe's labels)
                    if not labels['entry_signal']:
                        # Generate labels (simplified approach)
                        labels['entry_signal'] = self._generate_entry_labels(data)
                        labels['win_probability'] = self._generate_win_prob_labels(data)
                        labels['volatility'] = self._generate_volatility_labels(data)
            
            if not ensemble_predictions:
                return None, None
            
            # Convert to DataFrame
            ensemble_features = pd.DataFrame(ensemble_predictions).T
            ensemble_features = ensemble_features.fillna(0)  # Fill any NaN values
            
            return ensemble_features, labels
            
        except Exception as e:
            logger.error(f"Error preparing ensemble training data: {e}")
            return None, None
    
    def _generate_entry_labels(self, data: pd.DataFrame) -> List[int]:
        """Generate entry signal labels (simplified)"""
        if 'close' not in data.columns:
            return [0] * len(data)
        
        labels = []
        for i in range(len(data) - 1):
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + 1]
            labels.append(1 if future_price > current_price * 1.01 else 0)
        
        labels.append(0)  # Last point
        return labels
    
    def _generate_win_prob_labels(self, data: pd.DataFrame) -> List[float]:
        """Generate win probability labels (simplified)"""
        if 'close' not in data.columns:
            return [0.5] * len(data)
        
        labels = []
        for i in range(len(data) - 1):
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + 1]
            return_pct = (future_price - current_price) / current_price
            win_prob = max(0, min(1, (return_pct + 0.02) / 0.04))
            labels.append(win_prob)
        
        labels.append(0.5)  # Last point
        return labels
    
    def _generate_volatility_labels(self, data: pd.DataFrame) -> List[int]:
        """Generate volatility labels (simplified)"""
        if 'close' not in data.columns:
            return [1] * len(data)
        
        labels = []
        for i in range(len(data) - 1):
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + 1]
            volatility = abs((future_price - current_price) / current_price)
            
            if volatility < 0.01:
                labels.append(0)  # Low
            elif volatility < 0.03:
                labels.append(1)  # Medium
            else:
                labels.append(2)  # High
        
        labels.append(1)  # Last point
        return labels
    
    def predict_with_ensemble(self, features: pd.DataFrame, ensemble_name: str, model_type: str) -> Optional[np.ndarray]:
        """Make predictions using ensemble model"""
        try:
            if ensemble_name not in self.ensemble_models:
                logger.error(f"Ensemble {ensemble_name} not found")
                return None
            
            ensemble_data = self.ensemble_models[ensemble_name]
            
            if model_type not in ensemble_data['ensemble_models']:
                logger.error(f"Model type {model_type} not found in {ensemble_name}")
                return None
            
            ensemble_model_data = ensemble_data['ensemble_models'][model_type]
            ensemble_model = ensemble_model_data['ensemble']
            
            # Make prediction
            prediction = ensemble_model.predict(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return None
    
    def predict_with_all_ensembles(self, features: pd.DataFrame, model_type: str) -> Dict[str, np.ndarray]:
        """Make predictions using all ensemble models"""
        predictions = {}
        
        for ensemble_name in self.ensemble_models.keys():
            pred = self.predict_with_ensemble(features, ensemble_name, model_type)
            if pred is not None:
                predictions[ensemble_name] = pred
        
        return predictions
    
    def save_ensemble_models(self, ensemble_models: Dict[str, Dict], model_dir: str = "models/ensemble"):
        """Save ensemble models"""
        os.makedirs(model_dir, exist_ok=True)
        
        for ensemble_name, ensemble_data in ensemble_models.items():
            ensemble_dir = os.path.join(model_dir, ensemble_name)
            os.makedirs(ensemble_dir, exist_ok=True)
            
            # Save ensemble models
            for model_type, model_data in ensemble_data['ensemble_models'].items():
                if model_data is not None:
                    model_path = os.path.join(ensemble_dir, f"{model_type}.joblib")
                    joblib.dump(model_data['ensemble'], model_path)
                    logger.info(f"Saved {ensemble_name} {model_type} ensemble")
            
            # Save metadata
            metadata = {
                'config': {
                    'name': ensemble_data['config'].name,
                    'timeframes': ensemble_data['config'].timeframes,
                    'method': ensemble_data['config'].method,
                    'weights': ensemble_data['config'].weights
                }
            }
            
            import json
            metadata_path = os.path.join(ensemble_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"All ensemble models saved to {model_dir}")
    
    def build_complete_ensemble_system(self, symbols: List[str]) -> Dict[str, Dict]:
        """Build complete ensemble system"""
        logger.info("Building complete ensemble system...")
        
        # Step 1: Load timeframe models
        timeframe_models = self.load_timeframe_models()
        
        # Step 2: Create ensemble models
        ensemble_models = self.create_ensemble_models(timeframe_models)
        
        # Step 3: Prepare training data (simplified - would need real data in practice)
        training_data = self._prepare_training_data_for_ensembles(symbols)
        
        # Step 4: Train ensemble models
        trained_ensemble_models = self.train_ensemble_models(ensemble_models, training_data)
        
        # Step 5: Save ensemble models
        self.save_ensemble_models(trained_ensemble_models)
        
        # Store for predictions
        self.ensemble_models = trained_ensemble_models
        
        logger.info("Complete ensemble system built!")
        return trained_ensemble_models
    
    def _prepare_training_data_for_ensembles(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Prepare training data for ensembles (simplified)"""
        # This would normally collect fresh data
        # For now, return empty dict - would need real implementation
        return {}


if __name__ == "__main__":
    # Test the ensemble predictor
    predictor = EnsemblePredictor()
    
    # Build ensemble system for SPY and QQQ
    symbols = ["SPY", "QQQ"]
    ensemble_models = predictor.build_complete_ensemble_system(symbols)
    
    print(f"Built {len(ensemble_models)} ensemble models")
