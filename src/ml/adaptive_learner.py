"""
Adaptive learning system that monitors model performance and retrains when needed
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib
from loguru import logger

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score

from .multi_timeframe_trainer import MultiTimeframeTrainer
from .ensemble_predictor import EnsemblePredictor
from .model_loader import MLModelLoader


@dataclass
class PerformanceThresholds:
    """Performance thresholds for adaptive learning"""
    min_accuracy: float = 0.55  # Minimum accuracy for entry signals
    min_r2_score: float = 0.30  # Minimum R² for regression models
    max_accuracy_drop: float = 0.05  # Maximum allowed accuracy drop
    min_samples_for_retrain: int = 50  # Minimum new samples needed for retraining
    retrain_frequency_days: int = 7  # Minimum days between retraining


@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    model_type: str
    accuracy: float
    r2_score: Optional[float] = None
    last_updated: datetime = None
    sample_count: int = 0
    predictions_count: int = 0
    correct_predictions: int = 0


class AdaptiveLearner:
    """Adaptive learning system for ML models"""
    
    def __init__(self):
        # Initialize components without dependencies for testing
        self.multi_trainer = MultiTimeframeTrainer()
        self.ensemble_predictor = EnsemblePredictor()
        self.model_loader = MLModelLoader()
        
        # Performance tracking
        self.performance_history = {}
        self.thresholds = PerformanceThresholds()
        
        # Model paths
        self.model_base_path = "models"
        self.performance_log_path = "logs/adaptive_learning.json"
        
        # Initialize performance tracking
        self._load_performance_history()
        
    def monitor_model_performance(self, model_name: str, model_type: str, 
                                predictions: np.ndarray, actual_values: np.ndarray,
                                new_samples: int = 1) -> bool:
        """
        Monitor model performance and determine if retraining is needed
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('entry_signal', 'win_probability', 'volatility')
            predictions: Model predictions
            actual_values: Actual values
            new_samples: Number of new samples used for prediction
            
        Returns:
            bool: True if retraining is recommended
        """
        try:
            # Calculate current performance
            if model_type == "entry_signal":
                current_accuracy = accuracy_score(actual_values, predictions)
                current_r2 = None
            else:
                current_r2 = r2_score(actual_values, predictions)
                current_accuracy = None
            
            # Update performance tracking
            self._update_performance_tracking(
                model_name, model_type, current_accuracy, current_r2, new_samples
            )
            
            # Check if retraining is needed
            needs_retraining = self._should_retrain(model_name, model_type)
            
            if needs_retraining:
                logger.warning(f"Model {model_name} {model_type} needs retraining!")
                logger.info(f"Current performance - Accuracy: {current_accuracy}, R²: {current_r2}")
            
            return needs_retraining
            
        except Exception as e:
            logger.error(f"Error monitoring model performance: {e}")
            return False
    
    def _update_performance_tracking(self, model_name: str, model_type: str,
                                   accuracy: Optional[float], r2_score: Optional[float],
                                   new_samples: int):
        """Update performance tracking data"""
        key = f"{model_name}_{model_type}"
        
        if key not in self.performance_history:
            self.performance_history[key] = ModelPerformance(
                model_name=model_name,
                model_type=model_type,
                accuracy=accuracy or 0.0,
                r2_score=r2_score,
                last_updated=datetime.now(),
                sample_count=0,
                predictions_count=0,
                correct_predictions=0
            )
        
        perf = self.performance_history[key]
        perf.last_updated = datetime.now()
        perf.sample_count += new_samples
        perf.predictions_count += len(actual_values) if 'actual_values' in locals() else new_samples
        
        # Update accuracy/r2 with weighted average
        if accuracy is not None:
            if perf.accuracy == 0.0:
                perf.accuracy = accuracy
            else:
                # Weighted average based on sample count
                weight = min(new_samples / (perf.sample_count + new_samples), 0.1)
                perf.accuracy = (1 - weight) * perf.accuracy + weight * accuracy
        
        if r2_score is not None:
            if perf.r2_score is None:
                perf.r2_score = r2_score
            else:
                weight = min(new_samples / (perf.sample_count + new_samples), 0.1)
                perf.r2_score = (1 - weight) * perf.r2_score + weight * r2_score
        
        # Save updated performance
        self._save_performance_history()
    
    def _should_retrain(self, model_name: str, model_type: str) -> bool:
        """Determine if model should be retrained"""
        key = f"{model_name}_{model_type}"
        
        if key not in self.performance_history:
            return False
        
        perf = self.performance_history[key]
        
        # Check if enough time has passed since last retraining
        if perf.last_updated:
            days_since_update = (datetime.now() - perf.last_updated).days
            if days_since_update < self.thresholds.retrain_frequency_days:
                return False
        
        # Check performance thresholds
        if model_type == "entry_signal":
            if perf.accuracy < self.thresholds.min_accuracy:
                logger.warning(f"Accuracy {perf.accuracy:.3f} below threshold {self.thresholds.min_accuracy}")
                return True
        
        elif model_type in ["win_probability", "volatility"]:
            if perf.r2_score and perf.r2_score < self.thresholds.min_r2_score:
                logger.warning(f"R² score {perf.r2_score:.3f} below threshold {self.thresholds.min_r2_score}")
                return True
        
        # Check for performance degradation
        if self._has_performance_degraded(key):
            logger.warning(f"Performance degraded for {model_name} {model_type}")
            return True
        
        return False
    
    def _has_performance_degraded(self, model_key: str) -> bool:
        """Check if model performance has degraded significantly"""
        # This would compare current performance with historical performance
        # For now, simplified implementation
        return False
    
    def retrain_model(self, model_name: str, model_type: str, 
                     symbols: List[str], force_retrain: bool = False) -> bool:
        """
        Retrain a specific model
        
        Args:
            model_name: Name of the model to retrain
            model_type: Type of model
            symbols: Symbols to use for retraining
            force_retrain: Force retraining even if not needed
            
        Returns:
            bool: True if retraining was successful
        """
        try:
            logger.info(f"Retraining {model_name} {model_type} model...")
            
            # Check if retraining is needed (unless forced)
            if not force_retrain:
                key = f"{model_name}_{model_type}"
                if key in self.performance_history:
                    if not self._should_retrain(model_name, model_type):
                        logger.info(f"Retraining not needed for {model_name} {model_type}")
                        return True
            
            # Determine if this is a timeframe model or ensemble model
            if model_name in [config.name for config in self.multi_trainer.timeframe_configs]:
                # Retrain timeframe model
                return self._retrain_timeframe_model(model_name, symbols)
            else:
                # Retrain ensemble model
                return self._retrain_ensemble_model(model_name, symbols)
                
        except Exception as e:
            logger.error(f"Error retraining {model_name} {model_type}: {e}")
            return False
    
    def _retrain_timeframe_model(self, timeframe_name: str, symbols: List[str]) -> bool:
        """Retrain a specific timeframe model"""
        try:
            logger.info(f"Retraining {timeframe_name} timeframe model...")
            
            # Collect fresh data
            all_data = self.multi_trainer.collect_multi_timeframe_data(symbols)
            
            if timeframe_name not in all_data:
                logger.error(f"No data found for {timeframe_name}")
                return False
            
            # Find the config for this timeframe
            config = None
            for cfg in self.multi_trainer.timeframe_configs:
                if cfg.name == timeframe_name:
                    config = cfg
                    break
            
            if config is None:
                logger.error(f"No config found for {timeframe_name}")
                return False
            
            # Prepare features and labels
            features, entry_labels, win_prob_labels, volatility_labels = self.multi_trainer.prepare_features_and_labels(
                all_data[timeframe_name], config
            )
            
            if features is None:
                logger.error(f"Failed to prepare features for {timeframe_name}")
                return False
            
            # Train new models
            trained_models = self.multi_trainer.train_timeframe_models({timeframe_name: all_data[timeframe_name]})
            
            if timeframe_name in trained_models:
                # Save the retrained models
                self.multi_trainer.save_models(trained_models)
                
                # Update performance tracking
                model_data = trained_models[timeframe_name]
                
                # Update entry signal model
                if 'entry_signal' in model_data['models']:
                    accuracy = model_data['models']['entry_signal']['accuracy']
                    self._update_performance_tracking(timeframe_name, 'entry_signal', accuracy, None, len(features))
                
                # Update win probability model
                if 'win_probability' in model_data['models']:
                    r2_score = model_data['models']['win_probability']['r2_score']
                    self._update_performance_tracking(timeframe_name, 'win_probability', None, r2_score, len(features))
                
                # Update volatility model
                if 'volatility' in model_data['models']:
                    accuracy = model_data['models']['volatility']['accuracy']
                    self._update_performance_tracking(timeframe_name, 'volatility', accuracy, None, len(features))
                
                logger.info(f"✅ Successfully retrained {timeframe_name} model")
                return True
            else:
                logger.error(f"Failed to train {timeframe_name} model")
                return False
                
        except Exception as e:
            logger.error(f"Error retraining timeframe model {timeframe_name}: {e}")
            return False
    
    def _retrain_ensemble_model(self, ensemble_name: str, symbols: List[str]) -> bool:
        """Retrain a specific ensemble model"""
        try:
            logger.info(f"Retraining {ensemble_name} ensemble model...")
            
            # Build complete ensemble system
            ensemble_models = self.ensemble_predictor.build_complete_ensemble_system(symbols)
            
            if ensemble_name in ensemble_models:
                logger.info(f"✅ Successfully retrained {ensemble_name} ensemble")
                return True
            else:
                logger.error(f"Failed to retrain {ensemble_name} ensemble")
                return False
                
        except Exception as e:
            logger.error(f"Error retraining ensemble model {ensemble_name}: {e}")
            return False
    
    def auto_retrain_models(self, symbols: List[str]) -> Dict[str, bool]:
        """Automatically retrain all models that need retraining"""
        logger.info("Starting automatic model retraining...")
        
        retrain_results = {}
        
        # Check all timeframe models
        for config in self.multi_trainer.timeframe_configs:
            for model_type in ['entry_signal', 'win_probability', 'volatility']:
                model_name = config.name
                
                if self._should_retrain(model_name, model_type):
                    logger.info(f"Auto-retraining {model_name} {model_type}...")
                    success = self.retrain_model(model_name, model_type, symbols)
                    retrain_results[f"{model_name}_{model_type}"] = success
        
        # Check all ensemble models
        for config in self.ensemble_predictor.ensemble_configs:
            for model_type in ['entry_signal', 'win_probability', 'volatility']:
                ensemble_name = config.name
                
                if self._should_retrain(ensemble_name, model_type):
                    logger.info(f"Auto-retraining {ensemble_name} {model_type}...")
                    success = self.retrain_model(ensemble_name, model_type, symbols)
                    retrain_results[f"{ensemble_name}_{model_type}"] = success
        
        # Log results
        successful_retrains = sum(retrain_results.values())
        total_retrains = len(retrain_results)
        
        logger.info(f"Auto-retraining complete: {successful_retrains}/{total_retrains} successful")
        
        return retrain_results
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for all models"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'summary': {
                'total_models': len(self.performance_history),
                'models_needing_retrain': 0,
                'avg_accuracy': 0.0,
                'avg_r2_score': 0.0
            }
        }
        
        accuracies = []
        r2_scores = []
        needs_retrain = 0
        
        for key, perf in self.performance_history.items():
            model_report = {
                'model_name': perf.model_name,
                'model_type': perf.model_type,
                'accuracy': perf.accuracy,
                'r2_score': perf.r2_score,
                'last_updated': perf.last_updated.isoformat() if perf.last_updated else None,
                'sample_count': perf.sample_count,
                'predictions_count': perf.predictions_count,
                'needs_retraining': self._should_retrain(perf.model_name, perf.model_type)
            }
            
            report['models'][key] = model_report
            
            if perf.accuracy > 0:
                accuracies.append(perf.accuracy)
            
            if perf.r2_score is not None:
                r2_scores.append(perf.r2_score)
            
            if model_report['needs_retraining']:
                needs_retrain += 1
        
        # Calculate summary statistics
        report['summary']['models_needing_retrain'] = needs_retrain
        report['summary']['avg_accuracy'] = np.mean(accuracies) if accuracies else 0.0
        report['summary']['avg_r2_score'] = np.mean(r2_scores) if r2_scores else 0.0
        
        return report
    
    def _load_performance_history(self):
        """Load performance history from file"""
        try:
            if os.path.exists(self.performance_log_path):
                import json
                with open(self.performance_log_path, 'r') as f:
                    data = json.load(f)
                
                for key, perf_data in data.get('models', {}).items():
                    self.performance_history[key] = ModelPerformance(
                        model_name=perf_data['model_name'],
                        model_type=perf_data['model_type'],
                        accuracy=perf_data['accuracy'],
                        r2_score=perf_data.get('r2_score'),
                        last_updated=datetime.fromisoformat(perf_data['last_updated']) if perf_data.get('last_updated') else None,
                        sample_count=perf_data['sample_count'],
                        predictions_count=perf_data['predictions_count'],
                        correct_predictions=perf_data.get('correct_predictions', 0)
                    )
                
                logger.info(f"Loaded performance history for {len(self.performance_history)} models")
            else:
                logger.info("No performance history file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            self.performance_history = {}
    
    def _save_performance_history(self):
        """Save performance history to file"""
        try:
            os.makedirs(os.path.dirname(self.performance_log_path), exist_ok=True)
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'models': {}
            }
            
            for key, perf in self.performance_history.items():
                data['models'][key] = {
                    'model_name': perf.model_name,
                    'model_type': perf.model_type,
                    'accuracy': perf.accuracy,
                    'r2_score': perf.r2_score,
                    'last_updated': perf.last_updated.isoformat() if perf.last_updated else None,
                    'sample_count': perf.sample_count,
                    'predictions_count': perf.predictions_count,
                    'correct_predictions': perf.correct_predictions
                }
            
            import json
            with open(self.performance_log_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def schedule_periodic_retraining(self, symbols: List[str], interval_hours: int = 24):
        """Schedule periodic retraining of models"""
        logger.info(f"Scheduling periodic retraining every {interval_hours} hours")
        
        # This would integrate with a scheduler like APScheduler
        # For now, just log the schedule
        
        while True:
            try:
                import time
                time.sleep(interval_hours * 3600)  # Convert hours to seconds
                
                logger.info("Running scheduled model retraining...")
                results = self.auto_retrain_models(symbols)
                
                successful = sum(results.values())
                total = len(results)
                logger.info(f"Scheduled retraining complete: {successful}/{total} successful")
                
            except KeyboardInterrupt:
                logger.info("Stopping scheduled retraining...")
                break
            except Exception as e:
                logger.error(f"Error in scheduled retraining: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying


if __name__ == "__main__":
    # Test the adaptive learner
    learner = AdaptiveLearner()
    
    # Get performance report
    report = learner.get_model_performance_report()
    print(f"Performance report: {report['summary']}")
    
    # Test auto-retraining
    symbols = ["SPY", "QQQ"]
    results = learner.auto_retrain_models(symbols)
    print(f"Auto-retraining results: {results}")
