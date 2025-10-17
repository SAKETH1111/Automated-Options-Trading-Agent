#!/usr/bin/env python3
"""
Automatic ML Training System

This module handles automatic end-of-day ML training:
- Scheduled training at market close
- Model retraining based on performance
- Data collection and preprocessing
- Model evaluation and selection
"""

import os
import sys
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.data_pipeline import OptionsMLDataPipeline
from src.ml.backtesting import OptionsBacktester
from src.trading.account_adaptation import AccountAdaptationSystem
from src.market_data.polygon_options import PolygonOptionsClient
from src.market_data.collector import MarketDataCollector

class AutoMLTrainingSystem:
    """
    Automatic ML training system that runs end-of-day
    """
    
    def __init__(self, account_balance: float, config: Optional[Dict] = None):
        self.account_balance = account_balance
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.account_adaptation = AccountAdaptationSystem(account_balance)
        self.polygon_client = PolygonOptionsClient()
        self.market_collector = MarketDataCollector()
        self.ml_pipeline = OptionsMLDataPipeline()
        self.backtester = OptionsBacktester()
        
        # Training state
        self.is_training = False
        self.last_training = None
        self.training_thread = None
        self.stop_training = False
        
        # Model storage
        self.model_dir = Path("models/auto_training")
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info("AutoML Training System initialized")
        logger.info(f"Account Balance: ${account_balance:,.2f}")
        logger.info(f"Training Schedule: {self.config['training_schedule']}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "training_schedule": "16:30",  # 30 minutes after market close
            "retrain_threshold": 0.05,  # Retrain if performance drops 5%
            "min_training_data_days": 7,
            "max_training_data_days": 90,
            "model_types": ["random_forest", "xgboost", "neural_network"],
            "validation_metrics": ["accuracy", "precision", "recall", "f1", "sharpe_ratio"],
            "min_validation_score": 0.6,
            "backup_models": 3,
            "auto_deploy": True,
            "notification_email": None
        }
    
    def start_auto_training(self):
        """Start the automatic training scheduler"""
        logger.info("Starting automatic ML training scheduler...")
        
        # Schedule daily training
        schedule.every().day.at(self.config["training_schedule"]).do(self._scheduled_training)
        
        # Schedule weekly model evaluation
        schedule.every().monday.at("17:00").do(self._evaluate_models)
        
        # Start scheduler in background thread
        self.training_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.training_thread.start()
        
        logger.info("Auto training scheduler started")
    
    def stop_auto_training(self):
        """Stop the automatic training scheduler"""
        logger.info("Stopping automatic ML training scheduler...")
        self.stop_training = True
        schedule.clear()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        
        logger.info("Auto training scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in background"""
        while not self.stop_training:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _scheduled_training(self):
        """Scheduled training function"""
        if self.is_training:
            logger.warning("Training already in progress, skipping scheduled training")
            return
        
        logger.info("Starting scheduled ML training...")
        self.train_models()
    
    def _evaluate_models(self):
        """Evaluate model performance and retrain if needed"""
        logger.info("Evaluating model performance...")
        
        try:
            # Load current model performance
            performance_file = self.model_dir / "model_performance.json"
            if not performance_file.exists():
                logger.warning("No performance data found, skipping evaluation")
                return
            
            # Check if retraining is needed
            if self._should_retrain():
                logger.info("Performance degradation detected, retraining models...")
                self.train_models(force_retrain=True)
            else:
                logger.info("Model performance is acceptable, no retraining needed")
                
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        try:
            import json
            
            performance_file = self.model_dir / "model_performance.json"
            if not performance_file.exists():
                return True
            
            with open(performance_file, 'r') as f:
                performance = json.load(f)
            
            # Check if performance has degraded
            current_score = performance.get('current_score', 0)
            baseline_score = performance.get('baseline_score', 0)
            
            degradation = (baseline_score - current_score) / baseline_score
            return degradation > self.config["retrain_threshold"]
            
        except Exception as e:
            logger.error(f"Error checking retrain condition: {e}")
            return True
    
    def train_models(self, force_retrain: bool = False):
        """Train ML models with current data"""
        if self.is_training and not force_retrain:
            logger.warning("Training already in progress")
            return
        
        self.is_training = True
        start_time = datetime.now()
        
        try:
            logger.info("Starting ML model training...")
            
            # Get recommended symbols based on account size
            symbols = self.account_adaptation.get_recommended_symbols()
            logger.info(f"Training on symbols: {symbols}")
            
            # Get training configuration
            ml_config = self.account_adaptation.get_ml_training_config()
            logger.info(f"ML Config: {ml_config}")
            
            # Collect recent data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=ml_config["lookback_days"])
            
            logger.info(f"Collecting data from {start_date} to {end_date}")
            
            # Create comprehensive dataset
            dataset = self.ml_pipeline.create_comprehensive_dataset(
                symbols=symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                save_features=True
            )
            
            if dataset.empty:
                logger.error("No data available for training")
                return
            
            logger.info(f"Training dataset created: {len(dataset)} records")
            
            # Train multiple models
            trained_models = {}
            model_performance = {}
            
            for model_type in self.config["model_types"]:
                logger.info(f"Training {model_type} model...")
                
                try:
                    # Train model
                    model, metrics = self.ml_pipeline.train_model(
                        dataset=dataset,
                        model_type=model_type,
                        validation_split=ml_config["validation_split"],
                        test_split=ml_config["test_split"]
                    )
                    
                    # Save model
                    model_path = self.model_dir / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    self.ml_pipeline.save_model(model, model_path)
                    
                    trained_models[model_type] = model_path
                    model_performance[model_type] = metrics
                    
                    logger.info(f"{model_type} model trained successfully: {metrics}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_type} model: {e}")
                    continue
            
            # Select best model
            if trained_models:
                best_model_type = self._select_best_model(model_performance)
                logger.info(f"Best model: {best_model_type}")
                
                # Deploy best model
                if self.config["auto_deploy"]:
                    self._deploy_model(trained_models[best_model_type], best_model_type)
                
                # Save performance data
                self._save_performance_data(model_performance, best_model_type)
                
                # Cleanup old models
                self._cleanup_old_models()
            
            self.last_training = datetime.now()
            training_time = datetime.now() - start_time
            logger.info(f"ML training completed in {training_time}")
            
        except Exception as e:
            logger.error(f"Error during ML training: {e}")
        finally:
            self.is_training = False
    
    def _select_best_model(self, model_performance: Dict) -> str:
        """Select the best performing model"""
        best_score = -1
        best_model = None
        
        for model_type, metrics in model_performance.items():
            # Use weighted score combining multiple metrics
            score = self._calculate_model_score(metrics)
            
            if score > best_score:
                best_score = score
                best_model = model_type
        
        return best_model
    
    def _calculate_model_score(self, metrics: Dict) -> float:
        """Calculate overall model score from metrics"""
        weights = {
            "accuracy": 0.3,
            "precision": 0.2,
            "recall": 0.2,
            "f1": 0.2,
            "sharpe_ratio": 0.1
        }
        
        score = 0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            score += value * weight
        
        return score
    
    def _deploy_model(self, model_path: Path, model_type: str):
        """Deploy the best model for production use"""
        try:
            # Copy to production location
            prod_path = Path("models") / f"production_{model_type}.pkl"
            import shutil
            shutil.copy2(model_path, prod_path)
            
            # Update model metadata
            metadata = {
                "model_type": model_type,
                "deployed_at": datetime.now().isoformat(),
                "model_path": str(prod_path),
                "account_balance": self.account_balance,
                "account_tier": self.account_adaptation.current_tier.name
            }
            
            import json
            with open("models/model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model deployed: {model_type} -> {prod_path}")
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
    
    def _save_performance_data(self, model_performance: Dict, best_model: str):
        """Save model performance data"""
        try:
            import json
            
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "best_model": best_model,
                "all_models": model_performance,
                "account_balance": self.account_balance,
                "account_tier": self.account_adaptation.current_tier.name
            }
            
            # Save current performance
            with open(self.model_dir / "model_performance.json", "w") as f:
                json.dump(performance_data, f, indent=2)
            
            # Update baseline if this is the first training
            baseline_file = self.model_dir / "baseline_performance.json"
            if not baseline_file.exists():
                with open(baseline_file, "w") as f:
                    json.dump(performance_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _cleanup_old_models(self):
        """Cleanup old model files"""
        try:
            model_files = list(self.model_dir.glob("*.pkl"))
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent models
            keep_count = self.config["backup_models"]
            for old_model in model_files[keep_count:]:
                old_model.unlink()
                logger.info(f"Cleaned up old model: {old_model}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            "is_training": self.is_training,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "account_tier": self.account_adaptation.current_tier.name,
            "account_balance": self.account_balance,
            "next_scheduled": schedule.next_run().isoformat() if schedule.jobs else None
        }
    
    def force_training(self):
        """Force immediate training"""
        logger.info("Forcing immediate ML training...")
        self.train_models(force_retrain=True)

# Example usage
if __name__ == "__main__":
    # Test the auto training system
    auto_ml = AutoMLTrainingSystem(account_balance=25000)
    
    print("AutoML Training System Test")
    print("=" * 50)
    print(f"Account Balance: ${auto_ml.account_balance:,}")
    print(f"Account Tier: {auto_ml.account_adaptation.current_tier.name}")
    print(f"Recommended Symbols: {auto_ml.account_adaptation.get_recommended_symbols()}")
    print(f"ML Config: {auto_ml.account_adaptation.get_ml_training_config()}")
    
    # Start auto training
    auto_ml.start_auto_training()
    
    print("\nAuto training started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        auto_ml.stop_auto_training()
        print("\nAuto training stopped.")

