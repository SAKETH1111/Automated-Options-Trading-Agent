"""
ML Model Trainer
Comprehensive training pipeline for all ML models
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from loguru import logger

from .data_collector import HistoricalDataCollector
from .polygon_data_collector import PolygonDataCollector


class ModelTrainer:
    """
    Train and evaluate ML models on historical data
    Handles multiple model types and strategies
    """
    
    def __init__(self, db_session: Session, models_dir: str = "models"):
        """
        Initialize model trainer
        
        Args:
            db_session: Database session
            models_dir: Directory to save trained models
        """
        self.db = db_session
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Try to use Polygon if available, fallback to regular collector
        try:
            self.data_collector = PolygonDataCollector(db_session)
            logger.info("Using Polygon.io for data collection")
        except Exception as e:
            logger.warning(f"Polygon not available ({e}), using fallback collector")
            self.data_collector = HistoricalDataCollector(db_session)
        
        # Models
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.training_metadata = {}
        
        logger.info(f"Model Trainer initialized - models will be saved to {self.models_dir}")
    
    def train_all_models(
        self,
        symbols: List[str] = ['SPY', 'QQQ'],
        lookback_days: int = 365,
        strategy: str = 'bull_put_spread',
        timeframe: str = '1Day'
    ) -> Dict:
        """
        Train all ML models
        
        Args:
            symbols: Symbols to train on
            lookback_days: Days of historical data
            strategy: Strategy to optimize for
            
        Returns:
            Training results
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING COMPREHENSIVE ML TRAINING")
            logger.info("=" * 80)
            
            results = {}
            
            # 1. Collect training data
            logger.info("\n📊 Step 1/5: Collecting historical data...")
            df = self.data_collector.collect_training_data(
                symbols=symbols,
                lookback_days=lookback_days,
                timeframe=timeframe
            )
            
            if df.empty:
                logger.error("No data collected!")
                return {'success': False, 'error': 'No data'}
            
            logger.info(f"✅ Collected {len(df):,} samples")
            
            # 2. Create labels
            logger.info("\n🏷️  Step 2/5: Creating labels...")
            df = self.data_collector.create_labels(
                df,
                strategy=strategy,
                forward_periods=5,  # Reduced from 20 to 5 for daily data
                profit_threshold=0.01  # Adjusted for daily timeframe
            )
            
            # Drop rows with NaN (but keep enough for training)
            initial_count = len(df)
            
            # First, let's see which columns have the most NaN
            nan_counts = df.isna().sum()
            high_nan_cols = nan_counts[nan_counts > len(df) * 0.8].index.tolist()
            
            if high_nan_cols:
                logger.info(f"Dropping columns with >80% NaN: {high_nan_cols}")
                df = df.drop(columns=high_nan_cols)
            
            # Now drop rows with any remaining NaN
            df = df.dropna()
            dropped = initial_count - len(df)
            logger.info(f"✅ Created labels - {len(df):,} samples after cleaning (dropped {dropped} NaN rows)")
            
            if len(df) < 50:
                logger.error(f"Insufficient data after cleaning: {len(df)} samples (need at least 50)")
                logger.info("Try increasing lookback_days or using different timeframe")
                return {'success': False, 'error': 'Insufficient data after cleaning'}
            
            # 3. Split data
            logger.info("\n✂️  Step 3/5: Splitting data...")
            train_df, val_df, test_df = self.data_collector.split_train_test(df)
            
            # 4. Train models
            logger.info("\n🤖 Step 4/5: Training models...")
            
            # Train Entry Signal Model
            logger.info("\n  📈 Training Entry Signal Model...")
            entry_results = self._train_entry_model(train_df, val_df, test_df)
            results['entry_model'] = entry_results
            
            # Train Win Probability Model
            logger.info("\n  🎯 Training Win Probability Model...")
            win_prob_results = self._train_win_probability_model(train_df, val_df, test_df)
            results['win_probability_model'] = win_prob_results
            
            # Train Volatility Forecaster
            logger.info("\n  📊 Training Volatility Forecaster...")
            volatility_results = self._train_volatility_model(train_df, val_df, test_df)
            results['volatility_model'] = volatility_results
            
            # 5. Save models
            logger.info("\n💾 Step 5/5: Saving models...")
            self._save_all_models()
            
            # Summary
            logger.info("\n" + "=" * 80)
            logger.info("TRAINING COMPLETE! 🎉")
            logger.info("=" * 80)
            logger.info(f"\nEntry Model Accuracy: {entry_results.get('test_accuracy', 0):.2%}")
            logger.info(f"Win Probability R²: {win_prob_results.get('test_r2', 0):.3f}")
            logger.info(f"Volatility MAE: {volatility_results.get('test_mae', 0):.4f}")
            logger.info(f"\nModels saved to: {self.models_dir}/")
            
            results['success'] = True
            results['trained_on'] = len(train_df)
            results['tested_on'] = len(test_df)
            results['symbols'] = symbols
            results['timestamp'] = datetime.now().isoformat()
            
            # Save training summary
            self._save_training_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _train_entry_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """Train entry signal prediction model"""
        try:
            # Prepare features
            feature_cols = self._get_feature_columns(train_df)
            
            X_train = train_df[feature_cols]
            y_train = train_df['label']
            
            X_val = val_df[feature_cols]
            y_val = val_df['label']
            
            X_test = test_df[feature_cols]
            y_test = test_df['label']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            logger.info("    Training Random Forest...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val_scaled)
            val_proba = model.predict_proba(X_val_scaled)
            
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)
            val_auc = roc_auc_score(y_val, val_proba[:, 1])
            
            logger.info(f"    Validation - Accuracy: {val_accuracy:.2%}, F1: {val_f1:.2%}, AUC: {val_auc:.3f}")
            
            # Evaluate on test set
            test_pred = model.predict(X_test_scaled)
            test_proba = model.predict_proba(X_test_scaled)
            
            test_accuracy = accuracy_score(y_test, test_pred)
            test_precision = precision_score(y_test, test_pred, zero_division=0)
            test_recall = recall_score(y_test, test_pred, zero_division=0)
            test_f1 = f1_score(y_test, test_pred, zero_division=0)
            test_auc = roc_auc_score(y_test, test_proba[:, 1])
            
            logger.info(f"    Test - Accuracy: {test_accuracy:.2%}, F1: {test_f1:.2%}, AUC: {test_auc:.3f}")
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            logger.info("    Top 10 Features:")
            for feat, imp in top_features:
                logger.info(f"      {feat}: {imp:.4f}")
            
            # Save model and scaler
            self.models['entry_signal'] = model
            self.scalers['entry_signal'] = scaler
            self.feature_names['entry_signal'] = feature_cols
            
            return {
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'top_features': top_features[:10],
                'n_features': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Error training entry model: {e}")
            return {'error': str(e)}
    
    def _train_win_probability_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """Train win probability prediction model"""
        try:
            # Prepare features
            feature_cols = self._get_feature_columns(train_df)
            
            X_train = train_df[feature_cols]
            y_train = train_df['win_probability']
            
            X_val = val_df[feature_cols]
            y_val = val_df['win_probability']
            
            X_test = test_df[feature_cols]
            y_test = test_df['win_probability']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest Regressor
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("    Training Random Forest Regressor...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val_scaled)
            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            logger.info(f"    Validation - R²: {val_r2:.3f}, MAE: {val_mae:.4f}")
            
            # Evaluate on test set
            test_pred = model.predict(X_test_scaled)
            test_mse = mean_squared_error(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            logger.info(f"    Test - R²: {test_r2:.3f}, MAE: {test_mae:.4f}")
            
            # Save model and scaler
            self.models['win_probability'] = model
            self.scalers['win_probability'] = scaler
            self.feature_names['win_probability'] = feature_cols
            
            return {
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'n_features': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Error training win probability model: {e}")
            return {'error': str(e)}
    
    def _train_volatility_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """Train volatility forecasting model"""
        try:
            # Create volatility target (next period volatility)
            for df in [train_df, val_df, test_df]:
                df['future_volatility'] = df['returns'].rolling(20).std().shift(-20)
            
            train_df = train_df.dropna(subset=['future_volatility'])
            val_df = val_df.dropna(subset=['future_volatility'])
            test_df = test_df.dropna(subset=['future_volatility'])
            
            # Prepare features
            feature_cols = self._get_feature_columns(train_df)
            
            X_train = train_df[feature_cols]
            y_train = train_df['future_volatility']
            
            X_val = val_df[feature_cols]
            y_val = val_df['future_volatility']
            
            X_test = test_df[feature_cols]
            y_test = test_df['future_volatility']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Gradient Boosting Regressor
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            # Convert continuous volatility to classes (low, medium, high)
            def vol_to_class(vol):
                if vol < 0.01:
                    return 0  # Low
                elif vol < 0.02:
                    return 1  # Medium
                else:
                    return 2  # High
            
            y_train_class = y_train.apply(vol_to_class)
            y_val_class = y_val.apply(vol_to_class)
            y_test_class = y_test.apply(vol_to_class)
            
            logger.info("    Training Gradient Boosting...")
            model.fit(X_train_scaled, y_train_class)
            
            # Evaluate
            val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val_class, val_pred)
            
            test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test_class, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred * 0.01)  # Approximate
            
            logger.info(f"    Validation Accuracy: {val_accuracy:.2%}")
            logger.info(f"    Test Accuracy: {test_accuracy:.2%}")
            
            # Save model and scaler
            self.models['volatility'] = model
            self.scalers['volatility'] = scaler
            self.feature_names['volatility'] = feature_cols
            
            return {
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
                'test_mae': test_mae,
                'n_features': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Error training volatility model: {e}")
            return {'error': str(e)}
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns"""
        exclude_cols = [
            'timestamp', 'symbol', 'label', 'label_multiclass',
            'win_probability', 'future_return', 'future_volatility',
            'future_price', 'open', 'high', 'low', 'close', 'volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def _save_all_models(self):
        """Save all trained models to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for model_name, model in self.models.items():
                # Save model
                model_path = self.models_dir / f"{model_name}_{timestamp}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"    Saved {model_name} to {model_path}")
                
                # Save latest symlink
                latest_path = self.models_dir / f"{model_name}_latest.pkl"
                joblib.dump(model, latest_path)
                
                # Save scaler
                if model_name in self.scalers:
                    scaler_path = self.models_dir / f"{model_name}_scaler_{timestamp}.pkl"
                    joblib.dump(self.scalers[model_name], scaler_path)
                    
                    latest_scaler = self.models_dir / f"{model_name}_scaler_latest.pkl"
                    joblib.dump(self.scalers[model_name], latest_scaler)
                
                # Save feature names
                if model_name in self.feature_names:
                    features_path = self.models_dir / f"{model_name}_features_{timestamp}.json"
                    with open(features_path, 'w') as f:
                        json.dump(self.feature_names[model_name], f, indent=2)
                    
                    latest_features = self.models_dir / f"{model_name}_features_latest.json"
                    with open(latest_features, 'w') as f:
                        json.dump(self.feature_names[model_name], f, indent=2)
            
            logger.info("✅ All models saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _save_training_summary(self, results: Dict):
        """Save training summary to file"""
        try:
            summary_path = self.models_dir / "training_summary.json"
            
            # Load existing summaries
            summaries = []
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summaries = json.load(f)
            
            # Add new summary
            summaries.append(results)
            
            # Save
            with open(summary_path, 'w') as f:
                json.dump(summaries, f, indent=2)
            
            logger.info(f"Training summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
    
    def load_models(self) -> bool:
        """Load latest trained models"""
        try:
            for model_name in ['entry_signal', 'win_probability', 'volatility']:
                model_path = self.models_dir / f"{model_name}_latest.pkl"
                scaler_path = self.models_dir / f"{model_name}_scaler_latest.pkl"
                features_path = self.models_dir / f"{model_name}_features_latest.json"
                
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
                
                if scaler_path.exists():
                    self.scalers[model_name] = joblib.load(scaler_path)
                
                if features_path.exists():
                    with open(features_path, 'r') as f:
                        self.feature_names[model_name] = json.load(f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

