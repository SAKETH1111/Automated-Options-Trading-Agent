#!/usr/bin/env python3
"""
ML Model Training Script for Production Deployment
Trains all ML models for the institutional options trading system

Models:
- LSTM Volatility Forecaster
- Transformer Price Predictor
- Autoencoder Anomaly Detector
- Ensemble Models (XGBoost, LightGBM, Random Forest)
- Reinforcement Learning Agents
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import joblib
import json
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import ML modules
from src.ml.data_pipeline import OptionsMLDataPipeline
from src.ml.ensemble_options import EnsembleOptionsModel
from src.ml.options_deep_learning.volatility_lstm import LSTMVolatilityForecaster
from src.ml.options_deep_learning.price_transformer import TransformerPricePredictor
from src.ml.options_deep_learning.anomaly_detector import AutoencoderAnomalyDetector
from src.ml.rl.position_sizing_agent import PositionSizingAgent
from src.ml.rl.strategy_selector import StrategySelector
from src.data.polygon_advanced_production import PolygonAdvancedProducer

class MLModelTrainer:
    """Production ML model trainer"""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize data pipeline
        self.data_pipeline = OptionsMLDataPipeline()
        
        # Initialize Polygon producer
        self.polygon_producer = None
        
        logger.info("ML Model Trainer initialized")
    
    def _load_config(self) -> dict:
        """Load configuration"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    async def initialize(self):
        """Initialize connections"""
        self.polygon_producer = PolygonAdvancedProducer(self.config_path)
        await self.polygon_producer.initialize()
        logger.info("Connections initialized")
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.polygon_producer:
            await self.polygon_producer.cleanup()
        logger.info("Connections cleaned up")
    
    async def download_historical_data(self, symbols: list, days: int = 365) -> dict:
        """Download historical data for training"""
        logger.info(f"Downloading {days} days of historical data for {symbols}")
        
        historical_data = {}
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                # Download stock data
                stock_data = await self.polygon_producer.get_historical_data(
                    symbol, start_date, end_date, "day"
                )
                
                # Download options chain
                options_chain = await self.polygon_producer.get_options_chain(symbol)
                
                # Download flat files for options data
                flat_files = {}
                for i in range(min(days, 30)):  # Limit to 30 days of flat files
                    date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                    files = await self.polygon_producer.download_flat_files(date, [symbol])
                    if files:
                        flat_files[date] = files
                
                historical_data[symbol] = {
                    'stock': stock_data,
                    'options_chain': options_chain,
                    'flat_files': flat_files
                }
                
                logger.info(f"Downloaded data for {symbol}: {len(stock_data)} stock records, {len(options_chain)} options contracts")
                
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
                historical_data[symbol] = None
        
        return historical_data
    
    def prepare_training_data(self, historical_data: dict) -> tuple:
        """Prepare training data for all models"""
        logger.info("Preparing training data")
        
        # Combine all stock data
        all_stock_data = []
        for symbol, data in historical_data.items():
            if data and 'stock' in data and not data['stock'].empty:
                df = data['stock'].copy()
                df['symbol'] = symbol
                all_stock_data.append(df)
        
        if not all_stock_data:
            raise ValueError("No stock data available for training")
        
        combined_stock_data = pd.concat(all_stock_data, ignore_index=True)
        combined_stock_data = combined_stock_data.sort_values(['symbol', 'timestamp'])
        
        # Prepare options data
        options_data = []
        for symbol, data in historical_data.items():
            if data and 'flat_files' in data:
                for date, files in data['flat_files'].items():
                    if 'trades' in files:
                        df = files['trades'].copy()
                        df['symbol'] = symbol
                        df['date'] = date
                        options_data.append(df)
        
        if options_data:
            combined_options_data = pd.concat(options_data, ignore_index=True)
        else:
            # Create synthetic options data if none available
            combined_options_data = self._create_synthetic_options_data(combined_stock_data)
        
        # Engineer features
        features = self.data_pipeline.engineer_features(combined_stock_data, combined_options_data)
        
        logger.info(f"Prepared training data: {len(features)} samples")
        return features, combined_stock_data, combined_options_data
    
    def _create_synthetic_options_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic options data for training when real data is unavailable"""
        logger.info("Creating synthetic options data")
        
        options_data = []
        for symbol in stock_data['symbol'].unique():
            symbol_data = stock_data[stock_data['symbol'] == symbol].copy()
            
            for _, row in symbol_data.iterrows():
                # Create synthetic options for different strikes and expirations
                strikes = [row['c'] * 0.9, row['c'] * 0.95, row['c'], row['c'] * 1.05, row['c'] * 1.1]
                expirations = [7, 14, 21, 30, 45, 60]
                
                for strike in strikes:
                    for dte in expirations:
                        expiration_date = row['timestamp'] + timedelta(days=dte)
                        
                        # Synthetic Greeks calculation
                        iv = 0.2 + np.random.normal(0, 0.05)
                        delta = np.random.normal(0.5, 0.2)
                        gamma = np.random.uniform(0.01, 0.05)
                        theta = -np.random.uniform(0.1, 0.5)
                        vega = np.random.uniform(0.1, 0.3)
                        
                        for option_type in ['C', 'P']:
                            options_data.append({
                                'timestamp': row['timestamp'],
                                'symbol': symbol,
                                'strike': strike,
                                'expiration': expiration_date,
                                'option_type': option_type,
                                'bid': row['c'] * 0.1,
                                'ask': row['c'] * 0.12,
                                'last': row['c'] * 0.11,
                                'volume': np.random.randint(10, 1000),
                                'open_interest': np.random.randint(100, 10000),
                                'implied_volatility': iv,
                                'delta': delta if option_type == 'C' else delta - 1,
                                'gamma': gamma,
                                'theta': theta,
                                'vega': vega,
                                'rho': np.random.uniform(-0.1, 0.1)
                            })
        
        return pd.DataFrame(options_data)
    
    async def train_lstm_volatility_model(self, features: pd.DataFrame) -> str:
        """Train LSTM volatility forecasting model"""
        logger.info("Training LSTM volatility model")
        
        model = LSTMVolatilityForecaster(
            sequence_length=60,
            hidden_units=128,
            dropout=0.2,
            learning_rate=0.001
        )
        
        # Prepare data for LSTM
        X, y = self.data_pipeline.prepare_lstm_data(features, target='volatility')
        
        # Train model
        history = model.train(X, y, epochs=100, batch_size=32, validation_split=0.2)
        
        # Save model
        model_path = self.models_dir / "volatility_lstm_production.pkl"
        joblib.dump(model, model_path)
        
        # Save training history
        history_path = self.models_dir / "volatility_lstm_history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        
        logger.info(f"LSTM volatility model saved to {model_path}")
        return str(model_path)
    
    async def train_transformer_price_model(self, features: pd.DataFrame) -> str:
        """Train Transformer price prediction model"""
        logger.info("Training Transformer price model")
        
        model = TransformerPricePredictor(
            d_model=256,
            n_heads=8,
            n_layers=6,
            dropout=0.1,
            learning_rate=0.0001
        )
        
        # Prepare data for Transformer
        X, y = self.data_pipeline.prepare_transformer_data(features, target='price')
        
        # Train model
        history = model.train(X, y, epochs=50, batch_size=16, validation_split=0.2)
        
        # Save model
        model_path = self.models_dir / "price_transformer_production.pkl"
        joblib.dump(model, model_path)
        
        # Save training history
        history_path = self.models_dir / "price_transformer_history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        
        logger.info(f"Transformer price model saved to {model_path}")
        return str(model_path)
    
    async def train_anomaly_detector(self, features: pd.DataFrame) -> str:
        """Train autoencoder anomaly detector"""
        logger.info("Training anomaly detector model")
        
        model = AutoencoderAnomalyDetector(
            input_dim=features.shape[1],
            encoding_dim=64,
            learning_rate=0.001
        )
        
        # Prepare data for autoencoder
        X = self.data_pipeline.prepare_anomaly_data(features)
        
        # Train model
        history = model.train(X, epochs=100, batch_size=32, validation_split=0.2)
        
        # Save model
        model_path = self.models_dir / "anomaly_detector_production.pkl"
        joblib.dump(model, model_path)
        
        # Save training history
        history_path = self.models_dir / "anomaly_detector_history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        
        logger.info(f"Anomaly detector model saved to {model_path}")
        return str(model_path)
    
    async def train_ensemble_models(self, features: pd.DataFrame) -> str:
        """Train ensemble models"""
        logger.info("Training ensemble models")
        
        model = EnsembleOptionsModel()
        
        # Prepare data for ensemble
        X, y = self.data_pipeline.prepare_ensemble_data(features)
        
        # Train ensemble
        results = model.train(X, y)
        
        # Save model
        model_path = self.models_dir / "ensemble_production.pkl"
        joblib.dump(model, model_path)
        
        # Save results
        results_path = self.models_dir / "ensemble_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        logger.info(f"Ensemble models saved to {model_path}")
        return str(model_path)
    
    async def train_rl_agents(self, features: pd.DataFrame) -> dict:
        """Train reinforcement learning agents"""
        logger.info("Training RL agents")
        
        # Train position sizing agent
        position_agent = PositionSizingAgent()
        position_agent.train(features, episodes=1000)
        
        position_model_path = self.models_dir / "position_sizing_agent_production.pkl"
        joblib.dump(position_agent, position_model_path)
        
        # Train strategy selector
        strategy_selector = StrategySelector()
        strategy_selector.train(features, episodes=500)
        
        strategy_model_path = self.models_dir / "strategy_selector_production.pkl"
        joblib.dump(strategy_selector, strategy_model_path)
        
        logger.info(f"RL agents saved to {position_model_path} and {strategy_model_path}")
        return {
            'position_sizing': str(position_model_path),
            'strategy_selector': str(strategy_model_path)
        }
    
    def create_model_metadata(self, model_paths: dict) -> dict:
        """Create model metadata"""
        metadata = {
            'training_date': datetime.now().isoformat(),
            'config_path': self.config_path,
            'models': model_paths,
            'training_data_info': {
                'symbols': ['SPY', 'QQQ', 'IWM'],
                'period_days': 365,
                'features_count': 0,  # Will be updated
                'samples_count': 0    # Will be updated
            },
            'model_versions': {
                'lstm_volatility': '1.0.0',
                'transformer_price': '1.0.0',
                'anomaly_detector': '1.0.0',
                'ensemble': '1.0.0',
                'position_sizing_agent': '1.0.0',
                'strategy_selector': '1.0.0'
            },
            'performance_metrics': {
                'lstm_volatility': {'mse': 0.0, 'mae': 0.0},
                'transformer_price': {'mse': 0.0, 'mae': 0.0},
                'anomaly_detector': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0},
                'ensemble': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
            }
        }
        
        # Save metadata
        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        return metadata
    
    async def validate_models(self, model_paths: dict) -> dict:
        """Validate trained models"""
        logger.info("Validating trained models")
        
        validation_results = {}
        
        for model_name, model_path in model_paths.items():
            try:
                # Load model
                model = joblib.load(model_path)
                
                # Basic validation
                if hasattr(model, 'predict'):
                    # Create dummy data for testing
                    dummy_data = np.random.randn(10, 50)
                    predictions = model.predict(dummy_data)
                    
                    validation_results[model_name] = {
                        'status': 'valid',
                        'predictions_shape': predictions.shape,
                        'model_type': type(model).__name__
                    }
                else:
                    validation_results[model_name] = {
                        'status': 'warning',
                        'message': 'Model does not have predict method'
                    }
                
            except Exception as e:
                validation_results[model_name] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # Save validation results
        validation_path = self.models_dir / "validation_results.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to {validation_path}")
        return validation_results
    
    async def train_all_models(self):
        """Train all ML models"""
        logger.info("Starting ML model training pipeline")
        
        try:
            # Download historical data
            symbols = ['SPY', 'QQQ', 'IWM']
            historical_data = await self.download_historical_data(symbols, days=365)
            
            # Prepare training data
            features, stock_data, options_data = self.prepare_training_data(historical_data)
            
            # Train models
            model_paths = {}
            
            # LSTM Volatility Model
            lstm_path = await self.train_lstm_volatility_model(features)
            model_paths['lstm_volatility'] = lstm_path
            
            # Transformer Price Model
            transformer_path = await self.train_transformer_price_model(features)
            model_paths['transformer_price'] = transformer_path
            
            # Anomaly Detector
            anomaly_path = await self.train_anomaly_detector(features)
            model_paths['anomaly_detector'] = anomaly_path
            
            # Ensemble Models
            ensemble_path = await self.train_ensemble_models(features)
            model_paths['ensemble'] = ensemble_path
            
            # RL Agents
            rl_paths = await self.train_rl_agents(features)
            model_paths.update(rl_paths)
            
            # Create metadata
            metadata = self.create_model_metadata(model_paths)
            
            # Validate models
            validation_results = await self.validate_models(model_paths)
            
            logger.info("ML model training completed successfully")
            return {
                'status': 'success',
                'model_paths': model_paths,
                'metadata': metadata,
                'validation': validation_results
            }
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

async def main():
    """Main training function"""
    trainer = MLModelTrainer()
    
    try:
        await trainer.initialize()
        results = await trainer.train_all_models()
        
        if results['status'] == 'success':
            logger.info("All models trained successfully!")
            print("\n" + "="*50)
            print("ML MODEL TRAINING COMPLETED")
            print("="*50)
            print(f"Models trained: {len(results['model_paths'])}")
            print(f"Validation status: {results['validation']}")
            print("="*50)
        else:
            logger.error(f"Training failed: {results['message']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)
    finally:
        await trainer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())