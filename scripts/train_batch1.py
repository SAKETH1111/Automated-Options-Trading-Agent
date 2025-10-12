#!/usr/bin/env python3
"""
ML Model Training - Batch 1
Train primary symbols: SPY, QQQ, IWM, DIA, SQQQ
"""

import os
import sys
import argparse
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.multi_timeframe_trainer import MultiTimeframeTrainer
from src.ml.ensemble_predictor import EnsemblePredictor
from src.ml.adaptive_learner import AdaptiveLearner


def main():
    print("=" * 80)
    print("🤖 BATCH 1: Training Primary Symbols")
    print("=" * 80)
    print()
    
    # Batch 1: Primary symbols (major indices)
    # Removed SQQQ (leveraged ETF with non-standard options)
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF']
    
    logger.info("🚀 Starting Advanced ML Training - Batch 1")
    logger.info(f"Symbols: {symbols}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Train Multi-timeframe Models
        logger.info("📊 Training Multi-timeframe Models...")
        timeframe_trainer = MultiTimeframeTrainer()
        timeframe_models = timeframe_trainer.train_all_models(symbols)
        logger.info(f"✅ Trained {len(timeframe_models)} timeframe models")
        
        # Step 2: Train Ensemble Models
        logger.info("🧠 Training Ensemble Models...")
        ensemble_predictor = EnsemblePredictor()
        ensemble_models = ensemble_predictor.build_complete_ensemble_system(timeframe_models)
        logger.info(f"✅ Trained {len(ensemble_models)} ensemble models")
        
        # Step 3: Setup Adaptive Learning
        logger.info("🔄 Setting up Adaptive Learning...")
        adaptive_learner = AdaptiveLearner()
        
        # Get initial performance
        performance = adaptive_learner.get_model_performance_report()
        logger.info(f"📈 Initial Performance Report:")
        logger.info(f"   Total Models: {performance.get('total_models', 0)}")
        logger.info(f"   Avg Accuracy: {performance.get('avg_accuracy', 0):.3f}")
        logger.info(f"   Avg R² Score: {performance.get('avg_r2', 0):.3f}")
        
        # Run auto-retraining check
        retraining_results = adaptive_learner.auto_retrain_models()
        logger.info(f"🔄 Auto-retraining: {retraining_results['successful']}/{retraining_results['total']} successful")
        
        logger.info("✅ Adaptive learning system ready")
        
        # Summary
        logger.info("")
        logger.info("🎉 Batch 1 Training Complete!")
        logger.info("=" * 80)
        logger.info(f"📊 Multi-timeframe models: {len(timeframe_models)}")
        logger.info(f"   Symbols: {', '.join(symbols)}")
        logger.info("")
        logger.info(f"🧠 Ensemble models: {len(ensemble_models)}")
        logger.info("")
        logger.info(f"🔄 Adaptive learning: Active")
        logger.info("")
        logger.info(f"📁 Models saved to:")
        logger.info(f"   Timeframe models: models/multi_timeframe/")
        logger.info(f"   Ensemble models: models/ensemble/")
        logger.info(f"   Performance logs: logs/adaptive_learning.json")
        logger.info("")
        logger.info("🎯 Next: Run train_batch2.py for remaining symbols (GDX, XLF, TLT, XLE)")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

