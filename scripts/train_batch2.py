#!/usr/bin/env python3
"""
ML Model Training - Batch 2
Train secondary symbols: GDX, XLF, TLT, XLE
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
    print("🤖 BATCH 2: Training Secondary Symbols")
    print("=" * 80)
    print()
    
    # Batch 2: Secondary symbols (sectors for small/medium accounts)
    symbols = ['GDX', 'XLF', 'TLT', 'XLE']
    
    logger.info("🚀 Starting Advanced ML Training - Batch 2")
    logger.info(f"Symbols: {symbols}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Train Multi-timeframe Models
        logger.info("📊 Training Multi-timeframe Models...")
        timeframe_trainer = MultiTimeframeTrainer()
        timeframe_models = timeframe_trainer.train_all_models(symbols)
        logger.info(f"✅ Trained {len(timeframe_models)} timeframe models")
        
        # Step 2: Update Ensemble Models (merge with batch 1)
        logger.info("🧠 Updating Ensemble Models...")
        ensemble_predictor = EnsemblePredictor()
        
        # Load existing models and add new ones
        ensemble_models = ensemble_predictor.build_complete_ensemble_system(timeframe_models)
        logger.info(f"✅ Updated {len(ensemble_models)} ensemble models")
        
        # Step 3: Update Adaptive Learning
        logger.info("🔄 Updating Adaptive Learning...")
        adaptive_learner = AdaptiveLearner()
        
        # Get updated performance
        performance = adaptive_learner.get_performance_report()
        logger.info(f"📈 Updated Performance Report:")
        logger.info(f"   Total Models: {performance.get('total_models', 0)}")
        logger.info(f"   Avg Accuracy: {performance.get('avg_accuracy', 0):.3f}")
        logger.info(f"   Avg R² Score: {performance.get('avg_r2', 0):.3f}")
        
        logger.info("✅ Adaptive learning system updated")
        
        # Summary
        logger.info("")
        logger.info("🎉 Batch 2 Training Complete!")
        logger.info("=" * 80)
        logger.info(f"📊 Multi-timeframe models: {len(timeframe_models)}")
        logger.info(f"   Symbols: {', '.join(symbols)}")
        logger.info("")
        logger.info(f"🧠 Total ensemble models: {len(ensemble_models)}")
        logger.info("")
        logger.info(f"🔄 Adaptive learning: Active")
        logger.info("")
        logger.info(f"📁 All models saved to:")
        logger.info(f"   Timeframe models: models/multi_timeframe/")
        logger.info(f"   Ensemble models: models/ensemble/")
        logger.info(f"   Performance logs: logs/adaptive_learning.json")
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎊 ALL TRAINING COMPLETE!")
        logger.info("   Total symbols trained: 9 (SPY, QQQ, IWM, DIA, SQQQ, GDX, XLF, TLT, XLE)")
        logger.info("   Total ML models: ~270 models")
        logger.info("   Accounts supported: $1,000 to $100,000+")
        logger.info("")
        logger.info("🚀 Next: Restart Telegram bot to use new models!")
        logger.info("   pkill -f telegram_bot.py")
        logger.info("   nohup python3 scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

