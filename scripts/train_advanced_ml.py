#!/usr/bin/env python3
"""
Train advanced ML features: Multi-timeframe models, Ensemble predictions, Adaptive learning
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
    parser = argparse.ArgumentParser(description="Train advanced ML features")
    parser.add_argument("--symbols", nargs="+", 
                       default=["SPY", "QQQ", "IWM", "DIA", "SQQQ", "GDX", "XLF", "TLT", "XLE"],
                       help="Symbols to train on")
    parser.add_argument("--components", nargs="+", 
                       choices=["timeframe", "ensemble", "adaptive", "all"],
                       default=["all"],
                       help="Which components to train")
    parser.add_argument("--model-dir", default="models",
                       help="Directory to save models")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.info("🚀 Starting Advanced ML Training")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Components: {args.components}")
    
    try:
        # Step 1: Multi-timeframe models
        if "timeframe" in args.components or "all" in args.components:
            logger.info("📊 Training Multi-timeframe Models...")
            
            timeframe_trainer = MultiTimeframeTrainer()
            timeframe_models = timeframe_trainer.train_all_models(args.symbols)
            
            logger.info(f"✅ Trained {len(timeframe_models)} timeframe models")
        
        # Step 2: Ensemble models
        if "ensemble" in args.components or "all" in args.components:
            logger.info("🧠 Training Ensemble Models...")
            
            ensemble_predictor = EnsemblePredictor()
            ensemble_models = ensemble_predictor.build_complete_ensemble_system(args.symbols)
            
            logger.info(f"✅ Trained {len(ensemble_models)} ensemble models")
        
        # Step 3: Adaptive learning setup
        if "adaptive" in args.components or "all" in args.components:
            logger.info("🔄 Setting up Adaptive Learning...")
            
            adaptive_learner = AdaptiveLearner()
            
            # Get initial performance report
            report = adaptive_learner.get_model_performance_report()
            logger.info(f"📈 Initial Performance Report:")
            logger.info(f"   Total Models: {report['summary']['total_models']}")
            logger.info(f"   Avg Accuracy: {report['summary']['avg_accuracy']:.3f}")
            logger.info(f"   Avg R² Score: {report['summary']['avg_r2_score']:.3f}")
            
            # Test auto-retraining
            retrain_results = adaptive_learner.auto_retrain_models(args.symbols)
            successful_retrains = sum(retrain_results.values())
            total_retrains = len(retrain_results)
            logger.info(f"🔄 Auto-retraining: {successful_retrains}/{total_retrains} successful")
            
            logger.info("✅ Adaptive learning system ready")
        
        # Summary
        logger.info("🎉 Advanced ML Training Complete!")
        
        if "timeframe" in args.components or "all" in args.components:
            logger.info(f"📊 Multi-timeframe models: {len(timeframe_models) if 'timeframe_models' in locals() else 0}")
        
        if "ensemble" in args.components or "all" in args.components:
            logger.info(f"🧠 Ensemble models: {len(ensemble_models) if 'ensemble_models' in locals() else 0}")
        
        if "adaptive" in args.components or "all" in args.components:
            logger.info("🔄 Adaptive learning: Active")
        
        # Model locations
        logger.info("📁 Models saved to:")
        logger.info(f"   Timeframe models: {args.model_dir}/multi_timeframe/")
        logger.info(f"   Ensemble models: {args.model_dir}/ensemble/")
        logger.info(f"   Performance logs: logs/adaptive_learning.json")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
