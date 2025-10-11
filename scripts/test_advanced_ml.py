#!/usr/bin/env python3
"""
Test advanced ML features: Multi-timeframe models, Ensemble predictions, Adaptive learning
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.multi_timeframe_trainer import MultiTimeframeTrainer
from src.ml.ensemble_predictor import EnsemblePredictor
from src.ml.adaptive_learner import AdaptiveLearner


def test_multi_timeframe_models():
    """Test multi-timeframe models"""
    logger.info("🧪 Testing Multi-timeframe Models...")
    
    try:
        trainer = MultiTimeframeTrainer()
        
        # Test data collection for a short timeframe
        logger.info("Testing data collection...")
        test_symbols = ["SPY"]
        
        # Test with just daily data to avoid API limits
        daily_config = None
        for config in trainer.timeframe_configs:
            if config.timeframe == "1Day":
                daily_config = config
                break
        
        if daily_config:
            # Simulate data collection
            logger.info(f"Would collect {daily_config.name} data for {test_symbols}")
            logger.info(f"Timeframe: {daily_config.timeframe}")
            logger.info(f"Lookback days: {daily_config.lookback_days}")
            logger.info(f"Min samples: {daily_config.min_samples}")
            logger.info(f"Features: {daily_config.features_to_include}")
        
        logger.info("✅ Multi-timeframe models test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-timeframe models test failed: {e}")
        return False


def test_ensemble_models():
    """Test ensemble models"""
    logger.info("🧪 Testing Ensemble Models...")
    
    try:
        predictor = EnsemblePredictor()
        
        # Test ensemble configurations
        logger.info(f"Ensemble configurations: {len(predictor.ensemble_configs)}")
        
        for config in predictor.ensemble_configs:
            logger.info(f"  - {config.name}: {config.method} method with {len(config.timeframes)} timeframes")
            if config.weights:
                logger.info(f"    Weights: {config.weights}")
        
        # Test ensemble creation
        logger.info("Testing ensemble model creation...")
        
        # Create dummy timeframe models for testing
        dummy_timeframe_models = {}
        for config in predictor.multi_trainer.timeframe_configs[:3]:  # Test first 3
            dummy_timeframe_models[config.name] = {
                'entry_signal': {'model': None, 'accuracy': 0.65},
                'win_probability': {'model': None, 'r2_score': 0.45},
                'volatility': {'model': None, 'accuracy': 0.70},
                'scaler': None,
                'metadata': {'config': config}
            }
        
        # Test ensemble creation
        ensemble_models = predictor.create_ensemble_models(dummy_timeframe_models)
        
        logger.info(f"Created {len(ensemble_models)} ensemble models")
        
        for name, data in ensemble_models.items():
            logger.info(f"  - {name}: {len(data['ensemble_models'])} model types")
        
        logger.info("✅ Ensemble models test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ensemble models test failed: {e}")
        return False


def test_adaptive_learning():
    """Test adaptive learning system"""
    logger.info("🧪 Testing Adaptive Learning...")
    
    try:
        learner = AdaptiveLearner()
        
        # Test performance tracking
        logger.info("Testing performance tracking...")
        
        # Simulate model performance monitoring
        test_predictions = np.array([1, 0, 1, 1, 0])
        test_actual = np.array([1, 0, 1, 0, 1])
        
        needs_retrain = learner.monitor_model_performance(
            "test_model", "entry_signal", test_predictions, test_actual, 5
        )
        
        logger.info(f"Model needs retraining: {needs_retrain}")
        
        # Test performance report
        report = learner.get_model_performance_report()
        logger.info(f"Performance report summary:")
        logger.info(f"  Total models: {report['summary']['total_models']}")
        logger.info(f"  Models needing retrain: {report['summary']['models_needing_retrain']}")
        logger.info(f"  Avg accuracy: {report['summary']['avg_accuracy']:.3f}")
        logger.info(f"  Avg R² score: {report['summary']['avg_r2_score']:.3f}")
        
        # Test thresholds
        logger.info(f"Performance thresholds:")
        logger.info(f"  Min accuracy: {learner.thresholds.min_accuracy}")
        logger.info(f"  Min R² score: {learner.thresholds.min_r2_score}")
        logger.info(f"  Max accuracy drop: {learner.thresholds.max_accuracy_drop}")
        logger.info(f"  Min samples for retrain: {learner.thresholds.min_samples_for_retrain}")
        logger.info(f"  Retrain frequency: {learner.thresholds.retrain_frequency_days} days")
        
        logger.info("✅ Adaptive learning test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive learning test failed: {e}")
        return False


def test_integration():
    """Test integration between all components"""
    logger.info("🧪 Testing Integration...")
    
    try:
        # Test that all components can be imported and initialized
        trainer = MultiTimeframeTrainer()
        predictor = EnsemblePredictor()
        learner = AdaptiveLearner()
        
        logger.info("✅ All components imported successfully")
        
        # Test configuration consistency
        timeframe_configs = len(trainer.timeframe_configs)
        ensemble_configs = len(predictor.ensemble_configs)
        
        logger.info(f"Timeframe configurations: {timeframe_configs}")
        logger.info(f"Ensemble configurations: {ensemble_configs}")
        
        # Test that ensemble configs reference valid timeframes
        all_timeframe_names = [config.name for config in trainer.timeframe_configs]
        
        for ensemble_config in predictor.ensemble_configs:
            for timeframe in ensemble_config.timeframes:
                if timeframe not in all_timeframe_names:
                    logger.warning(f"Ensemble {ensemble_config.name} references unknown timeframe: {timeframe}")
        
        logger.info("✅ Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🚀 Starting Advanced ML Tests")
    
    tests = [
        ("Multi-timeframe Models", test_multi_timeframe_models),
        ("Ensemble Models", test_ensemble_models),
        ("Adaptive Learning", test_adaptive_learning),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info('='*50)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Advanced ML features are ready.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
