#!/usr/bin/env python3
"""
Simple test script for ML components without pagers
"""

import sys
sys.path.append('.')

print("🧪 Testing Advanced ML Components...")
print("=" * 50)

# Test 1: Multi-timeframe trainer
try:
    from src.ml.multi_timeframe_trainer import MultiTimeframeTrainer
    trainer = MultiTimeframeTrainer()
    print(f"✅ Multi-timeframe trainer: {len(trainer.timeframe_configs)} configurations")
    for config in trainer.timeframe_configs[:3]:
        print(f"   - {config.name}: {config.timeframe}")
except Exception as e:
    print(f"❌ Multi-timeframe trainer error: {e}")

# Test 2: Ensemble predictor
try:
    from src.ml.ensemble_predictor import EnsemblePredictor
    predictor = EnsemblePredictor()
    print(f"✅ Ensemble predictor: {len(predictor.ensemble_configs)} configurations")
    for config in predictor.ensemble_configs:
        print(f"   - {config.name}: {config.method}")
except Exception as e:
    print(f"❌ Ensemble predictor error: {e}")

# Test 3: Adaptive learner
try:
    from src.ml.adaptive_learner import AdaptiveLearner
    learner = AdaptiveLearner()
    print(f"✅ Adaptive learner initialized")
    print(f"   - Min accuracy: {learner.thresholds.min_accuracy}")
    print(f"   - Retrain frequency: {learner.thresholds.retrain_frequency_days} days")
except Exception as e:
    print(f"❌ Adaptive learner error: {e}")

print("=" * 50)
print("🎉 All ML components tested!")
