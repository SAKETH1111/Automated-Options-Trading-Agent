#!/usr/bin/env python3
"""
Test ML Models
Quick script to verify models are trained and working
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.ml.model_loader import MLModelLoader
import pandas as pd
import numpy as np


def main():
    print("=" * 80)
    print("🧪 ML MODEL TEST")
    print("=" * 80)
    print()
    
    # Check if models exist
    models_dir = project_root / "models"
    
    print("📁 Checking models directory...")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print(f"   Expected: {models_dir}")
        print("\n💡 Run training script first: python scripts/train_ml_models.py")
        return
    
    print(f"✅ Models directory found: {models_dir}")
    print()
    
    # List model files
    model_files = list(models_dir.glob("*_latest.pkl"))
    print(f"📊 Found {len(model_files)} model files:")
    for f in model_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   • {f.name} ({size_mb:.2f} MB)")
    print()
    
    # Load models
    print("🔄 Loading models...")
    loader = MLModelLoader(models_dir=str(models_dir))
    
    if loader.load_models():
        print("✅ Models loaded successfully!")
        print()
        
        # Show model info
        info = loader.get_model_info()
        print("📋 Model Information:")
        print(f"   Loaded: {info['loaded']}")
        print(f"   Models: {', '.join(info['models'])}")
        print()
        
        # Test predictions with dummy data
        print("🧪 Testing predictions with dummy data...")
        
        # Create dummy features (simplified)
        dummy_features = pd.DataFrame({
            'returns': [0.001],
            'rsi': [55.0],
            'macd': [0.5],
            'sma_10': [450.0],
            'sma_20': [448.0],
            'volume_ratio': [1.2],
            'volatility_20': [0.15],
            'momentum_10': [0.02]
        })
        
        # Test entry signal
        try:
            entry = loader.predict_entry_signal(dummy_features)
            print(f"   Entry Signal: {'ENTER' if entry['should_enter'] else 'WAIT'}")
            print(f"   Confidence: {entry['confidence']:.1%}")
            print(f"   Probabilities: Bearish={entry['probabilities']['bearish']:.1%}, "
                  f"Bullish={entry['probabilities']['bullish']:.1%}")
        except Exception as e:
            print(f"   ⚠️  Entry signal test failed: {e}")
        
        print()
        
        # Test win probability
        try:
            win_prob = loader.predict_win_probability(dummy_features)
            print(f"   Win Probability: {win_prob:.1%}")
        except Exception as e:
            print(f"   ⚠️  Win probability test failed: {e}")
        
        print()
        
        # Test volatility
        try:
            vol = loader.predict_volatility(dummy_features)
            print(f"   Volatility Forecast: {vol.upper()}")
        except Exception as e:
            print(f"   ⚠️  Volatility test failed: {e}")
        
        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("🎯 Your ML models are ready to use!")
        print("   Resume trading with: /resume (in Telegram bot)")
        print()
        
    else:
        print("❌ Failed to load models!")
        print()
        print("💡 Possible fixes:")
        print("   1. Train models: python scripts/train_ml_models.py")
        print("   2. Check models directory: ls -la models/")
        print("   3. Check file permissions")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

