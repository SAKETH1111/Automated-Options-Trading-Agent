#!/usr/bin/env python3
"""
ML Model Training with Real Options Data
Train models using actual Greeks, IV, and Open Interest from Polygon
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

from src.database.session import get_db
from src.ml.model_trainer import ModelTrainer


def main():
    """Train all ML models with options data"""
    
    # Banner
    print("=" * 80)
    print("🤖 ML TRAINING WITH REAL OPTIONS DATA (Greeks + IV)")
    print("=" * 80)
    print()
    
    # Configuration
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA']  # 4 symbols to avoid rate limiting
    lookback_days = 730  # 2 years
    strategy = 'bull_put_spread'
    timeframe = '1Day'
    
    print(f"📊 Training Configuration:")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Lookback: {lookback_days} days")
    print(f"   Timeframe: {timeframe}")
    print(f"   Strategy: {strategy}")
    print(f"   🆕 Options Data: ENABLED (Greeks + IV from Polygon)")
    print()
    
    print("⚠️  This will take 15-25 minutes (fetching options data takes longer)...")
    print()
    
    # Initialize database
    logger.info("Initializing database...")
    db = get_db()
    
    # Initialize trainer
    logger.info("Initializing model trainer with options features...")
    trainer = ModelTrainer(db, models_dir="models")
    
    # Train models
    print("🚀 Starting training with options data...\n")
    
    results = trainer.train_all_models(
        symbols=symbols,
        lookback_days=lookback_days,
        strategy=strategy,
        timeframe=timeframe
    )
    
    if results.get('success'):
        print("\n" + "=" * 80)
        print("✅ TRAINING WITH OPTIONS DATA COMPLETED!")
        print("=" * 80)
        print()
        print("📈 Model Performance Summary:")
        print()
        
        # Entry Model
        if 'entry_model' in results:
            entry = results['entry_model']
            print(f"  Entry Signal Model:")
            print(f"    Accuracy: {entry.get('test_accuracy', 0):.2%}")
            print(f"    Precision: {entry.get('test_precision', 0):.2%}")
            print(f"    F1 Score: {entry.get('test_f1', 0):.2%}")
            print(f"    AUC: {entry.get('test_auc', 0):.3f}")
            
            # Show top features
            if 'top_features' in entry:
                print(f"\n  Top 5 Features:")
                for feat, imp in entry['top_features'][:5]:
                    emoji = "📊" if 'atm_' in feat or 'iv_' in feat or 'option' in feat else "📈"
                    print(f"    {emoji} {feat}: {imp:.4f}")
            print()
        
        # Win Probability Model
        if 'win_probability_model' in results:
            win_prob = results['win_probability_model']
            print(f"  Win Probability Model:")
            print(f"    R² Score: {win_prob.get('test_r2', 0):.3f}")
            print(f"    MAE: {win_prob.get('test_mae', 0):.4f}")
            print()
        
        # Volatility Model
        if 'volatility_model' in results:
            vol = results['volatility_model']
            print(f"  Volatility Forecaster:")
            print(f"    Accuracy: {vol.get('test_accuracy', 0):.2%}")
            print()
        
        print(f"📦 Models saved to: ./models/")
        print()
        print("🎯 What's Different:")
        print("  ✅ Models trained on REAL options data")
        print("  ✅ Actual Greeks (not calculated)")
        print("  ✅ Real Implied Volatility")
        print("  ✅ Open Interest data")
        print("  ✅ Put/Call ratios")
        print()
        print("Expected improvement: +5-10% accuracy over price-only models")
        print()
        print("💡 Test models: python scripts/test_ml_models.py")
        print("💡 Deploy: ./DEPLOY_IMPROVED_ML.sh")
        print()
    else:
        print("\n❌ TRAINING FAILED!")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print()
        
        if "rate limit" in str(results.get('error', '')).lower():
            print("⚠️  Polygon rate limit hit. Solutions:")
            print("  1. Wait 1 minute and try again")
            print("  2. Reduce symbols to 2-3")
            print("  3. Use models without options data temporarily")
        else:
            print("Common fixes:")
            print("  - Check POLYGON_API_KEY in .env")
            print("  - Verify internet connection")
            print("  - Check Polygon plan includes options data")
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

