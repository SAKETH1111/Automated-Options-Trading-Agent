#!/usr/bin/env python3
"""
ML Model Training Script
Train all ML models on historical data
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
    """Train all ML models"""
    
    # Banner
    print("=" * 80)
    print("🤖 AUTOMATED OPTIONS TRADING AGENT - ML TRAINING")
    print("=" * 80)
    print()
    
    # Configuration
    # All symbols including new sector ETFs for small/medium accounts
    # Removed leveraged ETFs (SQQQ, UVXY, TZA) - they have non-standard options
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'GDX', 'TLT', 'XLE', 'EWZ']
    lookback_days = 730  # 2 years of data for better ML training
    strategy = 'bull_put_spread'  # Strategy to optimize for
    timeframe = '1Day'  # Use daily data (more reliable than intraday)
    
    print(f"🆕 Training {len(symbols)} symbols (added 5 new sector ETFs for all account sizes)")
    print(f"   Original: SPY, QQQ, IWM, DIA")
    print(f"   New Sectors: XLF, GDX, TLT, XLE, EWZ")
    
    print(f"📊 Training Configuration:")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Lookback: {lookback_days} days")
    print(f"   Timeframe: {timeframe}")
    print(f"   Strategy: {strategy}")
    print()
    
    print("⚠️  This will take 10-30 minutes depending on data volume...")
    print()
    
    # Initialize database
    logger.info("Initializing database...")
    db = get_db()
    
    # Initialize trainer
    logger.info("Initializing model trainer...")
    trainer = ModelTrainer(db, models_dir="models")
    
    # Train models
    print("🚀 Starting training...\n")
    
    results = trainer.train_all_models(
        symbols=symbols,
        lookback_days=lookback_days,
        strategy=strategy,
        timeframe=timeframe
    )
    
    if results.get('success'):
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
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
            print(f"    Recall: {entry.get('test_recall', 0):.2%}")
            print(f"    F1 Score: {entry.get('test_f1', 0):.2%}")
            print(f"    AUC: {entry.get('test_auc', 0):.3f}")
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
            print(f"    MAE: {vol.get('test_mae', 0):.4f}")
            print()
        
        print(f"📦 Models saved to: ./models/")
        print()
        print("🎯 What's Next:")
        print("  1. Review model performance above")
        print("  2. If accuracy < 60%, consider:")
        print("     - Collecting more data (increase lookback_days)")
        print("     - Adjusting strategy parameters")
        print("     - Adding more symbols for training")
        print("  3. Models will be automatically used in live trading")
        print("  4. Retrain periodically (weekly/monthly) with new data")
        print()
        print("💡 Tip: Run this script weekly to keep models up-to-date!")
        print()
    else:
        print("\n❌ TRAINING FAILED!")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print()
        print("Common fixes:")
        print("  - Make sure Alpaca API keys are configured")
        print("  - Check internet connection")
        print("  - Verify database is accessible")
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

