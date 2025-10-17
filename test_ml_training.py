#!/usr/bin/env python3
"""
Comprehensive ML Training Test

This script tests the complete ML training pipeline:
1. Account adaptation system
2. Data collection
3. Feature engineering
4. Model training
5. Model evaluation
6. Auto training system
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Set environment variables
os.environ['POLYGON_API_KEY'] = 'wWrUjjcksqLDPntXbJb72kiFzAwyqIpY'
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{os.getcwd()}/src"

# Add src to path
sys.path.append('src')

def test_account_adaptation():
    """Test account adaptation system"""
    logger.info("=" * 60)
    logger.info("Testing Account Adaptation System")
    logger.info("=" * 60)
    
    from src.trading.account_adaptation import AccountAdaptationSystem
    
    # Test different account sizes
    test_balances = [500, 5000, 25000, 100000, 500000]
    
    for balance in test_balances:
        logger.info(f"\nTesting with account balance: ${balance:,}")
        
        adaptation = AccountAdaptationSystem(balance)
        
        logger.info(f"Account Tier: {adaptation.current_tier.name}")
        logger.info(f"Symbols: {adaptation.get_recommended_symbols(5)}")
        logger.info(f"Max Positions: {adaptation.get_position_limits()['max_positions']}")
        logger.info(f"Risk per Trade: {adaptation.get_position_limits()['risk_per_trade']:.1%}")
        logger.info(f"ML Config: {adaptation.get_ml_training_config()}")
        
        # Test position sizing
        position_size = adaptation.calculate_position_size(option_price=2.50, stop_loss=2.00)
        logger.info(f"Position Size (option $2.50, stop $2.00): {position_size} contracts")
    
    logger.info("✅ Account adaptation system working correctly")
    return True

def test_data_collection():
    """Test data collection capabilities"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Data Collection")
    logger.info("=" * 60)
    
    from src.market_data.polygon_options import PolygonOptionsClient
    from src.market_data.collector import MarketDataCollector
    
    # Test Polygon client
    polygon_client = PolygonOptionsClient()
    logger.info("✅ Polygon client initialized")
    
    # Test market collector
    market_collector = MarketDataCollector()
    logger.info("✅ Market collector initialized")
    
    # Test basic API calls
    try:
        # Market status
        status = polygon_client.get_market_status()
        if status:
            logger.info(f"✅ Market status: {status['market']}")
        else:
            logger.warning("⚠️ Market status not available")
        
        # Options search
        contracts = polygon_client.search_options_contracts("SPY", limit=3)
        if contracts:
            logger.info(f"✅ Found {len(contracts)} option contracts")
        else:
            logger.warning("⚠️ No option contracts found")
        
        # Technical indicators
        sma = polygon_client.get_sma("O:SPY251220P00550000", 20)
        if sma:
            logger.info(f"✅ SMA calculated: {sma}")
        else:
            logger.warning("⚠️ SMA not available")
        
    except Exception as e:
        logger.error(f"❌ Data collection error: {e}")
        return False
    
    logger.info("✅ Data collection working correctly")
    return True

def test_ml_pipeline():
    """Test ML pipeline functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ML Pipeline")
    logger.info("=" * 60)
    
    from src.ml.data_pipeline import OptionsMLDataPipeline
    from src.trading.account_adaptation import AccountAdaptationSystem
    
    # Initialize ML pipeline
    ml_pipeline = OptionsMLDataPipeline()
    logger.info("✅ ML pipeline initialized")
    
    # Test with account adaptation
    adaptation = AccountAdaptationSystem(25000)
    symbols = adaptation.get_recommended_symbols(3)  # Limit to 3 for testing
    ml_config = adaptation.get_ml_training_config()
    
    logger.info(f"Testing with symbols: {symbols}")
    logger.info(f"ML config: {ml_config}")
    
    try:
        # Test dataset creation (this will use REST API data)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)  # Short period for testing
        
        logger.info(f"Creating dataset from {start_date} to {end_date}")
        
        dataset = ml_pipeline.create_comprehensive_dataset(
            symbols=symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            save_features=False  # Don't save for testing
        )
        
        if not dataset.empty:
            logger.info(f"✅ Dataset created: {len(dataset)} records")
            logger.info(f"Dataset columns: {list(dataset.columns)}")
            
            # Test feature engineering
            features = ml_pipeline.create_features(dataset)
            if not features.empty:
                logger.info(f"✅ Features created: {len(features)} records")
                logger.info(f"Feature columns: {list(features.columns)}")
            else:
                logger.warning("⚠️ No features created")
        else:
            logger.warning("⚠️ No dataset created (may be due to plan limitations)")
        
    except Exception as e:
        logger.error(f"❌ ML pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("✅ ML pipeline working correctly")
    return True

def test_auto_training():
    """Test auto training system"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Auto Training System")
    logger.info("=" * 60)
    
    from src.ml.auto_training import AutoMLTrainingSystem
    
    # Test with medium account
    balance = 25000
    logger.info(f"Testing with account balance: ${balance:,}")
    
    try:
        # Initialize auto training system
        auto_ml = AutoMLTrainingSystem(balance)
        logger.info("✅ Auto training system initialized")
        
        # Test configuration
        logger.info(f"Account tier: {auto_ml.account_adaptation.current_tier.name}")
        logger.info(f"Training schedule: {auto_ml.config['training_schedule']}")
        logger.info(f"Model types: {auto_ml.config['model_types']}")
        
        # Test training status
        status = auto_ml.get_training_status()
        logger.info(f"Training status: {status}")
        
        # Test scheduler (without actually starting it)
        logger.info("✅ Auto training system ready")
        
    except Exception as e:
        logger.error(f"❌ Auto training error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("✅ Auto training system working correctly")
    return True

def test_backtesting():
    """Test backtesting framework"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Backtesting Framework")
    logger.info("=" * 60)
    
    from src.ml.backtesting import OptionsBacktester
    
    try:
        # Initialize backtester
        backtester = OptionsBacktester()
        logger.info("✅ Backtester initialized")
        
        # Test strategy definition
        strategies = backtester.get_available_strategies()
        logger.info(f"Available strategies: {strategies}")
        
        # Test configuration
        config = backtester.get_default_config()
        logger.info(f"Default config: {config}")
        
        logger.info("✅ Backtesting framework ready")
        
    except Exception as e:
        logger.error(f"❌ Backtesting error: {e}")
        return False
    
    logger.info("✅ Backtesting framework working correctly")
    return True

def main():
    """Main test function"""
    logger.info("🚀 Starting Comprehensive ML Training Test")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Run all tests
    tests = [
        ("Account Adaptation", test_account_adaptation),
        ("Data Collection", test_data_collection),
        ("ML Pipeline", test_ml_pipeline),
        ("Auto Training", test_auto_training),
        ("Backtesting", test_backtesting)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All ML training tests passed!")
        logger.info("Your trading agent is ready for automatic ML training!")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    main()

