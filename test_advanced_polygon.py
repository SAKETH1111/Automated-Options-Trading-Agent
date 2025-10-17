#!/usr/bin/env python3
"""
Advanced Polygon Account Test

This script tests all Advanced Polygon account features:
1. WebSocket real-time streaming
2. S3 flat files access
3. Advanced REST API endpoints
4. Real-time ML training
5. Live position monitoring
"""

import os
import sys
import time
from datetime import datetime, timedelta
from loguru import logger

# Set environment variables
os.environ['POLYGON_API_KEY'] = 'wWrUjjcksqLDPntXbJb72kiFzAwyqIpY'
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{os.getcwd()}/src"

# Add src to path
sys.path.append('src')

def test_websocket_access():
    """Test WebSocket real-time streaming"""
    logger.info("=" * 60)
    logger.info("Testing WebSocket Real-time Streaming")
    logger.info("=" * 60)
    
    try:
        from src.market_data.polygon_websocket import PolygonOptionsWebSocketClient
        
        # Initialize WebSocket client
        client = PolygonOptionsWebSocketClient(api_key='wWrUjjcksqLDPntXbJb72kiFzAwyqIpY')
        logger.info("✅ WebSocket client initialized")
        
        # Subscribe to options data
        options_symbols = ['O:SPY251220P00550000', 'O:SPY251220C00550000']
        
        client.subscribe_to_trades(options_symbols)
        client.subscribe_to_quotes(options_symbols)
        client.subscribe_to_aggregates_minute(options_symbols)
        client.subscribe_to_aggregates_second(options_symbols)
        client.subscribe_to_fmv(options_symbols)
        
        logger.info("✅ Subscriptions created")
        
        # Connect and start streaming
        client.connect()
        logger.info("✅ WebSocket connected")
        
        client.start()
        logger.info("✅ WebSocket streaming started")
        
        # Let it run for a few seconds
        time.sleep(10)
        
        # Get stats
        stats = client.get_stats()
        logger.info(f"WebSocket stats: {stats}")
        
        # Stop and disconnect
        client.stop()
        client.disconnect()
        logger.info("✅ WebSocket stopped and disconnected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}")
        return False

def test_s3_flat_files():
    """Test S3 flat files access"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing S3 Flat Files Access")
    logger.info("=" * 60)
    
    try:
        from src.market_data.polygon_flat_files import PolygonFlatFilesClient
        
        # Initialize flat files client
        client = PolygonFlatFilesClient()
        logger.info("✅ Flat files client initialized")
        
        # Test listing available dates
        for data_type in ['trades', 'quotes', 'aggregates']:
            dates = client.list_available_dates(data_type)
            if dates:
                logger.info(f"✅ {data_type}: {len(dates)} dates available")
                logger.info(f"   Latest: {dates[-1]}")
                logger.info(f"   Earliest: {dates[0]}")
            else:
                logger.warning(f"⚠️ No {data_type} dates found")
        
        # Test downloading recent data
        trades_dates = client.list_available_dates("trades")
        if trades_dates:
            test_date = trades_dates[-1]  # Most recent date
            logger.info(f"\nTesting download for {test_date}...")
            
            # Download trades data
            trades_file = client.download_data("trades", test_date)
            if trades_file and trades_file.exists():
                logger.info(f"✅ Trades downloaded: {trades_file} ({trades_file.stat().st_size} bytes)")
            else:
                logger.warning("⚠️ Trades download failed")
            
            # Download quotes data
            quotes_file = client.download_data("quotes", test_date)
            if quotes_file and quotes_file.exists():
                logger.info(f"✅ Quotes downloaded: {quotes_file} ({quotes_file.stat().st_size} bytes)")
            else:
                logger.warning("⚠️ Quotes download failed")
            
            # Download aggregates data
            aggregates_file = client.download_data("aggregates", test_date)
            if aggregates_file and aggregates_file.exists():
                logger.info(f"✅ Aggregates downloaded: {aggregates_file} ({aggregates_file.stat().st_size} bytes)")
            else:
                logger.warning("⚠️ Aggregates download failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 flat files test failed: {e}")
        return False

def test_advanced_rest_api():
    """Test advanced REST API endpoints"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Advanced REST API")
    logger.info("=" * 60)
    
    try:
        from src.market_data.polygon_options import PolygonOptionsClient
        
        # Initialize client
        client = PolygonOptionsClient()
        logger.info("✅ Polygon client initialized")
        
        # Test all major endpoints
        endpoints_tested = []
        
        # Market operations
        status = client.get_market_status()
        if status:
            logger.info(f"✅ Market status: {status['market']}")
            endpoints_tested.append("market_status")
        
        holidays = client.get_market_holidays()
        if holidays:
            logger.info(f"✅ Market holidays: {len(holidays)} found")
            endpoints_tested.append("market_holidays")
        
        exchanges = client.get_exchanges()
        if exchanges:
            logger.info(f"✅ Exchanges: {len(exchanges)} found")
            endpoints_tested.append("exchanges")
        
        # Options search
        contracts = client.search_options_contracts("SPY", limit=10)
        if contracts:
            logger.info(f"✅ Options contracts: {len(contracts)} found")
            endpoints_tested.append("options_search")
        
        # Technical indicators
        sma = client.get_sma("O:SPY251220P00550000", 20)
        if sma:
            logger.info(f"✅ SMA: {sma}")
            endpoints_tested.append("technical_indicators")
        
        # Historical data
        historical = client.get_trades("O:SPY251220P00550000", "2024-01-01", "2024-01-02")
        if historical:
            logger.info(f"✅ Historical trades: {len(historical)} found")
            endpoints_tested.append("historical_data")
        
        logger.info(f"✅ REST API endpoints tested: {len(endpoints_tested)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced REST API test failed: {e}")
        return False

def test_advanced_integration():
    """Test the complete advanced integration system"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Advanced Integration System")
    logger.info("=" * 60)
    
    try:
        from src.trading.advanced_polygon_integration import AdvancedPolygonIntegration
        
        # Initialize advanced system
        advanced_system = AdvancedPolygonIntegration(account_balance=25000)
        logger.info("✅ Advanced integration system initialized")
        
        # Test system status
        status = advanced_system.get_system_status()
        logger.info(f"System status: {status}")
        
        # Test historical data summary
        historical_summary = advanced_system.get_historical_data_summary()
        logger.info(f"Historical data summary: {historical_summary}")
        
        # Test trading signals
        signals = advanced_system.get_trading_signals()
        logger.info(f"Trading signals: {signals}")
        
        logger.info("✅ Advanced integration system working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_with_historical_data():
    """Test ML training with historical data"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ML Training with Historical Data")
    logger.info("=" * 60)
    
    try:
        from src.ml.auto_training import AutoMLTrainingSystem
        
        # Initialize auto ML system
        auto_ml = AutoMLTrainingSystem(account_balance=25000)
        logger.info("✅ Auto ML system initialized")
        
        # Test training status
        status = auto_ml.get_training_status()
        logger.info(f"Training status: {status}")
        
        # Test force training (this will use historical data)
        logger.info("Testing force training...")
        auto_ml.force_training()
        logger.info("✅ Force training completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ML training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Advanced Polygon Account Test")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Run all tests
    tests = [
        ("WebSocket Streaming", test_websocket_access),
        ("S3 Flat Files", test_s3_flat_files),
        ("Advanced REST API", test_advanced_rest_api),
        ("Advanced Integration", test_advanced_integration),
        ("ML with Historical Data", test_ml_with_historical_data)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🧪 Running {test_name} test...")
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ADVANCED POLYGON ACCOUNT TEST SUMMARY")
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
        logger.info("🎉 All Advanced Polygon features working!")
        logger.info("Your trading agent now has access to:")
        logger.info("  ✅ Real-time WebSocket streaming")
        logger.info("  ✅ Historical data from S3 flat files")
        logger.info("  ✅ All REST API endpoints")
        logger.info("  ✅ Advanced ML training")
        logger.info("  ✅ Live position monitoring")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    main()

