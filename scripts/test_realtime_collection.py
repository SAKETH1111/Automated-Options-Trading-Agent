"""Test script for real-time data collection system"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.market_data.realtime_collector import RealTimeDataCollector
from src.market_data.tick_analyzer import TickDataAnalyzer


def test_collector():
    """Test the real-time collector"""
    logger.info("=" * 80)
    logger.info("TESTING REAL-TIME DATA COLLECTOR")
    logger.info("=" * 80)
    
    # Create collector
    collector = RealTimeDataCollector(
        symbols=['SPY', 'QQQ'],
        collect_interval=2.0,  # Slower for testing
        buffer_size=5
    )
    
    logger.info("\n1. Starting collector...")
    collector.start()
    
    # Let it run for 20 seconds
    logger.info("2. Collecting data for 20 seconds...")
    time.sleep(20)
    
    # Check stats
    logger.info("\n3. Checking statistics...")
    stats = collector.get_stats()
    logger.info(f"   Is Running: {stats['is_running']}")
    logger.info(f"   Symbols: {stats['symbols']}")
    logger.info(f"   Ticks Collected: {stats['total_ticks_collected']}")
    logger.info(f"   Ticks Stored: {stats['total_ticks_stored']}")
    logger.info(f"   Errors: {stats['collection_errors']}")
    logger.info(f"   Buffer Size: {stats['buffer_size']}")
    
    # Stop collector
    logger.info("\n4. Stopping collector...")
    collector.stop()
    
    # Final stats
    final_stats = collector.get_stats()
    logger.info(f"\n5. Final Statistics:")
    logger.info(f"   Total Ticks Collected: {final_stats['total_ticks_collected']}")
    logger.info(f"   Total Ticks Stored: {final_stats['total_ticks_stored']}")
    
    if final_stats['total_ticks_collected'] > 0:
        logger.info("\n✅ Collector test PASSED")
        return True
    else:
        logger.error("\n❌ Collector test FAILED - no data collected")
        logger.info("   Note: This is normal if market is closed")
        return False


def test_analyzer():
    """Test the tick analyzer"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING TICK DATA ANALYZER")
    logger.info("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    # Check data availability
    logger.info("\n1. Checking data availability...")
    for symbol in ['SPY', 'QQQ']:
        availability = analyzer.get_data_availability(symbol, days=1)
        logger.info(f"   {symbol}: {availability.get('total_ticks', 0)} ticks")
    
    # Get daily statistics
    logger.info("\n2. Getting daily statistics...")
    stats = analyzer.get_daily_statistics('SPY', datetime.now())
    
    if stats:
        logger.info(f"   SPY Open: ${stats.get('open', 0):.2f}")
        logger.info(f"   SPY High: ${stats.get('high', 0):.2f}")
        logger.info(f"   SPY Low: ${stats.get('low', 0):.2f}")
        logger.info(f"   SPY Close: ${stats.get('close', 0):.2f}")
        logger.info(f"   SPY Range: {stats.get('range_pct', 0):.2f}%")
        logger.info(f"   Tick Count: {stats.get('tick_count', 0):,}")
        logger.info("\n✅ Analyzer test PASSED")
        return True
    else:
        logger.warning("\n⚠️  No historical data available for analysis")
        logger.info("   This is normal if collector just started")
        return False


def main():
    """Run all tests"""
    try:
        logger.info("Starting Real-Time Data Collection Tests\n")
        
        # Test 1: Collector
        collector_passed = test_collector()
        
        # Test 2: Analyzer
        analyzer_passed = test_analyzer()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Collector Test: {'✅ PASSED' if collector_passed else '❌ FAILED'}")
        logger.info(f"Analyzer Test: {'✅ PASSED' if analyzer_passed else '⚠️  NO DATA'}")
        logger.info("\nNote: If tests fail due to market being closed, run during market hours (9:30-16:00 ET)")
        
        if collector_passed:
            logger.info("\n🎉 Real-time data collection system is working!")
            logger.info("\nNext steps:")
            logger.info("  1. Let the agent run to collect more data")
            logger.info("  2. Use: python scripts/view_tick_data.py")
            logger.info("  3. Analyze patterns and optimize strategies")
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

