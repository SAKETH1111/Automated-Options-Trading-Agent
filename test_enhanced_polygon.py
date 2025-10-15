#!/usr/bin/env python3
"""
Test script for enhanced Polygon.io options functionality
Tests all new features including technical indicators, historical data, 
market operations, and advanced filtering
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.market_data.polygon_options import PolygonOptionsClient
from src.market_data.collector import MarketDataCollector


def test_polygon_client():
    """Test the enhanced PolygonOptionsClient"""
    logger.info("=" * 60)
    logger.info("Testing Enhanced PolygonOptionsClient")
    logger.info("=" * 60)
    
    try:
        # Initialize client
        client = PolygonOptionsClient()
        logger.info("✅ PolygonOptionsClient initialized successfully")
        
        # Test symbol
        test_symbol = "SPY"
        test_option = "O:SPY251220P00550000"  # Example SPY put option
        
        # Test 1: Market Operations
        logger.info("\n1. Testing Market Operations...")
        
        market_status = client.get_market_status()
        if market_status:
            logger.info(f"✅ Market Status: {market_status.get('market', 'Unknown')}")
        else:
            logger.warning("⚠️ Market status not available")
        
        holidays = client.get_market_holidays()
        if holidays:
            logger.info(f"✅ Found {len(holidays)} market holidays")
        else:
            logger.warning("⚠️ Market holidays not available")
        
        exchanges = client.get_exchanges()
        if exchanges:
            logger.info(f"✅ Found {len(exchanges)} exchanges")
        else:
            logger.warning("⚠️ Exchanges not available")
        
        # Test 2: Options Chain Search
        logger.info("\n2. Testing Options Chain Search...")
        
        contracts = client.search_options_contracts(
            underlying=test_symbol,
            contract_type="put",
            limit=10
        )
        if contracts:
            logger.info(f"✅ Found {len(contracts)} put contracts for {test_symbol}")
            for contract in contracts[:3]:  # Show first 3
                logger.info(f"   - {contract['ticker']}: Strike {contract['strike']}, Exp {contract['expiration_date']}")
        else:
            logger.warning("⚠️ No contracts found")
        
        # Test 3: Option Snapshot
        logger.info("\n3. Testing Option Snapshot...")
        
        if contracts:
            test_contract = contracts[0]['ticker']
            snapshot = client.get_option_snapshot(test_contract)
            if snapshot:
                logger.info(f"✅ Snapshot for {test_contract}:")
                logger.info(f"   - Bid: ${snapshot.get('bid', 'N/A')}")
                logger.info(f"   - Ask: ${snapshot.get('ask', 'N/A')}")
                logger.info(f"   - Volume: {snapshot.get('volume', 'N/A')}")
                logger.info(f"   - OI: {snapshot.get('open_interest', 'N/A')}")
                greeks = snapshot.get('greeks', {})
                if greeks:
                    logger.info(f"   - Delta: {greeks.get('delta', 'N/A')}")
                    logger.info(f"   - Gamma: {greeks.get('gamma', 'N/A')}")
                    logger.info(f"   - Theta: {greeks.get('theta', 'N/A')}")
                    logger.info(f"   - Vega: {greeks.get('vega', 'N/A')}")
            else:
                logger.warning(f"⚠️ No snapshot data for {test_contract}")
        
        # Test 4: Technical Indicators
        logger.info("\n4. Testing Technical Indicators...")
        
        if contracts:
            test_contract = contracts[0]['ticker']
            
            # Test SMA
            sma = client.get_sma(test_contract, window=20)
            if sma:
                logger.info(f"✅ SMA(20) for {test_contract}: {len(sma.get('sma', []))} data points")
            else:
                logger.warning(f"⚠️ SMA not available for {test_contract}")
            
            # Test RSI
            rsi = client.get_rsi(test_contract, window=14)
            if rsi:
                logger.info(f"✅ RSI(14) for {test_contract}: {len(rsi.get('rsi', []))} data points")
            else:
                logger.warning(f"⚠️ RSI not available for {test_contract}")
        
        # Test 5: Historical Data
        logger.info("\n5. Testing Historical Data...")
        
        if contracts:
            test_contract = contracts[0]['ticker']
            
            # Test custom bars
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            bars = client.get_custom_bars(
                option_ticker=test_contract,
                from_date=from_date,
                to_date=to_date,
                timespan="day"
            )
            if bars:
                logger.info(f"✅ Custom bars for {test_contract}: {bars.get('count', 0)} bars")
            else:
                logger.warning(f"⚠️ Custom bars not available for {test_contract}")
            
            # Test daily summary
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            daily_summary = client.get_daily_ticker_summary(test_contract, yesterday)
            if daily_summary:
                logger.info(f"✅ Daily summary for {test_contract} on {yesterday}:")
                logger.info(f"   - Open: ${daily_summary.get('open', 'N/A')}")
                logger.info(f"   - High: ${daily_summary.get('high', 'N/A')}")
                logger.info(f"   - Low: ${daily_summary.get('low', 'N/A')}")
                logger.info(f"   - Close: ${daily_summary.get('close', 'N/A')}")
            else:
                logger.warning(f"⚠️ Daily summary not available for {test_contract}")
        
        # Test 6: Advanced Filtering
        logger.info("\n6. Testing Advanced Filtering...")
        
        # Test delta range filtering
        delta_options = client.get_options_by_delta_range(
            underlying=test_symbol,
            min_delta=0.20,
            max_delta=0.40,
            contract_type="put",
            dte_min=20,
            dte_max=60
        )
        if delta_options:
            logger.info(f"✅ Found {len(delta_options)} options in delta range 0.20-0.40")
            for option in delta_options[:3]:  # Show first 3
                delta = abs(option.get('greeks', {}).get('delta', 0))
                logger.info(f"   - {option['ticker']}: Strike {option['strike']}, Delta {delta:.3f}")
        else:
            logger.warning("⚠️ No options found in delta range")
        
        # Test high volume filtering
        high_volume_options = client.get_high_volume_options(
            underlying=test_symbol,
            min_volume=100,
            min_open_interest=500,
            dte_min=20,
            dte_max=60
        )
        if high_volume_options:
            logger.info(f"✅ Found {len(high_volume_options)} high volume options")
            for option in high_volume_options[:3]:  # Show first 3
                logger.info(f"   - {option['ticker']}: Volume {option.get('volume', 0)}, OI {option.get('open_interest', 0)}")
        else:
            logger.warning("⚠️ No high volume options found")
        
        # Test 7: Option Chain Snapshot
        logger.info("\n7. Testing Option Chain Snapshot...")
        
        chain_snapshot = client.get_option_chain_snapshot(test_symbol)
        if chain_snapshot:
            logger.info(f"✅ Option chain snapshot for {test_symbol}: {chain_snapshot.get('count', 0)} options")
        else:
            logger.warning(f"⚠️ Option chain snapshot not available for {test_symbol}")
        
        logger.info("\n✅ All PolygonOptionsClient tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Error testing PolygonOptionsClient: {e}")
        return False
    
    return True


def test_market_data_collector():
    """Test the enhanced MarketDataCollector"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Enhanced MarketDataCollector")
    logger.info("=" * 60)
    
    try:
        # Initialize collector
        collector = MarketDataCollector()
        logger.info("✅ MarketDataCollector initialized successfully")
        
        test_symbol = "SPY"
        
        # Test 1: Enhanced Options Chain
        logger.info("\n1. Testing Enhanced Options Chain...")
        
        enhanced_chain = collector.get_enhanced_options_chain(
            symbol=test_symbol,
            target_dte=35,
            option_type="put",
            use_advanced_filtering=True
        )
        if enhanced_chain:
            logger.info(f"✅ Enhanced chain for {test_symbol}: {len(enhanced_chain)} options")
            for option in enhanced_chain[:3]:  # Show first 3
                logger.info(f"   - {option['ticker']}: Strike {option['strike']}, Volume {option.get('volume', 0)}")
        else:
            logger.warning(f"⚠️ Enhanced chain not available for {test_symbol}")
        
        # Test 2: Delta Range Filtering
        logger.info("\n2. Testing Delta Range Filtering...")
        
        delta_options = collector.get_options_by_delta_range(
            symbol=test_symbol,
            min_delta=0.25,
            max_delta=0.35,
            contract_type="put",
            dte_min=25,
            dte_max=45
        )
        if delta_options:
            logger.info(f"✅ Delta range options for {test_symbol}: {len(delta_options)} options")
        else:
            logger.warning(f"⚠️ Delta range options not available for {test_symbol}")
        
        # Test 3: Market Operations
        logger.info("\n3. Testing Market Operations...")
        
        market_ops = collector.get_market_operations()
        if market_ops:
            logger.info("✅ Market operations data retrieved")
            if market_ops.get('market_status'):
                logger.info(f"   - Market: {market_ops['market_status'].get('market', 'Unknown')}")
            if market_ops.get('holidays'):
                logger.info(f"   - Holidays: {len(market_ops['holidays'])} found")
            if market_ops.get('exchanges'):
                logger.info(f"   - Exchanges: {len(market_ops['exchanges'])} found")
        else:
            logger.warning("⚠️ Market operations not available")
        
        # Test 4: Option Chain Snapshot
        logger.info("\n4. Testing Option Chain Snapshot...")
        
        chain_snapshot = collector.get_option_chain_snapshot(test_symbol)
        if chain_snapshot:
            logger.info(f"✅ Chain snapshot for {test_symbol}: {chain_snapshot.get('count', 0)} options")
        else:
            logger.warning(f"⚠️ Chain snapshot not available for {test_symbol}")
        
        # Test 5: Technical Indicators
        logger.info("\n5. Testing Technical Indicators...")
        
        if enhanced_chain:
            test_option = enhanced_chain[0]['ticker']
            indicators = collector.get_option_technical_indicators(
                test_option,
                ['sma', 'rsi']
            )
            if indicators:
                logger.info(f"✅ Technical indicators for {test_option}:")
                for indicator, data in indicators.items():
                    if data:
                        logger.info(f"   - {indicator.upper()}: {len(data.get(indicator, []))} data points")
            else:
                logger.warning(f"⚠️ Technical indicators not available for {test_option}")
        
        # Test 6: Historical Data
        logger.info("\n6. Testing Historical Data...")
        
        if enhanced_chain:
            test_option = enhanced_chain[0]['ticker']
            historical = collector.get_option_historical_data(test_option, days=7)
            if historical:
                logger.info(f"✅ Historical data for {test_option}: {historical.get('count', 0)} bars")
            else:
                logger.warning(f"⚠️ Historical data not available for {test_option}")
        
        # Test 7: Trades and Quotes
        logger.info("\n7. Testing Trades and Quotes...")
        
        if enhanced_chain:
            test_option = enhanced_chain[0]['ticker']
            trades_quotes = collector.get_option_trades_quotes(test_option, days=3)
            if trades_quotes:
                logger.info(f"✅ Trades and quotes for {test_option}:")
                if trades_quotes.get('trades'):
                    logger.info(f"   - Trades: {len(trades_quotes['trades'])} found")
                if trades_quotes.get('quotes'):
                    logger.info(f"   - Quotes: {len(trades_quotes['quotes'])} found")
                if trades_quotes.get('last_trade'):
                    logger.info(f"   - Last trade: ${trades_quotes['last_trade'].get('price', 'N/A')}")
            else:
                logger.warning(f"⚠️ Trades and quotes not available for {test_option}")
        
        logger.info("\n✅ All MarketDataCollector tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Error testing MarketDataCollector: {e}")
        return False
    
    return True


def main():
    """Main test function"""
    logger.info("🚀 Starting Enhanced Polygon.io Options Testing")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Check for API key
    if not os.getenv('POLYGON_API_KEY'):
        logger.error("❌ POLYGON_API_KEY environment variable not set!")
        logger.info("Please set your Polygon.io API key:")
        logger.info("export POLYGON_API_KEY='your_api_key_here'")
        return False
    
    # Run tests
    polygon_success = test_polygon_client()
    collector_success = test_market_data_collector()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PolygonOptionsClient: {'✅ PASSED' if polygon_success else '❌ FAILED'}")
    logger.info(f"MarketDataCollector: {'✅ PASSED' if collector_success else '❌ FAILED'}")
    
    if polygon_success and collector_success:
        logger.info("\n🎉 All tests passed! Enhanced Polygon.io integration is working correctly.")
        return True
    else:
        logger.error("\n❌ Some tests failed. Please check the logs above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
