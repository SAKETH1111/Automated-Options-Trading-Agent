#!/usr/bin/env python3
"""
Phase 1 Testing Script
Tests all technical analysis components
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.database.session import get_session
from src.analysis.analyzer import MarketAnalyzer


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_data_retrieval(analyzer: MarketAnalyzer, symbol: str):
    """Test data retrieval from database"""
    print_section(f"Testing Data Retrieval for {symbol}")
    
    df = analyzer.get_recent_data(symbol, minutes=60)
    
    if df.empty:
        print(f"❌ No data found for {symbol}")
        return False
    
    print(f"✅ Retrieved {len(df)} data points")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"   Current price: ${df['price'].iloc[-1]:.2f}")
    
    return True


def test_technical_indicators(analyzer: MarketAnalyzer, symbol: str):
    """Test technical indicators calculation"""
    print_section(f"Testing Technical Indicators for {symbol}")
    
    df = analyzer.get_recent_data(symbol, minutes=60)
    
    if df.empty:
        print(f"❌ No data available")
        return False
    
    # Calculate indicators
    df_with_ind = analyzer.indicators.calculate_all_indicators(df)
    
    # Check if indicators were calculated
    indicator_cols = ['sma_10', 'sma_20', 'rsi', 'macd', 'bb_upper', 'bb_lower']
    calculated = [col for col in indicator_cols if col in df_with_ind.columns]
    
    print(f"✅ Calculated {len(calculated)} indicator types")
    
    # Show latest values
    latest = df_with_ind.iloc[-1]
    print(f"\n📊 Latest Indicator Values:")
    print(f"   SMA(10): ${latest.get('sma_10', 0):.2f}")
    print(f"   SMA(20): ${latest.get('sma_20', 0):.2f}")
    print(f"   RSI: {latest.get('rsi', 0):.2f}")
    print(f"   MACD: {latest.get('macd', 0):.4f}")
    print(f"   BB Upper: ${latest.get('bb_upper', 0):.2f}")
    print(f"   BB Lower: ${latest.get('bb_lower', 0):.2f}")
    
    # Get signals
    signals = analyzer.indicators.get_indicator_signals(df_with_ind)
    print(f"\n🎯 Indicator Signals:")
    for key, value in signals.items():
        print(f"   {key}: {value}")
    
    return True


def test_pattern_recognition(analyzer: MarketAnalyzer, symbol: str):
    """Test pattern recognition"""
    print_section(f"Testing Pattern Recognition for {symbol}")
    
    df = analyzer.get_recent_data(symbol, minutes=60)
    
    if df.empty:
        print(f"❌ No data available")
        return False
    
    # Analyze patterns
    patterns = analyzer.patterns.analyze_all_patterns(df, 'price', 'volume')
    
    print(f"✅ Pattern analysis complete")
    
    # Support/Resistance
    if 'support_resistance' in patterns:
        sr = patterns['support_resistance']
        print(f"\n📍 Support/Resistance Levels:")
        print(f"   Support: {[f'${x:.2f}' for x in sr.get('support', [])]}")
        print(f"   Resistance: {[f'${x:.2f}' for x in sr.get('resistance', [])]}")
    
    # Trend
    if 'trend' in patterns:
        trend = patterns['trend']
        print(f"\n📈 Trend Analysis:")
        print(f"   Direction: {trend.get('direction')}")
        print(f"   Strength: {trend.get('strength')}")
        print(f"   Description: {trend.get('description')}")
    
    # Breakout
    if 'breakout' in patterns:
        breakout = patterns['breakout']
        if breakout.get('breakout'):
            print(f"\n🚀 Breakout Detected:")
            print(f"   Direction: {breakout.get('direction')}")
            print(f"   Volume Confirmed: {breakout.get('volume_confirmed')}")
    
    # Reversals
    if 'reversals' in patterns:
        reversals = patterns['reversals']
        if reversals.get('reversal_detected'):
            print(f"\n🔄 Reversal Pattern:")
            print(f"   Type: {reversals.get('type')}")
    
    return True


def test_market_regime(analyzer: MarketAnalyzer, symbol: str):
    """Test market regime detection"""
    print_section(f"Testing Market Regime Detection for {symbol}")
    
    df = analyzer.get_recent_data(symbol, minutes=120)
    
    if df.empty:
        print(f"❌ No data available")
        return False
    
    # Detect regime
    regime = analyzer.regime.detect_overall_regime(df, 'price', 'volume')
    
    print(f"✅ Market regime detected")
    
    # Volatility regime
    if 'volatility' in regime:
        vol = regime['volatility']
        print(f"\n💨 Volatility Regime:")
        print(f"   Regime: {vol.get('regime')}")
        print(f"   Description: {vol.get('description')}")
        print(f"   Percentile: {vol.get('percentile', 0):.2%}")
    
    # Trend regime
    if 'trend' in regime:
        trend = regime['trend']
        print(f"\n📊 Trend Regime:")
        print(f"   Regime: {trend.get('regime')}")
        print(f"   Description: {trend.get('description')}")
    
    # Momentum regime
    if 'momentum' in regime:
        momentum = regime['momentum']
        print(f"\n⚡ Momentum Regime:")
        print(f"   Regime: {momentum.get('regime')}")
        print(f"   RSI: {momentum.get('rsi', 0):.2f}")
    
    # Recommendation
    if 'recommendation' in regime:
        rec = regime['recommendation']
        print(f"\n💡 Trading Recommendation:")
        print(f"   Action: {rec.get('action')}")
        print(f"   Strategy: {rec.get('strategy')}")
    
    return True


def test_comprehensive_analysis(analyzer: MarketAnalyzer, symbol: str):
    """Test comprehensive analysis"""
    print_section(f"Testing Comprehensive Analysis for {symbol}")
    
    # Run full analysis
    analysis = analyzer.analyze_symbol(symbol, store_results=True)
    
    if 'error' in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
        return False
    
    print(f"✅ Comprehensive analysis complete")
    print(f"   Symbol: {analysis['symbol']}")
    print(f"   Current Price: ${analysis['current_price']:.2f}")
    print(f"   Data Points: {analysis['data_points']}")
    print(f"   Analysis stored in database: Yes")
    
    return True


def test_trading_signals(analyzer: MarketAnalyzer, symbol: str):
    """Test trading signal generation"""
    print_section(f"Testing Trading Signal Generation for {symbol}")
    
    # Generate signals
    signals = analyzer.generate_trading_signals(symbol)
    
    if 'error' in signals:
        print(f"❌ Signal generation failed: {signals['error']}")
        return False
    
    print(f"✅ Trading signals generated")
    print(f"\n🎯 Trading Signal:")
    print(f"   Overall Signal: {signals['overall_signal']}")
    print(f"   Confidence: {signals['confidence']:.1%}")
    print(f"   Entry Price: ${signals['entry_price']:.2f}")
    
    if signals['stop_loss']:
        print(f"   Stop Loss: ${signals['stop_loss']:.2f}")
    if signals['take_profit']:
        print(f"   Take Profit: ${signals['take_profit']:.2f}")
    
    print(f"\n📋 Reasons:")
    for reason in signals['reasons']:
        print(f"   • {reason}")
    
    return True


def test_market_summary(analyzer: MarketAnalyzer):
    """Test market summary generation"""
    print_section("Testing Market Summary")
    
    # Generate summary
    summary = analyzer.get_market_summary(['SPY', 'QQQ'])
    
    print(f"✅ Market summary generated")
    print(f"   Timestamp: {summary['timestamp']}")
    
    for symbol, data in summary['symbols'].items():
        print(f"\n📊 {symbol}:")
        print(f"   Price: ${data.get('price', 0):.2f}")
        print(f"   Trend: {data.get('trend')}")
        print(f"   Volatility: {data.get('volatility')}")
        print(f"   Signal: {data.get('signal')}")
        if data.get('confidence'):
            print(f"   Confidence: {data.get('confidence'):.1%}")
    
    return True


def main():
    """Run all Phase 1 tests"""
    print("\n" + "=" * 60)
    print("  🚀 Phase 1 Technical Analysis Testing")
    print("=" * 60)
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize
    db = get_session()
    analyzer = MarketAnalyzer(db)
    
    test_results = []
    symbols = ['SPY', 'QQQ']
    
    # Run tests for each symbol
    for symbol in symbols:
        test_results.append(("Data Retrieval", test_data_retrieval(analyzer, symbol)))
        test_results.append(("Technical Indicators", test_technical_indicators(analyzer, symbol)))
        test_results.append(("Pattern Recognition", test_pattern_recognition(analyzer, symbol)))
        test_results.append(("Market Regime", test_market_regime(analyzer, symbol)))
        test_results.append(("Comprehensive Analysis", test_comprehensive_analysis(analyzer, symbol)))
        test_results.append(("Trading Signals", test_trading_signals(analyzer, symbol)))
    
    # Run market summary test
    test_results.append(("Market Summary", test_market_summary(analyzer)))
    
    # Print summary
    print_section("Test Results Summary")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All Phase 1 tests passed! System is production-ready!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    print("\n" + "=" * 60)
    print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    db.close()
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

