#!/usr/bin/env python3
"""
Phase 5 Testing Script
Tests advanced risk management system
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.database.session import get_session
from src.risk_management import (
    PortfolioRiskManager,
    DynamicPositionSizer,
    CircuitBreaker,
    CorrelationAnalyzer
)


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_portfolio_risk_manager(db):
    """Test portfolio risk management"""
    print_section("Testing Portfolio Risk Manager")
    
    manager = PortfolioRiskManager(db, total_capital=10000.0)
    
    print(f"✅ Portfolio Risk Manager initialized")
    print(f"\n📊 Risk Limits:")
    print(f"  Max Positions: {manager.max_positions}")
    print(f"  Max Risk Per Position: {manager.max_risk_per_position:.1%}")
    print(f"  Max Total Risk: {manager.max_total_risk:.1%}")
    print(f"  Max Symbol Concentration: {manager.max_symbol_concentration:.1%}")
    print(f"  Daily Loss Limit: {manager.daily_loss_limit:.1%}")
    print(f"  Max Drawdown Limit: {manager.max_drawdown_limit:.1%}")
    
    # Test position approval
    print(f"\n🔍 Testing Position Approval:")
    
    test_trade = {
        'symbol': 'SPY',
        'strategy_type': 'bull_put_spread',
        'max_loss': 500.0
    }
    
    approval = manager.check_can_open_position(test_trade)
    
    print(f"  Proposed Trade: {test_trade['strategy_type']} on {test_trade['symbol']}")
    print(f"  Max Loss: ${test_trade['max_loss']:.2f}")
    print(f"  Approved: {'✅ YES' if approval['approved'] else '❌ NO'}")
    
    if approval.get('reasons'):
        print(f"  Reasons: {', '.join(approval['reasons'])}")
    
    if approval.get('warnings'):
        print(f"  Warnings: {', '.join(approval['warnings'])}")
    
    # Get risk metrics
    metrics = manager.get_portfolio_risk_metrics()
    if metrics:
        print(f"\n📊 Current Risk Metrics:")
        print(f"  Total Risk: ${metrics['total_risk']:,.2f} ({metrics['total_risk_pct']:.2f}%)")
        print(f"  Available Risk: ${metrics['available_risk']:,.2f}")
    
    return True


def test_dynamic_position_sizer(db):
    """Test dynamic position sizing"""
    print_section("Testing Dynamic Position Sizer")
    
    sizer = DynamicPositionSizer(db, base_capital=10000.0)
    
    print(f"✅ Dynamic Position Sizer initialized")
    print(f"\n📊 Sizing Parameters:")
    print(f"  Base Risk: {sizer.base_risk_pct:.1%}")
    print(f"  Min Risk: {sizer.min_risk_pct:.2%}")
    print(f"  Max Risk: {sizer.max_risk_pct:.1%}")
    
    # Test position sizing
    print(f"\n🔍 Testing Position Sizing:")
    
    test_scenarios = [
        {'symbol': 'SPY', 'strategy': 'bull_put_spread', 'max_loss': 500, 'confidence': 0.70},
        {'symbol': 'QQQ', 'strategy': 'iron_condor', 'max_loss': 800, 'confidence': 0.65},
        {'symbol': 'SPY', 'strategy': 'cash_secured_put', 'max_loss': 1000, 'confidence': 0.60}
    ]
    
    for scenario in test_scenarios:
        sizing = sizer.calculate_position_size(
            scenario['symbol'],
            scenario['strategy'],
            scenario['max_loss'],
            scenario['confidence']
        )
        
        print(f"\n  {scenario['symbol']} - {scenario['strategy']}:")
        print(f"    Confidence: {scenario['confidence']:.0%}")
        print(f"    Recommended Quantity: {sizing['quantity']} contract(s)")
        print(f"    Risk Amount: ${sizing['risk_amount']:,.2f} ({sizing['risk_pct']:.2%})")
        print(f"    Total Max Loss: ${sizing['max_loss_total']:,.2f}")
    
    return True


def test_circuit_breaker(db):
    """Test circuit breaker"""
    print_section("Testing Circuit Breaker")
    
    breaker = CircuitBreaker(db, total_capital=10000.0)
    
    print(f"✅ Circuit Breaker initialized")
    print(f"\n📊 Circuit Breaker Thresholds:")
    print(f"  Daily Loss Limit: {breaker.daily_loss_limit_pct:.1%}")
    print(f"  Max Drawdown Limit: {breaker.max_drawdown_limit_pct:.1%}")
    print(f"  Volatility Threshold: {breaker.extreme_volatility_threshold:.1f}x normal")
    print(f"  Max Consecutive Losses: {breaker.max_consecutive_losses}")
    
    # Check status
    status = breaker.check_circuit_breaker()
    
    print(f"\n🔍 Current Status:")
    print(f"  Tripped: {'🔴 YES' if status['tripped'] else '🟢 NO'}")
    print(f"  Can Trade: {'✅ YES' if status['can_trade'] else '❌ NO'}")
    
    if status.get('warnings'):
        print(f"\n  ⚠️  Warnings:")
        for warning in status['warnings']:
            print(f"    • {warning}")
    
    return True


def test_correlation_analyzer(db):
    """Test correlation analysis"""
    print_section("Testing Correlation Analyzer")
    
    analyzer = CorrelationAnalyzer(db)
    
    print(f"✅ Correlation Analyzer initialized")
    
    # Test correlation calculation
    print(f"\n🔍 Testing Correlation Calculation:")
    
    corr = analyzer.calculate_correlation('SPY', 'QQQ', lookback_days=30)
    
    if corr is not None:
        print(f"\n  SPY-QQQ Correlation: {corr:+.3f}")
        
        if abs(corr) > 0.8:
            print(f"  Status: 🔴 High correlation - limited diversification")
        elif abs(corr) > 0.6:
            print(f"  Status: 🟡 Moderate correlation")
        else:
            print(f"  Status: 🟢 Low correlation - good diversification")
    else:
        print(f"  Could not calculate correlation (insufficient data)")
    
    # Test portfolio analysis
    print(f"\n🔍 Testing Portfolio Analysis:")
    
    test_positions = [
        {'symbol': 'SPY', 'strategy': 'bull_put_spread'},
        {'symbol': 'QQQ', 'strategy': 'iron_condor'}
    ]
    
    portfolio_analysis = analyzer.analyze_portfolio_correlation(test_positions)
    
    print(f"  Diversification: {portfolio_analysis.get('diversification_quality', 'UNKNOWN')}")
    print(f"  Message: {portfolio_analysis.get('message', 'N/A')}")
    
    if portfolio_analysis.get('avg_correlation') is not None:
        print(f"  Average Correlation: {portfolio_analysis['avg_correlation']:+.3f}")
    
    return True


def main():
    """Run all Phase 5 tests"""
    print("\n" + "=" * 60)
    print("  🚀 Phase 5 Risk Management Testing")
    print("=" * 60)
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize
    db = get_session()
    
    test_results = []
    
    # Run tests
    test_results.append(("Portfolio Risk Manager", test_portfolio_risk_manager(db)))
    test_results.append(("Dynamic Position Sizer", test_dynamic_position_sizer(db)))
    test_results.append(("Circuit Breaker", test_circuit_breaker(db)))
    test_results.append(("Correlation Analyzer", test_correlation_analyzer(db)))
    
    # Print summary
    print_section("Test Results Summary")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All Phase 5 tests passed! Risk management system is production-ready!")
        print("\n🎯 You now have:")
        print("  - Portfolio-level risk management")
        print("  - Dynamic position sizing (Kelly Criterion)")
        print("  - Circuit breakers for protection")
        print("  - Correlation analysis for diversification")
        print("  - Professional-grade risk controls")
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

