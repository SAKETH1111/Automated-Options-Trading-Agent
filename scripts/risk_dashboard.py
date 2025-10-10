#!/usr/bin/env python3
"""
Risk Dashboard Script
Monitor portfolio risk in real-time
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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def display_portfolio_risk(risk_manager: PortfolioRiskManager):
    """Display portfolio risk metrics"""
    print_section("📊 PORTFOLIO RISK METRICS")
    
    metrics = risk_manager.get_portfolio_risk_metrics()
    
    if not metrics:
        print("  No risk metrics available")
        return
    
    print(f"\n💰 Capital:")
    print(f"  Total Capital: ${metrics['total_capital']:,.2f}")
    print(f"  Total Positions: {metrics['total_positions']}")
    
    print(f"\n📉 Risk Exposure:")
    print(f"  Total Risk: ${metrics['total_risk']:,.2f} ({metrics['total_risk_pct']:.2f}%)")
    print(f"  Available Risk: ${metrics['available_risk']:,.2f}")
    
    print(f"\n📊 By Symbol:")
    for symbol, data in metrics.get('by_symbol', {}).items():
        risk_pct = (data['risk'] / metrics['total_capital']) * 100
        print(f"  {symbol}: {data['count']} positions, ${data['risk']:,.2f} ({risk_pct:.1f}%)")
    
    print(f"\n📊 By Strategy:")
    for strategy, data in metrics.get('by_strategy', {}).items():
        risk_pct = (data['risk'] / metrics['total_capital']) * 100
        print(f"  {strategy}: {data['count']} positions, ${data['risk']:,.2f} ({risk_pct:.1f}%)")
    
    print(f"\n📅 Daily Performance:")
    print(f"  Daily P&L: ${metrics['daily_loss']:+,.2f} ({metrics['daily_loss_pct']:+.2f}%)")
    print(f"  Daily Limit: {metrics['limits']['daily_loss_limit']:.1%}")
    
    print(f"\n📉 Drawdown:")
    print(f"  Current Drawdown: {metrics['current_drawdown_pct']:.2f}%")
    print(f"  Max Drawdown Limit: {metrics['limits']['max_drawdown_limit']:.1%}")
    
    # Risk status
    risk_pct = metrics['total_risk_pct']
    if risk_pct < 5:
        status = "🟢 LOW RISK"
    elif risk_pct < 8:
        status = "🟡 MODERATE RISK"
    else:
        status = "🔴 HIGH RISK"
    
    print(f"\n🎯 Risk Status: {status}")


def display_circuit_breaker_status(circuit_breaker: CircuitBreaker):
    """Display circuit breaker status"""
    print_section("🚨 CIRCUIT BREAKER STATUS")
    
    status = circuit_breaker.check_circuit_breaker()
    
    if status['tripped']:
        print(f"\n  🔴 CIRCUIT BREAKER TRIPPED")
        print(f"  Reason: {status['reason']}")
        print(f"  Trading: PAUSED")
    else:
        print(f"\n  🟢 CIRCUIT BREAKER ACTIVE")
        print(f"  Trading: ALLOWED")
    
    if status.get('warnings'):
        print(f"\n  ⚠️  Warnings:")
        for warning in status['warnings']:
            print(f"    • {warning}")
    
    print(f"\n  📊 Thresholds:")
    print(f"    Daily Loss Limit: {circuit_breaker.daily_loss_limit_pct:.1%}")
    print(f"    Max Drawdown Limit: {circuit_breaker.max_drawdown_limit_pct:.1%}")
    print(f"    Volatility Threshold: {circuit_breaker.extreme_volatility_threshold:.1f}x")
    print(f"    Max Consecutive Losses: {circuit_breaker.max_consecutive_losses}")


def display_position_sizing(position_sizer: DynamicPositionSizer):
    """Display position sizing recommendations"""
    print_section("📏 DYNAMIC POSITION SIZING")
    
    # Test position sizing for different scenarios
    test_scenarios = [
        {'symbol': 'SPY', 'strategy': 'bull_put_spread', 'max_loss': 500, 'confidence': 0.70},
        {'symbol': 'QQQ', 'strategy': 'iron_condor', 'max_loss': 800, 'confidence': 0.65}
    ]
    
    print(f"\n  Base Risk: {position_sizer.base_risk_pct:.1%} of capital")
    print(f"  Range: {position_sizer.min_risk_pct:.2%} - {position_sizer.max_risk_pct:.1%}")
    
    print(f"\n  📊 Sizing Examples:")
    
    for scenario in test_scenarios:
        sizing = position_sizer.calculate_position_size(
            scenario['symbol'],
            scenario['strategy'],
            scenario['max_loss'],
            scenario['confidence']
        )
        
        print(f"\n  {scenario['symbol']} {scenario['strategy']}:")
        print(f"    Quantity: {sizing['quantity']} contract(s)")
        print(f"    Risk Amount: ${sizing['risk_amount']:,.2f} ({sizing['risk_pct']:.2%})")
        print(f"    Total Max Loss: ${sizing['max_loss_total']:,.2f}")
        print(f"    Adjustments:")
        adj = sizing.get('adjustments', {})
        print(f"      Volatility: {adj.get('volatility', 1.0):.2f}x")
        print(f"      Kelly: {adj.get('kelly', 1.0):.2f}x")
        print(f"      Regime: {adj.get('regime', 1.0):.2f}x")


def display_correlation_analysis(correlation_analyzer: CorrelationAnalyzer, positions: List[Dict]):
    """Display correlation analysis"""
    print_section("🔗 CORRELATION ANALYSIS")
    
    if not positions:
        print("  No open positions")
        return
    
    analysis = correlation_analyzer.analyze_portfolio_correlation(positions)
    
    print(f"\n  Diversification: {analysis.get('diversification_quality', 'UNKNOWN')}")
    print(f"  Status: {analysis.get('message', 'N/A')}")
    
    if analysis.get('correlations'):
        print(f"\n  📊 Pairwise Correlations:")
        for pair, corr in analysis['correlations'].items():
            corr_status = "🔴" if abs(corr) > 0.8 else "🟡" if abs(corr) > 0.6 else "🟢"
            print(f"    {corr_status} {pair}: {corr:+.3f}")
    
    if analysis.get('high_correlations'):
        print(f"\n  ⚠️  High Correlations Detected:")
        for item in analysis['high_correlations']:
            print(f"    • {item['pair']}: {item['correlation']:+.3f}")
            print(f"      {item['warning']}")


def main():
    """Main risk dashboard"""
    print("\n" + "=" * 70)
    print("  🛡️  RISK MANAGEMENT DASHBOARD")
    print("=" * 70)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize
    db = get_session()
    
    total_capital = 10000.0  # Adjust based on your account
    
    risk_manager = PortfolioRiskManager(db, total_capital)
    position_sizer = DynamicPositionSizer(db, total_capital)
    circuit_breaker = CircuitBreaker(db, total_capital)
    correlation_analyzer = CorrelationAnalyzer(db)
    
    # Display all risk metrics
    display_portfolio_risk(risk_manager)
    display_circuit_breaker_status(circuit_breaker)
    display_position_sizing(position_sizer)
    
    # Get positions for correlation analysis
    positions = risk_manager._get_open_positions()
    display_correlation_analysis(correlation_analyzer, positions)
    
    print("\n" + "=" * 70)
    print("  Risk dashboard complete")
    print("=" * 70)
    
    db.close()


if __name__ == '__main__':
    main()

