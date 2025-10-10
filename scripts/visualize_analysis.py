#!/usr/bin/env python3
"""
Visualization Script for Phase 1 Analysis
Creates charts showing price, indicators, and patterns
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.database.session import get_session
from src.analysis.analyzer import MarketAnalyzer

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not installed. Run: pip install matplotlib")


def plot_price_with_indicators(analyzer: MarketAnalyzer, symbol: str, minutes: int = 60):
    """
    Plot price chart with technical indicators
    
    Args:
        analyzer: Market analyzer instance
        symbol: Symbol to plot
        minutes: Minutes of data to plot
    """
    if not HAS_MATPLOTLIB:
        print("❌ Matplotlib not installed. Cannot create charts.")
        print("   Install with: pip install matplotlib")
        return
    
    # Get data
    df = analyzer.get_recent_data(symbol, minutes=minutes)
    
    if df.empty:
        print(f"❌ No data available for {symbol}")
        return
    
    # Calculate indicators
    df = analyzer.indicators.calculate_all_indicators(df)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    # Plot 1: Price with Moving Averages and Bollinger Bands
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['timestamp'], df['price'], label='Price', color='black', linewidth=2)
    
    if 'sma_10' in df.columns:
        ax1.plot(df['timestamp'], df['sma_10'], label='SMA(10)', color='blue', alpha=0.7)
    if 'sma_20' in df.columns:
        ax1.plot(df['timestamp'], df['sma_20'], label='SMA(20)', color='orange', alpha=0.7)
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        ax1.fill_between(df['timestamp'], df['bb_upper'], df['bb_lower'], 
                         alpha=0.2, color='gray', label='Bollinger Bands')
    
    ax1.set_title(f'{symbol} - Price Chart with Indicators', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI
    ax2 = fig.add_subplot(gs[1])
    if 'rsi' in df.columns:
        ax2.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax2.fill_between(df['timestamp'], 30, 70, alpha=0.1, color='gray')
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: MACD
    ax3 = fig.add_subplot(gs[2])
    if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
        ax3.plot(df['timestamp'], df['macd'], label='MACD', color='blue', linewidth=2)
        ax3.plot(df['timestamp'], df['macd_signal'], label='Signal', color='red', linewidth=2)
        ax3.bar(df['timestamp'], df['macd_histogram'], label='Histogram', 
                color='gray', alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('MACD', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volume
    ax4 = fig.add_subplot(gs[3])
    if 'volume' in df.columns:
        ax4.bar(df['timestamp'], df['volume'], color='steelblue', alpha=0.6)
        ax4.set_ylabel('Volume', fontsize=12)
        ax4.set_xlabel('Time', fontsize=12)
        ax4.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved: {filename}")
    
    # Show plot
    plt.show()


def print_analysis_report(analyzer: MarketAnalyzer, symbol: str):
    """
    Print text-based analysis report
    
    Args:
        analyzer: Market analyzer instance
        symbol: Symbol to analyze
    """
    print("\n" + "=" * 70)
    print(f"  📊 {symbol} - Technical Analysis Report")
    print("=" * 70)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run analysis
    analysis = analyzer.analyze_symbol(symbol, store_results=False)
    
    if 'error' in analysis:
        print(f"\n❌ Analysis failed: {analysis['error']}")
        return
    
    # Current Price
    print(f"\n💰 Current Price: ${analysis['current_price']:.2f}")
    print(f"   Data Points: {analysis['data_points']}")
    
    # Indicator Signals
    print(f"\n📊 Technical Indicators:")
    signals = analysis['indicators']['signals']
    for key, value in signals.items():
        print(f"   • {key.upper()}: {value}")
    
    # Pattern Analysis
    print(f"\n📈 Pattern Analysis:")
    patterns = analysis['patterns']
    
    if 'trend' in patterns:
        trend = patterns['trend']
        print(f"   • Trend: {trend.get('description')}")
    
    if 'support_resistance' in patterns:
        sr = patterns['support_resistance']
        if sr.get('support'):
            print(f"   • Support Levels: {[f'${x:.2f}' for x in sr['support'][:3]]}")
        if sr.get('resistance'):
            print(f"   • Resistance Levels: {[f'${x:.2f}' for x in sr['resistance'][:3]]}")
    
    if 'breakout' in patterns and patterns['breakout'].get('breakout'):
        breakout = patterns['breakout']
        print(f"   • Breakout: {breakout.get('direction')} (Volume: {breakout.get('volume_confirmed')})")
    
    if 'reversals' in patterns and patterns['reversals'].get('reversal_detected'):
        print(f"   • Reversal Pattern: {patterns['reversals'].get('type')}")
    
    # Market Regime
    print(f"\n🌐 Market Regime:")
    regime = analysis['regime']
    
    if 'trend' in regime:
        print(f"   • Trend: {regime['trend'].get('regime')}")
    if 'volatility' in regime:
        print(f"   • Volatility: {regime['volatility'].get('regime')}")
    if 'momentum' in regime:
        print(f"   • Momentum: {regime['momentum'].get('regime')}")
    
    # Trading Recommendation
    if 'recommendation' in regime:
        rec = regime['recommendation']
        print(f"\n💡 Trading Recommendation:")
        print(f"   • Action: {rec.get('action')}")
        print(f"   • Strategy: {rec.get('strategy')}")
    
    # Generate trading signals
    signals = analyzer.generate_trading_signals(symbol)
    
    print(f"\n🎯 Trading Signal:")
    print(f"   • Overall: {signals['overall_signal']}")
    print(f"   • Confidence: {signals['confidence']:.1%}")
    if signals.get('stop_loss'):
        print(f"   • Stop Loss: ${signals['stop_loss']:.2f}")
    if signals.get('take_profit'):
        print(f"   • Take Profit: ${signals['take_profit']:.2f}")
    
    print(f"\n📋 Signal Reasons:")
    for reason in signals['reasons']:
        print(f"   • {reason}")
    
    print("\n" + "=" * 70)


def main():
    """Main visualization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize technical analysis')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to analyze')
    parser.add_argument('--minutes', type=int, default=60, help='Minutes of data')
    parser.add_argument('--chart', action='store_true', help='Generate chart')
    parser.add_argument('--report', action='store_true', help='Print text report')
    parser.add_argument('--both', action='store_true', help='Both chart and report')
    
    args = parser.parse_args()
    
    # If no flags, do both
    if not (args.chart or args.report or args.both):
        args.both = True
    
    # Initialize
    db = get_session()
    analyzer = MarketAnalyzer(db)
    
    # Print report
    if args.report or args.both:
        print_analysis_report(analyzer, args.symbol)
    
    # Generate chart
    if args.chart or args.both:
        if HAS_MATPLOTLIB:
            plot_price_with_indicators(analyzer, args.symbol, args.minutes)
        else:
            print("\n⚠️  Chart generation skipped (matplotlib not installed)")
    
    db.close()


if __name__ == '__main__':
    main()

