"""Example: Analyzing collected tick data for trading insights"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market_data.tick_analyzer import TickDataAnalyzer


def example_1_daily_summary():
    """Example 1: Get daily trading summary"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Daily Trading Summary")
    print("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    # Get today's statistics for SPY
    stats = analyzer.get_daily_statistics('SPY', datetime.now())
    
    if not stats:
        print("No data available. Make sure the agent is running during market hours.")
        return
    
    print(f"\n{stats['symbol']} - {stats['date']}")
    print("-" * 40)
    print(f"Open:         ${stats['open']:.2f}")
    print(f"High:         ${stats['high']:.2f}")
    print(f"Low:          ${stats['low']:.2f}")
    print(f"Close:        ${stats['close']:.2f}")
    print(f"Range:        ${stats['range']:.2f} ({stats['range_pct']:.2f}%)")
    print(f"Volatility:   {stats['volatility']:.3f}%")
    print(f"Avg Spread:   {stats['avg_spread_pct']:.3f}%")
    print(f"Ticks:        {stats['tick_count']:,}")
    
    if stats['avg_vix']:
        print(f"Avg VIX:      {stats['avg_vix']:.2f}")
    
    # Trading insights
    print("\n📊 Trading Insights:")
    
    if stats['volatility'] > 0.2:
        print("  • High volatility - good for premium selling strategies")
    else:
        print("  • Low volatility - consider directional strategies")
    
    if stats['avg_spread_pct'] < 0.05:
        print("  • Tight spreads - excellent liquidity")
    elif stats['avg_spread_pct'] < 0.1:
        print("  • Normal spreads - good liquidity")
    else:
        print("  • Wide spreads - be careful with execution")
    
    if stats['range_pct'] > 1.0:
        print("  • Large daily range - strong trending day")
    else:
        print("  • Small daily range - choppy/sideways market")


def example_2_intraday_volatility():
    """Example 2: Analyze intraday volatility patterns"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Intraday Volatility Profile")
    print("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    # Get hourly volatility breakdown
    profile = analyzer.calculate_volatility_profile('SPY', datetime.now())
    
    if not profile:
        print("No data available for volatility analysis.")
        return
    
    print("\nHourly Volatility Breakdown:")
    print("-" * 60)
    print(f"{'Hour':<10} {'Avg Price':<15} {'Volatility':<15} {'Avg Spread':<15}")
    print("-" * 60)
    
    for hour, data in sorted(profile.items()):
        hour_str = f"{hour}:00"
        avg_price = f"${data.get('price_mean', 0):.2f}"
        volatility = f"{data.get('price_std', 0):.3f}"
        spread = f"{data.get('spread_pct_mean', 0):.3f}%"
        print(f"{hour_str:<10} {avg_price:<15} {volatility:<15} {spread:<15}")
    
    print("\n📊 Best Trading Hours:")
    # Find hours with highest volatility
    vol_by_hour = [(h, d.get('price_std', 0)) for h, d in profile.items()]
    vol_by_hour.sort(key=lambda x: x[1], reverse=True)
    
    if vol_by_hour:
        top_3 = vol_by_hour[:3]
        print("  Highest Volatility:")
        for hour, vol in top_3:
            print(f"    • {hour}:00 - {vol:.3f}")
        print("  → Best for: Opening positions, scalping")
        
        bottom_3 = vol_by_hour[-3:]
        print("\n  Lowest Volatility:")
        for hour, vol in bottom_3:
            print(f"    • {hour}:00 - {vol:.3f}")
        print("  → Best for: Avoiding whipsaws, tight stops")


def example_3_price_movements():
    """Example 3: Find significant price movements"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Significant Price Movements")
    print("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    # Look at last 2 hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)
    
    # Find moves > 0.3%
    reversals = analyzer.find_price_reversals(
        'SPY', start_time, end_time, threshold_pct=0.3
    )
    
    if not reversals:
        print("No significant price movements found in the last 2 hours.")
        return
    
    print(f"\nFound {len(reversals)} significant price movements:")
    print("-" * 80)
    print(f"{'Time':<20} {'Price':<12} {'Change':<12} {'VIX':<10}")
    print("-" * 80)
    
    for reversal in reversals[-10:]:  # Show last 10
        time_str = reversal['timestamp'].strftime('%H:%M:%S')
        price_str = f"${reversal['price']:.2f}"
        change_str = f"{reversal['returns']:+.3f}%"
        vix_str = f"{reversal['vix']:.2f}" if reversal['vix'] else "N/A"
        print(f"{time_str:<20} {price_str:<12} {change_str:<12} {vix_str:<10}")
    
    print("\n📊 Movement Analysis:")
    
    up_moves = [r for r in reversals if r['returns'] > 0]
    down_moves = [r for r in reversals if r['returns'] < 0]
    
    print(f"  Up moves:     {len(up_moves)}")
    print(f"  Down moves:   {len(down_moves)}")
    
    if len(up_moves) > len(down_moves) * 1.5:
        print("  → Bullish momentum detected")
    elif len(down_moves) > len(up_moves) * 1.5:
        print("  → Bearish momentum detected")
    else:
        print("  → Balanced/choppy market")


def example_4_vix_correlation():
    """Example 4: Analyze VIX correlation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: SPY/QQQ vs VIX Correlation")
    print("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    # Look at last 4 hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=4)
    
    print("\nCalculating correlations (last 4 hours)...")
    
    for symbol in ['SPY', 'QQQ']:
        corr = analyzer.get_correlation_with_vix(symbol, start_time, end_time)
        
        if corr is None:
            print(f"{symbol}: Insufficient data")
            continue
        
        print(f"\n{symbol} vs VIX: {corr:.3f}")
        
        # Interpret correlation
        if corr < -0.7:
            interpretation = "Strong negative (normal market behavior)"
            implication = "Safe to sell premium"
        elif corr < -0.5:
            interpretation = "Moderate negative (typical)"
            implication = "Normal market conditions"
        elif corr < -0.3:
            interpretation = "Weak negative"
            implication = "Watch for regime change"
        elif corr < 0.3:
            interpretation = "Near zero (unusual!)"
            implication = "Caution - abnormal conditions"
        else:
            interpretation = "Positive (very unusual!)"
            implication = "High alert - potential crisis"
        
        print(f"  Interpretation: {interpretation}")
        print(f"  Implication:    {implication}")


def example_5_minute_bars():
    """Example 5: Generate minute bars for charting"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Minute Bars (OHLCV)")
    print("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    # Get last hour of data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    bars = analyzer.get_minute_bars('SPY', start_time, end_time)
    
    if bars.empty:
        print("No data available for minute bars.")
        return
    
    print(f"\nLast 10 minute bars for SPY:")
    print("-" * 100)
    print(f"{'Time':<10} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Returns':<10} {'Ticks':<8}")
    print("-" * 100)
    
    for idx, row in bars.tail(10).iterrows():
        time_str = idx.strftime('%H:%M')
        print(
            f"{time_str:<10} "
            f"${row['open']:<9.2f} "
            f"${row['high']:<9.2f} "
            f"${row['low']:<9.2f} "
            f"${row['close']:<9.2f} "
            f"{row['returns']*100:+9.3f}% "
            f"{int(row['tick_count']):<8}"
        )
    
    print("\n📊 Bar Statistics:")
    print(f"  Total bars:      {len(bars)}")
    print(f"  Avg ticks/bar:   {bars['tick_count'].mean():.1f}")
    print(f"  Avg return:      {bars['returns'].mean()*100:+.3f}%")
    print(f"  Return std:      {bars['returns'].std()*100:.3f}%")
    print(f"  Largest gain:    {bars['returns'].max()*100:+.3f}%")
    print(f"  Largest loss:    {bars['returns'].min()*100:+.3f}%")


def example_6_data_availability():
    """Example 6: Check data availability"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Data Availability Check")
    print("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    print("\nChecking data for last 7 days...")
    
    for symbol in ['SPY', 'QQQ']:
        availability = analyzer.get_data_availability(symbol, days=7)
        
        print(f"\n{symbol}:")
        print("-" * 40)
        print(f"  Total ticks:     {availability.get('total_ticks', 0):,}")
        print(f"  Days with data:  {availability.get('days_with_data', 0)}")
        
        daily_counts = availability.get('daily_counts', {})
        if daily_counts:
            print("\n  Daily breakdown:")
            for date, count in sorted(daily_counts.items(), reverse=True):
                print(f"    {date}: {count:,} ticks")
                
                # Quality assessment
                if count > 20000:
                    print(f"      ✅ Full day")
                elif count > 5000:
                    print(f"      ⚠️  Partial day")
                else:
                    print(f"      ❌ Limited data")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("TICK DATA ANALYSIS EXAMPLES")
    print("=" * 80)
    print("\nThese examples show how to use collected tick data for trading insights.")
    print("Make sure the agent has been running for at least 1-2 hours during market hours.")
    
    try:
        example_1_daily_summary()
        example_2_intraday_volatility()
        example_3_price_movements()
        example_4_vix_correlation()
        example_5_minute_bars()
        example_6_data_availability()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed!")
        print("=" * 80)
        print("\nNext steps:")
        print("  • Use these patterns in your own analysis")
        print("  • Integrate insights into your strategies")
        print("  • Export data for further analysis")
        print("  • Build custom indicators from tick data")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

