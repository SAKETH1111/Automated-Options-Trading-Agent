"""View and analyze collected tick data"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.market_data.tick_analyzer import TickDataAnalyzer


def main():
    """View tick data and statistics"""
    logger.info("=" * 80)
    logger.info("TICK DATA VIEWER")
    logger.info("=" * 80)
    
    analyzer = TickDataAnalyzer()
    
    # Check data availability
    logger.info("\n📊 Data Availability (Last 7 Days)")
    logger.info("-" * 80)
    
    for symbol in ['SPY', 'QQQ']:
        availability = analyzer.get_data_availability(symbol, days=7)
        
        if availability:
            logger.info(f"\n{symbol}:")
            logger.info(f"  Total Ticks: {availability['total_ticks']:,}")
            logger.info(f"  Days with Data: {availability['days_with_data']}")
            
            if availability['daily_counts']:
                logger.info("  Daily Breakdown:")
                for date, count in sorted(availability['daily_counts'].items()):
                    logger.info(f"    {date}: {count:,} ticks")
    
    # Get today's statistics
    logger.info("\n📈 Today's Statistics")
    logger.info("-" * 80)
    
    today = datetime.now()
    
    for symbol in ['SPY', 'QQQ']:
        stats = analyzer.get_daily_statistics(symbol, today)
        
        if stats:
            logger.info(f"\n{symbol}:")
            logger.info(f"  Open: ${stats['open']:.2f}")
            logger.info(f"  High: ${stats['high']:.2f}")
            logger.info(f"  Low: ${stats['low']:.2f}")
            logger.info(f"  Close: ${stats['close']:.2f}")
            logger.info(f"  Range: ${stats['range']:.2f} ({stats['range_pct']:.2f}%)")
            logger.info(f"  Volatility: {stats['volatility']:.3f}%")
            logger.info(f"  Avg Spread: {stats['avg_spread_pct']:.3f}%")
            
            if stats['avg_vix']:
                logger.info(f"  Avg VIX: {stats['avg_vix']:.2f}")
            
            logger.info(f"  Tick Count: {stats['tick_count']:,}")
            logger.info(f"  Positive Moves: {stats['positive_moves']:,}")
            logger.info(f"  Negative Moves: {stats['negative_moves']:,}")
    
    # Interactive menu
    while True:
        logger.info("\n" + "=" * 80)
        logger.info("OPTIONS:")
        logger.info("  1. View recent ticks")
        logger.info("  2. View minute bars")
        logger.info("  3. Export to CSV")
        logger.info("  4. Find price reversals")
        logger.info("  5. Calculate VIX correlation")
        logger.info("  6. Exit")
        logger.info("=" * 80)
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                view_recent_ticks(analyzer)
            elif choice == '2':
                view_minute_bars(analyzer)
            elif choice == '3':
                export_to_csv(analyzer)
            elif choice == '4':
                find_reversals(analyzer)
            elif choice == '5':
                calculate_correlation(analyzer)
            elif choice == '6':
                logger.info("Exiting...")
                break
            else:
                logger.warning("Invalid choice")
        
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def view_recent_ticks(analyzer: TickDataAnalyzer):
    """View recent ticks"""
    symbol = input("Enter symbol (SPY/QQQ): ").strip().upper()
    limit = int(input("Number of ticks (default 20): ").strip() or "20")
    
    from src.market_data.realtime_collector import RealTimeDataCollector
    collector = RealTimeDataCollector()
    
    ticks = collector.get_recent_ticks(symbol, limit)
    
    if not ticks:
        logger.warning(f"No tick data found for {symbol}")
        return
    
    logger.info(f"\n📊 Last {len(ticks)} Ticks for {symbol}")
    logger.info("-" * 100)
    logger.info(f"{'Time':<20} {'Price':<10} {'Bid':<10} {'Ask':<10} {'Spread%':<10} {'Change%':<10}")
    logger.info("-" * 100)
    
    for tick in ticks:
        logger.info(
            f"{tick.timestamp.strftime('%H:%M:%S'):<20} "
            f"${tick.price:<9.2f} "
            f"${tick.bid:<9.2f} "
            f"${tick.ask:<9.2f} "
            f"{tick.spread_pct:<9.3f}% "
            f"{tick.price_change_pct:+9.3f}%"
        )


def view_minute_bars(analyzer: TickDataAnalyzer):
    """View minute bars"""
    symbol = input("Enter symbol (SPY/QQQ): ").strip().upper()
    minutes = int(input("Minutes back (default 60): ").strip() or "60")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=minutes)
    
    bars = analyzer.get_minute_bars(symbol, start_time, end_time)
    
    if bars.empty:
        logger.warning(f"No data found for {symbol}")
        return
    
    logger.info(f"\n📊 Minute Bars for {symbol} (Last {minutes} minutes)")
    logger.info("-" * 100)
    logger.info(f"{'Time':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Returns%':<12}")
    logger.info("-" * 100)
    
    for idx, row in bars.tail(20).iterrows():
        logger.info(
            f"{idx.strftime('%H:%M'):<20} "
            f"${row['open']:<9.2f} "
            f"${row['high']:<9.2f} "
            f"${row['low']:<9.2f} "
            f"${row['close']:<9.2f} "
            f"{row['returns']*100:+11.3f}%"
        )


def export_to_csv(analyzer: TickDataAnalyzer):
    """Export data to CSV"""
    symbol = input("Enter symbol (SPY/QQQ): ").strip().upper()
    hours = int(input("Hours back (default 1): ").strip() or "1")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    output_path = f"data/{symbol}_ticks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    success = analyzer.export_to_csv(symbol, start_time, end_time, output_path)
    
    if success:
        logger.info(f"✅ Data exported to {output_path}")
    else:
        logger.error("❌ Export failed")


def find_reversals(analyzer: TickDataAnalyzer):
    """Find price reversals"""
    symbol = input("Enter symbol (SPY/QQQ): ").strip().upper()
    hours = int(input("Hours back (default 1): ").strip() or "1")
    threshold = float(input("Threshold % (default 0.5): ").strip() or "0.5")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    reversals = analyzer.find_price_reversals(symbol, start_time, end_time, threshold)
    
    if not reversals:
        logger.info(f"No reversals found for {symbol} (threshold: {threshold}%)")
        return
    
    logger.info(f"\n📊 Found {len(reversals)} Reversals for {symbol}")
    logger.info("-" * 80)
    logger.info(f"{'Time':<20} {'Price':<12} {'Change%':<12} {'VIX':<10}")
    logger.info("-" * 80)
    
    for reversal in reversals:
        vix_str = f"{reversal['vix']:.2f}" if reversal['vix'] else "N/A"
        logger.info(
            f"{reversal['timestamp'].strftime('%H:%M:%S'):<20} "
            f"${reversal['price']:<11.2f} "
            f"{reversal['returns']:+11.3f}% "
            f"{vix_str:<10}"
        )


def calculate_correlation(analyzer: TickDataAnalyzer):
    """Calculate VIX correlation"""
    symbol = input("Enter symbol (SPY/QQQ): ").strip().upper()
    hours = int(input("Hours back (default 4): ").strip() or "4")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    correlation = analyzer.get_correlation_with_vix(symbol, start_time, end_time)
    
    if correlation is None:
        logger.warning("Unable to calculate correlation (insufficient data)")
        return
    
    logger.info(f"\n📊 {symbol} vs VIX Correlation")
    logger.info(f"  Period: Last {hours} hours")
    logger.info(f"  Correlation: {correlation:.3f}")
    
    if correlation < -0.5:
        logger.info("  Interpretation: Strong negative correlation (normal)")
    elif correlation < -0.3:
        logger.info("  Interpretation: Moderate negative correlation")
    elif correlation < 0.3:
        logger.info("  Interpretation: Weak correlation")
    else:
        logger.info("  Interpretation: Positive correlation (unusual)")


if __name__ == "__main__":
    main()

