#!/usr/bin/env python3
"""
Data Analysis Script - Phase 1 Starter
Analyze collected tick data and calculate basic statistics
"""

import os
import sys
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_database_url():
    """Get database URL from environment or use default"""
    return os.environ.get(
        'DATABASE_URL',
        'postgresql://trading_user:trading_password@localhost:5432/options_trading'
    )

def analyze_data_collection():
    """Analyze the collected tick data"""
    print("📊 Data Collection Analysis")
    print("=" * 50)
    
    engine = create_engine(get_database_url())
    
    # Overall statistics
    query = """
    SELECT 
        COUNT(*) as total_ticks,
        COUNT(DISTINCT symbol) as symbols,
        MIN(timestamp) as first_data,
        MAX(timestamp) as latest_data,
        MAX(timestamp) - MIN(timestamp) as collection_duration
    FROM index_tick_data;
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchone()
        
        print(f"\n📈 Overall Statistics:")
        print(f"  Total Ticks: {result[0]:,}")
        print(f"  Symbols: {result[1]}")
        print(f"  First Data: {result[2]}")
        print(f"  Latest Data: {result[3]}")
        print(f"  Duration: {result[4]}")

def analyze_by_symbol():
    """Analyze data by symbol"""
    print("\n\n💰 Analysis by Symbol")
    print("=" * 50)
    
    engine = create_engine(get_database_url())
    
    query = """
    SELECT 
        symbol,
        COUNT(*) as ticks,
        MIN(price) as min_price,
        MAX(price) as max_price,
        AVG(price) as avg_price,
        STDDEV(price) as price_stddev,
        MIN(timestamp) as first_tick,
        MAX(timestamp) as last_tick
    FROM index_tick_data
    GROUP BY symbol
    ORDER BY ticks DESC;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
        for _, row in df.iterrows():
            print(f"\n{row['symbol']}:")
            print(f"  Ticks Collected: {row['ticks']:,}")
            print(f"  Price Range: ${row['min_price']:.2f} - ${row['max_price']:.2f}")
            print(f"  Average Price: ${row['avg_price']:.2f}")
            print(f"  Price Volatility: ${row['price_stddev']:.2f}")
            print(f"  First Tick: {row['first_tick']}")
            print(f"  Last Tick: {row['last_tick']}")

def calculate_basic_indicators():
    """Calculate basic technical indicators"""
    print("\n\n📊 Basic Technical Indicators")
    print("=" * 50)
    
    engine = create_engine(get_database_url())
    
    # Get recent data for each symbol
    query = """
    SELECT symbol, price, timestamp
    FROM index_tick_data
    WHERE timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY symbol, timestamp;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Calculate simple moving averages
            symbol_data['sma_10'] = symbol_data['price'].rolling(window=10).mean()
            symbol_data['sma_30'] = symbol_data['price'].rolling(window=30).mean()
            
            # Calculate price change
            symbol_data['price_change'] = symbol_data['price'].diff()
            symbol_data['price_change_pct'] = symbol_data['price'].pct_change() * 100
            
            # Get latest values
            latest = symbol_data.iloc[-1]
            
            print(f"\n{symbol} (Last Hour):")
            print(f"  Current Price: ${latest['price']:.2f}")
            print(f"  SMA(10): ${latest['sma_10']:.2f}")
            print(f"  SMA(30): ${latest['sma_30']:.2f}")
            
            # Trend detection
            if pd.notna(latest['sma_10']) and pd.notna(latest['sma_30']):
                if latest['sma_10'] > latest['sma_30']:
                    trend = "🟢 BULLISH (Short-term > Long-term)"
                else:
                    trend = "🔴 BEARISH (Short-term < Long-term)"
                print(f"  Trend: {trend}")
            
            # Recent volatility
            recent_volatility = symbol_data['price_change'].std()
            print(f"  Recent Volatility: ${recent_volatility:.2f}")

def analyze_recent_activity():
    """Analyze recent trading activity"""
    print("\n\n⏰ Recent Activity (Last 5 Minutes)")
    print("=" * 50)
    
    engine = create_engine(get_database_url())
    
    query = """
    SELECT 
        symbol,
        COUNT(*) as ticks,
        MIN(price) as low,
        MAX(price) as high,
        (MAX(price) - MIN(price)) as range,
        AVG(volume) as avg_volume
    FROM index_tick_data
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    GROUP BY symbol;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
        for _, row in df.iterrows():
            print(f"\n{row['symbol']}:")
            print(f"  Ticks: {row['ticks']}")
            print(f"  Range: ${row['low']:.2f} - ${row['high']:.2f}")
            print(f"  Price Movement: ${row['range']:.2f}")
            print(f"  Avg Volume: {row['avg_volume']:,.0f}")

def generate_daily_summary():
    """Generate daily summary report"""
    print("\n\n📋 Daily Summary")
    print("=" * 50)
    
    engine = create_engine(get_database_url())
    
    query = """
    SELECT 
        symbol,
        COUNT(*) as ticks_today,
        MIN(price) as day_low,
        MAX(price) as day_high,
        (SELECT price FROM index_tick_data WHERE symbol = t.symbol ORDER BY timestamp ASC LIMIT 1) as open_price,
        (SELECT price FROM index_tick_data WHERE symbol = t.symbol ORDER BY timestamp DESC LIMIT 1) as close_price
    FROM index_tick_data t
    WHERE DATE(timestamp) = CURRENT_DATE
    GROUP BY symbol;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
        for _, row in df.iterrows():
            day_change = row['close_price'] - row['open_price']
            day_change_pct = (day_change / row['open_price']) * 100
            
            print(f"\n{row['symbol']} - Today's Performance:")
            print(f"  Open: ${row['open_price']:.2f}")
            print(f"  High: ${row['day_high']:.2f}")
            print(f"  Low: ${row['day_low']:.2f}")
            print(f"  Current: ${row['close_price']:.2f}")
            print(f"  Change: ${day_change:+.2f} ({day_change_pct:+.2f}%)")
            print(f"  Data Points: {row['ticks_today']:,}")

def main():
    """Main analysis function"""
    print("\n" + "=" * 50)
    print("🚀 Trading Agent Data Analysis")
    print("=" * 50)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        analyze_data_collection()
        analyze_by_symbol()
        calculate_basic_indicators()
        analyze_recent_activity()
        generate_daily_summary()
        
        print("\n\n" + "=" * 50)
        print("✅ Analysis Complete!")
        print("=" * 50)
        
        print("\n💡 Next Steps:")
        print("  1. Review the statistics above")
        print("  2. Look for patterns in price movements")
        print("  3. Consider which indicators to add")
        print("  4. Start building your trading strategy")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("\nMake sure:")
        print("  1. Database is running")
        print("  2. Data collection is active")
        print("  3. You have pandas installed: pip install pandas")

if __name__ == '__main__':
    main()
