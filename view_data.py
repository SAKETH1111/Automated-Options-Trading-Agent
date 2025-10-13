#!/usr/bin/env python3
"""
View collected market data with Central Time conversion
"""

import sys
from datetime import datetime
from pathlib import Path
import pytz

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.session import get_db
from src.database.models import IndexTickData

def main():
    """Display recent tick data in Central Time"""
    db = get_db()
    ct_tz = pytz.timezone('America/Chicago')
    
    print("=" * 80)
    print("📊 Recent Market Data (Central Time)")
    print("=" * 80)
    
    with db.get_session() as session:
        # Get recent ticks
        ticks = session.query(IndexTickData)\
            .order_by(IndexTickData.timestamp.desc())\
            .limit(20)\
            .all()
        
        print(f"\n{'Symbol':<8} {'Time (CT)':<20} {'Price':<10} {'State':<10}")
        print("-" * 80)
        
        for tick in reversed(ticks):
            # Convert UTC to Central Time
            utc_time = tick.timestamp.replace(tzinfo=pytz.UTC)
            ct_time = utc_time.astimezone(ct_tz)
            
            print(f"{tick.symbol:<8} {ct_time.strftime('%Y-%m-%d %H:%M:%S'):<20} "
                  f"${tick.price:<9.2f} {tick.market_state:<10}")
        
        # Get statistics
        print("\n" + "=" * 80)
        print("📈 Collection Statistics")
        print("=" * 80)
        
        total_ticks = session.query(IndexTickData).count()
        
        # Ticks by symbol
        from sqlalchemy import func
        symbol_counts = session.query(
            IndexTickData.symbol,
            func.count(IndexTickData.symbol).label('count'),
            func.max(IndexTickData.timestamp).label('latest')
        ).group_by(IndexTickData.symbol).all()
        
        print(f"\nTotal ticks collected: {total_ticks:,}")
        print(f"\nBy Symbol:")
        for symbol, count, latest_utc in symbol_counts:
            latest_ct = latest_utc.replace(tzinfo=pytz.UTC).astimezone(ct_tz)
            print(f"  {symbol}: {count:,} ticks (latest: {latest_ct.strftime('%Y-%m-%d %H:%M:%S %Z')})")

if __name__ == "__main__":
    main()

