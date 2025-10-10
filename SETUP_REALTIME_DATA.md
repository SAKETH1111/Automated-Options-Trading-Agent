# 🚀 Real-Time Data Collection - Setup Instructions

## What You Have Now

I've implemented a complete **real-time data collection system** that monitors SPY and QQQ **every second** during market hours and stores all data in your database.

## Quick Setup (3 Commands)

### 1. Create the Database Table
```bash
python scripts/migrate_add_tick_data.py
```

Expected output:
```
DATABASE MIGRATION: Adding IndexTickData table
Creating index_tick_data table...
✅ Table created successfully
✅ Migration completed successfully
Ready to collect real-time tick data! 📊
```

### 2. Start the Trading Agent
```bash
python main.py
```

You'll see new log messages:
```
RealTimeDataCollector initialized for ['SPY', 'QQQ']
✅ Real-time data collection started
Collection loop started
Collected tick: SPY @ $450.25 (Δ +0.015%)
Collected tick: QQQ @ $380.50 (Δ -0.008%)
```

### 3. View Your Data (After 10+ Minutes)
```bash
python scripts/view_tick_data.py
```

## What Gets Collected (Every Second)

For **SPY and QQQ**, the system captures:

✅ **Price Data**: Current price, bid, ask, spread ($ and %)  
✅ **Volume**: Cumulative daily volume + last trade size  
✅ **Market Context**: VIX level  
✅ **Technical Indicators**: 5-second SMA, 60-second SMA  
✅ **Momentum**: Price change from previous tick ($ and %)  
✅ **Market State**: open, pre_market, after_hours, closed  

## Files Created

```
New Files:
├── src/market_data/
│   ├── realtime_collector.py     # Real-time data collector
│   └── tick_analyzer.py          # Analysis tools
│
├── scripts/
│   ├── migrate_add_tick_data.py  # Database setup
│   ├── view_tick_data.py         # Interactive data viewer
│   └── test_realtime_collection.py  # Test script
│
├── examples/
│   └── analyze_tick_data.py      # Usage examples
│
└── docs/
    ├── REALTIME_DATA_COLLECTION.md     # Full documentation
    ├── REALTIME_DATA_QUICKSTART.md     # Quick start guide
    └── REALTIME_DATA_IMPLEMENTATION.md # Implementation details

Modified Files:
├── src/database/models.py        # Added IndexTickData model
├── src/orchestrator.py           # Integrated collector
└── config/spy_qqq_config.yaml    # Added config section
```

## Available Tools

### 1. Interactive Data Viewer
```bash
python scripts/view_tick_data.py
```

Menu options:
1. View recent ticks
2. View minute bars (OHLCV)
3. Export to CSV
4. Find price reversals
5. Calculate VIX correlation
6. Exit

### 2. Test the System
```bash
python scripts/test_realtime_collection.py
```

Runs for 20 seconds and verifies everything works.

### 3. See Examples
```bash
python examples/analyze_tick_data.py
```

Shows 6 practical examples:
1. Daily trading summary
2. Intraday volatility profile
3. Significant price movements
4. VIX correlation analysis
5. Minute bars generation
6. Data availability check

## Programmatic Usage

### Get Recent Data
```python
from src.market_data.tick_analyzer import TickDataAnalyzer
from datetime import datetime, timedelta

analyzer = TickDataAnalyzer()

# Get today's statistics
stats = analyzer.get_daily_statistics('SPY', datetime.now())
print(f"Range: {stats['range_pct']:.2f}%")
print(f"Volatility: {stats['volatility']:.3f}%")

# Get last hour as DataFrame
end = datetime.now()
start = end - timedelta(hours=1)
df = analyzer.get_tick_data_df('SPY', start, end)
print(df.describe())

# Get minute bars
bars = analyzer.get_minute_bars('SPY', start, end)
print(bars[['open', 'high', 'low', 'close']])
```

### Check Collection Status
```python
from src.orchestrator import TradingOrchestrator

orch = TradingOrchestrator()
status = orch.get_status()

print(status['realtime_data'])
# {
#   'is_running': True,
#   'total_ticks_collected': 23400,
#   'total_ticks_stored': 23400,
#   'collection_errors': 0,
#   'symbols': ['SPY', 'QQQ']
# }
```

## Configuration

Edit `config/spy_qqq_config.yaml`:

```yaml
realtime_data:
  enabled: true
  collect_interval_seconds: 1.0   # Collect every second
  buffer_size: 100                # Batch size for DB inserts
  retention_days: 30              # Keep 30 days of data
```

### Common Adjustments

**Collect every 5 seconds instead:**
```yaml
collect_interval_seconds: 5.0
```

**Keep only 7 days of data:**
```yaml
retention_days: 7
```

**Add more symbols:**
```yaml
scanning:
  watchlist:
    - SPY
    - QQQ
    - IWM
    - DIA
```

## Data Storage

### Expected Volume
- **Per Day**: ~23,400 ticks per symbol (~2-3 MB)
- **Per Month**: ~468,000 ticks per symbol (~50 MB)
- **SPY + QQQ per month**: ~100 MB

### Cleanup Old Data
```python
from src.market_data.tick_analyzer import TickDataAnalyzer

analyzer = TickDataAnalyzer()
deleted = analyzer.clean_old_data(days_to_keep=30)
print(f"Deleted {deleted} old records")
```

## Use Cases

### 1. Optimize Entry/Exit Timing
Find the best hours to trade based on volatility patterns.

### 2. Monitor Execution Costs
Track bid-ask spreads to optimize limit orders.

### 3. Pattern Recognition
Identify recurring intraday price patterns.

### 4. Backtest with Real Data
Use actual tick data for accurate backtest execution simulation.

### 5. Risk Management
Understand typical intraday ranges for position sizing.

### 6. Market Microstructure
Study order flow and price dynamics.

## Troubleshooting

### "No data collected"
✅ **Solution**: Market must be open (9:30-16:00 ET, Mon-Fri)
- Check: `python scripts/test_realtime_collection.py`

### "Collection errors increasing"
✅ **Solution**: Check Alpaca API
- Verify API keys in `.env`
- Check API rate limits

### "Database too large"
✅ **Solution**: Clean old data
```python
from src.market_data.tick_analyzer import TickDataAnalyzer
TickDataAnalyzer().clean_old_data(days_to_keep=7)
```

## Documentation

📚 **Full Guide**: `docs/REALTIME_DATA_COLLECTION.md`  
🚀 **Quick Start**: `REALTIME_DATA_QUICKSTART.md`  
🔧 **Implementation**: `REALTIME_DATA_IMPLEMENTATION.md`  

## Next Steps

1. ✅ **Run migration** (Step 1 above)
2. ✅ **Start agent** (Step 2 above)
3. ⏰ **Wait 1-2 hours** during market hours
4. 📊 **View data** (Step 3 above)
5. 🎯 **Analyze and optimize** your strategies

## Summary

You now have:
- ✅ Automatic data collection every second
- ✅ Complete historical tick database
- ✅ Powerful analysis tools
- ✅ Interactive data viewer
- ✅ Export capabilities
- ✅ Example scripts
- ✅ Full documentation

Start collecting data today. The more data you accumulate, the better insights you'll gain! 📊🚀

