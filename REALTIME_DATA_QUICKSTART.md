# 📊 Real-Time Data Collection - Quick Start Guide

## What Is It?

The trading agent now **monitors SPY and QQQ every second** and stores all data in a database for future analysis. This gives you:

- 📈 Second-by-second price tracking
- 💾 Complete historical tick data
- 🔍 Advanced analysis tools
- 📊 Pattern recognition capabilities
- 🎯 Better strategy optimization

## Quick Setup (3 Steps)

### Step 1: Run Migration

Add the new database table:

```bash
python scripts/migrate_add_tick_data.py
```

Output:
```
DATABASE MIGRATION: Adding IndexTickData table
✅ Table created successfully
Ready to collect real-time tick data! 📊
```

### Step 2: Start the Agent

The real-time collector starts automatically:

```bash
python main.py
```

You'll see:
```
RealTimeDataCollector initialized for ['SPY', 'QQQ']
✅ Real-time data collection started
Collection loop started
```

### Step 3: View Your Data

After a few minutes, view collected data:

```bash
python scripts/view_tick_data.py
```

## What Data Is Collected?

Every second, for each symbol (SPY, QQQ):
- **Price**: Current, bid, ask, spread
- **Volume**: Trade size and cumulative volume
- **VIX**: Market volatility indicator
- **Indicators**: Moving averages (5-sec, 60-sec)
- **Changes**: Price changes from previous tick

## Example Output

```
📊 Today's Statistics
SPY:
  Open: $450.25
  High: $452.30
  Low: $449.80
  Close: $451.95
  Range: $2.50 (0.55%)
  Volatility: 0.125%
  Tick Count: 23,400
  
QQQ:
  Open: $380.50
  High: $382.10
  Low: $379.90
  Close: $381.75
  Range: $2.20 (0.58%)
  Volatility: 0.142%
  Tick Count: 23,400
```

## Quick Commands

### Test the System
```bash
python scripts/test_realtime_collection.py
```

### View Recent Ticks
```bash
python scripts/view_tick_data.py
# Choose option 1: View recent ticks
```

### Export Data
```bash
python scripts/view_tick_data.py
# Choose option 3: Export to CSV
```

### Check Status (Python)
```python
from src.orchestrator import TradingOrchestrator

orch = TradingOrchestrator()
status = orch.get_status()
print(status['realtime_data'])
```

Output:
```python
{
    'is_running': True,
    'total_ticks_collected': 46800,  # ~2 hours of data
    'total_ticks_stored': 46800,
    'collection_errors': 0,
    'symbols': ['SPY', 'QQQ']
}
```

## Common Use Cases

### 1. Find Best Trading Hours
```python
from src.market_data.tick_analyzer import TickDataAnalyzer

analyzer = TickDataAnalyzer()
volatility = analyzer.calculate_volatility_profile('SPY', datetime.now())
# Shows which hours have highest volatility
```

### 2. Monitor Bid-Ask Spreads
```python
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(hours=1)
df = analyzer.get_tick_data_df('SPY', start, end)
print(f"Average spread: {df['spread_pct'].mean():.3f}%")
```

### 3. Analyze Price Reversals
```python
reversals = analyzer.find_price_reversals('SPY', start, end, threshold_pct=0.5)
print(f"Found {len(reversals)} significant reversals")
```

### 4. Check VIX Correlation
```python
corr = analyzer.get_correlation_with_vix('SPY', start, end)
print(f"SPY-VIX correlation: {corr:.3f}")
# Typical: -0.7 to -0.8 (negative correlation)
```

## Data Storage

### How Much Space?
- **Per day**: ~23,400 ticks per symbol (~2-3 MB)
- **Per month**: ~468,000 ticks per symbol (~50 MB)
- **Both SPY & QQQ**: ~100 MB per month

### Clean Old Data
```python
from src.market_data.tick_analyzer import TickDataAnalyzer

analyzer = TickDataAnalyzer()
deleted = analyzer.clean_old_data(days_to_keep=30)
print(f"Deleted {deleted} old records")
```

## Configuration

Edit `config/spy_qqq_config.yaml`:

```yaml
realtime_data:
  enabled: true
  collect_interval_seconds: 1.0   # How often to collect
  buffer_size: 100                # Batch size for database
  retention_days: 30              # How long to keep data
```

### Collect Less Frequently
```yaml
collect_interval_seconds: 5.0  # Collect every 5 seconds instead
```

### Keep More History
```yaml
retention_days: 90  # Keep 3 months of data
```

## Troubleshooting

### "No data collected"
- **Market must be open** (9:30-16:00 ET, Mon-Fri)
- Check logs: `tail -f logs/trading_agent.log`
- Run test: `python scripts/test_realtime_collection.py`

### "Collection errors increasing"
- Check Alpaca API status
- Verify API keys are valid
- Check internet connection

### "Database getting large"
```bash
# Clean old data
python -c "from src.market_data.tick_analyzer import TickDataAnalyzer; TickDataAnalyzer().clean_old_data(7)"
```

## Benefits for Trading

1. **Better Timing**: Identify optimal entry/exit times
2. **Spread Analysis**: Know typical bid-ask spreads
3. **Volatility Patterns**: Trade during favorable conditions
4. **Backtesting**: Use real tick data for accurate backtests
5. **Pattern Recognition**: Find recurring price patterns
6. **Risk Management**: Better understand intraday risk

## Advanced Usage

See full documentation: `docs/REALTIME_DATA_COLLECTION.md`

### Get Minute Bars
```python
bars = analyzer.get_minute_bars('SPY', start_time, end_time)
# Returns OHLCV bars aggregated from tick data
```

### Daily Statistics
```python
stats = analyzer.get_daily_statistics('SPY', datetime.now())
print(f"Today's range: {stats['range_pct']:.2f}%")
print(f"Volatility: {stats['volatility']:.3f}%")
```

### Export for Analysis
```python
analyzer.export_to_csv('SPY', start, end, 'spy_data.csv')
# Opens in Excel, Python pandas, etc.
```

## Summary

✅ **Automatic**: Starts with the agent, no manual intervention  
✅ **Efficient**: Batched inserts, minimal overhead  
✅ **Comprehensive**: Every data point captured  
✅ **Analyzable**: Powerful query and analysis tools  
✅ **Exportable**: CSV export for external tools  

Start collecting data now. The more data you have, the better your strategies will become! 🚀

## Next Steps

1. ✅ Run migration: `python scripts/migrate_add_tick_data.py`
2. ✅ Start agent: `python main.py`
3. ⏰ Wait 1-2 hours during market hours
4. 📊 View data: `python scripts/view_tick_data.py`
5. 🎯 Use data to optimize your strategies

Questions? Check `docs/REALTIME_DATA_COLLECTION.md` for detailed documentation.

