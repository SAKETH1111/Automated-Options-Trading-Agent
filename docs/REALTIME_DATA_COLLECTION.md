# Real-Time Data Collection System

## Overview

The trading agent now includes a comprehensive real-time data collection system that monitors indexes (SPY, QQQ) **every second** and stores the data in a database for future analysis and learning.

## Features

### 🔄 Real-Time Collection
- **Frequency**: Collects data every 1 second (configurable)
- **Symbols**: SPY, QQQ (configurable via watchlist)
- **Data Points**: Price, bid/ask, spread, volume, VIX, and more
- **Market Hours**: Automatically starts/stops based on market hours
- **Threading**: Runs in a separate thread, non-blocking

### 📊 Data Stored

Each tick includes:
- **Price Data**: Current price, bid, ask, spread, spread %
- **Volume**: Cumulative daily volume, last trade size
- **Market Context**: VIX level (for SPY/QQQ)
- **Technical Indicators**: 5-second SMA, 60-second SMA, RSI
- **Momentum**: Price change, price change %
- **Market State**: open, pre_market, after_hours, closed

### 💾 Database Schema

```sql
Table: index_tick_data
- tick_id (Primary Key)
- symbol (indexed)
- timestamp (indexed)
- price, bid, ask, spread, spread_pct
- bid_size, ask_size
- volume, last_trade_size
- vix
- sma_5, sma_60, rsi
- price_change, price_change_pct
- market_state
- created_at
```

### 📈 Analysis Capabilities

The `TickDataAnalyzer` provides powerful analysis tools:
1. **DataFrame Conversion**: Convert tick data to pandas DataFrame
2. **Minute Bars**: Aggregate ticks into OHLCV minute bars
3. **Volatility Profiles**: Intraday volatility by hour
4. **Price Reversals**: Find significant price movements
5. **VIX Correlation**: Calculate correlation with VIX
6. **Daily Statistics**: Comprehensive daily metrics
7. **Data Export**: Export to CSV for external analysis
8. **Data Cleanup**: Remove old data to save space

## Setup

### 1. Run Database Migration

First, add the new table to your database:

```bash
python scripts/migrate_add_tick_data.py
```

This creates the `index_tick_data` table.

### 2. Configure Collection

Edit `config/spy_qqq_config.yaml`:

```yaml
realtime_data:
  enabled: true
  collect_interval_seconds: 1.0  # Collect every second
  buffer_size: 100  # Batch insert every 100 ticks
  retention_days: 30  # Keep 30 days of data
```

### 3. Start the Agent

The real-time collector starts automatically when you start the trading agent:

```bash
python main.py
```

You'll see logs like:
```
RealTimeDataCollector initialized for ['SPY', 'QQQ']
✅ Real-time data collection started
Collected tick: SPY @ $450.25 (Δ +0.015%)
```

## Usage

### View Collected Data

Use the interactive viewer script:

```bash
python scripts/view_tick_data.py
```

This provides:
- Data availability overview
- Daily statistics
- Interactive menu with options:
  1. View recent ticks
  2. View minute bars
  3. Export to CSV
  4. Find price reversals
  5. Calculate VIX correlation

### Programmatic Access

```python
from src.market_data.realtime_collector import RealTimeDataCollector
from src.market_data.tick_analyzer import TickDataAnalyzer

# Get recent ticks
collector = RealTimeDataCollector()
recent_ticks = collector.get_recent_ticks('SPY', limit=100)

# Analyze data
analyzer = TickDataAnalyzer()

# Get as DataFrame
from datetime import datetime, timedelta
end_time = datetime.now()
start_time = end_time - timedelta(hours=1)
df = analyzer.get_tick_data_df('SPY', start_time, end_time)

# Get minute bars
bars = analyzer.get_minute_bars('SPY', start_time, end_time)

# Daily statistics
stats = analyzer.get_daily_statistics('SPY', datetime.now())
print(f"Range: {stats['range_pct']:.2f}%")
print(f"Volatility: {stats['volatility']:.3f}%")

# Find reversals
reversals = analyzer.find_price_reversals(
    'SPY', start_time, end_time, threshold_pct=0.5
)

# VIX correlation
correlation = analyzer.get_correlation_with_vix('SPY', start_time, end_time)
```

## Data Management

### Storage Considerations

At 1 tick/second during market hours (6.5 hours):
- **Per Day**: ~23,400 ticks per symbol
- **Per Symbol Per Month**: ~468,000 ticks (~50 MB)
- **Both SPY & QQQ Per Month**: ~936,000 ticks (~100 MB)

### Cleanup Old Data

Automatically clean old data:

```python
from src.market_data.tick_analyzer import TickDataAnalyzer

analyzer = TickDataAnalyzer()
deleted = analyzer.clean_old_data(days_to_keep=30)
print(f"Deleted {deleted} old records")
```

Or add to orchestrator as a scheduled task:

```python
# In orchestrator._schedule_tasks()
self.scheduler.add_job(
    self._cleanup_old_tick_data,
    'cron',
    day_of_week='sun',
    hour='22',
    minute='0',
    timezone=self.market_timezone,
    id='cleanup_tick_data'
)
```

## Use Cases

### 1. Pattern Recognition
Identify intraday patterns for entry/exit timing:
```python
# Analyze hourly volatility
volatility_profile = analyzer.calculate_volatility_profile('SPY', datetime.now())
# Use to determine best trading hours
```

### 2. Spread Analysis
Monitor bid-ask spreads for optimal execution:
```python
df = analyzer.get_tick_data_df('SPY', start_time, end_time)
avg_spread = df['spread_pct'].mean()
# Adjust limit orders based on typical spreads
```

### 3. VIX Relationship
Understand SPY/QQQ behavior relative to VIX:
```python
correlation = analyzer.get_correlation_with_vix('SPY', start_time, end_time)
# Adjust strategy based on correlation strength
```

### 4. Backtesting Enhancement
Use tick data for more accurate backtest fills:
```python
bars = analyzer.get_minute_bars('SPY', trade_date, trade_date)
# Use actual intraday prices for backtest execution
```

### 5. Market Microstructure
Study order flow and price dynamics:
```python
reversals = analyzer.find_price_reversals('SPY', start_time, end_time, 0.3)
# Identify support/resistance levels
```

### 6. Strategy Optimization
Optimize entry/exit timing using actual data:
```python
stats = analyzer.get_daily_statistics('SPY', date)
# Use volatility to adjust position sizing
```

## Performance

### Optimization Features
- **Batched Inserts**: Buffers 100 ticks before database write
- **Threaded Collection**: Non-blocking, runs in background
- **Indexed Queries**: Symbol and timestamp are indexed
- **Connection Pooling**: Efficient database connections

### Monitoring
Check collection status:
```python
from src.orchestrator import TradingOrchestrator

orchestrator = TradingOrchestrator()
status = orchestrator.get_status()
print(status['realtime_data'])
```

Output:
```python
{
    'is_running': True,
    'symbols': ['SPY', 'QQQ'],
    'total_ticks_collected': 15234,
    'total_ticks_stored': 15200,
    'collection_errors': 0,
    'buffer_size': 34,
    'last_collection_time': '2024-01-15T14:32:45'
}
```

## Troubleshooting

### No Data Being Collected

1. **Check if market is open**:
   ```python
   from src.orchestrator import TradingOrchestrator
   orch = TradingOrchestrator()
   print(orch.is_market_open())  # Should be True during market hours
   ```

2. **Check collector status**:
   ```python
   stats = orch.realtime_collector.get_stats()
   print(stats)
   ```

3. **Check logs**:
   ```bash
   tail -f logs/trading_agent.log | grep "RealTimeDataCollector"
   ```

### High Database Usage

1. **Reduce retention period**:
   ```yaml
   realtime_data:
     retention_days: 7  # Instead of 30
   ```

2. **Increase collection interval**:
   ```yaml
   realtime_data:
     collect_interval_seconds: 5.0  # Collect every 5 seconds
   ```

3. **Run cleanup regularly**:
   ```bash
   python scripts/view_tick_data.py
   # Then manually clean old data
   ```

### Collection Errors

Check the error count:
```python
stats = collector.get_stats()
if stats['collection_errors'] > 100:
    # Investigation needed
    # Check Alpaca API limits
    # Check network connectivity
```

## Future Enhancements

Potential additions:
- [ ] Order book data (Level 2)
- [ ] Trade size distribution analysis
- [ ] Real-time anomaly detection
- [ ] Automatic data compression for old data
- [ ] Multiple timeframe aggregation
- [ ] Real-time charting/visualization
- [ ] Alert on unusual price movements
- [ ] Integration with machine learning models

## Summary

The real-time data collection system provides:
- ✅ Second-by-second index monitoring
- ✅ Comprehensive data storage
- ✅ Powerful analysis tools
- ✅ Easy programmatic access
- ✅ Automatic market hours handling
- ✅ Efficient storage and retrieval
- ✅ Export and visualization capabilities

This data becomes invaluable for:
- Improving strategy timing
- Understanding market microstructure
- Backtesting with real intraday data
- Learning optimal entry/exit points
- Monitoring spread costs
- Detecting patterns and anomalies

Start collecting data now, and you'll have a rich historical dataset for future analysis and optimization! 📊

