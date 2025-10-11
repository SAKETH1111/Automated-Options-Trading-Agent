# Final Fix for Advanced ML Training

## Issues Found:
1. ❌ `PolygonDataCollector` doesn't support week/month/year timeframes
2. ❌ `multi_timeframe_trainer` doesn't properly handle Polygon's data structure

## Fixes Applied:
1. ✅ Added support for 1Week, 1Month, 3Month, 6Month, 1Year to `polygon_data_collector.py`
2. ✅ Fixed `multi_timeframe_trainer.py` to properly call Polygon API

## Manual Steps to Deploy on Server:

### **On DigitalOcean Server:**

```bash
cd /opt/trading-agent

# Pull latest fixes from GitHub (need to commit from local first)
git pull origin main

# If git pull doesn't work, manually edit the file:
nano src/ml/polygon_data_collector.py

# Find line 143-149 and change to:
            timespan_map = {
                "1Min": ("minute", 1),
                "5Min": ("minute", 5),
                "15Min": ("minute", 15),
                "1Hour": ("hour", 1),
                "1Day": ("day", 1),
                "1Week": ("week", 1),
                "1Month": ("month", 1),
                "3Month": ("month", 3),
                "6Month": ("month", 6),
                "1Year": ("year", 1),
            }

# Save and exit (Ctrl+X, Y, Enter)

# Now train:
source venv/bin/activate
python3 scripts/train_advanced_ml.py --symbols SPY QQQ IWM DIA --components all
```

## Expected Training Time:
- **15-30 minutes** for all timeframes and symbols

## What Will Happen:
```
2025-10-11 15:35:00 | INFO | Collecting 1day_swing data...
2025-10-11 15:35:05 | INFO | Fetching 1Day data for 4 symbols...
2025-10-11 15:35:10 | INFO | Fetching data for SPY...
2025-10-11 15:35:15 | INFO | Fetched 500 bars from Polygon for SPY
2025-10-11 15:35:17 | INFO | Added options features for SPY
2025-10-11 15:35:20 | INFO | ✅ Collected 500 bars for SPY
...
2025-10-11 15:40:00 | INFO | 🎯 Training entry_signal model...
2025-10-11 15:42:00 | INFO | 📊 Model accuracy: 72.5%
```

## Files Changed:
- `src/ml/polygon_data_collector.py` - Added week/month/year timeframe support
- `src/ml/multi_timeframe_trainer.py` - Fixed to properly handle Polygon data structure
