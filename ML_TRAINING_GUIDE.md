# 🤖 ML Training Guide

## Overview

This guide will show you how to train your ML models on historical data to make **smarter, data-driven trading decisions**.

Your system uses **3 main ML models**:
1. **Entry Signal Model** - Predicts if you should enter a trade
2. **Win Probability Model** - Estimates the probability of a winning trade
3. **Volatility Forecaster** - Predicts future market volatility

---

## 🚀 Quick Start (5 minutes)

### Step 1: Run the Training Script

```bash
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent

# Activate your virtual environment (if you have one)
source venv/bin/activate

# Run training
python scripts/train_ml_models.py
```

**What this does:**
- Downloads 1 year of historical data for SPY & QQQ
- Calculates 50+ technical indicators
- Creates labels (what was a good trade vs bad trade)
- Trains 3 ML models
- Saves models to `./models/` directory

**Expected time:** 10-30 minutes (depending on internet speed)

### Step 2: Check Results

After training completes, you'll see:

```
✅ TRAINING COMPLETED SUCCESSFULLY!

📈 Model Performance Summary:

  Entry Signal Model:
    Accuracy: 65.32%
    Precision: 68.45%
    Recall: 62.18%
    F1 Score: 65.19%
    AUC: 0.712

  Win Probability Model:
    R² Score: 0.543
    MAE: 0.0823

  Volatility Forecaster:
    Accuracy: 71.24%
    MAE: 0.0092
```

**What the numbers mean:**
- **Accuracy > 60%** = Good! Better than random guessing
- **Accuracy > 70%** = Excellent! Very predictive
- **F1 Score** = Balance of precision and recall (higher is better)
- **AUC > 0.7** = Strong predictive power

### Step 3: Models Are Automatically Used

Once trained, your models are **automatically loaded** by the trading system!

No code changes needed - just resume trading:
```bash
# In Telegram bot
/resume
```

---

## 📊 Understanding the Training Process

### 1. Data Collection

The system collects historical data:
- **Timeframe**: 5-minute bars (configurable)
- **Lookback**: 365 days (1 year)
- **Symbols**: SPY, QQQ (can add more)

### 2. Feature Engineering

Creates 50+ features from raw data:
- **Price features**: Returns, momentum, volatility
- **Technical indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Regime features**: Trend strength, market regime
- **IV features**: IV Rank, IV Percentile (if available)
- **Time features**: Hour, day of week, market session

### 3. Label Creation

Labels tell the model what was a "good" trade:

**For Bull Put Spreads:**
- ✅ **Good trade** = Price went up or stayed flat (profit)
- ❌ **Bad trade** = Price dropped significantly (loss)

**Labels are created by looking forward in time:**
```python
# If price increased by 2%+ in next 20 periods → Label = 1 (good)
# If price decreased significantly → Label = 0 (bad)
```

### 4. Training & Validation

- **Train set**: 70% of data (oldest)
- **Validation set**: 10% of data (middle)
- **Test set**: 20% of data (most recent)

**Important:** Time-based split, not random! This prevents "looking into the future".

---

## ⚙️ Customization Options

### Change Symbols

Edit `scripts/train_ml_models.py`:

```python
# Train on more symbols
symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF']
```

### Collect More Data

```python
# Train on 2 years instead of 1
lookback_days = 730  # Was 365
```

### Change Strategy Focus

```python
# Optimize for different strategy
strategy = 'iron_condor'  # Was 'bull_put_spread'
```

### Change Timeframe

Edit `src/ml/data_collector.py`:

```python
# Use 15-minute bars instead of 5-minute
timeframe="15Min"
```

---

## 📈 Improving Model Performance

### If Accuracy < 60%

**Problem:** Model isn't learning well

**Solutions:**
1. **Collect more data**: Increase `lookback_days` to 730 or more
2. **Add more symbols**: Train on SPY, QQQ, IWM, DIA together
3. **Use longer timeframes**: Switch to 15Min or 1Hour bars
4. **Check data quality**: Make sure API is returning good data

### If Accuracy 60-70%

**Good!** Model is working. To improve further:
1. Add more features (VIX, put/call ratio, market breadth)
2. Tune hyperparameters (n_estimators, max_depth)
3. Try different models (XGBoost, LightGBM)
4. Ensemble multiple models

### If Accuracy > 70%

**Excellent!** Be careful of overfitting:
1. Validate on out-of-sample data
2. Monitor live performance vs backtest
3. Retrain regularly with new data

---

## 🔄 Retraining Schedule

**Best practice:** Retrain models regularly to adapt to changing markets.

### Weekly Retraining (Recommended)

```bash
# Every Sunday before market opens
python scripts/train_ml_models.py
```

**Why weekly?**
- Market conditions change
- New patterns emerge
- Models stay current

### Monthly Retraining (Minimum)

If weekly is too much, at least retrain monthly.

### Automated Retraining (Advanced)

Create a cron job:

```bash
# Retrain every Sunday at 2 AM
0 2 * * 0 cd /path/to/project && python scripts/train_ml_models.py
```

---

## 📊 Monitoring Model Performance

### Check Feature Importance

After training, see which features matter most:

```python
# In the log output
Top 10 Features:
  rsi: 0.0823
  macd_hist: 0.0712
  bb_position: 0.0698
  momentum_10: 0.0654
  ...
```

**Use this to:**
- Understand what drives predictions
- Add similar features
- Remove low-importance features

### Live Performance Monitoring

Track model accuracy in real trades:

```bash
# Check trade journal
cat logs/trade_journal.jsonl | jq '.ml_confidence'
```

**Compare:**
- Backtest accuracy vs live accuracy
- Win rate on high-confidence (>70%) predictions
- Accuracy by market regime

---

## 🐛 Troubleshooting

### "No data collected" Error

**Cause:** Alpaca API not returning data

**Fix:**
1. Check API keys in `.env`
2. Verify internet connection
3. Check if markets are open (or use `end_date=datetime.now()`)
4. Try smaller `lookback_days` (e.g., 180)

### "Insufficient data for training" Error

**Cause:** Not enough samples after cleaning

**Fix:**
1. Increase `lookback_days`
2. Use longer timeframes (15Min, 1Hour)
3. Add more symbols

### Models Not Loading in Live Trading

**Cause:** Models directory not found

**Fix:**
```bash
# Make sure models directory exists
ls -la models/

# Should see:
# entry_signal_latest.pkl
# win_probability_latest.pkl
# volatility_latest.pkl
```

### Low Accuracy (<55%)

**Cause:** Market is random in short timeframes, or poor features

**Fix:**
1. Use longer timeframes (1Hour, 1Day)
2. Train on more data (2+ years)
3. Add more features (external data)
4. Check if labels are correct

---

## 📚 Advanced Topics

### Adding Custom Features

Edit `src/ml/data_collector.py`:

```python
def _calculate_indicators(self, df):
    # ... existing code ...
    
    # Add VIX correlation
    df['vix_correlation'] = self._fetch_vix_correlation()
    
    # Add put/call ratio
    df['put_call_ratio'] = self._fetch_put_call_ratio()
    
    return df
```

### Using Different Models

Edit `src/ml/model_trainer.py`:

```python
# Try XGBoost instead of Random Forest
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1
)
```

### Ensemble Multiple Models

Train multiple models and average predictions:

```python
model1 = RandomForestClassifier(...)
model2 = GradientBoostingClassifier(...)
model3 = XGBClassifier(...)

# Average predictions
final_pred = (model1.predict_proba() + 
              model2.predict_proba() + 
              model3.predict_proba()) / 3
```

---

## 🎯 Best Practices

### ✅ Do's

1. **Train regularly** - Weekly or monthly
2. **Use time-based splits** - Never shuffle time series data
3. **Monitor overfitting** - Validate on out-of-sample data
4. **Keep it simple** - Start with basic models, add complexity gradually
5. **Track performance** - Log predictions vs outcomes
6. **Use sufficient data** - At least 6-12 months

### ❌ Don'ts

1. **Don't overtrain** - More data ≠ always better (diminishing returns)
2. **Don't ignore regime changes** - Retrain after major market shifts
3. **Don't trust 100% accuracy** - It's overfitting!
4. **Don't use future data** - Leads to unrealistic results
5. **Don't ignore live performance** - Backtest ≠ live results

---

## 📞 Getting Help

### Model isn't training?
```bash
# Check logs
tail -f logs/trading_agent.log
```

### Want to see training in detail?
```bash
# Run with verbose logging
python scripts/train_ml_models.py 2>&1 | tee training.log
```

### Questions?
- Check `/logs/` directory for detailed errors
- Review `models/training_summary.json` for history
- Read sklearn documentation for model details

---

## 🎉 Next Steps

1. **Train your first model**: Run `python scripts/train_ml_models.py`
2. **Review performance**: Check accuracy metrics
3. **Resume trading**: Let the bot use trained models
4. **Monitor results**: Track win rate with ML vs without
5. **Iterate**: Retrain with more data, tune parameters
6. **Automate**: Set up weekly retraining

---

**Your ML models will get smarter over time as they learn from more data!** 🚀

**Pro Tip:** The first training might take 20-30 minutes. Future trainings are faster because data is cached.

