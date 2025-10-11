# 🚀 Deploy ML System to DigitalOcean

## Quick Deploy (One Command)

```bash
chmod +x DEPLOY_TO_DIGITALOCEAN.sh
./DEPLOY_TO_DIGITALOCEAN.sh
```

This will automatically:
1. ✅ Commit and push to GitHub
2. ✅ Pull on server
3. ✅ Install dependencies
4. ✅ Add Polygon key
5. ✅ Train models
6. ✅ Restart services
7. ✅ Test everything

**Time:** 10-15 minutes

---

## Manual Deploy (Step-by-Step)

### Step 1: Commit Local Changes

```bash
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent

git add -A
git commit -m "Add ML training with Polygon - 65.66% accuracy"
git push origin main
```

### Step 2: Connect to Server

```bash
ssh root@45.55.150.19
```

### Step 3: Pull Latest Code

```bash
cd /opt/trading-agent
git pull origin main
```

### Step 4: Install Dependencies

```bash
source venv/bin/activate
pip install polygon-api-client python-dotenv joblib
```

### Step 5: Add Polygon API Key

```bash
echo "POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY" >> .env
```

Or edit `.env` manually:
```bash
nano .env
# Add: POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY
```

### Step 6: Train Models on Server

```bash
python scripts/train_ml_models.py
```

**This will take 5-10 minutes.** You'll see:
```
✅ Collected 1,002 samples
✅ Entry Model Accuracy: 65.66%
✅ TRAINING COMPLETED SUCCESSFULLY!
```

### Step 7: Test Models

```bash
python scripts/test_ml_models.py
```

Should show:
```
✅ Models loaded successfully!
✅ ALL TESTS PASSED!
```

### Step 8: Restart Telegram Bot

```bash
pkill -f start_telegram_bot
nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &
```

### Step 9: Verify Services Running

```bash
ps aux | grep -E "telegram_bot|python" | grep -v grep
```

Should see telegram bot process running.

### Step 10: Exit Server

```bash
exit
```

---

## ✅ Test Deployment

### In Telegram, send:

1. **Check ML models:**
   ```
   /ml
   ```
   
   Should show:
   ```
   🤖 ML Models Status
   ✅ Status: LOADED
   📊 Models: 3
   ```

2. **Check status:**
   ```
   /status
   ```

3. **Resume trading:**
   ```
   /resume
   ```

---

## 📊 Monitor Logs

```bash
ssh root@45.55.150.19

# Training logs
tail -f /opt/trading-agent/logs/ml_training.log

# Telegram bot
tail -f /opt/trading-agent/logs/telegram_bot.log

# Trading agent
tail -f /opt/trading-agent/logs/trading_agent.log
```

---

## 🐛 Troubleshooting

### Models not loading

```bash
ssh root@45.55.150.19
cd /opt/trading-agent
ls -la models/

# Should see:
# entry_signal_latest.pkl
# win_probability_latest.pkl
# volatility_latest.pkl
```

### Telegram bot not responding

```bash
ssh root@45.55.150.19
cd /opt/trading-agent

# Check if running
ps aux | grep telegram_bot

# Restart
pkill -f start_telegram_bot
nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &

# Check logs
tail -20 logs/telegram_bot.log
```

### Training failed

```bash
# Check Polygon key
cat .env | grep POLYGON

# Try training with verbose output
python scripts/train_ml_models.py 2>&1 | tee training_output.log
```

---

## 🎯 Success Checklist

After deployment, verify:

- [ ] Code pushed to GitHub
- [ ] Server pulled latest code
- [ ] Dependencies installed
- [ ] Polygon key in .env
- [ ] Models trained successfully (65.66% accuracy)
- [ ] Models exist in models/ directory
- [ ] Test script passes
- [ ] Telegram bot restarted
- [ ] `/ml` command works in Telegram
- [ ] Models show as LOADED
- [ ] Trading can be resumed

---

## 📈 Expected Results

### Telegram `/ml` Output:

```
🤖 ML Models Status

✅ Status: LOADED
📊 Models: 3

🟢 Entry Signal
   Size: 0.3 MB
   Updated: 2025-10-10 (0d ago)

🟢 Win Probability  
   Size: 0.3 MB
   Updated: 2025-10-10 (0d ago)

🟢 Volatility
   Size: 0.9 MB
   Updated: 2025-10-10 (0d ago)
```

---

## 🎉 You're Done!

Your ML system is now:
- ✅ Deployed to production
- ✅ Using real Polygon data
- ✅ 65.66% prediction accuracy
- ✅ Monitored via Telegram
- ✅ Ready to trade

**Recommendation:** Monitor for 1 week, then retrain models weekly to keep them fresh!

