# ✅ Telegram Bot Setup - Complete Guide

## 🎯 What You Get

### **1. On-Demand Reports** 📱
Send these commands in Telegram anytime:
- `/report` - Full daily report right now
- `/summary` - Same as /report
- `/help` - Show commands

### **2. Automatic Daily Reports** ⏰
- Sent at 4:00 PM CT every trading day (Mon-Fri)
- Automatic via cron job
- No manual action needed

---

## 🚀 Complete Setup Steps

### **Step 1: Start the Telegram Bot (Server)**

SSH to your server and run:

```bash
cd /root/Automated-Options-Trading-Agent
./start_telegram_bot.sh
```

This starts the bot that listens for your commands.

### **Step 2: Test On-Demand Commands**

Open Telegram and send to your bot:

```
/start
```

You should get a welcome message!

Then try:
```
/report
```

You'll get a full daily report immediately!

### **Step 3: Verify Automatic Reports**

Your cron job is already set up to run at 4 PM CT daily.

To verify it's scheduled:
```bash
crontab -l
```

You should see:
```
0 16 * * 1-5 cd /root/Automated-Options-Trading-Agent && python3 send_daily_report.py >> logs/daily_report.log 2>&1
```

### **Step 4: Test Automatic Report (Optional)**

Don't wait until 4 PM! Test now:
```bash
python3 send_daily_report.py
```

You'll receive a report in Telegram immediately.

---

## 📱 How to Use

### **Get Report Anytime:**

Open Telegram, message your bot:
- `/report` - Get current daily report
- `/summary` - Same thing (shortcut)

### **Automatic Reports:**

You'll automatically receive reports at 4:00 PM CT every weekday:
- **Monday-Friday**: Automatic report at 4 PM
- **Weekends**: No reports (market closed)

---

## 🔧 Commands Reference

| Command | What It Does |
|---------|-------------|
| `/start` | Welcome message and command list |
| `/report` | Generate full daily report NOW |
| `/summary` | Same as /report (shorter to type) |
| `/help` | Show available commands |

---

## 📊 What's in the Report?

Every report includes:

✅ **Data Collection**
- Ticks collected today
- Data quality metrics
- Collection duration

✅ **Trading Activity**
- Trades opened (with reasons)
- Trades closed (with P&L)
- Why no trades (if none)

✅ **Current Positions**
- All open positions
- Unrealized P&L
- Days to expiration

✅ **Market Analysis**
- Today's conditions
- Volatility assessment
- Trend direction

✅ **Tomorrow's Outlook**
- What to expect
- Position plans
- Trading strategy

✅ **Weekly Performance**
- Last 7 days stats
- Win rate
- Total P&L

---

## 🔄 System Status

### **What's Running:**

1. **Data Collector** (start_simple.py)
   - Collecting market data
   - Stores in database
   - Runs continuously

2. **Telegram Bot** (telegram_report_bot.py)
   - Listens for your commands
   - Responds with reports
   - Runs continuously

3. **Cron Job** (automatic)
   - Sends daily report at 4 PM
   - Runs Monday-Friday
   - No manual action needed

### **Check Status:**

```bash
# Check data collector
ps aux | grep start_simple.py | grep -v grep

# Check Telegram bot
ps aux | grep telegram_report_bot.py | grep -v grep

# Check cron job
crontab -l

# View recent logs
tail -f logs/telegram_bot.log
tail -f logs/daily_report.log
```

---

## 🐛 Troubleshooting

### **Bot doesn't respond to commands:**

1. Check bot is running:
   ```bash
   ps aux | grep telegram_report_bot.py | grep -v grep
   ```

2. If not running, start it:
   ```bash
   ./start_telegram_bot.sh
   ```

3. Check logs for errors:
   ```bash
   tail -50 logs/telegram_bot.log
   ```

### **Automatic report not received at 4 PM:**

1. Check cron job exists:
   ```bash
   crontab -l
   ```

2. Check cron service is running:
   ```bash
   systemctl status cron
   ```

3. Check report logs:
   ```bash
   tail -50 logs/daily_report.log
   ```

4. Test manually:
   ```bash
   python3 send_daily_report.py
   ```

### **"Telegram credentials not found" error:**

1. Check .env file exists:
   ```bash
   cat .env | grep TELEGRAM
   ```

2. Should show:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

3. If missing, add them:
   ```bash
   nano .env
   ```

---

## 🎯 Quick Commands Summary

### **On Server:**

```bash
# Start Telegram bot (responds to commands)
./start_telegram_bot.sh

# Check if bot running
ps aux | grep telegram_report_bot | grep -v grep

# Stop bot
pkill -f telegram_report_bot.py

# Restart bot
pkill -f telegram_report_bot.py && ./start_telegram_bot.sh

# Send manual report
python3 send_daily_report.py

# View logs
tail -f logs/telegram_bot.log
tail -f logs/daily_report.log
```

### **In Telegram:**

```
/start    - Welcome
/report   - Get report now
/summary  - Get report now (shortcut)
/help     - Show commands
```

---

## 🔒 Security Notes

- Your bot token and chat ID are private
- Only you can access the bot (chat ID restricted)
- Keep .env file secure
- Don't share bot token publicly

---

## ✅ Setup Checklist

- [x] Cron job scheduled (4 PM CT daily)
- [ ] Telegram bot running (`./start_telegram_bot.sh`)
- [ ] Tested `/report` command in Telegram
- [ ] Received test report successfully
- [ ] Verified bot shows in `ps aux` command
- [ ] Checked logs for any errors

---

## 🚀 You're All Set!

**Your system now:**
1. ✅ Collects market data automatically
2. ✅ Sends daily reports at 4 PM CT
3. ✅ Responds to your Telegram commands anytime
4. ✅ Runs everything in the background

**To use:**
- Open Telegram anytime and send `/report`
- Or just wait for the automatic 4 PM report!

**Need help?** Check logs or test manually with:
```bash
python3 send_daily_report.py
```

---

**Enjoy your automated trading reports!** 📊🚀

