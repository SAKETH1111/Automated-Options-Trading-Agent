

# 📊 Daily Telegram Report Setup Guide

Get a comprehensive daily trading report sent to your Telegram every day at 4:00 PM CT!

---

## 📱 What You'll Receive Daily

### **Comprehensive Daily Report Includes:**

✅ **Data Collection Stats**
- How many ticks collected today
- Data quality metrics
- Collection duration

✅ **Trading Activity**
- Trades opened (with reasons)
- Trades closed (with P&L)
- Why no trades (if applicable)

✅ **Current Positions**
- All open positions
- Unrealized P&L
- Days to expiration

✅ **Market Analysis**
- Today's market conditions
- Volatility assessment
- Trend analysis

✅ **Tomorrow's Outlook**
- What to expect tomorrow
- Position management plans
- Trading strategy

✅ **Weekly Performance**
- Last 7 days stats
- Win rate
- Total P&L

---

## 🚀 Quick Setup (10 minutes)

### **Step 1: Create Telegram Bot** (5 min)

1. Open Telegram on your phone
2. Search for `@BotFather`
3. Send: `/newbot`
4. Choose a name: `My Trading Report Bot`
5. Choose username: `my_trading_report_bot` (must end with 'bot')
6. **Copy the token** you receive (looks like: `123456789:ABCdefGHIjkl...`)

### **Step 2: Get Your Chat ID** (2 min)

1. Start a chat with your new bot (click the link BotFather gave you)
2. Send any message to your bot (like "hello")
3. Go to this URL in your browser (replace YOUR_TOKEN):
   ```
   https://api.telegram.org/botYOUR_TOKEN/getUpdates
   ```
4. Look for `"chat":{"id":123456789}`
5. **Copy that number** (your chat ID)

### **Step 3: Configure on Server** (3 min)

SSH into your server and run:

```bash
cd /root/Automated-Options-Trading-Agent

# Create or edit .env file
nano .env
```

Add these lines (replace with your actual values):
```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

Save and exit (Ctrl+X, Y, Enter)

### **Step 4: Install Dependencies**

```bash
pip3 install python-telegram-bot python-dotenv
```

### **Step 5: Test the Report**

```bash
python3 send_daily_report.py
```

You should receive a report in Telegram within seconds!

---

## ⏰ Schedule Daily Reports (Automatic)

### **Option A: Using Cron (Recommended)**

Edit crontab:
```bash
crontab -e
```

Add this line (sends report at 4:00 PM CT daily):
```bash
0 16 * * 1-5 cd /root/Automated-Options-Trading-Agent && /usr/bin/python3 send_daily_report.py >> logs/daily_report.log 2>&1
```

Save and exit.

**Note:** This runs Monday-Friday at 4:00 PM CT (after market close)

### **Option B: Manual Testing**

Run anytime to test:
```bash
cd /root/Automated-Options-Trading-Agent
python3 send_daily_report.py
```

---

## 📊 Example Daily Report

```
══════════════════════════════════════════════════
📊 DAILY TRADING REPORT
📅 Monday, October 14, 2025
⏰ 04:00 PM CDT
══════════════════════════════════════════════════

📈 DATA COLLECTION
──────────────────────────────────────────────────
✅ Market data collected successfully
   Total ticks: 12,450
   Duration: 6.5 hours

   By Symbol:
   • EWZ: 3,112 ticks
   • GDX: 3,089 ticks
   • TLT: 3,125 ticks
   • XLF: 3,124 ticks

💼 TRADING ACTIVITY
──────────────────────────────────────────────────
📊 No Trades Opened Today

   Why no trades?
   • No high-quality signals met entry criteria
   • Risk limits may have been reached
   • Waiting for better market conditions

📋 OPEN POSITIONS
──────────────────────────────────────────────────
No open positions

🔍 MARKET ANALYSIS
──────────────────────────────────────────────────
✅ Good data quality - active market
   Collected 12,450 ticks
📊 Volatility: Normal
📈 Trend: Monitoring

🔮 TOMORROW'S OUTLOOK
──────────────────────────────────────────────────
🔔 Market opens: 8:30 AM CT
   Data collection will resume automatically

🎯 Looking for new trading opportunities

Strategy:
• Continue data collection
• Monitor for quality signals
• Manage existing positions
• Follow PDT compliance rules

📊 WEEKLY PERFORMANCE (Last 7 Days)
──────────────────────────────────────────────────
Total Trades: 0

══════════════════════════════════════════════════
🤖 Automated Trading Agent
Next report: Tomorrow at 4:00 PM CT
══════════════════════════════════════════════════
```

---

## 🎯 What Makes This Report Valuable?

### **1. Complete Transparency**
- See exactly what happened each day
- Understand every decision made
- Track performance over time

### **2. Actionable Insights**
- Know why trades were/weren't made
- Get tomorrow's outlook
- See position status at a glance

### **3. No Manual Checking**
- Automatic daily delivery
- Consistent timing (4 PM CT)
- Always up-to-date

### **4. Mobile-Friendly**
- Receive on your phone
- Read anywhere, anytime
- No need to log into server

---

## 🔧 Customization Options

### Change Report Time

Edit the cron schedule. For example, 3:30 PM CT:
```bash
30 15 * * 1-5 cd /root/Automated-Options-Trading-Agent && /usr/bin/python3 send_daily_report.py
```

### Add More Symbols to Track

Edit in `start_simple.py` or the orchestrator to add more symbols to collect.

### Adjust Report Format

The report generator is in: `src/alerts/daily_report.py`

You can customize:
- Which sections to include
- Metrics to show
- Analysis depth

---

## 📋 Troubleshooting

### **Report Not Received**

1. Check bot token is correct:
   ```bash
   cat .env | grep TELEGRAM_BOT_TOKEN
   ```

2. Check chat ID is correct:
   ```bash
   cat .env | grep TELEGRAM_CHAT_ID
   ```

3. Test manually:
   ```bash
   python3 send_daily_report.py
   ```

4. Check logs:
   ```bash
   tail -50 logs/daily_report.log
   ```

### **"Bot not found" Error**

- Verify token is exact copy from BotFather
- No extra spaces or characters
- Token should be on one line

### **Message Too Long**

The report automatically splits long messages. If issues:
- Check Telegram bot API status
- Try sending manual test report
- Check logs for errors

---

## 🎓 Advanced: Multiple Reports

### Morning Pre-Market Report

Add another cron job for 8:00 AM CT:
```bash
0 8 * * 1-5 cd /root/Automated-Options-Trading-Agent && /usr/bin/python3 send_morning_brief.py
```

### Real-Time Alerts

The system can also send:
- Trade execution alerts
- Position updates
- Risk warnings
- System errors

These are separate from the daily report.

---

## ✅ Setup Checklist

- [ ] Created Telegram bot with @BotFather
- [ ] Got bot token
- [ ] Got chat ID
- [ ] Added to .env file on server
- [ ] Installed python-telegram-bot
- [ ] Tested with `python3 send_daily_report.py`
- [ ] Received test report in Telegram
- [ ] Added cron job for daily reports
- [ ] Verified cron is working

---

## 🚀 Next Steps After Setup

1. **Tomorrow:** Check that you receive the 4 PM report
2. **Review:** Read through the daily insights
3. **Track:** Keep reports to see progress over time
4. **Adjust:** Modify report timing/content as needed
5. **Expand:** Add more alerts or custom reports

---

## 💡 Pro Tips

1. **Pin the daily report** in Telegram for quick access
2. **Create a Telegram group** if sharing with partners
3. **Archive old reports** to track historical performance
4. **Set up alerts** for notification when report arrives
5. **Review reports weekly** to spot patterns

---

## 📞 Support

If you need help:
1. Check logs: `tail -f logs/daily_report.log`
2. Test manually: `python3 send_daily_report.py`
3. Verify credentials in `.env`
4. Check Telegram bot is active with @BotFather

---

**Ready to receive your first daily report?** 📊

Just follow the 5 steps above, and you'll get comprehensive trading insights delivered to your phone every day!

