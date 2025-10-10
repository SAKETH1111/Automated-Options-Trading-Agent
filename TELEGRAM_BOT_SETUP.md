# 🤖 Telegram Bot Setup Guide

## ✅ **Telegram Bot Code: COMPLETE**

Your Telegram bot is built and ready! Just need to configure it.

---

## 🚀 **Quick Setup (5 minutes)**

### **Step 1: Create Your Bot**

1. **Open Telegram** on your phone or computer

2. **Search for** `@BotFather` (official Telegram bot)

3. **Start chat** and send: `/newbot`

4. **Choose name**: `My Trading Agent` (or any name you like)

5. **Choose username**: `my_trading_agent_bot` (must end with 'bot')

6. **Copy the token**: You'll get something like:
   ```
   123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```
   **Save this token!**

### **Step 2: Get Your Chat ID**

1. **Start chat** with your new bot (click the link BotFather gives you)

2. **Send any message** to your bot (like "hello")

3. **Get your chat ID**:
   - Go to: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Replace `<YOUR_BOT_TOKEN>` with your actual token
   - Look for `"chat":{"id":123456789}`
   - **Save this chat ID!**

### **Step 3: Configure on Droplet**

```bash
ssh root@45.55.150.19 "cat >> /opt/trading-agent/.env << 'EOF'

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EOF"
```

Replace with your actual token and chat ID.

### **Step 4: Install Python Telegram Bot**

```bash
ssh root@45.55.150.19 "cd /opt/trading-agent && source venv/bin/activate && pip install python-telegram-bot"
```

### **Step 5: Start the Bot**

```bash
ssh root@45.55.150.19 "cd /opt/trading-agent && source venv/bin/activate && nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &"
```

### **Step 6: Test It!**

Open Telegram and send to your bot:
```
/start
/status
/positions
/pnl
```

---

## 🎯 **Available Commands:**

| Command | What It Does |
|---------|--------------|
| `/start` | Welcome message and command list |
| `/status` | Current agent status, equity, positions |
| `/positions` | Detailed view of open positions |
| `/pnl` | P&L (today, week, all-time) |
| `/risk` | Risk metrics and circuit breaker status |
| `/stop` | Stop automated trading |
| `/resume` | Resume trading |
| `/help` | Show command list |

---

## 📱 **What You'll Receive:**

### **Instant Notifications:**
- 🔔 Trade executed
- ✅ Profit target reached
- ⚠️ Stop loss triggered
- ⏰ Position expiring
- 🚨 Circuit breaker tripped
- ❌ System errors

### **On-Demand Info:**
- Check status anytime with `/status`
- View positions with `/positions`
- Check P&L with `/pnl`
- Monitor risk with `/risk`

### **Control:**
- Stop trading with `/stop`
- Resume with `/resume`
- Full control from your phone!

---

## 🎯 **Example Interaction:**

```
You: /status

Bot: 📊 Trading Agent Status

🤖 Trading: ✅ ACTIVE
💰 Equity: $10,450.00
💵 Cash: $8,500.00
📈 Open Positions: 2
💼 Current P&L: +$150.00
📊 Symbols: SPY, QQQ

⏰ 2025-10-10 10:30:00
```

```
You: /positions

Bot: 📊 Open Positions (2)

1. SPY - bull_put_spread
   🟢 P&L: +$75.00
   📅 Days: 15
   💰 Max Profit: $125.00
   ⚠️ Max Loss: $375.00

2. QQQ - iron_condor
   🟢 P&L: +$50.00
   📅 Days: 10
   💰 Max Profit: $200.00
   ⚠️ Max Loss: $600.00
```

---

## 🔧 **Troubleshooting:**

### **Can't find BotFather:**
- Search exactly: `@BotFather`
- It's an official Telegram bot
- Has a blue checkmark

### **Bot doesn't respond:**
- Make sure you sent `/start` first
- Check bot is running on droplet
- Verify token is correct

### **Commands don't work:**
- Check logs: `tail -f /opt/trading-agent/logs/telegram_bot.log`
- Verify chat ID is correct
- Restart bot if needed

---

## 🎉 **Benefits:**

### **Better Than Email:**
- ✅ **Instant** - No delay
- ✅ **Interactive** - Two-way communication
- ✅ **Control** - Stop/start from phone
- ✅ **Convenient** - Always with you
- ✅ **FREE** - No costs
- ✅ **No port issues** - Works everywhere

### **Use Cases:**
- Check status while away from computer
- Monitor positions during market hours
- Stop trading if needed
- Get instant alerts
- Review performance anytime

---

## 📋 **Setup Checklist:**

- [ ] Create bot with @BotFather
- [ ] Get bot token
- [ ] Get chat ID
- [ ] Add to .env file on droplet
- [ ] Install python-telegram-bot
- [ ] Start bot
- [ ] Test with /start command
- [ ] Verify /status works

---

## 🎯 **Ready to Set Up?**

**I need from you:**
1. Bot token (from @BotFather)
2. Chat ID (from getUpdates)

**Then I'll:**
1. Configure it on your droplet
2. Install dependencies
3. Start the bot
4. Test it

**Or you can follow the steps above yourself!**

---

**Let me know when you have the bot token and chat ID!** 🚀

