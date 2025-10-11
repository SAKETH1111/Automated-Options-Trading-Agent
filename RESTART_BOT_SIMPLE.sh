#!/bin/bash

# Simple bot restart script
echo "🔄 Restarting Telegram Bot..."

# Kill any existing bot processes
ssh root@45.55.150.19 "pkill -f telegram_bot.py" 2>/dev/null

# Wait a moment
sleep 2

# Start the bot
ssh root@45.55.150.19 "cd /opt/trading-agent && nohup python3 scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &"

# Wait for it to start
sleep 3

# Check if it's running
echo "📊 Checking bot status..."
ssh root@45.55.150.19 "ps aux | grep telegram_bot.py | grep -v grep"

echo "✅ Bot restart complete!"
echo "📱 Try sending /status in Telegram now"
