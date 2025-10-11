#!/bin/bash

echo "🔄 Restarting Telegram Bot..."

ssh root@45.55.150.19 << 'ENDSSH'

cd /opt/trading-agent

echo "Killing old telegram bot processes..."
pkill -9 -f telegram_bot
pkill -9 -f start_telegram_bot
sleep 3

echo "Checking if any still running..."
ps aux | grep telegram | grep -v grep || echo "All stopped ✅"

echo ""
echo "Starting fresh telegram bot..."
source venv/bin/activate
nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &
sleep 3

echo ""
echo "Checking new process..."
ps aux | grep telegram_bot | grep -v grep

echo ""
echo "Last 20 lines of log:"
tail -20 logs/telegram_bot.log

echo ""
echo "✅ Bot restarted!"

ENDSSH

echo ""
echo "Try in Telegram: /status or /ml"

