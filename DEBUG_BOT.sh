#!/bin/bash

echo "🔍 Debugging Telegram Bot..."

# SSH into server and debug
ssh root@45.55.150.19 << 'EOF'
cd /opt/trading-agent

echo "📊 Checking bot process..."
ps aux | grep telegram_bot.py | grep -v grep

echo "📋 Checking bot logs..."
if [ -f logs/telegram_bot.log ]; then
    echo "Last 10 lines of bot log:"
    tail -10 logs/telegram_bot.log
else
    echo "No bot log file found"
fi

echo "🐍 Testing Python imports..."
source venv/bin/activate
python3 -c "
try:
    from src.alerts.telegram_bot import TradingAgentBot
    print('✅ Telegram bot import: SUCCESS')
except Exception as e:
    print(f'❌ Telegram bot import: {e}')

try:
    from src.database.session import get_db
    print('✅ Database import: SUCCESS')
except Exception as e:
    print(f'❌ Database import: {e}')

try:
    from src.brokers.alpaca_client import AlpacaClient
    print('✅ Alpaca import: SUCCESS')
except Exception as e:
    print(f'❌ Alpaca import: {e}')
"

echo "🔧 Restarting bot..."
pkill -f telegram_bot.py
sleep 2
nohup python3 scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &

echo "⏳ Waiting for bot to start..."
sleep 5

echo "📊 Final status check..."
ps aux | grep telegram_bot.py | grep -v grep

EOF

echo "✅ Debug complete! Check the output above."
