#!/bin/bash

echo "🔧 Fixing Dependencies on DigitalOcean Server..."

# SSH into server and fix dependencies
ssh root@45.55.150.19 << 'EOF'
cd /opt/trading-agent
source venv/bin/activate

echo "📦 Installing core dependencies..."
pip install sqlalchemy psycopg2-binary python-dotenv

echo "📦 Installing pandas-ta (stable version)..."
pip install pandas-ta==0.3.14b

echo "📦 Installing remaining requirements..."
pip install -r requirements.txt

echo "🚀 Starting Telegram Bot..."
pkill -f telegram_bot.py
sleep 2
nohup python3 scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &

echo "📊 Checking bot status..."
sleep 3
ps aux | grep telegram_bot.py | grep -v grep

echo "✅ Dependencies fixed and bot restarted!"
EOF

echo "🎉 Done! Try sending /status in Telegram now."
