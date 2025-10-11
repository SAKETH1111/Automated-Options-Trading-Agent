#!/bin/bash

echo "🔧 Deploying Database Fixes..."

# Add changes to git
git add .

# Commit changes
git commit -m "Fix database session query issues - use proper session context managers"

# Push to GitHub
git push origin main

echo "📦 Deploying to DigitalOcean..."

# SSH into server and pull changes
ssh root@45.55.150.19 << 'EOF'
cd /opt/trading-agent

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Restart the Telegram bot
echo "🔄 Restarting Telegram Bot..."
pkill -f telegram_bot.py
sleep 2

# Start the bot
nohup python3 scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &

# Wait for it to start
sleep 5

# Check if it's running
echo "📊 Checking bot status..."
ps aux | grep telegram_bot.py | grep -v grep

echo "✅ Database fixes deployed and bot restarted!"
EOF

echo "🎉 Database fixes deployed! Try /status in Telegram now."
