#!/bin/bash

echo "================================================================================"
echo "🔧 FORCE DEPLOYING ML SYSTEM"
echo "================================================================================"
echo ""

ssh root@45.55.150.19 << 'ENDSSH'

cd /opt/trading-agent

echo "🗑️  Removing conflicting files..."
rm -f scripts/start_telegram_bot.py
rm -f src/alerts/telegram_bot.py

echo ""
echo "🔄 Resetting to clean state..."
git reset --hard HEAD
git clean -fd

echo ""
echo "📥 Pulling latest code..."
git pull origin main

echo ""
echo "📋 Checking what we got..."
ls -la scripts/ | grep -E "train_ml|test_ml"
ls -la src/ml/

echo ""
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install -q polygon-api-client python-dotenv joblib

echo ""
echo "🔐 Ensuring Polygon API key..."
if ! grep -q "POLYGON_API_KEY" .env 2>/dev/null; then
    echo "POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY" >> .env
    echo "✅ Added POLYGON_API_KEY"
else
    echo "✅ POLYGON_API_KEY exists"
fi

echo ""
echo "🤖 Training ML models (this takes 5-10 minutes)..."
if [ -f "scripts/train_ml_models.py" ]; then
    python scripts/train_ml_models.py
else
    echo "❌ ERROR: train_ml_models.py not found after pull!"
    echo "Listing all scripts:"
    ls -la scripts/
    echo ""
    echo "This means the files weren't pushed to GitHub properly."
    echo "We'll need to fix this manually."
fi

echo ""
echo "🔄 Restarting services..."
pkill -f start_telegram_bot
pkill -f main.py
sleep 3

# Start trading agent
nohup python main.py > logs/trading_agent.log 2>&1 &
sleep 2

# Start telegram bot
nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &
sleep 2

echo ""
echo "📊 Running processes:"
ps aux | grep python | grep -E "main.py|telegram" | grep -v grep

echo ""
echo "================================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================================================"

ENDSSH

echo ""
echo "🎯 Next: Test in Telegram with /ml"
echo ""

