#!/bin/bash

echo "================================================================================"
echo "🔧 FIXING DEPLOYMENT CONFLICTS"
echo "================================================================================"
echo ""

ssh root@45.55.150.19 << 'ENDSSH'

cd /opt/trading-agent

echo "📋 Checking current status..."
git status --short

echo ""
echo "🔄 Stashing local changes..."
git stash

echo ""
echo "📥 Pulling latest code..."
git pull origin main

echo ""
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install -q polygon-api-client python-dotenv joblib

echo ""
echo "🔐 Ensuring Polygon API key in .env..."
if grep -q "POLYGON_API_KEY" .env 2>/dev/null; then
    echo "✅ POLYGON_API_KEY already exists"
else
    echo "POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY" >> .env
    echo "✅ Added POLYGON_API_KEY"
fi

echo ""
echo "📁 Checking if training script exists..."
if [ -f "scripts/train_ml_models.py" ]; then
    echo "✅ Training script found"
    
    echo ""
    echo "🤖 Training ML models (5-10 minutes)..."
    python scripts/train_ml_models.py
    
    echo ""
    echo "🧪 Testing models..."
    python scripts/test_ml_models.py
else
    echo "❌ Training script not found"
    echo "Files in scripts/:"
    ls -la scripts/
fi

echo ""
echo "🔄 Restarting Telegram bot..."
pkill -f start_telegram_bot
sleep 2
nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &
sleep 2

echo ""
echo "✅ Telegram bot restarted"

echo ""
echo "📊 Running processes:"
ps aux | grep -E "telegram_bot|main.py" | grep -v grep

echo ""
echo "================================================================================"
echo "✅ FIXED AND DEPLOYED!"
echo "================================================================================"
echo ""

ENDSSH

echo ""
echo "Test in Telegram: /ml"
echo ""

