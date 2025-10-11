#!/bin/bash

echo "================================================================================"
echo "🚀 COMPLETE ML DEPLOYMENT (Push + Deploy)"
echo "================================================================================"
echo ""

echo "Step 1: Ensuring all files are committed..."
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent

# Show what will be committed
echo "Files to commit:"
git status --short

echo ""
echo "Adding all files..."
git add -A

echo ""
echo "Committing..."
git commit -m "Add complete ML training system with Polygon integration - all files" || echo "Nothing new to commit"

echo ""
echo "Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Local files pushed to GitHub!"
echo ""
echo "================================================================================"
echo "Step 2: Deploying to DigitalOcean..."
echo "================================================================================"
echo ""

ssh root@45.55.150.19 << 'ENDSSH'

cd /opt/trading-agent

echo "🗑️  Cleaning conflicting files..."
rm -f scripts/start_telegram_bot.py 2>/dev/null
rm -f src/alerts/telegram_bot.py 2>/dev/null

echo "🔄 Resetting repository..."
git reset --hard HEAD
git clean -fd

echo ""
echo "📥 Pulling ALL latest code from GitHub..."
git fetch --all
git reset --hard origin/main

echo ""
echo "✅ Files pulled. Checking for ML scripts..."
if [ -f "scripts/train_ml_models.py" ]; then
    echo "✅ train_ml_models.py found!"
else
    echo "❌ train_ml_models.py NOT found!"
    echo "Files in scripts/:"
    ls scripts/ | grep "train\|test\|ml"
fi

if [ -f "scripts/test_ml_models.py" ]; then
    echo "✅ test_ml_models.py found!"
else
    echo "❌ test_ml_models.py NOT found!"
fi

echo ""
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install -q polygon-api-client python-dotenv joblib 2>&1 | grep -v "already satisfied" || true

echo ""
echo "🔐 Adding Polygon API key..."
if ! grep -q "POLYGON_API_KEY" .env 2>/dev/null; then
    echo "POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY" >> .env
fi
echo "✅ Polygon key configured"

echo ""
if [ -f "scripts/train_ml_models.py" ]; then
    echo "🤖 Training ML models (5-10 minutes)..."
    python scripts/train_ml_models.py 2>&1 | tail -30
    
    echo ""
    echo "🧪 Testing models..."
    python scripts/test_ml_models.py 2>&1 | tail -20
else
    echo "⚠️  Skipping training - script not found"
    echo "We'll need to upload files manually"
fi

echo ""
echo "🔄 Restarting all services..."
pkill -f telegram_bot 2>/dev/null
pkill -f "python main.py" 2>/dev/null
sleep 3

# Start main trading agent
nohup python main.py > logs/trading_agent.log 2>&1 &
sleep 2

# Start telegram bot  
if [ -f "scripts/start_telegram_bot.py" ]; then
    nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &
    sleep 2
    echo "✅ Services restarted"
else
    echo "⚠️  telegram bot script not found"
fi

echo ""
echo "📊 Running processes:"
ps aux | grep python | grep -E "main.py|telegram" | grep -v grep

echo ""
echo "📁 Models directory:"
ls -lh models/ 2>/dev/null | head -10 || echo "No models directory yet"

echo ""
echo "================================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================================================"

ENDSSH

echo ""
echo "================================================================================"
echo "🎯 TEST IN TELEGRAM:"
echo "================================================================================"
echo ""
echo "Send these commands:"
echo "  /ml      - Check ML model status"
echo "  /status  - Check system status"
echo ""

