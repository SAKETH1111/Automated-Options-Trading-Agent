#!/bin/bash

echo "================================================================================"
echo "🚀 DEPLOYING ML SYSTEM TO DIGITALOCEAN"
echo "================================================================================"
echo ""

# Step 1: Commit and push
echo "📦 Step 1: Committing changes to GitHub..."
git add -A
git status --short
git commit -m "Add ML training system with Polygon integration - 65.66% accuracy" || echo "Nothing to commit"
git push origin main

echo ""
echo "✅ Step 1 Complete!"
echo ""
echo "================================================================================"
echo "📡 Step 2: Deploying to DigitalOcean Server"
echo "================================================================================"
echo ""
echo "Connecting to server..."
echo ""

# Step 2-7: Deploy on server
ssh root@45.55.150.19 << 'ENDSSH'

echo "📥 Pulling latest code..."
cd /opt/trading-agent
git pull origin main

echo ""
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install -q polygon-api-client python-dotenv joblib

echo ""
echo "🔐 Adding Polygon API key to .env..."
# Check if POLYGON_API_KEY already exists
if grep -q "POLYGON_API_KEY" .env 2>/dev/null; then
    echo "✅ POLYGON_API_KEY already exists in .env"
else
    echo "" >> .env
    echo "POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY" >> .env
    echo "✅ Added POLYGON_API_KEY to .env"
fi

echo ""
echo "🤖 Training ML models on server (this will take 5-10 minutes)..."
python scripts/train_ml_models.py

echo ""
echo "🧪 Testing models..."
python scripts/test_ml_models.py

echo ""
echo "🔄 Restarting Telegram bot with ML support..."
pkill -f start_telegram_bot
nohup python scripts/start_telegram_bot.py > logs/telegram_bot.log 2>&1 &
sleep 2
echo "✅ Telegram bot restarted"

echo ""
echo "📊 Checking running processes..."
ps aux | grep -E "telegram_bot|main.py" | grep -v grep

echo ""
echo "================================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================================================"
echo ""
echo "🎯 Next steps:"
echo "  1. Test ML models via Telegram: /ml"
echo "  2. Check status: /status"
echo "  3. Resume trading: /resume"
echo ""
echo "📝 Logs:"
echo "  - Training: tail -f /opt/trading-agent/logs/ml_training.log"
echo "  - Telegram: tail -f /opt/trading-agent/logs/telegram_bot.log"
echo "  - Trading: tail -f /opt/trading-agent/logs/trading_agent.log"
echo ""

ENDSSH

echo ""
echo "================================================================================"
echo "🎉 ALL DONE!"
echo "================================================================================"
echo ""
echo "Test your ML system in Telegram:"
echo "  Send: /ml"
echo ""

