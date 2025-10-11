#!/bin/bash

echo "================================================================================"
echo "🚀 FINAL DEPLOYMENT - OPTIONS-ENHANCED ML TO SERVER"
echo "================================================================================"
echo ""

ssh root@45.55.150.19 << 'ENDSSH'

cd /opt/trading-agent

echo "📥 Pulling latest code with options integration..."
git stash
git pull origin main

echo ""
echo "🤖 Training ML with options data (takes 2-3 minutes)..."
source venv/bin/activate
python scripts/train_ml_models.py 2>&1 | grep -E "INFO|✅|Entry Model|Accuracy|TRAINING"

echo ""
echo "🧪 Testing models..."
python scripts/test_ml_models.py 2>&1 | grep -E "✅|Models|Accuracy|PASSED"

echo ""
echo "📊 Model files:"
ls -lh models/*latest.pkl

echo ""
echo "✅ Server deployment complete!"

ENDSSH

echo ""
echo "================================================================================"
echo "🎉 DEPLOYMENT COMPLETE!"
echo "================================================================================"
echo ""
echo "Test in Telegram:"
echo "  /ml      - Check models"
echo "  /resume  - Start trading"
echo ""

