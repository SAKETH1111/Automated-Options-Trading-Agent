#!/bin/bash

echo "================================================================================"
echo "🚀 ML MODEL TRAINING - STARTING NOW"
echo "================================================================================"
echo ""

# Navigate to project directory
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent

# Check if yfinance is installed
echo "📦 Checking dependencies..."
python3 -c "import yfinance" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  yfinance not found. Installing..."
    pip3 install yfinance
else
    echo "✅ yfinance installed"
fi

echo ""
echo "================================================================================"
echo "🤖 Starting ML Training (this will take 5-15 minutes)"
echo "================================================================================"
echo ""

# Run training
python3 scripts/train_ml_models.py

echo ""
echo "================================================================================"
echo "✅ Training Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Test models: python3 scripts/test_ml_models.py"
echo "  2. Deploy to server: ./deploy_ml.sh"
echo "  3. Check via Telegram: /ml"
echo ""


