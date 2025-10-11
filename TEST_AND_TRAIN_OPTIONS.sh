#!/bin/bash

echo "================================================================================"
echo "🚀 POLYGON OPTIONS DATA - TEST & TRAIN"
echo "================================================================================"
echo ""

cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent

# Make scripts executable
chmod +x scripts/test_polygon_options.py
chmod +x scripts/train_with_options_data.py

# Step 1: Test Polygon options access
echo "Step 1: Testing Polygon Options API Access..."
echo "================================================================================"
echo ""

python3 scripts/test_polygon_options.py

echo ""
read -p "Did the test pass? (Press Enter to continue or Ctrl+C to stop) "

echo ""
echo "================================================================================"
echo "Step 2: Training ML Models with Options Data"
echo "================================================================================"
echo ""
echo "⏱️  This will take 15-25 minutes (fetching options data + training)"
echo ""
read -p "Ready to start? (Press Enter to continue or Ctrl+C to stop) "

echo ""
python3 scripts/train_with_options_data.py

echo ""
echo "================================================================================"
echo "Step 3: Test New Models"
echo "================================================================================"
echo ""

python3 scripts/test_ml_models.py

echo ""
echo "================================================================================"
echo "✅ COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Compare accuracy (should be +5-10% better)"
echo "  2. Deploy to server if improved"
echo "  3. Test in Telegram: /ml"
echo ""

