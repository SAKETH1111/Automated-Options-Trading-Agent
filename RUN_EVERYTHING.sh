#!/bin/bash

echo "================================================================================"
echo "🚀 POLYGON ML TRAINING - COMPLETE SETUP & RUN"
echo "================================================================================"
echo ""

# Navigate to project
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -q polygon-api-client python-dotenv

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'EOF'
POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY
EOF
else
    # Add Polygon key if not exists
    if ! grep -q "POLYGON_API_KEY" .env; then
        echo "" >> .env
        echo "POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY" >> .env
    fi
fi

echo "✅ Setup complete!"
echo ""
echo "================================================================================"
echo "🤖 Starting ML Training with Polygon Data"
echo "================================================================================"
echo ""

# Run training
python3 scripts/train_ml_models.py

echo ""
echo "================================================================================"
echo "✅ Done!"
echo "================================================================================"



