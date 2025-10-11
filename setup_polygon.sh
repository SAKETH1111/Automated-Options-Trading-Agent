#!/bin/bash

echo "🔧 Setting up Polygon.io API Key..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Polygon.io API Key
POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY

# Polygon S3 Flat Files (Optional)
POLYGON_S3_ACCESS_KEY_ID=88976b8e-f103-4d07-959f-4d13ce686dd2
POLYGON_S3_SECRET_ACCESS_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY
POLYGON_S3_ENDPOINT=https://files.polygon.io
POLYGON_S3_BUCKET=flatfiles

# Add your other keys here...
# ALPACA_API_KEY=
# ALPACA_SECRET_KEY=
# TELEGRAM_BOT_TOKEN=
# TELEGRAM_CHAT_ID=
EOF
    echo "✅ Created .env file with Polygon API key"
else
    # Check if POLYGON_API_KEY already exists
    if grep -q "POLYGON_API_KEY" .env; then
        echo "⚠️  POLYGON_API_KEY already exists in .env"
        echo "Updating it..."
        sed -i.bak 's/^POLYGON_API_KEY=.*/POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY/' .env
    else
        echo "Adding POLYGON_API_KEY to .env..."
        echo "" >> .env
        echo "# Polygon.io API Key" >> .env
        echo "POLYGON_API_KEY=wWrUjjcksqLDPntXbJb72kiFzAwyqIpY" >> .env
    fi
    echo "✅ Added Polygon API key to .env"
fi

echo ""
echo "📦 Installing Polygon SDK..."
pip3 install polygon-api-client

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run training: python3 scripts/train_ml_models.py"
echo ""



