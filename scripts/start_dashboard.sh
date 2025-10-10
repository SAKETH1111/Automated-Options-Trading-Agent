#!/bin/bash

# Start Web Dashboard Script

cd /opt/trading-agent
export PYTHONPATH=/opt/trading-agent
source venv/bin/activate

echo "🚀 Starting Web Dashboard..."
echo "Access at: http://45.55.150.19:8000"

python src/dashboard/web_app.py

