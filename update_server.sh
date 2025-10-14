#!/bin/bash
#
# Server Update Script
# Updates code from GitHub and restarts the data collector
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                   📥 UPDATING SERVER                                       ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Pull latest code
echo "📥 Step 1/4: Pulling latest code from GitHub..."
git pull origin main
echo "✓ Code updated"
echo ""

# Step 2: Clear Python cache
echo "🧹 Step 2/4: Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "✓ Cache cleared"
echo ""

# Step 3: Restart collector
echo "🔄 Step 3/4: Restarting data collector..."

# Stop old process
OLD_PID=$(ps aux | grep '[s]tart_simple.py' | awk '{print $2}')
if [ ! -z "$OLD_PID" ]; then
    echo "  Stopping old collector (PID: $OLD_PID)..."
    pkill -f start_simple.py || true
    sleep 2
    echo "  ✓ Old collector stopped"
else
    echo "  No old collector running"
fi

# Start new process
echo "  Starting new collector..."
nohup python3 start_simple.py > logs/simple_collector.log 2>&1 &
NEW_PID=$!
sleep 2

# Verify it started
if ps -p $NEW_PID > /dev/null; then
    echo "  ✓ New collector started (PID: $NEW_PID)"
else
    echo "  ⚠ Collector may have crashed, checking logs..."
    tail -20 logs/simple_collector.log
    exit 1
fi
echo ""

# Step 4: Verify
echo "✅ Step 4/4: Verifying system..."
echo ""

# Check process
RUNNING=$(ps aux | grep '[s]tart_simple.py' | wc -l)
if [ $RUNNING -eq 1 ]; then
    echo "  ✓ Data collector is running"
else
    echo "  ⚠ Warning: Expected 1 collector process, found $RUNNING"
fi

# Show recent data
echo ""
echo "📊 Recent data (last 5 ticks):"
python3 view_data.py 2>/dev/null | tail -10 || echo "  (View data script not available)"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ SERVER UPDATE COMPLETE                                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  • Monitor logs: tail -f logs/simple_collector.log"
echo "  • View data: python3 view_data.py"
echo "  • Test timezone: python3 test_timezones.py"
echo ""

