#!/bin/bash

# Quick Health Check Script
SERVER_IP="45.55.150.19"

echo "🔍 Quick Health Check"
echo "===================="

# Check service status
STATUS=$(ssh root@$SERVER_IP "systemctl is-active trading-agent" 2>/dev/null)
if [ "$STATUS" = "active" ]; then
    echo "✅ Agent: RUNNING"
else
    echo "❌ Agent: NOT RUNNING"
    exit 1
fi

# Check data collection (last 5 minutes)
TICK_COUNT=$(ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT COUNT(*) FROM index_tick_data WHERE timestamp > NOW() - INTERVAL '5 minutes';\" 2>/dev/null | grep -o '[0-9]*' | tail -1")

if [ "$TICK_COUNT" -gt 100 ]; then
    echo "✅ Data Collection: ACTIVE ($TICK_COUNT ticks in last 5 min)"
else
    echo "⚠️  Data Collection: SLOW ($TICK_COUNT ticks in last 5 min)"
fi

# Check latest data timestamp
LATEST=$(ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT MAX(timestamp) FROM index_tick_data;\" 2>/dev/null | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1")
echo "📅 Latest Data: $LATEST"

echo ""
echo "🎯 Agent is healthy and collecting data!"
