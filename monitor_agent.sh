#!/bin/bash

# Trading Agent Monitor Script
# Usage: ./monitor_agent.sh

SERVER_IP="45.55.150.19"

echo "🚀 Trading Agent Status Monitor"
echo "================================"
echo "Server: $SERVER_IP"
echo "Time: $(date)"
echo ""

# Check if service is running
echo "📊 SERVICE STATUS:"
ssh root@$SERVER_IP "systemctl is-active trading-agent" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Trading Agent: RUNNING"
else
    echo "❌ Trading Agent: NOT RUNNING"
fi

echo ""

# Check data collection
echo "📈 DATA COLLECTION:"
ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT symbol, COUNT(*) as ticks, MAX(timestamp) as latest FROM index_tick_data GROUP BY symbol ORDER BY ticks DESC;\"" 2>/dev/null

echo ""

# Check latest prices
echo "💰 LATEST PRICES:"
ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT symbol, price, timestamp FROM index_tick_data ORDER BY timestamp DESC LIMIT 4;\"" 2>/dev/null

echo ""

# Check system resources
echo "💻 SYSTEM RESOURCES:"
ssh root@$SERVER_IP "ps aux | grep 'python main.py' | grep -v grep" 2>/dev/null

echo ""
echo "📝 To view live logs: ssh root@$SERVER_IP 'journalctl -u trading-agent -f'"
echo "🔄 To restart agent:  ssh root@$SERVER_IP 'systemctl restart trading-agent'"
