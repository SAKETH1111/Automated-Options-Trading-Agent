#!/bin/bash
#
# Start Full Trading Agent with Orchestrator
# Runs signal generation and paper trading
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              🚀 STARTING FULL TRADING AGENT                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if already running
if pgrep -f "main.py" > /dev/null; then
    echo "⚠️  Trading agent is already running"
    echo ""
    echo "To restart:"
    echo "  1. Stop: pkill -f main.py"
    echo "  2. Start: ./start_trading_agent.sh"
    exit 1
fi

# Check .env credentials
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo ""
    echo "Please create .env with:"
    echo "  ALPACA_API_KEY=your_key"
    echo "  ALPACA_SECRET_KEY=your_secret"
    echo "  TELEGRAM_BOT_TOKEN=your_token"
    echo "  TELEGRAM_CHAT_ID=your_chat_id"
    exit 1
fi

echo "✅ Configuration found"
echo ""

# Start trading agent
echo "🚀 Starting trading agent with full orchestrator..."
nohup python3 main.py > logs/trading_agent.log 2>&1 &
AGENT_PID=$!

sleep 3

# Check if running
if ps -p $AGENT_PID > /dev/null 2>&1; then
    echo "✅ Trading agent started (PID: $AGENT_PID)"
    echo ""
    echo "📊 Features enabled:"
    echo "   • Real-time data collection"
    echo "   • Signal generation"
    echo "   • Paper trade execution"
    echo "   • Position monitoring"
    echo "   • PDT compliance"
    echo "   • Risk management"
    echo ""
    echo "📋 Monitor logs:"
    echo "   tail -f logs/trading_agent.log"
    echo ""
    echo "📱 Telegram commands:"
    echo "   /status, /positions, /pnl, /signals, /report"
else
    echo "❌ Failed to start agent, checking logs..."
    tail -30 logs/trading_agent.log
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ TRADING AGENT RUNNING                                                  ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

