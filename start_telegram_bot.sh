#!/bin/bash
#
# Start Telegram Report Bot
# Runs in background and responds to commands
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              🤖 STARTING TELEGRAM REPORT BOT                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if already running
if pgrep -f "unified_telegram_bot.py" > /dev/null; then
    echo "⚠️  Telegram bot is already running"
    echo ""
    echo "To restart:"
    echo "  1. Stop: pkill -f unified_telegram_bot.py"
    echo "  2. Start: ./start_telegram_bot.sh"
    exit 1
fi

# Check credentials
if ! grep -q "TELEGRAM_BOT_TOKEN" .env 2>/dev/null; then
    echo "❌ Error: Telegram credentials not found in .env file"
    echo ""
    echo "Please add to .env:"
    echo "  TELEGRAM_BOT_TOKEN=your_token_here"
    echo "  TELEGRAM_CHAT_ID=your_chat_id_here"
    exit 1
fi

# Start bot
echo "🚀 Starting unified bot..."
nohup python3 unified_telegram_bot.py > logs/telegram_bot.log 2>&1 &
BOT_PID=$!

sleep 2

# Check if running
if ps -p $BOT_PID > /dev/null; then
    echo "✅ Telegram bot started (PID: $BOT_PID)"
    echo ""
    echo "📱 Available commands in Telegram:"
    echo "   Reports: /report, /summary, /signals"
    echo "   Trading: /status, /positions, /pnl, /risk"
    echo "   System: /ml, /pdt, /stop, /resume"
    echo "   /help - Show all commands"
    echo ""
    echo "📋 Bot logs: tail -f logs/telegram_bot.log"
    echo ""
    echo "To stop: pkill -f unified_telegram_bot.py"
else
    echo "❌ Failed to start bot, check logs:"
    tail -20 logs/telegram_bot.log
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ TELEGRAM BOT READY                                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

