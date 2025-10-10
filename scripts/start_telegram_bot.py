#!/usr/bin/env python3
"""
Start Telegram Bot
Run the Telegram bot for trading agent control
"""

import sys
import os

sys.path.append('/opt/trading-agent')

from src.database.session import get_session
from src.brokers.alpaca_client import AlpacaClient
from src.alerts.telegram_bot import TradingAgentBot

# Note: This requires auto_trader instance
# For now, create a standalone version

def main():
    """Start Telegram bot"""
    print("🤖 Starting Telegram Bot...")
    print("Note: Bot requires TELEGRAM_BOT_TOKEN in .env")
    
    db = get_session()
    alpaca = AlpacaClient()
    
    # Create bot (without auto_trader for now)
    bot = TradingAgentBot(db, alpaca, None)
    
    if bot.enabled:
        print("✅ Bot configured")
        print("🚀 Starting bot...")
        bot.run()
    else:
        print("❌ Bot not configured")
        print("Add TELEGRAM_BOT_TOKEN to .env file")

if __name__ == '__main__':
    main()

