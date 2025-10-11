#!/usr/bin/env python3
"""
Start the Telegram Bot for the Trading Agent
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.session import get_db
from src.brokers.alpaca_client import AlpacaClient
from src.automation.auto_trader import AutomatedTrader
from src.alerts.telegram_bot import TradingAgentBot
from loguru import logger

def main():
    """Start the Telegram bot"""
    logger.info("🤖 Starting Telegram Bot...")
    
    # Initialize database
    db = get_db()
    logger.info("✅ Database connected")
    
    # Initialize Alpaca client
    alpaca = AlpacaClient()
    logger.info("✅ Alpaca client initialized")
    
    # Initialize automated trader
    auto_trader = AutomatedTrader(db, alpaca)
    logger.info("✅ Automated trader initialized")
    
    # Initialize and run Telegram bot
    bot = TradingAgentBot(db, alpaca, auto_trader)
    
    if not bot.enabled:
        logger.error("❌ Telegram bot not configured. Please set TELEGRAM_BOT_TOKEN in .env")
        return
    
    logger.info(f"✅ Telegram bot configured")
    logger.info(f"📱 Bot will respond to chat ID: {bot.chat_id}")
    logger.info("🚀 Starting bot... Press Ctrl+C to stop")
    
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("👋 Telegram bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running bot: {e}")
        raise

if __name__ == "__main__":
    main()
