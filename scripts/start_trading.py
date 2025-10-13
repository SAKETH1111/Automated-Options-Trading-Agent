"""
Simple script to start the automated trading agent
Uses the existing AutomatedTrader class
"""

import sys
import os
import time
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.automation.auto_trader import AutomatedTrader
from src.brokers.alpaca_client import AlpacaClient
from src.database.session import get_db
from src.utils.symbol_selector import get_symbols_for_account

def main():
    """Start the automated trading agent"""
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting Automated Trading Agent")
        logger.info("=" * 80)
        
        # Initialize components
        db = get_db()
        alpaca = AlpacaClient()
        
        # Get account info and smart symbols
        account = alpaca.get_account()
        account_balance = float(account.get("equity", 0))
        symbols = get_symbols_for_account(account_balance)
        
        logger.info(f"💰 Account Balance: ${account_balance:,.2f}")
        logger.info(f"📊 Trading Symbols: {', '.join(symbols)}")
        
        # Get a database session (keep it open for the lifetime of the trader)
        session = db.SessionLocal()
        
        try:
            # Initialize automated trader
            trader = AutomatedTrader(
                db_session=session,
                alpaca_client=alpaca,
                symbols=symbols
            )
            
            logger.info("✅ Automated Trader initialized")
            
            # Start trading
            trader.start()
            logger.info("🎯 Trading agent is now ACTIVE")
            logger.info("📱 Monitor via Telegram bot: /status")
            
            # Keep running
            while True:
                time.sleep(60)  # Check every minute
                    
        except KeyboardInterrupt:
            logger.info("\n⏸️  Stopping trading agent...")
            trader.stop()
            logger.info("✅ Trading agent stopped")
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error starting trading agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

