#!/usr/bin/env python3
"""Check if setup is correct"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.brokers.alpaca_client import AlpacaClient
from src.database.session import get_db
from loguru import logger


def check_config():
    """Check configuration"""
    logger.info("Checking configuration...")
    
    try:
        settings = get_settings()
        
        if not settings.alpaca_api_key:
            logger.error("❌ ALPACA_API_KEY not set")
            return False
        
        if not settings.alpaca_secret_key:
            logger.error("❌ ALPACA_SECRET_KEY not set")
            return False
        
        logger.info("✅ Configuration OK")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False


def check_alpaca_connection():
    """Check Alpaca connection"""
    logger.info("Checking Alpaca connection...")
    
    try:
        client = AlpacaClient()
        account = client.get_account()
        
        logger.info(f"✅ Connected to Alpaca")
        logger.info(f"   Account Equity: ${account['equity']:,.2f}")
        logger.info(f"   Buying Power: ${account['buying_power']:,.2f}")
        return True
    except Exception as e:
        logger.error(f"❌ Alpaca connection error: {e}")
        return False


def check_database():
    """Check database connection"""
    logger.info("Checking database...")
    
    try:
        db = get_db()
        logger.info("✅ Database connection OK")
        return True
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False


def main():
    """Run all checks"""
    logger.info("=" * 80)
    logger.info("Setup Check")
    logger.info("=" * 80)
    
    checks = [
        check_config(),
        check_alpaca_connection(),
        check_database(),
    ]
    
    logger.info("=" * 80)
    
    if all(checks):
        logger.info("✅ All checks passed! Ready to start trading.")
        return 0
    else:
        logger.error("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())




