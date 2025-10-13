#!/usr/bin/env python3
"""
Simple startup script for real-time data collection only
This bypasses the full orchestrator to test timezone fixes
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.brokers.alpaca_client import AlpacaClient
from src.market_data.realtime_collector import RealTimeDataCollector
from src.utils.symbol_selector import get_symbols_for_account

def setup_logging():
    """Configure logging"""
    logger.add(
        "logs/trading_agent.log",
        rotation="100 MB",
        retention="30 days",
        level="INFO"
    )

def main():
    """Start real-time data collection"""
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("Starting Real-Time Data Collector (Simple Mode)")
    logger.info("Testing Central Time (Texas) timezone configuration")
    logger.info("=" * 80)
    
    try:
        # Initialize Alpaca client
        alpaca = AlpacaClient()
        account = alpaca.get_account()
        account_balance = float(account.get("equity", 0))
        
        logger.info(f"Connected to Alpaca - Account Equity: ${account_balance:,.2f}")
        
        # Get symbols for account
        watchlist = get_symbols_for_account(account_balance)
        logger.info(f"Watching symbols: {', '.join(watchlist)}")
        
        # Start real-time collector
        collector = RealTimeDataCollector(
            symbols=watchlist,
            alpaca_client=alpaca,
            collect_interval=1.0,
            buffer_size=100
        )
        
        collector.start()
        logger.info("✅ Real-time data collection STARTED")
        logger.info("   Timezone: America/Chicago (Central Time)")
        logger.info("   Market Hours: 8:30 AM - 3:00 PM CT")
        logger.info("")
        logger.info("   Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(60)
            stats = collector.get_stats()
            logger.info(f"📊 Stats: {stats['total_ticks_collected']} ticks collected, "
                       f"{stats['total_ticks_stored']} stored, "
                       f"{stats['collection_errors']} errors")
    
    except KeyboardInterrupt:
        logger.info("\nReceived shutdown signal")
        collector.stop()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

