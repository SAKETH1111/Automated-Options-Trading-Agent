#!/usr/bin/env python3
"""
Advanced Trading Agent Startup Script

This script starts the complete advanced trading system with all Polygon.io features:
- Real-time WebSocket streaming
- Historical data from S3 flat files
- Advanced ML training
- Live position monitoring
- Account size adaptation
"""

import os
import sys
import time
import signal
from datetime import datetime
from loguru import logger

# Set environment variables
os.environ['POLYGON_API_KEY'] = 'wWrUjjcksqLDPntXbJb72kiFzAwyqIpY'
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{os.getcwd()}/src"

# Add src to path
sys.path.append('src')

# Global variable to control the main loop
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    logger.info("🛑 Shutdown signal received. Stopping trading agent...")
    running = False

def main():
    """Main function to start the advanced trading agent"""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Advanced Polygon Trading Agent")
    logger.info("=" * 60)
    logger.info(f"Startup Time: {datetime.now()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    
    try:
        # Import the advanced integration system
        from src.trading.advanced_polygon_integration import AdvancedPolygonIntegration
        
        # Get account balance from environment or use default
        account_balance = float(os.environ.get('ACCOUNT_BALANCE', '25000'))
        logger.info(f"Account Balance: ${account_balance:,.2f}")
        
        # Initialize the advanced trading system
        logger.info("Initializing Advanced Trading System...")
        trading_system = AdvancedPolygonIntegration(account_balance=account_balance)
        
        # Get system configuration
        config = trading_system.config
        logger.info("System Configuration:")
        logger.info(f"  WebSocket Enabled: {config['websocket_enabled']}")
        logger.info(f"  Flat Files Enabled: {config['flat_files_enabled']}")
        logger.info(f"  Real-time ML: {config['real_time_ml']}")
        logger.info(f"  Live Position Monitoring: {config['live_position_monitoring']}")
        logger.info(f"  Data Refresh Interval: {config['data_refresh_interval']}s")
        logger.info(f"  ML Retrain Interval: {config['ml_retrain_interval']}s")
        
        # Get account tier information
        account_tier = trading_system.account_adaptation.current_tier
        logger.info(f"Account Tier: {account_tier.name}")
        logger.info(f"Recommended Symbols: {trading_system.account_adaptation.get_recommended_symbols()}")
        logger.info(f"Max Positions: {account_tier.max_positions}")
        logger.info(f"Risk per Trade: {account_tier.risk_per_trade:.1%}")
        
        # Get historical data summary
        historical_summary = trading_system.get_historical_data_summary()
        logger.info("Historical Data Available:")
        for data_type, info in historical_summary.items():
            logger.info(f"  {data_type.title()}: {info['total_dates']} dates")
            if info['latest']:
                logger.info(f"    Latest: {info['latest']}")
                logger.info(f"    Earliest: {info['earliest']}")
        
        # Start the advanced trading system
        logger.info("\n🚀 Starting Advanced Trading System...")
        trading_system.start_advanced_trading()
        
        logger.info("✅ Advanced Trading System started successfully!")
        logger.info("The system is now running with full Polygon.io Advanced account features:")
        logger.info("  🔴 Real-time WebSocket streaming")
        logger.info("  📊 Historical data from S3 flat files")
        logger.info("  🤖 Advanced ML training and predictions")
        logger.info("  📈 Live position monitoring")
        logger.info("  ⚖️ Account size adaptation")
        
        # Main monitoring loop
        logger.info("\n📊 Starting monitoring loop...")
        logger.info("Press Ctrl+C to stop the trading agent")
        
        while running:
            try:
                # Get system status
                status = trading_system.get_system_status()
                
                # Log status every 5 minutes
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info("📊 System Status Update:")
                    logger.info(f"  Running: {status['is_running']}")
                    logger.info(f"  WebSocket Connected: {status['websocket_connected']}")
                    logger.info(f"  Live Data Symbols: {len(status['live_data_symbols'])}")
                    logger.info(f"  Position Count: {status['position_count']}")
                    
                    # Get ML training status
                    ml_status = status.get('ml_training_status', {})
                    if ml_status.get('last_training'):
                        logger.info(f"  Last ML Training: {ml_status['last_training']}")
                    else:
                        logger.info("  Last ML Training: Never")
                
                # Get trading signals
                signals = trading_system.get_trading_signals()
                if signals['signals']:
                    logger.info(f"🚨 Trading Signals: {len(signals['signals'])} active")
                
                # Sleep for a short interval
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying
        
        # Shutdown sequence
        logger.info("\n🛑 Shutting down Advanced Trading System...")
        trading_system.stop_advanced_trading()
        
        # Final status
        final_status = trading_system.get_system_status()
        logger.info("Final System Status:")
        logger.info(f"  Running: {final_status['is_running']}")
        logger.info(f"  WebSocket Connected: {final_status['websocket_connected']}")
        logger.info(f"  Live Data Symbols: {len(final_status['live_data_symbols'])}")
        
        logger.info("✅ Advanced Trading System stopped successfully")
        logger.info("Thank you for using the Advanced Polygon Trading Agent!")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Please make sure all dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

