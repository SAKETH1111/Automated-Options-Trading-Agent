#!/usr/bin/env python3
"""
Start Automated Paper Trading
Easy-to-use script to start automated trading
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.database.session import get_session
from src.brokers.alpaca_client import AlpacaClient
from src.automation import AutomatedTrader


def main():
    """Start automated paper trading"""
    parser = argparse.ArgumentParser(description='Start automated paper trading')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ'],
                       help='Symbols to trade')
    parser.add_argument('--interval', type=int, default=5,
                       help='Minutes between trading cycles')
    parser.add_argument('--max-positions', type=int, default=5,
                       help='Maximum number of positions')
    parser.add_argument('--max-risk', type=float, default=0.02,
                       help='Maximum risk per trade (as decimal, e.g., 0.02 = 2%%)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no actual orders)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  🚀 AUTOMATED PAPER TRADING")
    print("=" * 70)
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Cycle Interval: {args.interval} minutes")
    print(f"  Max Positions: {args.max_positions}")
    print(f"  Max Risk Per Trade: {args.max_risk:.1%}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE PAPER TRADING'}")
    print("=" * 70)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No actual orders will be placed")
    else:
        print("\n✅ LIVE MODE - Orders will be placed in paper trading account")
    
    print("\n📋 Press Ctrl+C to stop automated trading")
    print("=" * 70)
    
    # Initialize
    db = get_session()
    alpaca = AlpacaClient()
    
    # Create automated trader
    trader = AutomatedTrader(db, alpaca, symbols=args.symbols)
    trader.max_positions = args.max_positions
    trader.max_risk_per_trade = args.max_risk
    
    # Start automated trading
    try:
        trader.start_automated_trading(interval_minutes=args.interval)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping automated trading...")
        trader.stop_automated_trading()
        
        # Print final summary
        print("\n" + "=" * 70)
        print("  📊 FINAL SUMMARY")
        print("=" * 70)
        
        status = trader.get_status()
        if status.get('performance'):
            perf = status['performance']
            print(f"\n  Total Trades: {perf.get('total_trades', 0)}")
            print(f"  Win Rate: {perf.get('win_rate', 0):.1%}")
            print(f"  Total P&L: ${perf.get('total_pnl', 0):+,.2f}")
        
        print("\n" + "=" * 70)
        print("  Automated trading stopped")
        print("=" * 70)
    
    db.close()


if __name__ == '__main__':
    main()

