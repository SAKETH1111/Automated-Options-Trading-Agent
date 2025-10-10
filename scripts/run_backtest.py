#!/usr/bin/env python3
"""
Run Backtest Script
Easy-to-use script for backtesting strategies
"""

import sys
import os
from datetime import datetime, timedelta
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.database.session import get_session
from src.backtesting import StrategyTester
from src.backtesting.reporter import BacktestReporter


def main():
    """Main backtest runner"""
    parser = argparse.ArgumentParser(description='Run strategy backtests')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to test')
    parser.add_argument('--strategy', type=str, default='bull_put_spread',
                       choices=['bull_put_spread', 'iron_condor', 'cash_secured_put', 'all'],
                       help='Strategy to test')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--export', action='store_true', help='Export trades to CSV')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(f"  🚀 Running Backtest - {args.strategy.upper()}")
    print("=" * 70)
    print(f"  Symbol: {args.symbol}")
    print(f"  Period: Last {args.days} days")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize
    db = get_session()
    tester = StrategyTester(db)
    reporter = BacktestReporter()
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Run backtest
    if args.strategy == 'all':
        print("\n📊 Running all strategies...")
        results = tester.compare_strategies(args.symbol, start_date, end_date)
        
        for strategy_name, result in results.items():
            print(f"\n{'=' * 70}")
            print(f"Strategy: {strategy_name.upper()}")
            print(f"{'=' * 70}")
            report = reporter.generate_text_report(result, strategy_name)
            print(report)
            
            if args.export:
                filename = f"backtest_{strategy_name}_{args.symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
                reporter.export_to_csv(result.trades, filename)
    
    else:
        # Run single strategy
        if args.strategy == 'bull_put_spread':
            result = tester.test_bull_put_spread_strategy(args.symbol, start_date, end_date)
        elif args.strategy == 'iron_condor':
            result = tester.test_iron_condor_strategy(args.symbol, start_date, end_date)
        elif args.strategy == 'cash_secured_put':
            result = tester.test_cash_secured_put_strategy(args.symbol, start_date, end_date)
        
        # Generate report
        report = reporter.generate_text_report(result, args.strategy)
        print(report)
        
        # Generate trade log
        if result.trades:
            trade_log = reporter.generate_trade_log(result.trades)
            print(trade_log)
            
            monthly = reporter.generate_monthly_summary(result.trades)
            print(monthly)
        
        # Export if requested
        if args.export and result.trades:
            filename = f"backtest_{args.strategy}_{args.symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            reporter.export_to_csv(result.trades, filename)
            print(f"\n✅ Trades exported to: {filename}")
    
    print("\n" + "=" * 70)
    print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    db.close()


if __name__ == '__main__':
    from typing import Optional, Dict
    main()

