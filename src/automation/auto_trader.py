"""
Automated Trader Module
Main orchestrator for automated paper trading
"""

from typing import Dict, List
from datetime import datetime
import time
from sqlalchemy.orm import Session
from loguru import logger

from .signal_generator import AutomatedSignalGenerator
from .order_executor import AutomatedOrderExecutor
from .position_manager import AutomatedPositionManager
from .trade_manager import AutomatedTradeManager
from .performance_tracker import PerformanceTracker


class AutomatedTrader:
    """
    Main automated trading orchestrator
    Coordinates signal generation, order execution, and position management
    """
    
    def __init__(
        self,
        db_session: Session,
        alpaca_client,
        symbols: List[str] = ['SPY', 'QQQ']
    ):
        """
        Initialize automated trader
        
        Args:
            db_session: Database session
            alpaca_client: Alpaca client
            symbols: Symbols to trade
        """
        self.db = db_session
        self.alpaca = alpaca_client
        self.symbols = symbols
        
        # Initialize components
        self.signal_generator = AutomatedSignalGenerator(db_session)
        self.order_executor = AutomatedOrderExecutor(db_session, alpaca_client)
        self.position_manager = AutomatedPositionManager(db_session, alpaca_client)
        self.trade_manager = AutomatedTradeManager(db_session)
        self.performance_tracker = PerformanceTracker(db_session)
        
        # Trading parameters
        self.max_positions = 5
        self.max_risk_per_trade = 0.02  # 2% of capital
        self.is_running = False
        
        logger.info(f"Automated Trader initialized for {symbols}")
    
    def run_trading_cycle(self) -> Dict:
        """
        Run one complete trading cycle
        
        Returns:
            Cycle summary
        """
        cycle_start = datetime.utcnow()
        logger.info("=" * 60)
        logger.info(f"Starting trading cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        summary = {
            'timestamp': cycle_start,
            'signals_generated': 0,
            'orders_executed': 0,
            'positions_closed': 0,
            'positions_managed': 0,
            'errors': []
        }
        
        try:
            # Step 1: Check if we should trade now
            if not self.signal_generator.should_trade_now():
                logger.info("Outside trading hours or restricted period")
                return summary
            
            # Step 2: Manage existing positions
            logger.info("Step 1: Managing existing positions...")
            managed = self._manage_existing_positions()
            summary['positions_managed'] = managed['managed']
            summary['positions_closed'] = managed['closed']
            
            # Step 3: Generate new entry signals
            logger.info("Step 2: Generating entry signals...")
            entry_signals = self.signal_generator.generate_entry_signals(self.symbols)
            summary['signals_generated'] = len(entry_signals)
            
            if entry_signals:
                logger.info(f"Found {len(entry_signals)} potential opportunities")
            
            # Step 4: Apply risk management filters
            logger.info("Step 3: Applying risk management...")
            account = self.alpaca.get_account()
            available_capital = float(account.get('cash', 10000))
            
            filtered_signals = self.signal_generator.filter_signals_by_risk(
                entry_signals,
                max_positions=self.max_positions,
                max_risk_per_trade=self.max_risk_per_trade,
                available_capital=available_capital
            )
            
            # Step 5: Execute approved signals
            logger.info("Step 4: Executing approved signals...")
            for signal in filtered_signals:
                result = self.order_executor.execute_entry_signal(signal)
                if result:
                    summary['orders_executed'] += 1
            
            # Step 6: Update portfolio status
            logger.info("Step 5: Updating portfolio status...")
            portfolio = self.position_manager.get_portfolio_summary()
            summary['portfolio'] = portfolio
            
            logger.info("=" * 60)
            logger.info(f"Trading cycle complete:")
            logger.info(f"  Signals: {summary['signals_generated']}")
            logger.info(f"  Orders: {summary['orders_executed']}")
            logger.info(f"  Closed: {summary['positions_closed']}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            summary['errors'].append(str(e))
        
        return summary
    
    def _manage_existing_positions(self) -> Dict:
        """Manage existing positions"""
        managed_count = 0
        closed_count = 0
        
        try:
            # Get open positions
            positions = self.position_manager.get_open_positions()
            
            if not positions:
                logger.info("No open positions to manage")
                return {'managed': 0, 'closed': 0}
            
            logger.info(f"Managing {len(positions)} open positions")
            
            # Update position values
            positions = self.position_manager.update_position_values(positions)
            
            # Generate exit signals
            exit_signals = self.signal_generator.generate_exit_signals(positions)
            
            # Execute exit signals
            for exit_signal in exit_signals:
                # Find corresponding position
                position = next(
                    (p for p in positions if p['position_id'] == exit_signal['position_id']),
                    None
                )
                
                if position:
                    success = self.order_executor.execute_exit_signal(exit_signal, position)
                    if success:
                        closed_count += 1
            
            # Manage remaining positions
            for position in positions:
                # Check if position needs management
                stock_data = self.alpaca.get_stock_data(position['symbol'])
                if stock_data:
                    action = self.trade_manager.manage_position(position, stock_data['price'])
                    
                    if action and action['action'] in ['CLOSE', 'MONITOR']:
                        managed_count += 1
                        logger.info(f"Position {position['position_id']}: {action['message']}")
            
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
        
        return {
            'managed': managed_count,
            'closed': closed_count
        }
    
    def start_automated_trading(self, interval_minutes: int = 5):
        """
        Start automated trading loop
        
        Args:
            interval_minutes: Minutes between trading cycles
        """
        logger.info("=" * 60)
        logger.info("🚀 STARTING AUTOMATED PAPER TRADING")
        logger.info("=" * 60)
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Max Positions: {self.max_positions}")
        logger.info(f"Max Risk Per Trade: {self.max_risk_per_trade:.1%}")
        logger.info(f"Cycle Interval: {interval_minutes} minutes")
        logger.info("=" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Run trading cycle
                summary = self.run_trading_cycle()
                
                # Wait for next cycle
                logger.info(f"Waiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Automated trading stopped by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"Error in automated trading loop: {e}")
            self.is_running = False
    
    def stop_automated_trading(self):
        """Stop automated trading"""
        logger.info("Stopping automated trading...")
        self.is_running = False
    
    def get_status(self) -> Dict:
        """Get current status of automated trader"""
        portfolio = self.position_manager.get_portfolio_summary()
        performance = self.performance_tracker.get_all_time_stats()
        
        return {
            'is_running': self.is_running,
            'timestamp': datetime.utcnow(),
            'symbols': self.symbols,
            'portfolio': portfolio,
            'performance': performance
        }

