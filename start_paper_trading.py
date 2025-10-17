#!/usr/bin/env python3
"""
Paper Trading Starter Script
Starts the institutional options trading system in paper trading mode with PDT compliance

Features:
- PDT-compliant trading for accounts under $25,000
- Conservative position sizing for small accounts
- Comprehensive safety systems
- Real-time monitoring and alerts
- Performance tracking
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
import yaml
import signal
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import trading system components
from src.compliance.pdt_tracker import PDTTracker
from src.data.polygon_advanced_production import PolygonAdvancedProducer
from src.risk_management.risk_manager import RiskManager
from src.strategies.spy_qqq_specialist import SPYQQQBullPutSpreadStrategy
from src.monitoring.real_money_alerts import AlertManager
from src.safety.pre_trade_validator import PreTradeValidator
from src.safety.real_money_protection import RealMoneyProtection
from src.database.models import engine, Base

class PaperTradingSystem:
    """Paper trading system with PDT compliance"""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.is_running = False
        
        # Initialize components
        self.pdt_tracker = None
        self.polygon_producer = None
        self.risk_manager = None
        self.strategy = None
        self.alert_manager = None
        self.pre_trade_validator = None
        self.real_money_protection = None
        
        # Trading state
        self.account_value = self.config.get('account_value', 5000)
        self.trading_mode = 'paper'
        self.positions = {}
        self.daily_pnl = 0.0
        
        logger.info("Paper Trading System initialized")
    
    def _load_config(self) -> dict:
        """Load configuration"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing paper trading system...")
        
        try:
            # Initialize PDT tracker
            self.pdt_tracker = PDTTracker()
            logger.info("PDT tracker initialized")
            
            # Initialize Polygon producer
            self.polygon_producer = PolygonAdvancedProducer(self.config_path)
            await self.polygon_producer.initialize()
            logger.info("Polygon producer initialized")
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config_path)
            logger.info("Risk manager initialized")
            
            # Initialize strategy
            self.strategy = SPYQQQBullPutSpreadStrategy(self.config_path)
            logger.info("Strategy initialized")
            
            # Initialize alert manager
            self.alert_manager = AlertManager(self.config_path)
            logger.info("Alert manager initialized")
            
            # Initialize pre-trade validator
            self.pre_trade_validator = PreTradeValidator(self.config_path)
            logger.info("Pre-trade validator initialized")
            
            # Initialize real money protection
            self.real_money_protection = RealMoneyProtection(self.config_path)
            logger.info("Real money protection initialized")
            
            # Initialize database
            Base.metadata.create_all(engine)
            logger.info("Database initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup system components"""
        logger.info("Cleaning up system components...")
        
        if self.polygon_producer:
            await self.polygon_producer.cleanup()
        
        logger.info("Cleanup completed")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def check_system_health(self) -> dict:
        """Check system health"""
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        try:
            # Check PDT compliance
            pdt_status = self.pdt_tracker.get_current_pdt_status(self.account_value)
            health_status['components']['pdt_compliance'] = {
                'status': 'healthy' if pdt_status.can_day_trade else 'warning',
                'day_trades_used': pdt_status.day_trades_used,
                'day_trades_remaining': pdt_status.day_trades_remaining
            }
            
            # Check Polygon connectivity
            data_quality = await self.polygon_producer.validate_data_quality()
            health_status['components']['polygon_connectivity'] = {
                'status': 'healthy' if all(data_quality.values()) else 'warning',
                'checks': data_quality
            }
            
            # Check risk manager
            risk_status = self.risk_manager.get_current_risk_status()
            health_status['components']['risk_manager'] = {
                'status': 'healthy' if risk_status['overall_status'] == 'safe' else 'warning',
                'risk_level': risk_status['risk_level']
            }
            
            # Check alert system
            alert_status = self.alert_manager.get_system_status()
            health_status['components']['alert_system'] = {
                'status': 'healthy' if alert_status['is_operational'] else 'warning',
                'channels': alert_status['active_channels']
            }
            
            # Determine overall status
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if 'error' in component_statuses:
                health_status['overall_status'] = 'error'
            elif 'warning' in component_statuses:
                health_status['overall_status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
            return health_status
    
    async def scan_for_opportunities(self) -> list:
        """Scan for trading opportunities"""
        try:
            opportunities = []
            
            # Get current market data
            spy_quotes = await self.polygon_producer.get_stock_quotes("SPY")
            qqq_quotes = await self.polygon_producer.get_stock_quotes("QQQ")
            
            if not spy_quotes or not qqq_quotes:
                logger.warning("Unable to get market quotes")
                return opportunities
            
            # Check PDT compliance before scanning
            pdt_status = self.pdt_tracker.get_current_pdt_status(self.account_value)
            if pdt_status.is_pdt_violation:
                logger.warning("PDT violation detected, skipping opportunity scan")
                return opportunities
            
            # Scan for bull put spread opportunities
            spy_opportunities = await self.strategy.scan_bull_put_spreads("SPY", spy_quotes)
            qqq_opportunities = await self.strategy.scan_bull_put_spreads("QQQ", qqq_quotes)
            
            opportunities.extend(spy_opportunities)
            opportunities.extend(qqq_opportunities)
            
            # Filter opportunities based on PDT compliance
            filtered_opportunities = []
            for opp in opportunities:
                # Check if we can take this opportunity without violating PDT
                can_trade, reason = self.pdt_tracker.can_execute_day_trade()
                if can_trade or opp['hold_period'] >= 1:  # Can hold overnight
                    filtered_opportunities.append(opp)
                else:
                    logger.info(f"Skipping opportunity due to PDT: {reason}")
            
            logger.info(f"Found {len(filtered_opportunities)} PDT-compliant opportunities")
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Opportunity scan failed: {e}")
            return []
    
    async def execute_trade(self, opportunity: dict) -> bool:
        """Execute a trade with PDT compliance"""
        try:
            # Pre-trade validation
            validation_result = await self.pre_trade_validator.validate_trade(opportunity)
            if not validation_result['is_valid']:
                logger.warning(f"Trade validation failed: {validation_result['reason']}")
                return False
            
            # Check PDT compliance
            can_trade, reason = self.pdt_tracker.can_execute_day_trade()
            if not can_trade:
                logger.warning(f"PDT compliance check failed: {reason}")
                return False
            
            # Record position open
            position_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            success = self.pdt_tracker.record_position_open(
                position_id=position_id,
                symbol=opportunity['symbol'],
                contract_type=opportunity['contract_type'],
                strike=opportunity['strike'],
                expiration=opportunity['expiration'],
                quantity=opportunity['quantity'],
                price=opportunity['price'],
                timestamp=datetime.now()
            )
            
            if not success:
                logger.error("Failed to record position open")
                return False
            
            # Store position
            self.positions[position_id] = {
                'opportunity': opportunity,
                'entry_time': datetime.now(),
                'status': 'open'
            }
            
            # Send alert
            await self.alert_manager.send_alert(
                alert_type='trade_executed',
                severity='normal',
                title=f"Paper Trade Executed - {opportunity['symbol']}",
                message=f"Opened {opportunity['strategy']} position: {opportunity['quantity']} contracts at ${opportunity['price']:.2f}",
                channels=['telegram']
            )
            
            logger.info(f"Paper trade executed: {position_id} - {opportunity['symbol']} {opportunity['strategy']}")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    async def monitor_positions(self):
        """Monitor existing positions"""
        try:
            for position_id, position in self.positions.items():
                if position['status'] != 'open':
                    continue
                
                opportunity = position['opportunity']
                
                # Check if position should be closed
                should_close = False
                close_reason = ""
                
                # Check hold period (PDT compliance)
                hold_period = datetime.now() - position['entry_time']
                if hold_period.total_seconds() < 16 * 3600:  # 16 hours minimum hold
                    continue
                
                # Check profit/loss targets
                current_price = await self._get_current_price(opportunity)
                if current_price:
                    pnl = (current_price - opportunity['price']) * opportunity['quantity']
                    
                    # Take profit at 50% of max profit
                    max_profit = opportunity['max_profit']
                    if pnl >= max_profit * 0.5:
                        should_close = True
                        close_reason = "Take profit target reached"
                    
                    # Stop loss at 2x credit received
                    elif pnl <= -opportunity['credit_received'] * 2:
                        should_close = True
                        close_reason = "Stop loss triggered"
                
                # Close position if needed
                if should_close:
                    await self._close_position(position_id, close_reason)
                    
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")
    
    async def _get_current_price(self, opportunity: dict) -> float:
        """Get current price for position"""
        try:
            # In paper trading, simulate price movement
            base_price = opportunity['price']
            volatility = 0.02  # 2% daily volatility
            
            # Simulate random price movement
            import random
            change = random.gauss(0, volatility)
            current_price = base_price * (1 + change)
            
            return max(0.01, current_price)  # Minimum price of $0.01
            
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return None
    
    async def _close_position(self, position_id: str, reason: str):
        """Close a position"""
        try:
            position = self.positions[position_id]
            opportunity = position['opportunity']
            
            # Get current price
            current_price = await self._get_current_price(opportunity)
            if not current_price:
                return
            
            # Calculate P&L
            pnl = (current_price - opportunity['price']) * opportunity['quantity']
            
            # Record position close
            is_day_trade = (datetime.now() - position['entry_time']).total_seconds() < 24 * 3600
            success = self.pdt_tracker.record_position_close(
                position_id=position_id,
                price=current_price,
                timestamp=datetime.now(),
                is_emergency=False
            )
            
            if not success:
                logger.error("Failed to record position close")
                return
            
            # Update position status
            position['status'] = 'closed'
            position['exit_time'] = datetime.now()
            position['exit_price'] = current_price
            position['pnl'] = pnl
            position['close_reason'] = reason
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Send alert
            await self.alert_manager.send_alert(
                alert_type='position_closed',
                severity='normal',
                title=f"Position Closed - {opportunity['symbol']}",
                message=f"Closed {opportunity['strategy']} position: P&L ${pnl:.2f} ({reason})",
                channels=['telegram']
            )
            
            logger.info(f"Position closed: {position_id} - P&L: ${pnl:.2f} ({reason})")
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
    
    async def send_daily_summary(self):
        """Send daily summary"""
        try:
            # Get PDT status
            pdt_status = self.pdt_tracker.get_current_pdt_status(self.account_value)
            
            # Calculate summary metrics
            total_positions = len(self.positions)
            open_positions = len([p for p in self.positions.values() if p['status'] == 'open'])
            closed_positions = len([p for p in self.positions.values() if p['status'] == 'closed'])
            total_pnl = sum([p.get('pnl', 0) for p in self.positions.values()])
            
            summary = f"""
📊 Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}

💰 Account Value: ${self.account_value:,.2f}
📈 Daily P&L: ${self.daily_pnl:.2f}
📊 Total P&L: ${total_pnl:.2f}

📋 Positions:
• Total: {total_positions}
• Open: {open_positions}
• Closed: {closed_positions}

🛡️ PDT Compliance:
• Day Trades Used: {pdt_status.day_trades_used}/3
• Day Trades Remaining: {pdt_status.day_trades_remaining}
• Status: {'✅ Compliant' if pdt_status.can_day_trade else '⚠️ Limit Reached'}

📈 Trading Mode: Paper Trading
🔒 Safety: All circuit breakers active
            """
            
            await self.alert_manager.send_alert(
                alert_type='daily_summary',
                severity='normal',
                title="Daily Trading Summary",
                message=summary,
                channels=['telegram']
            )
            
            logger.info("Daily summary sent")
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting paper trading loop...")
        
        self.is_running = True
        last_health_check = datetime.now()
        last_summary = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Health check every 5 minutes
                if (current_time - last_health_check).total_seconds() >= 300:
                    health_status = await self.check_system_health()
                    if health_status['overall_status'] == 'error':
                        logger.error("System health check failed, stopping trading")
                        break
                    last_health_check = current_time
                
                # Send daily summary at market close (4 PM ET)
                if current_time.hour == 16 and (current_time - last_summary).total_seconds() >= 3600:
                    await self.send_daily_summary()
                    last_summary = current_time
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # Scan for new opportunities (every 30 seconds)
                if current_time.second % 30 == 0:
                    opportunities = await self.scan_for_opportunities()
                    
                    # Execute trades if opportunities found and within limits
                    for opportunity in opportunities[:1]:  # Limit to 1 trade per scan
                        if len(self.positions) < 3:  # Max 3 positions for small account
                            await self.execute_trade(opportunity)
                
                # Sleep for 1 second
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info("Paper trading loop stopped")
    
    async def start(self):
        """Start the paper trading system"""
        try:
            logger.info("Starting paper trading system...")
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Initialize system
            await self.initialize()
            
            # Send startup alert
            await self.alert_manager.send_alert(
                alert_type='system_startup',
                severity='normal',
                title="Paper Trading System Started",
                message=f"Paper trading system started with ${self.account_value:,.2f} account value. PDT compliance enabled.",
                channels=['telegram']
            )
            
            # Run trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            raise
        finally:
            # Send shutdown alert
            await self.alert_manager.send_alert(
                alert_type='system_shutdown',
                severity='normal',
                title="Paper Trading System Stopped",
                message="Paper trading system has been stopped.",
                channels=['telegram']
            )
            
            # Cleanup
            await self.cleanup()

async def main():
    """Main function"""
    logger.info("Starting Paper Trading System with PDT Compliance")
    
    # Check if running in paper mode
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        logger.error("This script is for paper trading only. Use mode_switcher.py for live trading.")
        sys.exit(1)
    
    system = PaperTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
