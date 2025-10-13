"""Main orchestrator that coordinates all trading agent components"""

import time
from datetime import datetime, time as dt_time
from typing import Dict, Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from src.brokers.alpaca_client import AlpacaClient
from src.config.settings import get_config
from src.database.session import get_db
from src.execution.executor import TradeExecutor
from src.execution.monitor import PositionMonitor
from src.learning.analyzer import TradeAnalyzer
from src.learning.learner import StrategyLearner
from src.market_data.collector import MarketDataCollector
from src.market_data.realtime_collector import RealTimeDataCollector
from src.monitoring.alerts import AlertManager
from src.monitoring.logger import setup_logging
from src.risk.manager import RiskManager
from src.risk_management.pdt_compliance import PDTComplianceManager
from src.signals.generator import SignalGenerator
from src.utils.symbol_selector import get_symbols_for_account, get_symbol_info


class TradingOrchestrator:
    """Main orchestrator for the automated trading agent"""
    
    def __init__(self):
        # Setup logging first
        setup_logging()
        logger.info("=" * 80)
        logger.info("Initializing Trading Orchestrator")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = get_config()
        
        # Initialize database
        self.db = get_db()
        
        # Initialize components
        self.alpaca = AlpacaClient()
        self.market_data = MarketDataCollector(self.alpaca)
        self.risk_manager = RiskManager(self.config)
        
        # Initialize PDT compliance manager
        account = self.alpaca.get_account()
        account_balance = float(account.get("equity", 0))
        self.pdt_manager = PDTComplianceManager(account_balance)
        
        # Smart symbol selection based on account size
        self.watchlist = get_symbols_for_account(account_balance)
        symbol_info = get_symbol_info(account_balance)
        logger.info(f"📊 Smart Symbol Selection:")
        logger.info(f"   Account: ${account_balance:,.2f}")
        logger.info(f"   Tier: {symbol_info['tier']}")
        logger.info(f"   Symbols: {', '.join(self.watchlist)}")
        logger.info(f"   Max Stock Price: ${symbol_info['max_stock_price']}")
        logger.info(f"   Spread Width: {symbol_info['preferred_spread_width']}")
        
        self.signal_generator = SignalGenerator(self.market_data, self.config)
        self.trade_executor = TradeExecutor(self.alpaca, self.risk_manager, self.config)
        self.position_monitor = PositionMonitor(self.market_data)
        self.alert_manager = AlertManager()
        self.trade_analyzer = TradeAnalyzer()
        self.strategy_learner = StrategyLearner(self.config)
        
        # Initialize real-time data collector with smart symbols
        realtime_config = self.config.get("realtime_data", {})
        self.realtime_collector = RealTimeDataCollector(
            symbols=self.watchlist,
            alpaca_client=self.alpaca,
            collect_interval=realtime_config.get("collect_interval_seconds", 1.0),
            buffer_size=realtime_config.get("buffer_size", 100)
        )
        
        # Load strategies into position monitor
        for strategy_name, strategy_config in self.config.get("strategies", {}).items():
            self.position_monitor.load_strategy(
                strategy_name.replace("_", " ").title(),
                strategy_config
            )
        
        # Market hours configuration
        trading_config = self.config.get("trading", {})
        market_hours = trading_config.get("market_hours", {})
        self.market_timezone = pytz.timezone(market_hours.get("timezone", "America/New_York"))
        self.market_open = dt_time.fromisoformat(market_hours.get("start", "09:30"))
        self.market_close = dt_time.fromisoformat(market_hours.get("end", "16:00"))
        
        # Scheduler
        self.scheduler = BackgroundScheduler()
        
        # State
        self.is_running = False
        self.last_scan_time = None
        
        logger.info("Trading Orchestrator initialized successfully")
    
    def start(self):
        """Start the trading agent"""
        try:
            logger.info("Starting Trading Agent...")
            
            # Check account access
            account = self.alpaca.get_account()
            logger.info(f"Connected to Alpaca - Account Equity: ${account['equity']:,.2f}")
            
            # Schedule tasks
            self._schedule_tasks()
            
            # Start scheduler
            self.scheduler.start()
            self.is_running = True
            
            # Start real-time data collection
            self.realtime_collector.start()
            
            logger.info("✅ Trading Agent is now LIVE")
            self.alert_manager.send_alert(
                "system_startup",
                "Trading agent started successfully",
                "info"
            )
            
            # Run initial scan if market is open
            if self.is_market_open():
                self.run_trading_cycle()
        
        except Exception as e:
            logger.error(f"Error starting trading agent: {e}")
            self.alert_manager.alert_system_error("startup_error", str(e))
            raise
    
    def stop(self):
        """Stop the trading agent"""
        try:
            logger.info("Stopping Trading Agent...")
            
            self.is_running = False
            
            # Stop real-time data collection
            self.realtime_collector.stop()
            
            # Stop scheduler
            self.scheduler.shutdown()
            
            logger.info("✅ Trading Agent stopped")
            self.alert_manager.send_alert(
                "system_shutdown",
                "Trading agent stopped",
                "info"
            )
        
        except Exception as e:
            logger.error(f"Error stopping trading agent: {e}")
    
    def _schedule_tasks(self):
        """Schedule recurring tasks"""
        # Every 5 minutes during market hours: scan and monitor
        self.scheduler.add_job(
            self.run_trading_cycle,
            'cron',
            day_of_week='mon-fri',
            hour='9-16',
            minute='*/5',
            timezone=self.market_timezone,
            id='trading_cycle'
        )
        
        # Every 1 minute during market hours: monitor positions
        self.scheduler.add_job(
            self.monitor_positions,
            'cron',
            day_of_week='mon-fri',
            hour='9-16',
            minute='*',
            timezone=self.market_timezone,
            id='position_monitor'
        )
        
        # Daily after market close: analyze trades and learn
        self.scheduler.add_job(
            self.daily_analysis,
            'cron',
            day_of_week='mon-fri',
            hour='17',
            minute='0',
            timezone=self.market_timezone,
            id='daily_analysis'
        )
        
        # Weekly Sunday evening: deep learning analysis
        self.scheduler.add_job(
            self.weekly_learning,
            'cron',
            day_of_week='sun',
            hour='20',
            minute='0',
            timezone=self.market_timezone,
            id='weekly_learning'
        )
        
        logger.info("Scheduled tasks configured")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now(self.market_timezone)
            current_time = now.time()
            
            # Check if weekday
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Check market hours
            return self.market_open <= current_time <= self.market_close
        
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def run_trading_cycle(self):
        """Main trading cycle: scan, generate signals, execute trades"""
        if not self.is_market_open():
            logger.debug("Market is closed, skipping trading cycle")
            return
        
        try:
            logger.info("=" * 80)
            logger.info("Running Trading Cycle")
            logger.info("=" * 80)
            
            # Get account info
            account = self.alpaca.get_account()
            account_balance = float(account["equity"])
            
            # 🚨 PDT COMPLIANCE CHECK
            pdt_info = self.pdt_manager.get_pdt_status()
            logger.info(f"PDT Status: {pdt_info.status.value} ({pdt_info.day_trades_used}/{pdt_info.max_day_trades} day trades)")
            
            if not pdt_info.can_trade:
                logger.warning(f"🚨 Trading suspended: {pdt_info.suspension_reason}")
                # Send PDT alert
                self.alert_manager.send_alert(
                    "PDT Compliance",
                    f"Trading suspended: {pdt_info.suspension_reason}",
                    "warning"
                )
                return
            
            # Log PDT warnings
            pdt_warnings = self.pdt_manager.get_pdt_warnings()
            for warning in pdt_warnings:
                logger.warning(warning)
                self.alert_manager.send_alert("PDT Warning", warning, "warning")
            
            # Check risk summary
            risk_summary = self.risk_manager.get_risk_summary(account_balance)
            logger.info(f"Risk Summary: {risk_summary}")
            
            # Check if we can trade today (PDT-compliant limits)
            if pdt_info.is_pdt_account:
                # PDT accounts limited to 1 position per day
                if risk_summary.get("daily_trades_remaining", 0) <= 0:
                    logger.warning("PDT account: Daily trade limit reached (1 position per day)")
                    return
            else:
                # Non-PDT accounts use standard limits
                if risk_summary.get("daily_trades_remaining", 0) <= 0:
                    logger.warning("Daily trade limit reached")
                    return
            
            # Generate signals
            logger.info("Scanning for trading signals...")
            signals = self.signal_generator.get_best_signals(max_signals=5)
            
            if not signals:
                logger.info("No trading signals found")
                return
            
            logger.info(f"Found {len(signals)} potential signals")
            
            # Execute top signals (PDT-compliant limits)
            max_signals = 1 if pdt_info.is_pdt_account else 3  # PDT accounts: 1 position per day
            
            for i, signal in enumerate(signals[:max_signals], 1):
                logger.info(f"\n[Signal {i}] {signal['symbol']} - {signal['strategy_name']}")
                logger.info(f"  Quality: {signal['signal_quality']:.1f}/100")
                logger.info(f"  Max Profit: ${signal['max_profit']:.2f}")
                logger.info(f"  Max Loss: ${signal['max_loss']:.2f}")
                logger.info(f"  P(Profit): {signal['probability_of_profit']*100:.1f}%")
                
                # 🚨 PDT Compliance Check before execution
                can_open, reason = self.pdt_manager.can_open_position(
                    signal['symbol'], 
                    signal['strategy_name']
                )
                
                if not can_open:
                    logger.warning(f"❌ PDT Compliance: Cannot open position - {reason}")
                    continue
                
                # Execute signal
                trade_id = self.trade_executor.execute_signal(signal)
                
                if trade_id:
                    logger.info(f"✅ Trade executed: {trade_id}")
                    
                    # Log PDT status after trade
                    if pdt_info.is_pdt_account:
                        logger.info(f"🚨 PDT Status: Position opened - must hold overnight")
                    
                    self.alert_manager.alert_trade_executed(
                        trade_id,
                        signal['symbol'],
                        signal['strategy_name'],
                        signal['execution'].get('contracts', 1)
                    )
                    
                    # PDT accounts can only open 1 position per day
                    if pdt_info.is_pdt_account:
                        logger.info("PDT account: Daily position limit reached (1 per day)")
                        break
                else:
                    logger.warning(f"❌ Failed to execute signal for {signal['symbol']}")
                
                # Pause between executions
                time.sleep(2)
            
            self.last_scan_time = datetime.now()
            logger.info("Trading cycle complete")
        
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.alert_manager.alert_system_error("trading_cycle_error", str(e))
    
    def monitor_positions(self):
        """Monitor open positions and manage exits"""
        if not self.is_market_open():
            return
        
        try:
            logger.debug("Monitoring positions...")
            
            # Get exit signals
            exit_signals = self.position_monitor.monitor_positions()
            
            if not exit_signals:
                logger.debug("No exit signals")
                return
            
            logger.info(f"Processing {len(exit_signals)} exit signals")
            
            # Execute exits (PDT-compliant)
            for signal in exit_signals:
                if signal["action"] == "close":
                    # 🚨 PDT Compliance Check before closing
                    can_close, reason = self.pdt_manager.can_close_position(signal["trade_id"])
                    
                    if not can_close:
                        logger.warning(f"❌ PDT Compliance: Cannot close position - {reason}")
                        # For PDT accounts, we might want to wait until next day
                        if "day trade" in reason.lower():
                            logger.info("Position will be closed tomorrow (PDT compliance)")
                        continue
                    
                    logger.info(f"Closing position: {signal['trade_id']} - {signal['reason']}")
                    
                    success = self.trade_executor.close_trade(
                        signal["trade_id"],
                        signal["reason"]
                    )
                    
                    if success:
                        logger.info(f"✅ Position closed: ${signal['pnl']:.2f}")
                        
                        # Log PDT status after close
                        pdt_info = self.pdt_manager.get_pdt_status()
                        if pdt_info.is_pdt_account and "day trade" in reason.lower():
                            logger.info(f"🚨 PDT Status: Day trade completed - {pdt_info.day_trades_used + 1}/{pdt_info.max_day_trades}")
                        
                        self.alert_manager.alert_position_closed(
                            signal["trade_id"],
                            signal["symbol"],
                            signal["pnl"],
                            signal["reason"]
                        )
                    else:
                        logger.error(f"❌ Failed to close position: {signal['trade_id']}")
                
                elif signal["action"] == "roll":
                    # Rolling is generally not recommended for PDT accounts
                    pdt_info = self.pdt_manager.get_pdt_status()
                    if pdt_info.is_pdt_account:
                        logger.warning("PDT account: Rolling not recommended - may trigger day trade")
                    
                    logger.info(f"Rolling position: {signal['trade_id']} - {signal['reason']}")
                    # Rolling logic would go here
        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def daily_analysis(self):
        """Daily post-market analysis"""
        try:
            logger.info("=" * 80)
            logger.info("Running Daily Analysis")
            logger.info("=" * 80)
            
            # Get performance metrics
            metrics = self.trade_analyzer.calculate_performance_metrics(period_days=30)
            
            if metrics:
                logger.info("📊 Performance (Last 30 Days):")
                logger.info(f"  Total Trades: {metrics['total_trades']}")
                logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
                logger.info(f"  Total P&L: ${metrics['total_pnl']:.2f}")
                logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
                logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                
                if metrics.get("error_counts"):
                    logger.info("  Common Errors:")
                    for error, count in metrics["error_counts"].items():
                        logger.info(f"    - {error}: {count}")
            
            # Analyze recent closed trades
            with self.db.get_session() as session:
                from src.database.models import Trade
                recent_trades = session.query(Trade).filter(
                    Trade.status == "closed",
                    Trade.reason_tags == []
                ).limit(10).all()
                
                for trade in recent_trades:
                    self.trade_analyzer.analyze_trade(trade)
                    session.add(trade)
                
                session.commit()
            
            logger.info("Daily analysis complete")
        
        except Exception as e:
            logger.error(f"Error in daily analysis: {e}")
    
    def weekly_learning(self):
        """Weekly learning and strategy optimization"""
        try:
            logger.info("=" * 80)
            logger.info("Running Weekly Learning")
            logger.info("=" * 80)
            
            # Get learning insights
            insights = self.trade_analyzer.get_learning_insights()
            
            if insights.get("ready_for_learning"):
                logger.info("📚 Learning Insights:")
                
                for rec in insights.get("recommendations", []):
                    logger.info(f"  Issue: {rec['issue']} ({rec['frequency']})")
                    logger.info(f"  Action: {rec['action']}")
                
                # Generate adjustments for each strategy
                for strategy_name in ["Bull Put Spread", "Cash Secured Put", "Iron Condor"]:
                    adjustments = self.strategy_learner.analyze_and_learn(strategy_name)
                    
                    if adjustments:
                        logger.info(f"\n📈 Suggested adjustments for {strategy_name}:")
                        for change in adjustments["changes"]:
                            logger.info(f"  {change['parameter']}: {change['old_value']} → {change['new_value']}")
                        
                        logger.info(f"  Confidence: {adjustments['confidence']*100:.1f}%")
                        logger.info(f"  Reasoning: {', '.join(adjustments['reasoning'])}")
                        
                        # In production, apply adjustments here or queue for review
                        # self.strategy_learner.apply_adjustments(strategy_name, adjustments)
            
            logger.info("Weekly learning complete")
        
        except Exception as e:
            logger.error(f"Error in weekly learning: {e}")
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        try:
            account = self.alpaca.get_account()
            risk_summary = self.risk_manager.get_risk_summary(account["equity"])
            portfolio = self.position_monitor.get_portfolio_summary()
            realtime_stats = self.realtime_collector.get_stats()
            
            return {
                "is_running": self.is_running,
                "market_open": self.is_market_open(),
                "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
                "account": {
                    "equity": account["equity"],
                    "cash": account["cash"],
                    "buying_power": account["buying_power"],
                },
                "risk": risk_summary,
                "portfolio": portfolio,
                "realtime_data": realtime_stats,
            }
        
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}


def main():
    """Main entry point"""
    orchestrator = TradingOrchestrator()
    
    try:
        orchestrator.start()
        
        # Keep running
        while orchestrator.is_running:
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    
    finally:
        orchestrator.stop()


if __name__ == "__main__":
    main()


