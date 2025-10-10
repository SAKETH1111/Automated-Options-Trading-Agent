"""
Telegram Bot for Trading Agent
Control and monitor your trading agent from Telegram
"""

import os
from typing import Optional
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class TradingAgentBot:
    """
    Telegram bot for trading agent
    Commands: /status, /positions, /pnl, /stop, /start, /risk, /help
    """
    
    def __init__(self, db_session, alpaca_client, auto_trader):
        """
        Initialize Telegram bot
        
        Args:
            db_session: Database session
            alpaca_client: Alpaca client
            auto_trader: Automated trader instance
        """
        self.db = db_session
        self.alpaca = alpaca_client
        self.auto_trader = auto_trader
        
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram bot initialized")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "🚀 *Trading Agent Bot*\n\n"
            "Welcome! I'll help you monitor and control your trading agent.\n\n"
            "Available commands:\n"
            "/status - Get current status\n"
            "/positions - View open positions\n"
            "/pnl - Check P&L\n"
            "/risk - View risk metrics\n"
            "/stop - Stop trading\n"
            "/resume - Resume trading\n"
            "/help - Show this message",
            parse_mode='Markdown'
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Get account info
            account = self.alpaca.get_account()
            
            # Get trading status
            status = self.auto_trader.get_status()
            
            # Get open positions
            from src.automation.position_manager import AutomatedPositionManager
            pos_manager = AutomatedPositionManager(self.db, self.alpaca)
            portfolio = pos_manager.get_portfolio_summary()
            
            message = (
                "📊 *Trading Agent Status*\n\n"
                f"🤖 Trading: {'✅ ACTIVE' if status['is_running'] else '⏸️ PAUSED'}\n"
                f"💰 Equity: ${float(account.get('equity', 0)):,.2f}\n"
                f"💵 Cash: ${float(account.get('cash', 0)):,.2f}\n"
                f"📈 Open Positions: {portfolio.get('total_positions', 0)}\n"
                f"💼 Current P&L: ${portfolio.get('current_pnl', 0):+,.2f}\n"
                f"📊 Symbols: {', '.join(status['symbols'])}\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
            logger.error(f"Error in status command: {e}")
    
    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            from src.automation.position_manager import AutomatedPositionManager
            pos_manager = AutomatedPositionManager(self.db, self.alpaca)
            
            positions = pos_manager.get_open_positions()
            
            if not positions:
                await update.message.reply_text("📭 No open positions")
                return
            
            message = f"📊 *Open Positions* ({len(positions)})\n\n"
            
            for i, pos in enumerate(positions, 1):
                pnl_emoji = "🟢" if pos.get('current_pnl', 0) >= 0 else "🔴"
                message += (
                    f"{i}. *{pos['symbol']}* - {pos['strategy_type']}\n"
                    f"   {pnl_emoji} P&L: ${pos.get('current_pnl', 0):+,.2f}\n"
                    f"   📅 Days: {pos['days_held']}\n"
                    f"   💰 Max Profit: ${pos['max_profit']:.2f}\n"
                    f"   ⚠️ Max Loss: ${pos['max_loss']:.2f}\n\n"
                )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
            logger.error(f"Error in positions command: {e}")
    
    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command"""
        try:
            from src.automation.performance_tracker import PerformanceTracker
            tracker = PerformanceTracker(self.db)
            
            daily = tracker.get_daily_pnl()
            weekly = tracker.get_weekly_pnl()
            all_time = tracker.get_all_time_stats()
            
            message = "💰 *Performance Summary*\n\n"
            
            # Today
            if daily:
                pnl_emoji = "🟢" if daily['total_pnl'] >= 0 else "🔴"
                message += (
                    f"📅 *Today:*\n"
                    f"   Trades: {daily['total_trades']}\n"
                    f"   Win Rate: {daily['win_rate']:.1%}\n"
                    f"   {pnl_emoji} P&L: ${daily['total_pnl']:+,.2f}\n\n"
                )
            
            # This Week
            if weekly:
                pnl_emoji = "🟢" if weekly['total_pnl'] >= 0 else "🔴"
                message += (
                    f"📅 *This Week:*\n"
                    f"   Trades: {weekly['total_trades']}\n"
                    f"   Win Rate: {weekly['win_rate']:.1%}\n"
                    f"   {pnl_emoji} P&L: ${weekly['total_pnl']:+,.2f}\n\n"
                )
            
            # All Time
            if all_time and all_time.get('total_trades', 0) > 0:
                pnl_emoji = "🟢" if all_time['total_pnl'] >= 0 else "🔴"
                message += (
                    f"📊 *All Time:*\n"
                    f"   Trades: {all_time['total_trades']}\n"
                    f"   Win Rate: {all_time['win_rate']:.1%}\n"
                    f"   {pnl_emoji} P&L: ${all_time['total_pnl']:+,.2f}\n"
                    f"   Profit Factor: {all_time['profit_factor']:.2f}"
                )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
            logger.error(f"Error in pnl command: {e}")
    
    async def risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command"""
        try:
            from src.risk_management import PortfolioRiskManager, CircuitBreaker
            
            risk_manager = PortfolioRiskManager(self.db, total_capital=10000.0)
            circuit_breaker = CircuitBreaker(self.db, total_capital=10000.0)
            
            metrics = risk_manager.get_portfolio_risk_metrics()
            cb_status = circuit_breaker.check_circuit_breaker()
            
            cb_emoji = "🔴" if cb_status['tripped'] else "🟢"
            risk_emoji = "🟢" if metrics['total_risk_pct'] < 5 else "🟡" if metrics['total_risk_pct'] < 8 else "🔴"
            
            message = (
                "🛡️ *Risk Status*\n\n"
                f"{cb_emoji} Circuit Breaker: {'TRIPPED' if cb_status['tripped'] else 'ACTIVE'}\n"
                f"{risk_emoji} Portfolio Risk: {metrics['total_risk_pct']:.1f}%\n"
                f"💰 Total Risk: ${metrics['total_risk']:,.2f}\n"
                f"✅ Available Risk: ${metrics['available_risk']:,.2f}\n\n"
                f"📊 *Limits:*\n"
                f"   Max Positions: {metrics['limits']['max_positions']}\n"
                f"   Daily Loss Limit: {metrics['limits']['daily_loss_limit']:.1%}\n"
                f"   Max Drawdown: {metrics['limits']['max_drawdown_limit']:.1%}"
            )
            
            if cb_status.get('warnings'):
                message += "\n\n⚠️ *Warnings:*\n"
                for warning in cb_status['warnings']:
                    message += f"   • {warning}\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
            logger.error(f"Error in risk command: {e}")
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        try:
            self.auto_trader.stop_automated_trading()
            await update.message.reply_text(
                "⏸️ *Trading Stopped*\n\n"
                "Automated trading has been paused.\n"
                "Use /resume to restart.",
                parse_mode='Markdown'
            )
            logger.info("Trading stopped via Telegram command")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        try:
            self.auto_trader.is_running = True
            await update.message.reply_text(
                "✅ *Trading Resumed*\n\n"
                "Automated trading has been restarted.",
                parse_mode='Markdown'
            )
            logger.info("Trading resumed via Telegram command")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await update.message.reply_text(
            "🤖 *Trading Agent Commands*\n\n"
            "/status - Get current status\n"
            "/positions - View open positions\n"
            "/pnl - Check P&L (today, week, all-time)\n"
            "/risk - View risk metrics\n"
            "/stop - Stop automated trading\n"
            "/resume - Resume trading\n"
            "/help - Show this message\n\n"
            "Dashboard: http://45.55.150.19:8000",
            parse_mode='Markdown'
        )
    
    def send_notification(self, message: str):
        """Send notification to user"""
        if not self.enabled or not self.chat_id:
            return
        
        try:
            import asyncio
            from telegram import Bot
            
            bot = Bot(token=self.bot_token)
            asyncio.run(bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown'))
            
            logger.info(f"Telegram notification sent: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    def run(self):
        """Start the bot"""
        if not self.enabled:
            logger.error("Telegram bot not configured")
            return
        
        try:
            application = Application.builder().token(self.bot_token).build()
            
            # Add command handlers
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("status", self.status_command))
            application.add_handler(CommandHandler("positions", self.positions_command))
            application.add_handler(CommandHandler("pnl", self.pnl_command))
            application.add_handler(CommandHandler("risk", self.risk_command))
            application.add_handler(CommandHandler("stop", self.stop_command))
            application.add_handler(CommandHandler("resume", self.resume_command))
            application.add_handler(CommandHandler("help", self.help_command))
            
            logger.info("Telegram bot starting...")
            application.run_polling()
            
        except Exception as e:
            logger.error(f"Error running Telegram bot: {e}")

