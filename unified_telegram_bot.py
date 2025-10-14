#!/usr/bin/env python3
"""
Unified Telegram Bot for Trading Agent
All commands: reports + trading control + signal notifications
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
from loguru import logger

from src.alerts.daily_report import DailyReportGenerator
from src.brokers.alpaca_client import AlpacaClient
from src.database.session import get_db
from src.database.models import Trade
from src.signals.signal_logger import SignalLogger

# Load environment variables
load_dotenv()

# Get credentials
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️  Error: Telegram credentials not found in .env file")
    sys.exit(1)


class UnifiedTradingBot:
    """Unified bot with all commands"""
    
    def __init__(self):
        self.alpaca = AlpacaClient()
        self.db = get_db()
        self.report_generator = DailyReportGenerator(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        self.signal_logger = SignalLogger()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = """
🚀 *Trading Agent Bot*

Welcome! Monitor and control your trading agent.

*📊 Reports & Analysis:*
/report - Full daily report
/summary - Quick daily summary
/signals - Recent signals (last 10)

*💼 Trading Control:*
/status - Current status
/positions - Open positions
/pnl - Profit & Loss
/risk - Risk metrics
/ml - ML model status
/pdt - PDT compliance

*🎯 System:*
/help - Show all commands

*⏰ Automatic:*
Daily reports at 4:00 PM CT
Real-time signal notifications
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
📊 *Trading Agent Bot - All Commands*

*REPORTS:*
/report - Comprehensive daily report
/summary - Same as /report
/signals - Recent trading signals

*TRADING:*
/status - Current agent status
/positions - View all positions
/pnl - Profit & Loss
/risk - Risk metrics
/ml - ML model status
/pdt - PDT compliance

*SYSTEM:*
/help - This message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*Automatic Notifications:*
• Daily reports at 4 PM CT
• Signal generation alerts
• Trade execution confirmations
• Position close notifications
"""
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report and /summary commands"""
        status_msg = await update.message.reply_text("📊 Generating report... ⏳")
        
        try:
            report = self.report_generator.generate_daily_report()
            await status_msg.delete()
            
            # Send report (split if too long)
            max_length = 4000
            if len(report) <= max_length:
                await update.message.reply_text(f"```\n{report}\n```", parse_mode='Markdown')
            else:
                chunks = [report[i:i+max_length] for i in range(0, len(report), max_length)]
                for chunk in chunks:
                    await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
            
            logger.info(f"Report sent to user")
        
        except Exception as e:
            await status_msg.edit_text(f"❌ Error: {e}")
            logger.error(f"Error generating report: {e}")
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command - show recent signals"""
        try:
            signals = self.signal_logger.get_recent_signals(limit=10)
            
            if not signals:
                await update.message.reply_text("📊 No signals generated yet")
                return
            
            message = f"📊 *Recent Signals ({len(signals)})*\n\n"
            
            for i, sig in enumerate(signals[:10], 1):
                import pytz
                ct_tz = pytz.timezone('America/Chicago')
                time_ct = sig.timestamp.replace(tzinfo=pytz.UTC).astimezone(ct_tz)
                
                status_emoji = "✅" if sig.status == 'executed' else "📋"
                
                message += f"{i}. {status_emoji} *{sig.symbol}* - {sig.strategy_type}\n"
                message += f"   Quality: {sig.opportunity_score:.0f}/100\n"
                message += f"   PoP: {sig.pop*100:.0f}%, R:R: {sig.risk_reward_ratio:.1f}:1\n"
                message += f"   {time_ct.strftime('%m/%d %H:%M CT')}\n\n"
            
            # Get stats
            stats = self.signal_logger.get_signal_stats()
            message += f"📈 *Stats:* {stats.get('total_signals', 0)} total, "
            message += f"{stats.get('executed_signals', 0)} executed ({stats.get('execution_rate', 0):.0f}%)"
            
            await update.message.reply_text(message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Error in signals command: {e}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            account = self.alpaca.get_account()
            equity = float(account.get('equity', 0))
            cash = float(account.get('cash', 0))
            
            from src.risk_management.pdt_compliance import PDTComplianceManager
            from src.utils.symbol_selector import get_symbols_for_account, get_symbol_info
            
            pdt_manager = PDTComplianceManager(equity)
            pdt_info = pdt_manager.get_pdt_status()
            smart_symbols = get_symbols_for_account(equity)
            symbol_info = get_symbol_info(equity)
            
            # Check collector
            import subprocess
            collector_running = False
            try:
                result = subprocess.run(['pgrep', '-f', 'start_simple.py'], capture_output=True, text=True)
                collector_running = bool(result.stdout.strip())
            except:
                pass
            
            # Get positions
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter(Trade.status == 'open').all()
                open_positions = len(open_trades)
                current_pnl = sum(trade.pnl for trade in open_trades)
                
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                positions_today = session.query(Trade).filter(Trade.timestamp_enter >= today_start).count()
            
            trading_status = "✅ ACTIVE (Scanning Markets)" if collector_running else "⏸️  PAUSED (Not Running)"
            
            # PDT status
            if not pdt_info.is_pdt_account:
                pdt_emoji = "🔵"
                pdt_status = "PDT EXEMPT"
            else:
                pdt_emoji = "🟢" if pdt_info.status.value == "compliant" else "🟡" if pdt_info.status.value == "warning" else "🔴"
                pdt_status = pdt_info.status.value.upper()
            
            import pytz
            ct_tz = pytz.timezone('America/Chicago')
            ct_time = datetime.now(pytz.UTC).astimezone(ct_tz)
            
            status_message = f"""
📊 *Trading Agent Status*

🤖 Trading: {trading_status}
💰 Account: ${equity:,.2f}
💵 Cash: ${cash:,.2f}
📈 Open Positions: {open_positions}
💼 Current P&L: ${current_pnl:+,.2f}

{pdt_emoji} PDT Status: {pdt_status}
⚡ Day Trades: {pdt_info.day_trades_used}/{pdt_info.max_day_trades}
📅 Positions Today: {positions_today}/{'∞' if not pdt_info.is_pdt_account else '1'}

🎯 Account Tier: {symbol_info['tier'].upper()}
📊 Smart Symbols: {', '.join(smart_symbols)}
💲 Max Stock Price: ${symbol_info['max_stock_price']}

⏰ {ct_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
"""
            
            if not collector_running:
                status_message += "\n⚠️ _Data collector not running_"
            
            await update.message.reply_text(status_message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Error in status command: {e}")
    
    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            with self.db.get_session() as session:
                positions = session.query(Trade).filter(Trade.status == 'open').all()
                
                if not positions:
                    await update.message.reply_text("📊 No open positions")
                    return
                
                message = f"📊 *Open Positions ({len(positions)})*\n\n"
                
                for i, trade in enumerate(positions, 1):
                    days_open = (datetime.now() - trade.timestamp_enter).days
                    pnl_emoji = "🟢" if trade.pnl > 0 else "🔴" if trade.pnl < 0 else "⚪"
                    
                    message += f"{i}. *{trade.symbol}* - {trade.strategy}\n"
                    message += f"   {pnl_emoji} P&L: ${trade.pnl:+.2f}\n"
                    message += f"   📅 Days: {days_open}\n"
                    if trade.params.get('dte'):
                        message += f"   ⏱ DTE: {trade.params.get('dte')} days\n"
                    message += "\n"
                
                await update.message.reply_text(message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Error in positions command: {e}")
    
    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command"""
        try:
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter(Trade.status == 'open').all()
                open_pnl = sum(trade.pnl for trade in open_trades)
                
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                closed_today = session.query(Trade).filter(
                    Trade.timestamp_exit >= today_start,
                    Trade.status == 'closed'
                ).all()
                today_pnl = sum(trade.pnl for trade in closed_today)
                
                all_closed = session.query(Trade).filter(Trade.status == 'closed').all()
                total_pnl = sum(trade.pnl for trade in all_closed)
                
                open_emoji = "🟢" if open_pnl > 0 else "🔴" if open_pnl < 0 else "⚪"
                today_emoji = "🟢" if today_pnl > 0 else "🔴" if today_pnl < 0 else "⚪"
                total_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
                
                message = f"""
💰 *Profit & Loss*

*Open Positions:*
{open_emoji} ${open_pnl:+.2f} ({len(open_trades)} positions)

*Today:*
{today_emoji} ${today_pnl:+.2f} ({len(closed_today)} trades closed)

*All-Time:*
{total_emoji} ${total_pnl:+.2f} ({len(all_closed)} trades total)
"""
                await update.message.reply_text(message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Error in pnl command: {e}")
    
    async def risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command"""
        try:
            account = self.alpaca.get_account()
            equity = float(account.get('equity', 0))
            
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter(Trade.status == 'open').all()
                
                total_risk = sum(abs(trade.risk.get('max_loss', 0)) for trade in open_trades if isinstance(trade.risk, dict))
                risk_pct = (total_risk / equity * 100) if equity > 0 else 0
                
                message = f"""
⚠️ *Risk Metrics*

💰 Account Equity: ${equity:,.2f}
📊 Open Positions: {len(open_trades)}
💸 Total Risk: ${total_risk:,.2f}
📈 Risk %: {risk_pct:.1f}%

*Max Risk: 30% of equity*
{'✅ Within limits' if risk_pct <= 30 else '⚠️ HIGH RISK!'}
"""
                await update.message.reply_text(message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Error in risk command: {e}")
    
    async def ml_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ml command"""
        try:
            from pathlib import Path
            
            model_dir = Path("models")
            model_files = list(model_dir.glob("*.pkl")) if model_dir.exists() else []
            
            message = "🤖 *ML Model Status*\n\n"
            
            if model_files:
                message += f"✅ Models Found: {len(model_files)}\n\n"
                message += "*Recent Models:*\n"
                
                sorted_models = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)
                
                for i, model_file in enumerate(sorted_models[:5], 1):
                    import time
                    mod_time = time.ctime(model_file.stat().st_mtime)
                    size_kb = model_file.stat().st_size / 1024
                    message += f"{i}. `{model_file.name}`\n"
                    message += f"   Size: {size_kb:.1f} KB\n"
                    message += f"   Updated: {mod_time}\n\n"
                
                message += "\n*Status:* ML models available"
            else:
                message += "⚠️ No ML models found\n\n"
                message += "*Current Status:* Data collection phase\n"
                message += "Models generated after 30+ trades"
            
            await update.message.reply_text(message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Error in ml command: {e}")
    
    async def pdt_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pdt command"""
        try:
            account = self.alpaca.get_account()
            equity = float(account.get('equity', 0))
            
            from src.risk_management.pdt_compliance import PDTComplianceManager
            pdt_manager = PDTComplianceManager(equity)
            pdt_info = pdt_manager.get_pdt_status()
            
            # Status
            if not pdt_info.is_pdt_account:
                status_emoji = "🔵"
                account_type = "PDT Exempt (≥$25K)"
            else:
                status_emoji = "🟢" if pdt_info.status.value == "compliant" else "🟡" if pdt_info.status.value == "warning" else "🔴"
                account_type = "PDT Restricted (<$25K)"
            
            message = f"""
🚨 *PDT Compliance Status*

{status_emoji} *Account Type:* {account_type}
💰 *Equity:* ${equity:,.2f}

📊 *Day Trade Status:*
   Used: {pdt_info.day_trades_used}/{pdt_info.max_day_trades}
   Status: {pdt_info.status.value.upper()}

🔄 *Trading Limits:*
"""
            
            if pdt_info.is_pdt_account:
                message += f"   ⚠️ Max 1 new position per day\n"
                message += f"   ⚠️ Must hold overnight\n"
                message += f"   ⚠️ Max {pdt_info.max_day_trades} day trades per 5 days\n\n"
                
                if pdt_info.can_trade:
                    message += "✅ *Can open new positions*\n"
                else:
                    message += f"❌ *Cannot trade:* {pdt_info.suspension_reason}\n"
                
                warnings = pdt_manager.get_pdt_warnings()
                if warnings:
                    message += "\n⚠️ *Warnings:*\n"
                    for warning in warnings:
                        message += f"   • {warning}\n"
            else:
                message += "   ✅ No PDT restrictions\n"
                message += "   ✅ Unlimited day trades\n"
                message += "   ✅ Can close same-day positions\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Error in pdt command: {e}")
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        try:
            import subprocess
            
            # Check if agent is running
            result = subprocess.run(['pgrep', '-f', 'main.py'], capture_output=True, text=True)
            
            if not result.stdout.strip():
                await update.message.reply_text("⚠️ Trading agent is not running")
                return
            
            # Stop the agent
            subprocess.run(['pkill', '-f', 'main.py'], check=False)
            
            message = """
⏸️ *Trading Agent Stopped*

The trading agent has been stopped.

*What's stopped:*
• Signal generation
• Trade execution
• Position monitoring
• Scheduled tasks

*Data collector* continues running independently.

To restart: /resume
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            logger.info("Trading agent stopped via Telegram command")
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error stopping agent: {e}")
            logger.error(f"Error in stop command: {e}")
    
    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        try:
            import subprocess
            
            # Check if already running
            result = subprocess.run(['pgrep', '-f', 'main.py'], capture_output=True, text=True)
            
            if result.stdout.strip():
                await update.message.reply_text("✅ Trading agent is already running")
                return
            
            # Start the agent
            subprocess.Popen(
                ['nohup', 'python3', 'main.py'],
                stdout=open('logs/trading_agent.log', 'a'),
                stderr=subprocess.STDOUT,
                cwd='/root/Automated-Options-Trading-Agent'
            )
            
            import time
            time.sleep(2)
            
            # Verify it started
            result = subprocess.run(['pgrep', '-f', 'main.py'], capture_output=True, text=True)
            
            if result.stdout.strip():
                message = """
✅ *Trading Agent Started*

The trading agent is now running.

*Active features:*
• Signal generation
• Paper trade execution
• Position monitoring
• Risk management
• PDT compliance

Check status with: /status
"""
                await update.message.reply_text(message, parse_mode='Markdown')
                logger.info("Trading agent started via Telegram command")
            else:
                await update.message.reply_text("❌ Failed to start agent. Check logs:\n`tail -30 logs/trading_agent.log`", parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error starting agent: {e}")
            logger.error(f"Error in resume command: {e}")


def main():
    """Run the unified bot"""
    print("=" * 80)
    print("🤖 Starting Unified Trading Bot...")
    print("=" * 80)
    print()
    print("Commands: /report, /summary, /signals, /status, /positions, /pnl, /risk, /ml, /pdt")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        bot = UnifiedTradingBot()
        
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", bot.start_command))
        app.add_handler(CommandHandler("help", bot.help_command))
        app.add_handler(CommandHandler("report", bot.report_command))
        app.add_handler(CommandHandler("summary", bot.report_command))
        app.add_handler(CommandHandler("signals", bot.signals_command))
        app.add_handler(CommandHandler("status", bot.status_command))
        app.add_handler(CommandHandler("positions", bot.positions_command))
        app.add_handler(CommandHandler("pnl", bot.pnl_command))
        app.add_handler(CommandHandler("risk", bot.risk_command))
        app.add_handler(CommandHandler("ml", bot.ml_command))
        app.add_handler(CommandHandler("pdt", bot.pdt_command))
        app.add_handler(CommandHandler("stop", bot.stop_command))
        app.add_handler(CommandHandler("resume", bot.resume_command))
        
        logger.info("Unified Telegram bot started")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise


if __name__ == "__main__":
    main()

