#!/usr/bin/env python3
"""
Unified Telegram Bot for Trading Agent
Combines trading control commands + daily report commands
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
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = """
🚀 *Trading Agent Bot*

Welcome! I'll help you monitor and control your trading agent.

*📊 Reports & Analysis:*
/report - Generate full daily report
/summary - Quick daily summary
/help - Show all commands

*💼 Trading Control:*
/status - Get current status
/positions - View open positions
/pnl - Check P&L
/risk - View risk metrics
/stop - Stop trading
/resume - Resume trading

*🎯 System:*
/help - Show this message

You'll also receive automatic daily reports at 4:00 PM CT!
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
📊 *Trading Agent Bot - All Commands*

*REPORTS:*
/report - Comprehensive daily report
  • Data collection stats
  • Trading activity & reasons
  • Open positions
  • Market analysis
  • Tomorrow's outlook
  • Weekly performance

/summary - Same as /report

*TRADING:*
/status - Current agent status
/positions - View all open positions
/pnl - Profit & Loss summary
/risk - Risk metrics

*CONTROL:*
/stop - Stop automated trading
/resume - Resume trading

*SYSTEM:*
/help - This message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*Automatic Daily Reports:*
• Sent at 4:00 PM CT (Mon-Fri)
• Comprehensive end-of-day analysis
"""
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report and /summary commands"""
        status_msg = await update.message.reply_text("📊 Generating report... ⏳")
        
        try:
            # Generate report
            report = self.report_generator.generate_daily_report()
            
            # Delete status message
            await status_msg.delete()
            
            # Send report (split if too long)
            max_length = 4000
            if len(report) <= max_length:
                await update.message.reply_text(f"```\n{report}\n```", parse_mode='Markdown')
            else:
                chunks = [report[i:i+max_length] for i in range(0, len(report), max_length)]
                for chunk in chunks:
                    await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
            
            logger.info(f"Report sent to user {update.effective_user.id}")
        
        except Exception as e:
            await status_msg.edit_text(f"❌ Error generating report: {e}")
            logger.error(f"Error generating report: {e}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Get account info
            account = self.alpaca.get_account()
            equity = float(account.get('equity', 0))
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            
            # Get open positions count
            with self.db.get_session() as session:
                open_positions = session.query(Trade).filter(Trade.status == 'open').count()
            
            # Check data collector
            import subprocess
            collector_running = False
            try:
                result = subprocess.run(
                    ['pgrep', '-f', 'start_simple.py'],
                    capture_output=True,
                    text=True
                )
                collector_running = bool(result.stdout.strip())
            except:
                pass
            
            status_icon = "✅" if collector_running else "⚠️"
            
            status_message = f"""
📊 *Trading Agent Status*

🤖 Data Collector: {status_icon} {'RUNNING' if collector_running else 'STOPPED'}
💰 Equity: ${equity:,.2f}
💵 Cash: ${cash:,.2f}
💳 Buying Power: ${buying_power:,.2f}
📈 Open Positions: {open_positions}

⏰ {datetime.now().strftime('%Y-%m-%d %I:%M %p')}
"""
            await update.message.reply_text(status_message, parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting status: {e}")
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
            await update.message.reply_text(f"❌ Error getting positions: {e}")
            logger.error(f"Error in positions command: {e}")
    
    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command"""
        try:
            with self.db.get_session() as session:
                # Open positions P&L
                open_trades = session.query(Trade).filter(Trade.status == 'open').all()
                open_pnl = sum(trade.pnl for trade in open_trades)
                
                # Today's closed trades
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                closed_today = session.query(Trade)\
                    .filter(Trade.timestamp_exit >= today_start)\
                    .filter(Trade.status == 'closed')\
                    .all()
                today_pnl = sum(trade.pnl for trade in closed_today)
                
                # All-time closed
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
            await update.message.reply_text(f"❌ Error getting P&L: {e}")
            logger.error(f"Error in pnl command: {e}")
    
    async def risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command"""
        try:
            account = self.alpaca.get_account()
            equity = float(account.get('equity', 0))
            
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter(Trade.status == 'open').all()
                
                total_risk = sum(trade.risk.get('max_loss', 0) for trade in open_trades)
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
            await update.message.reply_text(f"❌ Error getting risk metrics: {e}")
            logger.error(f"Error in risk command: {e}")
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        message = """
⚠️ *Manual Trading Control*

The simple data collector doesn't have stop/resume functionality.

To stop data collection:
```
pkill -f start_simple.py
```

To restart:
```
./start_simple.py &
```

Note: Full trading orchestrator with stop/resume is not yet implemented.
"""
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        await self.stop_command(update, context)  # Same message


def main():
    """Run the unified bot"""
    print("=" * 80)
    print("🤖 Starting Unified Trading Bot...")
    print("=" * 80)
    print()
    print("Available commands:")
    print("  Reports: /report, /summary")
    print("  Trading: /status, /positions, /pnl, /risk")
    print("  Control: /stop, /resume")
    print("  Help: /start, /help")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        # Create bot instance
        bot = UnifiedTradingBot()
        
        # Create application
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", bot.start_command))
        app.add_handler(CommandHandler("help", bot.help_command))
        app.add_handler(CommandHandler("report", bot.report_command))
        app.add_handler(CommandHandler("summary", bot.report_command))
        app.add_handler(CommandHandler("status", bot.status_command))
        app.add_handler(CommandHandler("positions", bot.positions_command))
        app.add_handler(CommandHandler("pnl", bot.pnl_command))
        app.add_handler(CommandHandler("risk", bot.risk_command))
        app.add_handler(CommandHandler("stop", bot.stop_command))
        app.add_handler(CommandHandler("resume", bot.resume_command))
        
        # Start the bot
        logger.info("Unified Telegram bot started")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise


if __name__ == "__main__":
    main()

