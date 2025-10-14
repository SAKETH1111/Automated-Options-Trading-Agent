#!/usr/bin/env python3
"""
Telegram Bot for Daily Reports
Simple bot that responds to /report and /summary commands
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
from loguru import logger

from src.alerts.daily_report import DailyReportGenerator

# Load environment variables
load_dotenv()

# Get credentials
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️  Error: Telegram credentials not found in .env file")
    print("   Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    sys.exit(1)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_message = """
🤖 Welcome to Trading Agent Report Bot!

Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/report - Generate full daily report
/summary - Quick summary (same as /report)
/help - Show this message

The bot also sends automatic daily reports at 4:00 PM CT every trading day.

Use /report anytime to get the latest stats!
"""
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_message = """
📊 Trading Report Bot Commands:

/report - Generate comprehensive daily report
  • Data collection stats
  • Trading activity
  • Open positions
  • Market analysis
  • Tomorrow's outlook
  • Weekly performance

/summary - Same as /report (shortcut)

/help - Show this help message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automatic Reports:
• Daily at 4:00 PM CT (Mon-Fri)
• After market close
• Comprehensive end-of-day analysis

Need support? Check logs on server:
  tail -f logs/telegram_bot.log
"""
    await update.message.reply_text(help_message)


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /report and /summary commands"""
    # Send "generating" message
    status_msg = await update.message.reply_text("📊 Generating report... ⏳")
    
    try:
        # Generate report
        generator = DailyReportGenerator(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        report = generator.generate_daily_report()
        
        # Delete status message
        await status_msg.delete()
        
        # Send report (split if too long)
        max_length = 4000
        if len(report) <= max_length:
            await update.message.reply_text(f"```\n{report}\n```", parse_mode='Markdown')
        else:
            # Split into chunks
            chunks = [report[i:i+max_length] for i in range(0, len(report), max_length)]
            for chunk in chunks:
                await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
        
        logger.info(f"Report sent to user {update.effective_user.id}")
    
    except Exception as e:
        await status_msg.edit_text(f"❌ Error generating report: {e}")
        logger.error(f"Error generating report: {e}")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown commands"""
    await update.message.reply_text(
        "❓ Unknown command. Use /help to see available commands."
    )


def main():
    """Run the bot"""
    print("=" * 80)
    print("🤖 Starting Trading Report Bot...")
    print("=" * 80)
    print()
    print(f"✅ Bot Token: {TELEGRAM_TOKEN[:20]}...")
    print(f"✅ Chat ID: {TELEGRAM_CHAT_ID}")
    print()
    print("Available commands:")
    print("  /start - Welcome message")
    print("  /report - Generate daily report")
    print("  /summary - Generate daily report (alias)")
    print("  /help - Show help")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        # Create application
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("report", report_command))
        app.add_handler(CommandHandler("summary", report_command))  # Alias
        
        # Start the bot
        logger.info("Telegram bot started")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise


if __name__ == "__main__":
    main()

