#!/usr/bin/env python3
"""
Send Daily Trading Report
Run this script at end of day (4:00 PM CT) to send daily report
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.alerts.daily_report import DailyReportGenerator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Send daily report"""
    print("=" * 80)
    print("📊 Generating Daily Trading Report...")
    print("=" * 80)
    print()
    
    # Get Telegram credentials from environment
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not telegram_token or not chat_id:
        print("⚠️  Telegram not configured")
        print("   Report will be generated but not sent")
        print()
        print("   To enable Telegram:")
        print("   1. Set TELEGRAM_BOT_TOKEN in .env file")
        print("   2. Set TELEGRAM_CHAT_ID in .env file")
        print()
    
    # Generate and send report
    generator = DailyReportGenerator(telegram_token, chat_id)
    success = generator.send_daily_report()
    
    print()
    if success:
        print("✅ Daily report sent via Telegram!")
    else:
        print("✅ Daily report generated (check logs)")
    print("=" * 80)

if __name__ == "__main__":
    main()

