#!/usr/bin/env python3
"""Test email alerts"""
import sys
import os
sys.path.append('/opt/trading-agent')

from src.alerts import EmailAlertManager

email = EmailAlertManager()
print(f'Email enabled: {email.enabled}')
print(f'Sender: {email.sender_email}')
print(f'Recipient: {email.recipient_email}')

if email.enabled:
    print('\nSending test email...')
    email._send_email(
        '🎉 Trading Agent - Email Alerts Working!',
        '''Hello!

Your trading agent email alerts are configured and working!

You will receive notifications for:
✅ Trade executions
✅ Circuit breaker trips
✅ Position events
✅ Daily summaries

Dashboard: http://45.55.150.19:8000

---
Automated Options Trading Agent
'''
    )
    print('✅ Test email sent!')
else:
    print('❌ Email not configured')

