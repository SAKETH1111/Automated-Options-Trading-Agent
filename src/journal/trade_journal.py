"""
Trade Journal Module
Automatic trade logging and lesson tracking
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import Trade


class TradeJournal:
    """
    Automatic trade journal
    Logs trades, tracks lessons, identifies patterns
    """
    
    def __init__(self, db_session: Session):
        """Initialize trade journal"""
        self.db = db_session
        self.journal_file = "trade_journal.jsonl"
        logger.info("Trade Journal initialized")
    
    def log_trade_entry(self, trade: Dict):
        """Log trade entry"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'ENTRY',
            'symbol': trade.get('symbol'),
            'strategy': trade.get('strategy_type'),
            'entry_price': trade.get('entry_credit'),
            'strikes': trade.get('strikes'),
            'max_profit': trade.get('max_profit'),
            'max_loss': trade.get('max_loss'),
            'pop': trade.get('pop'),
            'iv_rank': trade.get('iv_rank'),
            'technical_signal': trade.get('technical_signal'),
            'market_regime': trade.get('market_regime'),
            'reasons': trade.get('reasons', [])
        }
        
        self._write_entry(entry)
        logger.info(f"Trade entry logged: {trade.get('strategy_type')} on {trade.get('symbol')}")
    
    def log_trade_exit(self, trade: Dict, exit_details: Dict):
        """Log trade exit"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'EXIT',
            'symbol': trade.get('symbol'),
            'strategy': trade.get('strategy'),
            'exit_reason': exit_details.get('exit_reason'),
            'pnl': exit_details.get('pnl'),
            'pnl_pct': exit_details.get('pnl_pct'),
            'days_held': exit_details.get('days_held'),
            'lessons': self._extract_lessons(trade, exit_details)
        }
        
        self._write_entry(entry)
        logger.info(f"Trade exit logged: {exit_details.get('exit_reason')}")
    
    def _extract_lessons(self, trade: Dict, exit_details: Dict) -> List[str]:
        """Extract lessons from trade"""
        lessons = []
        
        pnl = exit_details.get('pnl', 0)
        exit_reason = exit_details.get('exit_reason')
        
        # Winning trade lessons
        if pnl > 0:
            if exit_reason == 'TAKE_PROFIT':
                lessons.append("✅ Profit target worked well")
            if exit_reason == 'EXPIRATION':
                lessons.append("✅ Full profit captured at expiration")
            
            lessons.append(f"✅ Strategy {trade.get('strategy')} worked in this market condition")
        
        # Losing trade lessons
        else:
            if exit_reason == 'STOP_LOSS':
                lessons.append("⚠️ Stop loss triggered - review entry criteria")
            if exit_reason == 'TECHNICAL_REVERSAL':
                lessons.append("⚠️ Market reversed - consider tighter stops")
            
            lessons.append(f"⚠️ Review why {trade.get('strategy')} failed")
        
        return lessons
    
    def _write_entry(self, entry: Dict):
        """Write entry to journal file"""
        try:
            with open(self.journal_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Error writing to journal: {e}")
    
    def get_recent_entries(self, limit: int = 20) -> List[Dict]:
        """Get recent journal entries"""
        try:
            entries = []
            with open(self.journal_file, 'r') as f:
                for line in f:
                    entries.append(json.loads(line))
            
            return entries[-limit:]
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.error(f"Error reading journal: {e}")
            return []
    
    def generate_weekly_review(self) -> str:
        """Generate weekly review report"""
        entries = self.get_recent_entries(limit=100)
        
        # Filter last week
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        week_entries = [e for e in entries if e['timestamp'] >= week_ago]
        
        # Analyze
        trades = [e for e in week_entries if e['event'] == 'EXIT']
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        # Collect all lessons
        all_lessons = []
        for trade in trades:
            all_lessons.extend(trade.get('lessons', []))
        
        report = f"""
Weekly Trade Journal Review
===========================
Week ending: {datetime.now().strftime('%Y-%m-%d')}

Trades: {len(trades)}
Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}% if trades else 0)
Losses: {len(losses)}

Key Lessons:
{chr(10).join(f'  {lesson}' for lesson in all_lessons[:10])}

Action Items:
- Review losing trades
- Identify patterns
- Adjust strategy if needed
"""
        
        return report

