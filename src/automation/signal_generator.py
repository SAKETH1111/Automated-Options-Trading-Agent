"""
Automated Signal Generator Module
Generates entry and exit signals automatically based on analysis
"""

from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger

from src.analysis.analyzer import MarketAnalyzer
from src.options import IVTracker, OpportunityFinder


class AutomatedSignalGenerator:
    """
    Automatically generate trading signals
    Combines technical analysis, options analysis, and market regime
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize automated signal generator
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.market_analyzer = MarketAnalyzer(db_session)
        self.iv_tracker = IVTracker(db_session)
        self.opportunity_finder = OpportunityFinder(db_session)
        
        # Signal generation parameters
        self.min_opportunity_score = 65.0
        self.min_confidence = 0.60
        self.min_iv_rank = 50.0
        
        logger.info("Automated Signal Generator initialized")
    
    def generate_entry_signals(
        self,
        symbols: List[str] = ['SPY', 'QQQ']
    ) -> List[Dict]:
        """
        Generate entry signals for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            List of entry signals
        """
        signals = []
        
        for symbol in symbols:
            try:
                logger.info(f"Generating entry signals for {symbol}")
                
                # Get market analysis
                market_analysis = self.market_analyzer.analyze_symbol(symbol, store_results=True)
                
                if 'error' in market_analysis:
                    logger.warning(f"Could not analyze {symbol}: {market_analysis['error']}")
                    continue
                
                # Get IV metrics
                iv_metrics = self.iv_tracker.calculate_iv_metrics(symbol)
                
                if not iv_metrics:
                    logger.warning(f"No IV metrics for {symbol}")
                    continue
                
                # Get trading signals from technical analysis
                tech_signals = self.market_analyzer.generate_trading_signals(symbol)
                
                # Find options opportunities
                opportunities = self.opportunity_finder.find_opportunities(
                    symbol,
                    min_score=self.min_opportunity_score
                )
                
                # Generate signals based on conditions
                for opp in opportunities:
                    # Check if opportunity meets criteria
                    if opp['confidence'] < self.min_confidence:
                        continue
                    
                    # Check IV regime
                    iv_rank = iv_metrics.get('iv_rank', 0)
                    
                    # For credit spreads, want high IV
                    if opp['strategy_type'] in ['bull_put_spread', 'iron_condor']:
                        if iv_rank < self.min_iv_rank:
                            logger.debug(f"IV Rank too low for {opp['strategy_type']}: {iv_rank}")
                            continue
                    
                    # Check technical alignment
                    tech_signal = tech_signals.get('overall_signal', 'NEUTRAL')
                    
                    # Bull put spread needs bullish or neutral bias
                    if opp['strategy_type'] == 'bull_put_spread':
                        if tech_signal == 'BEARISH':
                            logger.debug(f"Technical signal bearish, skipping bull put spread")
                            continue
                    
                    # Create entry signal
                    signal = {
                        'timestamp': datetime.utcnow(),
                        'symbol': symbol,
                        'action': 'ENTRY',
                        'strategy_type': opp['strategy_type'],
                        'opportunity_score': opp['opportunity_score'],
                        'confidence': opp['confidence'],
                        'strikes': opp['strikes'],
                        'expiration': opp['expiration'],
                        'dte': opp['dte'],
                        'entry_credit': opp.get('entry_credit'),
                        'max_profit': opp['max_profit'],
                        'max_loss': opp['max_loss'],
                        'pop': opp['pop'],
                        'iv_rank': iv_rank,
                        'technical_signal': tech_signal,
                        'reasons': opp['reasons'] + [f"Technical: {tech_signal}"],
                        'market_regime': market_analysis.get('regime', {}).get('trend', {}).get('regime'),
                        'underlying_price': market_analysis['current_price']
                    }
                    
                    signals.append(signal)
                    logger.info(f"Generated entry signal: {opp['strategy_type']} on {symbol} "
                               f"(Score: {opp['opportunity_score']:.0f})")
                
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                continue
        
        # Sort by score
        signals.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        logger.info(f"Generated {len(signals)} entry signals")
        
        return signals
    
    def generate_exit_signals(
        self,
        open_positions: List[Dict]
    ) -> List[Dict]:
        """
        Generate exit signals for open positions
        
        Args:
            open_positions: List of open positions
            
        Returns:
            List of exit signals
        """
        exit_signals = []
        
        for position in open_positions:
            try:
                symbol = position['symbol']
                
                # Get current market data
                market_analysis = self.market_analyzer.analyze_symbol(symbol, store_results=False)
                
                if 'error' in market_analysis:
                    continue
                
                current_price = market_analysis['current_price']
                
                # Check exit conditions
                exit_reason = None
                
                # 1. Check expiration (close 1 day before)
                days_to_expiry = (position['expiration'] - datetime.utcnow()).days
                if days_to_expiry <= 1:
                    exit_reason = 'EXPIRATION'
                
                # 2. Check profit target (50% of max profit)
                elif position.get('current_pnl', 0) >= position['max_profit'] * 0.50:
                    exit_reason = 'TAKE_PROFIT'
                
                # 3. Check stop loss (200% of max loss)
                elif position.get('current_pnl', 0) <= -abs(position['max_loss']) * 2.0:
                    exit_reason = 'STOP_LOSS'
                
                # 4. Check technical reversal
                else:
                    tech_signals = self.market_analyzer.generate_trading_signals(symbol)
                    tech_signal = tech_signals.get('overall_signal')
                    
                    # If bull put spread and market turns bearish, exit
                    if position['strategy_type'] == 'bull_put_spread' and tech_signal == 'BEARISH':
                        exit_reason = 'TECHNICAL_REVERSAL'
                
                # Generate exit signal if needed
                if exit_reason:
                    exit_signal = {
                        'timestamp': datetime.utcnow(),
                        'symbol': symbol,
                        'action': 'EXIT',
                        'position_id': position['position_id'],
                        'strategy_type': position['strategy_type'],
                        'exit_reason': exit_reason,
                        'current_price': current_price,
                        'current_pnl': position.get('current_pnl', 0)
                    }
                    
                    exit_signals.append(exit_signal)
                    logger.info(f"Generated exit signal for {symbol}: {exit_reason}")
                
            except Exception as e:
                logger.error(f"Error generating exit signal: {e}")
                continue
        
        logger.info(f"Generated {len(exit_signals)} exit signals")
        
        return exit_signals
    
    def should_trade_now(self) -> bool:
        """
        Check if we should trade right now
        Based on market hours, volatility, etc.
        
        Returns:
            True if should trade
        """
        # Get current time in ET (Eastern Time)
        import pytz
        et_tz = pytz.timezone('America/New_York')
        current_et = datetime.now(pytz.UTC).astimezone(et_tz)
        current_hour = current_et.hour
        current_minute = current_et.minute
        
        # Only trade during market hours (9:30 AM - 4:00 PM ET)
        if current_hour < 9 or current_hour >= 16:
            logger.debug(f"Outside market hours (Current ET: {current_et.strftime('%H:%M')})")
            return False
        
        # Avoid first 15 minutes (high volatility)
        if current_hour == 9 and current_minute < 45:
            logger.debug(f"Too close to market open (Current ET: {current_et.strftime('%H:%M')})")
            return False
        
        # Avoid last 15 minutes (closing volatility)
        if current_hour == 15 and current_minute >= 45:
            logger.debug("Too close to market close")
            return False
        
        return True
    
    def filter_signals_by_risk(
        self,
        signals: List[Dict],
        max_positions: int = 5,
        max_risk_per_trade: float = 0.02,
        available_capital: float = 10000.0
    ) -> List[Dict]:
        """
        Filter signals based on risk management rules
        
        Args:
            signals: List of signals
            max_positions: Maximum number of positions
            max_risk_per_trade: Maximum risk per trade (as % of capital)
            available_capital: Available capital
            
        Returns:
            Filtered signals
        """
        filtered = []
        
        max_risk_amount = available_capital * max_risk_per_trade
        
        for signal in signals:
            # Check max loss
            if signal.get('max_loss', 0) > max_risk_amount:
                logger.debug(f"Signal rejected: max loss ${signal['max_loss']:.2f} > "
                           f"max allowed ${max_risk_amount:.2f}")
                continue
            
            # Check if we have room for more positions
            if len(filtered) >= max_positions:
                logger.debug(f"Signal rejected: already at max positions ({max_positions})")
                break
            
            filtered.append(signal)
        
        logger.info(f"Filtered {len(signals)} signals to {len(filtered)} after risk checks")
        
        return filtered

