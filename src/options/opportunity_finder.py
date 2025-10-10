"""
Options Opportunity Finder Module
Identify high-probability options trading opportunities
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import OptionsChain, OptionsOpportunity, ImpliedVolatility
from src.options.greeks import GreeksCalculator


class OpportunityFinder:
    """
    Identify and score options trading opportunities
    Supports: Bull Put Spreads, Iron Condors, Cash-Secured Puts, etc.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize opportunity finder
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.greeks_calc = GreeksCalculator()
        logger.info("Opportunity Finder initialized")
    
    def find_opportunities(
        self,
        symbol: str,
        min_score: float = 60.0
    ) -> List[Dict]:
        """
        Find all opportunities for a symbol
        
        Args:
            symbol: Symbol to analyze
            min_score: Minimum opportunity score (0-100)
            
        Returns:
            List of opportunities
        """
        opportunities = []
        
        # Get IV metrics
        iv_metrics = self._get_iv_metrics(symbol)
        
        if not iv_metrics:
            logger.warning(f"No IV metrics for {symbol}")
            return []
        
        iv_rank = iv_metrics.get('iv_rank', 50)
        
        # Get recent options data
        options = self._get_recent_options(symbol)
        
        if not options:
            logger.warning(f"No options data for {symbol}")
            return []
        
        # Find different strategy opportunities based on IV regime
        if iv_rank > 50:
            # High IV - look for credit spreads
            opportunities.extend(self._find_bull_put_spreads(symbol, options, iv_metrics))
            opportunities.extend(self._find_iron_condors(symbol, options, iv_metrics))
        else:
            # Low IV - look for debit spreads
            opportunities.extend(self._find_debit_spreads(symbol, options, iv_metrics))
        
        # Filter by minimum score
        opportunities = [opp for opp in opportunities if opp['opportunity_score'] >= min_score]
        
        # Sort by score
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        logger.info(f"Found {len(opportunities)} opportunities for {symbol}")
        
        return opportunities
    
    def _find_bull_put_spreads(
        self,
        symbol: str,
        options: List,
        iv_metrics: Dict
    ) -> List[Dict]:
        """Find bull put spread opportunities"""
        opportunities = []
        
        try:
            # Get puts only
            puts = [opt for opt in options if opt.option_type == 'PUT']
            
            # Group by expiration
            expirations = {}
            for put in puts:
                exp_key = put.expiration.strftime('%Y-%m-%d')
                if exp_key not in expirations:
                    expirations[exp_key] = []
                expirations[exp_key].append(put)
            
            # Look for spreads in each expiration
            for exp_date, exp_puts in expirations.items():
                # Sort by strike
                exp_puts.sort(key=lambda x: x.strike)
                
                # Look for short put (higher strike) and long put (lower strike)
                for i in range(len(exp_puts) - 1):
                    short_put = exp_puts[i]
                    
                    # Short put should be slightly OTM (delta around -0.30)
                    if not (0.25 <= abs(short_put.delta or 0) <= 0.35):
                        continue
                    
                    # Find long put (5-10 strikes lower)
                    for j in range(i + 1, min(i + 5, len(exp_puts))):
                        long_put = exp_puts[j]
                        
                        # Calculate spread metrics
                        width = short_put.strike - long_put.strike
                        credit = (short_put.bid or 0) - (long_put.ask or 0)
                        
                        if credit <= 0 or width <= 0:
                            continue
                        
                        max_profit = credit * 100  # Per contract
                        max_loss = (width - credit) * 100
                        
                        if max_loss <= 0:
                            continue
                        
                        risk_reward = max_profit / max_loss if max_loss > 0 else 0
                        
                        # Calculate probability of profit (short put delta)
                        pop = 1 - abs(short_put.delta or 0.30)
                        
                        # Score the opportunity
                        score = self._score_credit_spread(
                            pop=pop,
                            risk_reward=risk_reward,
                            iv_rank=iv_metrics.get('iv_rank', 50),
                            dte=short_put.dte,
                            credit=credit,
                            width=width
                        )
                        
                        if score >= 50:  # Minimum threshold
                            opportunities.append({
                                'symbol': symbol,
                                'strategy_type': 'bull_put_spread',
                                'opportunity_score': score,
                                'confidence': pop,
                                'strikes': [short_put.strike, long_put.strike],
                                'expiration': short_put.expiration,
                                'dte': short_put.dte,
                                'entry_credit': credit,
                                'entry_debit': None,
                                'max_profit': max_profit,
                                'max_loss': max_loss,
                                'breakeven': short_put.strike - credit,
                                'position_delta': (short_put.delta or 0) - (long_put.delta or 0),
                                'position_theta': (short_put.theta or 0) - (long_put.theta or 0),
                                'position_vega': (short_put.vega or 0) - (long_put.vega or 0),
                                'pop': pop,
                                'pop_50': pop * 0.8,  # Estimate for 50% max profit
                                'risk_reward_ratio': risk_reward,
                                'required_margin': max_loss,
                                'return_on_risk': (max_profit / max_loss * 100) if max_loss > 0 else 0,
                                'iv_rank': iv_metrics.get('iv_rank'),
                                'underlying_price': short_put.underlying_price,
                                'trend': 'BULLISH',
                                'reasons': self._get_spread_reasons(score, pop, iv_metrics, 'bull_put'),
                                'timestamp': datetime.utcnow()
                            })
            
        except Exception as e:
            logger.error(f"Error finding bull put spreads: {e}")
        
        return opportunities
    
    def _find_iron_condors(
        self,
        symbol: str,
        options: List,
        iv_metrics: Dict
    ) -> List[Dict]:
        """Find iron condor opportunities"""
        opportunities = []
        
        try:
            # Group by expiration
            expirations = {}
            for opt in options:
                exp_key = opt.expiration.strftime('%Y-%m-%d')
                if exp_key not in expirations:
                    expirations[exp_key] = {'calls': [], 'puts': []}
                
                if opt.option_type == 'CALL':
                    expirations[exp_key]['calls'].append(opt)
                else:
                    expirations[exp_key]['puts'].append(opt)
            
            # Look for iron condors in each expiration
            for exp_date, opts in expirations.items():
                calls = sorted(opts['calls'], key=lambda x: x.strike)
                puts = sorted(opts['puts'], key=lambda x: x.strike)
                
                if len(calls) < 2 or len(puts) < 2:
                    continue
                
                # Find put spread (lower strikes)
                for i in range(len(puts) - 1):
                    short_put = puts[i]
                    
                    # Short put around -0.30 delta
                    if not (0.25 <= abs(short_put.delta or 0) <= 0.35):
                        continue
                    
                    for j in range(i + 1, min(i + 3, len(puts))):
                        long_put = puts[j]
                        
                        # Find call spread (higher strikes)
                        for k in range(len(calls) - 1):
                            short_call = calls[k]
                            
                            # Short call around 0.30 delta
                            if not (0.25 <= abs(short_call.delta or 0) <= 0.35):
                                continue
                            
                            # Short call should be above current price
                            if short_call.strike <= short_put.underlying_price:
                                continue
                            
                            for l in range(k + 1, min(k + 3, len(calls))):
                                long_call = calls[l]
                                
                                # Calculate iron condor metrics
                                put_width = short_put.strike - long_put.strike
                                call_width = long_call.strike - short_call.strike
                                
                                put_credit = (short_put.bid or 0) - (long_put.ask or 0)
                                call_credit = (short_call.bid or 0) - (long_call.ask or 0)
                                
                                total_credit = put_credit + call_credit
                                
                                if total_credit <= 0:
                                    continue
                                
                                max_profit = total_credit * 100
                                max_loss = (max(put_width, call_width) - total_credit) * 100
                                
                                if max_loss <= 0:
                                    continue
                                
                                risk_reward = max_profit / max_loss
                                
                                # Probability of profit (both sides stay OTM)
                                put_pop = 1 - abs(short_put.delta or 0.30)
                                call_pop = 1 - abs(short_call.delta or 0.30)
                                pop = put_pop * call_pop  # Both must stay OTM
                                
                                # Score the opportunity
                                score = self._score_iron_condor(
                                    pop=pop,
                                    risk_reward=risk_reward,
                                    iv_rank=iv_metrics.get('iv_rank', 50),
                                    dte=short_put.dte,
                                    total_credit=total_credit
                                )
                                
                                if score >= 55:
                                    opportunities.append({
                                        'symbol': symbol,
                                        'strategy_type': 'iron_condor',
                                        'opportunity_score': score,
                                        'confidence': pop,
                                        'strikes': [long_put.strike, short_put.strike, 
                                                  short_call.strike, long_call.strike],
                                        'expiration': short_put.expiration,
                                        'dte': short_put.dte,
                                        'entry_credit': total_credit,
                                        'entry_debit': None,
                                        'max_profit': max_profit,
                                        'max_loss': max_loss,
                                        'breakeven': None,  # Two breakevens
                                        'position_delta': ((short_put.delta or 0) - (long_put.delta or 0) +
                                                         (short_call.delta or 0) - (long_call.delta or 0)),
                                        'position_theta': ((short_put.theta or 0) - (long_put.theta or 0) +
                                                         (short_call.theta or 0) - (long_call.theta or 0)),
                                        'position_vega': ((short_put.vega or 0) - (long_put.vega or 0) +
                                                        (short_call.vega or 0) - (long_call.vega or 0)),
                                        'pop': pop,
                                        'pop_50': pop * 0.7,
                                        'risk_reward_ratio': risk_reward,
                                        'required_margin': max_loss,
                                        'return_on_risk': (max_profit / max_loss * 100) if max_loss > 0 else 0,
                                        'iv_rank': iv_metrics.get('iv_rank'),
                                        'underlying_price': short_put.underlying_price,
                                        'trend': 'RANGING',
                                        'reasons': self._get_iron_condor_reasons(score, pop, iv_metrics),
                                        'timestamp': datetime.utcnow()
                                    })
        
        except Exception as e:
            logger.error(f"Error finding iron condors: {e}")
        
        return opportunities
    
    def _find_debit_spreads(
        self,
        symbol: str,
        options: List,
        iv_metrics: Dict
    ) -> List[Dict]:
        """Find debit spread opportunities (for low IV)"""
        # Placeholder for debit spread logic
        # Would implement bull call spreads, bear put spreads, etc.
        return []
    
    def _score_credit_spread(
        self,
        pop: float,
        risk_reward: float,
        iv_rank: float,
        dte: int,
        credit: float,
        width: float
    ) -> float:
        """Score a credit spread opportunity"""
        score = 0
        
        # Probability of profit (40% weight)
        if pop >= 0.70:
            score += 40
        elif pop >= 0.65:
            score += 30
        elif pop >= 0.60:
            score += 20
        else:
            score += 10
        
        # Risk/reward ratio (20% weight)
        if risk_reward >= 0.40:
            score += 20
        elif risk_reward >= 0.30:
            score += 15
        elif risk_reward >= 0.20:
            score += 10
        else:
            score += 5
        
        # IV rank (20% weight)
        if iv_rank >= 70:
            score += 20
        elif iv_rank >= 50:
            score += 15
        else:
            score += 5
        
        # DTE (10% weight)
        if 30 <= dte <= 45:
            score += 10
        elif 20 <= dte <= 60:
            score += 7
        else:
            score += 3
        
        # Credit relative to width (10% weight)
        credit_ratio = credit / width if width > 0 else 0
        if credit_ratio >= 0.35:
            score += 10
        elif credit_ratio >= 0.25:
            score += 7
        else:
            score += 3
        
        return min(100, score)
    
    def _score_iron_condor(
        self,
        pop: float,
        risk_reward: float,
        iv_rank: float,
        dte: int,
        total_credit: float
    ) -> float:
        """Score an iron condor opportunity"""
        score = 0
        
        # Probability of profit (35% weight)
        if pop >= 0.60:
            score += 35
        elif pop >= 0.50:
            score += 25
        elif pop >= 0.40:
            score += 15
        else:
            score += 5
        
        # Risk/reward (25% weight)
        if risk_reward >= 0.35:
            score += 25
        elif risk_reward >= 0.25:
            score += 18
        elif risk_reward >= 0.15:
            score += 10
        else:
            score += 5
        
        # IV rank (25% weight)
        if iv_rank >= 70:
            score += 25
        elif iv_rank >= 50:
            score += 18
        else:
            score += 8
        
        # DTE (10% weight)
        if 30 <= dte <= 45:
            score += 10
        elif 20 <= dte <= 60:
            score += 7
        else:
            score += 3
        
        # Total credit (5% weight)
        if total_credit >= 1.0:
            score += 5
        elif total_credit >= 0.5:
            score += 3
        else:
            score += 1
        
        return min(100, score)
    
    def _get_spread_reasons(
        self,
        score: float,
        pop: float,
        iv_metrics: Dict,
        spread_type: str
    ) -> List[str]:
        """Get reasons for spread opportunity"""
        reasons = []
        
        if pop >= 0.70:
            reasons.append(f"High probability of profit ({pop:.1%})")
        
        iv_rank = iv_metrics.get('iv_rank', 50)
        if iv_rank >= 70:
            reasons.append(f"Very high IV Rank ({iv_rank:.0f}) - excellent for selling premium")
        elif iv_rank >= 50:
            reasons.append(f"High IV Rank ({iv_rank:.0f}) - good for credit spreads")
        
        if score >= 80:
            reasons.append("Excellent risk/reward profile")
        elif score >= 70:
            reasons.append("Strong setup with good metrics")
        
        return reasons
    
    def _get_iron_condor_reasons(
        self,
        score: float,
        pop: float,
        iv_metrics: Dict
    ) -> List[str]:
        """Get reasons for iron condor opportunity"""
        reasons = []
        
        if pop >= 0.60:
            reasons.append(f"Good probability both sides stay OTM ({pop:.1%})")
        
        iv_rank = iv_metrics.get('iv_rank', 50)
        if iv_rank >= 70:
            reasons.append(f"Very high IV Rank ({iv_rank:.0f}) - ideal for iron condors")
        elif iv_rank >= 50:
            reasons.append(f"High IV Rank ({iv_rank:.0f}) - favorable for neutral strategies")
        
        reasons.append("Market appears range-bound")
        
        if score >= 75:
            reasons.append("Excellent iron condor setup")
        
        return reasons
    
    def _get_iv_metrics(self, symbol: str) -> Optional[Dict]:
        """Get latest IV metrics"""
        try:
            latest = self.db.query(ImpliedVolatility).filter(
                ImpliedVolatility.symbol == symbol
            ).order_by(ImpliedVolatility.timestamp.desc()).first()
            
            if not latest:
                return None
            
            return {
                'iv_rank': latest.iv_rank,
                'iv_percentile': latest.iv_percentile,
                'iv_30': latest.iv_30
            }
        except Exception as e:
            logger.error(f"Error getting IV metrics: {e}")
            return None
    
    def _get_recent_options(self, symbol: str) -> List:
        """Get recent options data"""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=1)
            
            options = self.db.query(OptionsChain).filter(
                OptionsChain.symbol == symbol,
                OptionsChain.timestamp >= cutoff
            ).all()
            
            return options
        except Exception as e:
            logger.error(f"Error getting options: {e}")
            return []
    
    def store_opportunity(self, opportunity: Dict) -> bool:
        """Store opportunity in database"""
        try:
            opp_record = OptionsOpportunity(
                symbol=opportunity['symbol'],
                timestamp=opportunity['timestamp'],
                strategy_type=opportunity['strategy_type'],
                opportunity_score=opportunity['opportunity_score'],
                confidence=opportunity['confidence'],
                strikes=opportunity['strikes'],
                expiration=opportunity['expiration'],
                dte=opportunity['dte'],
                entry_credit=opportunity.get('entry_credit'),
                entry_debit=opportunity.get('entry_debit'),
                max_profit=opportunity['max_profit'],
                max_loss=opportunity['max_loss'],
                breakeven=opportunity.get('breakeven'),
                position_delta=opportunity.get('position_delta'),
                position_theta=opportunity.get('position_theta'),
                position_vega=opportunity.get('position_vega'),
                pop=opportunity.get('pop'),
                pop_50=opportunity.get('pop_50'),
                risk_reward_ratio=opportunity.get('risk_reward_ratio'),
                required_margin=opportunity.get('required_margin'),
                return_on_risk=opportunity.get('return_on_risk'),
                iv_rank=opportunity.get('iv_rank'),
                underlying_price=opportunity.get('underlying_price'),
                trend=opportunity.get('trend'),
                reasons=opportunity.get('reasons')
            )
            
            self.db.add(opp_record)
            self.db.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error storing opportunity: {e}")
            self.db.rollback()
            return False

