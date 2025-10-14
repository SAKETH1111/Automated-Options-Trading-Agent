"""
Generic Bull Put Spread Strategy
Works with any symbol - adapts parameters based on stock price and account tier
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from .base import Strategy, StrategySignal


class BullPutSpreadStrategy(Strategy):
    """
    Bull Put Spread Strategy for all symbols
    Automatically adapts to stock price and account size
    """
    
    def __init__(self, config: Dict):
        super().__init__("Bull Put Spread", config)
        
        # Strategy parameters (can be overridden by config)
        self.dte_range = config.get("dte_range", [25, 45])
        self.short_delta_range = config.get("short_delta_range", [-0.30, -0.20])
        self.min_credit = config.get("min_credit", 0.25)  # Lower for cheaper stocks
        self.max_risk_reward = config.get("max_risk_reward", 4.0)
        
        # Exit parameters
        self.take_profit_pct = config.get("take_profit_pct", 50)
        self.stop_loss_pct = config.get("stop_loss_pct", 100)
        
        # Filters
        self.min_iv_rank = config.get("min_iv_rank", 25)
        self.min_oi = config.get("min_oi", 100)
        self.min_volume = config.get("min_volume", 50)
        self.max_spread_pct = config.get("max_spread_pct", 10.0)
        
        logger.info(f"BullPutSpreadStrategy initialized (enabled: {self.enabled})")
    
    def generate_signals(
        self,
        symbol: str,
        stock_data: Dict,
        options_chain: List[Dict]
    ) -> List[StrategySignal]:
        """Generate Bull Put Spread signals for any symbol"""
        
        if not self.enabled:
            return []
        
        try:
            signals = []
            
            # Check market conditions
            if not self._check_market_conditions(symbol, stock_data):
                return []
            
            stock_price = stock_data.get('price', 0)
            iv_rank = stock_data.get('iv_rank', 50)
            
            # Filter: IV Rank
            if iv_rank < self.min_iv_rank:
                logger.debug(f"{symbol}: IV Rank too low ({iv_rank:.1f} < {self.min_iv_rank})")
                return []
            
            # Determine spread width based on stock price
            if stock_price < 50:
                spread_widths = [2, 3]
            elif stock_price < 100:
                spread_widths = [3, 5]
            elif stock_price < 250:
                spread_widths = [5, 10]
            else:
                spread_widths = [10, 15]
            
            # Filter options by DTE
            filtered_options = [
                opt for opt in options_chain
                if self.dte_range[0] <= opt.get('dte', 0) <= self.dte_range[1]
                and opt.get('option_type') == 'put'
            ]
            
            if not filtered_options:
                logger.debug(f"{symbol}: No options in DTE range")
                return []
            
            # Group by expiration
            by_expiration = {}
            for opt in filtered_options:
                exp = opt.get('expiration')
                if exp not in by_expiration:
                    by_expiration[exp] = []
                by_expiration[exp].append(opt)
            
            # Generate signals for each expiration
            for expiration, opts in by_expiration.items():
                # Sort by strike
                opts.sort(key=lambda x: x['strike'])
                
                # Try different spread widths
                for width in spread_widths:
                    signal = self._find_best_spread(symbol, stock_data, opts, width, expiration)
                    if signal:
                        signals.append(signal)
            
            # Sort by signal quality
            signals.sort(key=lambda s: s.signal_quality, reverse=True)
            
            # Return top 3 signals
            return signals[:3]
        
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _check_market_conditions(self, symbol: str, stock_data: Dict) -> bool:
        """Check if market conditions are favorable"""
        try:
            # Check spread
            spread_pct = stock_data.get('spread_pct', 0)
            if spread_pct > 1.0:
                logger.debug(f"{symbol}: Spread too wide ({spread_pct:.2f}%)")
                return False
            
            # Check volume
            volume = stock_data.get('volume', 0)
            if volume < 100000:
                logger.debug(f"{symbol}: Volume too low ({volume:,})")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return False
    
    def _find_best_spread(
        self,
        symbol: str,
        stock_data: Dict,
        options: List[Dict],
        width: float,
        expiration: datetime
    ) -> Optional[StrategySignal]:
        """Find the best bull put spread for given width"""
        try:
            stock_price = stock_data['price']
            
            # Find short put (target delta in range)
            short_candidates = [
                opt for opt in options
                if self.short_delta_range[0] <= opt.get('delta', 0) <= self.short_delta_range[1]
            ]
            
            if not short_candidates:
                return None
            
            best_signal = None
            best_quality = 0
            
            for short_put in short_candidates:
                short_strike = short_put['strike']
                
                # Find long put (width away)
                long_strike = short_strike - width
                long_put = next((opt for opt in options if abs(opt['strike'] - long_strike) < 0.5), None)
                
                if not long_put:
                    continue
                
                # Check liquidity
                if short_put.get('open_interest', 0) < self.min_oi:
                    continue
                if short_put.get('volume', 0) < self.min_volume:
                    continue
                if short_put.get('bid_ask_spread_pct', 0) > self.max_spread_pct:
                    continue
                
                # Calculate spread metrics
                short_mid = short_put['mid']
                long_mid = long_put['mid']
                credit = short_mid - long_mid
                
                if credit < self.min_credit:
                    continue
                
                max_profit = credit * 100  # Per contract
                max_loss = (width - credit) * 100
                risk_reward = max_loss / max_profit if max_profit > 0 else 999
                
                if risk_reward > self.max_risk_reward:
                    continue
                
                # Probability of profit (using delta as approximation)
                pop = abs(short_put.get('delta', 0.25))  # Delta approximates PoP
                
                # Calculate signal quality (0-100)
                quality = self._calculate_quality(
                    iv_rank=stock_data.get('iv_rank', 50),
                    pop=pop,
                    risk_reward=risk_reward,
                    liquidity=short_put.get('open_interest', 0),
                    spread_pct=short_put.get('bid_ask_spread_pct', 0)
                )
                
                if quality > best_quality:
                    best_quality = quality
                    
                    # Create signal
                    best_signal = StrategySignal(
                        signal_id=str(uuid.uuid4()),
                        strategy_name="bull_put_spread",
                        symbol=symbol,
                        action="open",
                        timestamp=datetime.now(),
                        legs=[
                            {
                                'action': 'sell',
                                'option_type': 'put',
                                'strike': short_strike,
                                'quantity': 1,
                                'price': short_mid
                            },
                            {
                                'action': 'buy',
                                'option_type': 'put',
                                'strike': long_strike,
                                'quantity': 1,
                                'price': long_mid
                            }
                        ],
                        params={
                            'dte': short_put.get('dte', 35),
                            'expiration': expiration.isoformat() if isinstance(expiration, datetime) else str(expiration),
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'width': width,
                            'short_delta': short_put.get('delta', 0),
                        },
                        market_snapshot={
                            'price': stock_price,
                            'iv_rank': stock_data.get('iv_rank', 50),
                            'spread_pct': stock_data.get('spread_pct', 0),
                            'volume': stock_data.get('volume', 0),
                        },
                        max_profit=max_profit,
                        max_loss=max_loss,
                        probability_of_profit=pop,
                        expected_credit=credit,
                        risk_reward_ratio=risk_reward,
                        delta_exposure=short_put.get('delta', 0) - long_put.get('delta', 0),
                        theta_exposure=short_put.get('theta', 0) - long_put.get('theta', 0),
                        vega_exposure=short_put.get('vega', 0) - long_put.get('vega', 0),
                        signal_quality=quality,
                        reason=f"IV Rank: {stock_data.get('iv_rank', 50):.0f}, PoP: {pop*100:.0f}%, R:R: {risk_reward:.1f}",
                        notes=f"Credit: ${credit:.2f}, Spread: {short_strike:.0f}/{long_strike:.0f}"
                    )
            
            return best_signal
        
        except Exception as e:
            logger.error(f"Error finding best spread: {e}")
            return None
    
    def _calculate_quality(
        self,
        iv_rank: float,
        pop: float,
        risk_reward: float,
        liquidity: int,
        spread_pct: float
    ) -> float:
        """
        Calculate signal quality score (0-100)
        
        Higher is better
        """
        score = 0.0
        
        # IV Rank (15 points) - higher is better
        score += min(15, (iv_rank / 100) * 15)
        
        # Probability of Profit (20 points) - higher is better
        score += min(20, pop * 20)
        
        # Risk:Reward (15 points) - lower is better
        rr_score = max(0, 15 - (risk_reward * 3))
        score += rr_score
        
        # Liquidity (20 points) - higher OI is better
        liquidity_score = min(20, (liquidity / 500) * 20)
        score += liquidity_score
        
        # Bid-Ask Spread (15 points) - tighter is better
        spread_score = max(0, 15 - (spread_pct * 1.5))
        score += spread_score
        
        # Greeks Balance (15 points) - positive theta, manageable delta
        score += 10  # Placeholder - would check actual Greeks balance
        
        return min(100, max(0, score))
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary for serialization"""
        return {
            "signal_id": self.signal_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "legs": self.legs,
            "params": self.params,
            "market_snapshot": self.market_snapshot,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "probability_of_profit": self.probability_of_profit,
            "expected_credit": self.expected_credit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "signal_quality": self.signal_quality,
            "reason": self.reason,
            "notes": self.notes
        }

