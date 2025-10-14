"""Specialized strategies for SPY and QQQ index ETF options

These strategies are optimized for the unique characteristics of SPY and QQQ:
- Ultra-high liquidity
- Tight bid-ask spreads
- High options volume and open interest
- Predictable volatility patterns
- No earnings surprises (ETFs)
- VIX correlation for risk assessment
"""

import uuid
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional

from loguru import logger

from .base import Strategy, StrategySignal


class SPYQQQBullPutSpreadStrategy(Strategy):
    """
    Optimized Bull Put Spread for SPY/QQQ
    
    Key Differences from Generic Strategy:
    - Tighter spreads due to higher liquidity
    - Can use wider widths ($10-$15 vs $5-$10)
    - Better fill quality expected
    - More aggressive delta targets possible
    - Can scale into larger positions
    """
    
    def __init__(self, config: Dict):
        super().__init__("SPY/QQQ Bull Put Spread", config)
        
        # Strategy parameters
        self.dte_range = config.get("dte_range", [30, 45])
        self.short_delta_range = config.get("short_delta_range", [-0.25, -0.20])
        self.width_range = config.get("width_range", [5, 10, 15])
        self.min_credit = config.get("min_credit", 0.40)
        self.max_risk_reward = config.get("max_risk_reward", 3.5)
        
        # SPY/QQQ specific
        self.min_premium_dollars = config.get("min_premium_dollars", 40)
        self.max_spread_cost_pct = config.get("max_spread_cost_pct", 3.0)
        
        # Exit parameters
        self.take_profit_pct = config.get("take_profit_pct", 50)
        self.stop_loss_pct = config.get("stop_loss_pct", 100)
        
        # Filters
        self.min_iv_rank = config.get("min_iv_rank", 20)
        self.min_oi = config.get("min_oi", 500)
        self.min_volume = config.get("min_volume", 200)
        self.max_spread_pct = config.get("max_spread_pct", 5.0)
    
    def generate_signals(
        self,
        symbol: str,
        stock_data: Dict,
        options_chain: List[Dict]
    ) -> List[StrategySignal]:
        """Generate optimized SPY/QQQ Bull Put Spread signals"""
        
        if not self.enabled:
            return []
        
        # Only trade SPY and QQQ
        if symbol not in ["SPY", "QQQ"]:
            return []
        
        try:
            signals = []
            
            # Check market conditions
            if not self._check_market_conditions(symbol, stock_data):
                return []
            
            # Filter: IV Rank (can be lower for indices)
            iv_rank = stock_data.get("iv_rank", 50)
            if iv_rank < self.min_iv_rank:
                logger.debug(f"{symbol}: IV Rank too low ({iv_rank})")
                return []
            
            # Filter options by DTE
            options = self._filter_by_dte(
                options_chain,
                self.dte_range[0],
                self.dte_range[1]
            )
            
            if not options:
                return []
            
            # Filter: Only puts
            puts = [opt for opt in options if opt["option_type"] == "put"]
            
            if not puts:
                return []
            
            # Filter by liquidity (stricter for SPY/QQQ)
            liquid_puts = self._filter_by_liquidity(
                puts,
                self.min_oi,
                self.min_volume,
                self.max_spread_pct
            )
            
            if not liquid_puts:
                return []
            
            # Group by expiration
            expirations = {}
            for put in liquid_puts:
                exp_date = put["expiration"].date()
                if exp_date not in expirations:
                    expirations[exp_date] = []
                expirations[exp_date].append(put)
            
            # For each expiration, find optimal spreads
            for exp_date, exp_puts in expirations.items():
                exp_puts_sorted = sorted(exp_puts, key=lambda x: x["strike"], reverse=True)
                
                # Find short leg
                for short_put in exp_puts_sorted:
                    short_delta = short_put["delta"]
                    
                    if not (self.short_delta_range[0] <= short_delta <= self.short_delta_range[1]):
                        continue
                    
                    short_strike = short_put["strike"]
                    
                    # Try different widths (SPY/QQQ can handle wider spreads)
                    for width in self.width_range:
                        target_long_strike = short_strike - width
                        
                        # Find closest long put
                        long_put = min(
                            [p for p in exp_puts_sorted if p["strike"] <= target_long_strike],
                            key=lambda x: abs(x["strike"] - target_long_strike),
                            default=None
                        )
                        
                        if not long_put:
                            continue
                        
                        # Calculate spread metrics
                        metrics = self._calculate_spread_metrics(short_put, long_put)
                        
                        if not metrics:
                            continue
                        
                        # SPY/QQQ specific filters
                        if not self._passes_spy_qqq_filters(metrics, stock_data):
                            continue
                        
                        # Build signal
                        signal = self._build_signal(
                            symbol, stock_data, short_put, long_put, metrics
                        )
                        
                        if signal:
                            signals.append(signal)
                            break
                    
                    if signals:
                        break
            
            # Sort by signal quality
            signals.sort(key=lambda x: x.signal_quality, reverse=True)
            return signals[:3]
        
        except Exception as e:
            logger.error(f"Error generating SPY/QQQ Bull Put Spread signals for {symbol}: {e}")
            return []
    
    def _check_market_conditions(self, symbol: str, stock_data: Dict) -> bool:
        """Check if market conditions are suitable for entry"""
        try:
            # Check VIX if available (for SPY)
            vix = stock_data.get("vix")
            if vix and vix > 35:
                logger.warning(f"{symbol}: VIX too high ({vix:.1f}) - skipping")
                return False
            
            # Check time of day (prefer morning entries)
            now = datetime.now()
            market_open = dt_time(9, 30)
            late_morning = dt_time(11, 30)
            
            if market_open <= now.time() <= late_morning:
                logger.debug(f"{symbol}: Good entry time (morning)")
                return True
            
            return True  # Allow afternoon entries but prefer morning
        
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return True
    
    def _passes_spy_qqq_filters(self, metrics: Dict, stock_data: Dict) -> bool:
        """SPY/QQQ specific filtering"""
        try:
            # Minimum premium in dollars
            credit = metrics.get("net_credit", 0)
            if credit * 100 < self.min_premium_dollars:
                logger.debug(f"Credit too low: ${credit*100:.2f}")
                return False
            
            # Risk/reward check
            if metrics.get("risk_reward", 0) > self.max_risk_reward:
                return False
            
            # Minimum credit check
            if credit < self.min_credit:
                return False
            
            # Spread quality check (should be very tight for SPY/QQQ)
            spread_quality = stock_data.get("spread_pct", 0)
            if spread_quality > self.max_spread_cost_pct:
                logger.debug(f"Spread too wide: {spread_quality:.2f}%")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error in SPY/QQQ filters: {e}")
            return False
    
    def _build_signal(
        self,
        symbol: str,
        stock_data: Dict,
        short_put: Dict,
        long_put: Dict,
        metrics: Dict
    ) -> Optional[StrategySignal]:
        """Build optimized signal for SPY/QQQ"""
        try:
            legs = [
                {
                    "option_symbol": short_put["option_symbol"],
                    "option_type": "put",
                    "strike": short_put["strike"],
                    "expiration": short_put["expiration"].isoformat(),
                    "side": "short",
                    "qty": 1,
                    "delta": short_put["delta"],
                    "price": short_put["mid"],
                    "oi": short_put["open_interest"],
                    "volume": short_put["volume"],
                },
                {
                    "option_symbol": long_put["option_symbol"],
                    "option_type": "put",
                    "strike": long_put["strike"],
                    "expiration": long_put["expiration"].isoformat(),
                    "side": "long",
                    "qty": 1,
                    "delta": long_put["delta"],
                    "price": long_put["mid"],
                    "oi": long_put["open_interest"],
                    "volume": long_put["volume"],
                }
            ]
            
            params = {
                "symbol": symbol,
                "dte": short_put["dte"],
                "short_delta": short_put["delta"],
                "long_delta": long_put["delta"],
                "width": metrics["width"],
                "short_strike": short_put["strike"],
                "long_strike": long_put["strike"],
                "entry_type": "morning" if datetime.now().hour < 12 else "afternoon",
            }
            
            market_snapshot = {
                "stock_price": stock_data["price"],
                "vix": stock_data.get("vix"),
                "iv_rank": stock_data.get("iv_rank", 50),
                "short_iv": short_put["iv"],
                "long_iv": long_put["iv"],
                "short_oi": short_put["open_interest"],
                "long_oi": long_put["open_interest"],
                "short_volume": short_put["volume"],
                "long_volume": long_put["volume"],
                "underlying_spread": stock_data.get("spread_pct", 0),
            }
            
            filters_passed = {
                "iv_rank": True,
                "liquidity": True,
                "delta_range": True,
                "credit": True,
                "spread_quality": True,
                "spy_qqq_specific": True,
            }
            
            signal_quality = self._calculate_signal_quality(
                metrics, stock_data, filters_passed
            )
            
            # Boost quality for SPY/QQQ due to liquidity
            signal_quality = min(signal_quality * 1.1, 100)
            
            signal = StrategySignal(
                signal_id=str(uuid.uuid4()),
                strategy_name=self.name,
                symbol=symbol,
                action="open",
                legs=legs,
                params=params,
                market_snapshot=market_snapshot,
                max_profit=metrics["max_profit"],
                max_loss=metrics["max_loss"],
                probability_of_profit=metrics["probability_of_profit"],
                expected_credit=metrics["net_credit"],
                risk_reward_ratio=metrics["risk_reward"],
                delta_exposure=metrics["delta_exposure"],
                theta_exposure=metrics["theta_exposure"],
                vega_exposure=metrics["vega_exposure"],
                signal_quality=signal_quality,
                filters_passed=filters_passed,
                reason=f"{symbol} Bull Put Spread: {short_put['delta']:.2f}Δ @ ${short_put['strike']:.0f}/${long_put['strike']:.0f}, {short_put['dte']}DTE, ${metrics['net_credit']*100:.0f} credit",
            )
            
            return signal
        
        except Exception as e:
            logger.error(f"Error building signal: {e}")
            return None
    
    def should_exit(
        self,
        trade: Dict,
        current_positions: List[Dict],
        current_pnl: float,
        current_pnl_pct: float
    ) -> Optional[Dict]:
        """SPY/QQQ optimized exit logic"""
        try:
            params = trade.get("params", {})
            entry_credit = trade.get("execution", {}).get("fill_credit", 0)
            symbol = params.get("symbol", "")
            
            # Take Profit: 50% of max profit
            if current_pnl_pct >= self.take_profit_pct:
                return {
                    "action": "close",
                    "reason": "take_profit",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # SPY/QQQ specific: Quicker take profit if near expiration
            days_held = trade.get("days_held", 0)
            dte_at_entry = params.get("dte", 30)
            current_dte = dte_at_entry - days_held
            
            if current_dte <= 10 and current_pnl_pct >= 30:
                return {
                    "action": "close",
                    "reason": "take_profit_near_expiration",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # Stop Loss: 100% loss (2x credit)
            max_loss_threshold = -entry_credit * 100 * (self.stop_loss_pct / 100)
            if current_pnl <= max_loss_threshold:
                return {
                    "action": "close",
                    "reason": "stop_loss",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # VIX spike exit (if we have VIX data)
            vix_at_entry = trade.get("market_snapshot", {}).get("vix")
            # In production, get current VIX and check for spike
            
            # Expiration approaching
            if current_dte <= 7:
                if current_pnl >= -entry_credit * 50:
                    return {
                        "action": "close",
                        "reason": "expiration_approaching",
                        "pnl": current_pnl,
                        "pnl_pct": current_pnl_pct,
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def should_roll(
        self,
        trade: Dict,
        current_positions: List[Dict],
        options_chain: List[Dict]
    ) -> Optional[Dict]:
        """SPY/QQQ roll logic"""
        try:
            params = trade.get("params", {})
            days_held = trade.get("days_held", 0)
            dte_at_entry = params.get("dte", 30)
            current_dte = dte_at_entry - days_held
            
            # SPY/QQQ: Can roll more aggressively due to liquidity
            if current_dte < 14:
                stock_price = trade.get("market_snapshot", {}).get("stock_price", 0)
                short_strike = params.get("short_strike", 0)
                
                otm_pct = ((stock_price - short_strike) / stock_price) * 100
                
                # Roll if less than 5% OTM
                if otm_pct < 5:
                    return {
                        "action": "roll",
                        "reason": "threatened_near_expiration",
                        "current_dte": current_dte,
                        "otm_pct": otm_pct,
                        "can_collect_credit": True,  # SPY/QQQ usually can
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking roll conditions: {e}")
            return None










