"""Iron Condor Strategy Implementation"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from .base import Strategy, StrategySignal


class IronCondorStrategy(Strategy):
    """
    Iron Condor (Market Neutral)
    
    Entry Criteria:
    - Sell OTM put at target delta (e.g., -0.15 to -0.20)
    - Buy OTM put below it for protection
    - Sell OTM call at target delta (e.g., 0.15 to 0.20)
    - Buy OTM call above it for protection
    - Target DTE: 30-45 days
    - High IV Rank preferred (>30)
    - Expect stock to stay range-bound
    
    Exit Criteria:
    - Take profit at 50% of max profit
    - Stop loss at 100% of credit (2x loss)
    - Close if one side is threatened
    """
    
    def __init__(self, config: Dict):
        super().__init__("Iron Condor", config)
        
        # Strategy parameters
        self.dte_range = config.get("dte_range", [30, 45])
        self.short_put_delta_range = config.get("short_put_delta_range", [-0.20, -0.15])
        self.short_call_delta_range = config.get("short_call_delta_range", [0.15, 0.20])
        self.width = config.get("width", 5)
        self.min_credit = config.get("min_credit", 0.50)
        
        # Exit parameters
        self.take_profit_pct = config.get("take_profit_pct", 50)
        self.stop_loss_pct = config.get("stop_loss_pct", 100)
        
        # Filters
        self.min_iv_rank = config.get("min_iv_rank", 30)
        self.min_oi = config.get("min_oi", 100)
        self.min_volume = config.get("min_volume", 50)
        self.max_spread_pct = config.get("max_spread_pct", 10.0)
    
    def generate_signals(
        self,
        symbol: str,
        stock_data: Dict,
        options_chain: List[Dict]
    ) -> List[StrategySignal]:
        """Generate Iron Condor signals"""
        
        if not self.enabled:
            return []
        
        try:
            signals = []
            
            # Filter: IV Rank
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
            
            # Filter by liquidity
            liquid_options = self._filter_by_liquidity(
                options,
                self.min_oi,
                self.min_volume,
                self.max_spread_pct
            )
            
            if not liquid_options:
                return []
            
            # Split into puts and calls
            puts = [opt for opt in liquid_options if opt["option_type"] == "put"]
            calls = [opt for opt in liquid_options if opt["option_type"] == "call"]
            
            if not puts or not calls:
                return []
            
            # Group by expiration
            put_expirations = {}
            for put in puts:
                exp_date = put["expiration"].date()
                if exp_date not in put_expirations:
                    put_expirations[exp_date] = []
                put_expirations[exp_date].append(put)
            
            call_expirations = {}
            for call in calls:
                exp_date = call["expiration"].date()
                if exp_date not in call_expirations:
                    call_expirations[exp_date] = []
                call_expirations[exp_date].append(call)
            
            # Find expirations with both puts and calls
            common_expirations = set(put_expirations.keys()) & set(call_expirations.keys())
            
            for exp_date in common_expirations:
                exp_puts = put_expirations[exp_date]
                exp_calls = call_expirations[exp_date]
                
                # Sort by strike
                exp_puts_sorted = sorted(exp_puts, key=lambda x: x["strike"], reverse=True)
                exp_calls_sorted = sorted(exp_calls, key=lambda x: x["strike"])
                
                # Find short put
                short_put_candidates = [
                    p for p in exp_puts_sorted
                    if self.short_put_delta_range[0] <= p["delta"] <= self.short_put_delta_range[1]
                ]
                
                # Find short call
                short_call_candidates = [
                    c for c in exp_calls_sorted
                    if self.short_call_delta_range[0] <= c["delta"] <= self.short_call_delta_range[1]
                ]
                
                if not short_put_candidates or not short_call_candidates:
                    continue
                
                # Try to build iron condor
                for short_put in short_put_candidates[:3]:
                    for short_call in short_call_candidates[:3]:
                        # Find long put (protection)
                        target_long_put_strike = short_put["strike"] - self.width
                        long_put = min(
                            [p for p in exp_puts_sorted if p["strike"] <= target_long_put_strike],
                            key=lambda x: abs(x["strike"] - target_long_put_strike),
                            default=None
                        )
                        
                        # Find long call (protection)
                        target_long_call_strike = short_call["strike"] + self.width
                        long_call = min(
                            [c for c in exp_calls_sorted if c["strike"] >= target_long_call_strike],
                            key=lambda x: abs(x["strike"] - target_long_call_strike),
                            default=None
                        )
                        
                        if not long_put or not long_call:
                            continue
                        
                        # Calculate metrics
                        metrics = self._calculate_iron_condor_metrics(
                            short_put, long_put, short_call, long_call
                        )
                        
                        if not metrics:
                            continue
                        
                        # Filter: Minimum credit
                        if metrics["net_credit"] < self.min_credit:
                            continue
                        
                        # Build signal
                        signal = self._build_signal(
                            symbol, stock_data,
                            short_put, long_put, short_call, long_call,
                            metrics
                        )
                        
                        if signal:
                            signals.append(signal)
                            break  # Found a good iron condor
                    
                    if signals:
                        break
            
            # Sort by signal quality
            signals.sort(key=lambda x: x.signal_quality, reverse=True)
            return signals[:2]
        
        except Exception as e:
            logger.error(f"Error generating Iron Condor signals for {symbol}: {e}")
            return []
    
    def _calculate_iron_condor_metrics(
        self,
        short_put: Dict,
        long_put: Dict,
        short_call: Dict,
        long_call: Dict
    ) -> Dict:
        """Calculate iron condor metrics"""
        try:
            # Put spread metrics
            put_spread = self._calculate_spread_metrics(short_put, long_put)
            
            # Call spread metrics
            call_spread = self._calculate_spread_metrics(short_call, long_call)
            
            # Combined metrics
            net_credit = put_spread["net_credit"] + call_spread["net_credit"]
            max_profit = (put_spread["net_credit"] + call_spread["net_credit"]) * 100
            max_loss_put = put_spread["max_loss"]
            max_loss_call = call_spread["max_loss"]
            max_loss = max(max_loss_put, max_loss_call)  # Worst case on one side
            
            risk_reward = max_loss / max_profit if max_profit > 0 else 0
            
            # Probability of profit (approximate)
            # Iron condor wins if stock stays between short strikes
            pop = (1 - abs(short_put["delta"])) * (1 - abs(short_call["delta"]))
            
            # Greeks
            delta_exposure = (short_put["delta"] + long_put["delta"] + 
                             short_call["delta"] + long_call["delta"])
            theta_exposure = (short_put["theta"] + long_put["theta"] + 
                             short_call["theta"] + long_call["theta"])
            vega_exposure = (short_put["vega"] + long_put["vega"] + 
                            short_call["vega"] + long_call["vega"])
            
            return {
                "net_credit": net_credit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "risk_reward": risk_reward,
                "probability_of_profit": pop,
                "delta_exposure": delta_exposure,
                "theta_exposure": theta_exposure,
                "vega_exposure": vega_exposure,
                "put_width": put_spread["width"],
                "call_width": call_spread["width"],
            }
        
        except Exception as e:
            logger.error(f"Error calculating iron condor metrics: {e}")
            return {}
    
    def _build_signal(
        self,
        symbol: str,
        stock_data: Dict,
        short_put: Dict,
        long_put: Dict,
        short_call: Dict,
        long_call: Dict,
        metrics: Dict
    ) -> Optional[StrategySignal]:
        """Build a strategy signal"""
        try:
            # Prepare legs
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
                },
                {
                    "option_symbol": short_call["option_symbol"],
                    "option_type": "call",
                    "strike": short_call["strike"],
                    "expiration": short_call["expiration"].isoformat(),
                    "side": "short",
                    "qty": 1,
                    "delta": short_call["delta"],
                    "price": short_call["mid"],
                },
                {
                    "option_symbol": long_call["option_symbol"],
                    "option_type": "call",
                    "strike": long_call["strike"],
                    "expiration": long_call["expiration"].isoformat(),
                    "side": "long",
                    "qty": 1,
                    "delta": long_call["delta"],
                    "price": long_call["mid"],
                }
            ]
            
            # Strategy parameters
            params = {
                "dte": short_put["dte"],
                "short_put_delta": short_put["delta"],
                "short_call_delta": short_call["delta"],
                "put_width": metrics["put_width"],
                "call_width": metrics["call_width"],
                "short_put_strike": short_put["strike"],
                "long_put_strike": long_put["strike"],
                "short_call_strike": short_call["strike"],
                "long_call_strike": long_call["strike"],
            }
            
            # Market snapshot
            market_snapshot = {
                "stock_price": stock_data["price"],
                "iv_rank": stock_data.get("iv_rank", 50),
                "put_iv": short_put["iv"],
                "call_iv": short_call["iv"],
            }
            
            # Filters passed
            filters_passed = {
                "iv_rank": stock_data.get("iv_rank", 50) >= self.min_iv_rank,
                "liquidity": True,
                "delta_range": True,
                "credit": metrics["net_credit"] >= self.min_credit,
            }
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(
                metrics, stock_data, filters_passed
            )
            
            # Create signal
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
                reason=f"Iron Condor: {short_put['strike']}/{long_put['strike']} - {short_call['strike']}/{long_call['strike']}, {short_put['dte']}DTE",
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
        """Determine if position should be exited"""
        try:
            entry_credit = trade.get("execution", {}).get("fill_credit", 0)
            
            # Take Profit: 50% of max profit
            if current_pnl_pct >= self.take_profit_pct:
                return {
                    "action": "close",
                    "reason": "take_profit",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # Stop Loss: 100% loss
            max_loss_threshold = -entry_credit * 100 * (self.stop_loss_pct / 100)
            if current_pnl <= max_loss_threshold:
                return {
                    "action": "close",
                    "reason": "stop_loss",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # Check if one side is threatened
            params = trade.get("params", {})
            stock_price = trade.get("market_snapshot", {}).get("stock_price", 0)
            
            short_put_strike = params.get("short_put_strike", 0)
            short_call_strike = params.get("short_call_strike", 0)
            
            # If stock breaches short strikes, consider closing
            if stock_price <= short_put_strike or stock_price >= short_call_strike:
                if current_pnl_pct > -50:  # Close if loss isn't too large yet
                    return {
                        "action": "close",
                        "reason": "strike_breached",
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
        """Determine if position should be rolled"""
        # Iron condors are typically not rolled; manage via exit
        return None


