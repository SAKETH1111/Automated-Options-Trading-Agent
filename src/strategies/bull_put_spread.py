"""Bull Put Spread Strategy Implementation"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from .base import Strategy, StrategySignal


class BullPutSpreadStrategy(Strategy):
    """
    Bull Put Spread (Vertical Credit Spread)
    
    Entry Criteria:
    - Sell put at lower delta (e.g., -0.25)
    - Buy put below it for protection
    - Target DTE: 25-45 days
    - High IV Rank preferred (>25)
    - Good liquidity (OI, volume, spread)
    
    Exit Criteria:
    - Take profit at 50% of max profit
    - Stop loss at 100% of credit (2x loss)
    - Roll if threatened and IV still elevated
    """
    
    def __init__(self, config: Dict):
        super().__init__("Bull Put Spread", config)
        
        # Strategy parameters
        self.dte_range = config.get("dte_range", [25, 45])
        self.short_delta_range = config.get("short_delta_range", [-0.30, -0.20])
        self.width_range = config.get("width_range", [5, 10])
        self.min_credit = config.get("min_credit", 0.30)
        self.max_risk_reward = config.get("max_risk_reward", 4.0)
        
        # Exit parameters
        self.take_profit_pct = config.get("take_profit_pct", 50)
        self.stop_loss_pct = config.get("stop_loss_pct", 100)
        
        # Filters
        self.min_iv_rank = config.get("min_iv_rank", 25)
        self.min_oi = config.get("min_oi", 100)
        self.min_volume = config.get("min_volume", 50)
        self.max_spread_pct = config.get("max_spread_pct", 10.0)
    
    def generate_signals(
        self,
        symbol: str,
        stock_data: Dict,
        options_chain: List[Dict]
    ) -> List[StrategySignal]:
        """Generate Bull Put Spread signals"""
        
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
                logger.debug(f"{symbol}: No options in DTE range")
                return []
            
            # Filter: Only puts
            puts = [opt for opt in options if opt["option_type"] == "put"]
            
            if not puts:
                return []
            
            # Filter by liquidity
            liquid_puts = self._filter_by_liquidity(
                puts,
                self.min_oi,
                self.min_volume,
                self.max_spread_pct
            )
            
            if not liquid_puts:
                logger.debug(f"{symbol}: No liquid puts found")
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
                # Sort by strike
                exp_puts_sorted = sorted(exp_puts, key=lambda x: x["strike"], reverse=True)
                
                # Find short leg (target delta)
                for short_put in exp_puts_sorted:
                    short_delta = short_put["delta"]
                    
                    # Check if delta is in target range
                    if not (self.short_delta_range[0] <= short_delta <= self.short_delta_range[1]):
                        continue
                    
                    short_strike = short_put["strike"]
                    
                    # Find long leg (protection)
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
                        
                        # Filter: Minimum credit
                        if metrics["net_credit"] < self.min_credit:
                            continue
                        
                        # Filter: Max risk/reward
                        if metrics["risk_reward"] > self.max_risk_reward:
                            continue
                        
                        # Build signal
                        signal = self._build_signal(
                            symbol, stock_data, short_put, long_put, metrics
                        )
                        
                        if signal:
                            signals.append(signal)
                            # Only return best signal per expiration
                            break
                    
                    if signals:
                        break
            
            # Sort by signal quality and return top signals
            signals.sort(key=lambda x: x.signal_quality, reverse=True)
            return signals[:3]  # Return top 3 signals
        
        except Exception as e:
            logger.error(f"Error generating Bull Put Spread signals for {symbol}: {e}")
            return []
    
    def _build_signal(
        self,
        symbol: str,
        stock_data: Dict,
        short_put: Dict,
        long_put: Dict,
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
                }
            ]
            
            # Strategy parameters
            params = {
                "dte": short_put["dte"],
                "short_delta": short_put["delta"],
                "long_delta": long_put["delta"],
                "width": metrics["width"],
                "short_strike": short_put["strike"],
                "long_strike": long_put["strike"],
            }
            
            # Market snapshot
            market_snapshot = {
                "stock_price": stock_data["price"],
                "iv_rank": stock_data.get("iv_rank", 50),
                "short_iv": short_put["iv"],
                "long_iv": long_put["iv"],
                "short_oi": short_put["open_interest"],
                "long_oi": long_put["open_interest"],
                "short_spread": short_put["spread"],
                "long_spread": long_put["spread"],
            }
            
            # Filters passed
            filters_passed = {
                "iv_rank": stock_data.get("iv_rank", 50) >= self.min_iv_rank,
                "liquidity": True,
                "delta_range": True,
                "credit": metrics["net_credit"] >= self.min_credit,
                "risk_reward": metrics["risk_reward"] <= self.max_risk_reward,
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
                reason=f"Bull Put Spread: {short_put['delta']:.2f}Δ @ {short_put['strike']}/{long_put['strike']}, {short_put['dte']}DTE",
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
            # Get trade details
            params = trade.get("params", {})
            entry_credit = trade.get("execution", {}).get("fill_credit", 0)
            
            # Take Profit: 50% of max profit
            if current_pnl_pct >= self.take_profit_pct:
                return {
                    "action": "close",
                    "reason": "take_profit",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # Stop Loss: 100% loss (2x credit received)
            max_loss_threshold = -entry_credit * 100 * (self.stop_loss_pct / 100)
            if current_pnl <= max_loss_threshold:
                return {
                    "action": "close",
                    "reason": "stop_loss",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # Expiration approaching (7 DTE)
            days_held = trade.get("days_held", 0)
            dte_at_entry = params.get("dte", 30)
            current_dte = dte_at_entry - days_held
            
            if current_dte <= 7:
                # Close if profitable or at small loss
                if current_pnl >= -entry_credit * 50:  # Less than 50% loss
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
        """Determine if position should be rolled"""
        try:
            params = trade.get("params", {})
            days_held = trade.get("days_held", 0)
            dte_at_entry = params.get("dte", 30)
            current_dte = dte_at_entry - days_held
            
            # Consider rolling if:
            # 1. Close to expiration (< 10 DTE)
            # 2. Position is threatened (short strike < 5% OTM)
            # 3. IV is still elevated
            
            if current_dte < 10:
                # Check if we should roll to next expiration
                stock_price = trade.get("market_snapshot", {}).get("stock_price", 0)
                short_strike = params.get("short_strike", 0)
                
                otm_pct = ((stock_price - short_strike) / stock_price) * 100
                
                # If less than 5% OTM, consider rolling
                if otm_pct < 5:
                    return {
                        "action": "roll",
                        "reason": "threatened_near_expiration",
                        "current_dte": current_dte,
                        "otm_pct": otm_pct,
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking roll conditions: {e}")
            return None


