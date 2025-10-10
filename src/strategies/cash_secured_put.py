"""Cash Secured Put Strategy Implementation"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from .base import Strategy, StrategySignal


class CashSecuredPutStrategy(Strategy):
    """
    Cash Secured Put (Naked Put Sale)
    
    Entry Criteria:
    - Sell put at target delta (e.g., -0.25)
    - Target DTE: 30-45 days
    - High IV Rank preferred (>30)
    - Good liquidity
    - Must have cash to secure
    
    Exit Criteria:
    - Take profit at 50% of max profit
    - Stop loss at 100% loss (2x premium)
    - Assignment acceptable (buying stock at strike)
    """
    
    def __init__(self, config: Dict):
        super().__init__("Cash Secured Put", config)
        
        # Strategy parameters
        self.dte_range = config.get("dte_range", [30, 45])
        self.delta_range = config.get("delta_range", [-0.30, -0.20])
        self.min_premium = config.get("min_premium", 0.50)
        
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
        """Generate Cash Secured Put signals"""
        
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
                return []
            
            # Group by expiration
            expirations = {}
            for put in liquid_puts:
                exp_date = put["expiration"].date()
                if exp_date not in expirations:
                    expirations[exp_date] = []
                expirations[exp_date].append(put)
            
            # For each expiration, find optimal puts
            for exp_date, exp_puts in expirations.items():
                # Filter by delta range
                target_puts = [
                    p for p in exp_puts
                    if self.delta_range[0] <= p["delta"] <= self.delta_range[1]
                ]
                
                if not target_puts:
                    continue
                
                # Sort by premium (descending)
                target_puts.sort(key=lambda x: x["mid"], reverse=True)
                
                # Take top candidates
                for put in target_puts[:3]:
                    # Filter: Minimum premium
                    if put["mid"] < self.min_premium:
                        continue
                    
                    # Build signal
                    signal = self._build_signal(symbol, stock_data, put)
                    
                    if signal:
                        signals.append(signal)
            
            # Sort by signal quality
            signals.sort(key=lambda x: x.signal_quality, reverse=True)
            return signals[:3]
        
        except Exception as e:
            logger.error(f"Error generating Cash Secured Put signals for {symbol}: {e}")
            return []
    
    def _build_signal(
        self,
        symbol: str,
        stock_data: Dict,
        put: Dict
    ) -> Optional[StrategySignal]:
        """Build a strategy signal"""
        try:
            # Prepare leg
            legs = [
                {
                    "option_symbol": put["option_symbol"],
                    "option_type": "put",
                    "strike": put["strike"],
                    "expiration": put["expiration"].isoformat(),
                    "side": "short",
                    "qty": 1,
                    "delta": put["delta"],
                    "price": put["mid"],
                }
            ]
            
            # Calculate metrics
            premium = put["mid"]
            max_profit = premium * 100  # Per contract
            max_loss = (put["strike"] - premium) * 100  # If stock goes to 0
            risk_reward = max_loss / max_profit if max_profit > 0 else 0
            pop = 1 - abs(put["delta"])
            
            # Strategy parameters
            params = {
                "dte": put["dte"],
                "delta": put["delta"],
                "strike": put["strike"],
                "premium": premium,
            }
            
            # Market snapshot
            market_snapshot = {
                "stock_price": stock_data["price"],
                "iv_rank": stock_data.get("iv_rank", 50),
                "iv": put["iv"],
                "open_interest": put["open_interest"],
                "volume": put["volume"],
                "spread": put["spread"],
            }
            
            # Filters passed
            filters_passed = {
                "iv_rank": stock_data.get("iv_rank", 50) >= self.min_iv_rank,
                "liquidity": True,
                "delta_range": True,
                "premium": premium >= self.min_premium,
            }
            
            # Metrics dict for quality calculation
            metrics = {
                "max_profit": max_profit,
                "max_loss": max_loss,
                "risk_reward": risk_reward,
                "probability_of_profit": pop,
                "delta_exposure": put["delta"],
                "theta_exposure": put["theta"],
                "vega_exposure": put["vega"],
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
                max_profit=max_profit,
                max_loss=max_loss,
                probability_of_profit=pop,
                expected_credit=premium,
                risk_reward_ratio=risk_reward,
                delta_exposure=put["delta"],
                theta_exposure=put["theta"],
                vega_exposure=put["vega"],
                signal_quality=signal_quality,
                filters_passed=filters_passed,
                reason=f"Cash Secured Put: {put['delta']:.2f}Δ @ {put['strike']}, {put['dte']}DTE",
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
            premium = trade.get("params", {}).get("premium", 0)
            
            # Take Profit: 50% of max profit
            if current_pnl_pct >= self.take_profit_pct:
                return {
                    "action": "close",
                    "reason": "take_profit",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # Stop Loss: 100% loss
            max_loss_threshold = -premium * 100 * (self.stop_loss_pct / 100)
            if current_pnl <= max_loss_threshold:
                return {
                    "action": "close",
                    "reason": "stop_loss",
                    "pnl": current_pnl,
                    "pnl_pct": current_pnl_pct,
                }
            
            # Expiration approaching (7 DTE)
            params = trade.get("params", {})
            days_held = trade.get("days_held", 0)
            dte_at_entry = params.get("dte", 30)
            current_dte = dte_at_entry - days_held
            
            if current_dte <= 7:
                # Close if profitable
                if current_pnl > 0:
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
            
            # Consider rolling if close to expiration and position threatened
            if current_dte < 10:
                stock_price = trade.get("market_snapshot", {}).get("stock_price", 0)
                strike = params.get("strike", 0)
                
                otm_pct = ((stock_price - strike) / stock_price) * 100
                
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


