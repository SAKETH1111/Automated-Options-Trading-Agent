"""
Pre-Trade Validation Engine for Real Money Trading
Multi-layer validation system that must pass ALL checks before any trade executes
"""

import uuid
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from src.config.settings import get_config
from src.database.session import get_db
from src.database.models import Trade, Position


class ValidationResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ValidationCheck:
    """Individual validation check result"""
    check_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class PreTradeValidator:
    """
    Multi-layer validation system for real money trading
    
    ALL checks must pass before any trade is executed
    Logs all failures for analysis and improvement
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        self.db = get_db()
        
        # Validation thresholds by account size
        self.thresholds = self._get_validation_thresholds()
        
        # Market hours (NYSE)
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)
        
        logger.info("PreTradeValidator initialized with multi-layer protection")
    
    def _get_validation_thresholds(self) -> Dict[str, Dict]:
        """Get validation thresholds based on account size"""
        return {
            "micro": {
                "max_spread_pct": 15.0,  # Stricter for small accounts
                "min_volume": 100,
                "min_oi": 100,
                "max_iv_pct": 200,
                "min_iv_pct": 10,
                "max_price": 10.0,  # Avoid expensive options
                "min_dte": 7,
                "max_dte": 60
            },
            "small": {
                "max_spread_pct": 12.0,
                "min_volume": 50,
                "min_oi": 50,
                "max_iv_pct": 250,
                "min_iv_pct": 8,
                "max_price": 20.0,
                "min_dte": 7,
                "max_dte": 90
            },
            "medium": {
                "max_spread_pct": 10.0,
                "min_volume": 25,
                "min_oi": 25,
                "max_iv_pct": 300,
                "min_iv_pct": 5,
                "max_price": 50.0,
                "min_dte": 5,
                "max_dte": 120
            },
            "large": {
                "max_spread_pct": 8.0,
                "min_volume": 10,
                "min_oi": 10,
                "max_iv_pct": 400,
                "min_iv_pct": 3,
                "max_price": 100.0,
                "min_dte": 3,
                "max_dte": 180
            },
            "institutional": {
                "max_spread_pct": 5.0,
                "min_volume": 5,
                "min_oi": 5,
                "max_iv_pct": 500,
                "min_iv_pct": 1,
                "max_price": 200.0,
                "min_dte": 1,
                "max_dte": 365
            }
        }
    
    def validate_trade(
        self,
        trade_params: Dict,
        account_balance: float,
        current_positions: List[Dict] = None
    ) -> Tuple[bool, List[ValidationCheck], str]:
        """
        Validate a trade through all layers
        
        Returns:
            (can_trade, validation_results, summary_message)
        """
        try:
            validation_results = []
            
            # Get account tier for appropriate thresholds
            account_tier = self._get_account_tier(account_balance)
            thresholds = self.thresholds[account_tier]
            
            # Layer 1: Data Quality Validation
            data_checks = self._validate_data_quality(trade_params, thresholds)
            validation_results.extend(data_checks)
            
            # Layer 2: Risk Validation
            risk_checks = self._validate_risk_limits(
                trade_params, account_balance, current_positions
            )
            validation_results.extend(risk_checks)
            
            # Layer 3: Logic Validation
            logic_checks = self._validate_trade_logic(trade_params, thresholds)
            validation_results.extend(logic_checks)
            
            # Layer 4: Market Conditions
            market_checks = self._validate_market_conditions(trade_params)
            validation_results.extend(market_checks)
            
            # Layer 5: System Health
            system_checks = self._validate_system_health()
            validation_results.extend(system_checks)
            
            # Determine if trade can proceed
            failed_checks = [check for check in validation_results 
                           if check.result == ValidationResult.FAIL]
            warning_checks = [check for check in validation_results 
                            if check.result == ValidationResult.WARNING]
            
            can_trade = len(failed_checks) == 0
            
            # Log results
            self._log_validation_results(trade_params, validation_results, can_trade)
            
            # Create summary message
            summary = self._create_summary_message(
                can_trade, failed_checks, warning_checks, len(validation_results)
            )
            
            return can_trade, validation_results, summary
            
        except Exception as e:
            logger.error(f"Error in trade validation: {e}")
            return False, [], f"Validation error: {str(e)}"
    
    def _get_account_tier(self, account_balance: float) -> str:
        """Determine account tier based on balance"""
        if account_balance < 1000:
            return "micro"
        elif account_balance < 10000:
            return "small"
        elif account_balance < 100000:
            return "medium"
        elif account_balance < 1000000:
            return "large"
        else:
            return "institutional"
    
    def _validate_data_quality(
        self, trade_params: Dict, thresholds: Dict
    ) -> List[ValidationCheck]:
        """Layer 1: Data quality validation"""
        checks = []
        
        try:
            legs = trade_params.get("legs", [])
            
            for i, leg in enumerate(legs):
                leg_id = f"leg_{i+1}"
                
                # Check option price is not stale
                option_data = leg.get("option_data", {})
                if option_data:
                    last_quote_time = option_data.get("last_quote_time")
                    if last_quote_time:
                        time_diff = (datetime.utcnow() - last_quote_time).seconds
                        if time_diff > 300:  # 5 minutes
                            checks.append(ValidationCheck(
                                f"{leg_id}_stale_data",
                                ValidationResult.FAIL,
                                f"Option data is stale ({time_diff}s old)",
                                {"time_diff_seconds": time_diff}
                            ))
                
                # Check bid/ask prices are reasonable
                bid = leg.get("bid", 0)
                ask = leg.get("ask", 0)
                mid = (bid + ask) / 2 if bid and ask else 0
                
                if bid <= 0 or ask <= 0:
                    checks.append(ValidationCheck(
                        f"{leg_id}_invalid_prices",
                        ValidationResult.FAIL,
                        f"Invalid bid/ask prices: bid={bid}, ask={ask}"
                    ))
                
                # Check spread is not excessive
                if bid > 0 and ask > 0:
                    spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 100
                    max_spread = thresholds["max_spread_pct"]
                    
                    if spread_pct > max_spread:
                        checks.append(ValidationCheck(
                            f"{leg_id}_wide_spread",
                            ValidationResult.FAIL,
                            f"Spread too wide: {spread_pct:.1f}% > {max_spread}%",
                            {"spread_pct": spread_pct, "threshold": max_spread}
                        ))
                
                # Check volume and open interest
                volume = leg.get("volume", 0)
                oi = leg.get("open_interest", 0)
                
                min_volume = thresholds["min_volume"]
                min_oi = thresholds["min_oi"]
                
                if volume < min_volume:
                    checks.append(ValidationCheck(
                        f"{leg_id}_low_volume",
                        ValidationResult.FAIL,
                        f"Volume too low: {volume} < {min_volume}",
                        {"volume": volume, "threshold": min_volume}
                    ))
                
                if oi < min_oi:
                    checks.append(ValidationCheck(
                        f"{leg_id}_low_oi",
                        ValidationResult.FAIL,
                        f"Open interest too low: {oi} < {min_oi}",
                        {"open_interest": oi, "threshold": min_oi}
                    ))
                
                # Check implied volatility is reasonable
                iv = leg.get("implied_volatility", 0)
                if iv > 0:
                    iv_pct = iv * 100
                    max_iv = thresholds["max_iv_pct"]
                    min_iv = thresholds["min_iv_pct"]
                    
                    if iv_pct > max_iv:
                        checks.append(ValidationCheck(
                            f"{leg_id}_high_iv",
                            ValidationResult.FAIL,
                            f"IV too high: {iv_pct:.1f}% > {max_iv}%",
                            {"iv_pct": iv_pct, "threshold": max_iv}
                        ))
                    
                    if iv_pct < min_iv:
                        checks.append(ValidationCheck(
                            f"{leg_id}_low_iv",
                            ValidationResult.WARNING,
                            f"IV unusually low: {iv_pct:.1f}% < {min_iv}%",
                            {"iv_pct": iv_pct, "threshold": min_iv}
                        ))
                
                # Check option price is reasonable
                if mid > thresholds["max_price"]:
                    checks.append(ValidationCheck(
                        f"{leg_id}_expensive_option",
                        ValidationResult.FAIL,
                        f"Option too expensive: ${mid:.2f} > ${thresholds['max_price']}"
                    ))
                
                # Check DTE is reasonable
                dte = leg.get("days_to_expiration", 0)
                min_dte = thresholds["min_dte"]
                max_dte = thresholds["max_dte"]
                
                if dte < min_dte:
                    checks.append(ValidationCheck(
                        f"{leg_id}_too_short_dte",
                        ValidationResult.FAIL,
                        f"DTE too short: {dte} < {min_dte}",
                        {"dte": dte, "threshold": min_dte}
                    ))
                
                if dte > max_dte:
                    checks.append(ValidationCheck(
                        f"{leg_id}_too_long_dte",
                        ValidationResult.WARNING,
                        f"DTE very long: {dte} > {max_dte}",
                        {"dte": dte, "threshold": max_dte}
                    ))
            
            # If no data quality issues found
            if not any(check.result == ValidationResult.FAIL for check in checks):
                checks.append(ValidationCheck(
                    "data_quality",
                    ValidationResult.PASS,
                    "All data quality checks passed"
                ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                "data_quality_error",
                ValidationResult.FAIL,
                f"Data quality validation error: {str(e)}"
            ))
        
        return checks
    
    def _validate_risk_limits(
        self, trade_params: Dict, account_balance: float, 
        current_positions: List[Dict] = None
    ) -> List[ValidationCheck]:
        """Layer 2: Risk validation"""
        checks = []
        
        try:
            # Get risk config
            risk_config = self.config.get("trading", {}).get("risk", {})
            account_tier = self._get_account_tier(account_balance)
            
            # Calculate trade risk
            max_loss = abs(trade_params.get("max_loss", 0))
            risk_pct = (max_loss / account_balance * 100) if account_balance > 0 else 100
            
            # Check position size limits
            max_position_size_pct = risk_config.get("max_position_size_pct", 20.0)
            if risk_pct > max_position_size_pct:
                checks.append(ValidationCheck(
                    "position_size_limit",
                    ValidationResult.FAIL,
                    f"Position risk {risk_pct:.1f}% exceeds max {max_position_size_pct}%",
                    {"risk_pct": risk_pct, "limit": max_position_size_pct}
                ))
            
            # Check portfolio heat
            current_risk = self._calculate_current_portfolio_risk(current_positions)
            portfolio_heat = (current_risk + max_loss) / account_balance * 100
            
            max_portfolio_heat = risk_config.get("max_portfolio_heat", 30.0)
            if portfolio_heat > max_portfolio_heat:
                checks.append(ValidationCheck(
                    "portfolio_heat_limit",
                    ValidationResult.FAIL,
                    f"Portfolio heat {portfolio_heat:.1f}% exceeds max {max_portfolio_heat}%",
                    {"portfolio_heat": portfolio_heat, "limit": max_portfolio_heat}
                ))
            
            # Check daily trade limit
            trades_today = self._count_trades_today()
            max_trades_per_day = risk_config.get("max_trades_per_day", 10)
            
            if trades_today >= max_trades_per_day:
                checks.append(ValidationCheck(
                    "daily_trade_limit",
                    ValidationResult.FAIL,
                    f"Daily trade limit reached: {trades_today}/{max_trades_per_day}",
                    {"trades_today": trades_today, "limit": max_trades_per_day}
                ))
            
            # Check per-symbol position limit
            symbol = trade_params.get("symbol", "")
            symbol_positions = self._count_symbol_positions(symbol, current_positions)
            max_per_symbol = risk_config.get("max_positions_per_symbol", 2)
            
            if symbol_positions >= max_per_symbol:
                checks.append(ValidationCheck(
                    "symbol_position_limit",
                    ValidationResult.FAIL,
                    f"Max positions for {symbol} reached: {symbol_positions}/{max_per_symbol}",
                    {"symbol": symbol, "current": symbol_positions, "limit": max_per_symbol}
                ))
            
            # Check Greeks limits (for larger accounts)
            if account_tier in ["large", "institutional"]:
                greeks_checks = self._validate_greeks_limits(trade_params, current_positions)
                checks.extend(greeks_checks)
            
            # If no risk issues found
            if not any(check.result == ValidationResult.FAIL for check in checks):
                checks.append(ValidationCheck(
                    "risk_limits",
                    ValidationResult.PASS,
                    "All risk limits within bounds"
                ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                "risk_validation_error",
                ValidationResult.FAIL,
                f"Risk validation error: {str(e)}"
            ))
        
        return checks
    
    def _validate_trade_logic(self, trade_params: Dict, thresholds: Dict) -> List[ValidationCheck]:
        """Layer 3: Trade logic validation"""
        checks = []
        
        try:
            strategy = trade_params.get("strategy", "")
            legs = trade_params.get("legs", [])
            
            # Basic strategy validation
            if not strategy:
                checks.append(ValidationCheck(
                    "missing_strategy",
                    ValidationResult.FAIL,
                    "Strategy not specified"
                ))
            
            if not legs:
                checks.append(ValidationCheck(
                    "missing_legs",
                    ValidationResult.FAIL,
                    "No option legs specified"
                ))
            
            # Validate strategy-specific logic
            if strategy == "bull_put_spread":
                checks.extend(self._validate_bull_put_spread(legs))
            elif strategy == "bear_call_spread":
                checks.extend(self._validate_bear_call_spread(legs))
            elif strategy == "iron_condor":
                checks.extend(self._validate_iron_condor(legs))
            elif strategy == "cash_secured_put":
                checks.extend(self._validate_cash_secured_put(legs))
            
            # Check for duplicate orders
            if self._check_duplicate_order(trade_params):
                checks.append(ValidationCheck(
                    "duplicate_order",
                    ValidationResult.FAIL,
                    "Duplicate order detected"
                ))
            
            # If no logic issues found
            if not any(check.result == ValidationResult.FAIL for check in checks):
                checks.append(ValidationCheck(
                    "trade_logic",
                    ValidationResult.PASS,
                    "Trade logic is valid"
                ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                "logic_validation_error",
                ValidationResult.FAIL,
                f"Trade logic validation error: {str(e)}"
            ))
        
        return checks
    
    def _validate_market_conditions(self, trade_params: Dict) -> List[ValidationCheck]:
        """Layer 4: Market conditions validation"""
        checks = []
        
        try:
            # Check market is open
            now = datetime.now()
            current_time = now.time()
            
            if not (self.market_open <= current_time <= self.market_close):
                checks.append(ValidationCheck(
                    "market_closed",
                    ValidationResult.FAIL,
                    f"Market is closed. Current time: {current_time}"
                ))
            
            # Check day of week (no weekend trading)
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                checks.append(ValidationCheck(
                    "weekend_trading",
                    ValidationResult.FAIL,
                    "Weekend trading not allowed"
                ))
            
            # Check for flash crash conditions (if VIX data available)
            vix = trade_params.get("market_data", {}).get("vix")
            if vix and vix > 80:
                checks.append(ValidationCheck(
                    "flash_crash_conditions",
                    ValidationResult.FAIL,
                    f"Extreme volatility detected: VIX = {vix:.1f}"
                ))
            elif vix and vix > 50:
                checks.append(ValidationCheck(
                    "high_volatility",
                    ValidationResult.WARNING,
                    f"High volatility: VIX = {vix:.1f}"
                ))
            
            # Check for unusual spread conditions
            market_data = trade_params.get("market_data", {})
            underlying_spread = market_data.get("underlying_spread_pct", 0)
            if underlying_spread > 2.0:  # 2% spread on underlying
                checks.append(ValidationCheck(
                    "wide_underlying_spread",
                    ValidationResult.WARNING,
                    f"Wide underlying spread: {underlying_spread:.1f}%"
                ))
            
            # If no market condition issues found
            if not any(check.result == ValidationResult.FAIL for check in checks):
                checks.append(ValidationCheck(
                    "market_conditions",
                    ValidationResult.PASS,
                    "Market conditions acceptable"
                ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                "market_validation_error",
                ValidationResult.FAIL,
                f"Market conditions validation error: {str(e)}"
            ))
        
        return checks
    
    def _validate_system_health(self) -> List[ValidationCheck]:
        """Layer 5: System health validation"""
        checks = []
        
        try:
            # Check database connectivity
            try:
                with self.db.get_session() as session:
                    session.execute("SELECT 1")
                checks.append(ValidationCheck(
                    "database_health",
                    ValidationResult.PASS,
                    "Database connection healthy"
                ))
            except Exception as e:
                checks.append(ValidationCheck(
                    "database_health",
                    ValidationResult.FAIL,
                    f"Database connection failed: {str(e)}"
                ))
            
            # Check for recent system errors
            recent_errors = self._count_recent_errors()
            if recent_errors > 10:  # More than 10 errors in last hour
                checks.append(ValidationCheck(
                    "system_errors",
                    ValidationResult.FAIL,
                    f"Too many recent system errors: {recent_errors}"
                ))
            elif recent_errors > 5:
                checks.append(ValidationCheck(
                    "system_errors",
                    ValidationResult.WARNING,
                    f"Elevated system errors: {recent_errors}"
                ))
            
            # Check API connectivity (would need actual API check)
            # This is a placeholder - in real implementation, ping APIs
            
            # If no system health issues found
            if not any(check.result == ValidationResult.FAIL for check in checks):
                checks.append(ValidationCheck(
                    "system_health",
                    ValidationResult.PASS,
                    "System health checks passed"
                ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                "system_health_error",
                ValidationResult.FAIL,
                f"System health validation error: {str(e)}"
            ))
        
        return checks
    
    def _validate_greeks_limits(
        self, trade_params: Dict, current_positions: List[Dict]
    ) -> List[ValidationCheck]:
        """Validate Greeks limits for larger accounts"""
        checks = []
        
        try:
            # Calculate portfolio Greeks after this trade
            current_greeks = self._calculate_portfolio_greeks(current_positions)
            trade_greeks = self._calculate_trade_greeks(trade_params)
            
            # Combine Greeks
            new_greeks = {
                "delta": current_greeks["delta"] + trade_greeks["delta"],
                "gamma": current_greeks["gamma"] + trade_greeks["gamma"],
                "theta": current_greeks["theta"] + trade_greeks["theta"],
                "vega": current_greeks["vega"] + trade_greeks["vega"]
            }
            
            # Check limits (these would be configured based on account size)
            greeks_limits = {
                "delta": 100,  # Max delta exposure
                "gamma": 0.5,  # Max gamma exposure
                "theta": 200,  # Max theta exposure
                "vega": 300    # Max vega exposure
            }
            
            for greek, limit in greeks_limits.items():
                if abs(new_greeks[greek]) > limit:
                    checks.append(ValidationCheck(
                        f"greeks_{greek}_limit",
                        ValidationResult.FAIL,
                        f"{greek.capitalize()} limit exceeded: {new_greeks[greek]:.2f} > {limit}",
                        {"current": new_greeks[greek], "limit": limit}
                    ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                "greeks_validation_error",
                ValidationResult.FAIL,
                f"Greeks validation error: {str(e)}"
            ))
        
        return checks
    
    # Strategy-specific validation methods
    def _validate_bull_put_spread(self, legs: List[Dict]) -> List[ValidationCheck]:
        """Validate bull put spread logic"""
        checks = []
        
        if len(legs) != 2:
            checks.append(ValidationCheck(
                "bull_put_legs",
                ValidationResult.FAIL,
                f"Bull put spread requires 2 legs, got {len(legs)}"
            ))
            return checks
        
        short_leg = legs[0]  # Short put
        long_leg = legs[1]   # Long put
        
        # Check both are puts
        if short_leg.get("option_type") != "put" or long_leg.get("option_type") != "put":
            checks.append(ValidationCheck(
                "bull_put_types",
                ValidationResult.FAIL,
                "Bull put spread requires both legs to be puts"
            ))
        
        # Check strikes are correct (short > long)
        short_strike = short_leg.get("strike", 0)
        long_strike = long_leg.get("strike", 0)
        
        if short_strike <= long_strike:
            checks.append(ValidationCheck(
                "bull_put_strikes",
                ValidationResult.FAIL,
                f"Short strike ({short_strike}) must be higher than long strike ({long_strike})"
            ))
        
        # Check same expiration
        short_exp = short_leg.get("expiration")
        long_exp = long_leg.get("expiration")
        
        if short_exp != long_exp:
            checks.append(ValidationCheck(
                "bull_put_expiration",
                ValidationResult.FAIL,
                "Both legs must have same expiration"
            ))
        
        return checks
    
    def _validate_bear_call_spread(self, legs: List[Dict]) -> List[ValidationCheck]:
        """Validate bear call spread logic"""
        checks = []
        
        if len(legs) != 2:
            checks.append(ValidationCheck(
                "bear_call_legs",
                ValidationResult.FAIL,
                f"Bear call spread requires 2 legs, got {len(legs)}"
            ))
            return checks
        
        short_leg = legs[0]  # Short call
        long_leg = legs[1]   # Long call
        
        # Check both are calls
        if short_leg.get("option_type") != "call" or long_leg.get("option_type") != "call":
            checks.append(ValidationCheck(
                "bear_call_types",
                ValidationResult.FAIL,
                "Bear call spread requires both legs to be calls"
            ))
        
        # Check strikes are correct (short < long)
        short_strike = short_leg.get("strike", 0)
        long_strike = long_leg.get("strike", 0)
        
        if short_strike >= long_strike:
            checks.append(ValidationCheck(
                "bear_call_strikes",
                ValidationResult.FAIL,
                f"Short strike ({short_strike}) must be lower than long strike ({long_strike})"
            ))
        
        return checks
    
    def _validate_iron_condor(self, legs: List[Dict]) -> List[ValidationCheck]:
        """Validate iron condor logic"""
        checks = []
        
        if len(legs) != 4:
            checks.append(ValidationCheck(
                "iron_condor_legs",
                ValidationResult.FAIL,
                f"Iron condor requires 4 legs, got {len(legs)}"
            ))
            return checks
        
        # Should have: short put, long put, short call, long call
        # Strikes: long put < short put < short call < long call
        
        return checks  # Simplified for now
    
    def _validate_cash_secured_put(self, legs: List[Dict]) -> List[ValidationCheck]:
        """Validate cash secured put logic"""
        checks = []
        
        if len(legs) != 1:
            checks.append(ValidationCheck(
                "csp_legs",
                ValidationResult.FAIL,
                f"Cash secured put requires 1 leg, got {len(legs)}"
            ))
            return checks
        
        leg = legs[0]
        
        # Check it's a put
        if leg.get("option_type") != "put":
            checks.append(ValidationCheck(
                "csp_type",
                ValidationResult.FAIL,
                "Cash secured put must be a put option"
            ))
        
        return checks
    
    # Helper methods
    def _calculate_current_portfolio_risk(self, positions: List[Dict]) -> float:
        """Calculate current portfolio risk"""
        if not positions:
            return 0.0
        
        total_risk = 0.0
        for position in positions:
            risk = position.get("max_loss", 0)
            quantity = position.get("quantity", 1)
            total_risk += abs(risk) * quantity
        
        return total_risk
    
    def _count_trades_today(self) -> int:
        """Count trades executed today"""
        try:
            with self.db.get_session() as session:
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                count = session.query(Trade).filter(
                    Trade.timestamp_enter >= today_start
                ).count()
                return count
        except Exception as e:
            logger.error(f"Error counting trades today: {e}")
            return 0
    
    def _count_symbol_positions(self, symbol: str, positions: List[Dict]) -> int:
        """Count current positions for a symbol"""
        if not positions:
            return 0
        
        count = 0
        for position in positions:
            if position.get("symbol") == symbol:
                count += 1
        
        return count
    
    def _check_duplicate_order(self, trade_params: Dict) -> bool:
        """Check for duplicate orders (simplified)"""
        # In real implementation, would check recent orders
        return False
    
    def _calculate_portfolio_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate current portfolio Greeks"""
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }  # Simplified for now
    
    def _calculate_trade_greeks(self, trade_params: Dict) -> Dict[str, float]:
        """Calculate Greeks for this trade"""
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }  # Simplified for now
    
    def _count_recent_errors(self) -> int:
        """Count recent system errors"""
        # In real implementation, would check error logs
        return 0
    
    def _log_validation_results(
        self, trade_params: Dict, validation_results: List[ValidationCheck], 
        can_trade: bool
    ):
        """Log validation results for analysis"""
        try:
            failed_checks = [check for check in validation_results 
                           if check.result == ValidationResult.FAIL]
            warning_checks = [check for check in validation_results 
                            if check.result == ValidationResult.WARNING]
            
            logger.info(f"Trade validation: {'PASSED' if can_trade else 'FAILED'}")
            logger.info(f"  - Total checks: {len(validation_results)}")
            logger.info(f"  - Failed: {len(failed_checks)}")
            logger.info(f"  - Warnings: {len(warning_checks)}")
            
            if failed_checks:
                logger.warning("Failed validation checks:")
                for check in failed_checks:
                    logger.warning(f"  - {check.check_name}: {check.message}")
            
            if warning_checks:
                logger.info("Warning validation checks:")
                for check in warning_checks:
                    logger.info(f"  - {check.check_name}: {check.message}")
            
        except Exception as e:
            logger.error(f"Error logging validation results: {e}")
    
    def _create_summary_message(
        self, can_trade: bool, failed_checks: List[ValidationCheck],
        warning_checks: List[ValidationCheck], total_checks: int
    ) -> str:
        """Create human-readable summary message"""
        if can_trade:
            message = f"✅ Trade validation PASSED ({total_checks} checks)"
            if warning_checks:
                message += f" with {len(warning_checks)} warnings"
        else:
            message = f"❌ Trade validation FAILED ({len(failed_checks)}/{total_checks} checks failed)"
            message += f": {failed_checks[0].message}"
        
        return message


# Example usage
if __name__ == "__main__":
    # Test the validator
    validator = PreTradeValidator()
    
    # Sample trade parameters
    test_trade = {
        "strategy": "bull_put_spread",
        "symbol": "SPY",
        "max_loss": 500,
        "legs": [
            {
                "option_type": "put",
                "strike": 500,
                "expiration": "2025-02-21",
                "bid": 1.25,
                "ask": 1.35,
                "volume": 150,
                "open_interest": 500,
                "implied_volatility": 0.25,
                "days_to_expiration": 30
            },
            {
                "option_type": "put",
                "strike": 495,
                "expiration": "2025-02-21",
                "bid": 0.85,
                "ask": 0.95,
                "volume": 200,
                "open_interest": 750,
                "implied_volatility": 0.24,
                "days_to_expiration": 30
            }
        ],
        "market_data": {
            "vix": 18.5,
            "underlying_spread_pct": 0.1
        }
    }
    
    can_trade, results, summary = validator.validate_trade(
        test_trade, account_balance=10000
    )
    
    print(f"Can trade: {can_trade}")
    print(f"Summary: {summary}")
    print(f"Results: {len(results)} checks performed")
