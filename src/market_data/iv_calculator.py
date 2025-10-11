"""Implied Volatility calculation and ranking"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from loguru import logger


class IVCalculator:
    """Calculate Implied Volatility and IV Rank"""
    
    @staticmethod
    def black_scholes_price(
        option_type: str,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0
    ) -> float:
        """Calculate option price using Black-Scholes"""
        try:
            if time_to_expiry <= 0:
                # Intrinsic value
                if option_type.lower() == "call":
                    return max(stock_price - strike, 0)
                else:
                    return max(strike - stock_price, 0)
            
            S = stock_price * math.exp(-dividend_yield * time_to_expiry)
            K = strike
            t = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
            d2 = d1 - sigma * math.sqrt(t)
            
            if option_type.lower() == "call":
                price = S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return price
        
        except Exception as e:
            logger.error(f"Error calculating BS price: {e}")
            return 0.0
    
    @staticmethod
    def calculate_iv(
        option_type: str,
        market_price: float,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method
        
        Returns:
            Implied volatility (annualized) or None if calculation fails
        """
        try:
            if time_to_expiry <= 0 or market_price <= 0:
                return None
            
            # Intrinsic value
            if option_type.lower() == "call":
                intrinsic = max(stock_price - strike, 0)
            else:
                intrinsic = max(strike - stock_price, 0)
            
            # If market price is below intrinsic, return None
            if market_price < intrinsic:
                return None
            
            # Objective function
            def objective(sigma):
                return IVCalculator.black_scholes_price(
                    option_type, stock_price, strike,
                    time_to_expiry, risk_free_rate, sigma, dividend_yield
                ) - market_price
            
            # Solve for IV using Brent's method
            # Search between 0.01% and 500% volatility
            iv = brentq(objective, 0.0001, 5.0, maxiter=100)
            
            return round(iv, 4)
        
        except Exception as e:
            # If Brent fails, try Newton-Raphson with vega
            try:
                sigma = 0.3  # Initial guess
                for _ in range(50):
                    price = IVCalculator.black_scholes_price(
                        option_type, stock_price, strike,
                        time_to_expiry, risk_free_rate, sigma, dividend_yield
                    )
                    
                    diff = price - market_price
                    if abs(diff) < 0.001:
                        return round(sigma, 4)
                    
                    # Calculate vega for Newton step
                    S = stock_price * math.exp(-dividend_yield * time_to_expiry)
                    d1 = (math.log(S / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * math.sqrt(time_to_expiry))
                    vega = S * norm.pdf(d1) * math.sqrt(time_to_expiry)
                    
                    if vega > 0:
                        sigma -= diff / vega
                    else:
                        break
                    
                    # Bounds checking
                    sigma = max(0.0001, min(sigma, 5.0))
                
                return round(sigma, 4) if 0.01 <= sigma <= 3.0 else None
            
            except Exception:
                logger.debug(f"Could not calculate IV for price={market_price}")
                return None
    
    @staticmethod
    def calculate_iv_rank(historical_ivs: List[float], current_iv: float, lookback_days: int = 252) -> float:
        """
        Calculate IV Rank: where current IV stands relative to its range over lookback period
        
        IV Rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
        
        Returns:
            IV Rank between 0 and 100
        """
        try:
            if not historical_ivs or current_iv is None:
                return 50.0  # Default to midpoint
            
            # Use most recent data up to lookback_days
            recent_ivs = historical_ivs[-lookback_days:] if len(historical_ivs) > lookback_days else historical_ivs
            
            if len(recent_ivs) < 20:  # Need minimum data points
                return 50.0
            
            min_iv = min(recent_ivs)
            max_iv = max(recent_ivs)
            
            if max_iv == min_iv:
                return 50.0
            
            iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
            
            return round(max(0.0, min(100.0, iv_rank)), 2)
        
        except Exception as e:
            logger.error(f"Error calculating IV rank: {e}")
            return 50.0
    
    @staticmethod
    def calculate_iv_percentile(historical_ivs: List[float], current_iv: float, lookback_days: int = 252) -> float:
        """
        Calculate IV Percentile: percentage of days where IV was below current level
        
        Returns:
            IV Percentile between 0 and 100
        """
        try:
            if not historical_ivs or current_iv is None:
                return 50.0
            
            recent_ivs = historical_ivs[-lookback_days:] if len(historical_ivs) > lookback_days else historical_ivs
            
            if len(recent_ivs) < 20:
                return 50.0
            
            below_count = sum(1 for iv in recent_ivs if iv <= current_iv)
            percentile = (below_count / len(recent_ivs)) * 100
            
            return round(percentile, 2)
        
        except Exception as e:
            logger.error(f"Error calculating IV percentile: {e}")
            return 50.0
    
    @staticmethod
    def estimate_iv_from_historical_volatility(prices: List[float], window: int = 20) -> Optional[float]:
        """
        Estimate IV from historical price volatility
        
        Args:
            prices: List of historical prices
            window: Rolling window for volatility calculation
        
        Returns:
            Estimated IV (annualized)
        """
        try:
            if len(prices) < window + 1:
                return None
            
            # Calculate log returns
            returns = np.diff(np.log(prices))
            
            # Calculate rolling std
            rolling_std = np.std(returns[-window:])
            
            # Annualize (assuming 252 trading days)
            annualized_vol = rolling_std * np.sqrt(252)
            
            return round(annualized_vol, 4)
        
        except Exception as e:
            logger.error(f"Error estimating IV from HV: {e}")
            return None




