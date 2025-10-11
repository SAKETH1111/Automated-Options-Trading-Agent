"""Greeks calculation for options"""

import math
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm
from loguru import logger


class GreeksCalculator:
    """Calculate option Greeks using Black-Scholes model"""
    
    @staticmethod
    def calculate_greeks(
        option_type: str,
        stock_price: float,
        strike: float,
        time_to_expiry: float,  # in years
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Args:
            option_type: "call" or "put"
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Implied volatility (annual)
            dividend_yield: Dividend yield (annual)
        
        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        try:
            if time_to_expiry <= 0:
                # Expired or same day - handle edge case
                return {
                    "delta": 0.0,
                    "gamma": 0.0,
                    "theta": 0.0,
                    "vega": 0.0,
                    "rho": 0.0,
                }
            
            # Adjust for dividends
            S = stock_price * math.exp(-dividend_yield * time_to_expiry)
            K = strike
            t = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
            d2 = d1 - sigma * math.sqrt(t)
            
            # Standard normal CDF and PDF
            nd1 = norm.cdf(d1)
            nd2 = norm.cdf(d2)
            n_d1 = norm.pdf(d1)
            
            is_call = option_type.lower() == "call"
            
            # Delta
            if is_call:
                delta = math.exp(-dividend_yield * t) * nd1
            else:
                delta = -math.exp(-dividend_yield * t) * (1 - nd1)
            
            # Gamma (same for call and put)
            gamma = (math.exp(-dividend_yield * t) * n_d1) / (S * sigma * math.sqrt(t))
            
            # Theta
            term1 = -(S * n_d1 * sigma * math.exp(-dividend_yield * t)) / (2 * math.sqrt(t))
            if is_call:
                term2 = -r * K * math.exp(-r * t) * nd2
                term3 = dividend_yield * S * math.exp(-dividend_yield * t) * nd1
                theta = (term1 + term2 + term3) / 365  # Convert to daily
            else:
                term2 = r * K * math.exp(-r * t) * (1 - nd2)
                term3 = -dividend_yield * S * math.exp(-dividend_yield * t) * (1 - nd1)
                theta = (term1 + term2 + term3) / 365  # Convert to daily
            
            # Vega (same for call and put)
            vega = (S * math.exp(-dividend_yield * t) * n_d1 * math.sqrt(t)) / 100  # Per 1% change
            
            # Rho
            if is_call:
                rho = (K * t * math.exp(-r * t) * nd2) / 100  # Per 1% change
            else:
                rho = -(K * t * math.exp(-r * t) * (1 - nd2)) / 100  # Per 1% change
            
            return {
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
                "rho": round(rho, 4),
            }
        
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0,
            }
    
    @staticmethod
    def days_to_expiry(expiration_date: datetime) -> float:
        """Calculate days to expiry as a fraction of year"""
        now = datetime.now()
        days = (expiration_date - now).total_seconds() / 86400
        return max(days / 365.0, 0.0)
    
    @staticmethod
    def calculate_strike_from_delta(
        option_type: str,
        stock_price: float,
        target_delta: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """
        Calculate strike price for a given delta
        Uses Newton-Raphson method for approximation
        """
        try:
            # Initial guess
            if option_type.lower() == "call":
                strike = stock_price * (1 + abs(target_delta) * 0.1)
            else:
                strike = stock_price * (1 - abs(target_delta) * 0.1)
            
            # Newton-Raphson iteration
            max_iterations = 50
            tolerance = 0.0001
            
            for _ in range(max_iterations):
                greeks = GreeksCalculator.calculate_greeks(
                    option_type, stock_price, strike,
                    time_to_expiry, risk_free_rate, volatility, dividend_yield
                )
                
                delta_diff = greeks["delta"] - target_delta
                
                if abs(delta_diff) < tolerance:
                    return round(strike, 2)
                
                # Adjust strike using gamma
                if greeks["gamma"] != 0:
                    strike -= delta_diff / greeks["gamma"]
                else:
                    break
            
            return round(strike, 2)
        
        except Exception as e:
            logger.error(f"Error calculating strike from delta: {e}")
            return None




