"""
Greeks Calculator Module
Calculate option Greeks using Black-Scholes model
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
from datetime import datetime, timedelta
from loguru import logger


class GreeksCalculator:
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
    Uses Black-Scholes model for European-style options
    """
    
    def __init__(self):
        """Initialize Greeks calculator"""
        self.risk_free_rate = 0.05  # Default 5% annual risk-free rate
        logger.info("Greeks Calculator initialized")
    
    def set_risk_free_rate(self, rate: float):
        """Set risk-free rate"""
        self.risk_free_rate = rate
    
    def calculate_d1_d2(
        self,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> tuple:
        """
        Calculate d1 and d2 for Black-Scholes formula
        
        Args:
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (annual)
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Tuple of (d1, d2)
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if time_to_expiry <= 0:
            time_to_expiry = 0.0001  # Avoid division by zero
        
        d1 = (np.log(stock_price / strike) + 
              (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        return d1, d2
    
    def calculate_delta(
        self,
        option_type: str,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Delta
        
        Delta measures the rate of change of option price with respect to 
        changes in the underlying asset's price.
        
        Call Delta: 0 to 1
        Put Delta: -1 to 0
        
        Args:
            option_type: 'CALL' or 'PUT'
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Delta value
        """
        d1, _ = self.calculate_d1_d2(stock_price, strike, time_to_expiry, 
                                      volatility, risk_free_rate)
        
        if option_type.upper() == 'CALL':
            delta = norm.cdf(d1)
        else:  # PUT
            delta = norm.cdf(d1) - 1
        
        return delta
    
    def calculate_gamma(
        self,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Gamma
        
        Gamma measures the rate of change in Delta with respect to changes 
        in the underlying price. Same for calls and puts.
        
        Args:
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Gamma value
        """
        d1, _ = self.calculate_d1_d2(stock_price, strike, time_to_expiry,
                                      volatility, risk_free_rate)
        
        gamma = norm.pdf(d1) / (stock_price * volatility * np.sqrt(time_to_expiry))
        
        return gamma
    
    def calculate_theta(
        self,
        option_type: str,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Theta
        
        Theta measures the rate of decline in the value of an option due to 
        the passage of time (time decay). Usually expressed as daily decay.
        
        Args:
            option_type: 'CALL' or 'PUT'
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Theta value (daily)
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        d1, d2 = self.calculate_d1_d2(stock_price, strike, time_to_expiry,
                                       volatility, risk_free_rate)
        
        if option_type.upper() == 'CALL':
            theta = (-(stock_price * norm.pdf(d1) * volatility) / 
                    (2 * np.sqrt(time_to_expiry)) -
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * 
                    norm.cdf(d2))
        else:  # PUT
            theta = (-(stock_price * norm.pdf(d1) * volatility) / 
                    (2 * np.sqrt(time_to_expiry)) +
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * 
                    norm.cdf(-d2))
        
        # Convert to daily theta (divide by 365)
        theta_daily = theta / 365
        
        return theta_daily
    
    def calculate_vega(
        self,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Vega
        
        Vega measures sensitivity to volatility. Same for calls and puts.
        Expressed as change in option price for 1% change in IV.
        
        Args:
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Vega value
        """
        d1, _ = self.calculate_d1_d2(stock_price, strike, time_to_expiry,
                                      volatility, risk_free_rate)
        
        vega = stock_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
        
        # Convert to 1% change (divide by 100)
        vega = vega / 100
        
        return vega
    
    def calculate_rho(
        self,
        option_type: str,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Rho
        
        Rho measures sensitivity to interest rate changes.
        Expressed as change in option price for 1% change in interest rates.
        
        Args:
            option_type: 'CALL' or 'PUT'
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Rho value
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        _, d2 = self.calculate_d1_d2(stock_price, strike, time_to_expiry,
                                      volatility, risk_free_rate)
        
        if option_type.upper() == 'CALL':
            rho = (strike * time_to_expiry * 
                   np.exp(-risk_free_rate * time_to_expiry) * 
                   norm.cdf(d2))
        else:  # PUT
            rho = (-strike * time_to_expiry * 
                   np.exp(-risk_free_rate * time_to_expiry) * 
                   norm.cdf(-d2))
        
        # Convert to 1% change (divide by 100)
        rho = rho / 100
        
        return rho
    
    def calculate_all_greeks(
        self,
        option_type: str,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all Greeks at once
        
        Args:
            option_type: 'CALL' or 'PUT'
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Dictionary with all Greeks
        """
        try:
            greeks = {
                'delta': self.calculate_delta(option_type, stock_price, strike,
                                              time_to_expiry, volatility, risk_free_rate),
                'gamma': self.calculate_gamma(stock_price, strike, time_to_expiry,
                                              volatility, risk_free_rate),
                'theta': self.calculate_theta(option_type, stock_price, strike,
                                              time_to_expiry, volatility, risk_free_rate),
                'vega': self.calculate_vega(stock_price, strike, time_to_expiry,
                                           volatility, risk_free_rate),
                'rho': self.calculate_rho(option_type, stock_price, strike,
                                         time_to_expiry, volatility, risk_free_rate)
            }
            
            return greeks
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    def calculate_time_to_expiry(
        self,
        expiration_date: datetime,
        current_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate time to expiration in years
        
        Args:
            expiration_date: Option expiration date
            current_date: Current date (default: now)
            
        Returns:
            Time to expiry in years
        """
        if current_date is None:
            current_date = datetime.now()
        
        days_to_expiry = (expiration_date - current_date).days
        
        if days_to_expiry < 0:
            days_to_expiry = 0
        
        time_to_expiry = days_to_expiry / 365.0
        
        return time_to_expiry
    
    def calculate_intrinsic_value(
        self,
        option_type: str,
        stock_price: float,
        strike: float
    ) -> float:
        """
        Calculate intrinsic value
        
        Args:
            option_type: 'CALL' or 'PUT'
            stock_price: Current stock price
            strike: Strike price
            
        Returns:
            Intrinsic value
        """
        if option_type.upper() == 'CALL':
            intrinsic = max(0, stock_price - strike)
        else:  # PUT
            intrinsic = max(0, strike - stock_price)
        
        return intrinsic
    
    def calculate_extrinsic_value(
        self,
        option_price: float,
        intrinsic_value: float
    ) -> float:
        """
        Calculate extrinsic (time) value
        
        Args:
            option_price: Current option price
            intrinsic_value: Intrinsic value
            
        Returns:
            Extrinsic value
        """
        return max(0, option_price - intrinsic_value)
    
    def get_moneyness(
        self,
        option_type: str,
        stock_price: float,
        strike: float,
        threshold: float = 0.02
    ) -> str:
        """
        Determine if option is ITM, ATM, or OTM
        
        Args:
            option_type: 'CALL' or 'PUT'
            stock_price: Current stock price
            strike: Strike price
            threshold: ATM threshold (default 2%)
            
        Returns:
            'ITM', 'ATM', or 'OTM'
        """
        ratio = stock_price / strike
        
        if option_type.upper() == 'CALL':
            if ratio > (1 + threshold):
                return 'ITM'
            elif ratio < (1 - threshold):
                return 'OTM'
            else:
                return 'ATM'
        else:  # PUT
            if ratio < (1 - threshold):
                return 'ITM'
            elif ratio > (1 + threshold):
                return 'OTM'
            else:
                return 'ATM'
    
    def calculate_probability_itm(
        self,
        option_type: str,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        Calculate probability of finishing in-the-money
        
        Args:
            option_type: 'CALL' or 'PUT'
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            
        Returns:
            Probability (0-1)
        """
        _, d2 = self.calculate_d1_d2(stock_price, strike, time_to_expiry,
                                      volatility, self.risk_free_rate)
        
        if option_type.upper() == 'CALL':
            prob_itm = norm.cdf(d2)
        else:  # PUT
            prob_itm = norm.cdf(-d2)
        
        return prob_itm

