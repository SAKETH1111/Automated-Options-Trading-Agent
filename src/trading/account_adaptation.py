#!/usr/bin/env python3
"""
Account Size-Based Trading Adaptation System

This module adapts the trading agent's behavior based on account size:
- Symbol selection (high-cap vs small-cap)
- Risk management parameters
- Position sizing
- API usage optimization
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from datetime import datetime, timedelta

@dataclass
class AccountTier:
    """Account tier configuration"""
    name: str
    min_balance: float
    max_balance: float
    symbols: List[str]
    max_positions: int
    max_position_size: float
    risk_per_trade: float
    preferred_expiry_days: Tuple[int, int]
    min_volume: int
    min_open_interest: int

class AccountAdaptationSystem:
    """
    Adapts trading behavior based on account size and risk tolerance
    """
    
    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.current_tier = self._determine_account_tier()
        self.symbol_cache = {}
        self.last_update = None
        
        logger.info(f"Account Adaptation System initialized")
        logger.info(f"Account Balance: ${account_balance:,.2f}")
        logger.info(f"Account Tier: {self.current_tier.name}")
    
    def _determine_account_tier(self) -> AccountTier:
        """Determine account tier based on balance"""
        
        # Define account tiers
        tiers = [
            AccountTier(
                name="Micro Account",
                min_balance=0,
                max_balance=1000,
                symbols=["SPY", "QQQ", "IWM"],  # High liquidity ETFs
                max_positions=2,
                max_position_size=0.1,  # 10% of account
                risk_per_trade=0.02,   # 2% risk per trade
                preferred_expiry_days=(7, 30),
                min_volume=1000,
                min_open_interest=500
            ),
            AccountTier(
                name="Small Account",
                min_balance=1000,
                max_balance=10000,
                symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA"],
                max_positions=3,
                max_position_size=0.15,
                risk_per_trade=0.025,
                preferred_expiry_days=(14, 45),
                min_volume=2000,
                min_open_interest=1000
            ),
            AccountTier(
                name="Medium Account",
                min_balance=10000,
                max_balance=50000,
                symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"],
                max_positions=5,
                max_position_size=0.2,
                risk_per_trade=0.03,
                preferred_expiry_days=(21, 60),
                min_volume=5000,
                min_open_interest=2000
            ),
            AccountTier(
                name="Large Account",
                min_balance=50000,
                max_balance=250000,
                symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", 
                        "NFLX", "ADBE", "CRM", "PYPL", "INTC", "AMD", "ORCL", "CSCO", "IBM", "V"],
                max_positions=8,
                max_position_size=0.25,
                risk_per_trade=0.035,
                preferred_expiry_days=(30, 90),
                min_volume=10000,
                min_open_interest=5000
            ),
            AccountTier(
                name="Institutional Account",
                min_balance=250000,
                max_balance=float('inf'),
                symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META",
                        "NFLX", "ADBE", "CRM", "PYPL", "INTC", "AMD", "ORCL", "CSCO", "IBM", "V",
                        "JPM", "JNJ", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "WMT"],
                max_positions=12,
                max_position_size=0.3,
                risk_per_trade=0.04,
                preferred_expiry_days=(45, 120),
                min_volume=20000,
                min_open_interest=10000
            )
        ]
        
        # Find appropriate tier
        for tier in tiers:
            if tier.min_balance <= self.account_balance < tier.max_balance:
                return tier
        
        # Default to highest tier if balance exceeds all
        return tiers[-1]
    
    def get_recommended_symbols(self, limit: Optional[int] = None) -> List[str]:
        """Get recommended symbols based on account tier"""
        symbols = self.current_tier.symbols.copy()
        
        if limit:
            symbols = symbols[:limit]
        
        logger.info(f"Recommended symbols for {self.current_tier.name}: {symbols}")
        return symbols
    
    def get_position_limits(self) -> Dict[str, int]:
        """Get position limits based on account tier"""
        return {
            "max_positions": self.current_tier.max_positions,
            "max_position_size": self.current_tier.max_position_size,
            "risk_per_trade": self.current_tier.risk_per_trade
        }
    
    def get_options_criteria(self) -> Dict[str, any]:
        """Get options selection criteria based on account tier"""
        return {
            "min_volume": self.current_tier.min_volume,
            "min_open_interest": self.current_tier.min_open_interest,
            "preferred_expiry_days": self.current_tier.preferred_expiry_days,
            "min_delta": 0.1 if self.account_balance < 10000 else 0.05,
            "max_delta": 0.9 if self.account_balance < 10000 else 0.95
        }
    
    def calculate_position_size(self, option_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management rules"""
        risk_amount = self.account_balance * self.current_tier.risk_per_trade
        risk_per_contract = abs(option_price - stop_loss)
        
        if risk_per_contract <= 0:
            return 0
        
        position_size = int(risk_amount / risk_per_contract)
        max_position_value = self.account_balance * self.current_tier.max_position_size
        max_contracts = int(max_position_value / option_price)
        
        return min(position_size, max_contracts)
    
    def get_api_usage_limits(self) -> Dict[str, int]:
        """Get API usage limits based on account tier"""
        # These would typically come from your plan, but we'll set reasonable defaults
        base_limits = {
            "requests_per_minute": 5,
            "requests_per_day": 1000,
            "concurrent_requests": 2
        }
        
        # Scale based on account tier
        multiplier = {
            "Micro Account": 1.0,
            "Small Account": 1.5,
            "Medium Account": 2.0,
            "Large Account": 3.0,
            "Institutional Account": 5.0
        }.get(self.current_tier.name, 1.0)
        
        return {
            key: int(value * multiplier) 
            for key, value in base_limits.items()
        }
    
    def should_update_tier(self) -> bool:
        """Check if account tier should be updated"""
        if not self.last_update:
            return True
        
        # Update every hour
        return datetime.now() - self.last_update > timedelta(hours=1)
    
    def update_account_balance(self, new_balance: float):
        """Update account balance and tier if needed"""
        self.account_balance = new_balance
        
        if self.should_update_tier():
            old_tier = self.current_tier.name
            self.current_tier = self._determine_account_tier()
            self.last_update = datetime.now()
            
            if old_tier != self.current_tier.name:
                logger.info(f"Account tier updated: {old_tier} -> {self.current_tier.name}")
    
    def get_trading_schedule(self) -> Dict[str, str]:
        """Get trading schedule based on account tier"""
        # Smaller accounts might trade more frequently
        if self.account_balance < 10000:
            return {
                "market_open": "09:30",
                "market_close": "16:00",
                "scan_frequency": "5min",
                "rebalance_frequency": "daily"
            }
        else:
            return {
                "market_open": "09:30",
                "market_close": "16:00", 
                "scan_frequency": "15min",
                "rebalance_frequency": "daily"
            }
    
    def get_ml_training_config(self) -> Dict[str, any]:
        """Get ML training configuration based on account tier"""
        return {
            "retrain_frequency": "daily",
            "lookback_days": min(30, max(7, int(self.account_balance / 1000))),
            "min_data_points": max(100, int(self.account_balance / 100)),
            "model_complexity": "medium" if self.account_balance < 50000 else "high",
            "validation_split": 0.2,
            "test_split": 0.1
        }
    
    def get_risk_management_rules(self) -> Dict[str, any]:
        """Get risk management rules based on account tier"""
        return {
            "max_daily_loss": self.account_balance * 0.05,  # 5% max daily loss
            "max_portfolio_risk": self.account_balance * 0.15,  # 15% max portfolio risk
            "stop_loss_percentage": 0.5 if self.account_balance < 10000 else 0.3,
            "take_profit_percentage": 1.0 if self.account_balance < 10000 else 1.5,
            "max_correlation": 0.7,
            "position_sizing_method": "fixed_risk" if self.account_balance < 50000 else "kelly_criterion"
        }

def create_account_adaptation(account_balance: float) -> AccountAdaptationSystem:
    """Factory function to create account adaptation system"""
    return AccountAdaptationSystem(account_balance)

# Example usage
if __name__ == "__main__":
    # Test with different account sizes
    test_balances = [500, 5000, 25000, 100000, 500000]
    
    for balance in test_balances:
        print(f"\n{'='*50}")
        print(f"Testing with account balance: ${balance:,}")
        print(f"{'='*50}")
        
        adaptation = create_account_adaptation(balance)
        
        print(f"Account Tier: {adaptation.current_tier.name}")
        print(f"Recommended Symbols: {adaptation.get_recommended_symbols(5)}")
        print(f"Max Positions: {adaptation.get_position_limits()['max_positions']}")
        print(f"Risk per Trade: {adaptation.get_position_limits()['risk_per_trade']:.1%}")
        print(f"Options Criteria: {adaptation.get_options_criteria()}")

