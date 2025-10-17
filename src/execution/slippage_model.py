"""
Advanced Slippage and Execution Cost Modeling
Realistic transaction cost modeling for options trading

Features:
- Bid-ask spread modeling by symbol, strike, DTE
- Market impact modeling based on position size vs. volume
- Time-of-day execution cost variations
- Volatility-adjusted execution costs
- Liquidity tier classification
- Historical slippage analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import yaml
from loguru import logger
from scipy import stats
import sqlite3

class LiquidityTier(Enum):
    """Options liquidity tiers"""
    TIER_1 = "tier_1"    # SPY, QQQ - High volume, tight spreads
    TIER_2 = "tier_2"    # IWM, DIA - Medium volume, moderate spreads
    TIER_3 = "tier_3"    # Sector ETFs - Lower volume, wider spreads
    TIER_4 = "tier_4"    # Individual stocks - Low volume, wide spreads

class TimeOfDay(Enum):
    """Market session periods"""
    PRE_MARKET = "pre_market"      # 4:00 AM - 9:30 AM
    MARKET_OPEN = "market_open"    # 9:30 AM - 10:30 AM
    MID_DAY = "mid_day"           # 10:30 AM - 2:00 PM
    MARKET_CLOSE = "market_close"  # 2:00 PM - 4:00 PM
    AFTER_HOURS = "after_hours"   # 4:00 PM - 8:00 PM

@dataclass
class ExecutionCost:
    """Execution cost breakdown"""
    symbol: str
    contract_type: str
    strike: float
    expiration: str
    quantity: int
    market_price: float
    execution_price: float
    bid_ask_spread: float
    market_impact: float
    timing_cost: float
    total_slippage: float
    commission: float
    regulatory_fees: float
    total_cost: float
    cost_per_contract: float
    timestamp: datetime

@dataclass
class LiquidityMetrics:
    """Liquidity metrics for options contract"""
    symbol: str
    strike: float
    expiration: str
    avg_volume: float
    avg_open_interest: float
    avg_bid_ask_spread: float
    spread_percentile: float
    volume_percentile: float
    liquidity_score: float
    tier: LiquidityTier

class SlippageModel:
    """Advanced slippage and execution cost model"""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config = self._load_config(config_path)
        self.slippage_config = self.config.get('slippage_modeling', {})
        
        # Liquidity tier definitions
        self.liquidity_tiers = {
            LiquidityTier.TIER_1: {
                'symbols': ['SPY', 'QQQ'],
                'min_volume': 10000,
                'max_spread_percent': 2.0,
                'impact_multiplier': 0.5
            },
            LiquidityTier.TIER_2: {
                'symbols': ['IWM', 'DIA', 'XLF', 'XLK'],
                'min_volume': 5000,
                'max_spread_percent': 3.5,
                'impact_multiplier': 0.8
            },
            LiquidityTier.TIER_3: {
                'symbols': ['XLE', 'XLV', 'XLI', 'XLY'],
                'min_volume': 2000,
                'max_spread_percent': 5.0,
                'impact_multiplier': 1.2
            },
            LiquidityTier.TIER_4: {
                'symbols': [],  # Individual stocks
                'min_volume': 500,
                'max_spread_percent': 10.0,
                'impact_multiplier': 2.0
            }
        }
        
        # Time-of-day cost multipliers
        self.time_multipliers = {
            TimeOfDay.PRE_MARKET: 1.5,
            TimeOfDay.MARKET_OPEN: 1.8,
            TimeOfDay.MID_DAY: 1.0,
            TimeOfDay.MARKET_CLOSE: 1.6,
            TimeOfDay.AFTER_HOURS: 2.0
        }
        
        # Historical data storage
        self.liquidity_data = {}
        self.slippage_history = []
        
        logger.info("Slippage model initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def classify_liquidity_tier(self, symbol: str, avg_volume: float, avg_spread: float) -> LiquidityTier:
        """Classify options liquidity tier"""
        for tier, criteria in self.liquidity_tiers.items():
            if symbol in criteria['symbols']:
                return tier
            elif avg_volume >= criteria['min_volume'] and avg_spread <= criteria['max_spread_percent']:
                return tier
        
        return LiquidityTier.TIER_4  # Default to lowest tier
    
    def get_time_of_day(self, timestamp: datetime) -> TimeOfDay:
        """Determine time of day for execution cost calculation"""
        hour = timestamp.hour
        minute = timestamp.minute
        
        if 4 <= hour < 9 or (hour == 9 and minute < 30):
            return TimeOfDay.PRE_MARKET
        elif hour == 9 and minute >= 30:
            return TimeOfDay.MARKET_OPEN
        elif 10 <= hour < 14 or (hour == 14 and minute == 0):
            return TimeOfDay.MID_DAY
        elif hour == 14 and minute > 0 or hour == 15:
            return TimeOfDay.MARKET_CLOSE
        else:
            return TimeOfDay.AFTER_HOURS
    
    def calculate_bid_ask_slippage(self, symbol: str, strike: float, expiration: str, 
                                  quantity: int, market_price: float, vix: float = 20.0) -> float:
        """Calculate bid-ask spread slippage"""
        try:
            # Base spread calculation
            base_spread = self._get_base_spread(symbol, strike, expiration)
            
            # VIX adjustment (higher VIX = wider spreads)
            vix_adjustment = 1.0 + (vix - 20.0) * 0.02  # 2% per VIX point above 20
            
            # Liquidity tier adjustment
            tier = self.classify_liquidity_tier(symbol, 10000, base_spread)
            tier_multiplier = self.liquidity_tiers[tier]['impact_multiplier']
            
            # Calculate expected slippage
            expected_spread = base_spread * vix_adjustment * tier_multiplier
            
            # Quantity impact (larger orders = more slippage)
            quantity_impact = 1.0 + (quantity - 1) * 0.05  # 5% per additional contract
            
            total_slippage = expected_spread * quantity_impact
            
            return min(total_slippage, market_price * 0.1)  # Cap at 10% of market price
            
        except Exception as e:
            logger.error(f"Error calculating bid-ask slippage: {e}")
            return market_price * 0.05  # Default 5% slippage
    
    def _get_base_spread(self, symbol: str, strike: float, expiration: str) -> float:
        """Get base bid-ask spread for options contract"""
        # Simplified base spread calculation
        # In production, this would use historical data
        
        base_spreads = {
            'SPY': 0.02,  # $0.02 base spread
            'QQQ': 0.03,  # $0.03 base spread
            'IWM': 0.05,  # $0.05 base spread
        }
        
        base_spread = base_spreads.get(symbol, 0.10)  # Default $0.10
        
        # Adjust for strike proximity to current price
        # Assume current price around $400 for SPY/QQQ
        current_price = 400.0
        strike_ratio = strike / current_price
        
        if 0.95 <= strike_ratio <= 1.05:  # Near-the-money
            return base_spread
        elif 0.90 <= strike_ratio <= 1.10:  # Close-to-money
            return base_spread * 1.2
        else:  # Far out-of-money
            return base_spread * 2.0
    
    def calculate_market_impact(self, symbol: str, quantity: int, avg_volume: float, 
                               vix: float = 20.0) -> float:
        """Calculate market impact based on order size vs. liquidity"""
        try:
            # Volume ratio (our order size vs. average volume)
            volume_ratio = quantity / max(avg_volume, 1)
            
            # Base impact calculation (square root model)
            base_impact = np.sqrt(volume_ratio) * 0.01  # 1% base impact
            
            # VIX adjustment
            vix_adjustment = 1.0 + (vix - 20.0) * 0.03  # 3% per VIX point
            
            # Liquidity tier adjustment
            tier = self.classify_liquidity_tier(symbol, avg_volume, 2.0)
            tier_multiplier = self.liquidity_tiers[tier]['impact_multiplier']
            
            total_impact = base_impact * vix_adjustment * tier_multiplier
            
            return min(total_impact, 0.20)  # Cap at 20% impact
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.05  # Default 5% impact
    
    def calculate_timing_cost(self, timestamp: datetime, volatility_regime: str = "normal") -> float:
        """Calculate timing-based execution cost"""
        time_of_day = self.get_time_of_day(timestamp)
        base_multiplier = self.time_multipliers[time_of_day]
        
        # Volatility regime adjustment
        volatility_multipliers = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.5,
            "extreme": 2.0
        }
        
        volatility_multiplier = volatility_multipliers.get(volatility_regime, 1.0)
        
        return base_multiplier * volatility_multiplier
    
    def calculate_total_execution_cost(self, symbol: str, contract_type: str, strike: float,
                                     expiration: str, quantity: int, market_price: float,
                                     avg_volume: float = 10000, vix: float = 20.0,
                                     timestamp: Optional[datetime] = None) -> ExecutionCost:
        """Calculate total execution cost for an order"""
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Calculate individual cost components
            bid_ask_slippage = self.calculate_bid_ask_slippage(
                symbol, strike, expiration, quantity, market_price, vix
            )
            
            market_impact = self.calculate_market_impact(
                symbol, quantity, avg_volume, vix
            )
            
            timing_cost_multiplier = self.calculate_timing_cost(timestamp)
            
            # Calculate execution price
            execution_price = market_price + bid_ask_slippage + (market_price * market_impact)
            
            # Calculate total slippage
            total_slippage = execution_price - market_price
            
            # Calculate commission and fees
            commission = quantity * 0.65  # $0.65 per contract
            regulatory_fees = quantity * 0.000119  # Regulatory fees
            
            # Calculate total cost
            total_cost = total_slippage + commission + regulatory_fees
            cost_per_contract = total_cost / quantity if quantity > 0 else 0
            
            return ExecutionCost(
                symbol=symbol,
                contract_type=contract_type,
                strike=strike,
                expiration=expiration,
                quantity=quantity,
                market_price=market_price,
                execution_price=execution_price,
                bid_ask_spread=bid_ask_slippage,
                market_impact=market_price * market_impact,
                timing_cost=timing_cost_multiplier,
                total_slippage=total_slippage,
                commission=commission,
                regulatory_fees=regulatory_fees,
                total_cost=total_cost,
                cost_per_contract=cost_per_contract,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Error calculating execution cost: {e}")
            # Return default cost structure
            return ExecutionCost(
                symbol=symbol,
                contract_type=contract_type,
                strike=strike,
                expiration=expiration,
                quantity=quantity,
                market_price=market_price,
                execution_price=market_price * 1.05,  # 5% default slippage
                bid_ask_spread=market_price * 0.03,
                market_impact=market_price * 0.02,
                timing_cost=1.0,
                total_slippage=market_price * 0.05,
                commission=quantity * 0.65,
                regulatory_fees=quantity * 0.000119,
                total_cost=market_price * 0.05 + (quantity * 0.65) + (quantity * 0.000119),
                cost_per_contract=(market_price * 0.05 + (quantity * 0.65) + (quantity * 0.000119)) / quantity,
                timestamp=timestamp
            )
    
    def analyze_liquidity(self, symbol: str, strike: float, expiration: str,
                         historical_data: pd.DataFrame) -> LiquidityMetrics:
        """Analyze liquidity metrics for options contract"""
        try:
            # Filter data for specific contract
            contract_data = historical_data[
                (historical_data['symbol'] == symbol) &
                (historical_data['strike'] == strike) &
                (historical_data['expiration'] == expiration)
            ].copy()
            
            if contract_data.empty:
                return self._get_default_liquidity_metrics(symbol, strike, expiration)
            
            # Calculate metrics
            avg_volume = contract_data['volume'].mean()
            avg_open_interest = contract_data['open_interest'].mean()
            avg_spread = contract_data['bid_ask_spread'].mean()
            
            # Calculate percentiles
            spread_percentile = stats.percentileofscore(
                contract_data['bid_ask_spread'], avg_spread
            )
            volume_percentile = stats.percentileofscore(
                contract_data['volume'], avg_volume
            )
            
            # Calculate liquidity score (0-100)
            liquidity_score = (volume_percentile * 0.6 + (100 - spread_percentile) * 0.4)
            
            # Classify tier
            tier = self.classify_liquidity_tier(symbol, avg_volume, avg_spread)
            
            return LiquidityMetrics(
                symbol=symbol,
                strike=strike,
                expiration=expiration,
                avg_volume=avg_volume,
                avg_open_interest=avg_open_interest,
                avg_bid_ask_spread=avg_spread,
                spread_percentile=spread_percentile,
                volume_percentile=volume_percentile,
                liquidity_score=liquidity_score,
                tier=tier
            )
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {e}")
            return self._get_default_liquidity_metrics(symbol, strike, expiration)
    
    def _get_default_liquidity_metrics(self, symbol: str, strike: float, expiration: str) -> LiquidityMetrics:
        """Get default liquidity metrics when data is unavailable"""
        return LiquidityMetrics(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            avg_volume=1000,
            avg_open_interest=5000,
            avg_bid_ask_spread=0.05,
            spread_percentile=50.0,
            volume_percentile=50.0,
            liquidity_score=50.0,
            tier=LiquidityTier.TIER_4
        )
    
    def optimize_execution_timing(self, symbol: str, quantity: int, 
                                preferred_times: List[datetime]) -> datetime:
        """Optimize execution timing to minimize costs"""
        try:
            best_time = preferred_times[0]
            best_cost = float('inf')
            
            for time_candidate in preferred_times:
                # Calculate cost for this time
                cost_multiplier = self.calculate_timing_cost(time_candidate)
                
                if cost_multiplier < best_cost:
                    best_cost = cost_multiplier
                    best_time = time_candidate
            
            return best_time
            
        except Exception as e:
            logger.error(f"Error optimizing execution timing: {e}")
            return datetime.now()
    
    def estimate_execution_quality(self, symbol: str, quantity: int, 
                                  market_price: float, execution_price: float) -> Dict[str, float]:
        """Estimate execution quality metrics"""
        try:
            # Calculate slippage
            slippage = execution_price - market_price
            slippage_percent = (slippage / market_price) * 100
            
            # Get expected slippage for comparison
            expected_cost = self.calculate_total_execution_cost(
                symbol, "PUT", 400.0, "2024-01-19", quantity, market_price
            )
            expected_slippage_percent = (expected_cost.total_slippage / market_price) * 100
            
            # Calculate execution quality score (0-100)
            if expected_slippage_percent > 0:
                quality_score = max(0, 100 - (slippage_percent / expected_slippage_percent) * 100)
            else:
                quality_score = 50  # Default score
            
            return {
                'slippage_percent': slippage_percent,
                'expected_slippage_percent': expected_slippage_percent,
                'execution_quality_score': quality_score,
                'slippage_ratio': slippage_percent / expected_slippage_percent if expected_slippage_percent > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error estimating execution quality: {e}")
            return {
                'slippage_percent': 0.0,
                'expected_slippage_percent': 0.0,
                'execution_quality_score': 50.0,
                'slippage_ratio': 1.0
            }
    
    def get_execution_recommendations(self, symbol: str, quantity: int, 
                                    urgency: str = "normal") -> Dict[str, Any]:
        """Get execution recommendations based on market conditions"""
        try:
            recommendations = {
                'order_type': 'LIMIT',
                'time_in_force': 'DAY',
                'execution_strategy': 'TWAP',  # Time-Weighted Average Price
                'max_slippage': 0.05,  # 5% max slippage
                'timing_recommendation': 'MID_DAY'
            }
            
            # Adjust based on urgency
            if urgency == "high":
                recommendations['order_type'] = 'MARKET'
                recommendations['execution_strategy'] = 'IMMEDIATE'
                recommendations['max_slippage'] = 0.10  # 10% max slippage
            elif urgency == "low":
                recommendations['time_in_force'] = 'GTC'  # Good Till Cancel
                recommendations['execution_strategy'] = 'VWAP'  # Volume-Weighted Average Price
                recommendations['max_slippage'] = 0.03  # 3% max slippage
            
            # Adjust based on liquidity tier
            tier = self.classify_liquidity_tier(symbol, 10000, 2.0)
            if tier == LiquidityTier.TIER_4:
                recommendations['execution_strategy'] = 'ICE'  # Iceberg orders
                recommendations['max_slippage'] = 0.15  # 15% max slippage
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting execution recommendations: {e}")
            return {
                'order_type': 'LIMIT',
                'time_in_force': 'DAY',
                'execution_strategy': 'TWAP',
                'max_slippage': 0.05,
                'timing_recommendation': 'MID_DAY'
            }

# Example usage and testing
def main():
    """Test the slippage model"""
    model = SlippageModel()
    
    # Test execution cost calculation
    cost = model.calculate_total_execution_cost(
        symbol="SPY",
        contract_type="PUT",
        strike=400.0,
        expiration="2024-01-19",
        quantity=10,
        market_price=5.0,
        avg_volume=15000,
        vix=22.0
    )
    
    print(f"Execution Cost Analysis:")
    print(f"Market Price: ${cost.market_price:.2f}")
    print(f"Execution Price: ${cost.execution_price:.2f}")
    print(f"Total Slippage: ${cost.total_slippage:.2f}")
    print(f"Commission: ${cost.commission:.2f}")
    print(f"Total Cost: ${cost.total_cost:.2f}")
    print(f"Cost per Contract: ${cost.cost_per_contract:.2f}")
    
    # Test liquidity analysis
    liquidity = model.analyze_liquidity("SPY", 400.0, "2024-01-19", pd.DataFrame())
    print(f"\nLiquidity Analysis:")
    print(f"Tier: {liquidity.tier.value}")
    print(f"Liquidity Score: {liquidity.liquidity_score:.1f}")
    
    # Test execution recommendations
    recommendations = model.get_execution_recommendations("SPY", 10, "normal")
    print(f"\nExecution Recommendations:")
    print(f"Order Type: {recommendations['order_type']}")
    print(f"Strategy: {recommendations['execution_strategy']}")
    print(f"Max Slippage: {recommendations['max_slippage']:.1%}")

if __name__ == "__main__":
    main()
