"""
Universal Account Manager
Automatically adapts trading universe, strategies, and risk parameters based on account size
Scales from $100 (micro) to $10M+ (institutional)
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from loguru import logger

from src.config.settings import get_config


class AccountTier(Enum):
    """Account size tiers"""
    MICRO = "MICRO"              # $100 - $999
    SMALL = "SMALL"              # $1K - $9,999
    MEDIUM = "MEDIUM"            # $10K - $99,999
    LARGE = "LARGE"              # $100K - $999,999
    INSTITUTIONAL = "INSTITUTIONAL"  # $1M+


@dataclass
class AccountProfile:
    """Complete account profile with adaptive parameters"""
    tier: AccountTier
    balance: float
    buying_power: float
    
    # Trading universe
    allowed_symbols: List[str]
    max_symbols_active: int
    
    # Strategy configuration
    enabled_strategies: List[str]
    max_positions: int
    max_positions_per_symbol: int
    
    # Risk parameters
    max_position_size_pct: float
    max_daily_loss_pct: float
    max_portfolio_heat: float
    stop_loss_pct: float
    take_profit_pct: float
    
    # Execution parameters
    execution_frequency: str  # 'eod', 'daily', 'intraday', 'continuous'
    min_dte: int
    max_dte: int
    
    # Options parameters
    min_option_volume: int
    min_open_interest: int
    max_bid_ask_spread_pct: float
    delta_range: Tuple[float, float]
    
    # Costs
    commission_per_contract: float
    slippage_multiplier: float
    
    # Greeks limits
    max_portfolio_delta: float
    max_portfolio_gamma: float
    max_portfolio_vega: float
    max_portfolio_theta: float


class UniversalAccountManager:
    """
    Universal account manager that adapts all trading parameters based on account size
    
    Features:
    - Automatic tier classification
    - Dynamic symbol universe (SPY/QQQ only for micro → 500+ symbols for institutional)
    - Strategy enablement based on account size
    - Risk parameter scaling
    - Liquidity filtering by tier
    - Greeks limits by account size
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        
        # Account tier definitions
        self.tier_thresholds = {
            AccountTier.MICRO: (100, 999),
            AccountTier.SMALL: (1000, 9999),
            AccountTier.MEDIUM: (10000, 99999),
            AccountTier.LARGE: (100000, 999999),
            AccountTier.INSTITUTIONAL: (1000000, float('inf'))
        }
        
        # Symbol universes by tier
        self.symbol_universes = self._initialize_symbol_universes()
        
        # Strategy configurations by tier
        self.strategy_configs = self._initialize_strategy_configs()
        
        # Current account profile
        self.current_profile = None
        
        logger.info("UniversalAccountManager initialized")
    
    def _initialize_symbol_universes(self) -> Dict[AccountTier, Dict]:
        """Initialize symbol universes for each account tier"""
        return {
            AccountTier.MICRO: {
                'symbols': ['SPY', 'QQQ'],
                'max_active': 1,
                'focus': 'Ultra liquid only - SPY/QQQ',
                'min_avg_volume': 1000000,  # 1M+ daily volume
                'min_market_cap': 100_000_000_000  # $100B+
            },
            AccountTier.SMALL: {
                'symbols': ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA'],
                'max_active': 2,
                'focus': 'Major indices + mega cap tech',
                'min_avg_volume': 500000,
                'min_market_cap': 50_000_000_000  # $50B+
            },
            AccountTier.MEDIUM: {
                'symbols': self._get_top_50_liquid_symbols(),
                'max_active': 5,
                'focus': 'Top 50 most liquid stocks + indices',
                'min_avg_volume': 200000,
                'min_market_cap': 10_000_000_000  # $10B+
            },
            AccountTier.LARGE: {
                'symbols': self._get_top_200_liquid_symbols(),
                'max_active': 10,
                'focus': 'Top 200 liquid stocks, full sector coverage',
                'min_avg_volume': 100000,
                'min_market_cap': 5_000_000_000  # $5B+
            },
            AccountTier.INSTITUTIONAL: {
                'symbols': self._get_full_universe(),
                'max_active': 20,
                'focus': 'Full tradeable universe (500+ symbols)',
                'min_avg_volume': 50000,
                'min_market_cap': 1_000_000_000  # $1B+
            }
        }
    
    def _initialize_strategy_configs(self) -> Dict[AccountTier, Dict]:
        """Initialize strategy configurations for each tier"""
        return {
            AccountTier.MICRO: {
                'enabled_strategies': [
                    'bull_put_spread',
                    'cash_secured_put'
                ],
                'strategy_complexity': 'simple',
                'max_legs_per_trade': 2,
                'focus': 'High probability credit strategies only'
            },
            AccountTier.SMALL: {
                'enabled_strategies': [
                    'bull_put_spread',
                    'bear_call_spread',
                    'cash_secured_put',
                    'iron_condor'
                ],
                'strategy_complexity': 'basic',
                'max_legs_per_trade': 4,
                'focus': 'Credit spreads and simple iron condors'
            },
            AccountTier.MEDIUM: {
                'enabled_strategies': [
                    'bull_put_spread',
                    'bear_call_spread',
                    'cash_secured_put',
                    'iron_condor',
                    'iron_butterfly',
                    'calendar_spread',
                    'diagonal_spread'
                ],
                'strategy_complexity': 'intermediate',
                'max_legs_per_trade': 4,
                'focus': 'Multi-strategy portfolio with calendars'
            },
            AccountTier.LARGE: {
                'enabled_strategies': [
                    'bull_put_spread',
                    'bear_call_spread',
                    'cash_secured_put',
                    'iron_condor',
                    'iron_butterfly',
                    'calendar_spread',
                    'diagonal_spread',
                    'straddle',
                    'strangle',
                    'ratio_spread'
                ],
                'strategy_complexity': 'advanced',
                'max_legs_per_trade': 6,
                'focus': 'Advanced multi-leg strategies with Greeks management'
            },
            AccountTier.INSTITUTIONAL: {
                'enabled_strategies': [
                    'bull_put_spread',
                    'bear_call_spread',
                    'cash_secured_put',
                    'iron_condor',
                    'iron_butterfly',
                    'calendar_spread',
                    'diagonal_spread',
                    'straddle',
                    'strangle',
                    'ratio_spread',
                    'delta_neutral_portfolio',
                    'gamma_scalping',
                    'volatility_arbitrage',
                    'dispersion_trading'
                ],
                'strategy_complexity': 'institutional',
                'max_legs_per_trade': 8,
                'focus': 'Full strategy suite with advanced Greeks arbitrage'
            }
        }
    
    def classify_account(self, balance: float) -> AccountTier:
        """Classify account into appropriate tier based on balance"""
        for tier, (min_balance, max_balance) in self.tier_thresholds.items():
            if min_balance <= balance <= max_balance:
                return tier
        
        # Default to micro if below minimum
        if balance < 100:
            logger.warning(f"Account balance ${balance:.2f} below minimum, classified as MICRO")
            return AccountTier.MICRO
        
        # Default to institutional if above maximum
        return AccountTier.INSTITUTIONAL
    
    def create_account_profile(
        self, 
        balance: float, 
        buying_power: float = None
    ) -> AccountProfile:
        """
        Create complete account profile with all adaptive parameters
        
        Args:
            balance: Current account balance
            buying_power: Available buying power (defaults to balance)
        
        Returns:
            AccountProfile with all parameters configured for account size
        """
        try:
            # Classify account tier
            tier = self.classify_account(balance)
            
            # Default buying power
            if buying_power is None:
                buying_power = balance
            
            # Get universe and strategy configs
            universe_config = self.symbol_universes[tier]
            strategy_config = self.strategy_configs[tier]
            
            # Build risk parameters (scaled by tier)
            risk_params = self._get_risk_parameters(tier, balance)
            
            # Build execution parameters
            execution_params = self._get_execution_parameters(tier)
            
            # Build options parameters
            options_params = self._get_options_parameters(tier, balance)
            
            # Build cost parameters
            cost_params = self._get_cost_parameters(tier)
            
            # Build Greeks limits
            greeks_limits = self._get_greeks_limits(tier, balance)
            
            # Create profile
            profile = AccountProfile(
                tier=tier,
                balance=balance,
                buying_power=buying_power,
                
                # Universe
                allowed_symbols=universe_config['symbols'],
                max_symbols_active=universe_config['max_active'],
                
                # Strategies
                enabled_strategies=strategy_config['enabled_strategies'],
                max_positions=risk_params['max_positions'],
                max_positions_per_symbol=risk_params['max_positions_per_symbol'],
                
                # Risk
                max_position_size_pct=risk_params['max_position_size_pct'],
                max_daily_loss_pct=risk_params['max_daily_loss_pct'],
                max_portfolio_heat=risk_params['max_portfolio_heat'],
                stop_loss_pct=risk_params['stop_loss_pct'],
                take_profit_pct=risk_params['take_profit_pct'],
                
                # Execution
                execution_frequency=execution_params['frequency'],
                min_dte=execution_params['min_dte'],
                max_dte=execution_params['max_dte'],
                
                # Options
                min_option_volume=options_params['min_volume'],
                min_open_interest=options_params['min_oi'],
                max_bid_ask_spread_pct=options_params['max_spread_pct'],
                delta_range=options_params['delta_range'],
                
                # Costs
                commission_per_contract=cost_params['commission'],
                slippage_multiplier=cost_params['slippage_multiplier'],
                
                # Greeks
                max_portfolio_delta=greeks_limits['max_delta'],
                max_portfolio_gamma=greeks_limits['max_gamma'],
                max_portfolio_vega=greeks_limits['max_vega'],
                max_portfolio_theta=greeks_limits['max_theta']
            )
            
            # Store current profile
            self.current_profile = profile
            
            logger.info(f"Account profile created: {tier.value} tier (${balance:,.0f})")
            logger.info(f"  - Universe: {len(profile.allowed_symbols)} symbols, max {profile.max_symbols_active} active")
            logger.info(f"  - Strategies: {len(profile.enabled_strategies)} enabled")
            logger.info(f"  - Max positions: {profile.max_positions}")
            logger.info(f"  - Execution: {profile.execution_frequency}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating account profile: {e}")
            raise
    
    def _get_risk_parameters(self, tier: AccountTier, balance: float) -> Dict:
        """Get risk parameters for account tier"""
        params = {
            AccountTier.MICRO: {
                'max_positions': 1,
                'max_positions_per_symbol': 1,
                'max_position_size_pct': 50.0,  # 50% of account max
                'max_daily_loss_pct': 3.0,
                'max_portfolio_heat': 15.0,
                'stop_loss_pct': 50.0,
                'take_profit_pct': 50.0
            },
            AccountTier.SMALL: {
                'max_positions': 2,
                'max_positions_per_symbol': 1,
                'max_position_size_pct': 30.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_heat': 20.0,
                'stop_loss_pct': 50.0,
                'take_profit_pct': 50.0
            },
            AccountTier.MEDIUM: {
                'max_positions': 5,
                'max_positions_per_symbol': 2,
                'max_position_size_pct': 20.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_heat': 25.0,
                'stop_loss_pct': 50.0,
                'take_profit_pct': 50.0
            },
            AccountTier.LARGE: {
                'max_positions': 10,
                'max_positions_per_symbol': 3,
                'max_position_size_pct': 15.0,
                'max_daily_loss_pct': 7.0,
                'max_portfolio_heat': 30.0,
                'stop_loss_pct': 60.0,
                'take_profit_pct': 50.0
            },
            AccountTier.INSTITUTIONAL: {
                'max_positions': 20,
                'max_positions_per_symbol': 5,
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 10.0,
                'max_portfolio_heat': 35.0,
                'stop_loss_pct': 75.0,
                'take_profit_pct': 50.0
            }
        }
        
        return params[tier]
    
    def _get_execution_parameters(self, tier: AccountTier) -> Dict:
        """Get execution parameters for account tier"""
        params = {
            AccountTier.MICRO: {
                'frequency': 'eod',  # End of day (3:30 PM)
                'min_dte': 7,
                'max_dte': 60,
                'order_timeout_minutes': 60
            },
            AccountTier.SMALL: {
                'frequency': 'daily',  # 1-2x per day
                'min_dte': 7,
                'max_dte': 90,
                'order_timeout_minutes': 30
            },
            AccountTier.MEDIUM: {
                'frequency': 'intraday',  # 3-4x per day
                'min_dte': 5,
                'max_dte': 120,
                'order_timeout_minutes': 15
            },
            AccountTier.LARGE: {
                'frequency': 'frequent',  # Every 1-2 hours
                'min_dte': 3,
                'max_dte': 180,
                'order_timeout_minutes': 5
            },
            AccountTier.INSTITUTIONAL: {
                'frequency': 'continuous',  # Every 30-60 minutes
                'min_dte': 1,
                'max_dte': 365,
                'order_timeout_minutes': 2
            }
        }
        
        return params[tier]
    
    def _get_options_parameters(self, tier: AccountTier, balance: float) -> Dict:
        """Get options-specific parameters for account tier"""
        params = {
            AccountTier.MICRO: {
                'min_volume': 100,
                'min_oi': 100,
                'max_spread_pct': 15.0,
                'delta_range': (0.20, 0.40),  # Conservative deltas
                'max_option_price': 10.0
            },
            AccountTier.SMALL: {
                'min_volume': 50,
                'min_oi': 50,
                'max_spread_pct': 12.0,
                'delta_range': (0.15, 0.45),
                'max_option_price': 20.0
            },
            AccountTier.MEDIUM: {
                'min_volume': 25,
                'min_oi': 25,
                'max_spread_pct': 10.0,
                'delta_range': (0.10, 0.50),
                'max_option_price': 50.0
            },
            AccountTier.LARGE: {
                'min_volume': 10,
                'min_oi': 10,
                'max_spread_pct': 8.0,
                'delta_range': (0.05, 0.60),
                'max_option_price': 100.0
            },
            AccountTier.INSTITUTIONAL: {
                'min_volume': 5,
                'min_oi': 5,
                'max_spread_pct': 5.0,
                'delta_range': (0.01, 0.70),
                'max_option_price': 200.0
            }
        }
        
        return params[tier]
    
    def _get_cost_parameters(self, tier: AccountTier) -> Dict:
        """Get cost parameters for account tier"""
        params = {
            AccountTier.MICRO: {
                'commission': 2.00,  # $2.00 per contract round trip
                'slippage_multiplier': 1.5
            },
            AccountTier.SMALL: {
                'commission': 1.75,
                'slippage_multiplier': 1.3
            },
            AccountTier.MEDIUM: {
                'commission': 1.25,
                'slippage_multiplier': 1.0
            },
            AccountTier.LARGE: {
                'commission': 0.75,
                'slippage_multiplier': 0.8
            },
            AccountTier.INSTITUTIONAL: {
                'commission': 0.50,
                'slippage_multiplier': 0.7
            }
        }
        
        return params[tier]
    
    def _get_greeks_limits(self, tier: AccountTier, balance: float) -> Dict:
        """Get Greeks limits for account tier"""
        params = {
            AccountTier.MICRO: {
                'max_delta': 5,
                'max_gamma': 0.05,
                'max_vega': 10,
                'max_theta': 5
            },
            AccountTier.SMALL: {
                'max_delta': 20,
                'max_gamma': 0.2,
                'max_vega': 50,
                'max_theta': 20
            },
            AccountTier.MEDIUM: {
                'max_delta': 50,
                'max_gamma': 0.5,
                'max_vega': 150,
                'max_theta': 50
            },
            AccountTier.LARGE: {
                'max_delta': 100,
                'max_gamma': 1.0,
                'max_vega': 300,
                'max_theta': 100
            },
            AccountTier.INSTITUTIONAL: {
                'max_delta': 200,
                'max_gamma': 2.0,
                'max_vega': 500,
                'max_theta': 200
            }
        }
        
        return params[tier]
    
    def _get_top_50_liquid_symbols(self) -> List[str]:
        """Get top 50 most liquid symbols"""
        # Top 50 by options volume
        return [
            'SPY', 'QQQ', 'IWM', 'DIA',
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD',
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLY',
            'NFLX', 'DIS', 'PYPL', 'SQ', 'SHOP',
            'BA', 'CAT', 'GE', 'MMM',
            'PFE', 'JNJ', 'UNH', 'CVS',
            'XOM', 'CVX', 'COP',
            'V', 'MA', 'AXP',
            'COST', 'WMT', 'TGT', 'HD'
        ]
    
    def _get_top_200_liquid_symbols(self) -> List[str]:
        """Get top 200 most liquid symbols"""
        # Top 50 + another 150 liquid symbols
        top_50 = self._get_top_50_liquid_symbols()
        
        additional_150 = [
            'INTC', 'CSCO', 'ORCL', 'IBM', 'CRM', 'ADBE', 'QCOM', 'TXN', 'AVGO',
            'MU', 'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS',
            'C', 'USB', 'PNC', 'TFC', 'SCHW', 'BLK', 'SPGI', 'ICE', 'CME',
            'JD', 'BABA', 'PDD', 'NIO', 'XPEV', 'LI',
            'UBER', 'LYFT', 'ABNB', 'DASH',
            'ROKU', 'SPOT', 'SNAP', 'PINS', 'TWTR',
            'ZM', 'DOCU', 'OKTA', 'SNOW', 'DDOG', 'NET', 'CRWD',
            'SQ', 'COIN', 'HOOD', 'SOFI',
            'F', 'GM', 'RIVN', 'LCID',
            'DAL', 'UAL', 'AAL', 'LUV',
            'CCL', 'RCL', 'NCLH',
            'MGM', 'WYNN', 'LVS', 'CZR', 'PENN', 'DKNG',
            'MRK', 'ABBV', 'BMY', 'GILD', 'AMGN', 'REGN', 'VRTX', 'BIIB',
            'LLY', 'NVO', 'AZN', 'SNY',
            'ABT', 'TMO', 'DHR', 'ISRG', 'SYK', 'BSX', 'MDT', 'EW',
            'HON', 'RTX', 'LMT', 'NOC', 'GD',
            'DE', 'EMR', 'ITW', 'ROK', 'PH',
            'FCX', 'NEM', 'GOLD', 'AA', 'X', 'NUE', 'STLD',
            'DOW', 'DD', 'LYB', 'PPG', 'APD', 'ECL',
            'SLB', 'HAL', 'BKR', 'OXY', 'MPC', 'VLO', 'PSX',
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PCG',
            'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'SPG', 'O', 'WELL', 'AVB', 'EQR',
            'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'KHC', 'GIS', 'K',
            'PG', 'CL', 'KMB', 'CHD',
            'NKE', 'LULU', 'SBUX', 'MCD', 'YUM', 'CMG', 'DPZ',
            'LOW', 'TJX', 'ROST', 'DG', 'DLTR'
        ]
        
        return top_50 + additional_150
    
    def _get_full_universe(self) -> List[str]:
        """Get full tradeable universe (500+ symbols)"""
        # S&P 500 essentially - would be dynamically loaded in production
        return self._get_top_200_liquid_symbols() + ['...'] * 300  # Placeholder
    
    def get_position_size(
        self, 
        profile: AccountProfile, 
        trade_risk: float,
        strategy: str = None
    ) -> int:
        """
        Calculate appropriate position size based on account profile
        
        Args:
            profile: Account profile
            trade_risk: Maximum loss per contract
            strategy: Strategy name (optional, for strategy-specific adjustments)
        
        Returns:
            Number of contracts to trade
        """
        try:
            # Maximum risk per trade
            max_risk_dollars = profile.balance * (profile.max_position_size_pct / 100)
            
            # Calculate contracts based on risk
            if trade_risk > 0:
                contracts = int(max_risk_dollars / trade_risk)
            else:
                contracts = 0
            
            # Ensure at least 1 contract for viable trades
            contracts = max(1, contracts)
            
            # Apply tier-specific limits
            tier_limits = {
                AccountTier.MICRO: 1,
                AccountTier.SMALL: 2,
                AccountTier.MEDIUM: 5,
                AccountTier.LARGE: 10,
                AccountTier.INSTITUTIONAL: 50
            }
            
            max_contracts = tier_limits.get(profile.tier, 1)
            contracts = min(contracts, max_contracts)
            
            logger.debug(f"Position size calculated: {contracts} contracts (risk: ${trade_risk:.2f} per contract)")
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1  # Safe default
    
    def should_trade_symbol(
        self, 
        profile: AccountProfile, 
        symbol: str,
        current_positions: List[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Check if symbol is allowed to trade based on account profile
        
        Args:
            profile: Account profile
            symbol: Symbol to check
            current_positions: List of current positions
        
        Returns:
            (can_trade, reason)
        """
        try:
            # Check if symbol in allowed universe
            if symbol not in profile.allowed_symbols:
                return False, f"Symbol {symbol} not in allowed universe for {profile.tier.value} tier"
            
            # Check active symbol limit
            if current_positions:
                active_symbols = list(set([pos.get('symbol') for pos in current_positions]))
                if len(active_symbols) >= profile.max_symbols_active and symbol not in active_symbols:
                    return False, f"Max active symbols ({profile.max_symbols_active}) reached"
            
            # Check per-symbol position limit
            if current_positions:
                symbol_positions = [pos for pos in current_positions if pos.get('symbol') == symbol]
                if len(symbol_positions) >= profile.max_positions_per_symbol:
                    return False, f"Max positions per symbol ({profile.max_positions_per_symbol}) reached for {symbol}"
            
            return True, "Symbol allowed"
            
        except Exception as e:
            logger.error(f"Error checking symbol eligibility: {e}")
            return False, f"Error: {str(e)}"
    
    def should_use_strategy(
        self, 
        profile: AccountProfile, 
        strategy: str
    ) -> Tuple[bool, str]:
        """
        Check if strategy is enabled for account tier
        
        Args:
            profile: Account profile
            strategy: Strategy name
        
        Returns:
            (can_use, reason)
        """
        try:
            if strategy in profile.enabled_strategies:
                return True, f"Strategy {strategy} enabled for {profile.tier.value} tier"
            else:
                return False, f"Strategy {strategy} not enabled for {profile.tier.value} tier"
                
        except Exception as e:
            logger.error(f"Error checking strategy eligibility: {e}")
            return False, f"Error: {str(e)}"
    
    def get_profile_summary(self, profile: AccountProfile = None) -> Dict[str, Any]:
        """Get human-readable summary of account profile"""
        if profile is None:
            profile = self.current_profile
        
        if profile is None:
            return {"error": "No profile available"}
        
        return {
            'tier': profile.tier.value,
            'balance': f"${profile.balance:,.2f}",
            'buying_power': f"${profile.buying_power:,.2f}",
            'universe': {
                'total_symbols': len(profile.allowed_symbols),
                'max_active': profile.max_symbols_active,
                'sample_symbols': profile.allowed_symbols[:10]
            },
            'strategies': {
                'enabled_count': len(profile.enabled_strategies),
                'strategies': profile.enabled_strategies
            },
            'risk': {
                'max_positions': profile.max_positions,
                'max_position_size': f"{profile.max_position_size_pct}%",
                'max_daily_loss': f"{profile.max_daily_loss_pct}%",
                'max_portfolio_heat': f"{profile.max_portfolio_heat}%"
            },
            'execution': {
                'frequency': profile.execution_frequency,
                'dte_range': f"{profile.min_dte}-{profile.max_dte} days"
            },
            'greeks_limits': {
                'max_delta': profile.max_portfolio_delta,
                'max_gamma': profile.max_portfolio_gamma,
                'max_vega': profile.max_portfolio_vega,
                'max_theta': profile.max_portfolio_theta
            }
        }


# Example usage
if __name__ == "__main__":
    # Test account manager
    manager = UniversalAccountManager()
    
    # Test different account sizes
    test_balances = [500, 5000, 50000, 500000, 5000000]
    
    for balance in test_balances:
        print(f"\n{'='*60}")
        print(f"Testing account with balance: ${balance:,.0f}")
        print('='*60)
        
        profile = manager.create_account_profile(balance)
        summary = manager.get_profile_summary(profile)
        
        print(f"\nTier: {summary['tier']}")
        print(f"Universe: {summary['universe']['total_symbols']} symbols, max {summary['universe']['max_active']} active")
        print(f"Sample symbols: {', '.join(summary['universe']['sample_symbols'])}")
        print(f"Strategies: {summary['strategies']['enabled_count']} enabled")
        print(f"  - {', '.join(summary['strategies']['strategies'][:5])}")
        print(f"Risk: Max {summary['risk']['max_positions']} positions, {summary['risk']['max_position_size']} per trade")
        print(f"Execution: {summary['execution']['frequency']}, DTE: {summary['execution']['dte_range']}")
        print(f"Greeks: Delta ±{summary['greeks_limits']['max_delta']}, Gamma {summary['greeks_limits']['max_gamma']}")

