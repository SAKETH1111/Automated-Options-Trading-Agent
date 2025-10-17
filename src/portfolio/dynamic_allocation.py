"""
Dynamic Allocation System
Kelly criterion sizing and regime-based capital allocation with account-tier responses
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import json

from src.portfolio.account_manager import AccountProfile, AccountTier
from src.volatility.regime_detector import MarketRegimeDetector


@dataclass
class AllocationDecision:
    """Dynamic allocation decision"""
    strategy_name: str
    allocation_percent: float
    allocation_amount: float
    kelly_multiplier: float
    regime_factor: float
    confidence: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    recommendation: str


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    total_allocation: float
    cash_reserve: float
    strategy_allocations: List[AllocationDecision]
    regime: str
    kelly_fraction: float
    risk_budget: float
    rebalance_needed: bool
    timestamp: datetime


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    trade_count: int
    last_updated: datetime


class DynamicAllocationSystem:
    """
    Dynamic capital allocation system for options trading
    
    Features:
    - Kelly criterion position sizing with account-tier adjustments
    - Regime-based strategy allocation
    - Risk budget management
    - Automatic rebalancing
    - Performance-based strategy selection
    """
    
    def __init__(self, account_profile: AccountProfile, config: Dict = None):
        self.profile = account_profile
        
        # Configuration
        self.config = config or self._default_config()
        
        # Strategy performance tracking
        self.strategy_performance = {}
        self.allocation_history = []
        
        # Regime detection
        self.regime_detector = MarketRegimeDetector(account_profile)
        
        # Current allocation
        self.current_allocation = None
        self.last_rebalance = datetime.now()
        
        # Risk management
        self.risk_budget = self.profile.balance * self.config['max_risk_percent'] / 100
        self.available_strategies = self._get_available_strategies()
        
        logger.info(f"DynamicAllocationSystem initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'max_risk_percent': 25.0,  # Maximum 25% of account at risk
            'cash_reserve_percent': 10.0,  # Keep 10% in cash
            'rebalance_threshold': 0.05,  # 5% drift triggers rebalance
            'rebalance_frequency': 7,  # days
            'kelly_multipliers': {
                'micro': 0.1,      # 10% of Kelly for micro accounts
                'small': 0.25,     # 25% of Kelly for small accounts
                'medium': 0.35,    # 35% of Kelly for medium accounts
                'large': 0.45,     # 45% of Kelly for large accounts
                'institutional': 0.5  # 50% of Kelly for institutional accounts
            },
            'regime_allocation': {
                'LOW_VOL': {
                    'bull_put_spread': 0.4,
                    'bear_call_spread': 0.3,
                    'cash_secured_put': 0.2,
                    'iron_condor': 0.1
                },
                'NORMAL_VOL': {
                    'bull_put_spread': 0.3,
                    'bear_call_spread': 0.25,
                    'cash_secured_put': 0.2,
                    'iron_condor': 0.15,
                    'calendar_spread': 0.1
                },
                'HIGH_VOL': {
                    'iron_condor': 0.4,
                    'straddle_strangle': 0.3,
                    'calendar_spread': 0.2,
                    'bull_put_spread': 0.1
                },
                'CRISIS': {
                    'cash_secured_put': 0.5,
                    'calendar_spread': 0.3,
                    'bull_put_spread': 0.2
                }
            },
            'performance_weights': {
                'return_weight': 0.3,
                'sharpe_weight': 0.4,
                'drawdown_weight': 0.2,
                'consistency_weight': 0.1
            }
        }
    
    def _get_available_strategies(self) -> List[str]:
        """Get available strategies based on account tier"""
        strategies = {
            AccountTier.MICRO: ['bull_put_spread', 'cash_secured_put'],
            AccountTier.SMALL: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put'],
            AccountTier.MEDIUM: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put', 'iron_condor'],
            AccountTier.LARGE: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put', 'iron_condor', 'calendar_spread'],
            AccountTier.INSTITUTIONAL: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put', 'iron_condor', 
                                       'calendar_spread', 'diagonal_spread', 'straddle_strangle', 'gamma_scalping']
        }
        
        return strategies.get(self.profile.tier, strategies[AccountTier.MEDIUM])
    
    def update_strategy_performance(self, strategy_performance: Dict[str, StrategyPerformance]):
        """Update strategy performance data"""
        try:
            self.strategy_performance.update(strategy_performance)
            
            # Remove strategies not available for this account tier
            self.strategy_performance = {
                k: v for k, v in self.strategy_performance.items()
                if k in self.available_strategies
            }
            
            logger.info(f"Updated performance for {len(strategy_performance)} strategies")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def calculate_dynamic_allocation(self, market_data: Dict[str, Any] = None) -> PortfolioAllocation:
        """
        Calculate dynamic portfolio allocation
        
        Args:
            market_data: Current market data for regime detection
        
        Returns:
            PortfolioAllocation object
        """
        try:
            # Detect current market regime
            current_regime = self._detect_current_regime(market_data)
            
            # Calculate Kelly-based allocations
            kelly_allocations = self._calculate_kelly_allocations()
            
            # Apply regime-based adjustments
            regime_allocations = self._apply_regime_adjustments(kelly_allocations, current_regime)
            
            # Apply risk budget constraints
            constrained_allocations = self._apply_risk_constraints(regime_allocations)
            
            # Calculate cash reserve
            cash_reserve = self.profile.balance * self.config['cash_reserve_percent'] / 100
            
            # Create allocation decisions
            allocation_decisions = []
            total_allocation = 0.0
            
            for strategy, allocation in constrained_allocations.items():
                allocation_amount = allocation * self.profile.balance
                
                # Get strategy performance
                performance = self.strategy_performance.get(strategy)
                if performance:
                    decision = AllocationDecision(
                        strategy_name=strategy,
                        allocation_percent=allocation * 100,
                        allocation_amount=allocation_amount,
                        kelly_multiplier=self._get_kelly_multiplier(strategy, performance),
                        regime_factor=self._get_regime_factor(strategy, current_regime),
                        confidence=self._calculate_confidence(strategy, performance),
                        expected_return=performance.total_return,
                        expected_volatility=performance.volatility,
                        sharpe_ratio=performance.sharpe_ratio,
                        max_drawdown=performance.max_drawdown,
                        recommendation=self._get_allocation_recommendation(strategy, allocation, performance)
                    )
                    
                    allocation_decisions.append(decision)
                    total_allocation += allocation_amount
            
            # Check if rebalancing is needed
            rebalance_needed = self._check_rebalance_needed(allocation_decisions)
            
            # Create portfolio allocation
            portfolio_allocation = PortfolioAllocation(
                total_allocation=total_allocation,
                cash_reserve=cash_reserve,
                strategy_allocations=allocation_decisions,
                regime=current_regime,
                kelly_fraction=self._get_kelly_fraction(),
                risk_budget=self.risk_budget,
                rebalance_needed=rebalance_needed,
                timestamp=datetime.now()
            )
            
            # Store allocation history
            self.allocation_history.append(portfolio_allocation)
            self.current_allocation = portfolio_allocation
            
            # Clean up old history
            cutoff_date = datetime.now() - timedelta(days=90)
            self.allocation_history = [
                a for a in self.allocation_history if a.timestamp > cutoff_date
            ]
            
            return portfolio_allocation
            
        except Exception as e:
            logger.error(f"Error calculating dynamic allocation: {e}")
            return self._empty_allocation()
    
    def _detect_current_regime(self, market_data: Dict[str, Any] = None) -> str:
        """Detect current market regime"""
        try:
            if market_data:
                # Use regime detector if market data is available
                regime_result = self.regime_detector.detect_regime(market_data)
                return regime_result.regime_name
            else:
                # Default to normal volatility
                return 'NORMAL_VOL'
                
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'NORMAL_VOL'
    
    def _calculate_kelly_allocations(self) -> Dict[str, float]:
        """Calculate Kelly criterion-based allocations"""
        try:
            kelly_allocations = {}
            
            for strategy in self.available_strategies:
                performance = self.strategy_performance.get(strategy)
                
                if performance and performance.trade_count > 10:  # Minimum trades for Kelly calculation
                    # Calculate Kelly fraction
                    win_rate = performance.win_rate
                    avg_win = performance.total_return / max(performance.trade_count, 1)
                    avg_loss = -performance.max_drawdown / 10  # Approximate average loss
                    
                    if avg_loss != 0:
                        # Kelly formula: f = (bp - q) / b
                        # where b = odds (avg_win / avg_loss), p = win_rate, q = loss_rate
                        b = avg_win / abs(avg_loss)
                        p = win_rate
                        q = 1 - win_rate
                        
                        kelly_fraction = (b * p - q) / b
                        
                        # Apply account tier multiplier
                        kelly_multiplier = self._get_kelly_multiplier(strategy, performance)
                        adjusted_kelly = kelly_fraction * kelly_multiplier
                        
                        # Ensure positive allocation
                        kelly_allocations[strategy] = max(0.0, min(adjusted_kelly, 0.5))  # Cap at 50%
                    else:
                        kelly_allocations[strategy] = 0.0
                else:
                    # Default allocation for strategies with insufficient data
                    kelly_allocations[strategy] = 0.1  # 10% default
            
            # Normalize allocations
            total_allocation = sum(kelly_allocations.values())
            if total_allocation > 0:
                kelly_allocations = {
                    strategy: allocation / total_allocation
                    for strategy, allocation in kelly_allocations.items()
                }
            
            return kelly_allocations
            
        except Exception as e:
            logger.error(f"Error calculating Kelly allocations: {e}")
            return {strategy: 1.0 / len(self.available_strategies) for strategy in self.available_strategies}
    
    def _apply_regime_adjustments(self, kelly_allocations: Dict[str, float], regime: str) -> Dict[str, float]:
        """Apply regime-based adjustments to allocations"""
        try:
            regime_allocation = self.config['regime_allocation'].get(regime, {})
            
            if not regime_allocation:
                # No regime-specific allocation, use Kelly allocations
                return kelly_allocations
            
            # Blend Kelly allocations with regime allocations
            blended_allocations = {}
            
            for strategy in self.available_strategies:
                kelly_weight = kelly_allocations.get(strategy, 0.0)
                regime_weight = regime_allocation.get(strategy, 0.0)
                
                # Weighted average (70% Kelly, 30% regime)
                blended_weight = 0.7 * kelly_weight + 0.3 * regime_weight
                blended_allocations[strategy] = blended_weight
            
            # Normalize
            total_allocation = sum(blended_allocations.values())
            if total_allocation > 0:
                blended_allocations = {
                    strategy: allocation / total_allocation
                    for strategy, allocation in blended_allocations.items()
                }
            
            return blended_allocations
            
        except Exception as e:
            logger.error(f"Error applying regime adjustments: {e}")
            return kelly_allocations
    
    def _apply_risk_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Apply risk budget constraints"""
        try:
            constrained_allocations = {}
            
            for strategy, allocation in allocations.items():
                # Get strategy performance
                performance = self.strategy_performance.get(strategy)
                
                if performance:
                    # Calculate risk per strategy
                    strategy_risk = allocation * self.profile.balance
                    
                    # Check if risk exceeds budget
                    max_risk_per_strategy = self.risk_budget * 0.4  # Max 40% of risk budget per strategy
                    
                    if strategy_risk > max_risk_per_strategy:
                        # Scale down allocation
                        scaled_allocation = max_risk_per_strategy / self.profile.balance
                        constrained_allocations[strategy] = scaled_allocation
                    else:
                        constrained_allocations[strategy] = allocation
                else:
                    constrained_allocations[strategy] = allocation
            
            # Normalize after constraints
            total_allocation = sum(constrained_allocations.values())
            if total_allocation > 0:
                constrained_allocations = {
                    strategy: allocation / total_allocation
                    for strategy, allocation in constrained_allocations.items()
                }
            
            return constrained_allocations
            
        except Exception as e:
            logger.error(f"Error applying risk constraints: {e}")
            return allocations
    
    def _get_kelly_multiplier(self, strategy: str, performance: StrategyPerformance) -> float:
        """Get Kelly multiplier based on account tier and strategy performance"""
        try:
            # Base multiplier from config
            base_multiplier = self.config['kelly_multipliers'].get(self.profile.tier.value, 0.25)
            
            # Adjust based on strategy performance
            if performance.sharpe_ratio > 2.0:
                performance_multiplier = 1.2  # Increase for high-performing strategies
            elif performance.sharpe_ratio > 1.5:
                performance_multiplier = 1.1
            elif performance.sharpe_ratio < 0.5:
                performance_multiplier = 0.8  # Decrease for poor-performing strategies
            else:
                performance_multiplier = 1.0
            
            # Adjust based on consistency (win rate)
            if performance.win_rate > 0.7:
                consistency_multiplier = 1.1
            elif performance.win_rate < 0.5:
                consistency_multiplier = 0.9
            else:
                consistency_multiplier = 1.0
            
            final_multiplier = base_multiplier * performance_multiplier * consistency_multiplier
            
            return max(0.05, min(1.0, final_multiplier))  # Clamp between 5% and 100%
            
        except Exception as e:
            logger.error(f"Error calculating Kelly multiplier: {e}")
            return self.config['kelly_multipliers'].get(self.profile.tier.value, 0.25)
    
    def _get_regime_factor(self, strategy: str, regime: str) -> float:
        """Get regime adjustment factor for strategy"""
        try:
            regime_allocation = self.config['regime_allocation'].get(regime, {})
            
            if strategy in regime_allocation:
                # Strategy is favored in this regime
                return 1.2
            else:
                # Strategy is not favored in this regime
                return 0.8
                
        except Exception as e:
            logger.error(f"Error calculating regime factor: {e}")
            return 1.0
    
    def _calculate_confidence(self, strategy: str, performance: StrategyPerformance) -> float:
        """Calculate confidence in strategy allocation"""
        try:
            # Base confidence on trade count
            trade_confidence = min(1.0, performance.trade_count / 50)  # Full confidence at 50+ trades
            
            # Adjust based on performance consistency
            if performance.sharpe_ratio > 1.5 and performance.win_rate > 0.6:
                performance_confidence = 0.9
            elif performance.sharpe_ratio > 1.0 and performance.win_rate > 0.55:
                performance_confidence = 0.7
            elif performance.sharpe_ratio > 0.5:
                performance_confidence = 0.5
            else:
                performance_confidence = 0.3
            
            # Weighted combination
            confidence = 0.6 * trade_confidence + 0.4 * performance_confidence
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _get_allocation_recommendation(self, strategy: str, allocation: float, performance: StrategyPerformance) -> str:
        """Get allocation recommendation for strategy"""
        try:
            if allocation > 0.3:
                return "INCREASE" if performance.sharpe_ratio > 1.5 else "MAINTAIN"
            elif allocation > 0.1:
                return "MAINTAIN"
            elif allocation > 0.05:
                return "REDUCE"
            else:
                return "ELIMINATE"
                
        except Exception as e:
            logger.error(f"Error getting allocation recommendation: {e}")
            return "MAINTAIN"
    
    def _check_rebalance_needed(self, allocation_decisions: List[AllocationDecision]) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            # Check time-based rebalancing
            days_since_rebalance = (datetime.now() - self.last_rebalance).days
            if days_since_rebalance >= self.config['rebalance_frequency']:
                return True
            
            # Check drift-based rebalancing
            if self.current_allocation:
                total_drift = 0.0
                
                for decision in allocation_decisions:
                    # Find corresponding allocation in current portfolio
                    current_allocation = None
                    for current_decision in self.current_allocation.strategy_allocations:
                        if current_decision.strategy_name == decision.strategy_name:
                            current_allocation = current_decision.allocation_percent
                            break
                    
                    if current_allocation is not None:
                        drift = abs(decision.allocation_percent - current_allocation)
                        total_drift += drift
                
                if total_drift > self.config['rebalance_threshold'] * 100:  # Convert to percentage
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rebalance need: {e}")
            return False
    
    def _get_kelly_fraction(self) -> float:
        """Get overall Kelly fraction for portfolio"""
        try:
            tier_multiplier = self.config['kelly_multipliers'].get(self.profile.tier.value, 0.25)
            
            # Calculate average Kelly across strategies
            kelly_fractions = []
            for strategy in self.available_strategies:
                performance = self.strategy_performance.get(strategy)
                if performance and performance.trade_count > 10:
                    # Simplified Kelly calculation
                    win_rate = performance.win_rate
                    if win_rate > 0.5:  # Profitable strategy
                        kelly_fraction = (win_rate - 0.5) * 2  # Simplified Kelly
                        kelly_fractions.append(kelly_fraction)
            
            if kelly_fractions:
                avg_kelly = np.mean(kelly_fractions)
                return avg_kelly * tier_multiplier
            else:
                return tier_multiplier * 0.1  # Conservative default
                
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return self.config['kelly_multipliers'].get(self.profile.tier.value, 0.25)
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get allocation summary"""
        try:
            if not self.current_allocation:
                return {}
            
            summary = {
                'timestamp': self.current_allocation.timestamp,
                'regime': self.current_allocation.regime,
                'total_allocation': self.current_allocation.total_allocation,
                'cash_reserve': self.current_allocation.cash_reserve,
                'risk_budget': self.current_allocation.risk_budget,
                'kelly_fraction': self.current_allocation.kelly_fraction,
                'rebalance_needed': self.current_allocation.rebalance_needed,
                'strategy_allocations': [
                    {
                        'strategy': decision.strategy_name,
                        'allocation_percent': decision.allocation_percent,
                        'allocation_amount': decision.allocation_amount,
                        'confidence': decision.confidence,
                        'expected_return': decision.expected_return,
                        'sharpe_ratio': decision.sharpe_ratio,
                        'recommendation': decision.recommendation
                    }
                    for decision in self.current_allocation.strategy_allocations
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting allocation summary: {e}")
            return {}
    
    def get_performance_ranking(self) -> List[Tuple[str, float]]:
        """Get strategy performance ranking"""
        try:
            rankings = []
            
            for strategy, performance in self.strategy_performance.items():
                # Calculate composite score
                weights = self.config['performance_weights']
                
                # Normalize metrics
                return_score = min(1.0, max(0.0, performance.total_return / 0.5))  # 50% return = 1.0
                sharpe_score = min(1.0, max(0.0, performance.sharpe_ratio / 2.0))  # 2.0 Sharpe = 1.0
                drawdown_score = max(0.0, 1.0 - performance.max_drawdown / 0.2)  # 20% drawdown = 0.0
                consistency_score = performance.win_rate
                
                composite_score = (
                    weights['return_weight'] * return_score +
                    weights['sharpe_weight'] * sharpe_score +
                    weights['drawdown_weight'] * drawdown_score +
                    weights['consistency_weight'] * consistency_score
                )
                
                rankings.append((strategy, composite_score))
            
            # Sort by score
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting performance ranking: {e}")
            return []
    
    def _empty_allocation(self) -> PortfolioAllocation:
        """Return empty allocation"""
        return PortfolioAllocation(
            total_allocation=0.0,
            cash_reserve=self.profile.balance * self.config['cash_reserve_percent'] / 100,
            strategy_allocations=[],
            regime='NORMAL_VOL',
            kelly_fraction=0.0,
            risk_budget=0.0,
            rebalance_needed=False,
            timestamp=datetime.now()
        )


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create dynamic allocation system
    allocation_system = DynamicAllocationSystem(profile)
    
    # Create sample strategy performance data
    sample_performance = {
        'bull_put_spread': StrategyPerformance(
            strategy_name='bull_put_spread',
            total_return=0.15,
            volatility=0.12,
            sharpe_ratio=1.25,
            max_drawdown=0.08,
            win_rate=0.68,
            profit_factor=1.45,
            avg_trade_duration=15.5,
            trade_count=45,
            last_updated=datetime.now()
        ),
        'iron_condor': StrategyPerformance(
            strategy_name='iron_condor',
            total_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.80,
            max_drawdown=0.12,
            win_rate=0.58,
            profit_factor=1.25,
            avg_trade_duration=22.3,
            trade_count=32,
            last_updated=datetime.now()
        ),
        'cash_secured_put': StrategyPerformance(
            strategy_name='cash_secured_put',
            total_return=0.08,
            volatility=0.08,
            sharpe_ratio=1.00,
            max_drawdown=0.05,
            win_rate=0.72,
            profit_factor=1.35,
            avg_trade_duration=18.7,
            trade_count=28,
            last_updated=datetime.now()
        )
    }
    
    # Update strategy performance
    allocation_system.update_strategy_performance(sample_performance)
    
    # Calculate dynamic allocation
    market_data = {'vix': 20, 'spy_volatility': 0.15}
    allocation = allocation_system.calculate_dynamic_allocation(market_data)
    
    print("Dynamic Portfolio Allocation:")
    print(f"Regime: {allocation.regime}")
    print(f"Total Allocation: ${allocation.total_allocation:,.2f}")
    print(f"Cash Reserve: ${allocation.cash_reserve:,.2f}")
    print(f"Risk Budget: ${allocation.risk_budget:,.2f}")
    print(f"Kelly Fraction: {allocation.kelly_fraction:.3f}")
    print(f"Rebalance Needed: {allocation.rebalance_needed}")
    
    print(f"\nStrategy Allocations:")
    for decision in allocation.strategy_allocations:
        print(f"  {decision.strategy_name}:")
        print(f"    Allocation: {decision.allocation_percent:.1f}% (${decision.allocation_amount:,.2f})")
        print(f"    Confidence: {decision.confidence:.2%}")
        print(f"    Expected Return: {decision.expected_return:.1%}")
        print(f"    Sharpe Ratio: {decision.sharpe_ratio:.2f}")
        print(f"    Recommendation: {decision.recommendation}")
    
    # Get performance ranking
    ranking = allocation_system.get_performance_ranking()
    print(f"\nStrategy Performance Ranking:")
    for i, (strategy, score) in enumerate(ranking, 1):
        print(f"  {i}. {strategy}: {score:.3f}")
    
    # Get allocation summary
    summary = allocation_system.get_allocation_summary()
    print(f"\nAllocation Summary:")
    print(f"  Timestamp: {summary.get('timestamp')}")
    print(f"  Regime: {summary.get('regime')}")
    print(f"  Rebalance Needed: {summary.get('rebalance_needed')}")
