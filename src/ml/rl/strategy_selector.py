"""
Multi-Armed Bandit Strategy Selector
Test which strategies work best per account size with Thompson sampling
Capital allocation to winning strategies and separate bandits per account tier
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import json
import pickle
import os

from src.portfolio.account_manager import AccountProfile, AccountTier


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime


@dataclass
class BanditAction:
    """Bandit action result"""
    selected_strategy: str
    confidence: float
    expected_return: float
    exploration_bonus: float
    capital_allocation: float


@dataclass
class BanditMetrics:
    """Multi-armed bandit performance metrics"""
    total_rewards: float
    regret: float
    exploration_rate: float
    strategy_selections: Dict[str, int]
    best_strategy: str
    convergence_rate: float


class ThompsonSamplingBandit:
    """
    Multi-Armed Bandit with Thompson Sampling for strategy selection
    
    Features:
    - Test which strategies work best per account size
    - Capital allocation to winning strategies
    - Thompson sampling for explore/exploit balance
    - Separate bandits per account tier
    """
    
    def __init__(self, account_profile: AccountProfile, config: Dict = None):
        self.profile = account_profile
        
        # Configuration
        self.config = config or self._default_config()
        
        # Available strategies based on account tier
        self.available_strategies = self._get_available_strategies()
        
        # Bandit parameters
        self.alpha = {strategy: 1.0 for strategy in self.available_strategies}  # Success count
        self.beta = {strategy: 1.0 for strategy in self.available_strategies}   # Failure count
        
        # Performance tracking
        self.strategy_performance = {
            strategy: StrategyPerformance(
                strategy_name=strategy,
                total_trades=0,
                winning_trades=0,
                total_pnl=0.0,
                avg_pnl=0.0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                last_updated=datetime.now()
            )
            for strategy in self.available_strategies
        }
        
        # Selection history
        self.selection_history = []
        self.reward_history = []
        
        # Capital allocation weights
        self.capital_weights = {strategy: 1.0 / len(self.available_strategies) for strategy in self.available_strategies}
        
        logger.info(f"ThompsonSamplingBandit initialized for {account_profile.tier.value} tier with {len(self.available_strategies)} strategies")
    
    def _default_config(self) -> Dict:
        """Default configuration for bandit"""
        return {
            'update_frequency': 10,  # Update weights every N selections
            'min_trades_for_allocation': 5,  # Minimum trades before capital allocation
            'exploration_bonus': 0.1,  # Bonus for exploration
            'decay_factor': 0.95,  # Decay factor for old performance
            'confidence_threshold': 0.7,  # Confidence threshold for allocation
            'max_allocation_per_strategy': 0.4,  # Maximum 40% allocation to single strategy
            'min_allocation_per_strategy': 0.05,  # Minimum 5% allocation
            'convergence_threshold': 0.01,  # Convergence threshold for weights
            'reward_scaling': 100.0  # Scale rewards for better learning
        }
    
    def _get_available_strategies(self) -> List[str]:
        """Get available strategies based on account tier"""
        strategies = {
            AccountTier.MICRO: [
                'bull_put_spread',
                'cash_secured_put'
            ],
            AccountTier.SMALL: [
                'bull_put_spread',
                'bear_call_spread',
                'cash_secured_put'
            ],
            AccountTier.MEDIUM: [
                'bull_put_spread',
                'bear_call_spread',
                'cash_secured_put',
                'iron_condor'
            ],
            AccountTier.LARGE: [
                'bull_put_spread',
                'bear_call_spread',
                'cash_secured_put',
                'iron_condor',
                'calendar_spread'
            ],
            AccountTier.INSTITUTIONAL: [
                'bull_put_spread',
                'bear_call_spread',
                'cash_secured_put',
                'iron_condor',
                'calendar_spread',
                'diagonal_spread',
                'straddle_strangle',
                'gamma_scalping'
            ]
        }
        
        return strategies.get(self.profile.tier, strategies[AccountTier.MEDIUM])
    
    def select_strategy(self, context: Dict[str, Any] = None) -> BanditAction:
        """
        Select strategy using Thompson Sampling
        
        Args:
            context: Optional context information (market regime, etc.)
        
        Returns:
            BanditAction with selected strategy and allocation
        """
        try:
            # Sample from Beta distributions
            sampled_values = {}
            for strategy in self.available_strategies:
                # Sample from Beta(alpha, beta)
                sample = np.random.beta(self.alpha[strategy], self.beta[strategy])
                sampled_values[strategy] = sample
            
            # Select strategy with highest sampled value
            selected_strategy = max(sampled_values, key=sampled_values.get)
            
            # Calculate confidence based on total observations
            total_obs = self.alpha[selected_strategy] + self.beta[selected_strategy]
            confidence = min(0.99, total_obs / (total_obs + 10))  # Confidence increases with observations
            
            # Calculate expected return
            expected_return = self.alpha[selected_strategy] / (self.alpha[selected_strategy] + self.beta[selected_strategy])
            
            # Calculate exploration bonus
            exploration_bonus = self.config['exploration_bonus'] * (1 - confidence)
            
            # Calculate capital allocation
            capital_allocation = self._calculate_capital_allocation(selected_strategy, context)
            
            # Record selection
            self.selection_history.append({
                'timestamp': datetime.now(),
                'strategy': selected_strategy,
                'context': context,
                'confidence': confidence,
                'expected_return': expected_return
            })
            
            return BanditAction(
                selected_strategy=selected_strategy,
                confidence=confidence,
                expected_return=expected_return,
                exploration_bonus=exploration_bonus,
                capital_allocation=capital_allocation
            )
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            # Fallback to first available strategy
            return BanditAction(
                selected_strategy=self.available_strategies[0],
                confidence=0.1,
                expected_return=0.5,
                exploration_bonus=0.5,
                capital_allocation=0.2
            )
    
    def _calculate_capital_allocation(self, strategy: str, context: Dict[str, Any] = None) -> float:
        """
        Calculate capital allocation for selected strategy
        
        Args:
            strategy: Selected strategy
            context: Optional context information
        
        Returns:
            Capital allocation percentage (0.0 to 1.0)
        """
        try:
            # Base allocation from weights
            base_allocation = self.capital_weights.get(strategy, 0.1)
            
            # Adjust based on performance
            performance = self.strategy_performance[strategy]
            
            # Increase allocation for high-performing strategies
            if performance.total_trades >= self.config['min_trades_for_allocation']:
                if performance.win_rate > 0.7:
                    base_allocation *= 1.2
                elif performance.win_rate > 0.6:
                    base_allocation *= 1.1
                elif performance.win_rate < 0.4:
                    base_allocation *= 0.8
                
                # Adjust based on Sharpe ratio
                if performance.sharpe_ratio > 2.0:
                    base_allocation *= 1.3
                elif performance.sharpe_ratio > 1.5:
                    base_allocation *= 1.1
                elif performance.sharpe_ratio < 0.5:
                    base_allocation *= 0.7
            
            # Context-based adjustments
            if context:
                market_regime = context.get('market_regime', 'NORMAL_VOL')
                
                # Adjust allocation based on market regime
                if market_regime == 'HIGH_VOL':
                    if strategy in ['iron_condor', 'straddle_strangle']:
                        base_allocation *= 1.2  # Favor volatility strategies
                    elif strategy in ['bull_put_spread', 'bear_call_spread']:
                        base_allocation *= 0.8  # Reduce directional strategies
                elif market_regime == 'LOW_VOL':
                    if strategy in ['bull_put_spread', 'bear_call_spread']:
                        base_allocation *= 1.2  # Favor directional strategies
                    elif strategy in ['iron_condor', 'straddle_strangle']:
                        base_allocation *= 0.8  # Reduce volatility strategies
            
            # Apply constraints
            base_allocation = max(self.config['min_allocation_per_strategy'], base_allocation)
            base_allocation = min(self.config['max_allocation_per_strategy'], base_allocation)
            
            return base_allocation
            
        except Exception as e:
            logger.error(f"Error calculating capital allocation: {e}")
            return 0.2  # Default 20% allocation
    
    def update_performance(
        self,
        strategy: str,
        trade_result: Dict[str, Any]
    ) -> bool:
        """
        Update strategy performance based on trade result
        
        Args:
            strategy: Strategy name
            trade_result: Trade execution result
        
        Returns:
            Success status
        """
        try:
            if strategy not in self.available_strategies:
                logger.error(f"Unknown strategy: {strategy}")
                return False
            
            # Extract trade metrics
            pnl = trade_result.get('pnl', 0.0)
            max_loss = trade_result.get('max_loss', 1.0)
            is_winner = pnl > 0
            
            # Update Beta distribution parameters
            if is_winner:
                self.alpha[strategy] += 1
                reward = min(1.0, pnl / (max_loss * self.config['reward_scaling']))
            else:
                self.beta[strategy] += 1
                reward = max(0.0, pnl / (max_loss * self.config['reward_scaling']))
            
            # Record reward
            self.reward_history.append({
                'timestamp': datetime.now(),
                'strategy': strategy,
                'reward': reward,
                'pnl': pnl,
                'is_winner': is_winner
            })
            
            # Update strategy performance
            performance = self.strategy_performance[strategy]
            performance.total_trades += 1
            
            if is_winner:
                performance.winning_trades += 1
            
            performance.total_pnl += pnl
            performance.avg_pnl = performance.total_pnl / performance.total_trades
            performance.win_rate = performance.winning_trades / performance.total_trades
            
            # Update Sharpe ratio (simplified)
            if len(self.reward_history) >= 10:
                strategy_rewards = [r['reward'] for r in self.reward_history[-50:] if r['strategy'] == strategy]
                if len(strategy_rewards) > 1:
                    performance.sharpe_ratio = np.mean(strategy_rewards) / np.std(strategy_rewards) if np.std(strategy_rewards) > 0 else 0
            
            performance.last_updated = datetime.now()
            
            # Update capital weights periodically
            if performance.total_trades % self.config['update_frequency'] == 0:
                self._update_capital_weights()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
            return False
    
    def _update_capital_weights(self):
        """Update capital allocation weights based on performance"""
        try:
            # Calculate performance scores for each strategy
            performance_scores = {}
            
            for strategy in self.available_strategies:
                performance = self.strategy_performance[strategy]
                
                if performance.total_trades >= self.config['min_trades_for_allocation']:
                    # Combined score based on win rate and Sharpe ratio
                    win_rate_score = performance.win_rate
                    sharpe_score = min(2.0, max(0.0, performance.sharpe_ratio)) / 2.0
                    pnl_score = min(1.0, max(0.0, performance.avg_pnl / 100))  # Normalize P&L
                    
                    # Weighted combination
                    performance_scores[strategy] = (
                        0.4 * win_rate_score +
                        0.4 * sharpe_score +
                        0.2 * pnl_score
                    )
                else:
                    # Default score for strategies with insufficient data
                    performance_scores[strategy] = 0.5
            
            # Convert scores to weights using softmax
            scores_array = np.array(list(performance_scores.values()))
            weights_array = np.exp(scores_array) / np.sum(np.exp(scores_array))
            
            # Apply decay factor to smooth updates
            for i, strategy in enumerate(self.available_strategies):
                old_weight = self.capital_weights[strategy]
                new_weight = weights_array[i]
                
                # Smooth update
                self.capital_weights[strategy] = (
                    self.config['decay_factor'] * old_weight +
                    (1 - self.config['decay_factor']) * new_weight
                )
            
            # Normalize weights
            total_weight = sum(self.capital_weights.values())
            for strategy in self.available_strategies:
                self.capital_weights[strategy] /= total_weight
            
            logger.info(f"Updated capital weights: {self.capital_weights}")
            
        except Exception as e:
            logger.error(f"Error updating capital weights: {e}")
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """
        Get current strategy ranking based on performance
        
        Returns:
            List of (strategy, score) tuples sorted by performance
        """
        try:
            rankings = []
            
            for strategy in self.available_strategies:
                performance = self.strategy_performance[strategy]
                
                # Calculate composite score
                if performance.total_trades >= self.config['min_trades_for_allocation']:
                    score = (
                        0.3 * performance.win_rate +
                        0.3 * min(1.0, performance.sharpe_ratio / 2.0) +
                        0.2 * min(1.0, performance.avg_pnl / 100) +
                        0.2 * self.capital_weights[strategy]
                    )
                else:
                    # Lower score for strategies with insufficient data
                    score = 0.2
                
                rankings.append((strategy, score))
            
            # Sort by score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting strategy ranking: {e}")
            return [(strategy, 0.0) for strategy in self.available_strategies]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                'total_selections': len(self.selection_history),
                'total_rewards': sum(r['reward'] for r in self.reward_history),
                'strategy_performance': {},
                'capital_allocation': self.capital_weights.copy(),
                'strategy_ranking': self.get_strategy_ranking(),
                'best_strategy': None,
                'convergence_metrics': self._calculate_convergence_metrics()
            }
            
            # Strategy performance details
            for strategy in self.available_strategies:
                performance = self.strategy_performance[strategy]
                summary['strategy_performance'][strategy] = {
                    'total_trades': performance.total_trades,
                    'winning_trades': performance.winning_trades,
                    'total_pnl': performance.total_pnl,
                    'avg_pnl': performance.avg_pnl,
                    'win_rate': performance.win_rate,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown,
                    'alpha': self.alpha[strategy],
                    'beta': self.beta[strategy],
                    'capital_allocation': self.capital_weights[strategy]
                }
            
            # Best strategy
            if summary['strategy_ranking']:
                summary['best_strategy'] = summary['strategy_ranking'][0][0]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics"""
        try:
            if len(self.selection_history) < 20:
                return {'convergence_rate': 0.0, 'exploration_rate': 1.0}
            
            # Calculate selection frequency in recent history
            recent_selections = self.selection_history[-20:]
            selection_counts = {}
            
            for selection in recent_selections:
                strategy = selection['strategy']
                selection_counts[strategy] = selection_counts.get(strategy, 0) + 1
            
            # Calculate exploration rate (frequency of non-dominant strategy)
            total_recent = len(recent_selections)
            dominant_strategy = max(selection_counts, key=selection_counts.get)
            dominant_count = selection_counts[dominant_strategy]
            exploration_rate = 1.0 - (dominant_count / total_recent)
            
            # Calculate convergence rate (stability of weights)
            if len(self.selection_history) >= 40:
                old_weights = self.capital_weights.copy()
                # Simulate old weights (simplified)
                weight_changes = []
                for strategy in self.available_strategies:
                    weight_change = abs(self.capital_weights[strategy] - 0.2)  # Assume old weight was 0.2
                    weight_changes.append(weight_change)
                
                convergence_rate = 1.0 - np.mean(weight_changes)
            else:
                convergence_rate = 0.0
            
            return {
                'convergence_rate': max(0.0, min(1.0, convergence_rate)),
                'exploration_rate': exploration_rate
            }
            
        except Exception as e:
            logger.error(f"Error calculating convergence metrics: {e}")
            return {'convergence_rate': 0.0, 'exploration_rate': 0.5}
    
    def calculate_regret(self) -> float:
        """Calculate cumulative regret"""
        try:
            if not self.reward_history:
                return 0.0
            
            # Find best strategy in hindsight
            strategy_avg_rewards = {}
            for strategy in self.available_strategies:
                strategy_rewards = [r['reward'] for r in self.reward_history if r['strategy'] == strategy]
                if strategy_rewards:
                    strategy_avg_rewards[strategy] = np.mean(strategy_rewards)
                else:
                    strategy_avg_rewards[strategy] = 0.0
            
            best_strategy = max(strategy_avg_rewards, key=strategy_avg_rewards.get)
            best_avg_reward = strategy_avg_rewards[best_strategy]
            
            # Calculate regret
            total_regret = 0.0
            for reward_data in self.reward_history:
                strategy = reward_data['strategy']
                actual_reward = reward_data['reward']
                optimal_reward = best_avg_reward
                regret = optimal_reward - actual_reward
                total_regret += regret
            
            return total_regret
            
        except Exception as e:
            logger.error(f"Error calculating regret: {e}")
            return 0.0
    
    def save_model(self, model_path: str = None) -> bool:
        """Save bandit model"""
        try:
            if model_path is None:
                model_path = f"models/strategy_bandit_{self.profile.tier.value}.pkl"
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'alpha': self.alpha,
                'beta': self.beta,
                'strategy_performance': self.strategy_performance,
                'capital_weights': self.capital_weights,
                'selection_history': self.selection_history[-100:],  # Keep last 100 selections
                'reward_history': self.reward_history[-200:],  # Keep last 200 rewards
                'config': self.config,
                'available_strategies': self.available_strategies,
                'account_profile': self.profile
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Bandit model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str = None) -> bool:
        """Load bandit model"""
        try:
            if model_path is None:
                model_path = f"models/strategy_bandit_{self.profile.tier.value}.pkl"
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore components
            self.alpha = model_data['alpha']
            self.beta = model_data['beta']
            self.strategy_performance = model_data['strategy_performance']
            self.capital_weights = model_data['capital_weights']
            self.selection_history = model_data['selection_history']
            self.reward_history = model_data['reward_history']
            self.config = model_data['config']
            self.available_strategies = model_data['available_strategies']
            
            logger.info(f"Bandit model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create strategy selector
    selector = ThompsonSamplingBandit(profile)
    
    print("Testing Strategy Selector...")
    
    # Test strategy selection
    context = {'market_regime': 'NORMAL_VOL', 'iv_rank': 0.6}
    action = selector.select_strategy(context)
    
    print(f"Selected Strategy: {action.selected_strategy}")
    print(f"Confidence: {action.confidence:.2%}")
    print(f"Expected Return: {action.expected_return:.3f}")
    print(f"Exploration Bonus: {action.exploration_bonus:.3f}")
    print(f"Capital Allocation: {action.capital_allocation:.1%}")
    
    # Simulate some trades
    print("\nSimulating trades...")
    
    strategies = ['bull_put_spread', 'bear_call_spread', 'cash_secured_put']
    for i in range(20):
        # Select strategy
        action = selector.select_strategy(context)
        selected_strategy = action.selected_strategy
        
        # Simulate trade result
        is_winner = np.random.random() > 0.4  # 60% win rate
        pnl = np.random.uniform(-500, 1000) if is_winner else np.random.uniform(-1000, 0)
        max_loss = 500
        
        trade_result = {
            'pnl': pnl,
            'max_loss': max_loss,
            'is_winner': is_winner
        }
        
        # Update performance
        selector.update_performance(selected_strategy, trade_result)
        
        if i % 5 == 0:
            print(f"Trade {i+1}: {selected_strategy}, P&L: ${pnl:.0f}")
    
    # Get performance summary
    print("\nPerformance Summary:")
    summary = selector.get_performance_summary()
    
    print(f"Total Selections: {summary['total_selections']}")
    print(f"Total Rewards: {summary['total_rewards']:.2f}")
    print(f"Best Strategy: {summary['best_strategy']}")
    print(f"Convergence Rate: {summary['convergence_metrics']['convergence_rate']:.2%}")
    print(f"Exploration Rate: {summary['convergence_metrics']['exploration_rate']:.2%}")
    
    print("\nStrategy Performance:")
    for strategy, performance in summary['strategy_performance'].items():
        print(f"{strategy}:")
        print(f"  Trades: {performance['total_trades']}")
        print(f"  Win Rate: {performance['win_rate']:.1%}")
        print(f"  Avg P&L: ${performance['avg_pnl']:.0f}")
        print(f"  Capital Allocation: {performance['capital_allocation']:.1%}")
    
    print("\nStrategy Ranking:")
    for i, (strategy, score) in enumerate(summary['strategy_ranking'], 1):
        print(f"{i}. {strategy}: {score:.3f}")
