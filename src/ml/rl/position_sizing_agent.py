"""
Position Sizing Agent
RL agent for optimal position sizing using Proximal Policy Optimization (PPO)
Account-aware state/action spaces and Sharpe-based rewards
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import os

from src.portfolio.account_manager import AccountProfile, AccountTier


@dataclass
class PositionAction:
    """Position sizing action"""
    position_size: float  # 0, 0.5x, 1x, 1.5x, 2x base size
    strategy_selection: str
    entry_timing: str  # 'immediate', 'wait_1h', 'wait_4h', 'wait_1d'
    confidence: float


@dataclass
class TradingState:
    """Trading state for RL agent"""
    account_size: float
    portfolio_greeks: Dict[str, float]  # delta, gamma, theta, vega
    market_regime: str
    iv_rank: float
    recent_pnl: List[float]  # Last 10 trades P&L
    volatility: float
    trend_strength: float
    volume_profile: float
    time_of_day: float
    day_of_week: float


@dataclass
class RLMetrics:
    """RL agent performance metrics"""
    total_reward: float
    average_reward: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    episode_length: float
    exploration_rate: float


class PolicyNetwork(nn.Module):
    """Policy network for PPO agent"""
    
    def __init__(self, state_size: int, action_sizes: Dict[str, int], hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_sizes = action_sizes
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Action heads
        self.position_size_head = nn.Sequential(
            nn.Linear(hidden_size // 2, action_sizes['position_size']),
            nn.Softmax(dim=-1)
        )
        
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_size // 2, action_sizes['strategy']),
            nn.Softmax(dim=-1)
        )
        
        self.timing_head = nn.Sequential(
            nn.Linear(hidden_size // 2, action_sizes['timing']),
            nn.Softmax(dim=-1)
        )
        
        # Value head for critic
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, state):
        # Extract features
        features = self.feature_extractor(state)
        
        # Action probabilities
        position_size_probs = self.position_size_head(features)
        strategy_probs = self.strategy_head(features)
        timing_probs = self.timing_head(features)
        
        # Value estimation
        value = self.value_head(features)
        
        return {
            'position_size': position_size_probs,
            'strategy': strategy_probs,
            'timing': timing_probs,
            'value': value
        }


class PositionSizingAgent:
    """
    RL agent for position sizing using Proximal Policy Optimization (PPO)
    
    Features:
    - State space: account size, portfolio Greeks, market regime, IV rank, recent P&L
    - Action space (account-specific): position size (0, 0.5x, 1x, 1.5x, 2x), strategy selection, entry timing
    - Reward function: Sharpe ratio with drawdown penalties and theta collection bonuses
    - Proximal Policy Optimization (PPO) algorithm
    """
    
    def __init__(self, account_profile: AccountProfile, config: Dict = None):
        self.profile = account_profile
        
        # Configuration
        self.config = config or self._default_config()
        
        # State and action spaces
        self.state_size = self._calculate_state_size()
        self.action_sizes = self._get_action_sizes()
        
        # Networks
        self.policy_net = PolicyNetwork(self.state_size, self.action_sizes)
        self.old_policy_net = PolicyNetwork(self.state_size, self.action_sizes)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config['learning_rate'])
        
        # PPO parameters
        self.clip_ratio = self.config['clip_ratio']
        self.value_loss_coef = self.config['value_loss_coef']
        self.entropy_coef = self.config['entropy_coef']
        self.max_grad_norm = self.config['max_grad_norm']
        
        # Training data
        self.episodes = []
        self.current_episode = []
        
        # Performance tracking
        self.training_metrics = []
        self.exploration_rate = 1.0
        
        # Available strategies
        self.available_strategies = self._get_available_strategies()
        
        logger.info(f"PositionSizingAgent initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default configuration for RL agent"""
        return {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'update_epochs': 10,
            'clip_ratio': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'gamma': 0.99,  # Discount factor
            'lambda': 0.95,  # GAE lambda
            'episode_length': 100,
            'exploration_decay': 0.995,
            'min_exploration': 0.1,
            'reward_scaling': 100.0
        }
    
    def _calculate_state_size(self) -> int:
        """Calculate state space size"""
        return 15  # account_size, portfolio_greeks(4), market_regime, iv_rank, 
                   # recent_pnl(5), volatility, trend_strength, volume_profile, 
                   # time_of_day, day_of_week
    
    def _get_action_sizes(self) -> Dict[str, int]:
        """Get action space sizes based on account tier"""
        base_actions = {
            'position_size': 5,  # 0, 0.5x, 1x, 1.5x, 2x
            'strategy': len(self._get_available_strategies()),
            'timing': 4  # immediate, wait_1h, wait_4h, wait_1d
        }
        
        # Adjust for account tier
        if self.profile.tier in [AccountTier.MICRO, AccountTier.SMALL]:
            base_actions['position_size'] = 3  # 0, 0.5x, 1x only
        elif self.profile.tier == AccountTier.INSTITUTIONAL:
            base_actions['position_size'] = 7  # 0, 0.5x, 1x, 1.5x, 2x, 2.5x, 3x
        
        return base_actions
    
    def _get_available_strategies(self) -> List[str]:
        """Get available strategies based on account tier"""
        strategies = {
            AccountTier.MICRO: ['bull_put_spread', 'cash_secured_put'],
            AccountTier.SMALL: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put'],
            AccountTier.MEDIUM: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put', 'iron_condor'],
            AccountTier.LARGE: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put', 'iron_condor', 'calendar_spread'],
            AccountTier.INSTITUTIONAL: ['bull_put_spread', 'bear_call_spread', 'cash_secured_put', 'iron_condor', 
                                       'calendar_spread', 'diagonal_spread', 'straddle_strangle']
        }
        
        return strategies.get(self.profile.tier, strategies[AccountTier.MEDIUM])
    
    def encode_state(self, state: TradingState) -> np.ndarray:
        """Encode trading state to neural network input"""
        try:
            # Encode portfolio Greeks
            greeks = state.portfolio_greeks
            greeks_vector = [
                greeks.get('delta', 0.0),
                greeks.get('gamma', 0.0),
                greeks.get('theta', 0.0),
                greeks.get('vega', 0.0)
            ]
            
            # Encode market regime (one-hot)
            regime_encoding = {
                'LOW_VOL': 0.0,
                'NORMAL_VOL': 0.5,
                'HIGH_VOL': 1.0,
                'CRISIS': 1.0
            }
            regime_value = regime_encoding.get(state.market_regime, 0.5)
            
            # Encode recent P&L (last 5 trades, padded with zeros)
            recent_pnl = state.recent_pnl[-5:] if state.recent_pnl else [0.0] * 5
            while len(recent_pnl) < 5:
                recent_pnl.insert(0, 0.0)
            
            # Normalize account size (log scale)
            normalized_account_size = np.log(state.account_size) / 10.0  # Rough normalization
            
            # Create state vector
            state_vector = np.array([
                normalized_account_size,
                *greeks_vector,
                regime_value,
                state.iv_rank,
                *recent_pnl,
                state.volatility,
                state.trend_strength,
                state.volume_profile,
                state.time_of_day,
                state.day_of_week
            ], dtype=np.float32)
            
            return state_vector
            
        except Exception as e:
            logger.error(f"Error encoding state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def decode_action(self, action_probs: Dict[str, torch.Tensor]) -> PositionAction:
        """Decode action probabilities to PositionAction"""
        try:
            # Sample actions
            position_size_dist = Categorical(action_probs['position_size'])
            strategy_dist = Categorical(action_probs['strategy'])
            timing_dist = Categorical(action_probs['timing'])
            
            # Apply exploration
            if np.random.random() < self.exploration_rate:
                # Random actions for exploration
                position_size_idx = np.random.randint(0, self.action_sizes['position_size'])
                strategy_idx = np.random.randint(0, self.action_sizes['strategy'])
                timing_idx = np.random.randint(0, self.action_sizes['timing'])
            else:
                # Greedy actions
                position_size_idx = position_size_dist.sample().item()
                strategy_idx = strategy_dist.sample().item()
                timing_idx = timing_dist.sample().item()
            
            # Convert indices to actual values
            position_size_map = {
                0: 0.0,    # No position
                1: 0.5,    # Half size
                2: 1.0,    # Full size
                3: 1.5,    # 1.5x size
                4: 2.0,    # 2x size
                5: 2.5,    # 2.5x size (institutional only)
                6: 3.0     # 3x size (institutional only)
            }
            
            timing_map = {
                0: 'immediate',
                1: 'wait_1h',
                2: 'wait_4h',
                3: 'wait_1d'
            }
            
            position_size = position_size_map.get(position_size_idx, 1.0)
            strategy = self.available_strategies[strategy_idx] if strategy_idx < len(self.available_strategies) else 'bull_put_spread'
            entry_timing = timing_map.get(timing_idx, 'immediate')
            
            # Calculate confidence based on action probabilities
            confidence = (
                action_probs['position_size'][position_size_idx].item() *
                action_probs['strategy'][strategy_idx].item() *
                action_probs['timing'][timing_idx].item()
            )
            
            return PositionAction(
                position_size=position_size,
                strategy_selection=strategy,
                entry_timing=entry_timing,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error decoding action: {e}")
            return PositionAction(0.0, 'bull_put_spread', 'immediate', 0.0)
    
    def get_action(self, state: TradingState) -> PositionAction:
        """Get action from current state"""
        try:
            # Encode state
            state_vector = self.encode_state(state)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            
            # Get action probabilities
            with torch.no_grad():
                action_probs = self.policy_net(state_tensor)
            
            # Decode action
            action = self.decode_action(action_probs)
            
            return action
            
        except Exception as e:
            logger.error(f"Error getting action: {e}")
            return PositionAction(0.0, 'bull_put_spread', 'immediate', 0.0)
    
    def calculate_reward(
        self,
        action: PositionAction,
        next_state: TradingState,
        trade_result: Dict[str, Any]
    ) -> float:
        """
        Calculate reward based on trade performance
        
        Args:
            action: Action taken
            next_state: Next state after action
            trade_result: Trade execution result
        
        Returns:
            Reward value
        """
        try:
            reward = 0.0
            
            # Base reward from trade P&L
            pnl = trade_result.get('pnl', 0.0)
            max_loss = trade_result.get('max_loss', 1.0)
            
            if max_loss > 0:
                # Risk-adjusted return
                risk_adjusted_return = pnl / max_loss
                reward += risk_adjusted_return * self.config['reward_scaling']
            
            # Sharpe ratio bonus
            if trade_result.get('sharpe_contribution', 0) > 0:
                reward += trade_result['sharpe_contribution'] * 10.0
            
            # Theta collection bonus
            theta_collected = trade_result.get('theta_collected', 0.0)
            if theta_collected > 0:
                reward += theta_collected * 5.0
            
            # Drawdown penalty
            portfolio_drawdown = trade_result.get('portfolio_drawdown', 0.0)
            if portfolio_drawdown > 0.05:  # 5% drawdown threshold
                reward -= portfolio_drawdown * 20.0
            
            # Position size penalty (encourage appropriate sizing)
            if self.profile.tier in [AccountTier.MICRO, AccountTier.SMALL]:
                if action.position_size > 1.0:
                    reward -= 2.0  # Penalty for oversized positions
            elif action.position_size > 2.0:
                reward -= 1.0  # Penalty for very large positions
            
            # Strategy selection bonus
            strategy_performance = trade_result.get('strategy_performance', {})
            if action.strategy_selection in strategy_performance:
                strategy_score = strategy_performance[action.strategy_selection]
                reward += strategy_score * 2.0
            
            # Timing bonus (reward patience in volatile markets)
            if next_state.market_regime == 'HIGH_VOL' and action.entry_timing != 'immediate':
                reward += 1.0
            elif next_state.market_regime == 'LOW_VOL' and action.entry_timing == 'immediate':
                reward += 0.5
            
            # Confidence bonus
            if action.confidence > 0.8:
                reward += 1.0
            elif action.confidence < 0.3:
                reward -= 0.5
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def store_experience(
        self,
        state: TradingState,
        action: PositionAction,
        reward: float,
        next_state: TradingState,
        done: bool
    ):
        """Store experience for training"""
        try:
            experience = {
                'state': self.encode_state(state),
                'action': {
                    'position_size': self._action_to_index(action.position_size, 'position_size'),
                    'strategy': self._action_to_index(action.strategy_selection, 'strategy'),
                    'timing': self._action_to_index(action.entry_timing, 'timing')
                },
                'reward': reward,
                'next_state': self.encode_state(next_state),
                'done': done
            }
            
            self.current_episode.append(experience)
            
            if done:
                self.episodes.append(self.current_episode.copy())
                self.current_episode.clear()
                
                # Decay exploration
                self.exploration_rate = max(
                    self.config['min_exploration'],
                    self.exploration_rate * self.config['exploration_decay']
                )
            
        except Exception as e:
            logger.error(f"Error storing experience: {e}")
    
    def _action_to_index(self, action_value: Any, action_type: str) -> int:
        """Convert action value to index"""
        try:
            if action_type == 'position_size':
                position_size_map = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 2.5: 5, 3.0: 6}
                return position_size_map.get(action_value, 2)
            elif action_type == 'strategy':
                return self.available_strategies.index(action_value) if action_value in self.available_strategies else 0
            elif action_type == 'timing':
                timing_map = {'immediate': 0, 'wait_1h': 1, 'wait_4h': 2, 'wait_1d': 3}
                return timing_map.get(action_value, 0)
            else:
                return 0
        except:
            return 0
    
    def train(self, num_episodes: int = 1000) -> RLMetrics:
        """
        Train the RL agent using PPO
        
        Args:
            num_episodes: Number of training episodes
        
        Returns:
            Training metrics
        """
        try:
            logger.info(f"Starting RL training for {num_episodes} episodes")
            
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(num_episodes):
                episode_reward = 0.0
                episode_length = 0
                
                # Run episode (simplified - would need actual market simulation)
                # This is a placeholder for the actual episode simulation
                for step in range(self.config['episode_length']):
                    # Generate random state for demonstration
                    state = self._generate_random_state()
                    
                    # Get action
                    action = self.get_action(state)
                    
                    # Simulate trade result
                    trade_result = self._simulate_trade_result(action, state)
                    
                    # Generate next state
                    next_state = self._generate_next_state(state, action)
                    
                    # Calculate reward
                    reward = self.calculate_reward(action, next_state, trade_result)
                    
                    # Store experience
                    done = (step == self.config['episode_length'] - 1)
                    self.store_experience(state, action, reward, next_state, done)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Update policy every few episodes
                if len(self.episodes) >= 10:
                    self._update_policy()
                    self.episodes.clear()
                
                # Log progress
                if episode % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                              f"Exploration: {self.exploration_rate:.3f}")
            
            # Calculate final metrics
            metrics = self._calculate_metrics(episode_rewards, episode_lengths)
            
            # Save model
            self.save_model()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training agent: {e}")
            return self._empty_metrics()
    
    def _update_policy(self):
        """Update policy using PPO"""
        try:
            if len(self.episodes) == 0:
                return
            
            # Collect all experiences
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            for episode in self.episodes:
                for exp in episode:
                    states.append(exp['state'])
                    actions.append(exp['action'])
                    rewards.append(exp['reward'])
                    next_states.append(exp['next_state'])
                    dones.append(exp['done'])
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states))
            actions = {
                'position_size': torch.LongTensor([a['position_size'] for a in actions]),
                'strategy': torch.LongTensor([a['strategy'] for a in actions]),
                'timing': torch.LongTensor([a['timing'] for a in actions])
            }
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.BoolTensor(dones)
            
            # Calculate advantages and returns
            returns = self._calculate_returns(rewards, dones)
            advantages = self._calculate_advantages(states, returns)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update old policy
            self.old_policy_net.load_state_dict(self.policy_net.state_dict())
            
            # PPO update
            for _ in range(self.config['update_epochs']):
                # Get current policy outputs
                current_outputs = self.policy_net(states)
                
                # Get old policy outputs
                with torch.no_grad():
                    old_outputs = self.old_policy_net(states)
                
                # Calculate action probabilities
                current_probs = (
                    current_outputs['position_size'][range(len(actions['position_size'])), actions['position_size']] *
                    current_outputs['strategy'][range(len(actions['strategy'])), actions['strategy']] *
                    current_outputs['timing'][range(len(actions['timing'])), actions['timing']]
                )
                
                old_probs = (
                    old_outputs['position_size'][range(len(actions['position_size'])), actions['position_size']] *
                    old_outputs['strategy'][range(len(actions['strategy'])), actions['strategy']] *
                    old_outputs['timing'][range(len(actions['timing'])), actions['timing']]
                )
                
                # Calculate ratio
                ratio = current_probs / (old_probs + 1e-8)
                
                # Calculate clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(current_outputs['value'].squeeze(), returns)
                
                # Entropy loss
                entropy_loss = -(
                    (current_outputs['position_size'] * torch.log(current_outputs['position_size'] + 1e-8)).sum(dim=1).mean() +
                    (current_outputs['strategy'] * torch.log(current_outputs['strategy'] + 1e-8)).sum(dim=1).mean() +
                    (current_outputs['timing'] * torch.log(current_outputs['timing'] + 1e-8)).sum(dim=1).mean()
                )
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
        except Exception as e:
            logger.error(f"Error updating policy: {e}")
    
    def _calculate_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.config['gamma'] * running_return
            returns[t] = running_return
        
        return returns
    
    def _calculate_advantages(self, states: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using GAE"""
        with torch.no_grad():
            values = self.policy_net(states)['value'].squeeze()
        
        # Simplified GAE calculation
        advantages = returns - values
        return advantages
    
    def _generate_random_state(self) -> TradingState:
        """Generate random state for training (placeholder)"""
        return TradingState(
            account_size=self.profile.balance,
            portfolio_greeks={'delta': np.random.uniform(-50, 50), 'gamma': np.random.uniform(0, 10),
                            'theta': np.random.uniform(-100, 0), 'vega': np.random.uniform(0, 50)},
            market_regime=np.random.choice(['LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL', 'CRISIS']),
            iv_rank=np.random.uniform(0, 1),
            recent_pnl=np.random.uniform(-1000, 1000, 10).tolist(),
            volatility=np.random.uniform(0.1, 0.5),
            trend_strength=np.random.uniform(-1, 1),
            volume_profile=np.random.uniform(0.5, 2.0),
            time_of_day=np.random.uniform(0, 1),
            day_of_week=np.random.uniform(0, 1)
        )
    
    def _generate_next_state(self, state: TradingState, action: PositionAction) -> TradingState:
        """Generate next state (placeholder)"""
        return self._generate_random_state()
    
    def _simulate_trade_result(self, action: PositionAction, state: TradingState) -> Dict[str, Any]:
        """Simulate trade result (placeholder)"""
        return {
            'pnl': np.random.uniform(-1000, 1000),
            'max_loss': 500,
            'sharpe_contribution': np.random.uniform(-0.1, 0.1),
            'theta_collected': np.random.uniform(0, 50),
            'portfolio_drawdown': np.random.uniform(0, 0.1),
            'strategy_performance': {strategy: np.random.uniform(-0.1, 0.1) for strategy in self.available_strategies}
        }
    
    def _calculate_metrics(self, episode_rewards: List[float], episode_lengths: List[float]) -> RLMetrics:
        """Calculate training metrics"""
        try:
            total_reward = sum(episode_rewards)
            average_reward = np.mean(episode_rewards)
            
            # Win rate (episodes with positive reward)
            win_rate = np.mean([r > 0 for r in episode_rewards])
            
            # Sharpe ratio
            if len(episode_rewards) > 1:
                sharpe_ratio = np.mean(episode_rewards) / np.std(episode_rewards) * np.sqrt(len(episode_rewards))
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown
            cumulative_rewards = np.cumsum(episode_rewards)
            running_max = np.maximum.accumulate(cumulative_rewards)
            drawdowns = (running_max - cumulative_rewards) / (running_max + 1e-8)
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            
            return RLMetrics(
                total_reward=total_reward,
                average_reward=average_reward,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                episode_length=np.mean(episode_lengths),
                exploration_rate=self.exploration_rate
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> RLMetrics:
        """Return empty metrics"""
        return RLMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def save_model(self, model_path: str = None) -> bool:
        """Save trained model"""
        try:
            if model_path is None:
                model_path = f"models/position_sizing_agent_{self.profile.tier.value}.pkl"
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'config': self.config,
                'action_sizes': self.action_sizes,
                'available_strategies': self.available_strategies,
                'exploration_rate': self.exploration_rate,
                'account_profile': self.profile
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str = None) -> bool:
        """Load trained model"""
        try:
            if model_path is None:
                model_path = f"models/position_sizing_agent_{self.profile.tier.value}.pkl"
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore components
            self.config = model_data['config']
            self.action_sizes = model_data['action_sizes']
            self.available_strategies = model_data['available_strategies']
            self.exploration_rate = model_data['exploration_rate']
            
            # Load policy network
            self.policy_net = PolicyNetwork(self.state_size, self.action_sizes)
            self.policy_net.load_state_dict(model_data['policy_net_state_dict'])
            
            logger.info(f"Model loaded from {model_path}")
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
    
    # Create RL agent
    agent = PositionSizingAgent(profile)
    
    # Test state encoding and action generation
    test_state = TradingState(
        account_size=25000,
        portfolio_greeks={'delta': 15, 'gamma': 2, 'theta': -25, 'vega': 30},
        market_regime='NORMAL_VOL',
        iv_rank=0.6,
        recent_pnl=[150, -75, 200, 100, -50, 75, 125, -25, 175, 50],
        volatility=0.25,
        trend_strength=0.3,
        volume_profile=1.2,
        time_of_day=0.5,
        day_of_week=0.6
    )
    
    print("Testing Position Sizing Agent...")
    action = agent.get_action(test_state)
    
    print(f"Position Size: {action.position_size}x")
    print(f"Strategy: {action.strategy_selection}")
    print(f"Entry Timing: {action.entry_timing}")
    print(f"Confidence: {action.confidence:.2%}")
    
    # Train agent
    print("\nTraining RL agent...")
    metrics = agent.train(num_episodes=100)
    
    print(f"\nTraining Results:")
    print(f"Total Reward: {metrics.total_reward:.2f}")
    print(f"Average Reward: {metrics.average_reward:.2f}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Episode Length: {metrics.episode_length:.1f}")
    print(f"Exploration Rate: {metrics.exploration_rate:.3f}")
