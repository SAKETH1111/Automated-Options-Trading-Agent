"""
Market Regime Detector
Hidden Markov Model with VIX analysis, put/call ratios, and account-tier-specific responses
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from loguru import logger
import warnings

from src.portfolio.account_manager import AccountProfile, AccountTier


@dataclass
class RegimeState:
    """Market regime state"""
    state_id: int
    name: str
    description: str
    vix_range: Tuple[float, float]
    probability: float
    expected_duration: int  # days
    strategy_preferences: List[str]
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'


@dataclass
class RegimeSignal:
    """Regime transition signal"""
    signal_type: str
    strength: float  # 0-1
    direction: str  # 'transitioning_to', 'staying_in', 'uncertain'
    target_regime: Optional[int]
    confidence: float  # 0-1
    time_horizon: int  # days
    recommended_actions: List[str]


class MarketRegimeDetector:
    """
    Advanced market regime detector using Hidden Markov Models
    
    Features:
    - Hidden Markov Model with 3-5 volatility states
    - VIX term structure analysis
    - Put/call ratio and skew signals
    - Regime-specific strategy recommendations
    - Account-tier-specific responses (ON/OFF for micro, full rebalancing for institutional)
    - Dynamic regime transition detection
    """
    
    def __init__(self, account_profile: AccountProfile):
        self.profile = account_profile
        
        # Regime states definition
        self.regime_states = self._initialize_regime_states()
        
        # Model parameters
        self.n_states = len(self.regime_states)
        self.lookback_days = 252  # 1 year of daily data
        self.min_state_duration = 5  # minimum days in a state
        
        # Model components
        self.hmm_model = None
        self.state_transition_matrix = None
        self.observation_model = None
        
        # Current state
        self.current_regime = 0
        self.regime_history = []
        self.state_probabilities = np.ones(self.n_states) / self.n_states
        
        # Account-specific parameters
        self.account_responses = self._initialize_account_responses()
        
        logger.info(f"MarketRegimeDetector initialized for {account_profile.tier.value} tier")
        logger.info(f"  States: {self.n_states}, Lookback: {self.lookback_days} days")
    
    def _initialize_regime_states(self) -> List[RegimeState]:
        """Initialize market regime states"""
        return [
            RegimeState(
                state_id=0,
                name="LOW_VOL",
                description="Low volatility, trending markets",
                vix_range=(8, 16),
                probability=0.25,
                expected_duration=30,
                strategy_preferences=[
                    "iron_condor", "credit_spread", "calendar_spread",
                    "cash_secured_put", "covered_call"
                ],
                risk_level="LOW"
            ),
            RegimeState(
                state_id=1,
                name="NORMAL_VOL",
                description="Normal volatility, mixed markets",
                vix_range=(16, 24),
                probability=0.40,
                expected_duration=20,
                strategy_preferences=[
                    "bull_put_spread", "bear_call_spread", "iron_condor",
                    "straddle", "strangle"
                ],
                risk_level="MEDIUM"
            ),
            RegimeState(
                state_id=2,
                name="HIGH_VOL",
                description="High volatility, uncertain markets",
                vix_range=(24, 35),
                probability=0.25,
                expected_duration=15,
                strategy_preferences=[
                    "debit_spread", "long_straddle", "long_strangle",
                    "calendar_spread", "volatility_arbitrage"
                ],
                risk_level="HIGH"
            ),
            RegimeState(
                state_id=3,
                name="CRISIS_VOL",
                description="Extreme volatility, panic markets",
                vix_range=(35, 100),
                probability=0.10,
                expected_duration=10,
                strategy_preferences=[
                    "cash", "defensive_hedges", "long_puts",
                    "volatility_arbitrage", "dispersion_trading"
                ],
                risk_level="EXTREME"
            )
        ]
    
    def _initialize_account_responses(self) -> Dict[AccountTier, Dict]:
        """Initialize account-tier-specific regime responses"""
        return {
            AccountTier.MICRO: {
                'regime_switching': 'OFF',  # Turn off trading during regime changes
                'strategy_adjustment': 'MINIMAL',  # Minimal strategy changes
                'position_sizing': 'CONSERVATIVE',  # Reduce size during uncertainty
                'rebalance_frequency': 'WEEKLY'  # Weekly rebalancing
            },
            AccountTier.SMALL: {
                'regime_switching': 'CONSERVATIVE',  # Conservative changes
                'strategy_adjustment': 'MODERATE',  # Moderate strategy changes
                'position_sizing': 'ADAPTIVE',  # Adapt size to regime
                'rebalance_frequency': 'BIWEEKLY'  # Bi-weekly rebalancing
            },
            AccountTier.MEDIUM: {
                'regime_switching': 'MODERATE',  # Moderate regime adaptation
                'strategy_adjustment': 'FULL',  # Full strategy adjustments
                'position_sizing': 'DYNAMIC',  # Dynamic sizing
                'rebalance_frequency': 'WEEKLY'  # Weekly rebalancing
            },
            AccountTier.LARGE: {
                'regime_switching': 'ACTIVE',  # Active regime adaptation
                'strategy_adjustment': 'FULL',  # Full strategy adjustments
                'position_sizing': 'OPTIMAL',  # Optimal sizing per regime
                'rebalance_frequency': 'BIWEEKLY'  # Bi-weekly rebalancing
            },
            AccountTier.INSTITUTIONAL: {
                'regime_switching': 'AGGRESSIVE',  # Aggressive regime adaptation
                'strategy_adjustment': 'ADVANCED',  # Advanced strategy adjustments
                'position_sizing': 'OPTIMAL',  # Optimal sizing per regime
                'rebalance_frequency': 'DAILY'  # Daily rebalancing
            }
        }
    
    def train_model(self, historical_data: pd.DataFrame) -> bool:
        """
        Train the Hidden Markov Model on historical data
        
        Args:
            historical_data: DataFrame with columns ['date', 'vix', 'spy_return', 'put_call_ratio', 'skew']
        
        Returns:
            Success boolean
        """
        try:
            if len(historical_data) < self.lookback_days:
                logger.warning(f"Insufficient data for training: {len(historical_data)} < {self.lookback_days}")
                return False
            
            # Prepare features for HMM
            features = self._prepare_features(historical_data)
            
            if features is None or len(features) < 50:
                logger.error("Insufficient features for HMM training")
                return False
            
            # Train Gaussian Mixture Model (simplified HMM)
            self.hmm_model = GaussianMixture(
                n_components=self.n_states,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            
            self.hmm_model.fit(features)
            
            # Extract state transition matrix
            self.state_transition_matrix = self._estimate_transition_matrix(features)
            
            # Initialize observation model
            self.observation_model = self.hmm_model
            
            logger.info(f"HMM model trained successfully on {len(features)} observations")
            logger.info(f"  States: {self.n_states}, Features: {features.shape[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training HMM model: {e}")
            return False
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for HMM training"""
        try:
            # Required columns
            required_cols = ['vix', 'spy_return', 'put_call_ratio']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"Missing required columns: {required_cols}")
                return None
            
            # Calculate features
            features = []
            
            for i in range(len(data)):
                if i < 20:  # Need at least 20 days for rolling calculations
                    continue
                
                row_features = []
                
                # VIX features
                vix = data.iloc[i]['vix']
                row_features.extend([
                    vix,
                    np.log(vix + 1),  # Log VIX
                    data.iloc[i-20:i]['vix'].mean(),  # 20-day VIX mean
                    data.iloc[i-5:i]['vix'].std()  # 5-day VIX volatility
                ])
                
                # SPY return features
                spy_return = data.iloc[i]['spy_return']
                row_features.extend([
                    spy_return,
                    data.iloc[i-20:i]['spy_return'].mean(),  # 20-day return mean
                    data.iloc[i-20:i]['spy_return'].std()  # 20-day return volatility
                ])
                
                # Put/call ratio features
                put_call_ratio = data.iloc[i]['put_call_ratio']
                row_features.extend([
                    put_call_ratio,
                    np.log(put_call_ratio + 0.1),  # Log put/call ratio
                    data.iloc[i-10:i]['put_call_ratio'].mean()  # 10-day mean
                ])
                
                # Additional regime indicators
                if 'skew' in data.columns:
                    skew = data.iloc[i]['skew']
                    row_features.append(skew)
                else:
                    row_features.append(0.0)
                
                # VIX term structure (if available)
                if 'vix_9d' in data.columns and 'vix_30d' in data.columns:
                    vix_term = data.iloc[i]['vix_30d'] - data.iloc[i]['vix_9d']
                    row_features.append(vix_term)
                else:
                    row_features.append(0.0)
                
                features.append(row_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _estimate_transition_matrix(self, features: np.ndarray) -> np.ndarray:
        """Estimate state transition matrix"""
        try:
            # Predict states for all observations
            states = self.hmm_model.predict(features)
            
            # Count transitions
            transition_counts = np.zeros((self.n_states, self.n_states))
            
            for i in range(1, len(states)):
                from_state = states[i-1]
                to_state = states[i]
                transition_counts[from_state, to_state] += 1
            
            # Normalize to get probabilities
            transition_matrix = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-8)
            
            # Add small amount of uniform noise to avoid zero probabilities
            transition_matrix = 0.9 * transition_matrix + 0.1 / self.n_states
            
            logger.debug(f"Transition matrix estimated:")
            for i, row in enumerate(transition_matrix):
                logger.debug(f"  State {i}: {row}")
            
            return transition_matrix
            
        except Exception as e:
            logger.error(f"Error estimating transition matrix: {e}")
            # Return uniform transition matrix as fallback
            return np.ones((self.n_states, self.n_states)) / self.n_states
    
    def update_regime(self, current_data: Dict[str, float]) -> RegimeSignal:
        """
        Update current regime based on new market data
        
        Args:
            current_data: Dictionary with current market indicators
        
        Returns:
            RegimeSignal with regime transition information
        """
        try:
            if self.hmm_model is None:
                logger.warning("HMM model not trained, using simple regime detection")
                return self._simple_regime_detection(current_data)
            
            # Prepare current features
            current_features = self._prepare_current_features(current_data)
            
            if current_features is None:
                return self._simple_regime_detection(current_data)
            
            # Predict current state probabilities
            state_probs = self.hmm_model.predict_proba(current_features.reshape(1, -1))[0]
            
            # Update state probabilities using transition matrix
            if self.state_transition_matrix is not None:
                state_probs = self.state_transition_matrix.T @ state_probs
                state_probs = state_probs / state_probs.sum()  # Normalize
            
            # Determine current regime
            new_regime = np.argmax(state_probs)
            regime_confidence = state_probs[new_regime]
            
            # Detect regime changes
            regime_change = new_regime != self.current_regime
            change_strength = abs(regime_confidence - self.state_probabilities[new_regime])
            
            # Update state
            old_regime = self.current_regime
            self.current_regime = new_regime
            self.state_probabilities = state_probs
            
            # Record regime history
            self.regime_history.append({
                'timestamp': datetime.utcnow(),
                'regime': new_regime,
                'confidence': regime_confidence,
                'state_probabilities': state_probs.copy()
            })
            
            # Generate regime signal
            signal = self._generate_regime_signal(
                old_regime, new_regime, regime_change, change_strength, regime_confidence
            )
            
            logger.info(f"Regime update: {old_regime} → {new_regime} (confidence: {regime_confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error updating regime: {e}")
            return self._simple_regime_detection(current_data)
    
    def _prepare_current_features(self, current_data: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare features for current data point"""
        try:
            features = []
            
            # VIX features
            vix = current_data.get('vix', 20.0)
            features.extend([
                vix,
                np.log(vix + 1),
                current_data.get('vix_20d_mean', vix),
                current_data.get('vix_5d_std', 2.0)
            ])
            
            # SPY return features
            spy_return = current_data.get('spy_return', 0.0)
            features.extend([
                spy_return,
                current_data.get('spy_20d_mean', 0.0),
                current_data.get('spy_20d_std', 0.02)
            ])
            
            # Put/call ratio features
            put_call_ratio = current_data.get('put_call_ratio', 1.0)
            features.extend([
                put_call_ratio,
                np.log(put_call_ratio + 0.1),
                current_data.get('put_call_10d_mean', put_call_ratio)
            ])
            
            # Additional features
            features.extend([
                current_data.get('skew', 0.0),
                current_data.get('vix_term_structure', 0.0)
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing current features: {e}")
            return None
    
    def _generate_regime_signal(
        self, 
        old_regime: int, 
        new_regime: int, 
        regime_change: bool,
        change_strength: float,
        confidence: float
    ) -> RegimeSignal:
        """Generate regime transition signal"""
        try:
            current_state = self.regime_states[new_regime]
            
            if regime_change:
                signal_type = "REGIME_TRANSITION"
                direction = f"transitioning_to_{current_state.name}"
                target_regime = new_regime
                strength = change_strength
                
                # Generate recommended actions based on account tier
                recommended_actions = self._get_regime_transition_actions(old_regime, new_regime)
                
            else:
                signal_type = "REGIME_CONFIRMATION"
                direction = "staying_in"
                target_regime = new_regime
                strength = confidence
                recommended_actions = ["Continue current strategy allocation"]
            
            return RegimeSignal(
                signal_type=signal_type,
                strength=strength,
                direction=direction,
                target_regime=target_regime,
                confidence=confidence,
                time_horizon=current_state.expected_duration,
                recommended_actions=recommended_actions
            )
            
        except Exception as e:
            logger.error(f"Error generating regime signal: {e}")
            return RegimeSignal("ERROR", 0.0, "error", None, 0.0, 0, [])
    
    def _get_regime_transition_actions(self, old_regime: int, new_regime: int) -> List[str]:
        """Get recommended actions for regime transition based on account tier"""
        try:
            old_state = self.regime_states[old_regime]
            new_state = self.regime_states[new_regime]
            
            # Get account-specific response configuration
            account_response = self.account_responses.get(self.profile.tier, {})
            
            actions = []
            
            # Strategy adjustment actions
            strategy_adjustment = account_response.get('strategy_adjustment', 'MODERATE')
            
            if strategy_adjustment == 'MINIMAL':
                actions.append("Continue current strategies with minor adjustments")
            elif strategy_adjustment == 'MODERATE':
                actions.append(f"Gradually shift to {new_state.name} strategies")
                actions.append(f"Reduce exposure to {old_state.name} strategies")
            elif strategy_adjustment in ['FULL', 'ADVANCED']:
                actions.append(f"Rebalance to {new_state.name} strategy preferences")
                actions.append(f"Exit {old_state.name} positions over 1-2 weeks")
                actions.append(f"Enter {new_state.name} positions gradually")
            
            # Position sizing actions
            position_sizing = account_response.get('position_sizing', 'ADAPTIVE')
            
            if position_sizing == 'CONSERVATIVE':
                if new_state.risk_level in ['HIGH', 'EXTREME']:
                    actions.append("Reduce position sizes by 25%")
            elif position_sizing == 'ADAPTIVE':
                if new_state.risk_level == 'EXTREME':
                    actions.append("Reduce position sizes by 50%")
                elif new_state.risk_level == 'HIGH':
                    actions.append("Reduce position sizes by 25%")
            elif position_sizing in ['DYNAMIC', 'OPTIMAL']:
                actions.append(f"Adjust position sizes for {new_state.risk_level} risk environment")
            
            # Regime switching actions
            regime_switching = account_response.get('regime_switching', 'MODERATE')
            
            if regime_switching == 'OFF':
                actions.append("Pause new trades during regime transition")
            elif regime_switching == 'CONSERVATIVE':
                actions.append("Reduce trading frequency during transition")
            elif regime_switching in ['MODERATE', 'ACTIVE', 'AGGRESSIVE']:
                actions.append("Continue trading with regime-appropriate strategies")
            
            return actions
            
        except Exception as e:
            logger.error(f"Error getting regime transition actions: {e}")
            return ["Monitor market conditions carefully"]
    
    def _simple_regime_detection(self, current_data: Dict[str, float]) -> RegimeSignal:
        """Simple regime detection when HMM model is not available"""
        try:
            vix = current_data.get('vix', 20.0)
            
            # Simple VIX-based regime classification
            if vix < 16:
                regime = 0  # LOW_VOL
            elif vix < 24:
                regime = 1  # NORMAL_VOL
            elif vix < 35:
                regime = 2  # HIGH_VOL
            else:
                regime = 3  # CRISIS_VOL
            
            regime_change = regime != self.current_regime
            self.current_regime = regime
            
            return RegimeSignal(
                signal_type="SIMPLE_REGIME",
                strength=0.8 if regime_change else 0.5,
                direction="transitioning_to" if regime_change else "staying_in",
                target_regime=regime,
                confidence=0.7,
                time_horizon=20,
                recommended_actions=self._get_regime_transition_actions(self.current_regime, regime)
            )
            
        except Exception as e:
            logger.error(f"Error in simple regime detection: {e}")
            return RegimeSignal("ERROR", 0.0, "error", None, 0.0, 0, [])
    
    def get_strategy_recommendations(self, regime_id: Optional[int] = None) -> Dict[str, Any]:
        """Get strategy recommendations for current or specified regime"""
        try:
            if regime_id is None:
                regime_id = self.current_regime
            
            if regime_id >= len(self.regime_states):
                regime_id = 0
            
            state = self.regime_states[regime_id]
            
            # Filter strategies by account tier
            account_strategies = self.profile.enabled_strategies
            recommended_strategies = [
                strategy for strategy in state.strategy_preferences 
                if strategy in account_strategies
            ]
            
            return {
                'regime': state.name,
                'regime_id': regime_id,
                'risk_level': state.risk_level,
                'description': state.description,
                'recommended_strategies': recommended_strategies,
                'expected_duration': state.expected_duration,
                'confidence': self.state_probabilities[regime_id],
                'account_tier': self.profile.tier.value,
                'strategy_count': len(recommended_strategies)
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            return {}
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime summary"""
        try:
            current_state = self.regime_states[self.current_regime]
            
            # Calculate regime statistics
            if len(self.regime_history) > 0:
                recent_history = self.regime_history[-30:]  # Last 30 observations
                regime_counts = {}
                for obs in recent_history:
                    regime = obs['regime']
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                most_common_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else self.current_regime
                regime_stability = regime_counts.get(self.current_regime, 0) / len(recent_history)
            else:
                most_common_regime = self.current_regime
                regime_stability = 1.0
            
            return {
                'current_regime': {
                    'id': self.current_regime,
                    'name': current_state.name,
                    'description': current_state.description,
                    'risk_level': current_state.risk_level,
                    'confidence': self.state_probabilities[self.current_regime]
                },
                'state_probabilities': {
                    state.name: float(prob) for state, prob in zip(self.regime_states, self.state_probabilities)
                },
                'regime_statistics': {
                    'most_common_recent': most_common_regime,
                    'stability_score': regime_stability,
                    'total_observations': len(self.regime_history),
                    'days_in_current_regime': len([obs for obs in self.regime_history if obs['regime'] == self.current_regime])
                },
                'account_configuration': self.account_responses.get(self.profile.tier, {}),
                'model_status': {
                    'trained': self.hmm_model is not None,
                    'states': self.n_states,
                    'lookback_days': self.lookback_days
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting regime summary: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create regime detector
    detector = MarketRegimeDetector(profile)
    
    # Sample historical data for training
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'vix': np.random.normal(20, 5, len(dates)),
        'spy_return': np.random.normal(0.001, 0.02, len(dates)),
        'put_call_ratio': np.random.normal(1.0, 0.2, len(dates)),
        'skew': np.random.normal(0.0, 0.1, len(dates))
    })
    
    # Train model
    trained = detector.train_model(sample_data)
    print(f"Model trained: {trained}")
    
    # Update with current data
    current_data = {
        'vix': 18.5,
        'spy_return': 0.02,
        'put_call_ratio': 1.1,
        'skew': 0.05
    }
    
    signal = detector.update_regime(current_data)
    print(f"\nRegime Signal:")
    print(f"  Type: {signal.signal_type}")
    print(f"  Strength: {signal.strength:.2f}")
    print(f"  Direction: {signal.direction}")
    print(f"  Confidence: {signal.confidence:.2f}")
    print(f"  Recommended Actions:")
    for action in signal.recommended_actions:
        print(f"    - {action}")
    
    # Get strategy recommendations
    recommendations = detector.get_strategy_recommendations()
    print(f"\nStrategy Recommendations:")
    print(f"  Regime: {recommendations['regime']}")
    print(f"  Risk Level: {recommendations['risk_level']}")
    print(f"  Recommended Strategies: {recommendations['recommended_strategies']}")
    
    # Get regime summary
    summary = detector.get_regime_summary()
    print(f"\nRegime Summary:")
    print(f"  Current: {summary['current_regime']['name']} ({summary['current_regime']['confidence']:.2f})")
    print(f"  Stability: {summary['regime_statistics']['stability_score']:.2f}")
    print(f"  Model Trained: {summary['model_status']['trained']}")
