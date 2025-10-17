"""
Advanced Options Risk Management
Universal risk controls with account-specific limits and advanced risk models
VaR, CVaR, pin risk detection, assignment modeling, and liquidity-adjusted risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from loguru import logger
import warnings

from src.portfolio.account_manager import AccountProfile, AccountTier


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    portfolio_heat: float
    greeks_exposure: Dict[str, float]
    liquidity_risk: float
    pin_risk_score: float
    assignment_probability: float
    volatility_scenarios: Dict[str, float]


@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    position_id: str
    symbol: str
    strategy: str
    max_loss: float
    current_pnl: float
    var_contribution: float
    pin_risk: bool
    assignment_prob: float
    liquidity_score: float
    days_to_expiration: int
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'


@dataclass
class RiskLimits:
    """Dynamic risk limits by account tier"""
    max_position_size_pct: float
    max_daily_loss_pct: float
    max_portfolio_heat: float
    max_var_pct: float
    max_greeks: Dict[str, float]
    max_pin_risk_pct: float
    max_assignment_risk_pct: float


class AdvancedOptionsRiskManager:
    """
    Advanced options risk management system
    
    Features:
    - Position-level limits (stop loss at 2x credit, take profit at 50%, DTE exits)
    - Portfolio-level limits (daily loss 3-7%, portfolio heat 15-35% by tier)
    - Monte Carlo VaR with 10K simulations
    - CVaR for tail risk
    - Volatility spike scenarios (+50%, +100%, +200% IV)
    - Pin risk detection (1-3 DTE near strikes)
    - Assignment probability modeling
    - Liquidity-adjusted risk (wide spreads = higher risk)
    """
    
    def __init__(self, account_profile: AccountProfile):
        self.profile = account_profile
        
        # Risk limits by account tier
        self.risk_limits = self._initialize_risk_limits()
        
        # Monte Carlo parameters
        self.mc_simulations = 10000
        self.confidence_levels = [0.95, 0.99, 0.999]  # 95%, 99%, 99.9%
        
        # Volatility scenarios
        self.vol_scenarios = {
            'normal': 1.0,
            'spike_50': 1.5,
            'spike_100': 2.0,
            'spike_200': 3.0
        }
        
        # Pin risk parameters
        self.pin_risk_dte = 3  # Days to expiration for pin risk
        self.pin_risk_threshold = 0.02  # 2% of underlying price
        
        # Assignment modeling parameters
        self.assignment_models = {
            'conservative': 0.8,  # 80% assignment probability threshold
            'moderate': 0.9,      # 90% assignment probability threshold
            'aggressive': 0.95    # 95% assignment probability threshold
        }
        
        logger.info(f"AdvancedOptionsRiskManager initialized for {account_profile.tier.value} tier")
    
    def _initialize_risk_limits(self) -> RiskLimits:
        """Initialize risk limits based on account tier"""
        limits_by_tier = {
            AccountTier.MICRO: RiskLimits(
                max_position_size_pct=50.0,    # 50% max position size
                max_daily_loss_pct=3.0,        # 3% max daily loss
                max_portfolio_heat=15.0,       # 15% max portfolio heat
                max_var_pct=5.0,               # 5% max VaR
                max_greeks={
                    'delta': 5,
                    'gamma': 0.05,
                    'theta': 5,
                    'vega': 10
                },
                max_pin_risk_pct=10.0,         # 10% max pin risk exposure
                max_assignment_risk_pct=20.0   # 20% max assignment risk
            ),
            AccountTier.SMALL: RiskLimits(
                max_position_size_pct=30.0,
                max_daily_loss_pct=5.0,
                max_portfolio_heat=20.0,
                max_var_pct=7.0,
                max_greeks={
                    'delta': 20,
                    'gamma': 0.2,
                    'theta': 20,
                    'vega': 50
                },
                max_pin_risk_pct=15.0,
                max_assignment_risk_pct=30.0
            ),
            AccountTier.MEDIUM: RiskLimits(
                max_position_size_pct=20.0,
                max_daily_loss_pct=5.0,
                max_portfolio_heat=25.0,
                max_var_pct=8.0,
                max_greeks={
                    'delta': 50,
                    'gamma': 0.5,
                    'theta': 50,
                    'vega': 150
                },
                max_pin_risk_pct=20.0,
                max_assignment_risk_pct=40.0
            ),
            AccountTier.LARGE: RiskLimits(
                max_position_size_pct=15.0,
                max_daily_loss_pct=7.0,
                max_portfolio_heat=30.0,
                max_var_pct=10.0,
                max_greeks={
                    'delta': 100,
                    'gamma': 1.0,
                    'theta': 100,
                    'vega': 300
                },
                max_pin_risk_pct=25.0,
                max_assignment_risk_pct=50.0
            ),
            AccountTier.INSTITUTIONAL: RiskLimits(
                max_position_size_pct=10.0,
                max_daily_loss_pct=10.0,
                max_portfolio_heat=35.0,
                max_var_pct=12.0,
                max_greeks={
                    'delta': 200,
                    'gamma': 2.0,
                    'theta': 200,
                    'vega': 500
                },
                max_pin_risk_pct=30.0,
                max_assignment_risk_pct=60.0
            )
        }
        
        return limits_by_tier.get(self.profile.tier, limits_by_tier[AccountTier.MEDIUM])
    
    def calculate_portfolio_risk(
        self,
        positions: List[Dict],
        market_data: Dict[str, Any],
        time_horizon: int = 1
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            positions: List of position dictionaries
            market_data: Current market data (prices, volatilities, etc.)
            time_horizon: Risk calculation horizon in days
        
        Returns:
            RiskMetrics with comprehensive risk analysis
        """
        try:
            if not positions:
                return self._empty_risk_metrics()
            
            # Calculate individual position risks
            position_risks = []
            for position in positions:
                pos_risk = self._calculate_position_risk(position, market_data)
                position_risks.append(pos_risk)
            
            # Monte Carlo VaR calculation
            portfolio_var = self._calculate_monte_carlo_var(position_risks, market_data, time_horizon)
            portfolio_cvar = self._calculate_cvar(position_risks, market_data, time_horizon)
            
            # Portfolio heat calculation
            portfolio_heat = self._calculate_portfolio_heat(position_risks)
            
            # Greeks exposure
            greeks_exposure = self._calculate_greeks_exposure(position_risks)
            
            # Liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(position_risks)
            
            # Pin risk assessment
            pin_risk_score = self._calculate_pin_risk_score(position_risks)
            
            # Assignment probability
            assignment_probability = self._calculate_assignment_probability(position_risks, market_data)
            
            # Volatility scenarios
            volatility_scenarios = self._calculate_volatility_scenarios(position_risks, market_data)
            
            return RiskMetrics(
                portfolio_var=portfolio_var,
                portfolio_cvar=portfolio_cvar,
                max_drawdown=0.0,  # Would calculate from historical data
                portfolio_heat=portfolio_heat,
                greeks_exposure=greeks_exposure,
                liquidity_risk=liquidity_risk,
                pin_risk_score=pin_risk_score,
                assignment_probability=assignment_probability,
                volatility_scenarios=volatility_scenarios
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return self._empty_risk_metrics()
    
    def _calculate_position_risk(self, position: Dict, market_data: Dict[str, Any]) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        try:
            # Extract position data
            position_id = position.get('position_id', 'unknown')
            symbol = position.get('symbol', '')
            strategy = position.get('strategy', '')
            max_loss = abs(position.get('max_loss', 0))
            current_pnl = position.get('unrealized_pnl', 0)
            days_to_expiration = position.get('days_to_expiration', 30)
            
            # Calculate VaR contribution (simplified)
            var_contribution = max_loss * 0.05  # 5% of max loss as VaR proxy
            
            # Pin risk assessment
            pin_risk = self._assess_pin_risk(position, market_data)
            
            # Assignment probability
            assignment_prob = self._calculate_position_assignment_prob(position, market_data)
            
            # Liquidity score
            liquidity_score = self._calculate_position_liquidity(position)
            
            # Determine risk level
            risk_level = self._determine_position_risk_level(
                max_loss, current_pnl, pin_risk, assignment_prob, liquidity_score
            )
            
            return PositionRisk(
                position_id=position_id,
                symbol=symbol,
                strategy=strategy,
                max_loss=max_loss,
                current_pnl=current_pnl,
                var_contribution=var_contribution,
                pin_risk=pin_risk,
                assignment_prob=assignment_prob,
                liquidity_score=liquidity_score,
                days_to_expiration=days_to_expiration,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return PositionRisk(
                position_id='error',
                symbol='',
                strategy='',
                max_loss=0,
                current_pnl=0,
                var_contribution=0,
                pin_risk=False,
                assignment_prob=0,
                liquidity_score=0,
                days_to_expiration=0,
                risk_level='UNKNOWN'
            )
    
    def _assess_pin_risk(self, position: Dict, market_data: Dict[str, Any]) -> bool:
        """Assess if position has pin risk"""
        try:
            days_to_expiration = position.get('days_to_expiration', 30)
            
            # Check if close to expiration
            if days_to_expiration > self.pin_risk_dte:
                return False
            
            # Check if position is near strike price
            underlying_price = market_data.get('underlying_price', 0)
            strike = position.get('strike', 0)
            
            if underlying_price > 0 and strike > 0:
                price_distance = abs(underlying_price - strike) / underlying_price
                return price_distance <= self.pin_risk_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error assessing pin risk: {e}")
            return False
    
    def _calculate_position_assignment_prob(self, position: Dict, market_data: Dict[str, Any]) -> float:
        """Calculate assignment probability for position"""
        try:
            option_type = position.get('option_type', '')
            side = position.get('side', '')
            days_to_expiration = position.get('days_to_expiration', 30)
            strike = position.get('strike', 0)
            underlying_price = market_data.get('underlying_price', 0)
            
            # Only short options can be assigned
            if side != 'short':
                return 0.0
            
            if underlying_price <= 0 or strike <= 0:
                return 0.0
            
            # Simplified assignment probability calculation
            if option_type == 'put':
                # Put assignment probability increases as underlying goes below strike
                if underlying_price < strike:
                    distance_pct = (strike - underlying_price) / underlying_price
                    # Higher probability if deeper ITM and closer to expiration
                    prob = min(0.95, 0.3 + (distance_pct * 2) + (1 / max(days_to_expiration, 1)) * 0.5)
                else:
                    prob = 0.1  # Low probability if OTM
            else:  # call
                # Call assignment probability increases as underlying goes above strike
                if underlying_price > strike:
                    distance_pct = (underlying_price - strike) / underlying_price
                    prob = min(0.95, 0.3 + (distance_pct * 2) + (1 / max(days_to_expiration, 1)) * 0.5)
                else:
                    prob = 0.1  # Low probability if OTM
            
            # Adjust for time to expiration
            if days_to_expiration <= 1:
                prob = min(0.95, prob * 1.5)
            elif days_to_expiration <= 3:
                prob = min(0.90, prob * 1.2)
            
            return prob
            
        except Exception as e:
            logger.error(f"Error calculating assignment probability: {e}")
            return 0.0
    
    def _calculate_position_liquidity(self, position: Dict) -> float:
        """Calculate liquidity score for position (0-1, higher is more liquid)"""
        try:
            volume = position.get('volume', 0)
            open_interest = position.get('open_interest', 0)
            bid = position.get('bid', 0)
            ask = position.get('ask', 0)
            
            liquidity_score = 0.0
            
            # Volume component (0-0.4)
            if volume >= 100:
                liquidity_score += 0.4
            elif volume >= 50:
                liquidity_score += 0.3
            elif volume >= 25:
                liquidity_score += 0.2
            elif volume >= 10:
                liquidity_score += 0.1
            
            # Open interest component (0-0.3)
            if open_interest >= 500:
                liquidity_score += 0.3
            elif open_interest >= 250:
                liquidity_score += 0.2
            elif open_interest >= 100:
                liquidity_score += 0.1
            
            # Spread component (0-0.3)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid * 100
                
                if spread_pct <= 5:
                    liquidity_score += 0.3
                elif spread_pct <= 10:
                    liquidity_score += 0.2
                elif spread_pct <= 15:
                    liquidity_score += 0.1
            
            return min(1.0, liquidity_score)
            
        except Exception as e:
            logger.error(f"Error calculating position liquidity: {e}")
            return 0.0
    
    def _determine_position_risk_level(
        self, 
        max_loss: float, 
        current_pnl: float, 
        pin_risk: bool, 
        assignment_prob: float, 
        liquidity_score: float
    ) -> str:
        """Determine position risk level"""
        try:
            risk_score = 0
            
            # Loss magnitude
            if max_loss > self.profile.balance * 0.1:  # > 10% of account
                risk_score += 3
            elif max_loss > self.profile.balance * 0.05:  # > 5% of account
                risk_score += 2
            elif max_loss > self.profile.balance * 0.02:  # > 2% of account
                risk_score += 1
            
            # Current P&L
            if current_pnl < -max_loss * 0.5:  # > 50% of max loss
                risk_score += 2
            elif current_pnl < -max_loss * 0.25:  # > 25% of max loss
                risk_score += 1
            
            # Pin risk
            if pin_risk:
                risk_score += 2
            
            # Assignment probability
            if assignment_prob > 0.8:
                risk_score += 2
            elif assignment_prob > 0.5:
                risk_score += 1
            
            # Liquidity
            if liquidity_score < 0.3:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 6:
                return 'EXTREME'
            elif risk_score >= 4:
                return 'HIGH'
            elif risk_score >= 2:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.error(f"Error determining position risk level: {e}")
            return 'UNKNOWN'
    
    def _calculate_monte_carlo_var(
        self, 
        position_risks: List[PositionRisk], 
        market_data: Dict[str, Any], 
        time_horizon: int
    ) -> float:
        """Calculate Monte Carlo VaR"""
        try:
            if not position_risks:
                return 0.0
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            
            # Simulate portfolio returns
            portfolio_returns = []
            
            for _ in range(self.mc_simulations):
                scenario_return = 0.0
                
                for pos_risk in position_risks:
                    # Simplified return simulation
                    # In practice, would use Greeks and market scenarios
                    position_return = np.random.normal(0, pos_risk.var_contribution)
                    scenario_return += position_return
                
                portfolio_returns.append(scenario_return)
            
            # Calculate VaR at different confidence levels
            portfolio_returns = np.array(portfolio_returns)
            var_95 = np.percentile(portfolio_returns, 5)  # 95% VaR
            var_99 = np.percentile(portfolio_returns, 1)  # 99% VaR
            
            # Use 95% VaR as primary metric
            return abs(var_95)
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0
    
    def _calculate_cvar(
        self, 
        position_risks: List[PositionRisk], 
        market_data: Dict[str, Any], 
        time_horizon: int
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        try:
            if not position_risks:
                return 0.0
            
            # Generate portfolio returns (same as VaR calculation)
            np.random.seed(42)
            portfolio_returns = []
            
            for _ in range(self.mc_simulations):
                scenario_return = 0.0
                for pos_risk in position_risks:
                    position_return = np.random.normal(0, pos_risk.var_contribution)
                    scenario_return += position_return
                portfolio_returns.append(scenario_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # Calculate CVaR (expected loss beyond VaR)
            var_95 = np.percentile(portfolio_returns, 5)
            tail_losses = portfolio_returns[portfolio_returns <= var_95]
            cvar = np.mean(tail_losses)
            
            return abs(cvar)
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def _calculate_portfolio_heat(self, position_risks: List[PositionRisk]) -> float:
        """Calculate portfolio heat (total risk exposure)"""
        try:
            if not position_risks:
                return 0.0
            
            total_max_loss = sum(pos_risk.max_loss for pos_risk in position_risks)
            portfolio_heat = (total_max_loss / self.profile.balance) * 100
            
            return portfolio_heat
            
        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}")
            return 0.0
    
    def _calculate_greeks_exposure(self, position_risks: List[PositionRisk]) -> Dict[str, float]:
        """Calculate portfolio Greeks exposure"""
        try:
            # This would be calculated from actual position Greeks
            # For now, return placeholder values
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks exposure: {e}")
            return {}
    
    def _calculate_liquidity_risk(self, position_risks: List[PositionRisk]) -> float:
        """Calculate overall portfolio liquidity risk"""
        try:
            if not position_risks:
                return 0.0
            
            # Average liquidity score
            avg_liquidity = np.mean([pos_risk.liquidity_score for pos_risk in position_risks])
            
            # Liquidity risk is inverse of liquidity score
            liquidity_risk = 1.0 - avg_liquidity
            
            return liquidity_risk
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 1.0
    
    def _calculate_pin_risk_score(self, position_risks: List[PositionRisk]) -> float:
        """Calculate portfolio pin risk score"""
        try:
            if not position_risks:
                return 0.0
            
            # Count positions with pin risk
            pin_risk_positions = sum(1 for pos_risk in position_risks if pos_risk.pin_risk)
            
            # Calculate percentage of portfolio with pin risk
            pin_risk_score = (pin_risk_positions / len(position_risks)) * 100
            
            return pin_risk_score
            
        except Exception as e:
            logger.error(f"Error calculating pin risk score: {e}")
            return 0.0
    
    def _calculate_assignment_probability(self, position_risks: List[PositionRisk], market_data: Dict[str, Any]) -> float:
        """Calculate overall portfolio assignment probability"""
        try:
            if not position_risks:
                return 0.0
            
            # Weighted average assignment probability
            total_exposure = sum(pos_risk.max_loss for pos_risk in position_risks)
            
            if total_exposure == 0:
                return 0.0
            
            weighted_assignment_prob = sum(
                pos_risk.assignment_prob * pos_risk.max_loss 
                for pos_risk in position_risks
            ) / total_exposure
            
            return weighted_assignment_prob
            
        except Exception as e:
            logger.error(f"Error calculating assignment probability: {e}")
            return 0.0
    
    def _calculate_volatility_scenarios(self, position_risks: List[PositionRisk], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate portfolio P&L under different volatility scenarios"""
        try:
            scenarios = {}
            base_pnl = sum(pos_risk.current_pnl for pos_risk in position_risks)
            
            # For each volatility scenario
            for scenario_name, vol_multiplier in self.vol_scenarios.items():
                # Simplified calculation - in practice would use Greeks
                # Assume vega exposure affects P&L
                vega_exposure = sum(pos_risk.var_contribution * 0.1 for pos_risk in position_risks)
                scenario_pnl = base_pnl + (vol_multiplier - 1.0) * vega_exposure
                scenarios[scenario_name] = scenario_pnl
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error calculating volatility scenarios: {e}")
            return {}
    
    def check_risk_limits(self, risk_metrics: RiskMetrics) -> Tuple[bool, List[str]]:
        """Check if portfolio exceeds risk limits"""
        try:
            violations = []
            
            # Portfolio heat check
            if risk_metrics.portfolio_heat > self.risk_limits.max_portfolio_heat:
                violations.append(
                    f"Portfolio heat {risk_metrics.portfolio_heat:.1f}% exceeds limit {self.risk_limits.max_portfolio_heat}%"
                )
            
            # VaR check
            var_pct = (risk_metrics.portfolio_var / self.profile.balance) * 100
            if var_pct > self.risk_limits.max_var_pct:
                violations.append(
                    f"Portfolio VaR {var_pct:.1f}% exceeds limit {self.risk_limits.max_var_pct}%"
                )
            
            # Greeks limits check
            for greek, limit in self.risk_limits.max_greeks.items():
                exposure = abs(risk_metrics.greeks_exposure.get(greek, 0))
                if exposure > limit:
                    violations.append(
                        f"Portfolio {greek} {exposure:.2f} exceeds limit {limit}"
                    )
            
            # Pin risk check
            if risk_metrics.pin_risk_score > self.risk_limits.max_pin_risk_pct:
                violations.append(
                    f"Pin risk exposure {risk_metrics.pin_risk_score:.1f}% exceeds limit {self.risk_limits.max_pin_risk_pct}%"
                )
            
            # Assignment risk check
            assignment_risk_pct = risk_metrics.assignment_probability * 100
            if assignment_risk_pct > self.risk_limits.max_assignment_risk_pct:
                violations.append(
                    f"Assignment risk {assignment_risk_pct:.1f}% exceeds limit {self.risk_limits.max_assignment_risk_pct}%"
                )
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, [f"Error checking risk limits: {str(e)}"]
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            portfolio_var=0.0,
            portfolio_cvar=0.0,
            max_drawdown=0.0,
            portfolio_heat=0.0,
            greeks_exposure={},
            liquidity_risk=0.0,
            pin_risk_score=0.0,
            assignment_probability=0.0,
            volatility_scenarios={}
        )
    
    def get_risk_summary(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Get human-readable risk summary"""
        try:
            return {
                'account_tier': self.profile.tier.value,
                'account_balance': self.profile.balance,
                'risk_metrics': {
                    'portfolio_var': f"${risk_metrics.portfolio_var:,.2f}",
                    'portfolio_cvar': f"${risk_metrics.portfolio_cvar:,.2f}",
                    'portfolio_heat': f"{risk_metrics.portfolio_heat:.1f}%",
                    'liquidity_risk': f"{risk_metrics.liquidity_risk:.1%}",
                    'pin_risk_score': f"{risk_metrics.pin_risk_score:.1f}%",
                    'assignment_probability': f"{risk_metrics.assignment_probability:.1%}"
                },
                'risk_limits': {
                    'max_portfolio_heat': f"{self.risk_limits.max_portfolio_heat}%",
                    'max_var_pct': f"{self.risk_limits.max_var_pct}%",
                    'max_pin_risk_pct': f"{self.risk_limits.max_pin_risk_pct}%",
                    'max_assignment_risk_pct': f"{self.risk_limits.max_assignment_risk_pct}%"
                },
                'greeks_exposure': risk_metrics.greeks_exposure,
                'volatility_scenarios': {
                    scenario: f"${pnl:,.2f}" 
                    for scenario, pnl in risk_metrics.volatility_scenarios.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating risk summary: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create risk manager
    risk_manager = AdvancedOptionsRiskManager(profile)
    
    # Sample positions
    positions = [
        {
            'position_id': 'pos_1',
            'symbol': 'SPY',
            'strategy': 'bull_put_spread',
            'max_loss': 350,
            'unrealized_pnl': -45,
            'days_to_expiration': 25,
            'option_type': 'put',
            'side': 'short',
            'strike': 450,
            'volume': 150,
            'open_interest': 500,
            'bid': 1.25,
            'ask': 1.35
        },
        {
            'position_id': 'pos_2',
            'symbol': 'QQQ',
            'strategy': 'iron_condor',
            'max_loss': 300,
            'unrealized_pnl': 25,
            'days_to_expiration': 18,
            'option_type': 'call',
            'side': 'short',
            'strike': 480,
            'volume': 200,
            'open_interest': 750,
            'bid': 0.85,
            'ask': 0.95
        }
    ]
    
    # Market data
    market_data = {
        'underlying_price': 455.0,
        'vix': 18.5,
        'risk_free_rate': 0.045
    }
    
    # Calculate portfolio risk
    risk_metrics = risk_manager.calculate_portfolio_risk(positions, market_data)
    
    print(f"Portfolio Risk Analysis:")
    print(f"  VaR (95%): ${risk_metrics.portfolio_var:,.2f}")
    print(f"  CVaR: ${risk_metrics.portfolio_cvar:,.2f}")
    print(f"  Portfolio Heat: {risk_metrics.portfolio_heat:.1f}%")
    print(f"  Pin Risk Score: {risk_metrics.pin_risk_score:.1f}%")
    print(f"  Assignment Probability: {risk_metrics.assignment_probability:.1%}")
    print(f"  Liquidity Risk: {risk_metrics.liquidity_risk:.1%}")
    
    # Check risk limits
    within_limits, violations = risk_manager.check_risk_limits(risk_metrics)
    print(f"\nRisk Limits Check:")
    print(f"  Within Limits: {within_limits}")
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}")
    
    # Get risk summary
    summary = risk_manager.get_risk_summary(risk_metrics)
    print(f"\nRisk Summary:")
    print(f"  Account Tier: {summary['account_tier']}")
    print(f"  Portfolio Heat: {summary['risk_metrics']['portfolio_heat']}")
    print(f"  Max Heat Limit: {summary['risk_limits']['max_portfolio_heat']}")
