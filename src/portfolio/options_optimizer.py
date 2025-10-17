"""
Greeks-Based Portfolio Optimizer
Mean-variance optimization, Black-Litterman, and risk parity for options portfolios
Account-aware Greeks constraints and multi-strategy correlation management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import minimize
from loguru import logger

from src.portfolio.account_manager import AccountProfile, AccountTier


@dataclass
class PortfolioPosition:
    """Portfolio position with Greeks"""
    symbol: str
    strategy: str
    quantity: int
    entry_price: float
    current_price: float
    max_loss: float
    max_profit: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0
    days_to_expiration: int = 0
    implied_volatility: float = 0.0


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    delta_per_dollar: float
    theta_per_day: float
    vega_per_vol_point: float


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    greeks: PortfolioGreeks
    strategy_allocation: Dict[str, float]
    constraints_satisfied: bool
    optimization_method: str


class GreeksPortfolioOptimizer:
    """
    Advanced portfolio optimizer for options trading
    
    Features:
    - Mean-variance optimization with Greeks constraints
    - Black-Litterman model for views incorporation
    - Risk parity for balanced Greeks exposure
    - Account-size-aware Greek limits
    - Multi-strategy correlation management
    - Margin efficiency optimization
    - Dynamic rebalancing triggers
    """
    
    def __init__(self, account_profile: AccountProfile):
        self.profile = account_profile
        
        # Optimization parameters
        self.risk_free_rate = 0.045  # 4.5% annual
        self.target_sharpe = 2.0
        
        # Greeks constraints from profile
        self.greeks_limits = {
            'max_delta': account_profile.max_portfolio_delta,
            'max_gamma': account_profile.max_portfolio_gamma,
            'max_theta': account_profile.max_portfolio_theta,
            'max_vega': account_profile.max_portfolio_vega
        }
        
        logger.info(f"GreeksPortfolioOptimizer initialized for {account_profile.tier.value} tier")
        logger.info(f"  Greeks limits: Delta ±{self.greeks_limits['max_delta']}, Gamma {self.greeks_limits['max_gamma']}")
    
    def calculate_portfolio_greeks(
        self, 
        positions: List[PortfolioPosition]
    ) -> PortfolioGreeks:
        """
        Calculate aggregated portfolio Greeks
        
        Args:
            positions: List of portfolio positions
        
        Returns:
            PortfolioGreeks with aggregated values
        """
        try:
            if not positions:
                return PortfolioGreeks(0, 0, 0, 0, 0, 0, 0, 0)
            
            total_delta = sum(pos.delta * pos.quantity for pos in positions)
            total_gamma = sum(pos.gamma * pos.quantity for pos in positions)
            total_theta = sum(pos.theta * pos.quantity for pos in positions)
            total_vega = sum(pos.vega * pos.quantity for pos in positions)
            total_rho = sum(pos.rho * pos.quantity for pos in positions)
            
            # Calculate per-dollar exposures
            total_value = sum(abs(pos.current_price * pos.quantity * 100) for pos in positions)
            
            delta_per_dollar = total_delta / (total_value / 100) if total_value > 0 else 0
            theta_per_day = total_theta
            vega_per_vol_point = total_vega
            
            return PortfolioGreeks(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_theta=total_theta,
                total_vega=total_vega,
                total_rho=total_rho,
                delta_per_dollar=delta_per_dollar,
                theta_per_day=theta_per_day,
                vega_per_vol_point=vega_per_vol_point
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            return PortfolioGreeks(0, 0, 0, 0, 0, 0, 0, 0)
    
    def optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        positions: List[PortfolioPosition],
        target_return: float = None
    ) -> OptimizationResult:
        """
        Mean-variance optimization with Greeks constraints
        
        Args:
            expected_returns: Expected returns for each position
            covariance_matrix: Return covariance matrix
            positions: List of positions to optimize
            target_return: Target portfolio return (optional)
        
        Returns:
            OptimizationResult with optimal weights
        """
        try:
            n_positions = len(positions)
            
            if n_positions == 0:
                raise ValueError("No positions to optimize")
            
            # Extract Greeks for constraints
            deltas = np.array([pos.delta * pos.quantity for pos in positions])
            gammas = np.array([pos.gamma * pos.quantity for pos in positions])
            thetas = np.array([pos.theta * pos.quantity for pos in positions])
            vegas = np.array([pos.vega * pos.quantity for pos in positions])
            
            # Objective function: minimize portfolio variance
            def portfolio_variance(weights):
                return weights.T @ covariance_matrix @ weights
            
            # Constraints
            constraints = []
            
            # Weights sum to 1
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Target return constraint (if specified)
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: w @ expected_returns - target_return
                })
            
            # Greeks constraints
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.greeks_limits['max_delta'] - abs(w @ deltas)
            })
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.greeks_limits['max_gamma'] - abs(w @ gammas)
            })
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.greeks_limits['max_vega'] - abs(w @ vegas)
            })
            
            # Bounds: 0 <= weight <= 1 for each position
            bounds = tuple((0, 1) for _ in range(n_positions))
            
            # Initial guess: equal weights
            w0 = np.array([1.0 / n_positions] * n_positions)
            
            # Optimize
            result = minimize(
                portfolio_variance,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
            
            # Calculate portfolio metrics
            optimal_weights = result.x
            expected_return = optimal_weights @ expected_returns
            expected_vol = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
            sharpe = (expected_return - self.risk_free_rate) / expected_vol if expected_vol > 0 else 0
            
            # Calculate portfolio Greeks with optimal weights
            portfolio_greeks = PortfolioGreeks(
                total_delta=optimal_weights @ deltas,
                total_gamma=optimal_weights @ gammas,
                total_theta=optimal_weights @ thetas,
                total_vega=optimal_weights @ vegas,
                total_rho=0.0,
                delta_per_dollar=0.0,
                theta_per_day=optimal_weights @ thetas,
                vega_per_vol_point=optimal_weights @ vegas
            )
            
            # Strategy allocation
            strategy_allocation = {}
            for i, pos in enumerate(positions):
                if pos.strategy not in strategy_allocation:
                    strategy_allocation[pos.strategy] = 0.0
                strategy_allocation[pos.strategy] += optimal_weights[i]
            
            # Check constraints
            constraints_satisfied = (
                abs(portfolio_greeks.total_delta) <= self.greeks_limits['max_delta'] and
                abs(portfolio_greeks.total_gamma) <= self.greeks_limits['max_gamma'] and
                abs(portfolio_greeks.total_vega) <= self.greeks_limits['max_vega']
            )
            
            return OptimizationResult(
                weights=optimal_weights,
                expected_return=expected_return,
                expected_volatility=expected_vol,
                sharpe_ratio=sharpe,
                greeks=portfolio_greeks,
                strategy_allocation=strategy_allocation,
                constraints_satisfied=constraints_satisfied,
                optimization_method='mean_variance'
            )
            
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            # Return equal weights as fallback
            n = len(positions)
            equal_weights = np.array([1.0/n] * n)
            return OptimizationResult(
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                greeks=self.calculate_portfolio_greeks(positions),
                strategy_allocation={},
                constraints_satisfied=False,
                optimization_method='equal_weight_fallback'
            )
    
    def optimize_black_litterman(
        self,
        market_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        positions: List[PortfolioPosition],
        views: Dict[str, float] = None,
        view_confidence: float = 0.5
    ) -> OptimizationResult:
        """
        Black-Litterman optimization incorporating market views
        
        Args:
            market_returns: Market equilibrium returns
            covariance_matrix: Return covariance matrix
            positions: List of positions
            views: Dictionary of views {strategy: expected_return}
            view_confidence: Confidence in views (0-1)
        
        Returns:
            OptimizationResult with BL weights
        """
        try:
            n_positions = len(positions)
            
            # If no views, default to market returns
            if views is None or len(views) == 0:
                adjusted_returns = market_returns
            else:
                # Build views matrix P and views vector Q
                view_strategies = list(views.keys())
                P = np.zeros((len(view_strategies), n_positions))
                Q = np.zeros(len(view_strategies))
                
                for i, strategy in enumerate(view_strategies):
                    Q[i] = views[strategy]
                    for j, pos in enumerate(positions):
                        if pos.strategy == strategy:
                            P[i, j] = 1.0
                
                # View uncertainty (Omega)
                tau = 0.025  # Scaling factor
                omega = np.diag(np.diag(P @ (tau * covariance_matrix) @ P.T)) / view_confidence
                
                # Black-Litterman formula
                # E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
                tau_sigma_inv = np.linalg.inv(tau * covariance_matrix)
                p_omega_p = P.T @ np.linalg.inv(omega) @ P
                
                middle_term = np.linalg.inv(tau_sigma_inv + p_omega_p)
                right_term = tau_sigma_inv @ market_returns + P.T @ np.linalg.inv(omega) @ Q
                
                adjusted_returns = middle_term @ right_term
            
            # Now optimize with adjusted returns
            return self.optimize_mean_variance(
                expected_returns=adjusted_returns,
                covariance_matrix=covariance_matrix,
                positions=positions
            )
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return self.optimize_mean_variance(market_returns, covariance_matrix, positions)
    
    def optimize_risk_parity(
        self,
        covariance_matrix: np.ndarray,
        positions: List[PortfolioPosition]
    ) -> OptimizationResult:
        """
        Risk parity optimization - equal risk contribution from each position
        
        Args:
            covariance_matrix: Return covariance matrix
            positions: List of positions
        
        Returns:
            OptimizationResult with risk parity weights
        """
        try:
            n_positions = len(positions)
            
            # Objective: minimize sum of squared differences in risk contributions
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
                marginal_contrib = covariance_matrix @ weights
                risk_contrib = weights * marginal_contrib / portfolio_vol
                
                # Target: equal risk contribution
                target_risk = portfolio_vol / n_positions
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]
            
            # Greeks constraints
            deltas = np.array([pos.delta * pos.quantity for pos in positions])
            gammas = np.array([pos.gamma * pos.quantity for pos in positions])
            vegas = np.array([pos.vega * pos.quantity for pos in positions])
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.greeks_limits['max_delta'] - abs(w @ deltas)
            })
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.greeks_limits['max_gamma'] - abs(w @ gammas)
            })
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.greeks_limits['max_vega'] - abs(w @ vegas)
            })
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(n_positions))
            
            # Initial guess
            w0 = np.array([1.0 / n_positions] * n_positions)
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            optimal_weights = result.x
            portfolio_vol = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
            
            # Calculate portfolio Greeks
            portfolio_greeks = PortfolioGreeks(
                total_delta=optimal_weights @ deltas,
                total_gamma=optimal_weights @ gammas,
                total_theta=optimal_weights @ np.array([pos.theta * pos.quantity for pos in positions]),
                total_vega=optimal_weights @ vegas,
                total_rho=0.0,
                delta_per_dollar=0.0,
                theta_per_day=optimal_weights @ np.array([pos.theta * pos.quantity for pos in positions]),
                vega_per_vol_point=optimal_weights @ vegas
            )
            
            # Strategy allocation
            strategy_allocation = {}
            for i, pos in enumerate(positions):
                if pos.strategy not in strategy_allocation:
                    strategy_allocation[pos.strategy] = 0.0
                strategy_allocation[pos.strategy] += optimal_weights[i]
            
            constraints_satisfied = (
                abs(portfolio_greeks.total_delta) <= self.greeks_limits['max_delta'] and
                abs(portfolio_greeks.total_gamma) <= self.greeks_limits['max_gamma'] and
                abs(portfolio_greeks.total_vega) <= self.greeks_limits['max_vega']
            )
            
            return OptimizationResult(
                weights=optimal_weights,
                expected_return=0.0,
                expected_volatility=portfolio_vol,
                sharpe_ratio=0.0,
                greeks=portfolio_greeks,
                strategy_allocation=strategy_allocation,
                constraints_satisfied=constraints_satisfied,
                optimization_method='risk_parity'
            )
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            n = len(positions)
            equal_weights = np.array([1.0/n] * n)
            return OptimizationResult(
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                greeks=self.calculate_portfolio_greeks(positions),
                strategy_allocation={},
                constraints_satisfied=False,
                optimization_method='equal_weight_fallback'
            )
    
    def calculate_theta_projection(
        self,
        positions: List[PortfolioPosition],
        days_ahead: int = 30
    ) -> Dict[str, float]:
        """
        Project theta collection over time
        
        Args:
            positions: List of positions
            days_ahead: Number of days to project
        
        Returns:
            Dictionary with theta projections
        """
        try:
            daily_theta = sum(pos.theta * pos.quantity for pos in positions)
            
            # Assume theta decay accelerates as expiration approaches
            # Simple linear projection for now
            cumulative_theta = {}
            for day in range(1, days_ahead + 1):
                cumulative_theta[f"day_{day}"] = daily_theta * day
            
            return {
                'daily_theta': daily_theta,
                'weekly_theta': daily_theta * 7,
                'monthly_theta': daily_theta * 30,
                'projections': cumulative_theta
            }
            
        except Exception as e:
            logger.error(f"Error calculating theta projection: {e}")
            return {'daily_theta': 0, 'weekly_theta': 0, 'monthly_theta': 0}
    
    def check_rebalance_needed(
        self,
        current_greeks: PortfolioGreeks,
        threshold_pct: float = 0.8
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio rebalancing is needed
        
        Args:
            current_greeks: Current portfolio Greeks
            threshold_pct: Threshold as percentage of limits (0-1)
        
        Returns:
            (needs_rebalance, reasons)
        """
        try:
            reasons = []
            
            # Check delta
            delta_threshold = self.greeks_limits['max_delta'] * threshold_pct
            if abs(current_greeks.total_delta) > delta_threshold:
                reasons.append(f"Delta {current_greeks.total_delta:.2f} exceeds {delta_threshold:.2f}")
            
            # Check gamma
            gamma_threshold = self.greeks_limits['max_gamma'] * threshold_pct
            if abs(current_greeks.total_gamma) > gamma_threshold:
                reasons.append(f"Gamma {current_greeks.total_gamma:.4f} exceeds {gamma_threshold:.4f}")
            
            # Check vega
            vega_threshold = self.greeks_limits['max_vega'] * threshold_pct
            if abs(current_greeks.total_vega) > vega_threshold:
                reasons.append(f"Vega {current_greeks.total_vega:.2f} exceeds {vega_threshold:.2f}")
            
            needs_rebalance = len(reasons) > 0
            
            return needs_rebalance, reasons
            
        except Exception as e:
            logger.error(f"Error checking rebalance: {e}")
            return False, []
    
    def suggest_hedge_positions(
        self,
        current_greeks: PortfolioGreeks,
        underlying_price: float
    ) -> List[Dict[str, Any]]:
        """
        Suggest hedging positions to neutralize Greeks
        
        Args:
            current_greeks: Current portfolio Greeks
            underlying_price: Current underlying price
        
        Returns:
            List of suggested hedge positions
        """
        try:
            suggestions = []
            
            # Delta hedge
            if abs(current_greeks.total_delta) > self.greeks_limits['max_delta'] * 0.5:
                # Calculate shares needed to hedge delta
                shares_to_hedge = -int(current_greeks.total_delta)
                suggestions.append({
                    'type': 'stock_hedge',
                    'action': 'buy' if shares_to_hedge > 0 else 'sell',
                    'quantity': abs(shares_to_hedge),
                    'reason': f'Hedge delta exposure of {current_greeks.total_delta:.2f}',
                    'estimated_cost': abs(shares_to_hedge) * underlying_price
                })
            
            # Gamma hedge
            if abs(current_greeks.total_gamma) > self.greeks_limits['max_gamma'] * 0.7:
                suggestions.append({
                    'type': 'gamma_hedge',
                    'action': 'buy_atm_straddle' if current_greeks.total_gamma < 0 else 'sell_atm_straddle',
                    'reason': f'Hedge gamma exposure of {current_greeks.total_gamma:.4f}',
                    'note': 'ATM options have highest gamma'
                })
            
            # Vega hedge
            if abs(current_greeks.total_vega) > self.greeks_limits['max_vega'] * 0.7:
                suggestions.append({
                    'type': 'vega_hedge',
                    'action': 'buy_options' if current_greeks.total_vega < 0 else 'sell_options',
                    'reason': f'Hedge vega exposure of {current_greeks.total_vega:.2f}',
                    'note': 'Long-dated options have highest vega'
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting hedge positions: {e}")
            return []
    
    def calculate_margin_efficiency(
        self,
        positions: List[PortfolioPosition],
        available_margin: float
    ) -> Dict[str, Any]:
        """
        Calculate margin efficiency metrics
        
        Args:
            positions: List of positions
            available_margin: Available margin
        
        Returns:
            Dictionary with margin efficiency metrics
        """
        try:
            # Calculate total margin used
            total_margin_used = sum(abs(pos.max_loss * pos.quantity) for pos in positions)
            
            # Calculate total theta collection
            total_theta = sum(pos.theta * pos.quantity for pos in positions)
            
            # Margin efficiency = theta per dollar of margin
            margin_efficiency = total_theta / total_margin_used if total_margin_used > 0 else 0
            
            # Utilization
            margin_utilization = (total_margin_used / available_margin) * 100 if available_margin > 0 else 0
            
            return {
                'total_margin_used': total_margin_used,
                'available_margin': available_margin,
                'margin_utilization_pct': margin_utilization,
                'total_daily_theta': total_theta,
                'margin_efficiency': margin_efficiency,
                'theta_per_1k_margin': (total_theta / total_margin_used * 1000) if total_margin_used > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating margin efficiency: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager, AccountTier
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create optimizer
    optimizer = GreeksPortfolioOptimizer(profile)
    
    # Sample positions
    positions = [
        PortfolioPosition(
            symbol='SPY',
            strategy='bull_put_spread',
            quantity=1,
            entry_price=1.50,
            current_price=1.20,
            max_loss=350,
            max_profit=150,
            delta=15,
            gamma=0.03,
            theta=-8,
            vega=12,
            days_to_expiration=30,
            implied_volatility=0.20
        ),
        PortfolioPosition(
            symbol='QQQ',
            strategy='iron_condor',
            quantity=1,
            entry_price=2.00,
            current_price=1.50,
            max_loss=300,
            max_profit=200,
            delta=-5,
            gamma=0.02,
            theta=-10,
            vega=15,
            days_to_expiration=25,
            implied_volatility=0.22
        )
    ]
    
    # Calculate portfolio Greeks
    portfolio_greeks = optimizer.calculate_portfolio_greeks(positions)
    print(f"\nPortfolio Greeks:")
    print(f"  Delta: {portfolio_greeks.total_delta:.2f}")
    print(f"  Gamma: {portfolio_greeks.total_gamma:.4f}")
    print(f"  Theta: {portfolio_greeks.total_theta:.2f}")
    print(f"  Vega: {portfolio_greeks.total_vega:.2f}")
    
    # Check if rebalancing needed
    needs_rebalance, reasons = optimizer.check_rebalance_needed(portfolio_greeks)
    print(f"\nRebalance needed: {needs_rebalance}")
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")
    
    # Theta projection
    theta_proj = optimizer.calculate_theta_projection(positions, days_ahead=30)
    print(f"\nTheta Projection:")
    print(f"  Daily: ${theta_proj['daily_theta']:.2f}")
    print(f"  Weekly: ${theta_proj['weekly_theta']:.2f}")
    print(f"  Monthly: ${theta_proj['monthly_theta']:.2f}")
    
    # Margin efficiency
    margin_metrics = optimizer.calculate_margin_efficiency(positions, available_margin=5000)
    print(f"\nMargin Efficiency:")
    print(f"  Utilization: {margin_metrics['margin_utilization_pct']:.1f}%")
    print(f"  Theta per $1K margin: ${margin_metrics['theta_per_1k_margin']:.2f}")

