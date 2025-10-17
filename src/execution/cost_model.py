"""
Transaction Cost Model
Pre-trade cost estimation and post-trade analysis with account-specific multipliers
Realistic execution simulation with slippage and market impact modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import json

from src.portfolio.account_manager import AccountProfile, AccountTier


@dataclass
class CostEstimate:
    """Transaction cost estimate"""
    symbol: str
    side: str
    quantity: int
    estimated_commission: float
    estimated_slippage: float
    estimated_market_impact: float
    estimated_opportunity_cost: float
    total_estimated_cost: float
    cost_per_contract: float
    cost_as_pct_of_notional: float
    confidence_interval: Tuple[float, float]
    model_version: str


@dataclass
class ExecutionAnalysis:
    """Post-execution cost analysis"""
    order_id: str
    symbol: str
    estimated_cost: CostEstimate
    actual_cost: float
    actual_commission: float
    actual_slippage: float
    actual_market_impact: float
    cost_variance: float
    execution_quality_score: float
    improvement_recommendations: List[str]


@dataclass
class CostModelMetrics:
    """Cost model performance metrics"""
    prediction_accuracy: float
    mae: float
    rmse: float
    over_estimate_rate: float
    under_estimate_rate: float
    confidence_calibration: float


class TransactionCostModel:
    """
    Advanced transaction cost model for options trading
    
    Features:
    - Pre-trade cost estimate vs actual
    - Slippage by time of day and symbol
    - Fill rate and partial fill analysis
    - Market impact by order size
    - Opportunity cost (missed fills)
    - Account-size-specific benchmarks
    """
    
    def __init__(self, account_profile: AccountProfile, config: Dict = None):
        self.profile = account_profile
        
        # Configuration
        self.config = config or self._default_config()
        
        # Cost model parameters by account tier
        self.cost_parameters = self._initialize_cost_parameters()
        
        # Historical execution data
        self.execution_history = []
        self.cost_predictions = []
        
        # Model components
        self.commission_model = None
        self.slippage_model = None
        self.market_impact_model = None
        self.opportunity_cost_model = None
        
        # Performance tracking
        self.model_metrics = {}
        self.calibration_data = []
        
        logger.info(f"TransactionCostModel initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'model_update_frequency': 30,  # days
            'min_samples_for_calibration': 50,
            'confidence_level': 0.95,
            'cost_components': {
                'commission': True,
                'slippage': True,
                'market_impact': True,
                'opportunity_cost': True
            },
            'slippage_factors': {
                'time_of_day': True,
                'volume_profile': True,
                'volatility_regime': True,
                'order_size': True,
                'spread_width': True
            },
            'market_impact_factors': {
                'order_size': True,
                'liquidity': True,
                'volatility': True,
                'time_horizon': True
            }
        }
    
    def _initialize_cost_parameters(self) -> Dict[AccountTier, Dict]:
        """Initialize cost parameters by account tier"""
        return {
            AccountTier.MICRO: {
                'base_commission': 2.00,
                'commission_per_contract': 2.00,
                'slippage_multiplier': 1.5,
                'market_impact_factor': 0.002,
                'opportunity_cost_factor': 0.001,
                'min_spread_threshold': 0.05,
                'max_spread_threshold': 0.20
            },
            AccountTier.SMALL: {
                'base_commission': 1.75,
                'commission_per_contract': 1.75,
                'slippage_multiplier': 1.3,
                'market_impact_factor': 0.0015,
                'opportunity_cost_factor': 0.0008,
                'min_spread_threshold': 0.03,
                'max_spread_threshold': 0.15
            },
            AccountTier.MEDIUM: {
                'base_commission': 1.25,
                'commission_per_contract': 1.25,
                'slippage_multiplier': 1.0,
                'market_impact_factor': 0.001,
                'opportunity_cost_factor': 0.0005,
                'min_spread_threshold': 0.02,
                'max_spread_threshold': 0.10
            },
            AccountTier.LARGE: {
                'base_commission': 0.75,
                'commission_per_contract': 0.75,
                'slippage_multiplier': 0.8,
                'market_impact_factor': 0.0008,
                'opportunity_cost_factor': 0.0003,
                'min_spread_threshold': 0.015,
                'max_spread_threshold': 0.08
            },
            AccountTier.INSTITUTIONAL: {
                'base_commission': 0.50,
                'commission_per_contract': 0.50,
                'slippage_multiplier': 0.7,
                'market_impact_factor': 0.0005,
                'opportunity_cost_factor': 0.0002,
                'min_spread_threshold': 0.01,
                'max_spread_threshold': 0.05
            }
        }
    
    def estimate_pre_trade_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        market_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> CostEstimate:
        """
        Estimate transaction costs before trade execution
        
        Args:
            symbol: Option symbol
            side: 'buy' or 'sell'
            quantity: Number of contracts
            current_price: Current market price
            market_data: Market data dictionary
            context: Optional context information
        
        Returns:
            CostEstimate object
        """
        try:
            # Get cost parameters for account tier
            params = self.cost_parameters[self.profile.tier]
            
            # Extract market data
            bid = market_data.get('bid', current_price * 0.99)
            ask = market_data.get('ask', current_price * 1.01)
            volume = market_data.get('volume', 1000)
            spread = ask - bid
            
            # Calculate commission
            commission = self._calculate_commission(quantity, params)
            
            # Calculate slippage
            slippage = self._calculate_slippage(
                symbol, side, quantity, current_price, market_data, context, params
            )
            
            # Calculate market impact
            market_impact = self._calculate_market_impact(
                symbol, side, quantity, current_price, market_data, context, params
            )
            
            # Calculate opportunity cost
            opportunity_cost = self._calculate_opportunity_cost(
                symbol, side, quantity, current_price, market_data, context, params
            )
            
            # Total estimated cost
            total_cost = commission + slippage + market_impact + opportunity_cost
            
            # Cost per contract
            cost_per_contract = total_cost / quantity if quantity > 0 else 0
            
            # Cost as percentage of notional
            notional_value = quantity * current_price * 100  # Options are per 100 shares
            cost_as_pct = (total_cost / notional_value) * 100 if notional_value > 0 else 0
            
            # Confidence interval
            confidence_interval = self._calculate_confidence_interval(
                total_cost, symbol, quantity, context
            )
            
            return CostEstimate(
                symbol=symbol,
                side=side,
                quantity=quantity,
                estimated_commission=commission,
                estimated_slippage=slippage,
                estimated_market_impact=market_impact,
                estimated_opportunity_cost=opportunity_cost,
                total_estimated_cost=total_cost,
                cost_per_contract=cost_per_contract,
                cost_as_pct_of_notional=cost_as_pct,
                confidence_interval=confidence_interval,
                model_version=f"cost_model_v1_{self.profile.tier.value}"
            )
            
        except Exception as e:
            logger.error(f"Error estimating pre-trade cost: {e}")
            return self._empty_cost_estimate(symbol, side, quantity)
    
    def _calculate_commission(self, quantity: int, params: Dict) -> float:
        """Calculate commission costs"""
        try:
            return params['commission_per_contract'] * quantity
            
        except Exception as e:
            logger.error(f"Error calculating commission: {e}")
            return 0.0
    
    def _calculate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        market_data: Dict[str, Any],
        context: Dict[str, Any],
        params: Dict
    ) -> float:
        """Calculate slippage costs"""
        try:
            bid = market_data.get('bid', current_price * 0.99)
            ask = market_data.get('ask', current_price * 1.01)
            spread = ask - bid
            
            # Base slippage (half the spread)
            base_slippage = spread / 2
            
            # Apply account tier multiplier
            slippage_multiplier = params['slippage_multiplier']
            
            # Time of day adjustment
            time_of_day = context.get('time_of_day', 'normal') if context else 'normal'
            time_multiplier = self._get_time_of_day_multiplier(time_of_day)
            
            # Volume adjustment
            volume = market_data.get('volume', 1000)
            volume_factor = self._get_volume_factor(volume, quantity)
            
            # Volatility adjustment
            volatility = context.get('volatility', 0.2) if context else 0.2
            vol_multiplier = self._get_volatility_multiplier(volatility)
            
            # Order size adjustment
            size_multiplier = self._get_order_size_multiplier(quantity)
            
            # Calculate total slippage
            total_slippage = (
                base_slippage * 
                slippage_multiplier * 
                time_multiplier * 
                volume_factor * 
                vol_multiplier * 
                size_multiplier * 
                quantity * 100  # Convert to dollar amount
            )
            
            return total_slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0.0
    
    def _calculate_market_impact(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        market_data: Dict[str, Any],
        context: Dict[str, Any],
        params: Dict
    ) -> float:
        """Calculate market impact costs"""
        try:
            # Market impact based on order size relative to average volume
            avg_volume = market_data.get('avg_volume', 1000)
            volume_ratio = quantity / avg_volume if avg_volume > 0 else 0
            
            # Base market impact
            market_impact_factor = params['market_impact_factor']
            
            # Liquidity adjustment
            liquidity_score = self._get_liquidity_score(symbol, market_data)
            liquidity_multiplier = 1.0 / (liquidity_score + 0.1)  # Higher liquidity = lower impact
            
            # Volatility adjustment
            volatility = context.get('volatility', 0.2) if context else 0.2
            vol_impact_multiplier = 1.0 + volatility  # Higher vol = higher impact
            
            # Time horizon adjustment
            time_horizon = context.get('time_horizon', 'normal') if context else 'normal'
            time_multiplier = self._get_time_horizon_multiplier(time_horizon)
            
            # Calculate market impact
            market_impact = (
                quantity * 
                current_price * 
                100 *  # Options multiplier
                market_impact_factor * 
                volume_ratio * 
                liquidity_multiplier * 
                vol_impact_multiplier * 
                time_multiplier
            )
            
            return market_impact
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    def _calculate_opportunity_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        market_data: Dict[str, Any],
        context: Dict[str, Any],
        params: Dict
    ) -> float:
        """Calculate opportunity cost of delayed execution"""
        try:
            # Opportunity cost based on price movement risk
            opportunity_cost_factor = params['opportunity_cost_factor']
            
            # Volatility-based opportunity cost
            volatility = context.get('volatility', 0.2) if context else 0.2
            expected_volatility = volatility * current_price
            
            # Time decay for options
            days_to_expiry = context.get('days_to_expiry', 30) if context else 30
            time_decay_factor = 1.0 / (days_to_expiry + 1)  # Higher decay for shorter expiry
            
            # Market regime adjustment
            market_regime = context.get('market_regime', 'NORMAL_VOL') if context else 'NORMAL_VOL'
            regime_multiplier = self._get_regime_multiplier(market_regime)
            
            # Calculate opportunity cost
            opportunity_cost = (
                quantity * 
                current_price * 
                100 *  # Options multiplier
                opportunity_cost_factor * 
                expected_volatility * 
                time_decay_factor * 
                regime_multiplier
            )
            
            return opportunity_cost
            
        except Exception as e:
            logger.error(f"Error calculating opportunity cost: {e}")
            return 0.0
    
    def _get_time_of_day_multiplier(self, time_of_day: str) -> float:
        """Get time of day multiplier for slippage"""
        multipliers = {
            'open': 1.5,      # Higher slippage at market open
            'normal': 1.0,    # Normal slippage during regular hours
            'close': 1.3,     # Higher slippage near market close
            'after_hours': 2.0  # Much higher slippage after hours
        }
        return multipliers.get(time_of_day, 1.0)
    
    def _get_volume_factor(self, volume: int, order_size: int) -> float:
        """Get volume factor for slippage calculation"""
        try:
            # Higher volume relative to order size = lower slippage
            if volume > 0:
                volume_ratio = min(order_size / volume, 1.0)
                return max(0.5, 1.0 - volume_ratio)
            else:
                return 1.5  # High slippage for no volume
                
        except Exception as e:
            logger.error(f"Error calculating volume factor: {e}")
            return 1.0
    
    def _get_volatility_multiplier(self, volatility: float) -> float:
        """Get volatility multiplier for slippage"""
        try:
            # Higher volatility = higher slippage
            return 1.0 + (volatility - 0.2) * 2  # Base vol of 20%
            
        except Exception as e:
            logger.error(f"Error calculating volatility multiplier: {e}")
            return 1.0
    
    def _get_order_size_multiplier(self, quantity: int) -> float:
        """Get order size multiplier for slippage"""
        try:
            # Larger orders = higher slippage
            if quantity <= 10:
                return 1.0
            elif quantity <= 50:
                return 1.0 + (quantity - 10) * 0.01  # 1% increase per 10 contracts
            else:
                return 1.4 + (quantity - 50) * 0.005  # 0.5% increase per 10 contracts above 50
                
        except Exception as e:
            logger.error(f"Error calculating order size multiplier: {e}")
            return 1.0
    
    def _get_liquidity_score(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Get liquidity score for market impact calculation"""
        try:
            volume = market_data.get('volume', 1000)
            spread = market_data.get('ask', 0) - market_data.get('bid', 0)
            price = market_data.get('last', (market_data.get('bid', 0) + market_data.get('ask', 0)) / 2)
            
            # Simple liquidity score based on volume and spread
            volume_score = min(volume / 10000, 1.0)  # Normalize volume
            spread_score = max(0.1, 1.0 - (spread / price) * 100)  # Lower spread = higher score
            
            return (volume_score + spread_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    def _get_time_horizon_multiplier(self, time_horizon: str) -> float:
        """Get time horizon multiplier for market impact"""
        multipliers = {
            'immediate': 1.0,    # Immediate execution
            'short': 0.8,        # Short time horizon
            'normal': 0.6,       # Normal time horizon
            'long': 0.4,         # Long time horizon
            'patient': 0.2       # Very patient execution
        }
        return multipliers.get(time_horizon, 0.6)
    
    def _get_regime_multiplier(self, market_regime: str) -> float:
        """Get market regime multiplier for opportunity cost"""
        multipliers = {
            'LOW_VOL': 0.5,      # Lower opportunity cost in low vol
            'NORMAL_VOL': 1.0,   # Normal opportunity cost
            'HIGH_VOL': 1.5,     # Higher opportunity cost in high vol
            'CRISIS': 2.0        # Much higher opportunity cost in crisis
        }
        return multipliers.get(market_regime, 1.0)
    
    def _calculate_confidence_interval(
        self,
        total_cost: float,
        symbol: str,
        quantity: int,
        context: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for cost estimate"""
        try:
            # Base confidence interval width
            base_width = 0.2  # 20% width
            
            # Adjust based on historical accuracy
            historical_accuracy = self._get_historical_accuracy(symbol)
            accuracy_adjustment = 1.0 - historical_accuracy
            
            # Adjust based on order size
            size_adjustment = min(0.1, quantity / 100)  # Larger orders = wider interval
            
            # Adjust based on market conditions
            volatility = context.get('volatility', 0.2) if context else 0.2
            vol_adjustment = (volatility - 0.2) * 0.5  # Higher vol = wider interval
            
            # Calculate final width
            interval_width = base_width + accuracy_adjustment + size_adjustment + vol_adjustment
            interval_width = max(0.1, min(0.5, interval_width))  # Clamp between 10% and 50%
            
            # Calculate bounds
            lower_bound = total_cost * (1 - interval_width / 2)
            upper_bound = total_cost * (1 + interval_width / 2)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (total_cost * 0.9, total_cost * 1.1)
    
    def _get_historical_accuracy(self, symbol: str) -> float:
        """Get historical prediction accuracy for symbol"""
        try:
            # Filter predictions for this symbol
            symbol_predictions = [p for p in self.cost_predictions if p['symbol'] == symbol]
            
            if len(symbol_predictions) < 5:
                return 0.7  # Default accuracy for new symbols
            
            # Calculate accuracy
            total_error = 0.0
            for pred in symbol_predictions:
                estimated = pred['estimated_cost']
                actual = pred['actual_cost']
                if estimated > 0:
                    error = abs(estimated - actual) / estimated
                    total_error += error
            
            accuracy = 1.0 - (total_error / len(symbol_predictions))
            return max(0.1, min(0.95, accuracy))  # Clamp between 10% and 95%
            
        except Exception as e:
            logger.error(f"Error getting historical accuracy: {e}")
            return 0.7
    
    def analyze_execution(self, cost_estimate: CostEstimate, actual_execution: Dict[str, Any]) -> ExecutionAnalysis:
        """
        Analyze actual execution vs estimate
        
        Args:
            cost_estimate: Pre-trade cost estimate
            actual_execution: Actual execution results
        
        Returns:
            ExecutionAnalysis object
        """
        try:
            # Extract actual costs
            actual_total_cost = actual_execution.get('total_cost', 0.0)
            actual_commission = actual_execution.get('commission', 0.0)
            actual_slippage = actual_execution.get('slippage', 0.0)
            actual_market_impact = actual_execution.get('market_impact', 0.0)
            
            # Calculate cost variance
            cost_variance = actual_total_cost - cost_estimate.total_estimated_cost
            
            # Calculate execution quality score
            quality_score = self._calculate_execution_quality_score(
                cost_estimate, actual_execution
            )
            
            # Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations(
                cost_estimate, actual_execution, quality_score
            )
            
            # Store for model improvement
            self._store_execution_data(cost_estimate, actual_execution)
            
            return ExecutionAnalysis(
                order_id=actual_execution.get('order_id', ''),
                symbol=cost_estimate.symbol,
                estimated_cost=cost_estimate,
                actual_cost=actual_total_cost,
                actual_commission=actual_commission,
                actual_slippage=actual_slippage,
                actual_market_impact=actual_market_impact,
                cost_variance=cost_variance,
                execution_quality_score=quality_score,
                improvement_recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing execution: {e}")
            return self._empty_execution_analysis(cost_estimate)
    
    def _calculate_execution_quality_score(
        self,
        cost_estimate: CostEstimate,
        actual_execution: Dict[str, Any]
    ) -> float:
        """Calculate execution quality score (0-1)"""
        try:
            # Accuracy score (how close estimate was to actual)
            estimated_cost = cost_estimate.total_estimated_cost
            actual_cost = actual_execution.get('total_cost', 0.0)
            
            if estimated_cost > 0:
                accuracy = 1.0 - abs(estimated_cost - actual_cost) / estimated_cost
            else:
                accuracy = 0.5
            
            # Cost efficiency score (lower is better)
            notional = cost_estimate.quantity * actual_execution.get('avg_price', 1.0) * 100
            cost_efficiency = max(0, 1.0 - (actual_cost / notional) * 100) if notional > 0 else 0.5
            
            # Fill rate score
            fill_rate = actual_execution.get('fill_rate', 1.0)
            
            # Slippage score (lower is better)
            slippage = actual_execution.get('slippage', 0.0)
            slippage_score = max(0, 1.0 - slippage / (estimated_cost * 0.1)) if estimated_cost > 0 else 0.5
            
            # Weighted combination
            quality_score = (
                0.3 * accuracy +
                0.3 * cost_efficiency +
                0.2 * fill_rate +
                0.2 * slippage_score
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating execution quality score: {e}")
            return 0.5
    
    def _generate_improvement_recommendations(
        self,
        cost_estimate: CostEstimate,
        actual_execution: Dict[str, Any],
        quality_score: float
    ) -> List[str]:
        """Generate improvement recommendations"""
        try:
            recommendations = []
            
            # Cost accuracy recommendations
            cost_variance = actual_execution.get('total_cost', 0.0) - cost_estimate.total_estimated_cost
            if abs(cost_variance) > cost_estimate.total_estimated_cost * 0.2:  # 20% variance
                if cost_variance > 0:
                    recommendations.append(
                        "Cost estimates are consistently underestimating. Consider increasing "
                        "market impact and slippage factors."
                    )
                else:
                    recommendations.append(
                        "Cost estimates are consistently overestimating. Consider reducing "
                        "market impact and slippage factors."
                    )
            
            # Slippage recommendations
            actual_slippage = actual_execution.get('slippage', 0.0)
            estimated_slippage = cost_estimate.estimated_slippage
            
            if actual_slippage > estimated_slippage * 1.5:
                recommendations.append(
                    "Actual slippage is significantly higher than estimates. Consider "
                    "using more conservative execution algorithms or better timing."
                )
            
            # Fill rate recommendations
            fill_rate = actual_execution.get('fill_rate', 1.0)
            if fill_rate < 0.8:
                recommendations.append(
                    f"Fill rate of {fill_rate:.1%} is below target. Consider more aggressive "
                    "pricing or different execution timing."
                )
            
            # Quality score recommendations
            if quality_score < 0.6:
                recommendations.append(
                    "Overall execution quality is below target. Review execution strategy "
                    "and consider working with a different broker."
                )
            
            # Account tier specific recommendations
            if self.profile.tier in [AccountTier.MICRO, AccountTier.SMALL]:
                recommendations.append(
                    "For smaller accounts, consider end-of-day execution to minimize costs "
                    "and improve fill rates."
                )
            elif self.profile.tier in [AccountTier.LARGE, AccountTier.INSTITUTIONAL]:
                recommendations.append(
                    "For larger accounts, consider TWAP or VWAP execution to minimize "
                    "market impact."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _store_execution_data(self, cost_estimate: CostEstimate, actual_execution: Dict[str, Any]):
        """Store execution data for model improvement"""
        try:
            execution_data = {
                'timestamp': datetime.now(),
                'symbol': cost_estimate.symbol,
                'estimated_cost': cost_estimate.total_estimated_cost,
                'actual_cost': actual_execution.get('total_cost', 0.0),
                'estimated_commission': cost_estimate.estimated_commission,
                'actual_commission': actual_execution.get('commission', 0.0),
                'estimated_slippage': cost_estimate.estimated_slippage,
                'actual_slippage': actual_execution.get('slippage', 0.0),
                'quantity': cost_estimate.quantity,
                'side': cost_estimate.side,
                'account_tier': self.profile.tier.value
            }
            
            self.execution_history.append(execution_data)
            
            # Store prediction for accuracy tracking
            self.cost_predictions.append({
                'symbol': cost_estimate.symbol,
                'estimated_cost': cost_estimate.total_estimated_cost,
                'actual_cost': actual_execution.get('total_cost', 0.0),
                'timestamp': datetime.now()
            })
            
            # Keep only recent data
            max_history = 1000
            if len(self.execution_history) > max_history:
                self.execution_history = self.execution_history[-max_history:]
            
            if len(self.cost_predictions) > max_history:
                self.cost_predictions = self.cost_predictions[-max_history:]
            
        except Exception as e:
            logger.error(f"Error storing execution data: {e}")
    
    def update_model(self) -> bool:
        """Update cost model based on historical data"""
        try:
            if len(self.execution_history) < self.config['min_samples_for_calibration']:
                logger.warning(f"Insufficient data for model update: {len(self.execution_history)} < {self.config['min_samples_for_calibration']}")
                return False
            
            # Analyze prediction accuracy
            self._analyze_prediction_accuracy()
            
            # Update cost parameters if needed
            self._update_cost_parameters()
            
            # Recalculate model metrics
            self._calculate_model_metrics()
            
            logger.info("Cost model updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
    
    def _analyze_prediction_accuracy(self):
        """Analyze prediction accuracy by component"""
        try:
            if not self.execution_history:
                return
            
            # Calculate accuracy metrics
            total_error = 0.0
            commission_error = 0.0
            slippage_error = 0.0
            
            for execution in self.execution_history:
                # Total cost accuracy
                estimated = execution['estimated_cost']
                actual = execution['actual_cost']
                if estimated > 0:
                    total_error += abs(estimated - actual) / estimated
                
                # Commission accuracy
                est_comm = execution['estimated_commission']
                act_comm = execution['actual_commission']
                if est_comm > 0:
                    commission_error += abs(est_comm - act_comm) / est_comm
                
                # Slippage accuracy
                est_slip = execution['estimated_slippage']
                act_slip = execution['actual_slippage']
                if est_slip > 0:
                    slippage_error += abs(est_slip - act_slip) / est_slip
            
            # Store metrics
            n_executions = len(self.execution_history)
            self.model_metrics = {
                'total_cost_accuracy': 1.0 - (total_error / n_executions),
                'commission_accuracy': 1.0 - (commission_error / n_executions),
                'slippage_accuracy': 1.0 - (slippage_error / n_executions),
                'sample_size': n_executions,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prediction accuracy: {e}")
    
    def _update_cost_parameters(self):
        """Update cost parameters based on historical accuracy"""
        try:
            if not self.model_metrics:
                return
            
            # Get current parameters
            current_params = self.cost_parameters[self.profile.tier]
            
            # Adjust slippage multiplier based on accuracy
            slippage_accuracy = self.model_metrics.get('slippage_accuracy', 0.8)
            if slippage_accuracy < 0.7:  # Low accuracy
                adjustment_factor = 1.1  # Increase multiplier
            elif slippage_accuracy > 0.9:  # High accuracy
                adjustment_factor = 0.95  # Slight decrease
            else:
                adjustment_factor = 1.0  # No change
            
            # Apply adjustment
            new_slippage_multiplier = current_params['slippage_multiplier'] * adjustment_factor
            
            # Update parameters
            self.cost_parameters[self.profile.tier]['slippage_multiplier'] = new_slippage_multiplier
            
            logger.info(f"Updated slippage multiplier: {current_params['slippage_multiplier']:.3f} -> {new_slippage_multiplier:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating cost parameters: {e}")
    
    def _calculate_model_metrics(self) -> CostModelMetrics:
        """Calculate comprehensive model metrics"""
        try:
            if not self.execution_history:
                return self._empty_metrics()
            
            # Calculate prediction errors
            errors = []
            over_estimates = 0
            under_estimates = 0
            
            for execution in self.execution_history:
                estimated = execution['estimated_cost']
                actual = execution['actual_cost']
                
                if estimated > 0:
                    error = actual - estimated
                    errors.append(error)
                    
                    if error > 0:
                        over_estimates += 1
                    else:
                        under_estimates += 1
            
            if not errors:
                return self._empty_metrics()
            
            # Calculate metrics
            mae = np.mean([abs(e) for e in errors])
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            
            total_predictions = len(errors)
            over_estimate_rate = over_estimates / total_predictions
            under_estimate_rate = under_estimates / total_predictions
            
            # Prediction accuracy
            prediction_accuracy = self.model_metrics.get('total_cost_accuracy', 0.8)
            
            # Confidence calibration (simplified)
            confidence_calibration = 0.8  # Placeholder
            
            return CostModelMetrics(
                prediction_accuracy=prediction_accuracy,
                mae=mae,
                rmse=rmse,
                over_estimate_rate=over_estimate_rate,
                under_estimate_rate=under_estimate_rate,
                confidence_calibration=confidence_calibration
            )
            
        except Exception as e:
            logger.error(f"Error calculating model metrics: {e}")
            return self._empty_metrics()
    
    def _empty_cost_estimate(self, symbol: str, side: str, quantity: int) -> CostEstimate:
        """Return empty cost estimate"""
        return CostEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            estimated_commission=0.0,
            estimated_slippage=0.0,
            estimated_market_impact=0.0,
            estimated_opportunity_cost=0.0,
            total_estimated_cost=0.0,
            cost_per_contract=0.0,
            cost_as_pct_of_notional=0.0,
            confidence_interval=(0.0, 0.0),
            model_version="empty"
        )
    
    def _empty_execution_analysis(self, cost_estimate: CostEstimate) -> ExecutionAnalysis:
        """Return empty execution analysis"""
        return ExecutionAnalysis(
            order_id="",
            symbol=cost_estimate.symbol,
            estimated_cost=cost_estimate,
            actual_cost=0.0,
            actual_commission=0.0,
            actual_slippage=0.0,
            actual_market_impact=0.0,
            cost_variance=0.0,
            execution_quality_score=0.0,
            improvement_recommendations=[]
        )
    
    def _empty_metrics(self) -> CostModelMetrics:
        """Return empty metrics"""
        return CostModelMetrics(0, 0, 0, 0, 0, 0)
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_metrics.copy()
    
    def get_execution_history(self, limit: int = 100) -> List[Dict]:
        """Get execution history"""
        return self.execution_history[-limit:] if self.execution_history else []


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create cost model
    cost_model = TransactionCostModel(profile)
    
    # Test cost estimation
    symbol = 'SPY240315C00500000'
    side = 'buy'
    quantity = 10
    current_price = 1.50
    
    market_data = {
        'bid': 1.45,
        'ask': 1.55,
        'last': 1.50,
        'volume': 1500,
        'avg_volume': 2000
    }
    
    context = {
        'time_of_day': 'normal',
        'volatility': 0.25,
        'market_regime': 'NORMAL_VOL',
        'days_to_expiry': 30,
        'time_horizon': 'normal'
    }
    
    print("Testing Transaction Cost Model...")
    print(f"Account Tier: {profile.tier.value}")
    
    # Estimate costs
    cost_estimate = cost_model.estimate_pre_trade_cost(
        symbol=symbol,
        side=side,
        quantity=quantity,
        current_price=current_price,
        market_data=market_data,
        context=context
    )
    
    print(f"\nCost Estimate for {symbol}:")
    print(f"Commission: ${cost_estimate.estimated_commission:.2f}")
    print(f"Slippage: ${cost_estimate.estimated_slippage:.2f}")
    print(f"Market Impact: ${cost_estimate.estimated_market_impact:.2f}")
    print(f"Opportunity Cost: ${cost_estimate.estimated_opportunity_cost:.2f}")
    print(f"Total Cost: ${cost_estimate.total_estimated_cost:.2f}")
    print(f"Cost per Contract: ${cost_estimate.cost_per_contract:.2f}")
    print(f"Cost as % of Notional: {cost_estimate.cost_as_pct_of_notional:.3f}%")
    print(f"Confidence Interval: ${cost_estimate.confidence_interval[0]:.2f} - ${cost_estimate.confidence_interval[1]:.2f}")
    print(f"Model Version: {cost_estimate.model_version}")
    
    # Simulate actual execution
    actual_execution = {
        'order_id': 'test_order_123',
        'total_cost': cost_estimate.total_estimated_cost * 1.1,  # 10% higher than estimate
        'commission': cost_estimate.estimated_commission,
        'slippage': cost_estimate.estimated_slippage * 1.2,  # 20% higher slippage
        'market_impact': cost_estimate.estimated_market_impact * 0.8,  # 20% lower market impact
        'avg_price': current_price,
        'fill_rate': 1.0
    }
    
    # Analyze execution
    analysis = cost_model.analyze_execution(cost_estimate, actual_execution)
    
    print(f"\nExecution Analysis:")
    print(f"Cost Variance: ${analysis.cost_variance:.2f}")
    print(f"Execution Quality Score: {analysis.execution_quality_score:.3f}")
    print(f"Recommendations:")
    for i, rec in enumerate(analysis.improvement_recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Get model metrics
    metrics = cost_model.get_model_metrics()
    print(f"\nModel Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
