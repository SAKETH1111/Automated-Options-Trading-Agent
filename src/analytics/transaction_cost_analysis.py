"""
Transaction Cost Analysis (TCA)
Execution quality measurement and optimization with account-size-specific benchmarks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from src.portfolio.account_manager import AccountProfile, AccountTier


@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    order_id: str
    symbol: str
    strategy: str
    side: str
    quantity: int
    estimated_cost: float
    actual_cost: float
    slippage: float
    commission: float
    market_impact: float
    opportunity_cost: float
    fill_rate: float
    time_to_fill: float  # seconds
    execution_quality_score: float


@dataclass
class TCAReport:
    """Comprehensive TCA report"""
    analysis_period: Dict[str, Any]
    cost_breakdown: Dict[str, float]
    execution_quality: Dict[str, float]
    benchmark_comparison: Dict[str, float]
    recommendations: List[str]
    detailed_metrics: List[ExecutionMetrics]


class TransactionCostAnalysis:
    """
    Advanced transaction cost analysis for options trading
    
    Features:
    - Pre-trade cost estimate vs actual
    - Slippage by time of day and symbol
    - Fill rate and partial fill analysis
    - Market impact by order size
    - Opportunity cost (missed fills)
    - Account-size-specific benchmarks
    """
    
    def __init__(self, account_profile: AccountProfile):
        self.profile = account_profile
        
        # TCA parameters
        self.analysis_window = 30  # days
        self.min_orders_for_analysis = 10
        
        # Cost model parameters by account tier
        self.cost_models = self._initialize_cost_models()
        
        # Benchmark data
        self.benchmarks = self._initialize_benchmarks()
        
        logger.info(f"TransactionCostAnalysis initialized for {account_profile.tier.value} tier")
    
    def _initialize_cost_models(self) -> Dict[AccountTier, Dict]:
        """Initialize cost models by account tier"""
        return {
            AccountTier.MICRO: {
                'base_commission': 2.00,
                'commission_per_contract': 2.00,
                'slippage_multiplier': 1.5,
                'market_impact_factor': 0.002,
                'opportunity_cost_factor': 0.001
            },
            AccountTier.SMALL: {
                'base_commission': 1.75,
                'commission_per_contract': 1.75,
                'slippage_multiplier': 1.3,
                'market_impact_factor': 0.0015,
                'opportunity_cost_factor': 0.0008
            },
            AccountTier.MEDIUM: {
                'base_commission': 1.25,
                'commission_per_contract': 1.25,
                'slippage_multiplier': 1.0,
                'market_impact_factor': 0.001,
                'opportunity_cost_factor': 0.0005
            },
            AccountTier.LARGE: {
                'base_commission': 0.75,
                'commission_per_contract': 0.75,
                'slippage_multiplier': 0.8,
                'market_impact_factor': 0.0008,
                'opportunity_cost_factor': 0.0003
            },
            AccountTier.INSTITUTIONAL: {
                'base_commission': 0.50,
                'commission_per_contract': 0.50,
                'slippage_multiplier': 0.7,
                'market_impact_factor': 0.0005,
                'opportunity_cost_factor': 0.0002
            }
        }
    
    def _initialize_benchmarks(self) -> Dict[AccountTier, Dict]:
        """Initialize performance benchmarks by account tier"""
        return {
            AccountTier.MICRO: {
                'target_slippage_pct': 2.0,
                'target_fill_rate': 0.85,
                'target_time_to_fill': 300,  # 5 minutes
                'target_execution_score': 0.70
            },
            AccountTier.SMALL: {
                'target_slippage_pct': 1.5,
                'target_fill_rate': 0.90,
                'target_time_to_fill': 240,  # 4 minutes
                'target_execution_score': 0.75
            },
            AccountTier.MEDIUM: {
                'target_slippage_pct': 1.2,
                'target_fill_rate': 0.92,
                'target_time_to_fill': 180,  # 3 minutes
                'target_execution_score': 0.80
            },
            AccountTier.LARGE: {
                'target_slippage_pct': 1.0,
                'target_fill_rate': 0.94,
                'target_time_to_fill': 120,  # 2 minutes
                'target_execution_score': 0.85
            },
            AccountTier.INSTITUTIONAL: {
                'target_slippage_pct': 0.8,
                'target_fill_rate': 0.96,
                'target_time_to_fill': 60,   # 1 minute
                'target_execution_score': 0.90
            }
        }
    
    def estimate_pre_trade_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        bid: float,
        ask: float,
        volume: int = 1000,
        time_of_day: str = "normal"
    ) -> Dict[str, float]:
        """
        Estimate transaction costs before trade execution
        
        Args:
            symbol: Option symbol
            side: 'buy' or 'sell'
            quantity: Number of contracts
            current_price: Current market price
            bid: Bid price
            ask: Ask price
            volume: Current volume
            time_of_day: 'open', 'normal', 'close'
        
        Returns:
            Dictionary with estimated costs
        """
        try:
            cost_model = self.cost_models.get(self.profile.tier, self.cost_models[AccountTier.MEDIUM])
            
            # Commission
            commission = cost_model['commission_per_contract'] * quantity
            
            # Spread cost (half the bid-ask spread)
            spread = ask - bid
            mid_price = (bid + ask) / 2
            spread_cost = (spread / 2) * quantity * 100
            
            # Slippage estimation
            slippage_multiplier = cost_model['slippage_multiplier']
            base_slippage = spread * 0.1  # 10% of spread as base slippage
            
            # Adjust for time of day
            time_multiplier = self._get_time_of_day_multiplier(time_of_day)
            
            # Adjust for volume (lower volume = higher slippage)
            volume_factor = max(0.5, 1.0 - (volume / 10000))  # More volume = lower slippage
            
            estimated_slippage = base_slippage * slippage_multiplier * time_multiplier * volume_factor * quantity * 100
            
            # Market impact
            market_impact = cost_model['market_impact_factor'] * quantity * mid_price * 100
            
            # Opportunity cost (simplified)
            opportunity_cost = cost_model['opportunity_cost_factor'] * quantity * mid_price * 100
            
            # Total estimated cost
            total_estimated_cost = commission + spread_cost + estimated_slippage + market_impact + opportunity_cost
            
            return {
                'commission': commission,
                'spread_cost': spread_cost,
                'estimated_slippage': estimated_slippage,
                'market_impact': market_impact,
                'opportunity_cost': opportunity_cost,
                'total_estimated_cost': total_estimated_cost,
                'cost_per_contract': total_estimated_cost / quantity if quantity > 0 else 0,
                'cost_as_pct_of_notional': (total_estimated_cost / (quantity * mid_price * 100)) * 100 if quantity > 0 and mid_price > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error estimating pre-trade cost: {e}")
            return {}
    
    def _get_time_of_day_multiplier(self, time_of_day: str) -> float:
        """Get slippage multiplier based on time of day"""
        multipliers = {
            'open': 1.5,      # Higher slippage at market open
            'normal': 1.0,    # Normal slippage during regular hours
            'close': 1.3,     # Higher slippage near market close
            'after_hours': 2.0  # Much higher slippage after hours
        }
        return multipliers.get(time_of_day, 1.0)
    
    def analyze_execution_quality(
        self,
        orders: List[Dict],
        start_date: datetime = None,
        end_date: datetime = None
    ) -> TCAReport:
        """
        Analyze execution quality across all orders
        
        Args:
            orders: List of order dictionaries with execution data
            start_date: Analysis start date
            end_date: Analysis end date
        
        Returns:
            TCAReport with comprehensive execution analysis
        """
        try:
            if not orders:
                return self._empty_tca_report()
            
            # Filter orders by date range
            if start_date or end_date:
                orders = self._filter_orders_by_date(orders, start_date, end_date)
            
            if len(orders) < self.min_orders_for_analysis:
                logger.warning(f"Insufficient orders for analysis: {len(orders)} < {self.min_orders_for_analysis}")
                return self._empty_tca_report()
            
            # Convert to DataFrame for analysis
            orders_df = pd.DataFrame(orders)
            
            # Calculate execution metrics for each order
            execution_metrics = []
            for _, order in orders_df.iterrows():
                metrics = self._calculate_order_metrics(order)
                if metrics:
                    execution_metrics.append(metrics)
            
            if not execution_metrics:
                return self._empty_tca_report()
            
            # Calculate aggregate metrics
            cost_breakdown = self._calculate_cost_breakdown(execution_metrics)
            execution_quality = self._calculate_execution_quality_metrics(execution_metrics)
            benchmark_comparison = self._compare_to_benchmarks(execution_quality)
            recommendations = self._generate_recommendations(execution_quality, benchmark_comparison)
            
            # Analysis period
            analysis_period = {
                'start_date': orders_df['timestamp'].min() if 'timestamp' in orders_df.columns else None,
                'end_date': orders_df['timestamp'].max() if 'timestamp' in orders_df.columns else None,
                'total_orders': len(orders_df),
                'total_contracts': orders_df['quantity'].sum() if 'quantity' in orders_df.columns else 0
            }
            
            return TCAReport(
                analysis_period=analysis_period,
                cost_breakdown=cost_breakdown,
                execution_quality=execution_quality,
                benchmark_comparison=benchmark_comparison,
                recommendations=recommendations,
                detailed_metrics=execution_metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing execution quality: {e}")
            return self._empty_tca_report()
    
    def _filter_orders_by_date(self, orders: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Filter orders by date range"""
        try:
            filtered_orders = []
            
            for order in orders:
                order_date = order.get('timestamp')
                if isinstance(order_date, str):
                    order_date = datetime.fromisoformat(order_date.replace('Z', '+00:00'))
                
                if start_date and order_date < start_date:
                    continue
                if end_date and order_date > end_date:
                    continue
                
                filtered_orders.append(order)
            
            return filtered_orders
            
        except Exception as e:
            logger.error(f"Error filtering orders by date: {e}")
            return orders
    
    def _calculate_order_metrics(self, order: pd.Series) -> Optional[ExecutionMetrics]:
        """Calculate execution metrics for individual order"""
        try:
            # Extract order data
            order_id = order.get('order_id', 'unknown')
            symbol = order.get('symbol', '')
            strategy = order.get('strategy', '')
            side = order.get('side', '')
            quantity = order.get('quantity', 0)
            
            # Execution data
            estimated_cost = order.get('estimated_cost', 0)
            actual_cost = order.get('actual_cost', 0)
            commission = order.get('commission', 0)
            slippage = order.get('slippage', 0)
            fill_rate = order.get('fill_rate', 1.0)
            time_to_fill = order.get('time_to_fill', 0)
            
            # Calculate market impact
            market_impact = actual_cost - commission - slippage
            
            # Calculate opportunity cost (simplified)
            opportunity_cost = estimated_cost - actual_cost if actual_cost < estimated_cost else 0
            
            # Calculate execution quality score
            execution_quality_score = self._calculate_execution_quality_score(
                slippage, fill_rate, time_to_fill, quantity
            )
            
            return ExecutionMetrics(
                order_id=order_id,
                symbol=symbol,
                strategy=strategy,
                side=side,
                quantity=quantity,
                estimated_cost=estimated_cost,
                actual_cost=actual_cost,
                slippage=slippage,
                commission=commission,
                market_impact=market_impact,
                opportunity_cost=opportunity_cost,
                fill_rate=fill_rate,
                time_to_fill=time_to_fill,
                execution_quality_score=execution_quality_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating order metrics: {e}")
            return None
    
    def _calculate_execution_quality_score(
        self,
        slippage: float,
        fill_rate: float,
        time_to_fill: float,
        quantity: int
    ) -> float:
        """Calculate overall execution quality score (0-1)"""
        try:
            # Get benchmarks for account tier
            benchmark = self.benchmarks.get(self.profile.tier, self.benchmarks[AccountTier.MEDIUM])
            
            # Slippage score (lower is better)
            slippage_score = max(0, 1 - (slippage / (benchmark['target_slippage_pct'] / 100)))
            
            # Fill rate score (higher is better)
            fill_rate_score = fill_rate / benchmark['target_fill_rate']
            
            # Time to fill score (lower is better)
            time_score = max(0, 1 - (time_to_fill / benchmark['target_time_to_fill']))
            
            # Weighted combination
            execution_score = (
                0.3 * slippage_score +
                0.4 * fill_rate_score +
                0.3 * time_score
            )
            
            return min(1.0, max(0.0, execution_score))
            
        except Exception as e:
            logger.error(f"Error calculating execution quality score: {e}")
            return 0.5
    
    def _calculate_cost_breakdown(self, execution_metrics: List[ExecutionMetrics]) -> Dict[str, float]:
        """Calculate cost breakdown across all orders"""
        try:
            if not execution_metrics:
                return {}
            
            total_commission = sum(metric.commission for metric in execution_metrics)
            total_slippage = sum(metric.slippage for metric in execution_metrics)
            total_market_impact = sum(metric.market_impact for metric in execution_metrics)
            total_opportunity_cost = sum(metric.opportunity_cost for metric in execution_metrics)
            total_actual_cost = sum(metric.actual_cost for metric in execution_metrics)
            total_estimated_cost = sum(metric.estimated_cost for metric in execution_metrics)
            
            total_contracts = sum(metric.quantity for metric in execution_metrics)
            
            return {
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_market_impact': total_market_impact,
                'total_opportunity_cost': total_opportunity_cost,
                'total_actual_cost': total_actual_cost,
                'total_estimated_cost': total_estimated_cost,
                'cost_vs_estimate': total_actual_cost - total_estimated_cost,
                'avg_commission_per_contract': total_commission / total_contracts if total_contracts > 0 else 0,
                'avg_slippage_per_contract': total_slippage / total_contracts if total_contracts > 0 else 0,
                'avg_total_cost_per_contract': total_actual_cost / total_contracts if total_contracts > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost breakdown: {e}")
            return {}
    
    def _calculate_execution_quality_metrics(self, execution_metrics: List[ExecutionMetrics]) -> Dict[str, float]:
        """Calculate aggregate execution quality metrics"""
        try:
            if not execution_metrics:
                return {}
            
            # Average metrics
            avg_slippage = np.mean([metric.slippage for metric in execution_metrics])
            avg_fill_rate = np.mean([metric.fill_rate for metric in execution_metrics])
            avg_time_to_fill = np.mean([metric.time_to_fill for metric in execution_metrics])
            avg_execution_score = np.mean([metric.execution_quality_score for metric in execution_metrics])
            
            # Slippage by side
            buy_orders = [m for m in execution_metrics if m.side == 'buy']
            sell_orders = [m for m in execution_metrics if m.side == 'sell']
            
            buy_slippage = np.mean([m.slippage for m in buy_orders]) if buy_orders else 0
            sell_slippage = np.mean([m.slippage for m in sell_orders]) if sell_orders else 0
            
            # Slippage by symbol
            symbol_slippage = {}
            for symbol in set(m.symbol for m in execution_metrics):
                symbol_orders = [m for m in execution_metrics if m.symbol == symbol]
                if symbol_orders:
                    symbol_slippage[symbol] = np.mean([m.slippage for m in symbol_orders])
            
            # Time of day analysis (simplified)
            # This would require timestamp data to be more accurate
            
            return {
                'avg_slippage': avg_slippage,
                'avg_fill_rate': avg_fill_rate,
                'avg_time_to_fill': avg_time_to_fill,
                'avg_execution_score': avg_execution_score,
                'buy_slippage': buy_slippage,
                'sell_slippage': sell_slippage,
                'symbol_slippage': symbol_slippage,
                'total_orders': len(execution_metrics),
                'successful_fills': sum(1 for m in execution_metrics if m.fill_rate >= 0.8),
                'partial_fills': sum(1 for m in execution_metrics if 0 < m.fill_rate < 0.8),
                'failed_fills': sum(1 for m in execution_metrics if m.fill_rate == 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating execution quality metrics: {e}")
            return {}
    
    def _compare_to_benchmarks(self, execution_quality: Dict[str, float]) -> Dict[str, float]:
        """Compare execution quality to benchmarks"""
        try:
            benchmark = self.benchmarks.get(self.profile.tier, self.benchmarks[AccountTier.MEDIUM])
            
            avg_slippage = execution_quality.get('avg_slippage', 0)
            avg_fill_rate = execution_quality.get('avg_fill_rate', 0)
            avg_time_to_fill = execution_quality.get('avg_time_to_fill', 0)
            avg_execution_score = execution_quality.get('avg_execution_score', 0)
            
            return {
                'slippage_vs_benchmark': avg_slippage - benchmark['target_slippage_pct'] / 100,
                'fill_rate_vs_benchmark': avg_fill_rate - benchmark['target_fill_rate'],
                'time_to_fill_vs_benchmark': avg_time_to_fill - benchmark['target_time_to_fill'],
                'execution_score_vs_benchmark': avg_execution_score - benchmark['target_execution_score'],
                'overall_performance': 'ABOVE' if avg_execution_score >= benchmark['target_execution_score'] else 'BELOW'
            }
            
        except Exception as e:
            logger.error(f"Error comparing to benchmarks: {e}")
            return {}
    
    def _generate_recommendations(
        self,
        execution_quality: Dict[str, float],
        benchmark_comparison: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for improving execution quality"""
        try:
            recommendations = []
            
            # Slippage recommendations
            slippage_vs_benchmark = benchmark_comparison.get('slippage_vs_benchmark', 0)
            if slippage_vs_benchmark > 0:
                recommendations.append(
                    f"Slippage is {slippage_vs_benchmark:.2%} above benchmark. "
                    "Consider using limit orders closer to mid-price or trading during higher volume periods."
                )
            
            # Fill rate recommendations
            fill_rate_vs_benchmark = benchmark_comparison.get('fill_rate_vs_benchmark', 0)
            if fill_rate_vs_benchmark < 0:
                recommendations.append(
                    f"Fill rate is {abs(fill_rate_vs_benchmark):.1%} below benchmark. "
                    "Consider adjusting order prices or using different execution strategies."
                )
            
            # Time to fill recommendations
            time_vs_benchmark = benchmark_comparison.get('time_to_fill_vs_benchmark', 0)
            if time_vs_benchmark > 0:
                recommendations.append(
                    f"Time to fill is {time_vs_benchmark:.0f} seconds above benchmark. "
                    "Consider using more aggressive pricing or trading during peak liquidity hours."
                )
            
            # Account tier specific recommendations
            if self.profile.tier in [AccountTier.MICRO, AccountTier.SMALL]:
                recommendations.append(
                    "For smaller accounts, consider end-of-day execution to minimize costs and improve fill rates."
                )
            elif self.profile.tier in [AccountTier.LARGE, AccountTier.INSTITUTIONAL]:
                recommendations.append(
                    "For larger accounts, consider TWAP execution and break large orders into smaller chunks."
                )
            
            # Execution score recommendations
            execution_score = execution_quality.get('avg_execution_score', 0)
            if execution_score < 0.7:
                recommendations.append(
                    "Overall execution quality is below target. Review execution strategy and consider "
                    "working with a different broker or adjusting order management approach."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _empty_tca_report(self) -> TCAReport:
        """Return empty TCA report"""
        return TCAReport(
            analysis_period={'start_date': None, 'end_date': None, 'total_orders': 0, 'total_contracts': 0},
            cost_breakdown={},
            execution_quality={},
            benchmark_comparison={},
            recommendations=[],
            detailed_metrics=[]
        )
    
    def generate_tca_report(self, tca_report: TCAReport) -> str:
        """Generate human-readable TCA report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("TRANSACTION COST ANALYSIS (TCA) REPORT")
            report.append("=" * 60)
            
            # Analysis period
            period = tca_report.analysis_period
            report.append(f"\nAnalysis Period: {period['start_date']} to {period['end_date']}")
            report.append(f"Account Tier: {self.profile.tier.value}")
            report.append(f"Total Orders: {period['total_orders']}")
            report.append(f"Total Contracts: {period['total_contracts']:,}")
            
            # Cost breakdown
            costs = tca_report.cost_breakdown
            if costs:
                report.append(f"\n{'COST BREAKDOWN':<40}")
                report.append("-" * 40)
                report.append(f"Total Commission:     ${costs.get('total_commission', 0):>10,.2f}")
                report.append(f"Total Slippage:       ${costs.get('total_slippage', 0):>10,.2f}")
                report.append(f"Market Impact:        ${costs.get('total_market_impact', 0):>10,.2f}")
                report.append(f"Opportunity Cost:     ${costs.get('total_opportunity_cost', 0):>10,.2f}")
                report.append(f"Total Actual Cost:    ${costs.get('total_actual_cost', 0):>10,.2f}")
                report.append(f"Total Estimated Cost: ${costs.get('total_estimated_cost', 0):>10,.2f}")
                report.append(f"Cost vs Estimate:     ${costs.get('cost_vs_estimate', 0):>10,.2f}")
                report.append(f"Avg Cost/Contract:    ${costs.get('avg_total_cost_per_contract', 0):>10,.2f}")
            
            # Execution quality
            quality = tca_report.execution_quality
            if quality:
                report.append(f"\n{'EXECUTION QUALITY':<40}")
                report.append("-" * 40)
                report.append(f"Avg Slippage:         ${quality.get('avg_slippage', 0):>10,.2f}")
                report.append(f"Avg Fill Rate:        {quality.get('avg_fill_rate', 0):>10.1%}")
                report.append(f"Avg Time to Fill:     {quality.get('avg_time_to_fill', 0):>10.0f}s")
                report.append(f"Avg Execution Score:  {quality.get('avg_execution_score', 0):>10.2f}")
                report.append(f"Buy Slippage:         ${quality.get('buy_slippage', 0):>10,.2f}")
                report.append(f"Sell Slippage:        ${quality.get('sell_slippage', 0):>10,.2f}")
                report.append(f"Successful Fills:     {quality.get('successful_fills', 0):>10}")
                report.append(f"Partial Fills:        {quality.get('partial_fills', 0):>10}")
                report.append(f"Failed Fills:         {quality.get('failed_fills', 0):>10}")
            
            # Benchmark comparison
            benchmark = tca_report.benchmark_comparison
            if benchmark:
                report.append(f"\n{'BENCHMARK COMPARISON':<40}")
                report.append("-" * 40)
                report.append(f"Slippage vs Benchmark: {benchmark.get('slippage_vs_benchmark', 0):>10.2%}")
                report.append(f"Fill Rate vs Benchmark: {benchmark.get('fill_rate_vs_benchmark', 0):>10.1%}")
                report.append(f"Time vs Benchmark:     {benchmark.get('time_to_fill_vs_benchmark', 0):>10.0f}s")
                report.append(f"Score vs Benchmark:    {benchmark.get('execution_score_vs_benchmark', 0):>10.2f}")
                report.append(f"Overall Performance:   {benchmark.get('overall_performance', 'UNKNOWN'):>10}")
            
            # Recommendations
            recommendations = tca_report.recommendations
            if recommendations:
                report.append(f"\n{'RECOMMENDATIONS':<40}")
                report.append("-" * 40)
                for i, rec in enumerate(recommendations, 1):
                    report.append(f"{i}. {rec}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating TCA report: {e}")
            return "Error generating TCA report"


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create TCA analyzer
    tca = TransactionCostAnalysis(profile)
    
    # Estimate pre-trade cost
    cost_estimate = tca.estimate_pre_trade_cost(
        symbol='SPY240315C00500000',
        side='buy',
        quantity=10,
        current_price=1.50,
        bid=1.45,
        ask=1.55,
        volume=1500,
        time_of_day='normal'
    )
    
    print("Pre-Trade Cost Estimate:")
    for key, value in cost_estimate.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:,.2f}" if 'cost' in key else f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Sample orders data
    sample_orders = [
        {
            'order_id': 'order_1',
            'timestamp': '2023-01-15T10:30:00Z',
            'symbol': 'SPY240315C00500000',
            'strategy': 'bull_put_spread',
            'side': 'sell',
            'quantity': 5,
            'estimated_cost': 25.00,
            'actual_cost': 27.50,
            'commission': 12.50,
            'slippage': 10.00,
            'fill_rate': 1.0,
            'time_to_fill': 180
        },
        {
            'order_id': 'order_2',
            'timestamp': '2023-01-16T14:15:00Z',
            'symbol': 'QQQ240315P00450000',
            'strategy': 'iron_condor',
            'side': 'buy',
            'quantity': 3,
            'estimated_cost': 15.00,
            'actual_cost': 14.25,
            'commission': 7.50,
            'slippage': 5.00,
            'fill_rate': 1.0,
            'time_to_fill': 120
        }
    ]
    
    # Analyze execution quality
    tca_report = tca.analyze_execution_quality(sample_orders)
    
    # Generate report
    report = tca.generate_tca_report(tca_report)
    print(f"\n{report}")
