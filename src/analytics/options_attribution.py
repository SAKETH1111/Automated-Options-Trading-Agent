"""
Options Performance Attribution
P&L decomposition by Greeks, strategies, and risk factors with Brinson attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from src.portfolio.account_manager import AccountProfile


@dataclass
class GreeksAttribution:
    """Greeks-based P&L attribution"""
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    total_greeks_pnl: float
    unexplained_pnl: float


@dataclass
class StrategyAttribution:
    """Strategy-level performance attribution"""
    strategy_name: str
    total_pnl: float
    pnl_pct: float
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    sharpe_ratio: float
    contribution_to_portfolio: float


@dataclass
class BrinsonAttribution:
    """Brinson-style attribution for multi-strategy portfolios"""
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_attribution: float


@dataclass
class RegimeAttribution:
    """Performance attribution by market regime"""
    regime_name: str
    regime_duration: int
    regime_pnl: float
    regime_return_pct: float
    best_strategy: str
    worst_strategy: str
    regime_characteristics: Dict[str, float]


class OptionsPerformanceAttribution:
    """
    Advanced performance attribution for options trading
    
    Features:
    - Greeks attribution (Delta P&L + Gamma P&L + Theta P&L + Vega P&L)
    - Strategy-level performance breakdown
    - Best/worst performers by regime
    - Strategy correlation matrix
    - Brinson attribution for multi-strategy portfolios
    """
    
    def __init__(self, account_profile: AccountProfile):
        self.profile = account_profile
        
        # Attribution parameters
        self.risk_free_rate = 0.045  # 4.5% annual
        
        # Performance tracking
        self.trades_data = []
        self.portfolio_snapshots = []
        self.regime_periods = []
        
        logger.info(f"OptionsPerformanceAttribution initialized for {account_profile.tier.value} tier")
    
    def analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive trade analysis and attribution
        
        Args:
            trades: List of trade dictionaries with performance data
        
        Returns:
            Dictionary with comprehensive attribution analysis
        """
        try:
            if not trades:
                return self._empty_analysis()
            
            # Store trades data
            self.trades_data = trades
            
            # Convert to DataFrame for analysis
            trades_df = pd.DataFrame(trades)
            
            # Greeks attribution
            greeks_attribution = self._calculate_greeks_attribution(trades_df)
            
            # Strategy attribution
            strategy_attribution = self._calculate_strategy_attribution(trades_df)
            
            # Brinson attribution
            brinson_attribution = self._calculate_brinson_attribution(trades_df)
            
            # Regime attribution
            regime_attribution = self._calculate_regime_attribution(trades_df)
            
            # Correlation analysis
            correlation_matrix = self._calculate_strategy_correlations(trades_df)
            
            # Risk-adjusted metrics
            risk_metrics = self._calculate_risk_adjusted_metrics(trades_df)
            
            # Best/worst analysis
            best_worst = self._analyze_best_worst_performers(trades_df)
            
            return {
                'analysis_period': {
                    'start_date': trades_df['entry_date'].min() if 'entry_date' in trades_df.columns else None,
                    'end_date': trades_df['exit_date'].max() if 'exit_date' in trades_df.columns else None,
                    'total_trades': len(trades_df),
                    'total_pnl': trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
                },
                'greeks_attribution': greeks_attribution,
                'strategy_attribution': strategy_attribution,
                'brinson_attribution': brinson_attribution,
                'regime_attribution': regime_attribution,
                'correlation_matrix': correlation_matrix,
                'risk_metrics': risk_metrics,
                'best_worst_analysis': best_worst,
                'summary_statistics': self._calculate_summary_statistics(trades_df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return self._empty_analysis()
    
    def _calculate_greeks_attribution(self, trades_df: pd.DataFrame) -> GreeksAttribution:
        """Calculate P&L attribution by Greeks"""
        try:
            if not self._has_greeks_data(trades_df):
                return GreeksAttribution(0, 0, 0, 0, 0, 0, 0)
            
            # Calculate Greeks P&L contributions
            delta_pnl = self._calculate_greek_pnl(trades_df, 'delta')
            gamma_pnl = self._calculate_greek_pnl(trades_df, 'gamma')
            theta_pnl = self._calculate_greek_pnl(trades_df, 'theta')
            vega_pnl = self._calculate_greek_pnl(trades_df, 'vega')
            rho_pnl = self._calculate_greek_pnl(trades_df, 'rho')
            
            total_greeks_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
            
            # Calculate unexplained P&L
            total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
            unexplained_pnl = total_pnl - total_greeks_pnl
            
            return GreeksAttribution(
                delta_pnl=delta_pnl,
                gamma_pnl=gamma_pnl,
                theta_pnl=theta_pnl,
                vega_pnl=vega_pnl,
                rho_pnl=rho_pnl,
                total_greeks_pnl=total_greeks_pnl,
                unexplained_pnl=unexplained_pnl
            )
            
        except Exception as e:
            logger.error(f"Error calculating Greeks attribution: {e}")
            return GreeksAttribution(0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_greek_pnl(self, trades_df: pd.DataFrame, greek_name: str) -> float:
        """Calculate P&L contribution for a specific Greek"""
        try:
            if f'{greek_name}_pnl' not in trades_df.columns:
                # Estimate based on Greek exposure and underlying movement
                if greek_name in trades_df.columns and 'underlying_change' in trades_df.columns:
                    greek_exposure = trades_df[greek_name] * trades_df['quantity']
                    greek_pnl = (greek_exposure * trades_df['underlying_change']).sum()
                else:
                    greek_pnl = 0.0
            else:
                greek_pnl = trades_df[f'{greek_name}_pnl'].sum()
            
            return greek_pnl
            
        except Exception as e:
            logger.error(f"Error calculating {greek_name} P&L: {e}")
            return 0.0
    
    def _calculate_strategy_attribution(self, trades_df: pd.DataFrame) -> List[StrategyAttribution]:
        """Calculate strategy-level performance attribution"""
        try:
            if 'strategy' not in trades_df.columns:
                return []
            
            strategy_attributions = []
            
            for strategy in trades_df['strategy'].unique():
                strategy_trades = trades_df[trades_df['strategy'] == strategy]
                
                if len(strategy_trades) == 0:
                    continue
                
                # Calculate strategy metrics
                total_pnl = strategy_trades['pnl'].sum() if 'pnl' in strategy_trades.columns else 0
                total_portfolio_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 1
                pnl_pct = (total_pnl / total_portfolio_pnl) * 100 if total_portfolio_pnl != 0 else 0
                
                trade_count = len(strategy_trades)
                
                # Win rate
                winning_trades = strategy_trades[strategy_trades['pnl'] > 0] if 'pnl' in strategy_trades.columns else pd.DataFrame()
                win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0
                
                # Average win/loss
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                losing_trades = strategy_trades[strategy_trades['pnl'] < 0] if 'pnl' in strategy_trades.columns else pd.DataFrame()
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                
                # Max win/loss
                max_win = strategy_trades['pnl'].max() if 'pnl' in strategy_trades.columns else 0
                max_loss = strategy_trades['pnl'].min() if 'pnl' in strategy_trades.columns else 0
                
                # Sharpe ratio (simplified)
                if 'pnl' in strategy_trades.columns and len(strategy_trades) > 1:
                    returns = strategy_trades['pnl'] / strategy_trades.get('max_loss', 1)
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Contribution to portfolio
                contribution_to_portfolio = pnl_pct
                
                strategy_attr = StrategyAttribution(
                    strategy_name=strategy,
                    total_pnl=total_pnl,
                    pnl_pct=pnl_pct,
                    trade_count=trade_count,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    max_win=max_win,
                    max_loss=max_loss,
                    sharpe_ratio=sharpe_ratio,
                    contribution_to_portfolio=contribution_to_portfolio
                )
                
                strategy_attributions.append(strategy_attr)
            
            # Sort by total P&L
            strategy_attributions.sort(key=lambda x: x.total_pnl, reverse=True)
            
            return strategy_attributions
            
        except Exception as e:
            logger.error(f"Error calculating strategy attribution: {e}")
            return []
    
    def _calculate_brinson_attribution(self, trades_df: pd.DataFrame) -> BrinsonAttribution:
        """Calculate Brinson-style attribution for multi-strategy portfolios"""
        try:
            if 'strategy' not in trades_df.columns or 'pnl' not in trades_df.columns:
                return BrinsonAttribution(0, 0, 0, 0)
            
            # Get benchmark (equal-weighted strategy allocation)
            strategies = trades_df['strategy'].unique()
            n_strategies = len(strategies)
            
            if n_strategies == 0:
                return BrinsonAttribution(0, 0, 0, 0)
            
            # Calculate benchmark allocation (equal weight)
            benchmark_weight = 1.0 / n_strategies
            
            # Calculate actual vs benchmark performance
            total_pnl = trades_df['pnl'].sum()
            
            allocation_effect = 0.0
            selection_effect = 0.0
            interaction_effect = 0.0
            
            for strategy in strategies:
                strategy_trades = trades_df[trades_df['strategy'] == strategy]
                strategy_pnl = strategy_trades['pnl'].sum()
                
                # Actual weight (based on capital allocation)
                actual_weight = self._calculate_strategy_weight(strategy, trades_df)
                
                # Strategy return
                strategy_capital = strategy_trades.get('max_loss', 1).sum()
                strategy_return = strategy_pnl / strategy_capital if strategy_capital > 0 else 0
                
                # Benchmark return for this strategy
                benchmark_pnl = total_pnl * benchmark_weight
                benchmark_return = benchmark_pnl / strategy_capital if strategy_capital > 0 else 0
                
                # Allocation effect: (actual_weight - benchmark_weight) * benchmark_return
                allocation_effect += (actual_weight - benchmark_weight) * benchmark_return
                
                # Selection effect: benchmark_weight * (strategy_return - benchmark_return)
                selection_effect += benchmark_weight * (strategy_return - benchmark_return)
                
                # Interaction effect: (actual_weight - benchmark_weight) * (strategy_return - benchmark_return)
                interaction_effect += (actual_weight - benchmark_weight) * (strategy_return - benchmark_return)
            
            total_attribution = allocation_effect + selection_effect + interaction_effect
            
            return BrinsonAttribution(
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect,
                total_attribution=total_attribution
            )
            
        except Exception as e:
            logger.error(f"Error calculating Brinson attribution: {e}")
            return BrinsonAttribution(0, 0, 0, 0)
    
    def _calculate_strategy_weight(self, strategy: str, trades_df: pd.DataFrame) -> float:
        """Calculate weight of strategy in portfolio"""
        try:
            strategy_trades = trades_df[trades_df['strategy'] == strategy]
            
            if len(strategy_trades) == 0:
                return 0.0
            
            # Weight based on capital allocation (max loss)
            strategy_capital = strategy_trades.get('max_loss', 1).sum()
            total_capital = trades_df.get('max_loss', 1).sum()
            
            return strategy_capital / total_capital if total_capital > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating strategy weight: {e}")
            return 0.0
    
    def _calculate_regime_attribution(self, trades_df: pd.DataFrame) -> List[RegimeAttribution]:
        """Calculate performance attribution by market regime"""
        try:
            if 'regime' not in trades_df.columns or 'pnl' not in trades_df.columns:
                return []
            
            regime_attributions = []
            
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                
                if len(regime_trades) == 0:
                    continue
                
                # Calculate regime metrics
                regime_pnl = regime_trades['pnl'].sum()
                regime_capital = regime_trades.get('max_loss', 1).sum()
                regime_return_pct = (regime_pnl / regime_capital) * 100 if regime_capital > 0 else 0
                
                # Duration (simplified - based on trade count)
                regime_duration = len(regime_trades)
                
                # Best and worst strategies in this regime
                if 'strategy' in regime_trades.columns:
                    strategy_pnl = regime_trades.groupby('strategy')['pnl'].sum()
                    best_strategy = strategy_pnl.idxmax() if len(strategy_pnl) > 0 else "None"
                    worst_strategy = strategy_pnl.idxmin() if len(strategy_pnl) > 0 else "None"
                else:
                    best_strategy = "Unknown"
                    worst_strategy = "Unknown"
                
                # Regime characteristics
                regime_characteristics = {
                    'avg_trade_size': regime_trades.get('max_loss', 0).mean(),
                    'win_rate': len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades),
                    'avg_holding_period': regime_trades.get('days_held', 0).mean(),
                    'volatility': regime_trades['pnl'].std() if 'pnl' in regime_trades.columns else 0
                }
                
                regime_attr = RegimeAttribution(
                    regime_name=regime,
                    regime_duration=regime_duration,
                    regime_pnl=regime_pnl,
                    regime_return_pct=regime_return_pct,
                    best_strategy=best_strategy,
                    worst_strategy=worst_strategy,
                    regime_characteristics=regime_characteristics
                )
                
                regime_attributions.append(regime_attr)
            
            # Sort by regime P&L
            regime_attributions.sort(key=lambda x: x.regime_pnl, reverse=True)
            
            return regime_attributions
            
        except Exception as e:
            logger.error(f"Error calculating regime attribution: {e}")
            return []
    
    def _calculate_strategy_correlations(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        try:
            if 'strategy' not in trades_df.columns or 'pnl' not in trades_df.columns:
                return pd.DataFrame()
            
            # Pivot trades to get strategy returns
            strategy_returns = trades_df.pivot_table(
                index='entry_date' if 'entry_date' in trades_df.columns else trades_df.index,
                columns='strategy',
                values='pnl',
                aggfunc='sum',
                fill_value=0
            )
            
            if strategy_returns.empty:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            correlation_matrix = strategy_returns.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating strategy correlations: {e}")
            return pd.DataFrame()
    
    def _calculate_risk_adjusted_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        try:
            if 'pnl' not in trades_df.columns or len(trades_df) < 2:
                return {}
            
            # Calculate returns
            returns = trades_df['pnl'] / trades_df.get('max_loss', 1)
            returns = returns.dropna()
            
            if len(returns) < 2:
                return {}
            
            # Basic metrics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Sharpe ratio
            sharpe_ratio = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(252) if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else 0
            sortino_ratio = (mean_return * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Information ratio (vs benchmark)
            benchmark_return = 0.08 / 252  # 8% annual benchmark
            excess_returns = returns - benchmark_return
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'max_drawdown': max_drawdown,
                'volatility': std_return * np.sqrt(252),
                'mean_return': mean_return * 252
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}
    
    def _analyze_best_worst_performers(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze best and worst performing trades"""
        try:
            if 'pnl' not in trades_df.columns:
                return {}
            
            # Best performers
            best_trades = trades_df.nlargest(5, 'pnl')
            
            # Worst performers
            worst_trades = trades_df.nsmallest(5, 'pnl')
            
            # Best strategies
            if 'strategy' in trades_df.columns:
                strategy_performance = trades_df.groupby('strategy')['pnl'].agg(['sum', 'mean', 'count'])
                best_strategy = strategy_performance.loc[strategy_performance['sum'].idxmax()]
                worst_strategy = strategy_performance.loc[strategy_performance['sum'].idxmin()]
            else:
                best_strategy = None
                worst_strategy = None
            
            # Best/worst by regime
            if 'regime' in trades_df.columns:
                regime_performance = trades_df.groupby('regime')['pnl'].agg(['sum', 'mean', 'count'])
                best_regime = regime_performance.loc[regime_performance['sum'].idxmax()]
                worst_regime = regime_performance.loc[regime_performance['sum'].idxmin()]
            else:
                best_regime = None
                worst_regime = None
            
            return {
                'best_trades': {
                    'count': len(best_trades),
                    'total_pnl': best_trades['pnl'].sum(),
                    'avg_pnl': best_trades['pnl'].mean(),
                    'strategies': best_trades['strategy'].value_counts().to_dict() if 'strategy' in best_trades.columns else {}
                },
                'worst_trades': {
                    'count': len(worst_trades),
                    'total_pnl': worst_trades['pnl'].sum(),
                    'avg_pnl': worst_trades['pnl'].mean(),
                    'strategies': worst_trades['strategy'].value_counts().to_dict() if 'strategy' in worst_trades.columns else {}
                },
                'best_strategy': {
                    'name': best_strategy.name if best_strategy is not None else 'Unknown',
                    'total_pnl': best_strategy['sum'] if best_strategy is not None else 0,
                    'avg_pnl': best_strategy['mean'] if best_strategy is not None else 0,
                    'trade_count': best_strategy['count'] if best_strategy is not None else 0
                },
                'worst_strategy': {
                    'name': worst_strategy.name if worst_strategy is not None else 'Unknown',
                    'total_pnl': worst_strategy['sum'] if worst_strategy is not None else 0,
                    'avg_pnl': worst_strategy['mean'] if worst_strategy is not None else 0,
                    'trade_count': worst_strategy['count'] if worst_strategy is not None else 0
                },
                'best_regime': {
                    'name': best_regime.name if best_regime is not None else 'Unknown',
                    'total_pnl': best_regime['sum'] if best_regime is not None else 0,
                    'avg_pnl': best_regime['mean'] if best_regime is not None else 0,
                    'trade_count': best_regime['count'] if best_regime is not None else 0
                },
                'worst_regime': {
                    'name': worst_regime.name if worst_regime is not None else 'Unknown',
                    'total_pnl': worst_regime['sum'] if worst_regime is not None else 0,
                    'avg_pnl': worst_regime['mean'] if worst_regime is not None else 0,
                    'trade_count': worst_regime['count'] if worst_regime is not None else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing best/worst performers: {e}")
            return {}
    
    def _calculate_summary_statistics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        try:
            if 'pnl' not in trades_df.columns:
                return {}
            
            total_pnl = trades_df['pnl'].sum()
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            return {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                'total_pnl': total_pnl,
                'gross_profit': winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0,
                'gross_loss': losing_trades['pnl'].sum() if len(losing_trades) > 0 else 0,
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf'),
                'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'largest_win': trades_df['pnl'].max(),
                'largest_loss': trades_df['pnl'].min(),
                'avg_trade': trades_df['pnl'].mean(),
                'trade_frequency': len(trades_df) / 30 if len(trades_df) > 0 else 0  # trades per month
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            return {}
    
    def _has_greeks_data(self, trades_df: pd.DataFrame) -> bool:
        """Check if trades have Greeks data"""
        greeks_columns = ['delta', 'gamma', 'theta', 'vega', 'rho']
        return any(col in trades_df.columns for col in greeks_columns)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'analysis_period': {'start_date': None, 'end_date': None, 'total_trades': 0, 'total_pnl': 0},
            'greeks_attribution': GreeksAttribution(0, 0, 0, 0, 0, 0, 0),
            'strategy_attribution': [],
            'brinson_attribution': BrinsonAttribution(0, 0, 0, 0),
            'regime_attribution': [],
            'correlation_matrix': pd.DataFrame(),
            'risk_metrics': {},
            'best_worst_analysis': {},
            'summary_statistics': {}
        }
    
    def generate_attribution_report(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable attribution report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("OPTIONS PERFORMANCE ATTRIBUTION REPORT")
            report.append("=" * 60)
            
            # Analysis period
            period = analysis['analysis_period']
            report.append(f"\nAnalysis Period: {period['start_date']} to {period['end_date']}")
            report.append(f"Total Trades: {period['total_trades']}")
            report.append(f"Total P&L: ${period['total_pnl']:,.2f}")
            
            # Greeks attribution
            greeks = analysis['greeks_attribution']
            report.append(f"\n{'GREEKS ATTRIBUTION':<30}")
            report.append("-" * 30)
            report.append(f"Delta P&L:     ${greeks.delta_pnl:>10,.2f}")
            report.append(f"Gamma P&L:     ${greeks.gamma_pnl:>10,.2f}")
            report.append(f"Theta P&L:     ${greeks.theta_pnl:>10,.2f}")
            report.append(f"Vega P&L:      ${greeks.vega_pnl:>10,.2f}")
            report.append(f"Rho P&L:       ${greeks.rho_pnl:>10,.2f}")
            report.append(f"{'Total Greeks:':<15} ${greeks.total_greeks_pnl:>10,.2f}")
            report.append(f"{'Unexplained:':<15} ${greeks.unexplained_pnl:>10,.2f}")
            
            # Strategy attribution
            strategies = analysis['strategy_attribution']
            if strategies:
                report.append(f"\n{'STRATEGY ATTRIBUTION':<50}")
                report.append("-" * 50)
                report.append(f"{'Strategy':<20} {'P&L':<10} {'%':<6} {'Trades':<6} {'Win%':<6}")
                report.append("-" * 50)
                
                for strategy in strategies[:10]:  # Top 10 strategies
                    report.append(
                        f"{strategy.strategy_name:<20} "
                        f"${strategy.total_pnl:>8,.0f} "
                        f"{strategy.pnl_pct:>5.1f}% "
                        f"{strategy.trade_count:>5} "
                        f"{strategy.win_rate:>5.1%}"
                    )
            
            # Risk metrics
            risk_metrics = analysis['risk_metrics']
            if risk_metrics:
                report.append(f"\n{'RISK-ADJUSTED METRICS':<30}")
                report.append("-" * 30)
                report.append(f"Sharpe Ratio:      {risk_metrics.get('sharpe_ratio', 0):>8.2f}")
                report.append(f"Sortino Ratio:     {risk_metrics.get('sortino_ratio', 0):>8.2f}")
                report.append(f"Calmar Ratio:      {risk_metrics.get('calmar_ratio', 0):>8.2f}")
                report.append(f"Max Drawdown:      {risk_metrics.get('max_drawdown', 0):>8.1%}")
                report.append(f"Volatility:        {risk_metrics.get('volatility', 0):>8.1%}")
            
            # Best/worst analysis
            best_worst = analysis['best_worst_analysis']
            if best_worst:
                report.append(f"\n{'BEST/WORST ANALYSIS':<40}")
                report.append("-" * 40)
                
                best_strategy = best_worst.get('best_strategy', {})
                worst_strategy = best_worst.get('worst_strategy', {})
                
                report.append(f"Best Strategy:     {best_strategy.get('name', 'Unknown'):<15} ${best_strategy.get('total_pnl', 0):>8,.0f}")
                report.append(f"Worst Strategy:    {worst_strategy.get('name', 'Unknown'):<15} ${worst_strategy.get('total_pnl', 0):>8,.0f}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return "Error generating attribution report"


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create attribution analyzer
    attribution = OptionsPerformanceAttribution(profile)
    
    # Sample trades data
    sample_trades = [
        {
            'entry_date': '2023-01-15',
            'exit_date': '2023-01-25',
            'strategy': 'bull_put_spread',
            'symbol': 'SPY',
            'pnl': 150,
            'max_loss': 350,
            'days_held': 10,
            'delta': 15,
            'gamma': 0.03,
            'theta': -8,
            'vega': 12,
            'regime': 'NORMAL_VOL'
        },
        {
            'entry_date': '2023-01-20',
            'exit_date': '2023-02-05',
            'strategy': 'iron_condor',
            'symbol': 'QQQ',
            'pnl': -75,
            'max_loss': 300,
            'days_held': 16,
            'delta': -5,
            'gamma': 0.02,
            'theta': -10,
            'vega': 15,
            'regime': 'HIGH_VOL'
        },
        {
            'entry_date': '2023-02-10',
            'exit_date': '2023-02-20',
            'strategy': 'bull_put_spread',
            'symbol': 'SPY',
            'pnl': 200,
            'max_loss': 400,
            'days_held': 10,
            'delta': 18,
            'gamma': 0.04,
            'theta': -12,
            'vega': 14,
            'regime': 'LOW_VOL'
        }
    ]
    
    # Analyze trades
    analysis = attribution.analyze_trades(sample_trades)
    
    # Generate report
    report = attribution.generate_attribution_report(analysis)
    print(report)
