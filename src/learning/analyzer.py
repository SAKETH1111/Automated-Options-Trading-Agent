"""Trade analysis and error taxonomy"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from sqlalchemy import func

from src.database.models import Trade, PerformanceMetric
from src.database.session import get_db


class TradeAnalyzer:
    """Analyze trades and categorize errors"""
    
    # Error taxonomy
    ERROR_CATEGORIES = [
        "entry_quality",
        "liquidity_execution",
        "volatility",
        "risk_policy",
        "timing",
        "greek_risk",
    ]
    
    def __init__(self):
        self.db = get_db()
        logger.info("Trade Analyzer initialized")
    
    def analyze_trade(self, trade: Trade) -> Dict:
        """
        Analyze a closed trade and assign reason tags
        
        Returns:
            Dict with analysis results and reason tags
        """
        try:
            if trade.status != "closed":
                return {}
            
            analysis = {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "strategy": trade.strategy,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "reason_tags": [],
                "insights": [],
            }
            
            # Analyze entry quality
            entry_analysis = self._analyze_entry_quality(trade)
            if entry_analysis["has_issues"]:
                analysis["reason_tags"].append("entry_quality")
                analysis["insights"].extend(entry_analysis["insights"])
            
            # Analyze execution/liquidity
            execution_analysis = self._analyze_execution(trade)
            if execution_analysis["has_issues"]:
                analysis["reason_tags"].append("liquidity_execution")
                analysis["insights"].extend(execution_analysis["insights"])
            
            # Analyze volatility impact
            volatility_analysis = self._analyze_volatility(trade)
            if volatility_analysis["has_issues"]:
                analysis["reason_tags"].append("volatility")
                analysis["insights"].extend(volatility_analysis["insights"])
            
            # Analyze risk management
            risk_analysis = self._analyze_risk_policy(trade)
            if risk_analysis["has_issues"]:
                analysis["reason_tags"].append("risk_policy")
                analysis["insights"].extend(risk_analysis["insights"])
            
            # Analyze timing
            timing_analysis = self._analyze_timing(trade)
            if timing_analysis["has_issues"]:
                analysis["reason_tags"].append("timing")
                analysis["insights"].extend(timing_analysis["insights"])
            
            # Update trade with reason tags
            trade.reason_tags = analysis["reason_tags"]
            trade.notes = "; ".join(analysis["insights"])
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing trade: {e}")
            return {}
    
    def _analyze_entry_quality(self, trade: Trade) -> Dict:
        """Analyze if entry parameters were optimal"""
        issues = []
        
        try:
            params = trade.params
            market_snapshot = trade.market_snapshot
            
            # Check IV rank at entry
            iv_rank = market_snapshot.get("iv_rank", 50)
            if iv_rank < 30:
                issues.append("IV rank too low at entry")
            
            # Check delta selection
            if trade.strategy == "Bull Put Spread":
                short_delta = params.get("short_delta", 0)
                if abs(short_delta) > 0.35:
                    issues.append("Short delta too high (risky)")
            
            # Check DTE
            dte = params.get("dte", 30)
            if dte < 20:
                issues.append("DTE too short at entry")
            elif dte > 60:
                issues.append("DTE too long at entry")
            
            return {
                "has_issues": len(issues) > 0,
                "insights": issues,
            }
        
        except Exception as e:
            logger.error(f"Error in entry quality analysis: {e}")
            return {"has_issues": False, "insights": []}
    
    def _analyze_execution(self, trade: Trade) -> Dict:
        """Analyze execution quality and slippage"""
        issues = []
        
        try:
            execution = trade.execution
            slippage = execution.get("slippage", 0)
            
            # Check slippage
            if abs(slippage) > 10:  # More than 10% slippage
                issues.append(f"High slippage: {slippage:.1f}%")
            
            # Check if filled at poor price
            limit_credit = execution.get("limit_credit", 0)
            fill_credit = execution.get("fill_credit", 0)
            
            if fill_credit < limit_credit * 0.85:  # Got less than 85% of expected
                issues.append("Poor fill price")
            
            return {
                "has_issues": len(issues) > 0,
                "insights": issues,
            }
        
        except Exception as e:
            logger.error(f"Error in execution analysis: {e}")
            return {"has_issues": False, "insights": []}
    
    def _analyze_volatility(self, trade: Trade) -> Dict:
        """Analyze volatility impact"""
        issues = []
        
        try:
            # Check if volatility crushed after entry
            entry_iv = trade.market_snapshot.get("short_iv", 0)
            
            # In production, get exit IV
            # For now, check if IV rank dropped significantly
            iv_rank_entry = trade.market_snapshot.get("iv_rank", 50)
            
            # If losing trade and IV rank was moderate
            if trade.pnl < 0 and iv_rank_entry < 40:
                issues.append("Entered during low IV period")
            
            return {
                "has_issues": len(issues) > 0,
                "insights": issues,
            }
        
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {"has_issues": False, "insights": []}
    
    def _analyze_risk_policy(self, trade: Trade) -> Dict:
        """Analyze if risk rules were followed"""
        issues = []
        
        try:
            risk = trade.risk
            exit_reason = trade.exit_reason
            
            # Check if stop loss was hit
            if exit_reason == "stop_loss":
                issues.append("Stop loss triggered - position management issue")
            
            # Check if position size was appropriate
            risk_pct = risk.get("risk_pct", 0)
            if risk_pct > 2.5:  # More than 2.5% at risk
                issues.append(f"Position size too large: {risk_pct:.1f}% at risk")
            
            return {
                "has_issues": len(issues) > 0,
                "insights": issues,
            }
        
        except Exception as e:
            logger.error(f"Error in risk policy analysis: {e}")
            return {"has_issues": False, "insights": []}
    
    def _analyze_timing(self, trade: Trade) -> Dict:
        """Analyze if timing was good"""
        issues = []
        
        try:
            params = trade.params
            exit_reason = trade.exit_reason
            days_held = trade.days_held
            dte_entry = params.get("dte", 30)
            
            # Check if held too long
            if days_held > dte_entry * 0.8:
                issues.append("Held too close to expiration")
            
            # Check if exited too early
            if exit_reason == "take_profit" and days_held < 3:
                issues.append("Exited very early - could have held longer?")
            
            # Check if exit was late
            if exit_reason == "stop_loss" and trade.pnl < -trade.execution.get("fill_credit", 0) * 150:
                issues.append("Should have exited earlier")
            
            return {
                "has_issues": len(issues) > 0,
                "insights": issues,
            }
        
        except Exception as e:
            logger.error(f"Error in timing analysis: {e}")
            return {"has_issues": False, "insights": []}
    
    def calculate_performance_metrics(self, period_days: int = 30) -> Dict:
        """Calculate performance metrics for a period"""
        try:
            with self.db.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=period_days)
                
                closed_trades = session.query(Trade).filter(
                    Trade.timestamp_exit >= cutoff_date,
                    Trade.status == "closed"
                ).all()
                
                if not closed_trades:
                    return {}
                
                # Basic stats
                total_trades = len(closed_trades)
                winning_trades = [t for t in closed_trades if t.pnl > 0]
                losing_trades = [t for t in closed_trades if t.pnl < 0]
                
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                
                total_pnl = sum(t.pnl for t in closed_trades)
                avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
                
                profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0
                
                # Sharpe ratio
                returns = [t.pnl for t in closed_trades]
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = [r for r in returns if r < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1
                sortino = (np.mean(returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
                
                # Max drawdown
                cumulative_pnl = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = running_max - cumulative_pnl
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                
                # Error analysis
                error_counts = defaultdict(int)
                for trade in closed_trades:
                    for tag in trade.reason_tags:
                        error_counts[tag] += 1
                
                # Strategy breakdown
                strategy_performance = {}
                strategies = set(t.strategy for t in closed_trades)
                for strategy in strategies:
                    strategy_trades = [t for t in closed_trades if t.strategy == strategy]
                    strategy_pnl = sum(t.pnl for t in strategy_trades)
                    strategy_win_rate = len([t for t in strategy_trades if t.pnl > 0]) / len(strategy_trades) if strategy_trades else 0
                    
                    strategy_performance[strategy] = {
                        "total_trades": len(strategy_trades),
                        "pnl": strategy_pnl,
                        "win_rate": strategy_win_rate,
                    }
                
                return {
                    "period_days": period_days,
                    "total_trades": total_trades,
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "win_rate": round(win_rate * 100, 2),
                    "total_pnl": round(total_pnl, 2),
                    "avg_win": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "profit_factor": round(profit_factor, 2),
                    "sharpe_ratio": round(sharpe, 2),
                    "sortino_ratio": round(sortino, 2),
                    "max_drawdown": round(max_drawdown, 2),
                    "error_counts": dict(error_counts),
                    "strategy_performance": strategy_performance,
                }
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_learning_insights(self, min_sample_size: int = 20) -> Dict:
        """Generate insights for strategy improvement"""
        try:
            metrics = self.calculate_performance_metrics(period_days=90)
            
            if not metrics or metrics.get("total_trades", 0) < min_sample_size:
                return {
                    "ready_for_learning": False,
                    "reason": "insufficient_data",
                }
            
            insights = {
                "ready_for_learning": True,
                "recommendations": [],
            }
            
            # Analyze common errors
            error_counts = metrics.get("error_counts", {})
            total_trades = metrics["total_trades"]
            
            for error, count in error_counts.items():
                error_rate = count / total_trades
                if error_rate > 0.3:  # More than 30% of trades
                    insights["recommendations"].append({
                        "issue": error,
                        "frequency": f"{error_rate*100:.1f}%",
                        "action": self._get_error_recommendation(error),
                    })
            
            # Analyze strategy performance
            strategy_perf = metrics.get("strategy_performance", {})
            for strategy, perf in strategy_perf.items():
                if perf["win_rate"] < 0.5 and perf["total_trades"] >= 10:
                    insights["recommendations"].append({
                        "issue": f"low_win_rate_{strategy}",
                        "frequency": f"{perf['win_rate']*100:.1f}%",
                        "action": f"Review {strategy} parameters or disable temporarily",
                    })
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
            return {}
    
    def _get_error_recommendation(self, error_category: str) -> str:
        """Get recommendation for error category"""
        recommendations = {
            "entry_quality": "Increase IV rank minimum or adjust delta selection",
            "liquidity_execution": "Increase minimum OI/volume requirements or use limit orders with wider spreads",
            "volatility": "Enter only during high IV rank periods (>40)",
            "risk_policy": "Reduce position size or tighten stop losses",
            "timing": "Adjust DTE or exit rules",
            "greek_risk": "Better delta or vega management",
        }
        return recommendations.get(error_category, "Review strategy parameters")


