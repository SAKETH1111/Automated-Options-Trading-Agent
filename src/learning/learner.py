"""Strategy learning and parameter optimization"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from src.config.settings import get_config
from src.database.models import LearningLog, Trade
from src.database.session import get_db
from .analyzer import TradeAnalyzer


class StrategyLearner:
    """Learn from trades and adjust strategy parameters"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.learning_config = self.config.get("learning", {})
        
        self.enabled = self.learning_config.get("enabled", True)
        self.max_change_pct = self.learning_config.get("adjustment", {}).get("max_change_pct", 10)
        self.min_sample_size = self.learning_config.get("adjustment", {}).get("min_sample_size", 30)
        self.confidence_threshold = self.learning_config.get("adjustment", {}).get("confidence_threshold", 0.8)
        
        self.analyzer = TradeAnalyzer()
        self.db = get_db()
        
        logger.info(f"Strategy Learner initialized (enabled: {self.enabled})")
    
    def analyze_and_learn(self, strategy_name: str) -> Optional[Dict]:
        """
        Analyze trades and suggest parameter adjustments
        
        Returns:
            Dict with suggested parameter changes
        """
        if not self.enabled:
            return None
        
        try:
            # Get learning insights
            insights = self.analyzer.get_learning_insights(self.min_sample_size)
            
            if not insights.get("ready_for_learning"):
                logger.info("Not enough data for learning")
                return None
            
            # Get performance metrics
            metrics = self.analyzer.calculate_performance_metrics(period_days=90)
            
            # Get strategy-specific trades
            with self.db.get_session() as session:
                strategy_trades = session.query(Trade).filter(
                    Trade.strategy == strategy_name,
                    Trade.status == "closed"
                ).order_by(Trade.timestamp_exit.desc()).limit(100).all()
                
                if len(strategy_trades) < self.min_sample_size:
                    logger.info(f"Not enough {strategy_name} trades for learning")
                    return None
            
            # Analyze common issues
            recommendations = insights.get("recommendations", [])
            
            if not recommendations:
                logger.info("No significant issues found")
                return None
            
            # Generate parameter adjustments
            adjustments = self._generate_adjustments(
                strategy_name, strategy_trades, recommendations, metrics
            )
            
            if adjustments and adjustments.get("changes"):
                # Log the learning update
                self._log_learning_update(strategy_name, adjustments, metrics)
                
                return adjustments
            
            return None
        
        except Exception as e:
            logger.error(f"Error in learning process: {e}")
            return None
    
    def _generate_adjustments(
        self,
        strategy_name: str,
        trades: List[Trade],
        recommendations: List[Dict],
        metrics: Dict
    ) -> Optional[Dict]:
        """Generate parameter adjustment recommendations"""
        try:
            adjustments = {
                "strategy": strategy_name,
                "changes": [],
                "reasoning": [],
                "confidence": 0.0,
            }
            
            # Analyze common error patterns
            error_counts = {}
            for trade in trades:
                for tag in trade.reason_tags:
                    error_counts[tag] = error_counts.get(tag, 0) + 1
            
            # Most common error
            if error_counts:
                top_error = max(error_counts.items(), key=lambda x: x[1])
                error_type, error_count = top_error
                error_rate = error_count / len(trades)
                
                if error_rate > 0.25:  # More than 25% of trades
                    # Generate specific adjustments
                    if error_type == "entry_quality":
                        # Increase IV rank threshold
                        adjustments["changes"].append({
                            "parameter": "min_iv_rank",
                            "old_value": 25,
                            "new_value": 35,
                            "change_pct": 40,
                        })
                        adjustments["reasoning"].append(
                            f"Entry quality issues in {error_rate*100:.1f}% of trades"
                        )
                    
                    elif error_type == "liquidity_execution":
                        # Increase liquidity requirements
                        adjustments["changes"].append({
                            "parameter": "min_open_interest",
                            "old_value": 100,
                            "new_value": 200,
                            "change_pct": 100,
                        })
                        adjustments["reasoning"].append(
                            f"Liquidity issues in {error_rate*100:.1f}% of trades"
                        )
                    
                    elif error_type == "timing":
                        # Adjust DTE range
                        adjustments["changes"].append({
                            "parameter": "dte_range",
                            "old_value": [25, 45],
                            "new_value": [30, 50],
                            "change_pct": 10,
                        })
                        adjustments["reasoning"].append(
                            f"Timing issues in {error_rate*100:.1f}% of trades"
                        )
            
            # Check win rate
            win_rate = metrics.get("win_rate", 0) / 100
            if win_rate < 0.55:  # Less than 55% win rate
                # Be more conservative
                adjustments["changes"].append({
                    "parameter": "short_delta_range",
                    "old_value": [-0.30, -0.20],
                    "new_value": [-0.25, -0.15],
                    "change_pct": 17,
                })
                adjustments["reasoning"].append(
                    f"Win rate below target: {win_rate*100:.1f}%"
                )
            
            # Calculate confidence based on sample size and consistency
            sample_size = len(trades)
            confidence = min(sample_size / 50, 1.0) * 0.8  # Max 80% confidence
            
            adjustments["confidence"] = round(confidence, 2)
            
            # Only return if confidence is high enough
            if confidence >= self.confidence_threshold and adjustments["changes"]:
                return adjustments
            
            return None
        
        except Exception as e:
            logger.error(f"Error generating adjustments: {e}")
            return None
    
    def _log_learning_update(
        self,
        strategy_name: str,
        adjustments: Dict,
        metrics: Dict
    ):
        """Log learning update to database"""
        try:
            with self.db.get_session() as session:
                # Get current parameters (simplified - in production, fetch from config)
                old_params = {}
                new_params = {}
                
                for change in adjustments["changes"]:
                    param = change["parameter"]
                    old_params[param] = change["old_value"]
                    new_params[param] = change["new_value"]
                
                log = LearningLog(
                    log_id=str(datetime.now().timestamp()),
                    timestamp=datetime.now(),
                    update_type="parameter_adjustment",
                    strategy=strategy_name,
                    old_params=old_params,
                    new_params=new_params,
                    reason="; ".join(adjustments["reasoning"]),
                    confidence=adjustments["confidence"],
                    trades_analyzed=metrics.get("total_trades", 0),
                    expected_improvement=5.0,  # Placeholder
                    status="active",
                )
                
                session.add(log)
                session.commit()
                
                logger.info(f"Learning update logged for {strategy_name}")
        
        except Exception as e:
            logger.error(f"Error logging learning update: {e}")
    
    def apply_adjustments(self, strategy_name: str, adjustments: Dict) -> bool:
        """
        Apply parameter adjustments to strategy configuration
        
        In production, this would update the config file or database
        For now, just log the adjustment
        """
        try:
            logger.info(f"Applying adjustments to {strategy_name}:")
            for change in adjustments["changes"]:
                logger.info(f"  {change['parameter']}: {change['old_value']} -> {change['new_value']}")
            
            # In production:
            # 1. Update config file
            # 2. Reload strategies
            # 3. Start A/B test
            
            return True
        
        except Exception as e:
            logger.error(f"Error applying adjustments: {e}")
            return False
    
    def get_learning_history(self, strategy_name: Optional[str] = None) -> List[Dict]:
        """Get history of learning updates"""
        try:
            with self.db.get_session() as session:
                query = session.query(LearningLog)
                
                if strategy_name:
                    query = query.filter_by(strategy=strategy_name)
                
                logs = query.order_by(LearningLog.timestamp.desc()).limit(20).all()
                
                return [
                    {
                        "log_id": log.log_id,
                        "timestamp": log.timestamp.isoformat(),
                        "strategy": log.strategy,
                        "update_type": log.update_type,
                        "old_params": log.old_params,
                        "new_params": log.new_params,
                        "reason": log.reason,
                        "confidence": log.confidence,
                        "status": log.status,
                    }
                    for log in logs
                ]
        
        except Exception as e:
            logger.error(f"Error getting learning history: {e}")
            return []









