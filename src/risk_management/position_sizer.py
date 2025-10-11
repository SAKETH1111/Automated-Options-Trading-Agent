"""
Dynamic Position Sizer Module
Calculate optimal position sizes based on volatility and Kelly Criterion
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import Trade, IndexTickData


class DynamicPositionSizer:
    """
    Calculate dynamic position sizes
    Adjusts based on volatility, Kelly Criterion, and market conditions
    """
    
    def __init__(
        self,
        db_session: Session,
        base_capital: float = 10000.0
    ):
        """
        Initialize dynamic position sizer
        
        Args:
            db_session: Database session
            base_capital: Base trading capital
        """
        self.db = db_session
        self.base_capital = base_capital
        
        # Base risk parameters
        self.base_risk_pct = 0.02  # 2% base risk
        self.min_risk_pct = 0.005  # 0.5% minimum
        self.max_risk_pct = 0.05  # 5% maximum
        
        logger.info("Dynamic Position Sizer initialized")
    
    def calculate_position_size(
        self,
        symbol: str,
        strategy: str,
        max_loss: float,
        confidence: float = 0.70
    ) -> Dict:
        """
        Calculate optimal position size
        
        Args:
            symbol: Symbol to trade
            strategy: Strategy type
            max_loss: Max loss per contract
            confidence: Confidence in trade (0-1)
            
        Returns:
            Position sizing recommendation
        """
        try:
            # Get volatility adjustment
            vol_adjustment = self._get_volatility_adjustment(symbol)
            
            # Get Kelly Criterion sizing
            kelly_size = self._calculate_kelly_size(strategy, confidence)
            
            # Get market regime adjustment
            regime_adjustment = self._get_regime_adjustment(symbol)
            
            # Calculate adjusted risk
            adjusted_risk_pct = self.base_risk_pct * vol_adjustment * kelly_size * regime_adjustment
            
            # Clamp to min/max
            adjusted_risk_pct = max(self.min_risk_pct, min(self.max_risk_pct, adjusted_risk_pct))
            
            # Calculate position size
            risk_amount = self.base_capital * adjusted_risk_pct
            quantity = int(risk_amount / abs(max_loss)) if max_loss != 0 else 1
            quantity = max(1, quantity)  # At least 1 contract
            
            return {
                'quantity': quantity,
                'risk_amount': risk_amount,
                'risk_pct': adjusted_risk_pct,
                'adjustments': {
                    'volatility': vol_adjustment,
                    'kelly': kelly_size,
                    'regime': regime_adjustment
                },
                'max_loss_total': abs(max_loss) * quantity
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'quantity': 1,
                'risk_amount': self.base_capital * self.base_risk_pct,
                'risk_pct': self.base_risk_pct
            }
    
    def _get_volatility_adjustment(self, symbol: str) -> float:
        """
        Calculate volatility-based adjustment
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Adjustment factor (0.5 to 1.5)
        """
        try:
            # Get recent price data
            cutoff = datetime.utcnow() - timedelta(days=20)
            
            with self.db.get_session() as session:
                query = session.query(IndexTickData).filter(
                    IndexTickData.symbol == symbol,
                    IndexTickData.timestamp >= cutoff
                ).order_by(IndexTickData.timestamp.asc())
            
            data = query.all()
            
            if len(data) < 100:
                return 1.0  # Default if insufficient data
            
            # Calculate volatility
            prices = [d.price for d in data]
            returns = np.diff(np.log(prices))
            current_vol = np.std(returns)
            
            # Calculate historical average volatility
            historical_vol = np.mean([np.std(returns[i:i+100]) 
                                     for i in range(0, len(returns)-100, 100)])
            
            if historical_vol == 0:
                return 1.0
            
            # Adjustment: reduce size in high volatility
            vol_ratio = current_vol / historical_vol
            
            if vol_ratio > 1.5:
                adjustment = 0.5  # Reduce size by 50% in high vol
            elif vol_ratio > 1.2:
                adjustment = 0.75  # Reduce size by 25%
            elif vol_ratio < 0.8:
                adjustment = 1.25  # Increase size by 25% in low vol
            else:
                adjustment = 1.0  # Normal volatility
            
            logger.debug(f"Volatility adjustment for {symbol}: {adjustment:.2f}x")
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    def _calculate_kelly_size(
        self,
        strategy: str,
        confidence: float
    ) -> float:
        """
        Calculate Kelly Criterion position size
        
        Args:
            strategy: Strategy type
            confidence: Confidence in trade
            
        Returns:
            Kelly fraction (0.25 to 1.5)
        """
        try:
            # Get historical win rate and avg win/loss for strategy
            with self.db.get_session() as session:
                recent_trades = session.query(Trade).filter(
                    Trade.strategy == strategy,
                    Trade.status == 'closed'
                ).order_by(Trade.timestamp_exit.desc()).limit(50).all()
            
            if len(recent_trades) < 10:
                # Not enough history, use confidence
                return confidence
            
            # Calculate win rate
            wins = [t for t in recent_trades if t.pnl > 0]
            losses = [t for t in recent_trades if t.pnl < 0]
            
            win_rate = len(wins) / len(recent_trades)
            
            # Calculate avg win/loss ratio
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1
            
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            
            # Kelly Criterion: f = (p * b - q) / b
            # where p = win rate, q = loss rate, b = win/loss ratio
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Use fractional Kelly (25% of full Kelly for safety)
            kelly_fraction = kelly_fraction * 0.25
            
            # Clamp to reasonable range
            kelly_fraction = max(0.25, min(1.5, kelly_fraction))
            
            logger.debug(f"Kelly sizing for {strategy}: {kelly_fraction:.2f}x")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly size: {e}")
            return 1.0
    
    def _get_regime_adjustment(self, symbol: str) -> float:
        """
        Calculate market regime-based adjustment
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Adjustment factor (0.5 to 1.5)
        """
        try:
            # Get recent market regime data
            from src.database.models import MarketRegime
            
            with self.db.get_session() as session:
                latest_regime = session.query(MarketRegime).filter(
                    MarketRegime.symbol == symbol
                ).order_by(MarketRegime.timestamp.desc()).first()
            
            if not latest_regime:
                return 1.0
            
            trend_regime = latest_regime.trend_regime
            vol_regime = latest_regime.volatility_regime
            
            # Adjust based on regime
            adjustment = 1.0
            
            # Favorable regimes: increase size
            if 'UPTREND' in trend_regime and 'NORMAL' in vol_regime:
                adjustment = 1.25
            elif 'STRONG_UPTREND' in trend_regime:
                adjustment = 1.5
            
            # Unfavorable regimes: decrease size
            elif 'HIGH_VOLATILITY' in vol_regime:
                adjustment = 0.75
            elif 'VERY_HIGH_VOLATILITY' in vol_regime:
                adjustment = 0.5
            
            logger.debug(f"Regime adjustment for {symbol}: {adjustment:.2f}x")
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating regime adjustment: {e}")
            return 1.0
    
    def _get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            with self.db.get_session() as session:
                trades = session.query(Trade).filter(
                    Trade.status == 'open'
                ).all()
            
            return [{
                'symbol': t.symbol,
                'strategy': t.strategy,
                'max_loss': t.risk.get('max_loss') if isinstance(t.risk, dict) else 0
            } for t in trades]
            
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

