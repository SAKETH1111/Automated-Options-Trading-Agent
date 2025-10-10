"""
Ensemble Predictor Module
Combine multiple models for better predictions
"""

from typing import Dict, List
from sqlalchemy.orm import Session
from loguru import logger

from .signal_predictor import SignalPredictor
from .volatility_forecaster import VolatilityForecaster
from .strike_optimizer import StrikeOptimizer


class EnsemblePredictor:
    """
    Ensemble multiple ML models for robust predictions
    Combines signal predictor, volatility forecaster, and strike optimizer
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize ensemble predictor
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        
        # Initialize component models
        self.signal_predictor = SignalPredictor(db_session)
        self.volatility_forecaster = VolatilityForecaster(db_session)
        self.strike_optimizer = StrikeOptimizer(db_session)
        
        logger.info("Ensemble Predictor initialized")
    
    def get_comprehensive_prediction(
        self,
        symbol: str,
        strategy: str,
        current_price: float,
        iv_rank: float,
        trend: str
    ) -> Dict:
        """
        Get comprehensive prediction from all models
        
        Args:
            symbol: Symbol to predict
            strategy: Strategy type
            current_price: Current price
            iv_rank: IV Rank
            trend: Market trend
            
        Returns:
            Comprehensive prediction
        """
        try:
            logger.info(f"Getting ensemble prediction for {symbol}")
            
            # Get predictions from each model
            entry_signal = self.signal_predictor.predict_entry_signal(symbol)
            vol_forecast = self.volatility_forecaster.forecast_volatility(symbol)
            optimal_strikes = self.strike_optimizer.find_optimal_strikes(
                symbol, current_price, strategy, iv_rank, trend
            )
            win_prob = self.signal_predictor.predict_win_probability(
                symbol, strategy, [], 35
            )
            
            # Combine predictions
            ensemble = {
                'symbol': symbol,
                'strategy': strategy,
                'entry_signal': entry_signal,
                'volatility_forecast': vol_forecast,
                'optimal_strikes': optimal_strikes,
                'win_probability': win_prob,
                'recommendation': self._generate_recommendation(
                    entry_signal, vol_forecast, win_prob, iv_rank
                )
            }
            
            logger.info(f"Ensemble prediction complete: "
                       f"Entry={entry_signal.get('should_enter')}, "
                       f"Win Prob={win_prob:.1%}")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {}
    
    def _generate_recommendation(
        self,
        entry_signal: Dict,
        vol_forecast: Dict,
        win_prob: float,
        iv_rank: float
    ) -> Dict:
        """Generate trading recommendation from ensemble"""
        # Combine all signals
        score = 0
        reasons = []
        
        # Entry signal
        if entry_signal.get('should_enter'):
            score += 30
            reasons.append(f"ML entry signal (confidence: {entry_signal.get('confidence', 0):.1%})")
        
        # Win probability
        if win_prob >= 0.65:
            score += 25
            reasons.append(f"High win probability ({win_prob:.1%})")
        elif win_prob >= 0.55:
            score += 15
        
        # Volatility forecast
        if vol_forecast.get('regime') == 'STABLE':
            score += 20
            reasons.append("Stable volatility forecast")
        elif vol_forecast.get('regime') == 'DECREASING':
            score += 15
            reasons.append("Decreasing volatility forecast")
        
        # IV rank
        if iv_rank > 60:
            score += 25
            reasons.append(f"High IV Rank ({iv_rank:.0f})")
        elif iv_rank > 50:
            score += 15
        
        # Determine recommendation
        if score >= 75:
            action = 'STRONG_BUY'
            confidence = 'HIGH'
        elif score >= 60:
            action = 'BUY'
            confidence = 'MEDIUM'
        elif score >= 40:
            action = 'NEUTRAL'
            confidence = 'LOW'
        else:
            action = 'AVOID'
            confidence = 'VERY_LOW'
        
        return {
            'action': action,
            'confidence': confidence,
            'score': score,
            'reasons': reasons
        }

