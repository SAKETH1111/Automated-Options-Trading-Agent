"""
Market Analysis Service
Integrates indicators, patterns, and regime detection for comprehensive market analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from .indicators import TechnicalIndicators
from .patterns import PatternRecognition
from .regime import MarketRegimeDetector
from src.database.models import IndexTickData, TechnicalIndicators as TechIndModel
from src.database.models import MarketRegime as MarketRegimeModel
from src.database.models import PatternDetection as PatternDetectionModel


class MarketAnalyzer:
    """
    Comprehensive market analysis service
    Combines indicators, patterns, and regime detection
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize market analyzer
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.indicators = TechnicalIndicators()
        self.patterns = PatternRecognition()
        self.regime = MarketRegimeDetector()
        logger.info("Market Analyzer initialized")
    
    def get_recent_data(
        self,
        symbol: str,
        minutes: int = 60
    ) -> pd.DataFrame:
        """
        Get recent tick data from database
        
        Args:
            symbol: Symbol to analyze
            minutes: Minutes of data to retrieve
            
        Returns:
            DataFrame with tick data
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        query = self.db.query(IndexTickData).filter(
            IndexTickData.symbol == symbol,
            IndexTickData.timestamp >= cutoff_time
        ).order_by(IndexTickData.timestamp.asc())
        
        data = query.all()
        
        if not data:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'price': d.price,
            'volume': d.volume or 0,
            'bid': d.bid,
            'ask': d.ask,
            'spread': d.spread
        } for d in data])
        
        # Add OHLC approximation (using price as close)
        df['high'] = df['price']
        df['low'] = df['price']
        df['close'] = df['price']
        
        return df
    
    def analyze_symbol(
        self,
        symbol: str,
        store_results: bool = True
    ) -> Dict[str, any]:
        """
        Perform comprehensive analysis on a symbol
        
        Args:
            symbol: Symbol to analyze
            store_results: Whether to store results in database
            
        Returns:
            Dictionary with complete analysis
        """
        try:
            logger.info(f"Analyzing {symbol}...")
            
            # Get recent data
            df = self.get_recent_data(symbol, minutes=120)
            
            if df.empty:
                return {'error': 'No data available'}
            
            # Calculate indicators
            df_with_indicators = self.indicators.calculate_all_indicators(df)
            indicator_signals = self.indicators.get_indicator_signals(df_with_indicators)
            
            # Analyze patterns
            pattern_analysis = self.patterns.analyze_all_patterns(df, 'price', 'volume')
            
            # Detect market regime
            regime_analysis = self.regime.detect_overall_regime(df, 'price', 'volume')
            
            # Compile results
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'current_price': df['price'].iloc[-1],
                'data_points': len(df),
                'indicators': {
                    'values': df_with_indicators.iloc[-1].to_dict(),
                    'signals': indicator_signals
                },
                'patterns': pattern_analysis,
                'regime': regime_analysis
            }
            
            # Store in database
            if store_results:
                self._store_analysis(symbol, df_with_indicators, pattern_analysis, regime_analysis)
            
            logger.info(f"Analysis complete for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'error': str(e)}
    
    def _store_analysis(
        self,
        symbol: str,
        df_with_indicators: pd.DataFrame,
        pattern_analysis: Dict,
        regime_analysis: Dict
    ):
        """Store analysis results in database"""
        try:
            timestamp = datetime.utcnow()
            latest = df_with_indicators.iloc[-1]
            
            # Store technical indicators
            tech_ind = TechIndModel(
                symbol=symbol,
                timestamp=timestamp,
                sma_10=latest.get('sma_10'),
                sma_20=latest.get('sma_20'),
                sma_50=latest.get('sma_50'),
                sma_200=latest.get('sma_200'),
                ema_12=latest.get('ema_12'),
                ema_26=latest.get('ema_26'),
                ema_50=latest.get('ema_50'),
                rsi_14=latest.get('rsi'),
                macd=latest.get('macd'),
                macd_signal=latest.get('macd_signal'),
                macd_histogram=latest.get('macd_histogram'),
                bb_upper=latest.get('bb_upper'),
                bb_middle=latest.get('bb_middle'),
                bb_lower=latest.get('bb_lower'),
                bb_width=latest.get('bb_width'),
                atr_14=latest.get('atr'),
                adx=latest.get('adx'),
                plus_di=latest.get('plus_di'),
                minus_di=latest.get('minus_di'),
                obv=latest.get('obv')
            )
            self.db.add(tech_ind)
            
            # Store market regime
            if regime_analysis:
                regime_model = MarketRegimeModel(
                    symbol=symbol,
                    timestamp=timestamp,
                    volatility_regime=regime_analysis.get('volatility', {}).get('regime'),
                    trend_regime=regime_analysis.get('trend', {}).get('regime'),
                    momentum_regime=regime_analysis.get('momentum', {}).get('regime'),
                    volume_regime=regime_analysis.get('volume', {}).get('regime'),
                    market_hours_regime=regime_analysis.get('market_hours', {}).get('regime'),
                    volatility_percentile=regime_analysis.get('volatility', {}).get('percentile'),
                    trend_strength=regime_analysis.get('trend', {}).get('trend_strength'),
                    rsi_value=regime_analysis.get('momentum', {}).get('rsi'),
                    volume_ratio=regime_analysis.get('volume', {}).get('volume_ratio'),
                    recommended_action=regime_analysis.get('recommendation', {}).get('action'),
                    recommended_strategy=regime_analysis.get('recommendation', {}).get('strategy')
                )
                self.db.add(regime_model)
            
            # Store pattern detection
            if pattern_analysis:
                pattern_model = PatternDetectionModel(
                    symbol=symbol,
                    timestamp=timestamp,
                    support_levels=pattern_analysis.get('support_resistance', {}).get('support'),
                    resistance_levels=pattern_analysis.get('support_resistance', {}).get('resistance'),
                    near_support=pattern_analysis.get('near_levels', {}).get('near_support', False),
                    near_resistance=pattern_analysis.get('near_levels', {}).get('near_resistance', False),
                    trend_direction=pattern_analysis.get('trend', {}).get('direction'),
                    trend_strength=pattern_analysis.get('trend', {}).get('strength'),
                    trend_angle=pattern_analysis.get('trend', {}).get('angle'),
                    higher_highs=pattern_analysis.get('higher_highs_lows', {}).get('higher_highs', False),
                    higher_lows=pattern_analysis.get('higher_highs_lows', {}).get('higher_lows', False),
                    lower_highs=pattern_analysis.get('higher_highs_lows', {}).get('lower_highs', False),
                    lower_lows=pattern_analysis.get('higher_highs_lows', {}).get('lower_lows', False),
                    pattern_type=pattern_analysis.get('higher_highs_lows', {}).get('pattern'),
                    breakout_detected=pattern_analysis.get('breakout', {}).get('breakout', False),
                    breakout_direction=pattern_analysis.get('breakout', {}).get('direction'),
                    volume_confirmed=pattern_analysis.get('breakout', {}).get('volume_confirmed', False),
                    is_consolidating=pattern_analysis.get('consolidation', {}).get('consolidating', False),
                    consolidation_range=pattern_analysis.get('consolidation', {}).get('range_pct'),
                    reversal_detected=pattern_analysis.get('reversals', {}).get('reversal_detected', False),
                    reversal_type=pattern_analysis.get('reversals', {}).get('type')
                )
                self.db.add(pattern_model)
            
            self.db.commit()
            logger.info(f"Stored analysis for {symbol}")
            
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
            self.db.rollback()
    
    def generate_trading_signals(
        self,
        symbol: str
    ) -> Dict[str, any]:
        """
        Generate trading signals based on analysis
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with trading signals
        """
        analysis = self.analyze_symbol(symbol, store_results=False)
        
        if 'error' in analysis:
            return analysis
        
        signals = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'overall_signal': 'NEUTRAL',
            'confidence': 0,
            'reasons': [],
            'entry_price': analysis['current_price'],
            'stop_loss': None,
            'take_profit': None
        }
        
        # Collect bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Check indicator signals
        ind_signals = analysis['indicators']['signals']
        
        if ind_signals.get('rsi') == 'OVERSOLD (Bullish)':
            bullish_signals += 1
            signals['reasons'].append('RSI oversold')
        elif ind_signals.get('rsi') == 'OVERBOUGHT (Bearish)':
            bearish_signals += 1
            signals['reasons'].append('RSI overbought')
        
        if ind_signals.get('macd') == 'BULLISH':
            bullish_signals += 1
            signals['reasons'].append('MACD bullish')
        elif ind_signals.get('macd') == 'BEARISH':
            bearish_signals += 1
            signals['reasons'].append('MACD bearish')
        
        if 'BULLISH' in ind_signals.get('trend', ''):
            bullish_signals += 1
            signals['reasons'].append('Bullish trend')
        elif 'BEARISH' in ind_signals.get('trend', ''):
            bearish_signals += 1
            signals['reasons'].append('Bearish trend')
        
        # Check pattern signals
        patterns = analysis['patterns']
        
        if patterns.get('reversals', {}).get('type') == 'BULLISH_REVERSAL':
            bullish_signals += 2
            signals['reasons'].append('Bullish reversal pattern')
        elif patterns.get('reversals', {}).get('type') == 'BEARISH_REVERSAL':
            bearish_signals += 2
            signals['reasons'].append('Bearish reversal pattern')
        
        if patterns.get('breakout', {}).get('direction') == 'UP':
            bullish_signals += 1
            signals['reasons'].append('Upward breakout')
        elif patterns.get('breakout', {}).get('direction') == 'DOWN':
            bearish_signals += 1
            signals['reasons'].append('Downward breakout')
        
        # Check regime
        regime = analysis['regime']
        regime_action = regime.get('recommendation', {}).get('action')
        
        if regime_action == 'BULLISH':
            bullish_signals += 1
            signals['reasons'].append('Bullish market regime')
        elif regime_action == 'BEARISH':
            bearish_signals += 1
            signals['reasons'].append('Bearish market regime')
        
        # Determine overall signal
        total_signals = bullish_signals + bearish_signals
        
        if total_signals > 0:
            if bullish_signals > bearish_signals and bullish_signals >= 3:
                signals['overall_signal'] = 'BULLISH'
                signals['confidence'] = bullish_signals / total_signals
            elif bearish_signals > bullish_signals and bearish_signals >= 3:
                signals['overall_signal'] = 'BEARISH'
                signals['confidence'] = bearish_signals / total_signals
        
        # Calculate stop loss and take profit
        current_price = analysis['current_price']
        atr = analysis['indicators']['values'].get('atr', current_price * 0.02)
        
        if signals['overall_signal'] == 'BULLISH':
            signals['stop_loss'] = current_price - (2 * atr)
            signals['take_profit'] = current_price + (3 * atr)
        elif signals['overall_signal'] == 'BEARISH':
            signals['stop_loss'] = current_price + (2 * atr)
            signals['take_profit'] = current_price - (3 * atr)
        
        return signals
    
    def get_market_summary(
        self,
        symbols: List[str] = ['SPY', 'QQQ']
    ) -> Dict[str, any]:
        """
        Get market summary for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary with market summary
        """
        summary = {
            'timestamp': datetime.utcnow(),
            'symbols': {}
        }
        
        for symbol in symbols:
            analysis = self.analyze_symbol(symbol, store_results=True)
            signals = self.generate_trading_signals(symbol)
            
            summary['symbols'][symbol] = {
                'price': analysis.get('current_price'),
                'trend': analysis.get('regime', {}).get('trend', {}).get('regime'),
                'volatility': analysis.get('regime', {}).get('volatility', {}).get('regime'),
                'signal': signals.get('overall_signal'),
                'confidence': signals.get('confidence'),
                'recommendation': analysis.get('regime', {}).get('recommendation', {})
            }
        
        return summary

