"""
Enhanced Market Regime Detector
Advanced market regime detection with VIX analysis, strategy adaptation, and account-tier responses

Features:
- VIX-based regime classification
- Market structure analysis (Bull, Bear, Sideways)
- Dynamic strategy switching
- Position sizing adjustments
- Correlation monitoring
- Account-tier specific responses
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import yaml
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import asyncio

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    FLASH_CRASH = "flash_crash"
    REGIME_TRANSITION = "regime_transition"
    UNKNOWN = "unknown"

class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    VERY_LOW = "very_low"    # VIX < 12
    LOW = "low"              # VIX 12-15
    NORMAL = "normal"        # VIX 15-25
    HIGH = "high"            # VIX 25-35
    VERY_HIGH = "very_high"  # VIX > 35
    EXTREME = "extreme"      # VIX > 50

@dataclass
class RegimeAnalysis:
    """Market regime analysis result"""
    current_regime: MarketRegime
    volatility_regime: VolatilityRegime
    confidence: float
    vix_level: float
    put_call_ratio: float
    market_structure_score: float
    correlation_breakdown: bool
    regime_duration_days: int
    regime_transition_probability: float
    recommended_strategies: List[str]
    position_size_multiplier: float
    risk_multiplier: float
    timestamp: datetime

@dataclass
class MarketStructure:
    """Market structure analysis"""
    trend_direction: str  # "bullish", "bearish", "neutral"
    support_level: float
    resistance_level: float
    breakout_probability: float
    consolidation_strength: float
    volume_profile: str  # "high", "normal", "low"
    momentum_score: float

class EnhancedRegimeDetector:
    """Enhanced market regime detector with ML and statistical analysis"""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config = self._load_config(config_path)
        self.regime_config = self.config.get('market_regime', {})
        
        # Regime detection parameters
        self.vix_thresholds = {
            'very_low': 12,
            'low': 15,
            'normal': 25,
            'high': 35,
            'very_high': 50
        }
        
        self.regime_switching_threshold = self.regime_config.get('switching_threshold', 0.7)
        self.lookback_period = self.regime_config.get('lookback_period', 60)
        
        # ML models for regime detection
        self.scaler = StandardScaler()
        self.regime_model = None
        self.volatility_model = None
        
        # Historical data storage
        self.market_data = pd.DataFrame()
        self.regime_history = []
        
        # Account tier configurations
        self.account_tiers = self.config.get('account_tiers', {})
        
        logger.info("Enhanced regime detector initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def train_models(self, historical_data: pd.DataFrame):
        """Train ML models for regime detection"""
        try:
            logger.info("Training regime detection models...")
            
            # Prepare features
            features = self._prepare_features(historical_data)
            
            if len(features) < 100:
                logger.warning("Insufficient data for model training")
                return
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Train regime detection model (Gaussian Mixture)
            self.regime_model = GaussianMixture(
                n_components=4,  # Bull, Bear, Sideways, High Vol
                covariance_type='full',
                random_state=42
            )
            self.regime_model.fit(scaled_features)
            
            # Train volatility regime model (KMeans)
            vix_features = features[['vix', 'vix_change', 'vix_volatility']].values
            self.volatility_model = KMeans(
                n_clusters=5,  # Very Low, Low, Normal, High, Very High
                random_state=42
            )
            self.volatility_model.fit(vix_features)
            
            logger.info("Regime detection models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        # VIX features
        features['vix'] = data.get('vix', 20)
        features['vix_change'] = data.get('vix', 20).pct_change()
        features['vix_volatility'] = data.get('vix', 20).rolling(20).std()
        
        # Market structure features
        features['spy_return'] = data.get('spy_close', 400).pct_change()
        features['spy_volatility'] = data.get('spy_close', 400).rolling(20).std()
        features['spy_momentum'] = data.get('spy_close', 400).pct_change(5)
        
        # Put/Call ratio
        features['put_call_ratio'] = data.get('put_call_ratio', 1.0)
        features['put_call_change'] = data.get('put_call_ratio', 1.0).pct_change()
        
        # Volume features
        features['volume_ratio'] = data.get('volume', 1000000) / data.get('volume', 1000000).rolling(20).mean()
        
        # Correlation features
        if 'qqq_close' in data.columns:
            features['spy_qqq_correlation'] = data['spy_close'].rolling(20).corr(data['qqq_close'])
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data.get('spy_close', 400))
        features['macd'] = self._calculate_macd(data.get('spy_close', 400))
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    async def analyze_current_regime(self, market_data: Dict[str, Any]) -> RegimeAnalysis:
        """Analyze current market regime"""
        try:
            # Extract current market data
            current_vix = market_data.get('vix', 20)
            current_spy_price = market_data.get('spy_price', 400)
            put_call_ratio = market_data.get('put_call_ratio', 1.0)
            volume = market_data.get('volume', 1000000)
            
            # Determine volatility regime
            volatility_regime = self._classify_volatility_regime(current_vix)
            
            # Analyze market structure
            market_structure = self._analyze_market_structure(market_data)
            
            # Determine primary regime
            current_regime = self._classify_market_regime(
                current_vix, market_structure, put_call_ratio
            )
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(
                current_vix, market_structure, put_call_ratio
            )
            
            # Check for regime transition
            transition_prob = self._calculate_transition_probability(current_regime)
            
            # Get regime duration
            regime_duration = self._get_regime_duration(current_regime)
            
            # Check correlation breakdown
            correlation_breakdown = self._check_correlation_breakdown(market_data)
            
            # Get recommendations
            recommendations = self._get_strategy_recommendations(
                current_regime, volatility_regime, market_structure
            )
            
            # Calculate position sizing
            position_multiplier = self._calculate_position_multiplier(
                current_regime, volatility_regime
            )
            
            # Calculate risk multiplier
            risk_multiplier = self._calculate_risk_multiplier(
                current_regime, volatility_regime
            )
            
            analysis = RegimeAnalysis(
                current_regime=current_regime,
                volatility_regime=volatility_regime,
                confidence=confidence,
                vix_level=current_vix,
                put_call_ratio=put_call_ratio,
                market_structure_score=market_structure.momentum_score,
                correlation_breakdown=correlation_breakdown,
                regime_duration_days=regime_duration,
                regime_transition_probability=transition_prob,
                recommended_strategies=recommendations,
                position_size_multiplier=position_multiplier,
                risk_multiplier=risk_multiplier,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.regime_history.append(analysis)
            
            # Keep only recent history
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            logger.info(f"Regime analysis: {current_regime.value} (confidence: {confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing regime: {e}")
            return self._get_default_regime_analysis()
    
    def _classify_volatility_regime(self, vix: float) -> VolatilityRegime:
        """Classify volatility regime based on VIX"""
        if vix < 12:
            return VolatilityRegime.VERY_LOW
        elif vix < 15:
            return VolatilityRegime.LOW
        elif vix < 25:
            return VolatilityRegime.NORMAL
        elif vix < 35:
            return VolatilityRegime.HIGH
        elif vix < 50:
            return VolatilityRegime.VERY_HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _analyze_market_structure(self, market_data: Dict[str, Any]) -> MarketStructure:
        """Analyze market structure"""
        try:
            # Get price data
            spy_prices = market_data.get('spy_prices', [400])
            volume_data = market_data.get('volume_data', [1000000])
            
            # Calculate trend
            if len(spy_prices) >= 20:
                short_ma = np.mean(spy_prices[-10:])
                long_ma = np.mean(spy_prices[-20:])
                trend_direction = "bullish" if short_ma > long_ma else "bearish"
            else:
                trend_direction = "neutral"
            
            # Calculate support and resistance
            recent_prices = spy_prices[-20:] if len(spy_prices) >= 20 else spy_prices
            support_level = np.percentile(recent_prices, 20)
            resistance_level = np.percentile(recent_prices, 80)
            
            # Calculate breakout probability
            current_price = spy_prices[-1] if spy_prices else 400
            price_range = resistance_level - support_level
            breakout_probability = min(1.0, abs(current_price - (support_level + resistance_level) / 2) / (price_range / 2))
            
            # Calculate consolidation strength
            price_std = np.std(recent_prices)
            consolidation_strength = 1.0 - min(1.0, price_std / (current_price * 0.02))
            
            # Analyze volume
            avg_volume = np.mean(volume_data[-10:]) if len(volume_data) >= 10 else volume_data[0] if volume_data else 1000000
            current_volume = volume_data[-1] if volume_data else 1000000
            volume_profile = "high" if current_volume > avg_volume * 1.5 else "low" if current_volume < avg_volume * 0.5 else "normal"
            
            # Calculate momentum score
            momentum_score = 0.5  # Default neutral
            if len(spy_prices) >= 5:
                recent_returns = [spy_prices[i] / spy_prices[i-1] - 1 for i in range(1, min(6, len(spy_prices)))]
                momentum_score = np.mean(recent_returns) * 100 + 0.5
            
            return MarketStructure(
                trend_direction=trend_direction,
                support_level=support_level,
                resistance_level=resistance_level,
                breakout_probability=breakout_probability,
                consolidation_strength=consolidation_strength,
                volume_profile=volume_profile,
                momentum_score=momentum_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return MarketStructure(
                trend_direction="neutral",
                support_level=400,
                resistance_level=420,
                breakout_probability=0.5,
                consolidation_strength=0.5,
                volume_profile="normal",
                momentum_score=0.5
            )
    
    def _classify_market_regime(self, vix: float, market_structure: MarketStructure, put_call_ratio: float) -> MarketRegime:
        """Classify primary market regime"""
        # High volatility regime
        if vix > 35:
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        if vix < 15:
            return MarketRegime.LOW_VOLATILITY
        
        # Flash crash detection
        if vix > 50 or put_call_ratio > 2.0:
            return MarketRegime.FLASH_CRASH
        
        # Market structure-based classification
        if market_structure.trend_direction == "bullish" and vix < 25:
            return MarketRegime.BULL_MARKET
        elif market_structure.trend_direction == "bearish" and vix > 20:
            return MarketRegime.BEAR_MARKET
        elif market_structure.consolidation_strength > 0.7:
            return MarketRegime.SIDEWAYS_MARKET
        else:
            return MarketRegime.NORMAL_VOLATILITY
    
    def _calculate_regime_confidence(self, vix: float, market_structure: MarketStructure, put_call_ratio: float) -> float:
        """Calculate confidence in regime classification"""
        confidence = 0.5  # Base confidence
        
        # VIX confidence
        if 15 <= vix <= 25:
            confidence += 0.2
        elif vix > 35 or vix < 12:
            confidence += 0.3
        
        # Market structure confidence
        if market_structure.consolidation_strength > 0.8:
            confidence += 0.2
        
        # Put/call ratio confidence
        if 0.7 <= put_call_ratio <= 1.3:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_transition_probability(self, current_regime: MarketRegime) -> float:
        """Calculate probability of regime transition"""
        if not self.regime_history:
            return 0.5
        
        # Look at recent regime changes
        recent_regimes = [r.current_regime for r in self.regime_history[-10:]]
        regime_changes = sum(1 for i in range(1, len(recent_regimes)) if recent_regimes[i] != recent_regimes[i-1])
        
        # Higher change rate = higher transition probability
        transition_prob = min(1.0, regime_changes / len(recent_regimes))
        
        return transition_prob
    
    def _get_regime_duration(self, current_regime: MarketRegime) -> int:
        """Get duration of current regime in days"""
        if not self.regime_history:
            return 0
        
        duration = 0
        for analysis in reversed(self.regime_history):
            if analysis.current_regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _check_correlation_breakdown(self, market_data: Dict[str, Any]) -> bool:
        """Check for correlation breakdown between assets"""
        try:
            spy_prices = market_data.get('spy_prices', [])
            qqq_prices = market_data.get('qqq_prices', [])
            
            if len(spy_prices) < 20 or len(qqq_prices) < 20:
                return False
            
            # Calculate rolling correlation
            spy_returns = pd.Series(spy_prices).pct_change().dropna()
            qqq_returns = pd.Series(qqq_prices).pct_change().dropna()
            
            min_length = min(len(spy_returns), len(qqq_returns), 20)
            spy_returns = spy_returns[-min_length:]
            qqq_returns = qqq_returns[-min_length:]
            
            correlation = spy_returns.corr(qqq_returns)
            
            # Correlation breakdown if correlation < 0.5
            return correlation < 0.5
            
        except Exception as e:
            logger.error(f"Error checking correlation: {e}")
            return False
    
    def _get_strategy_recommendations(self, regime: MarketRegime, vol_regime: VolatilityRegime, market_structure: MarketStructure) -> List[str]:
        """Get strategy recommendations based on regime"""
        recommendations = []
        
        if regime == MarketRegime.BULL_MARKET:
            recommendations.extend([
                "bull_put_spread",
                "cash_secured_put",
                "covered_call"
            ])
        elif regime == MarketRegime.BEAR_MARKET:
            recommendations.extend([
                "bear_call_spread",
                "protective_put",
                "cash_secured_put"  # Wait for better prices
            ])
        elif regime == MarketRegime.SIDEWAYS_MARKET:
            recommendations.extend([
                "iron_condor",
                "calendar_spread",
                "butterfly_spread"
            ])
        elif regime == MarketRegime.HIGH_VOLATILITY:
            recommendations.extend([
                "iron_condor",  # Sell volatility
                "strangle",     # Sell volatility
                "straddle"      # Sell volatility
            ])
        elif regime == MarketRegime.LOW_VOLATILITY:
            recommendations.extend([
                "long_straddle",
                "long_strangle",
                "calendar_spread"
            ])
        
        # Adjust based on volatility regime
        if vol_regime in [VolatilityRegime.VERY_HIGH, VolatilityRegime.EXTREME]:
            # Reduce position sizes
            recommendations = [f"reduced_size_{rec}" for rec in recommendations]
        
        return recommendations
    
    def _calculate_position_multiplier(self, regime: MarketRegime, vol_regime: VolatilityRegime) -> float:
        """Calculate position size multiplier based on regime"""
        multiplier = 1.0
        
        # Volatility adjustments
        if vol_regime == VolatilityRegime.VERY_LOW:
            multiplier *= 1.2  # Increase size in low vol
        elif vol_regime == VolatilityRegime.LOW:
            multiplier *= 1.1
        elif vol_regime == VolatilityRegime.HIGH:
            multiplier *= 0.8
        elif vol_regime in [VolatilityRegime.VERY_HIGH, VolatilityRegime.EXTREME]:
            multiplier *= 0.5  # Reduce size significantly in high vol
        
        # Regime adjustments
        if regime == MarketRegime.FLASH_CRASH:
            multiplier *= 0.3  # Minimal positions during flash crash
        elif regime == MarketRegime.HIGH_VOLATILITY:
            multiplier *= 0.7
        
        return max(0.1, multiplier)  # Minimum 10% position size
    
    def _calculate_risk_multiplier(self, regime: MarketRegime, vol_regime: VolatilityRegime) -> float:
        """Calculate risk multiplier based on regime"""
        multiplier = 1.0
        
        # Volatility adjustments
        if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH, VolatilityRegime.EXTREME]:
            multiplier *= 1.5  # Increase risk limits in high vol
        
        # Regime adjustments
        if regime == MarketRegime.FLASH_CRASH:
            multiplier *= 2.0  # Double risk limits during flash crash
        elif regime == MarketRegime.HIGH_VOLATILITY:
            multiplier *= 1.3
        
        return multiplier
    
    def _get_default_regime_analysis(self) -> RegimeAnalysis:
        """Get default regime analysis when errors occur"""
        return RegimeAnalysis(
            current_regime=MarketRegime.NORMAL_VOLATILITY,
            volatility_regime=VolatilityRegime.NORMAL,
            confidence=0.5,
            vix_level=20.0,
            put_call_ratio=1.0,
            market_structure_score=0.5,
            correlation_breakdown=False,
            regime_duration_days=0,
            regime_transition_probability=0.5,
            recommended_strategies=["bull_put_spread"],
            position_size_multiplier=1.0,
            risk_multiplier=1.0,
            timestamp=datetime.now()
        )
    
    def get_regime_history(self, days: int = 30) -> List[RegimeAnalysis]:
        """Get regime history for specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [r for r in self.regime_history if r.timestamp >= cutoff_date]
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime statistics"""
        if not self.regime_history:
            return {}
        
        regimes = [r.current_regime.value for r in self.regime_history]
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        
        avg_confidence = np.mean([r.confidence for r in self.regime_history])
        avg_vix = np.mean([r.vix_level for r in self.regime_history])
        
        return {
            'regime_distribution': regime_counts,
            'average_confidence': avg_confidence,
            'average_vix': avg_vix,
            'total_observations': len(self.regime_history),
            'most_common_regime': max(regime_counts, key=regime_counts.get) if regime_counts else None
        }

# Example usage and testing
async def main():
    """Test the enhanced regime detector"""
    detector = EnhancedRegimeDetector()
    
    # Mock market data
    market_data = {
        'vix': 18.5,
        'spy_price': 425.50,
        'put_call_ratio': 1.2,
        'volume': 1500000,
        'spy_prices': [420, 422, 425, 424, 426],
        'qqq_prices': [380, 382, 385, 384, 386],
        'volume_data': [1200000, 1300000, 1400000, 1350000, 1500000]
    }
    
    # Analyze current regime
    analysis = await detector.analyze_current_regime(market_data)
    
    print(f"Current Regime: {analysis.current_regime.value}")
    print(f"Volatility Regime: {analysis.volatility_regime.value}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"VIX Level: {analysis.vix_level}")
    print(f"Position Multiplier: {analysis.position_size_multiplier:.2f}")
    print(f"Risk Multiplier: {analysis.risk_multiplier:.2f}")
    print(f"Recommended Strategies: {analysis.recommended_strategies}")

if __name__ == "__main__":
    asyncio.run(main())
