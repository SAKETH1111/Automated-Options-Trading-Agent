"""
Correlation Analyzer Module
Analyze correlations between positions for diversification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import IndexTickData


class CorrelationAnalyzer:
    """
    Analyze correlations between symbols
    Ensure portfolio diversification
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize correlation analyzer
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.correlation_cache = {}
        self.cache_ttl_hours = 24
        
        logger.info("Correlation Analyzer initialized")
    
    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        lookback_days: int = 30
    ) -> Optional[float]:
        """
        Calculate correlation between two symbols
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            lookback_days: Days to look back
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        try:
            # Check cache
            cache_key = f"{symbol1}_{symbol2}_{lookback_days}"
            if cache_key in self.correlation_cache:
                cached_time, cached_corr = self.correlation_cache[cache_key]
                if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl_hours * 3600:
                    return cached_corr
            
            # Get price data for both symbols
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            
            data1 = self._get_price_data(symbol1, cutoff)
            data2 = self._get_price_data(symbol2, cutoff)
            
            if len(data1) < 20 or len(data2) < 20:
                logger.warning(f"Insufficient data for correlation: {symbol1}, {symbol2}")
                return None
            
            # Align data by timestamp (simplified - use same length)
            min_len = min(len(data1), len(data2))
            prices1 = data1[-min_len:]
            prices2 = data2[-min_len:]
            
            # Calculate returns
            returns1 = np.diff(np.log(prices1))
            returns2 = np.diff(np.log(prices2))
            
            # Calculate correlation
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            # Cache result
            self.correlation_cache[cache_key] = (datetime.utcnow(), correlation)
            
            logger.debug(f"Correlation {symbol1}-{symbol2}: {correlation:.3f}")
            
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None
    
    def analyze_portfolio_correlation(
        self,
        positions: List[Dict]
    ) -> Dict:
        """
        Analyze correlation across portfolio
        
        Args:
            positions: List of open positions
            
        Returns:
            Correlation analysis
        """
        try:
            # Get unique symbols
            symbols = list(set(p['symbol'] for p in positions))
            
            if len(symbols) < 2:
                return {
                    'diversified': True,
                    'message': 'Only one symbol in portfolio'
                }
            
            # Calculate pairwise correlations
            correlations = {}
            high_correlations = []
            
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    corr = self.calculate_correlation(sym1, sym2)
                    
                    if corr is not None:
                        pair = f"{sym1}-{sym2}"
                        correlations[pair] = corr
                        
                        # Flag high correlations
                        if abs(corr) > 0.8:
                            high_correlations.append({
                                'pair': pair,
                                'correlation': corr,
                                'warning': 'High correlation - limited diversification'
                            })
            
            # Calculate average correlation
            avg_correlation = np.mean(list(correlations.values())) if correlations else 0
            
            # Determine diversification quality
            if avg_correlation > 0.7:
                diversification = 'POOR'
                message = 'Portfolio highly correlated - limited diversification'
            elif avg_correlation > 0.5:
                diversification = 'MODERATE'
                message = 'Portfolio moderately correlated'
            else:
                diversification = 'GOOD'
                message = 'Portfolio well diversified'
            
            return {
                'diversified': diversification in ['GOOD', 'MODERATE'],
                'diversification_quality': diversification,
                'message': message,
                'symbols': symbols,
                'correlations': correlations,
                'avg_correlation': avg_correlation,
                'high_correlations': high_correlations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio correlation: {e}")
            return {'diversified': True, 'error': str(e)}
    
    def check_position_correlation(
        self,
        new_symbol: str,
        existing_positions: List[Dict]
    ) -> Dict:
        """
        Check if new position adds diversification
        
        Args:
            new_symbol: Symbol of proposed position
            existing_positions: Current open positions
            
        Returns:
            Correlation check result
        """
        try:
            existing_symbols = list(set(p['symbol'] for p in existing_positions))
            
            if not existing_symbols:
                return {
                    'adds_diversification': True,
                    'message': 'First position'
                }
            
            # Check correlation with existing symbols
            correlations = []
            for existing_symbol in existing_symbols:
                corr = self.calculate_correlation(new_symbol, existing_symbol)
                if corr is not None:
                    correlations.append({
                        'symbol': existing_symbol,
                        'correlation': corr
                    })
            
            if not correlations:
                return {
                    'adds_diversification': True,
                    'message': 'Could not calculate correlations'
                }
            
            # Check if any high correlations
            max_corr = max(abs(c['correlation']) for c in correlations)
            
            if max_corr > 0.8:
                return {
                    'adds_diversification': False,
                    'message': f'High correlation ({max_corr:.2f}) with existing positions',
                    'correlations': correlations
                }
            elif max_corr > 0.6:
                return {
                    'adds_diversification': True,
                    'message': f'Moderate correlation ({max_corr:.2f}) - acceptable',
                    'warning': 'Consider diversifying further',
                    'correlations': correlations
                }
            else:
                return {
                    'adds_diversification': True,
                    'message': f'Low correlation ({max_corr:.2f}) - good diversification',
                    'correlations': correlations
                }
            
        except Exception as e:
            logger.error(f"Error checking position correlation: {e}")
            return {'adds_diversification': True, 'error': str(e)}
    
    def _get_price_data(
        self,
        symbol: str,
        start_date: datetime
    ) -> List[float]:
        """Get price data for a symbol"""
        try:
            with self.db.get_session() as session:
                data = session.query(IndexTickData).filter(
                    IndexTickData.symbol == symbol,
                    IndexTickData.timestamp >= start_date
                ).order_by(IndexTickData.timestamp.asc()).all()
            
            return [d.price for d in data]
            
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return []

