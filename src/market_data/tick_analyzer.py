"""Utilities for analyzing historical tick data"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from sqlalchemy import func

from src.database.models import IndexTickData
from src.database.session import get_db


class TickDataAnalyzer:
    """Analyze historical tick data for patterns and insights"""
    
    def __init__(self):
        self.db = get_db()
    
    def get_tick_data_df(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Get tick data as pandas DataFrame
        
        Args:
            symbol: Symbol to query
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            DataFrame with tick data
        """
        try:
            with self.db.get_session() as session:
                ticks = session.query(IndexTickData)\
                    .filter(
                        IndexTickData.symbol == symbol,
                        IndexTickData.timestamp >= start_time,
                        IndexTickData.timestamp <= end_time
                    )\
                    .order_by(IndexTickData.timestamp.asc())\
                    .all()
                
                if not ticks:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for tick in ticks:
                    data.append({
                        'timestamp': tick.timestamp,
                        'symbol': tick.symbol,
                        'price': tick.price,
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'spread': tick.spread,
                        'spread_pct': tick.spread_pct,
                        'volume': tick.volume,
                        'vix': tick.vix,
                        'price_change': tick.price_change,
                        'price_change_pct': tick.price_change_pct,
                        'market_state': tick.market_state,
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                
                return df
        
        except Exception as e:
            logger.error(f"Error getting tick data as DataFrame: {e}")
            return pd.DataFrame()
    
    def get_minute_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Aggregate tick data into minute bars
        
        Args:
            symbol: Symbol to query
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            DataFrame with OHLCV minute bars
        """
        try:
            df = self.get_tick_data_df(symbol, start_time, end_time)
            
            if df.empty:
                return pd.DataFrame()
            
            # Resample to 1-minute bars
            bars = df.resample('1min').agg({
                'price': ['first', 'max', 'min', 'last', 'count'],
                'volume': 'last',
                'vix': 'mean',
                'spread_pct': 'mean',
            })
            
            # Flatten column names
            bars.columns = ['open', 'high', 'low', 'close', 'tick_count', 
                          'volume', 'avg_vix', 'avg_spread_pct']
            
            # Calculate minute returns
            bars['returns'] = bars['close'].pct_change()
            
            # Drop rows with no ticks
            bars = bars[bars['tick_count'] > 0]
            
            return bars
        
        except Exception as e:
            logger.error(f"Error creating minute bars: {e}")
            return pd.DataFrame()
    
    def calculate_volatility_profile(
        self,
        symbol: str,
        date: datetime
    ) -> Dict:
        """
        Calculate intraday volatility profile
        
        Args:
            symbol: Symbol to analyze
            date: Trading day to analyze
        
        Returns:
            Dict with volatility statistics by hour
        """
        try:
            start_time = date.replace(hour=9, minute=30, second=0, microsecond=0)
            end_time = date.replace(hour=16, minute=0, second=0, microsecond=0)
            
            df = self.get_tick_data_df(symbol, start_time, end_time)
            
            if df.empty:
                return {}
            
            # Group by hour
            df['hour'] = df.index.hour
            
            hourly_stats = df.groupby('hour').agg({
                'price': ['std', 'mean', 'min', 'max'],
                'price_change_pct': ['std', 'mean'],
                'spread_pct': 'mean',
            })
            
            hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns.values]
            
            return hourly_stats.to_dict('index')
        
        except Exception as e:
            logger.error(f"Error calculating volatility profile: {e}")
            return {}
    
    def find_price_reversals(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        threshold_pct: float = 0.5
    ) -> List[Dict]:
        """
        Find significant price reversals in tick data
        
        Args:
            symbol: Symbol to analyze
            start_time: Start timestamp
            end_time: End timestamp
            threshold_pct: Minimum price change % to consider
        
        Returns:
            List of reversal events
        """
        try:
            df = self.get_tick_data_df(symbol, start_time, end_time)
            
            if df.empty:
                return []
            
            # Find local peaks and troughs
            df['returns'] = df['price'].pct_change() * 100
            df['direction_change'] = (df['returns'].shift(1) * df['returns']) < 0
            
            reversals = []
            for idx, row in df[df['direction_change']].iterrows():
                if abs(row['returns']) >= threshold_pct:
                    reversals.append({
                        'timestamp': idx,
                        'price': row['price'],
                        'returns': row['returns'],
                        'vix': row['vix'],
                    })
            
            return reversals
        
        except Exception as e:
            logger.error(f"Error finding price reversals: {e}")
            return []
    
    def get_correlation_with_vix(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[float]:
        """
        Calculate correlation between price changes and VIX
        
        Args:
            symbol: Symbol to analyze
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            Correlation coefficient (-1 to 1)
        """
        try:
            df = self.get_tick_data_df(symbol, start_time, end_time)
            
            if df.empty or df['vix'].isna().all():
                return None
            
            # Remove rows with missing VIX
            df_clean = df.dropna(subset=['vix', 'price_change_pct'])
            
            if len(df_clean) < 10:
                return None
            
            correlation = df_clean['price_change_pct'].corr(df_clean['vix'])
            
            return float(correlation)
        
        except Exception as e:
            logger.error(f"Error calculating VIX correlation: {e}")
            return None
    
    def get_daily_statistics(
        self,
        symbol: str,
        date: datetime
    ) -> Dict:
        """
        Get comprehensive daily statistics
        
        Args:
            symbol: Symbol to analyze
            date: Trading day to analyze
        
        Returns:
            Dict with daily statistics
        """
        try:
            start_time = date.replace(hour=9, minute=30, second=0, microsecond=0)
            end_time = date.replace(hour=16, minute=0, second=0, microsecond=0)
            
            df = self.get_tick_data_df(symbol, start_time, end_time)
            
            if df.empty:
                return {}
            
            stats = {
                'date': date.date().isoformat(),
                'symbol': symbol,
                'tick_count': len(df),
                'open': float(df['price'].iloc[0]),
                'high': float(df['price'].max()),
                'low': float(df['price'].min()),
                'close': float(df['price'].iloc[-1]),
                'range': float(df['price'].max() - df['price'].min()),
                'range_pct': float((df['price'].max() - df['price'].min()) / df['price'].iloc[0] * 100),
                'avg_spread_pct': float(df['spread_pct'].mean()),
                'max_spread_pct': float(df['spread_pct'].max()),
                'volatility': float(df['price_change_pct'].std()),
                'avg_vix': float(df['vix'].mean()) if not df['vix'].isna().all() else None,
                'total_price_moves': int((df['price_change'] != 0).sum()),
                'positive_moves': int((df['price_change'] > 0).sum()),
                'negative_moves': int((df['price_change'] < 0).sum()),
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error calculating daily statistics: {e}")
            return {}
    
    def get_data_availability(self, symbol: str, days: int = 7) -> Dict:
        """
        Check data availability for recent days
        
        Args:
            symbol: Symbol to check
            days: Number of recent days to check
        
        Returns:
            Dict with availability statistics
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            with self.db.get_session() as session:
                # Count ticks per day
                daily_counts = session.query(
                    func.date(IndexTickData.timestamp).label('date'),
                    func.count(IndexTickData.tick_id).label('count')
                )\
                .filter(
                    IndexTickData.symbol == symbol,
                    IndexTickData.timestamp >= start_time,
                    IndexTickData.timestamp <= end_time
                )\
                .group_by(func.date(IndexTickData.timestamp))\
                .all()
                
                availability = {
                    'symbol': symbol,
                    'period_days': days,
                    'daily_counts': {str(date): count for date, count in daily_counts},
                    'total_ticks': sum(count for _, count in daily_counts),
                    'days_with_data': len(daily_counts),
                }
                
                return availability
        
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return {}
    
    def export_to_csv(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        output_path: str
    ) -> bool:
        """
        Export tick data to CSV file
        
        Args:
            symbol: Symbol to export
            start_time: Start timestamp
            end_time: End timestamp
            output_path: Path to save CSV file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            df = self.get_tick_data_df(symbol, start_time, end_time)
            
            if df.empty:
                logger.warning(f"No data to export for {symbol}")
                return False
            
            df.to_csv(output_path)
            logger.info(f"Exported {len(df)} ticks to {output_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {e}")
            return False
    
    def clean_old_data(self, days_to_keep: int = 30) -> int:
        """
        Delete tick data older than specified days
        
        Args:
            days_to_keep: Number of days to retain
        
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.db.get_session() as session:
                deleted_count = session.query(IndexTickData)\
                    .filter(IndexTickData.timestamp < cutoff_date)\
                    .delete()
                
                session.commit()
                
                logger.info(f"Deleted {deleted_count} old tick records")
                
                return deleted_count
        
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
            return 0

