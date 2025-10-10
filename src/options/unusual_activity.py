"""
Unusual Options Activity Detector Module
Detect unusual volume, sweeps, and block trades
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from src.database.models import OptionsChain, UnusualOptionsActivity


class UnusualActivityDetector:
    """
    Detect unusual options activity
    Identifies: Unusual volume, sweeps, block trades, large premium flows
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize unusual activity detector
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        logger.info("Unusual Activity Detector initialized")
    
    def detect_unusual_activity(
        self,
        symbol: str,
        volume_threshold: float = 3.0,
        min_premium: float = 100000
    ) -> List[Dict]:
        """
        Detect unusual options activity
        
        Args:
            symbol: Symbol to analyze
            volume_threshold: Volume ratio threshold (current / average)
            min_premium: Minimum premium spent ($)
            
        Returns:
            List of unusual activity events
        """
        unusual_activities = []
        
        try:
            logger.info(f"Detecting unusual activity for {symbol}")
            
            # Get recent options data
            recent_options = self._get_recent_options(symbol)
            
            if not recent_options:
                logger.warning(f"No recent options data for {symbol}")
                return []
            
            # Check each option for unusual activity
            for opt in recent_options:
                # Calculate metrics
                volume = opt.volume or 0
                open_interest = opt.open_interest or 1  # Avoid division by zero
                
                # Volume/OI ratio
                volume_oi_ratio = volume / open_interest if open_interest > 0 else 0
                
                # Get historical average volume
                avg_volume = self._get_average_volume(
                    symbol,
                    opt.option_symbol,
                    lookback_days=20
                )
                
                if avg_volume is None or avg_volume == 0:
                    avg_volume = volume * 0.3  # Estimate if no history
                
                # Volume ratio
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
                
                # Calculate premium spent
                mid_price = opt.mid_price or 0
                premium_spent = mid_price * volume * 100  # Per contract
                
                # Check for unusual activity
                is_unusual = False
                is_sweep = False
                is_block = False
                
                # Unusual volume check
                if volume_ratio >= volume_threshold:
                    is_unusual = True
                
                # Sweep detection (high volume with tight spread)
                spread_pct = opt.bid_ask_spread_pct or 100
                if volume >= 100 and spread_pct < 5:
                    is_sweep = True
                    is_unusual = True
                
                # Block trade detection (large single trades)
                if volume >= 500 and volume_oi_ratio > 0.5:
                    is_block = True
                    is_unusual = True
                
                # Large premium check
                if premium_spent >= min_premium:
                    is_unusual = True
                
                if is_unusual:
                    # Determine sentiment
                    sentiment = self._determine_sentiment(
                        opt.option_type,
                        opt.delta or 0,
                        opt.moneyness
                    )
                    
                    activity = {
                        'symbol': symbol,
                        'timestamp': datetime.utcnow(),
                        'option_symbol': opt.option_symbol,
                        'option_type': opt.option_type,
                        'strike': opt.strike,
                        'expiration': opt.expiration,
                        'volume': volume,
                        'open_interest': open_interest,
                        'volume_oi_ratio': volume_oi_ratio,
                        'avg_volume_20d': int(avg_volume),
                        'volume_ratio': volume_ratio,
                        'is_unusual_volume': volume_ratio >= volume_threshold,
                        'is_sweep': is_sweep,
                        'is_block_trade': is_block,
                        'premium_spent': premium_spent,
                        'avg_price': mid_price,
                        'sentiment': sentiment,
                        'delta': opt.delta,
                        'implied_volatility': opt.implied_volatility
                    }
                    
                    unusual_activities.append(activity)
            
            # Sort by premium spent (largest first)
            unusual_activities.sort(key=lambda x: x['premium_spent'], reverse=True)
            
            logger.info(f"Found {len(unusual_activities)} unusual activity events for {symbol}")
            
            return unusual_activities
            
        except Exception as e:
            logger.error(f"Error detecting unusual activity: {e}")
            return []
    
    def _get_recent_options(
        self,
        symbol: str,
        hours: int = 1
    ) -> List:
        """Get recent options data"""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            # Get latest data for each option symbol
            subquery = self.db.query(
                OptionsChain.option_symbol,
                func.max(OptionsChain.timestamp).label('max_timestamp')
            ).filter(
                OptionsChain.symbol == symbol,
                OptionsChain.timestamp >= cutoff
            ).group_by(OptionsChain.option_symbol).subquery()
            
            options = self.db.query(OptionsChain).join(
                subquery,
                (OptionsChain.option_symbol == subquery.c.option_symbol) &
                (OptionsChain.timestamp == subquery.c.max_timestamp)
            ).all()
            
            return options
            
        except Exception as e:
            logger.error(f"Error getting recent options: {e}")
            return []
    
    def _get_average_volume(
        self,
        symbol: str,
        option_symbol: str,
        lookback_days: int = 20
    ) -> Optional[float]:
        """Get average volume for an option"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            
            result = self.db.query(
                func.avg(OptionsChain.volume)
            ).filter(
                OptionsChain.symbol == symbol,
                OptionsChain.option_symbol == option_symbol,
                OptionsChain.timestamp >= cutoff,
                OptionsChain.volume.isnot(None)
            ).scalar()
            
            return float(result) if result else None
            
        except Exception as e:
            logger.error(f"Error getting average volume: {e}")
            return None
    
    def _determine_sentiment(
        self,
        option_type: str,
        delta: float,
        moneyness: str
    ) -> str:
        """
        Determine sentiment from option activity
        
        Args:
            option_type: 'CALL' or 'PUT'
            delta: Option delta
            moneyness: 'ITM', 'ATM', or 'OTM'
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        # Buying calls = bullish
        # Buying puts = bearish
        # But need to consider if buying or selling
        
        # Use delta as proxy for direction
        abs_delta = abs(delta)
        
        if option_type == 'CALL':
            # OTM calls with high volume = bullish
            if moneyness == 'OTM' or abs_delta < 0.40:
                return 'bullish'
            # ITM calls might be hedging
            elif moneyness == 'ITM':
                return 'neutral'
            else:
                return 'bullish'
        
        else:  # PUT
            # OTM puts with high volume = bearish
            if moneyness == 'OTM' or abs_delta < 0.40:
                return 'bearish'
            # ITM puts might be protective
            elif moneyness == 'ITM':
                return 'neutral'
            else:
                return 'bearish'
    
    def store_activity(
        self,
        activity: Dict
    ) -> bool:
        """
        Store unusual activity in database
        
        Args:
            activity: Activity dictionary
            
        Returns:
            Success status
        """
        try:
            activity_record = UnusualOptionsActivity(
                symbol=activity['symbol'],
                timestamp=activity['timestamp'],
                option_symbol=activity['option_symbol'],
                option_type=activity['option_type'],
                strike=activity['strike'],
                expiration=activity['expiration'],
                volume=activity['volume'],
                open_interest=activity['open_interest'],
                volume_oi_ratio=activity['volume_oi_ratio'],
                avg_volume_20d=activity['avg_volume_20d'],
                volume_ratio=activity['volume_ratio'],
                is_unusual_volume=activity['is_unusual_volume'],
                is_sweep=activity['is_sweep'],
                is_block_trade=activity['is_block_trade'],
                premium_spent=activity['premium_spent'],
                avg_price=activity['avg_price'],
                sentiment=activity['sentiment'],
                delta=activity.get('delta'),
                implied_volatility=activity.get('implied_volatility')
            )
            
            self.db.add(activity_record)
            self.db.commit()
            
            logger.info(f"Stored unusual activity for {activity['option_symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing unusual activity: {e}")
            self.db.rollback()
            return False
    
    def get_activity_summary(
        self,
        symbol: str,
        hours: int = 24
    ) -> Dict:
        """
        Get summary of unusual activity
        
        Args:
            symbol: Symbol to analyze
            hours: Hours to look back
            
        Returns:
            Summary dictionary
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            activities = self.db.query(UnusualOptionsActivity).filter(
                UnusualOptionsActivity.symbol == symbol,
                UnusualOptionsActivity.timestamp >= cutoff
            ).all()
            
            if not activities:
                return {
                    'symbol': symbol,
                    'total_events': 0,
                    'bullish_events': 0,
                    'bearish_events': 0,
                    'total_premium': 0,
                    'sweeps': 0,
                    'blocks': 0
                }
            
            # Calculate summary
            total_events = len(activities)
            bullish_events = sum(1 for a in activities if a.sentiment == 'bullish')
            bearish_events = sum(1 for a in activities if a.sentiment == 'bearish')
            total_premium = sum(a.premium_spent or 0 for a in activities)
            sweeps = sum(1 for a in activities if a.is_sweep)
            blocks = sum(1 for a in activities if a.is_block_trade)
            
            # Determine overall sentiment
            if bullish_events > bearish_events * 1.5:
                overall_sentiment = 'BULLISH'
            elif bearish_events > bullish_events * 1.5:
                overall_sentiment = 'BEARISH'
            else:
                overall_sentiment = 'MIXED'
            
            return {
                'symbol': symbol,
                'total_events': total_events,
                'bullish_events': bullish_events,
                'bearish_events': bearish_events,
                'overall_sentiment': overall_sentiment,
                'total_premium': total_premium,
                'sweeps': sweeps,
                'blocks': blocks,
                'top_activities': [
                    {
                        'option_symbol': a.option_symbol,
                        'type': a.option_type,
                        'strike': a.strike,
                        'premium': a.premium_spent,
                        'sentiment': a.sentiment,
                        'is_sweep': a.is_sweep,
                        'is_block': a.is_block_trade
                    }
                    for a in sorted(activities, key=lambda x: x.premium_spent or 0, reverse=True)[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting activity summary: {e}")
            return {}
    
    def detect_and_store(
        self,
        symbol: str,
        volume_threshold: float = 3.0
    ) -> int:
        """
        Detect and store unusual activity
        
        Args:
            symbol: Symbol to analyze
            volume_threshold: Volume ratio threshold
            
        Returns:
            Number of activities stored
        """
        activities = self.detect_unusual_activity(symbol, volume_threshold)
        
        stored_count = 0
        for activity in activities:
            if self.store_activity(activity):
                stored_count += 1
        
        return stored_count

