"""Real-time data collector for second-by-second index monitoring"""

import threading
import time
from collections import deque
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Deque

import pytz
from loguru import logger

from src.brokers.alpaca_client import AlpacaClient
from src.database.models import IndexTickData
from src.database.session import get_db


class RealTimeDataCollector:
    """
    Collects second-by-second market data for indexes (SPY, QQQ)
    Stores data in database for future analysis and learning
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        collect_interval: float = 1.0,
        buffer_size: int = 100
    ):
        """
        Initialize real-time data collector
        
        Args:
            symbols: List of symbols to track (default: ['SPY', 'QQQ'])
            alpaca_client: Alpaca client instance
            collect_interval: Collection interval in seconds (default: 1.0)
            buffer_size: Number of ticks to buffer before batch insert
        """
        self.symbols = symbols or ['SPY', 'QQQ']
        self.alpaca = alpaca_client or AlpacaClient()
        self.collect_interval = collect_interval
        self.buffer_size = buffer_size
        self.db = get_db()
        
        # State
        self.is_running = False
        self.collection_thread = None
        self._stop_event = threading.Event()
        
        # Buffered data for batch inserts
        self._tick_buffer: Deque[IndexTickData] = deque(maxlen=buffer_size)
        
        # Previous tick cache for calculating changes
        self._previous_ticks: Dict[str, Dict] = {}
        
        # Rolling window for technical indicators
        self._price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=300) for symbol in self.symbols  # 5 minutes at 1 tick/sec
        }
        
        # Market hours - Note: Market times are still ET (9:30-16:00 ET), but we store in Central Time
        self.market_timezone = pytz.timezone("America/Chicago")  # Central Time (Texas)
        self.market_open = dt_time(8, 30)  # 9:30 ET = 8:30 CT
        self.market_close = dt_time(15, 0)  # 16:00 ET = 15:00 CT
        
        # Statistics
        self.stats = {
            'total_ticks_collected': 0,
            'total_ticks_stored': 0,
            'collection_errors': 0,
            'last_collection_time': None,
        }
        
        logger.info(f"RealTimeDataCollector initialized for {self.symbols}")
    
    def start(self):
        """Start real-time data collection"""
        if self.is_running:
            logger.warning("RealTimeDataCollector is already running")
            return
        
        logger.info(f"Starting real-time data collection for {self.symbols}")
        self.is_running = True
        self._stop_event.clear()
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="RealTimeDataCollector"
        )
        self.collection_thread.start()
        
        logger.info("✅ Real-time data collection started")
    
    def stop(self):
        """Stop real-time data collection"""
        if not self.is_running:
            return
        
        logger.info("Stopping real-time data collection...")
        self.is_running = False
        self._stop_event.set()
        
        # Wait for thread to finish
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        # Flush remaining buffer
        self._flush_buffer()
        
        logger.info("✅ Real-time data collection stopped")
        logger.info(f"Final stats: {self.stats}")
    
    def _collection_loop(self):
        """Main collection loop - runs in separate thread"""
        logger.info("Collection loop started")
        
        while self.is_running and not self._stop_event.is_set():
            try:
                # Check if market is open
                if not self._is_market_open():
                    time.sleep(60)  # Check every minute during closed hours
                    continue
                
                # Collect data for all symbols
                collection_start = time.time()
                
                for symbol in self.symbols:
                    try:
                        self._collect_tick(symbol)
                    except Exception as e:
                        logger.debug(f"Error collecting tick for {symbol}: {e}")
                        self.stats['collection_errors'] += 1
                
                # Update stats (naive datetime for consistency)
                self.stats['last_collection_time'] = datetime.now(self.market_timezone).replace(tzinfo=None)
                
                # Flush buffer if full
                if len(self._tick_buffer) >= self.buffer_size:
                    self._flush_buffer()
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - collection_start
                sleep_time = max(0, self.collect_interval - elapsed)
                
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self.stats['collection_errors'] += 1
                time.sleep(1)
        
        logger.info("Collection loop ended")
    
    def _collect_tick(self, symbol: str):
        """Collect a single tick for a symbol"""
        try:
            # Get snapshot from Alpaca
            snapshot = self.alpaca.get_stock_snapshot(symbol)
            
            if not snapshot:
                return
            
            # Extract data
            latest_trade = snapshot.get('latest_trade', {})
            latest_quote = snapshot.get('latest_quote', {})
            daily_bar = snapshot.get('daily_bar', {})
            
            if not latest_trade or not latest_quote:
                return
            
            price = latest_trade.get('price', 0)
            if price <= 0:
                return
            
            bid = latest_quote.get('bid', 0)
            ask = latest_quote.get('ask', 0)
            spread = ask - bid if (bid and ask) else 0
            spread_pct = (spread / price * 100) if price > 0 else 0
            
            # Calculate change from previous tick
            price_change = 0
            price_change_pct = 0
            if symbol in self._previous_ticks:
                prev_price = self._previous_ticks[symbol].get('price', price)
                price_change = price - prev_price
                price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
            
            # Update price history
            self._price_history[symbol].append(price)
            
            # Calculate simple moving averages
            sma_5 = self._calculate_sma(symbol, 5)
            sma_60 = self._calculate_sma(symbol, 60)
            
            # Get VIX if available
            vix = None
            if symbol in ['SPY', 'QQQ']:
                try:
                    vix_snapshot = self.alpaca.get_stock_snapshot('VIX')
                    if vix_snapshot and vix_snapshot.get('latest_trade'):
                        vix = vix_snapshot['latest_trade'].get('price')
                except:
                    pass
            
            # Determine market state
            market_state = self._get_market_state()
            
            # Create tick data object
            # Use naive datetime (without tzinfo) so SQLite stores Central Time directly
            current_time_ct = datetime.now(self.market_timezone).replace(tzinfo=None)
            tick_data = IndexTickData(
                symbol=symbol,
                timestamp=current_time_ct,
                price=price,
                bid=bid,
                ask=ask,
                bid_size=latest_quote.get('bid_size'),
                ask_size=latest_quote.get('ask_size'),
                spread=spread,
                spread_pct=spread_pct,
                volume=daily_bar.get('volume', 0),
                last_trade_size=latest_trade.get('size'),
                vix=vix,
                sma_5=sma_5,
                sma_60=sma_60,
                price_change=price_change,
                price_change_pct=price_change_pct,
                market_state=market_state,
            )
            
            # Add to buffer
            self._tick_buffer.append(tick_data)
            
            # Update previous tick (naive datetime)
            self._previous_ticks[symbol] = {
                'price': price,
                'timestamp': datetime.now(self.market_timezone).replace(tzinfo=None)
            }
            
            # Update stats
            self.stats['total_ticks_collected'] += 1
            
            logger.debug(f"Collected tick: {symbol} @ ${price:.2f} (Δ {price_change_pct:+.3f}%)")
        
        except Exception as e:
            logger.debug(f"Error collecting tick for {symbol}: {e}")
            raise
    
    def _flush_buffer(self):
        """Flush buffered ticks to database"""
        if not self._tick_buffer:
            return
        
        try:
            ticks_to_insert = list(self._tick_buffer)
            self._tick_buffer.clear()
            
            with self.db.get_session() as session:
                session.bulk_save_objects(ticks_to_insert)
                session.commit()
            
            self.stats['total_ticks_stored'] += len(ticks_to_insert)
            logger.debug(f"Flushed {len(ticks_to_insert)} ticks to database")
        
        except Exception as e:
            logger.error(f"Error flushing buffer to database: {e}")
    
    def _calculate_sma(self, symbol: str, period: int) -> Optional[float]:
        """Calculate simple moving average"""
        try:
            prices = list(self._price_history[symbol])
            if len(prices) < period:
                return None
            
            return sum(prices[-period:]) / period
        
        except Exception:
            return None
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now(self.market_timezone)
            current_time = now.time()
            
            # Check if weekday
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Check market hours
            return self.market_open <= current_time <= self.market_close
        
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def _get_market_state(self) -> str:
        """Get current market state"""
        try:
            now = datetime.now(self.market_timezone)
            current_time = now.time()
            
            if now.weekday() >= 5:
                return "closed"
            
            pre_market_open = dt_time(3, 0)  # 4:00 ET = 3:00 CT
            after_hours_close = dt_time(19, 0)  # 20:00 ET = 19:00 CT
            
            if current_time < pre_market_open:
                return "closed"
            elif current_time < self.market_open:
                return "pre_market"
            elif current_time <= self.market_close:
                return "open"
            elif current_time < after_hours_close:
                return "after_hours"
            else:
                return "closed"
        
        except Exception:
            return "unknown"
    
    def get_recent_ticks(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[IndexTickData]:
        """
        Get recent ticks from database
        
        Args:
            symbol: Symbol to query
            limit: Maximum number of ticks to return
        
        Returns:
            List of IndexTickData objects
        """
        try:
            with self.db.get_session() as session:
                ticks = session.query(IndexTickData)\
                    .filter(IndexTickData.symbol == symbol)\
                    .order_by(IndexTickData.timestamp.desc())\
                    .limit(limit)\
                    .all()
                
                return list(reversed(ticks))
        
        except Exception as e:
            logger.error(f"Error querying recent ticks: {e}")
            return []
    
    def get_ticks_in_timerange(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[IndexTickData]:
        """
        Get ticks within a time range
        
        Args:
            symbol: Symbol to query
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            List of IndexTickData objects
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
                
                return ticks
        
        except Exception as e:
            logger.error(f"Error querying ticks in timerange: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'symbols': self.symbols,
            'buffer_size': len(self._tick_buffer),
            'collect_interval': self.collect_interval,
        }

