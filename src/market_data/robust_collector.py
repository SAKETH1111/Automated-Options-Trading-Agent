"""Robust real-time data collector with retry logic, validation, and monitoring"""

import json
import threading
import time
from collections import deque
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Deque

import pytz
from loguru import logger

from src.brokers.alpaca_client import AlpacaClient
from src.database.models import IndexTickData
from src.database.session import get_db
from src.monitoring.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError
from src.monitoring.health_checker import get_health_monitor


class DataValidator:
    """Validate tick data before storing"""
    
    def __init__(
        self,
        max_price_change_pct: float = 10.0,
        min_price: float = 1.0,
        max_price: float = 10000.0
    ):
        self.max_price_change_pct = max_price_change_pct
        self.min_price = min_price
        self.max_price = max_price
        self.previous_prices: Dict[str, float] = {}
    
    def validate(self, symbol: str, tick_data: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate tick data
        
        Returns:
            (is_valid, error_message)
        """
        try:
            price = tick_data.get('price', 0)
            
            # Check price range
            if price < self.min_price or price > self.max_price:
                return False, f"Price {price} out of range [{self.min_price}, {self.max_price}]"
            
            # Check for None values
            if price is None or price <= 0:
                return False, f"Invalid price: {price}"
            
            # Check bid/ask sanity
            bid = tick_data.get('bid', 0)
            ask = tick_data.get('ask', 0)
            
            if bid and ask and bid > ask:
                return False, f"Bid ({bid}) > Ask ({ask})"
            
            if ask and price and abs(price - ask) / price > 0.1:  # 10% difference
                return False, f"Price {price} too far from ask {ask}"
            
            # Check price change from previous
            if symbol in self.previous_prices:
                prev_price = self.previous_prices[symbol]
                pct_change = abs((price - prev_price) / prev_price * 100)
                
                if pct_change > self.max_price_change_pct:
                    return False, f"Price changed {pct_change:.1f}% (threshold: {self.max_price_change_pct}%)"
            
            # Update previous price
            self.previous_prices[symbol] = price
            
            return True, None
        
        except Exception as e:
            return False, f"Validation error: {e}"


class RobustRealTimeCollector:
    """Enhanced real-time collector with robustness features"""
    
    def __init__(
        self,
        symbols: List[str] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        collect_interval: float = 1.0,
        buffer_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_validation: bool = True,
        enable_backup: bool = True,
        backup_path: str = "data/tick_backups"
    ):
        """
        Initialize robust collector
        
        Args:
            symbols: Symbols to collect
            alpaca_client: Alpaca client
            collect_interval: Collection interval in seconds
            buffer_size: Buffer size for batch inserts
            max_retries: Max retry attempts
            retry_delay: Delay between retries
            enable_validation: Enable data validation
            enable_backup: Enable file backup
            backup_path: Path for backup files
        """
        self.symbols = symbols or ['SPY', 'QQQ']
        self.alpaca = alpaca_client or AlpacaClient()
        self.collect_interval = collect_interval
        self.buffer_size = buffer_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_validation = enable_validation
        self.enable_backup = enable_backup
        self.backup_path = Path(backup_path)
        
        # Create backup directory
        if self.enable_backup:
            self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Database
        self.db = get_db()
        
        # State
        self.is_running = False
        self.collection_thread = None
        self._stop_event = threading.Event()
        
        # Buffers
        self._tick_buffer: Deque[IndexTickData] = deque(maxlen=buffer_size)
        self._backup_buffer: List[Dict] = []
        
        # Cache
        self._previous_ticks: Dict[str, Dict] = {}
        self._price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=300) for symbol in self.symbols
        }
        
        # Validation
        if self.enable_validation:
            self.validator = DataValidator()
        
        # Circuit breakers for each symbol
        self.circuit_breakers = {
            symbol: get_circuit_breaker(
                f"alpaca_{symbol}",
                failure_threshold=5,
                timeout_seconds=60
            )
            for symbol in self.symbols
        }
        
        # Health monitoring
        self.health_monitor = get_health_monitor()
        self.health_checker = self.health_monitor.register_component(
            "RobustDataCollector",
            max_errors_per_minute=10,
            max_stale_seconds=300,
            min_success_rate=0.85
        )
        
        # Market hours
        self.market_timezone = pytz.timezone("America/New_York")
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)
        
        # Statistics
        self.stats = {
            'total_ticks_collected': 0,
            'total_ticks_stored': 0,
            'total_ticks_backed_up': 0,
            'collection_errors': 0,
            'validation_failures': 0,
            'retry_successes': 0,
            'circuit_breaker_trips': 0,
            'last_collection_time': None,
        }
        
        logger.info(f"RobustRealTimeCollector initialized for {self.symbols}")
    
    def start(self):
        """Start collection"""
        if self.is_running:
            logger.warning("Collector already running")
            return
        
        logger.info("Starting robust data collection...")
        self.is_running = True
        self._stop_event.clear()
        
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="RobustDataCollector"
        )
        self.collection_thread.start()
        
        logger.info("✅ Robust data collection started")
    
    def stop(self):
        """Stop collection"""
        if not self.is_running:
            return
        
        logger.info("Stopping robust data collection...")
        self.is_running = False
        self._stop_event.set()
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        # Flush buffers
        self._flush_buffer()
        if self.enable_backup and self._backup_buffer:
            self._flush_backup()
        
        logger.info("✅ Robust data collection stopped")
        logger.info(f"Final stats: {self.stats}")
    
    def _collection_loop(self):
        """Main collection loop"""
        logger.info("Collection loop started")
        
        while self.is_running and not self._stop_event.is_set():
            try:
                if not self._is_market_open():
                    time.sleep(60)
                    continue
                
                collection_start = time.time()
                
                for symbol in self.symbols:
                    try:
                        self._collect_tick_with_retry(symbol)
                    except Exception as e:
                        logger.debug(f"Failed to collect {symbol}: {e}")
                        self.stats['collection_errors'] += 1
                        self.health_checker.record_failure(str(e))
                
                # Update health
                self.stats['last_collection_time'] = datetime.now()
                
                # Flush buffers if needed
                if len(self._tick_buffer) >= self.buffer_size:
                    self._flush_buffer()
                
                if self.enable_backup and len(self._backup_buffer) >= self.buffer_size:
                    self._flush_backup()
                
                # Maintain interval
                elapsed = time.time() - collection_start
                sleep_time = max(0, self.collect_interval - elapsed)
                
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self.stats['collection_errors'] += 1
                time.sleep(1)
        
        logger.info("Collection loop ended")
    
    def _collect_tick_with_retry(self, symbol: str):
        """Collect tick with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Use circuit breaker
                circuit_breaker = self.circuit_breakers[symbol]
                
                snapshot = circuit_breaker.call(
                    self.alpaca.get_stock_snapshot,
                    symbol
                )
                
                if snapshot:
                    self._process_snapshot(symbol, snapshot)
                    
                    # Record success
                    if attempt > 0:
                        self.stats['retry_successes'] += 1
                    
                    self.health_checker.record_success()
                    return
            
            except CircuitBreakerOpenError as e:
                self.stats['circuit_breaker_trips'] += 1
                logger.debug(f"Circuit breaker open for {symbol}")
                return
            
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} failed for {symbol}: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    self.stats['collection_errors'] += 1
                    self.health_checker.record_failure(f"{symbol}: {e}")
                    raise
    
    def _process_snapshot(self, symbol: str, snapshot: Dict):
        """Process and validate snapshot data"""
        try:
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
            
            # Prepare data for validation
            data = {
                'price': price,
                'bid': bid,
                'ask': ask,
                'spread': spread,
                'spread_pct': spread_pct,
            }
            
            # Validate if enabled
            if self.enable_validation:
                is_valid, error = self.validator.validate(symbol, data)
                
                if not is_valid:
                    logger.warning(f"Validation failed for {symbol}: {error}")
                    self.stats['validation_failures'] += 1
                    return
            
            # Calculate changes
            price_change = 0
            price_change_pct = 0
            if symbol in self._previous_ticks:
                prev_price = self._previous_ticks[symbol].get('price', price)
                price_change = price - prev_price
                price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
            
            # Update history
            self._price_history[symbol].append(price)
            
            # Calculate SMAs
            sma_5 = self._calculate_sma(symbol, 5)
            sma_60 = self._calculate_sma(symbol, 60)
            
            # Get VIX
            vix = None
            if symbol in ['SPY', 'QQQ']:
                try:
                    vix_snapshot = self.alpaca.get_stock_snapshot('VIX')
                    if vix_snapshot and vix_snapshot.get('latest_trade'):
                        vix = vix_snapshot['latest_trade'].get('price')
                except:
                    pass
            
            # Market state
            market_state = self._get_market_state()
            
            # Create tick object
            tick_data = IndexTickData(
                symbol=symbol,
                timestamp=datetime.now(),
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
            
            # Backup if enabled
            if self.enable_backup:
                self._backup_buffer.append({
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'price': price,
                    'bid': bid,
                    'ask': ask,
                    'volume': daily_bar.get('volume', 0),
                })
            
            # Update previous tick
            self._previous_ticks[symbol] = data
            self._previous_ticks[symbol]['price'] = price
            
            # Update stats
            self.stats['total_ticks_collected'] += 1
        
        except Exception as e:
            logger.error(f"Error processing snapshot for {symbol}: {e}")
            raise
    
    def _flush_buffer(self):
        """Flush ticks to database"""
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
            logger.error(f"Error flushing buffer: {e}")
            # Keep ticks in buffer for retry
            self._tick_buffer.extend(ticks_to_insert)
    
    def _flush_backup(self):
        """Flush backup to file"""
        if not self._backup_buffer:
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_path / f"ticks_{timestamp}.jsonl"
            
            with open(backup_file, 'w') as f:
                for tick in self._backup_buffer:
                    f.write(json.dumps(tick) + '\n')
            
            self.stats['total_ticks_backed_up'] += len(self._backup_buffer)
            self._backup_buffer.clear()
            
            logger.debug(f"Backed up ticks to {backup_file}")
        
        except Exception as e:
            logger.error(f"Error backing up data: {e}")
    
    def _calculate_sma(self, symbol: str, period: int) -> Optional[float]:
        """Calculate simple moving average"""
        try:
            prices = list(self._price_history[symbol])
            if len(prices) < period:
                return None
            
            return sum(prices[-period:]) / period
        except:
            return None
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            now = datetime.now(self.market_timezone)
            current_time = now.time()
            
            if now.weekday() >= 5:
                return False
            
            return self.market_open <= current_time <= self.market_close
        except:
            return False
    
    def _get_market_state(self) -> str:
        """Get market state"""
        try:
            now = datetime.now(self.market_timezone)
            current_time = now.time()
            
            if now.weekday() >= 5:
                return "closed"
            
            if current_time < dt_time(4, 0):
                return "closed"
            elif current_time < self.market_open:
                return "pre_market"
            elif current_time <= self.market_close:
                return "open"
            elif current_time < dt_time(20, 0):
                return "after_hours"
            else:
                return "closed"
        except:
            return "unknown"
    
    def get_stats(self) -> Dict:
        """Get collector statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'symbols': self.symbols,
            'buffer_size': len(self._tick_buffer),
            'backup_buffer_size': len(self._backup_buffer) if self.enable_backup else 0,
            'validation_enabled': self.enable_validation,
            'backup_enabled': self.enable_backup,
        }
    
    def get_health_status(self) -> Dict:
        """Get health status"""
        return self.health_checker.check_health()

