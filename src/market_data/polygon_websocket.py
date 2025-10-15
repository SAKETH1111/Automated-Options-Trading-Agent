"""
Polygon.io WebSocket Client for Real-Time Options Data
Provides real-time streaming of trades, quotes, aggregates, and Fair Market Value
"""

import os
import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    from polygon import WebSocketClient
    from polygon.websocket.models import WebSocketMessage
    POLYGON_WEBSOCKET_AVAILABLE = True
except ImportError:
    POLYGON_WEBSOCKET_AVAILABLE = False
    logger.warning("Polygon WebSocket client not available. Install with: pip install polygon-api-client")


class WebSocketChannel(Enum):
    """WebSocket channel types for options data"""
    TRADES = "T"           # Trades
    QUOTES = "Q"           # Quotes
    AGGREGATES_MINUTE = "AM"  # Aggregates per minute
    AGGREGATES_SECOND = "AS"  # Aggregates per second
    FAIR_MARKET_VALUE = "FMV"  # Fair Market Value


@dataclass
class OptionsTrade:
    """Real-time options trade data"""
    ticker: str
    price: float
    size: int
    exchange: str
    timestamp: int
    conditions: List[str]
    participant_timestamp: Optional[int] = None
    
    @classmethod
    def from_websocket_message(cls, msg: Dict) -> 'OptionsTrade':
        """Create OptionsTrade from WebSocket message"""
        return cls(
            ticker=msg.get('sym', ''),
            price=float(msg.get('p', 0)),
            size=int(msg.get('s', 0)),
            exchange=msg.get('x', ''),
            timestamp=int(msg.get('t', 0)),
            conditions=msg.get('c', []),
            participant_timestamp=msg.get('z')
        )


@dataclass
class OptionsQuote:
    """Real-time options quote data"""
    ticker: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    exchange: str
    timestamp: int
    conditions: List[str]
    
    @classmethod
    def from_websocket_message(cls, msg: Dict) -> 'OptionsQuote':
        """Create OptionsQuote from WebSocket message"""
        return cls(
            ticker=msg.get('sym', ''),
            bid=float(msg.get('bp', 0)),
            ask=float(msg.get('ap', 0)),
            bid_size=int(msg.get('bs', 0)),
            ask_size=int(msg.get('as', 0)),
            exchange=msg.get('x', ''),
            timestamp=int(msg.get('t', 0)),
            conditions=msg.get('c', [])
        )


@dataclass
class OptionsAggregate:
    """Real-time options aggregate data (OHLCV)"""
    ticker: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    vwap: Optional[float]
    timestamp: int
    transactions: Optional[int] = None
    
    @classmethod
    def from_websocket_message(cls, msg: Dict) -> 'OptionsAggregate':
        """Create OptionsAggregate from WebSocket message"""
        return cls(
            ticker=msg.get('sym', ''),
            open_price=float(msg.get('o', 0)),
            high_price=float(msg.get('h', 0)),
            low_price=float(msg.get('l', 0)),
            close_price=float(msg.get('c', 0)),
            volume=int(msg.get('v', 0)),
            vwap=float(msg.get('vw', 0)) if msg.get('vw') else None,
            timestamp=int(msg.get('t', 0)),
            transactions=int(msg.get('n', 0)) if msg.get('n') else None
        )


@dataclass
class FairMarketValue:
    """Real-time Fair Market Value data"""
    ticker: str
    fmv: float
    timestamp: int
    confidence: Optional[float] = None
    
    @classmethod
    def from_websocket_message(cls, msg: Dict) -> 'FairMarketValue':
        """Create FairMarketValue from WebSocket message"""
        return cls(
            ticker=msg.get('sym', ''),
            fmv=float(msg.get('fmv', 0)),
            timestamp=int(msg.get('t', 0)),
            confidence=float(msg.get('conf', 0)) if msg.get('conf') else None
        )


class PolygonOptionsWebSocketClient:
    """
    Real-time WebSocket client for Polygon.io options data
    
    Features:
    - Real-time trades, quotes, and aggregates streaming
    - Fair Market Value (FMV) data (Business plan required)
    - Automatic reconnection and error handling
    - Subscription management (max 1000 contracts per connection)
    - Data validation and parsing
    - Callback-based event handling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize WebSocket client
        
        Args:
            api_key: Polygon.io API key (defaults to environment variable)
        """
        if not POLYGON_WEBSOCKET_AVAILABLE:
            raise ImportError("Polygon WebSocket client not available. Install with: pip install polygon-api-client")
        
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment!")
        
        self.client: Optional[WebSocketClient] = None
        self.is_connected = False
        self.is_running = False
        
        # Subscription management
        self.subscriptions: List[str] = []
        self.max_subscriptions = 1000
        
        # Callback handlers
        self.trade_handler: Optional[Callable[[OptionsTrade], None]] = None
        self.quote_handler: Optional[Callable[[OptionsQuote], None]] = None
        self.aggregate_handler: Optional[Callable[[OptionsAggregate], None]] = None
        self.fmv_handler: Optional[Callable[[FairMarketValue], None]] = None
        self.error_handler: Optional[Callable[[Exception], None]] = None
        
        # Data storage for recent data
        self.recent_trades: Dict[str, List[OptionsTrade]] = {}
        self.recent_quotes: Dict[str, OptionsQuote] = {}
        self.recent_aggregates: Dict[str, OptionsAggregate] = {}
        self.recent_fmv: Dict[str, FairMarketValue] = {}
        
        # Threading
        self._ws_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info("PolygonOptionsWebSocketClient initialized")
    
    def set_trade_handler(self, handler: Callable[[OptionsTrade], None]):
        """Set callback for trade data"""
        self.trade_handler = handler
    
    def set_quote_handler(self, handler: Callable[[OptionsQuote], None]):
        """Set callback for quote data"""
        self.quote_handler = handler
    
    def set_aggregate_handler(self, handler: Callable[[OptionsAggregate], None]):
        """Set callback for aggregate data"""
        self.aggregate_handler = handler
    
    def set_fmv_handler(self, handler: Callable[[FairMarketValue], None]):
        """Set callback for Fair Market Value data"""
        self.fmv_handler = handler
    
    def set_error_handler(self, handler: Callable[[Exception], None]):
        """Set callback for error handling"""
        self.error_handler = handler
    
    def subscribe_to_trades(self, tickers: List[str]) -> bool:
        """
        Subscribe to trade data for option contracts
        
        Args:
            tickers: List of option ticker symbols
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.subscriptions) + len(tickers) > self.max_subscriptions:
            logger.error(f"Subscription limit exceeded. Max {self.max_subscriptions} contracts per connection.")
            return False
        
        trade_subscriptions = [f"{WebSocketChannel.TRADES.value}.{ticker}" for ticker in tickers]
        self.subscriptions.extend(trade_subscriptions)
        
        logger.info(f"Subscribed to trades for {len(tickers)} contracts")
        return True
    
    def subscribe_to_quotes(self, tickers: List[str]) -> bool:
        """
        Subscribe to quote data for option contracts
        
        Args:
            tickers: List of option ticker symbols
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.subscriptions) + len(tickers) > self.max_subscriptions:
            logger.error(f"Subscription limit exceeded. Max {self.max_subscriptions} contracts per connection.")
            return False
        
        quote_subscriptions = [f"{WebSocketChannel.QUOTES.value}.{ticker}" for ticker in tickers]
        self.subscriptions.extend(quote_subscriptions)
        
        logger.info(f"Subscribed to quotes for {len(tickers)} contracts")
        return True
    
    def subscribe_to_aggregates_minute(self, tickers: List[str]) -> bool:
        """
        Subscribe to per-minute aggregate data for option contracts
        
        Args:
            tickers: List of option ticker symbols
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.subscriptions) + len(tickers) > self.max_subscriptions:
            logger.error(f"Subscription limit exceeded. Max {self.max_subscriptions} contracts per connection.")
            return False
        
        agg_subscriptions = [f"{WebSocketChannel.AGGREGATES_MINUTE.value}.{ticker}" for ticker in tickers]
        self.subscriptions.extend(agg_subscriptions)
        
        logger.info(f"Subscribed to minute aggregates for {len(tickers)} contracts")
        return True
    
    def subscribe_to_aggregates_second(self, tickers: List[str]) -> bool:
        """
        Subscribe to per-second aggregate data for option contracts
        
        Args:
            tickers: List of option ticker symbols
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.subscriptions) + len(tickers) > self.max_subscriptions:
            logger.error(f"Subscription limit exceeded. Max {self.max_subscriptions} contracts per connection.")
            return False
        
        agg_subscriptions = [f"{WebSocketChannel.AGGREGATES_SECOND.value}.{ticker}" for ticker in tickers]
        self.subscriptions.extend(agg_subscriptions)
        
        logger.info(f"Subscribed to second aggregates for {len(tickers)} contracts")
        return True
    
    def subscribe_to_fmv(self, tickers: List[str]) -> bool:
        """
        Subscribe to Fair Market Value data for option contracts
        Requires Business plan subscription
        
        Args:
            tickers: List of option ticker symbols
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.subscriptions) + len(tickers) > self.max_subscriptions:
            logger.error(f"Subscription limit exceeded. Max {self.max_subscriptions} contracts per connection.")
            return False
        
        fmv_subscriptions = [f"{WebSocketChannel.FAIR_MARKET_VALUE.value}.{ticker}" for ticker in tickers]
        self.subscriptions.extend(fmv_subscriptions)
        
        logger.info(f"Subscribed to FMV for {len(tickers)} contracts")
        return True
    
    def _handle_websocket_message(self, messages: List[WebSocketMessage]):
        """
        Handle incoming WebSocket messages
        
        Args:
            messages: List of WebSocket messages
        """
        try:
            for message in messages:
                if not hasattr(message, 'data') or not message.data:
                    continue
                
                data = message.data
                message_type = data.get('ev', '')
                
                # Parse based on message type
                if message_type == 'T':  # Trade
                    self._handle_trade_message(data)
                elif message_type == 'Q':  # Quote
                    self._handle_quote_message(data)
                elif message_type == 'AM':  # Aggregate minute
                    self._handle_aggregate_message(data)
                elif message_type == 'AS':  # Aggregate second
                    self._handle_aggregate_message(data)
                elif message_type == 'FMV':  # Fair Market Value
                    self._handle_fmv_message(data)
                else:
                    logger.debug(f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            if self.error_handler:
                self.error_handler(e)
    
    def _handle_trade_message(self, data: Dict):
        """Handle trade message"""
        try:
            trade = OptionsTrade.from_websocket_message(data)
            
            # Store recent trade
            ticker = trade.ticker
            if ticker not in self.recent_trades:
                self.recent_trades[ticker] = []
            
            self.recent_trades[ticker].append(trade)
            
            # Keep only last 100 trades per ticker
            if len(self.recent_trades[ticker]) > 100:
                self.recent_trades[ticker] = self.recent_trades[ticker][-100:]
            
            # Call handler
            if self.trade_handler:
                self.trade_handler(trade)
            
            logger.debug(f"Trade: {trade.ticker} @ ${trade.price} x {trade.size}")
        
        except Exception as e:
            logger.error(f"Error parsing trade message: {e}")
    
    def _handle_quote_message(self, data: Dict):
        """Handle quote message"""
        try:
            quote = OptionsQuote.from_websocket_message(data)
            
            # Store recent quote
            self.recent_quotes[quote.ticker] = quote
            
            # Call handler
            if self.quote_handler:
                self.quote_handler(quote)
            
            logger.debug(f"Quote: {quote.ticker} ${quote.bid}/${quote.ask}")
        
        except Exception as e:
            logger.error(f"Error parsing quote message: {e}")
    
    def _handle_aggregate_message(self, data: Dict):
        """Handle aggregate message"""
        try:
            aggregate = OptionsAggregate.from_websocket_message(data)
            
            # Store recent aggregate
            self.recent_aggregates[aggregate.ticker] = aggregate
            
            # Call handler
            if self.aggregate_handler:
                self.aggregate_handler(aggregate)
            
            logger.debug(f"Aggregate: {aggregate.ticker} O:{aggregate.open_price} H:{aggregate.high_price} L:{aggregate.low_price} C:{aggregate.close_price} V:{aggregate.volume}")
        
        except Exception as e:
            logger.error(f"Error parsing aggregate message: {e}")
    
    def _handle_fmv_message(self, data: Dict):
        """Handle Fair Market Value message"""
        try:
            fmv = FairMarketValue.from_websocket_message(data)
            
            # Store recent FMV
            self.recent_fmv[fmv.ticker] = fmv
            
            # Call handler
            if self.fmv_handler:
                self.fmv_handler(fmv)
            
            logger.debug(f"FMV: {fmv.ticker} = ${fmv.fmv}")
        
        except Exception as e:
            logger.error(f"Error parsing FMV message: {e}")
    
    def connect(self) -> bool:
        """
        Connect to WebSocket and start streaming
        
        Returns:
            True if successful, False otherwise
        """
        if not self.subscriptions:
            logger.error("No subscriptions found. Add subscriptions before connecting.")
            return False
        
        try:
            self.client = WebSocketClient(
                api_key=self.api_key,
                subscriptions=self.subscriptions
            )
            
            self.is_connected = True
            logger.info(f"Connected to Polygon WebSocket with {len(self.subscriptions)} subscriptions")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            if self.error_handler:
                self.error_handler(e)
            return False
    
    def start(self) -> bool:
        """
        Start WebSocket streaming in a separate thread
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            if not self.connect():
                return False
        
        if self.is_running:
            logger.warning("WebSocket is already running")
            return True
        
        try:
            self._stop_event.clear()
            self.is_running = True
            
            self._ws_thread = threading.Thread(
                target=self._run_websocket,
                daemon=True
            )
            self._ws_thread.start()
            
            logger.info("WebSocket streaming started")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            if self.error_handler:
                self.error_handler(e)
            return False
    
    def _run_websocket(self):
        """Run WebSocket client with reconnection logic"""
        while not self._stop_event.is_set():
            try:
                if self.client:
                    self.client.run(handle_msg=self._handle_websocket_message)
                else:
                    logger.error("WebSocket client not initialized")
                    break
            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.error_handler:
                    self.error_handler(e)
                
                # Wait before reconnecting
                if not self._stop_event.wait(5):
                    logger.info("Attempting to reconnect...")
                    if not self.connect():
                        logger.error("Reconnection failed")
                        break
        
        self.is_running = False
        logger.info("WebSocket streaming stopped")
    
    def stop(self):
        """Stop WebSocket streaming"""
        if self.is_running:
            self._stop_event.set()
            self.is_running = False
            logger.info("Stopping WebSocket streaming...")
            
            if self._ws_thread and self._ws_thread.is_alive():
                self._ws_thread.join(timeout=5)
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        self.stop()
        self.is_connected = False
        self.client = None
        logger.info("Disconnected from WebSocket")
    
    def get_recent_trades(self, ticker: str, limit: int = 10) -> List[OptionsTrade]:
        """Get recent trades for a ticker"""
        return self.recent_trades.get(ticker, [])[-limit:]
    
    def get_latest_quote(self, ticker: str) -> Optional[OptionsQuote]:
        """Get latest quote for a ticker"""
        return self.recent_quotes.get(ticker)
    
    def get_latest_aggregate(self, ticker: str) -> Optional[OptionsAggregate]:
        """Get latest aggregate for a ticker"""
        return self.recent_aggregates.get(ticker)
    
    def get_latest_fmv(self, ticker: str) -> Optional[FairMarketValue]:
        """Get latest Fair Market Value for a ticker"""
        return self.recent_fmv.get(ticker)
    
    def get_subscription_count(self) -> int:
        """Get current subscription count"""
        return len(self.subscriptions)
    
    def clear_subscriptions(self):
        """Clear all subscriptions"""
        self.subscriptions.clear()
        logger.info("Cleared all subscriptions")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


class OptionsDataStreamer:
    """
    High-level options data streamer with trading agent integration
    
    Features:
    - Easy subscription management
    - Data aggregation and analysis
    - Trading signal generation
    - Real-time monitoring
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize options data streamer
        
        Args:
            api_key: Polygon.io API key
        """
        self.ws_client = PolygonOptionsWebSocketClient(api_key)
        self.setup_handlers()
        
        # Data storage
        self.trade_data: Dict[str, List[OptionsTrade]] = {}
        self.quote_data: Dict[str, OptionsQuote] = {}
        self.aggregate_data: Dict[str, List[OptionsAggregate]] = {}
        self.fmv_data: Dict[str, FairMarketValue] = {}
        
        # Statistics
        self.message_count = 0
        self.start_time = None
        
        logger.info("OptionsDataStreamer initialized")
    
    def setup_handlers(self):
        """Setup default data handlers"""
        self.ws_client.set_trade_handler(self._on_trade)
        self.ws_client.set_quote_handler(self._on_quote)
        self.ws_client.set_aggregate_handler(self._on_aggregate)
        self.ws_client.set_fmv_handler(self._on_fmv)
        self.ws_client.set_error_handler(self._on_error)
    
    def _on_trade(self, trade: OptionsTrade):
        """Handle trade data"""
        ticker = trade.ticker
        if ticker not in self.trade_data:
            self.trade_data[ticker] = []
        
        self.trade_data[ticker].append(trade)
        
        # Keep only last 1000 trades per ticker
        if len(self.trade_data[ticker]) > 1000:
            self.trade_data[ticker] = self.trade_data[ticker][-1000:]
        
        self.message_count += 1
    
    def _on_quote(self, quote: OptionsQuote):
        """Handle quote data"""
        self.quote_data[quote.ticker] = quote
        self.message_count += 1
    
    def _on_aggregate(self, aggregate: OptionsAggregate):
        """Handle aggregate data"""
        ticker = aggregate.ticker
        if ticker not in self.aggregate_data:
            self.aggregate_data[ticker] = []
        
        self.aggregate_data[ticker].append(aggregate)
        
        # Keep only last 100 aggregates per ticker
        if len(self.aggregate_data[ticker]) > 100:
            self.aggregate_data[ticker] = self.aggregate_data[ticker][-100:]
        
        self.message_count += 1
    
    def _on_fmv(self, fmv: FairMarketValue):
        """Handle Fair Market Value data"""
        self.fmv_data[fmv.ticker] = fmv
        self.message_count += 1
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        logger.error(f"WebSocket error: {error}")
    
    def start_streaming(
        self,
        tickers: List[str],
        include_trades: bool = True,
        include_quotes: bool = True,
        include_aggregates: bool = True,
        include_fmv: bool = False
    ) -> bool:
        """
        Start streaming data for specified tickers
        
        Args:
            tickers: List of option ticker symbols
            include_trades: Whether to include trade data
            include_quotes: Whether to include quote data
            include_aggregates: Whether to include aggregate data
            include_fmv: Whether to include Fair Market Value data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add subscriptions
            if include_trades:
                self.ws_client.subscribe_to_trades(tickers)
            
            if include_quotes:
                self.ws_client.subscribe_to_quotes(tickers)
            
            if include_aggregates:
                self.ws_client.subscribe_to_aggregates_minute(tickers)
            
            if include_fmv:
                self.ws_client.subscribe_to_fmv(tickers)
            
            # Start streaming
            success = self.ws_client.start()
            if success:
                self.start_time = datetime.now()
                logger.info(f"Started streaming data for {len(tickers)} tickers")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop streaming data"""
        self.ws_client.stop()
        logger.info("Stopped streaming data")
    
    def get_streaming_stats(self) -> Dict:
        """Get streaming statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'is_running': self.ws_client.is_running,
            'subscription_count': self.ws_client.get_subscription_count(),
            'message_count': self.message_count,
            'runtime_seconds': runtime,
            'messages_per_second': self.message_count / runtime if runtime > 0 else 0,
            'tickers_with_trades': len(self.trade_data),
            'tickers_with_quotes': len(self.quote_data),
            'tickers_with_aggregates': len(self.aggregate_data),
            'tickers_with_fmv': len(self.fmv_data)
        }
    
    def get_trade_summary(self, ticker: str) -> Optional[Dict]:
        """Get trade summary for a ticker"""
        trades = self.trade_data.get(ticker, [])
        if not trades:
            return None
        
        prices = [trade.price for trade in trades]
        volumes = [trade.size for trade in trades]
        
        return {
            'ticker': ticker,
            'trade_count': len(trades),
            'last_price': trades[-1].price if trades else None,
            'last_size': trades[-1].size if trades else None,
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': sum(prices) / len(prices),
            'total_volume': sum(volumes),
            'last_trade_time': trades[-1].timestamp if trades else None
        }
    
    def get_quote_summary(self, ticker: str) -> Optional[Dict]:
        """Get quote summary for a ticker"""
        quote = self.quote_data.get(ticker)
        if not quote:
            return None
        
        return {
            'ticker': ticker,
            'bid': quote.bid,
            'ask': quote.ask,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
            'spread': quote.ask - quote.bid,
            'mid_price': (quote.bid + quote.ask) / 2,
            'timestamp': quote.timestamp
        }
    
    def get_volume_profile(self, ticker: str, lookback_minutes: int = 60) -> Optional[Dict]:
        """Get volume profile for a ticker"""
        aggregates = self.aggregate_data.get(ticker, [])
        if not aggregates:
            return None
        
        # Filter by time window
        cutoff_time = int(time.time() * 1000) - (lookback_minutes * 60 * 1000)
        recent_aggregates = [agg for agg in aggregates if agg.timestamp >= cutoff_time]
        
        if not recent_aggregates:
            return None
        
        total_volume = sum(agg.volume for agg in recent_aggregates)
        avg_volume = total_volume / len(recent_aggregates)
        
        return {
            'ticker': ticker,
            'lookback_minutes': lookback_minutes,
            'total_volume': total_volume,
            'avg_volume_per_minute': avg_volume,
            'data_points': len(recent_aggregates),
            'time_range': {
                'start': min(agg.timestamp for agg in recent_aggregates),
                'end': max(agg.timestamp for agg in recent_aggregates)
            }
        }
