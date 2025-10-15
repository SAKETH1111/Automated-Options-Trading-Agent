"""
Real-Time Data Integration for Trading Agent
Integrates WebSocket data with trading strategies and signal generation
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from loguru import logger

from src.market_data.polygon_websocket import (
    PolygonOptionsWebSocketClient, 
    OptionsDataStreamer,
    OptionsTrade,
    OptionsQuote,
    OptionsAggregate,
    FairMarketValue
)


@dataclass
class RealtimeSignal:
    """Real-time trading signal based on live data"""
    ticker: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    reason: str
    timestamp: datetime
    data: Dict  # Supporting data
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'ticker': self.ticker,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }


class RealtimeOptionsMonitor:
    """
    Real-time options monitoring and signal generation
    
    Features:
    - Live data monitoring
    - Signal generation based on real-time data
    - Volume spike detection
    - Price movement alerts
    - Greeks monitoring
    - Integration with trading strategies
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize real-time options monitor
        
        Args:
            api_key: Polygon.io API key
        """
        self.streamer = OptionsDataStreamer(api_key)
        self.is_monitoring = False
        
        # Signal generation
        self.signal_handlers: List[Callable[[RealtimeSignal], None]] = []
        
        # Monitoring thresholds
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.price_change_threshold = 0.05  # 5% price change
        self.spread_threshold = 0.10  # 10% spread threshold
        
        # Data storage for analysis
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[int]] = {}
        self.spread_history: Dict[str, List[float]] = {}
        
        # Statistics
        self.signals_generated = 0
        self.start_time = None
        
        logger.info("RealtimeOptionsMonitor initialized")
    
    def add_signal_handler(self, handler: Callable[[RealtimeSignal], None]):
        """Add signal handler"""
        self.signal_handlers.append(handler)
    
    def start_monitoring(
        self,
        tickers: List[str],
        include_trades: bool = True,
        include_quotes: bool = True,
        include_aggregates: bool = True,
        include_fmv: bool = False
    ) -> bool:
        """
        Start monitoring options
        
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
            # Setup data handlers
            self._setup_data_handlers()
            
            # Start streaming
            success = self.streamer.start_streaming(
                tickers=tickers,
                include_trades=include_trades,
                include_quotes=include_quotes,
                include_aggregates=include_aggregates,
                include_fmv=include_fmv
            )
            
            if success:
                self.is_monitoring = True
                self.start_time = datetime.now()
                logger.info(f"Started monitoring {len(tickers)} options")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.streamer.stop_streaming()
        self.is_monitoring = False
        logger.info("Stopped monitoring")
    
    def _setup_data_handlers(self):
        """Setup data handlers for signal generation"""
        # Override default handlers to include signal generation
        self.streamer.ws_client.set_trade_handler(self._on_trade_with_signals)
        self.streamer.ws_client.set_quote_handler(self._on_quote_with_signals)
        self.streamer.ws_client.set_aggregate_handler(self._on_aggregate_with_signals)
        self.streamer.ws_client.set_fmv_handler(self._on_fmv_with_signals)
    
    def _on_trade_with_signals(self, trade: OptionsTrade):
        """Handle trade data with signal generation"""
        # Call original handler
        self.streamer._on_trade(trade)
        
        # Generate signals
        self._analyze_trade(trade)
    
    def _on_quote_with_signals(self, quote: OptionsQuote):
        """Handle quote data with signal generation"""
        # Call original handler
        self.streamer._on_quote(quote)
        
        # Generate signals
        self._analyze_quote(quote)
    
    def _on_aggregate_with_signals(self, aggregate: OptionsAggregate):
        """Handle aggregate data with signal generation"""
        # Call original handler
        self.streamer._on_aggregate(aggregate)
        
        # Generate signals
        self._analyze_aggregate(aggregate)
    
    def _on_fmv_with_signals(self, fmv: FairMarketValue):
        """Handle FMV data with signal generation"""
        # Call original handler
        self.streamer._on_fmv(fmv)
        
        # Generate signals
        self._analyze_fmv(fmv)
    
    def _analyze_trade(self, trade: OptionsTrade):
        """Analyze trade data for signals"""
        ticker = trade.ticker
        
        # Update price history
        if ticker not in self.price_history:
            self.price_history[ticker] = []
        self.price_history[ticker].append(trade.price)
        
        # Keep only last 100 prices
        if len(self.price_history[ticker]) > 100:
            self.price_history[ticker] = self.price_history[ticker][-100:]
        
        # Update volume history
        if ticker not in self.volume_history:
            self.volume_history[ticker] = []
        self.volume_history[ticker].append(trade.size)
        
        # Keep only last 100 volumes
        if len(self.volume_history[ticker]) > 100:
            self.volume_history[ticker] = self.volume_history[ticker][-100:]
        
        # Check for volume spike
        if len(self.volume_history[ticker]) >= 10:
            recent_avg = sum(self.volume_history[ticker][-10:]) / 10
            if trade.size > recent_avg * self.volume_spike_threshold:
                signal = RealtimeSignal(
                    ticker=ticker,
                    signal_type='buy' if trade.price > self.price_history[ticker][-2] else 'sell',
                    strength=min(1.0, trade.size / recent_avg / self.volume_spike_threshold),
                    reason=f"Volume spike: {trade.size} vs avg {recent_avg:.0f}",
                    timestamp=datetime.now(),
                    data={
                        'trade_price': trade.price,
                        'trade_size': trade.size,
                        'volume_ratio': trade.size / recent_avg,
                        'signal_type': 'volume_spike'
                    }
                )
                self._emit_signal(signal)
    
    def _analyze_quote(self, quote: OptionsQuote):
        """Analyze quote data for signals"""
        ticker = quote.ticker
        
        # Calculate spread
        spread = quote.ask - quote.bid
        mid_price = (quote.bid + quote.ask) / 2
        spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        # Update spread history
        if ticker not in self.spread_history:
            self.spread_history[ticker] = []
        self.spread_history[ticker].append(spread_pct)
        
        # Keep only last 100 spreads
        if len(self.spread_history[ticker]) > 100:
            self.spread_history[ticker] = self.spread_history[ticker][-100:]
        
        # Check for spread tightening (good for trading)
        if len(self.spread_history[ticker]) >= 5:
            recent_avg_spread = sum(self.spread_history[ticker][-5:]) / 5
            if spread_pct < recent_avg_spread * 0.5:  # 50% tighter than recent average
                signal = RealtimeSignal(
                    ticker=ticker,
                    signal_type='buy',
                    strength=min(1.0, (recent_avg_spread - spread_pct) / recent_avg_spread),
                    reason=f"Spread tightening: {spread_pct:.2f}% vs avg {recent_avg_spread:.2f}%",
                    timestamp=datetime.now(),
                    data={
                        'bid': quote.bid,
                        'ask': quote.ask,
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'signal_type': 'spread_tightening'
                    }
                )
                self._emit_signal(signal)
    
    def _analyze_aggregate(self, aggregate: OptionsAggregate):
        """Analyze aggregate data for signals"""
        ticker = aggregate.ticker
        
        # Check for significant price movement
        if len(self.price_history.get(ticker, [])) >= 2:
            prev_price = self.price_history[ticker][-2]
            price_change = abs(aggregate.close_price - prev_price) / prev_price
            
            if price_change > self.price_change_threshold:
                signal_type = 'buy' if aggregate.close_price > prev_price else 'sell'
                signal = RealtimeSignal(
                    ticker=ticker,
                    signal_type=signal_type,
                    strength=min(1.0, price_change / self.price_change_threshold),
                    reason=f"Price movement: {price_change:.2%}",
                    timestamp=datetime.now(),
                    data={
                        'prev_price': prev_price,
                        'current_price': aggregate.close_price,
                        'price_change': price_change,
                        'volume': aggregate.volume,
                        'signal_type': 'price_movement'
                    }
                )
                self._emit_signal(signal)
    
    def _analyze_fmv(self, fmv: FairMarketValue):
        """Analyze Fair Market Value data for signals"""
        ticker = fmv.ticker
        
        # Get current quote for comparison
        quote = self.streamer.quote_data.get(ticker)
        if not quote:
            return
        
        mid_price = (quote.bid + quote.ask) / 2
        fmv_diff = abs(fmv.fmv - mid_price) / mid_price if mid_price > 0 else 0
        
        # If FMV is significantly different from market price
        if fmv_diff > 0.05:  # 5% difference
            signal_type = 'buy' if fmv.fmv > mid_price else 'sell'
            signal = RealtimeSignal(
                ticker=ticker,
                signal_type=signal_type,
                strength=min(1.0, fmv_diff / 0.05),
                reason=f"FMV divergence: FMV ${fmv.fmv:.2f} vs Market ${mid_price:.2f}",
                timestamp=datetime.now(),
                data={
                    'fmv': fmv.fmv,
                    'market_price': mid_price,
                    'fmv_diff': fmv_diff,
                    'confidence': fmv.confidence,
                    'signal_type': 'fmv_divergence'
                }
            )
            self._emit_signal(signal)
    
    def _emit_signal(self, signal: RealtimeSignal):
        """Emit signal to all handlers"""
        self.signals_generated += 1
        
        logger.info(f"Signal generated: {signal.ticker} {signal.signal_type} "
                   f"(strength: {signal.strength:.2f}) - {signal.reason}")
        
        # Call all signal handlers
        for handler in self.signal_handlers:
            try:
                handler(signal)
            except Exception as e:
                logger.error(f"Error in signal handler: {e}")
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring statistics"""
        stats = self.streamer.get_streaming_stats()
        stats.update({
            'is_monitoring': self.is_monitoring,
            'signals_generated': self.signals_generated,
            'tickers_with_price_history': len(self.price_history),
            'tickers_with_volume_history': len(self.volume_history),
            'tickers_with_spread_history': len(self.spread_history)
        })
        return stats
    
    def get_ticker_analysis(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive analysis for a ticker"""
        if ticker not in self.streamer.trade_data:
            return None
        
        # Get recent data
        recent_trades = self.streamer.get_recent_trades(ticker, 20)
        latest_quote = self.streamer.get_latest_quote(ticker)
        latest_aggregate = self.streamer.get_latest_aggregate(ticker)
        latest_fmv = self.streamer.get_latest_fmv(ticker)
        
        # Calculate metrics
        analysis = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'trade_count': len(recent_trades),
            'latest_quote': latest_quote.__dict__ if latest_quote else None,
            'latest_aggregate': latest_aggregate.__dict__ if latest_aggregate else None,
            'latest_fmv': latest_fmv.__dict__ if latest_fmv else None
        }
        
        # Price analysis
        if self.price_history.get(ticker):
            prices = self.price_history[ticker]
            analysis['price_analysis'] = {
                'current_price': prices[-1],
                'price_change_5min': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else None,
                'price_volatility': self._calculate_volatility(prices),
                'price_trend': self._calculate_trend(prices)
            }
        
        # Volume analysis
        if self.volume_history.get(ticker):
            volumes = self.volume_history[ticker]
            analysis['volume_analysis'] = {
                'current_volume': volumes[-1],
                'avg_volume_5min': sum(volumes[-5:]) / 5 if len(volumes) >= 5 else None,
                'volume_trend': self._calculate_trend(volumes)
            }
        
        # Spread analysis
        if self.spread_history.get(ticker):
            spreads = self.spread_history[ticker]
            analysis['spread_analysis'] = {
                'current_spread_pct': spreads[-1],
                'avg_spread_pct': sum(spreads[-5:]) / 5 if len(spreads) >= 5 else None,
                'spread_trend': self._calculate_trend(spreads)
            }
        
        return analysis
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return sum(returns) / len(returns) if returns else 0.0
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return 'neutral'
        
        recent = values[-3:]
        if recent[-1] > recent[0]:
            return 'upward'
        elif recent[-1] < recent[0]:
            return 'downward'
        else:
            return 'neutral'


class TradingAgentWebSocketIntegration:
    """
    Integration between WebSocket data and the trading agent
    
    Features:
    - Real-time signal generation
    - Strategy integration
    - Risk management
    - Position monitoring
    """
    
    def __init__(self, trading_agent, api_key: Optional[str] = None):
        """
        Initialize WebSocket integration with trading agent
        
        Args:
            trading_agent: Instance of the trading agent
            api_key: Polygon.io API key
        """
        self.trading_agent = trading_agent
        self.monitor = RealtimeOptionsMonitor(api_key)
        
        # Setup signal handler
        self.monitor.add_signal_handler(self._handle_realtime_signal)
        
        # Monitoring state
        self.monitored_positions: Dict[str, Dict] = {}
        self.signal_history: List[RealtimeSignal] = []
        
        logger.info("TradingAgentWebSocketIntegration initialized")
    
    def start_position_monitoring(self, positions: List[Dict]) -> bool:
        """
        Start monitoring specific positions
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract option tickers from positions
            tickers = []
            for position in positions:
                option_symbol = position.get('option_symbol')
                if option_symbol:
                    tickers.append(option_symbol)
                    self.monitored_positions[option_symbol] = position
            
            if not tickers:
                logger.warning("No valid option symbols found in positions")
                return False
            
            # Start monitoring
            success = self.monitor.start_monitoring(
                tickers=tickers,
                include_trades=True,
                include_quotes=True,
                include_aggregates=True,
                include_fmv=False  # Requires Business plan
            )
            
            if success:
                logger.info(f"Started monitoring {len(tickers)} positions")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to start position monitoring: {e}")
            return False
    
    def stop_position_monitoring(self):
        """Stop monitoring positions"""
        self.monitor.stop_monitoring()
        self.monitored_positions.clear()
        logger.info("Stopped position monitoring")
    
    def _handle_realtime_signal(self, signal: RealtimeSignal):
        """Handle real-time signals"""
        # Store signal
        self.signal_history.append(signal)
        
        # Keep only last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        # Check if this is a monitored position
        if signal.ticker in self.monitored_positions:
            position = self.monitored_positions[signal.ticker]
            
            # Log signal for monitored position
            logger.info(f"Signal for monitored position {signal.ticker}: {signal.signal_type} "
                       f"(strength: {signal.strength:.2f}) - {signal.reason}")
            
            # Here you could integrate with the trading agent's decision logic
            # For example, trigger position adjustments, exits, or alerts
            self._process_position_signal(signal, position)
    
    def _process_position_signal(self, signal: RealtimeSignal, position: Dict):
        """Process signal for a specific position"""
        # This is where you would integrate with your trading agent's logic
        # For example:
        
        if signal.signal_type == 'sell' and signal.strength > 0.7:
            # Strong sell signal - consider closing position
            logger.warning(f"Strong sell signal for {signal.ticker} - consider closing position")
        
        elif signal.signal_type == 'buy' and signal.strength > 0.7:
            # Strong buy signal - consider adjusting position
            logger.info(f"Strong buy signal for {signal.ticker} - consider position adjustment")
        
        # You could also trigger alerts, notifications, or automated actions here
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        monitor_stats = self.monitor.get_monitoring_stats()
        
        return {
            **monitor_stats,
            'monitored_positions': len(self.monitored_positions),
            'signal_history_count': len(self.signal_history),
            'recent_signals': [s.to_dict() for s in self.signal_history[-10:]]
        }
    
    def get_position_analysis(self, option_symbol: str) -> Optional[Dict]:
        """Get real-time analysis for a specific position"""
        if option_symbol not in self.monitored_positions:
            return None
        
        return self.monitor.get_ticker_analysis(option_symbol)
