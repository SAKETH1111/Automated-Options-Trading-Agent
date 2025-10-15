"""
Market Data Collector - Main interface for stock and options data
Wrapper that combines RealTimeDataCollector, Alpaca, and Polygon data sources
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import time

from loguru import logger

from src.brokers.alpaca_client import AlpacaClient
from src.market_data.greeks import GreeksCalculator
from src.market_data.iv_calculator import IVCalculator

try:
    from src.market_data.polygon_options import PolygonOptionsClient
    POLYGON_AVAILABLE = True
except Exception as e:
    POLYGON_AVAILABLE = False
    logger.warning(f"Polygon not available - using simulated options data. Error: {e}")


class MarketDataCollector:
    """
    Main market data collector for stocks and options
    Provides unified interface for signal generation
    """
    
    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        """
        Initialize market data collector
        
        Args:
            alpaca_client: Alpaca client instance
        """
        self.alpaca = alpaca_client or AlpacaClient()
        
        # Try to initialize Polygon for real options data
        self.polygon = None
        if POLYGON_AVAILABLE and os.getenv('POLYGON_API_KEY'):
            try:
                self.polygon = PolygonOptionsClient()
                logger.info("Polygon options data enabled")
            except Exception as e:
                logger.warning(f"Polygon init failed: {e}")
        
        self.greeks_calc = GreeksCalculator()
        self.iv_calc = IVCalculator()
        
        logger.info("MarketDataCollector initialized")
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Get current stock data with technical indicators
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with price, volume, spread, indicators
        """
        try:
            # Get snapshot
            snapshot = self.alpaca.get_stock_snapshot(symbol)
            
            if not snapshot:
                return None
            
            latest_trade = snapshot.get('latest_trade', {})
            latest_quote = snapshot.get('latest_quote', {})
            daily_bar = snapshot.get('daily_bar', {})
            
            price = latest_trade.get('price', 0)
            bid = latest_quote.get('bid', 0)
            ask = latest_quote.get('ask', 0)
            
            if price <= 0:
                return None
            
            spread = ask - bid if (bid and ask) else 0
            spread_pct = (spread / price * 100) if price > 0 else 0
            
            # Get historical bars for IV rank calculation
            from alpaca.data.timeframe import TimeFrame
            bars = self.alpaca.get_historical_bars(
                symbol,
                start_date=(datetime.now() - timedelta(days=365)).isoformat(),
                timeframe=TimeFrame.Day
            )
            
            # Calculate IV rank (simplified - would use historical IV in production)
            iv_rank = 50.0  # Default placeholder
            if bars and len(bars) > 20:
                # Simple volatility calculation
                prices = [bar['close'] for bar in bars[-252:]]  # Last year
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                import numpy as np
                volatility = np.std(returns) * np.sqrt(252) * 100
                iv_rank = min(100, max(0, volatility * 2))  # Rough approximation
            
            return {
                "symbol": symbol,
                "price": price,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "spread_pct": spread_pct,
                "volume": daily_bar.get('volume', 0),
                "open": daily_bar.get('open', price),
                "high": daily_bar.get('high', price),
                "low": daily_bar.get('low', price),
                "close": price,
                "iv_rank": iv_rank,
                "timestamp": datetime.now(),
            }
        
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def get_options_chain_enriched(
        self,
        symbol: str,
        target_dte: int = 35,
        option_type: str = "put"
    ) -> List[Dict]:
        """
        Get options chain with Greeks, IV, and analytics
        
        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            option_type: 'put' or 'call'
            
        Returns:
            List of enriched option contracts
        """
        try:
            logger.info(f"Fetching options chain for {symbol} (DTE ~{target_dte})")
            
            # Get current stock price
            stock_data = self.get_stock_data(symbol)
            if not stock_data:
                logger.warning(f"No stock data for {symbol}")
                return []
            
            stock_price = stock_data['price']
            
            # If Polygon available, use real data
            if self.polygon:
                return self._get_polygon_options(symbol, stock_price, target_dte, option_type)
            else:
                return self._get_simulated_options(symbol, stock_price, target_dte, option_type)
        
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []
    
    def _get_polygon_options(
        self,
        symbol: str,
        stock_price: float,
        target_dte: int,
        option_type: str
    ) -> List[Dict]:
        """Get real options data from Polygon"""
        try:
            # Find expiration date closest to target DTE
            target_date = datetime.now() + timedelta(days=target_dte)
            expiration_str = target_date.strftime('%Y-%m-%d')
            
            # Get options chain
            contracts = self.polygon.get_options_chain(
                underlying=symbol,
                option_type=option_type
            )
            
            logger.debug(f"Polygon returned {len(contracts)} contracts for {symbol}")
            
            # Debug: show available DTE ranges
            dte_counts = {}
            for contract in contracts[:20]:  # Check first 20 contracts
                exp_date = datetime.strptime(contract['expiration_date'], '%Y-%m-%d')
                dte = (exp_date - datetime.now()).days
                dte_counts[dte] = dte_counts.get(dte, 0) + 1
            logger.debug(f"Available DTE ranges: {sorted(dte_counts.keys())}")
            
            enriched_options = []
            
            for contract in contracts:
                # Calculate time to expiration
                exp_date = datetime.strptime(contract['expiration_date'], '%Y-%m-%d')
                dte = (exp_date - datetime.now()).days
                
                # Filter by DTE range - include expired options for testing
                if not (-5 <= dte <= 60):  # Accept -5 to 60 DTE for testing (include some expired)
                    continue
                
                # Rate limiting for Polygon API (5 calls per second max)
                import time
                time.sleep(0.25)  # 250ms between calls (4 calls per second)
                
                # Get REAL option snapshot for pricing
                snapshot = self.polygon.get_option_snapshot(contract['ticker'])
                
                if not snapshot:
                    logger.debug(f"No real snapshot data for {contract['ticker']}")
                    continue
                
                bid = snapshot.get('bid', 0)
                ask = snapshot.get('ask', 0)
                mid = (bid + ask) / 2 if (bid and ask) else 0
                
                logger.debug(f"Option {contract['ticker']}: bid={bid}, ask={ask}, mid={mid}")
                
                if mid <= 0:
                    logger.debug(f"Skipping {contract['ticker']}: no valid mid price")
                    continue
                
                # Calculate Greeks if not provided
                greeks = snapshot.get('greeks', {})
                if not greeks or not greeks.get('delta'):
                    tte = dte / 365.0
                    greeks = self.greeks_calc.calculate_greeks(
                        option_type=option_type,
                        stock_price=stock_price,
                        strike=contract['strike'],
                        time_to_expiry=tte,
                        risk_free_rate=0.05,
                        volatility=0.25  # Default
                    )
                
                # Calculate IV if not provided
                iv = snapshot.get('implied_volatility')
                if not iv:
                    tte = dte / 365.0
                    iv = self.iv_calc.calculate_iv(
                        option_type=option_type,
                        market_price=mid,
                        stock_price=stock_price,
                        strike=contract['strike'],
                        time_to_expiry=tte,
                        risk_free_rate=0.05
                    )
                
                enriched_options.append({
                    'symbol': contract['ticker'],
                    'underlying': symbol,
                    'strike': contract['strike'],
                    'expiration': exp_date,
                    'dte': dte,
                    'option_type': option_type,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'last': snapshot.get('last_price', mid),
                    'volume': snapshot.get('volume', 0),
                    'open_interest': snapshot.get('open_interest', 0),
                    'delta': greeks.get('delta', 0),
                    'gamma': greeks.get('gamma', 0),
                    'theta': greeks.get('theta', 0),
                    'vega': greeks.get('vega', 0),
                    'rho': greeks.get('rho', 0),
                    'implied_volatility': iv,
                    'bid_ask_spread': ask - bid,
                    'bid_ask_spread_pct': ((ask - bid) / mid * 100) if mid > 0 else 0,
                })
            
            logger.info(f"Loaded {len(enriched_options)} options for {symbol}")
            return enriched_options
        
        except Exception as e:
            logger.error(f"Error loading real Polygon options: {e}")
            return []  # Return empty list - no simulated data for real trading
    
    def _get_simulated_options(
        self,
        symbol: str,
        stock_price: float,
        target_dte: int,
        option_type: str
    ) -> List[Dict]:
        """
        Generate simulated options chain for testing/demo
        Used when Polygon is not available
        """
        try:
            logger.info(f"Generating simulated options chain for {symbol}")
            
            enriched_options = []
            
            # Generate strikes around current price
            # For puts: strikes below current price
            strike_increment = 1 if stock_price < 50 else 2.5 if stock_price < 200 else 5
            num_strikes = 10
            
            for i in range(num_strikes):
                # Generate strikes below price for puts
                strike = stock_price - (i * strike_increment)
                
                if strike <= 0:
                    continue
                
                # Calculate Greeks
                tte = target_dte / 365.0
                greeks = self.greeks_calc.calculate_greeks(
                    option_type=option_type,
                    stock_price=stock_price,
                    strike=strike,
                    time_to_expiry=tte,
                    risk_free_rate=0.05,
                    volatility=0.25  # Default 25% IV
                )
                
                # Estimate option price using Black-Scholes
                from src.market_data.iv_calculator import IVCalculator
                option_price = IVCalculator.black_scholes_price(
                    option_type=option_type,
                    stock_price=stock_price,
                    strike=strike,
                    time_to_expiry=tte,
                    risk_free_rate=0.05,
                    volatility=0.25
                )
                
                # Simulate bid-ask spread
                spread_pct = 0.05  # 5% spread
                mid = option_price
                bid = mid * (1 - spread_pct / 2)
                ask = mid * (1 + spread_pct / 2)
                
                # Simulate volume and OI
                # Higher for ATM options
                moneyness = abs(stock_price - strike) / stock_price
                volume = int(1000 * (1 - moneyness * 2))
                open_interest = int(5000 * (1 - moneyness * 2))
                
                expiration = datetime.now() + timedelta(days=target_dte)
                
                enriched_options.append({
                    'symbol': f"{symbol}{expiration.strftime('%y%m%d')}P{int(strike*1000):08d}",
                    'underlying': symbol,
                    'strike': strike,
                    'expiration': expiration,
                    'dte': target_dte,
                    'option_type': option_type,
                    'bid': round(bid, 2),
                    'ask': round(ask, 2),
                    'mid': round(mid, 2),
                    'last': round(mid, 2),
                    'volume': max(100, volume),
                    'open_interest': max(500, open_interest),
                    'delta': greeks.get('delta', 0),
                    'gamma': greeks.get('gamma', 0),
                    'theta': greeks.get('theta', 0),
                    'vega': greeks.get('vega', 0),
                    'rho': greeks.get('rho', 0),
                    'implied_volatility': 0.25,
                    'bid_ask_spread': ask - bid,
                    'bid_ask_spread_pct': spread_pct * 100,
                })
            
            logger.info(f"Generated {len(enriched_options)} simulated options for {symbol}")
            return enriched_options
        
        except Exception as e:
            logger.error(f"Error generating simulated options: {e}")
            return []
    
    def get_stock_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get stock snapshot - delegates to Alpaca client"""
        return self.alpaca.get_stock_snapshot(symbol)
    
    def get_historical_bars(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical bars - delegates to Alpaca client"""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        return self.alpaca.get_historical_bars(symbol, start_date, timeframe="1Day")
    
    # ==================== Enhanced Polygon Options Methods ====================
    
    def get_option_technical_indicators(
        self,
        option_ticker: str,
        indicators: List[str] = None
    ) -> Dict:
        """
        Get technical indicators for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            indicators: List of indicators to fetch ['sma', 'ema', 'macd', 'rsi']
            
        Returns:
            Dict with technical indicator data
        """
        if not self.polygon:
            logger.warning("Polygon not available for technical indicators")
            return {}
        
        if indicators is None:
            indicators = ['sma', 'ema', 'macd', 'rsi']
        
        result = {}
        
        try:
            for indicator in indicators:
                if indicator == 'sma':
                    result['sma'] = self.polygon.get_sma(option_ticker)
                elif indicator == 'ema':
                    result['ema'] = self.polygon.get_ema(option_ticker)
                elif indicator == 'macd':
                    result['macd'] = self.polygon.get_macd(option_ticker)
                elif indicator == 'rsi':
                    result['rsi'] = self.polygon.get_rsi(option_ticker)
                
                # Rate limiting
                time.sleep(0.1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {option_ticker}: {e}")
            return {}
    
    def get_option_historical_data(
        self,
        option_ticker: str,
        days: int = 30,
        timespan: str = "day"
    ) -> Optional[Dict]:
        """
        Get historical data for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            days: Number of days to look back
            timespan: minute, hour, day, week, month, quarter, year
            
        Returns:
            Dict with historical data
        """
        if not self.polygon:
            logger.warning("Polygon not available for historical data")
            return None
        
        try:
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            return self.polygon.get_custom_bars(
                option_ticker=option_ticker,
                from_date=from_date,
                to_date=to_date,
                timespan=timespan
            )
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {option_ticker}: {e}")
            return None
    
    def get_market_operations(self) -> Dict:
        """
        Get market operations data (status, holidays, exchanges)
        
        Returns:
            Dict with market operations data
        """
        if not self.polygon:
            logger.warning("Polygon not available for market operations")
            return {}
        
        try:
            return {
                'market_status': self.polygon.get_market_status(),
                'holidays': self.polygon.get_market_holidays(),
                'exchanges': self.polygon.get_exchanges(),
                'condition_codes': self.polygon.get_condition_codes()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market operations: {e}")
            return {}
    
    def get_option_chain_snapshot(self, underlying: str) -> Optional[Dict]:
        """
        Get option chain snapshot for an underlying asset
        
        Args:
            underlying: Underlying ticker symbol
            
        Returns:
            Dict with option chain snapshot
        """
        if not self.polygon:
            logger.warning("Polygon not available for option chain snapshot")
            return None
        
        try:
            return self.polygon.get_option_chain_snapshot(underlying)
            
        except Exception as e:
            logger.error(f"Error fetching option chain snapshot for {underlying}: {e}")
            return None
    
    def get_enhanced_options_chain(
        self,
        symbol: str,
        target_dte: int = 35,
        option_type: str = "put",
        use_advanced_filtering: bool = True
    ) -> List[Dict]:
        """
        Get enhanced options chain with advanced filtering and analytics
        
        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            option_type: 'put' or 'call'
            use_advanced_filtering: Whether to use advanced filtering
            
        Returns:
            List of enhanced option contracts
        """
        try:
            logger.info(f"Fetching enhanced options chain for {symbol}")
            
            if not self.polygon:
                logger.warning("Polygon not available, falling back to basic chain")
                return self.get_options_chain_enriched(symbol, target_dte, option_type)
            
            # Get current stock price
            stock_data = self.get_stock_data(symbol)
            if not stock_data:
                logger.warning(f"No stock data for {symbol}")
                return []
            
            stock_price = stock_data['price']
            
            if use_advanced_filtering:
                # Use advanced filtering to get high-quality options
                options = self.polygon.get_high_volume_options(
                    underlying=symbol,
                    min_volume=500,
                    min_open_interest=2000,
                    dte_min=target_dte - 10,
                    dte_max=target_dte + 10
                )
                
                # Filter by option type
                options = [opt for opt in options if opt.get('contract_type') == option_type]
                
                # Add additional analytics
                for option in options:
                    # Add technical indicators
                    indicators = self.get_option_technical_indicators(
                        option['ticker'],
                        ['rsi', 'sma']
                    )
                    option['technical_indicators'] = indicators
                    
                    # Add historical data
                    historical = self.get_option_historical_data(option['ticker'], days=30)
                    option['historical_data'] = historical
                    
                    # Rate limiting
                    time.sleep(0.1)
                
                logger.info(f"Enhanced chain loaded {len(options)} high-quality options for {symbol}")
                return options
            else:
                # Use basic chain
                return self._get_polygon_options(symbol, stock_price, target_dte, option_type)
                
        except Exception as e:
            logger.error(f"Error fetching enhanced options chain for {symbol}: {e}")
            return []
    
    def get_options_by_delta_range(
        self,
        symbol: str,
        min_delta: float,
        max_delta: float,
        contract_type: str = "put",
        dte_min: int = 20,
        dte_max: int = 60
    ) -> List[Dict]:
        """
        Get options within a specific delta range
        
        Args:
            symbol: Underlying symbol
            min_delta: Minimum delta value
            max_delta: Maximum delta value
            contract_type: 'call' or 'put'
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            
        Returns:
            List of options in delta range
        """
        if not self.polygon:
            logger.warning("Polygon not available for delta filtering")
            return []
        
        try:
            return self.polygon.get_options_by_delta_range(
                underlying=symbol,
                min_delta=min_delta,
                max_delta=max_delta,
                contract_type=contract_type,
                dte_min=dte_min,
                dte_max=dte_max
            )
            
        except Exception as e:
            logger.error(f"Error filtering options by delta range: {e}")
            return []
    
    def get_option_trades_quotes(
        self,
        option_ticker: str,
        days: int = 7
    ) -> Dict:
        """
        Get historical trades and quotes for an option
        
        Args:
            option_ticker: Option ticker symbol
            days: Number of days to look back
            
        Returns:
            Dict with trades and quotes data
        """
        if not self.polygon:
            logger.warning("Polygon not available for trades/quotes")
            return {}
        
        try:
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            return {
                'trades': self.polygon.get_trades(option_ticker, from_date, to_date),
                'quotes': self.polygon.get_quotes(option_ticker, from_date, to_date),
                'last_trade': self.polygon.get_last_trade(option_ticker)
            }
            
        except Exception as e:
            logger.error(f"Error fetching trades/quotes for {option_ticker}: {e}")
            return {}
    
    def get_unified_market_snapshot(
        self,
        symbols: List[str]
    ) -> Optional[Dict]:
        """
        Get unified snapshot for multiple symbols (stocks and options)
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dict with unified snapshot data
        """
        if not self.polygon:
            logger.warning("Polygon not available for unified snapshot")
            return None
        
        try:
            return self.polygon.get_unified_snapshot(symbols)
            
        except Exception as e:
            logger.error(f"Error fetching unified snapshot: {e}")
            return None

