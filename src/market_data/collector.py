"""
Market Data Collector - Main interface for stock and options data
Wrapper that combines RealTimeDataCollector, Alpaca, and Polygon data sources
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

from loguru import logger

from src.brokers.alpaca_client import AlpacaClient
from src.market_data.greeks import GreeksCalculator
from src.market_data.iv_calculator import IVCalculator

try:
    from src.market_data.polygon_options import PolygonOptionsClient
    POLYGON_AVAILABLE = True
except:
    POLYGON_AVAILABLE = False
    logger.warning("Polygon not available - using simulated options data")


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
            bars = self.alpaca.get_historical_bars(
                symbol,
                start_date=(datetime.now() - timedelta(days=365)).isoformat(),
                timeframe="1Day"
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
            
            enriched_options = []
            
            for contract in contracts:
                # Calculate time to expiration
                exp_date = datetime.strptime(contract['expiration_date'], '%Y-%m-%d')
                dte = (exp_date - datetime.now()).days
                
                # Filter by DTE range
                if not (target_dte - 10 <= dte <= target_dte + 10):
                    continue
                
                # Get option snapshot for pricing
                snapshot = self.polygon.get_option_snapshot(contract['ticker'])
                
                if not snapshot:
                    continue
                
                bid = snapshot.get('bid', 0)
                ask = snapshot.get('ask', 0)
                mid = (bid + ask) / 2 if (bid and ask) else 0
                
                if mid <= 0:
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
            logger.error(f"Error loading Polygon options: {e}")
            return self._get_simulated_options(symbol, stock_price, target_dte, option_type)
    
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

