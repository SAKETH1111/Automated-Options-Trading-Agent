"""Market data collector with caching and enrichment"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

from loguru import logger
import pandas as pd

from src.brokers.alpaca_client import AlpacaClient
from src.config.settings import get_config
from .greeks import GreeksCalculator
from .iv_calculator import IVCalculator


class MarketDataCollector:
    """Collect and enrich market data with Greeks, IV, and rankings"""
    
    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        self.alpaca = alpaca_client or AlpacaClient()
        self.config = get_config()
        self.greeks_calc = GreeksCalculator()
        self.iv_calc = IVCalculator()
        
        # Cache for market data
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._cache_ttl = 60  # seconds
    
    def get_stock_data(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """Get comprehensive stock data"""
        cache_key = f"stock_{symbol}"
        
        if use_cache and cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_data
        
        try:
            # Get snapshot
            snapshot = self.alpaca.get_stock_snapshot(symbol)
            if not snapshot:
                return None
            
            # Get historical data for IV calculation
            hist_start = datetime.now() - timedelta(days=365)
            historical_bars = self.alpaca.get_historical_bars(
                symbol, hist_start, datetime.now()
            )
            
            if not historical_bars:
                logger.warning(f"No historical data for {symbol}")
                return None
            
            # Calculate historical volatility
            prices = [bar["close"] for bar in historical_bars]
            current_price = snapshot["latest_trade"]["price"] if snapshot["latest_trade"] else prices[-1]
            
            hv = self.iv_calc.estimate_iv_from_historical_volatility(prices, window=20)
            
            # Get VIX for market context (if symbol is SPY/QQQ)
            vix = None
            if symbol in ["SPY", "QQQ", "IWM"]:
                try:
                    vix_data = self.alpaca.get_stock_quote("VIX")
                    if vix_data:
                        vix = vix_data["last"]
                except:
                    pass
            
            data = {
                "symbol": symbol,
                "price": current_price,
                "bid": snapshot["latest_quote"]["bid"],
                "ask": snapshot["latest_quote"]["ask"],
                "bid_size": snapshot["latest_quote"]["bid_size"],
                "ask_size": snapshot["latest_quote"]["ask_size"],
                "spread": snapshot["latest_quote"]["ask"] - snapshot["latest_quote"]["bid"],
                "spread_pct": ((snapshot["latest_quote"]["ask"] - snapshot["latest_quote"]["bid"]) / current_price) * 100,
                "volume": snapshot["daily_bar"]["volume"] if snapshot["daily_bar"] else 0,
                "historical_volatility": hv,
                "vix": vix,
                "timestamp": datetime.now(),
            }
            
            self._cache[cache_key] = (datetime.now(), data)
            return data
        
        except Exception as e:
            logger.error(f"Error collecting stock data for {symbol}: {e}")
            return None
    
    def get_options_chain_enriched(
        self,
        symbol: str,
        target_dte: Optional[int] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get options chain with enriched data: Greeks, IV, IV Rank
        
        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration (optional)
            use_cache: Use cached data if available
        
        Returns:
            List of enriched option contracts
        """
        cache_key = f"options_{symbol}_{target_dte}"
        
        if use_cache and cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_data
        
        try:
            # Get stock data
            stock_data = self.get_stock_data(symbol, use_cache)
            if not stock_data:
                return []
            
            stock_price = stock_data["price"]
            
            # Get options chain
            expiration_date = None
            if target_dte:
                expiration_date = datetime.now() + timedelta(days=target_dte)
            
            options = self.alpaca.get_option_chain(symbol, expiration_date)
            
            if not options:
                logger.warning(f"No options data for {symbol}")
                return []
            
            # Parse and enrich options
            enriched_options = []
            
            for opt in options:
                try:
                    # Parse option symbol (OCC format)
                    parsed = self._parse_option_symbol(opt["option_symbol"])
                    if not parsed:
                        continue
                    
                    option_type, strike, expiration = parsed
                    
                    # Calculate time to expiry
                    dte_years = self.greeks_calc.days_to_expiry(expiration)
                    dte_days = (expiration - datetime.now()).days
                    
                    if target_dte and abs(dte_days - target_dte) > 5:
                        continue  # Filter by DTE if specified
                    
                    # Get mid price
                    mid_price = (opt["bid"] + opt["ask"]) / 2 if opt["bid"] and opt["ask"] else 0
                    
                    if mid_price <= 0:
                        continue
                    
                    # Calculate IV if not provided
                    iv = opt.get("implied_volatility")
                    if not iv or iv == 0:
                        iv = self.iv_calc.calculate_iv(
                            option_type, mid_price, stock_price, strike,
                            dte_years, 0.05, 0.0
                        )
                    
                    if not iv:
                        continue
                    
                    # Calculate Greeks
                    greeks = self.greeks_calc.calculate_greeks(
                        option_type, stock_price, strike,
                        dte_years, 0.05, iv, 0.0
                    )
                    
                    # Calculate IV Rank (using historical IV)
                    # For now, use a simplified calculation
                    # In production, store historical IV in database
                    iv_rank = 50.0  # Placeholder
                    
                    enriched_options.append({
                        "option_symbol": opt["option_symbol"],
                        "underlying_symbol": symbol,
                        "underlying_price": stock_price,
                        "option_type": option_type,
                        "strike": strike,
                        "expiration": expiration,
                        "dte": dte_days,
                        "bid": opt["bid"],
                        "ask": opt["ask"],
                        "mid": mid_price,
                        "last": opt["last"],
                        "bid_size": opt["bid_size"],
                        "ask_size": opt["ask_size"],
                        "volume": opt["volume"],
                        "open_interest": opt["open_interest"],
                        "spread": opt["ask"] - opt["bid"],
                        "spread_pct": ((opt["ask"] - opt["bid"]) / mid_price * 100) if mid_price > 0 else 0,
                        "iv": iv,
                        "iv_rank": iv_rank,
                        "delta": greeks["delta"],
                        "gamma": greeks["gamma"],
                        "theta": greeks["theta"],
                        "vega": greeks["vega"],
                        "rho": greeks["rho"],
                        "liquidity_score": self._calculate_liquidity_score(opt),
                        "timestamp": datetime.now(),
                    })
                
                except Exception as e:
                    logger.debug(f"Error enriching option {opt.get('option_symbol')}: {e}")
                    continue
            
            self._cache[cache_key] = (datetime.now(), enriched_options)
            return enriched_options
        
        except Exception as e:
            logger.error(f"Error getting enriched options chain for {symbol}: {e}")
            return []
    
    def _parse_option_symbol(self, option_symbol: str) -> Optional[Tuple[str, float, datetime]]:
        """
        Parse OCC option symbol format
        Example: AAPL230120C00150000
        """
        try:
            # Format: SYMBOL + YYMMDD + C/P + STRIKE (8 digits, price * 1000)
            if len(option_symbol) < 15:
                return None
            
            # Find where the date starts (6 digits)
            for i in range(len(option_symbol) - 15):
                try:
                    date_str = option_symbol[i:i+6]
                    year = int("20" + date_str[0:2])
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    
                    option_type_char = option_symbol[i+6]
                    if option_type_char not in ['C', 'P']:
                        continue
                    
                    option_type = "call" if option_type_char == 'C' else "put"
                    
                    strike_str = option_symbol[i+7:i+15]
                    strike = float(strike_str) / 1000
                    
                    expiration = datetime(year, month, day)
                    
                    return option_type, strike, expiration
                
                except (ValueError, IndexError):
                    continue
            
            return None
        
        except Exception as e:
            logger.debug(f"Error parsing option symbol {option_symbol}: {e}")
            return None
    
    def _calculate_liquidity_score(self, option: Dict) -> float:
        """
        Calculate liquidity score (0-100)
        Based on: volume, open interest, bid-ask spread
        """
        try:
            volume = option["volume"] or 0
            oi = option["open_interest"] or 0
            bid = option["bid"] or 0
            ask = option["ask"] or 0
            mid = (bid + ask) / 2 if bid and ask else 0
            
            # Volume score (0-33)
            volume_score = min(volume / 100, 33)
            
            # OI score (0-33)
            oi_score = min(oi / 500, 33)
            
            # Spread score (0-34)
            if mid > 0:
                spread_pct = ((ask - bid) / mid) * 100
                spread_score = max(34 - spread_pct * 3, 0)
            else:
                spread_score = 0
            
            total_score = volume_score + oi_score + spread_score
            
            return round(min(total_score, 100), 2)
        
        except Exception:
            return 0.0
    
    def scan_symbols(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Scan multiple symbols and return market data
        
        Args:
            symbols: List of symbols to scan
        
        Returns:
            Dict mapping symbol to market data
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Scanning {symbol}...")
            try:
                data = self.get_stock_data(symbol)
                if data:
                    results[symbol] = data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return results


