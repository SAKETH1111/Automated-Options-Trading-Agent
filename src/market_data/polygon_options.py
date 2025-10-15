"""
Enhanced Polygon Options Data Client
Fetches comprehensive options data including Greeks, IV, technical indicators, 
historical data, market operations, and advanced filtering capabilities
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from loguru import logger
from polygon import RESTClient


class PolygonOptionsClient:
    """
    Enhanced client for fetching comprehensive options data from Polygon.io
    
    Features:
    - Real options chains with advanced filtering
    - Actual Greeks (not calculated)
    - Implied Volatility and IV Rank
    - Open Interest and Volume
    - Real-time and historical snapshots
    - Technical indicators (SMA, EMA, MACD, RSI)
    - Historical data (custom bars, daily summaries)
    - Market operations (status, holidays, exchanges)
    - Trades and quotes data
    - Condition codes interpretation
    - Unified snapshots for multiple assets
    """
    
    def __init__(self):
        """Initialize Polygon options client"""
        api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            raise ValueError("POLYGON_API_KEY not found in environment!")
        
        self.client = RESTClient(api_key)
        logger.info("Polygon Options Client initialized")
    
    def get_options_chain(
        self,
        underlying: str,
        expiration_date: Optional[str] = None,
        strike_price: Optional[float] = None,
        option_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get options chain for an underlying symbol
        
        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiration_date: Expiration date (YYYY-MM-DD) - optional
            strike_price: Strike price - optional
            option_type: 'call' or 'put' - optional
            
        Returns:
            List of option contracts with data
        """
        try:
            logger.info(f"Fetching options chain for {underlying}")
            
            # Get options contracts
            contracts = []
            
            for contract in self.client.list_options_contracts(
                underlying_ticker=underlying,
                expiration_date=expiration_date,
                strike_price=strike_price,
                contract_type=option_type,
                limit=1000
            ):
                contracts.append({
                    'ticker': contract.ticker,
                    'underlying': contract.underlying_ticker,
                    'expiration_date': contract.expiration_date,
                    'strike': float(contract.strike_price),
                    'contract_type': contract.contract_type,
                    'shares_per_contract': contract.shares_per_contract
                })
            
            logger.info(f"Found {len(contracts)} option contracts for {underlying}")
            return contracts
            
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return []
    
    def get_option_snapshot(
        self,
        option_ticker: str
    ) -> Optional[Dict]:
        """
        Get real-time snapshot of an option contract from Polygon
        Includes: Real Greeks, IV, Open Interest, Last Price
        
        Args:
            option_ticker: Option ticker (e.g., 'O:GDX251219P00080000')
            
        Returns:
            Dict with real option data including Greeks
        """
        try:
            logger.debug(f"Fetching REAL snapshot for {option_ticker}")
            
            # Extract underlying and option details from ticker
            # Format: O:GDX251219P00080000 -> GDX, 2025-12-19, P, 80.00
            ticker_parts = option_ticker.split(':')[1] if ':' in option_ticker else option_ticker
            underlying = ticker_parts[:3]
            
            # Get real options snapshot from Polygon
            try:
                snapshot = self.client.get_snapshot_option(
                    underlying_asset=underlying,
                    option_contract=option_ticker
                )
                
                if not snapshot:
                    logger.debug(f"No snapshot data for {option_ticker}")
                    return None
                
                # Extract real market data with safe attribute access
                last_quote = snapshot.last_quote if hasattr(snapshot, 'last_quote') else None
                day_data = snapshot.day if hasattr(snapshot, 'day') else None
                
                # Real bid/ask prices with safe access
                bid = 0.0
                ask = 0.0
                last_price = 0.0
                
                if last_quote:
                    try:
                        bid = float(last_quote.bid) if hasattr(last_quote, 'bid') and last_quote.bid else 0.0
                        ask = float(last_quote.ask) if hasattr(last_quote, 'ask') and last_quote.ask else 0.0
                        last_price = float(last_quote.last) if hasattr(last_quote, 'last') and last_quote.last else (bid + ask) / 2 if bid and ask else 0.0
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing quote data for {option_ticker}: {e}")
                        bid = ask = last_price = 0.0
                
                # Real volume and open interest with safe access
                volume = 0
                open_interest = 0
                
                if day_data:
                    try:
                        volume = int(day_data.volume) if hasattr(day_data, 'volume') and day_data.volume else 0
                    except (ValueError, TypeError):
                        volume = 0
                
                try:
                    open_interest = int(snapshot.open_interest) if hasattr(snapshot, 'open_interest') and snapshot.open_interest else 0
                except (ValueError, TypeError):
                    open_interest = 0
                
                # Real implied volatility with safe access
                iv = None
                try:
                    iv = float(snapshot.implied_volatility) if hasattr(snapshot, 'implied_volatility') and snapshot.implied_volatility else None
                except (ValueError, TypeError):
                    iv = None
                
                # Real Greeks from Polygon (handle missing attributes gracefully)
                greeks = {}
                if hasattr(snapshot, 'greeks') and snapshot.greeks:
                    greeks_data = snapshot.greeks
                    greeks = {
                        'delta': float(greeks_data.delta) if hasattr(greeks_data, 'delta') and greeks_data.delta else None,
                        'gamma': float(greeks_data.gamma) if hasattr(greeks_data, 'gamma') and greeks_data.gamma else None,
                        'theta': float(greeks_data.theta) if hasattr(greeks_data, 'theta') and greeks_data.theta else None,
                        'vega': float(greeks_data.vega) if hasattr(greeks_data, 'vega') and greeks_data.vega else None,
                        'rho': float(greeks_data.rho) if hasattr(greeks_data, 'rho') and greeks_data.rho else None,
                    }
                
                # Validate data quality
                if not bid or not ask or (ask - bid) <= 0:
                    logger.debug(f"Invalid bid/ask for {option_ticker}: bid={bid}, ask={ask}")
                    return None
                
                data = {
                    'ticker': option_ticker,
                    'last_price': last_price,
                    'bid': bid,
                    'ask': ask,
                    'volume': volume,
                    'open_interest': open_interest,
                    'implied_volatility': iv,
                    'greeks': greeks,
                    'timestamp': datetime.now(),
                    'data_source': 'polygon_real'
                }
                
                logger.debug(f"REAL snapshot for {option_ticker}: bid={bid}, ask={ask}, delta={greeks.get('delta', 'N/A')}")
                return data
                
            except Exception as api_error:
                logger.warning(f"Polygon API error for {option_ticker}: {api_error}")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching real option snapshot for {option_ticker}: {e}")
            return None
    
    def get_options_chain_with_greeks(
        self,
        underlying: str,
        days_to_expiration: int = 30,
        min_dte: int = 20,
        max_dte: int = 45
    ) -> List[Dict]:
        """
        Get options chain with Greeks for options expiring in target range
        
        Args:
            underlying: Underlying symbol
            days_to_expiration: Target DTE
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            List of options with Greeks and IV
        """
        try:
            logger.info(f"Fetching options chain with Greeks for {underlying}")
            
            # Calculate target expiration range
            today = datetime.now()
            min_exp = (today + timedelta(days=min_dte)).strftime("%Y-%m-%d")
            max_exp = (today + timedelta(days=max_dte)).strftime("%Y-%m-%d")
            
            # Get contracts
            contracts = []
            
            for contract in self.client.list_options_contracts(
                underlying_ticker=underlying,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                limit=500
            ):
                contracts.append({
                    'ticker': contract.ticker,
                    'strike': float(contract.strike_price),
                    'type': contract.contract_type,
                    'expiration': contract.expiration_date
                })
            
            logger.info(f"Found {len(contracts)} contracts, fetching Greeks...")
            
            # Get Greeks for each contract (with rate limiting)
            options_data = []
            
            for i, contract in enumerate(contracts[:50]):  # Limit to 50 to avoid rate limits
                # Rate limiting
                if i > 0 and i % 5 == 0:
                    import time
                    time.sleep(1)
                
                snapshot = self.get_option_snapshot(contract['ticker'])
                
                if snapshot:
                    snapshot.update(contract)
                    options_data.append(snapshot)
            
            logger.info(f"Retrieved Greeks for {len(options_data)} options")
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options with Greeks: {e}")
            return []
    
    def get_iv_rank(
        self,
        underlying: str,
        lookback_days: int = 252
    ) -> Optional[Dict]:
        """
        Calculate IV Rank using real IV data from Polygon
        
        Args:
            underlying: Underlying symbol
            lookback_days: Days to look back (default: 1 year)
            
        Returns:
            Dict with IV metrics
        """
        try:
            logger.info(f"Calculating IV Rank for {underlying}")
            
            # Get current ATM option IV
            # First, get underlying price
            snapshot = self.client.get_snapshot_ticker("stocks", underlying)
            if not snapshot or not snapshot.ticker:
                return None
            
            current_price = float(snapshot.ticker.day.close)
            
            # Find ATM option
            # Get nearest expiration (30-45 DTE)
            target_exp = (datetime.now() + timedelta(days=35)).strftime("%Y-%m-%d")
            
            # Get ATM call
            atm_options = []
            for contract in self.client.list_options_contracts(
                underlying_ticker=underlying,
                strike_price_gte=current_price * 0.98,
                strike_price_lte=current_price * 1.02,
                contract_type='call',
                limit=10
            ):
                atm_options.append(contract.ticker)
            
            if not atm_options:
                logger.warning(f"No ATM options found for {underlying}")
                return None
            
            # Get IV for ATM option
            snapshot = self.get_option_snapshot(atm_options[0])
            
            if not snapshot or not snapshot.get('implied_volatility'):
                return None
            
            current_iv = snapshot['implied_volatility']
            
            # For now, return current IV
            # TODO: Calculate IV Rank by fetching historical IV
            return {
                'current_iv': current_iv,
                'underlying_price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV Rank: {e}")
            return None
    
    def find_optimal_strikes(
        self,
        underlying: str,
        strategy: str = 'bull_put_spread',
        target_delta: float = 0.30,
        dte_min: int = 30,
        dte_max: int = 45
    ) -> Optional[Dict]:
        """
        Find optimal strike prices using real Greeks
        
        Args:
            underlying: Underlying symbol
            strategy: Strategy type
            target_delta: Target delta for short strike
            dte_min: Minimum DTE
            dte_max: Maximum DTE
            
        Returns:
            Dict with recommended strikes and Greeks
        """
        try:
            logger.info(f"Finding optimal strikes for {underlying} {strategy}")
            
            # Get options with Greeks
            options = self.get_options_chain_with_greeks(
                underlying,
                min_dte=dte_min,
                max_dte=dte_max
            )
            
            if not options:
                return None
            
            # Filter by strategy
            if strategy == 'bull_put_spread':
                # Find puts with delta around -0.30
                candidates = [
                    opt for opt in options
                    if opt['type'] == 'put'
                    and opt.get('greeks', {}).get('delta')
                    and abs(abs(opt['greeks']['delta']) - target_delta) < 0.10
                ]
                
                if candidates:
                    # Sort by delta closest to target
                    candidates.sort(key=lambda x: abs(abs(x['greeks']['delta']) - target_delta))
                    
                    best = candidates[0]
                    
                    return {
                        'short_strike': best['strike'],
                        'short_delta': best['greeks']['delta'],
                        'short_theta': best['greeks'].get('theta'),
                        'short_vega': best['greeks'].get('vega'),
                        'iv': best.get('implied_volatility'),
                        'expiration': best['expiration'],
                        'recommendation': f"Sell {best['strike']} put (Δ={best['greeks']['delta']:.3f})"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding optimal strikes: {e}")
            return None
    
    # ==================== Technical Indicators ====================
    
    def get_sma(
        self,
        option_ticker: str,
        timespan: str = "day",
        window: int = 20,
        adjusted: bool = True
    ) -> Optional[Dict]:
        """
        Get Simple Moving Average for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            timespan: minute, hour, day, week, month, quarter, year
            window: Number of periods for SMA calculation
            adjusted: Whether to use adjusted prices
            
        Returns:
            Dict with SMA data
        """
        try:
            logger.debug(f"Fetching SMA for {option_ticker}")
            
            sma_data = self.client.get_sma(
                ticker=option_ticker,
                timespan=timespan,
                window=window,
                adjusted=adjusted
            )
            
            if not sma_data or not sma_data.results:
                return None
            
            return {
                'ticker': option_ticker,
                'sma': sma_data.results.values,
                'timestamps': sma_data.results.timestamps,
                'window': window,
                'timespan': timespan
            }
            
        except Exception as e:
            logger.error(f"Error fetching SMA for {option_ticker}: {e}")
            return None
    
    def get_ema(
        self,
        option_ticker: str,
        timespan: str = "day",
        window: int = 20,
        adjusted: bool = True
    ) -> Optional[Dict]:
        """
        Get Exponential Moving Average for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            timespan: minute, hour, day, week, month, quarter, year
            window: Number of periods for EMA calculation
            adjusted: Whether to use adjusted prices
            
        Returns:
            Dict with EMA data
        """
        try:
            logger.debug(f"Fetching EMA for {option_ticker}")
            
            ema_data = self.client.get_ema(
                ticker=option_ticker,
                timespan=timespan,
                window=window,
                adjusted=adjusted
            )
            
            if not ema_data or not ema_data.results:
                return None
            
            return {
                'ticker': option_ticker,
                'ema': ema_data.results.values,
                'timestamps': ema_data.results.timestamps,
                'window': window,
                'timespan': timespan
            }
            
        except Exception as e:
            logger.error(f"Error fetching EMA for {option_ticker}: {e}")
            return None
    
    def get_macd(
        self,
        option_ticker: str,
        timespan: str = "day",
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        adjusted: bool = True
    ) -> Optional[Dict]:
        """
        Get MACD for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            timespan: minute, hour, day, week, month, quarter, year
            short_window: Short EMA window
            long_window: Long EMA window
            signal_window: Signal line window
            adjusted: Whether to use adjusted prices
            
        Returns:
            Dict with MACD data
        """
        try:
            logger.debug(f"Fetching MACD for {option_ticker}")
            
            macd_data = self.client.get_macd(
                ticker=option_ticker,
                timespan=timespan,
                short_window=short_window,
                long_window=long_window,
                signal_window=signal_window,
                adjusted=adjusted
            )
            
            if not macd_data or not macd_data.results:
                return None
            
            return {
                'ticker': option_ticker,
                'macd': macd_data.results.values,
                'signal': macd_data.results.signal,
                'histogram': macd_data.results.histogram,
                'timestamps': macd_data.results.timestamps,
                'short_window': short_window,
                'long_window': long_window,
                'signal_window': signal_window
            }
            
        except Exception as e:
            logger.error(f"Error fetching MACD for {option_ticker}: {e}")
            return None
    
    def get_rsi(
        self,
        option_ticker: str,
        timespan: str = "day",
        window: int = 14,
        adjusted: bool = True
    ) -> Optional[Dict]:
        """
        Get RSI for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            timespan: minute, hour, day, week, month, quarter, year
            window: Number of periods for RSI calculation
            adjusted: Whether to use adjusted prices
            
        Returns:
            Dict with RSI data
        """
        try:
            logger.debug(f"Fetching RSI for {option_ticker}")
            
            rsi_data = self.client.get_rsi(
                ticker=option_ticker,
                timespan=timespan,
                window=window,
                adjusted=adjusted
            )
            
            if not rsi_data or not rsi_data.results:
                return None
            
            return {
                'ticker': option_ticker,
                'rsi': rsi_data.results.values,
                'timestamps': rsi_data.results.timestamps,
                'window': window,
                'timespan': timespan
            }
            
        except Exception as e:
            logger.error(f"Error fetching RSI for {option_ticker}: {e}")
            return None
    
    # ==================== Historical Data ====================
    
    def get_custom_bars(
        self,
        option_ticker: str,
        from_date: str,
        to_date: str,
        timespan: str = "day",
        multiplier: int = 1,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 5000
    ) -> Optional[Dict]:
        """
        Get custom bars (OHLC) for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timespan: minute, hour, day, week, month, quarter, year
            multiplier: Size of the timespan multiplier
            adjusted: Whether to use adjusted prices
            sort: Sort order (asc, desc)
            limit: Maximum number of results
            
        Returns:
            Dict with OHLC data
        """
        try:
            logger.debug(f"Fetching custom bars for {option_ticker}")
            
            bars_data = self.client.get_aggs(
                ticker=option_ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                adjusted=adjusted,
                sort=sort,
                limit=limit
            )
            
            if not bars_data or not bars_data.results:
                return None
            
            bars = []
            for bar in bars_data.results:
                bars.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap if hasattr(bar, 'vwap') else None,
                    'transactions': bar.transactions if hasattr(bar, 'transactions') else None
                })
            
            return {
                'ticker': option_ticker,
                'bars': bars,
                'count': len(bars),
                'timespan': timespan,
                'multiplier': multiplier,
                'from_date': from_date,
                'to_date': to_date
            }
            
        except Exception as e:
            logger.error(f"Error fetching custom bars for {option_ticker}: {e}")
            return None
    
    def get_daily_ticker_summary(
        self,
        option_ticker: str,
        date: str
    ) -> Optional[Dict]:
        """
        Get daily ticker summary for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            date: Date (YYYY-MM-DD)
            
        Returns:
            Dict with daily summary data
        """
        try:
            logger.debug(f"Fetching daily summary for {option_ticker} on {date}")
            
            summary_data = self.client.get_daily_open_close(
                ticker=option_ticker,
                date=date,
                adjusted=True
            )
            
            if not summary_data:
                return None
            
            return {
                'ticker': option_ticker,
                'date': date,
                'open': summary_data.open,
                'high': summary_data.high,
                'low': summary_data.low,
                'close': summary_data.close,
                'volume': summary_data.volume,
                'after_hours': summary_data.after_hours if hasattr(summary_data, 'after_hours') else None,
                'pre_market': summary_data.pre_market if hasattr(summary_data, 'pre_market') else None
            }
            
        except Exception as e:
            logger.error(f"Error fetching daily summary for {option_ticker}: {e}")
            return None
    
    def get_previous_day_bar(
        self,
        option_ticker: str
    ) -> Optional[Dict]:
        """
        Get previous day bar for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            
        Returns:
            Dict with previous day OHLC data
        """
        try:
            logger.debug(f"Fetching previous day bar for {option_ticker}")
            
            # Get yesterday's date
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            return self.get_daily_ticker_summary(option_ticker, yesterday)
            
        except Exception as e:
            logger.error(f"Error fetching previous day bar for {option_ticker}: {e}")
            return None
    
    # ==================== Market Operations ====================
    
    def get_market_status(self) -> Optional[Dict]:
        """
        Get current market status
        
        Returns:
            Dict with market status information
        """
        try:
            logger.debug("Fetching market status")
            
            status_data = self.client.get_market_status()
            
            if not status_data:
                return None
            
            return {
                'market': getattr(status_data, 'market', 'unknown'),
                'server_time': getattr(status_data, 'serverTime', None),
                'exchanges': getattr(status_data, 'exchanges', None),
                'currencies': getattr(status_data, 'currencies', None)
            }
            
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return None
    
    def get_market_holidays(self) -> Optional[List[Dict]]:
        """
        Get market holidays
        
        Returns:
            List of market holidays
        """
        try:
            logger.debug("Fetching market holidays")
            
            holidays_data = self.client.get_market_holidays()
            
            if not holidays_data:
                return None
            
            # Handle both list and object with results attribute
            holidays_list = holidays_data if isinstance(holidays_data, list) else getattr(holidays_data, 'results', [])
            
            if not holidays_list:
                return None
            
            holidays = []
            for holiday in holidays_list:
                holidays.append({
                    'date': holiday.date,
                    'name': holiday.name,
                    'exchange': holiday.exchange,
                    'status': holiday.status
                })
            
            return holidays
            
        except Exception as e:
            logger.error(f"Error fetching market holidays: {e}")
            return None
    
    def get_exchanges(self) -> Optional[List[Dict]]:
        """
        Get list of exchanges
        
        Returns:
            List of exchange information
        """
        try:
            logger.debug("Fetching exchanges")
            
            exchanges_data = self.client.get_exchanges()
            
            if not exchanges_data:
                return None
            
            # Handle both list and object with results attribute
            exchanges_list = exchanges_data if isinstance(exchanges_data, list) else getattr(exchanges_data, 'results', [])
            
            if not exchanges_list:
                return None
            
            exchanges = []
            for exchange in exchanges_list:
                exchanges.append({
                    'id': exchange.id,
                    'type': exchange.type,
                    'market': exchange.market,
                    'mic': exchange.mic,
                    'name': exchange.name,
                    'tape': exchange.tape if hasattr(exchange, 'tape') else None
                })
            
            return exchanges
            
        except Exception as e:
            logger.error(f"Error fetching exchanges: {e}")
            return None
    
    def get_condition_codes(self) -> Optional[Dict]:
        """
        Get condition codes for trades and quotes
        
        Returns:
            Dict with condition codes
        """
        try:
            logger.debug("Fetching condition codes")
            
            # Note: This might not be directly available in the Python client
            # You may need to make a direct API call
            import requests
            
            api_key = os.getenv('POLYGON_API_KEY')
            url = f"https://api.polygon.io/v3/reference/conditions?apikey={api_key}"
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                logger.warning(f"Failed to fetch condition codes: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching condition codes: {e}")
            return None
    
    # ==================== Trades and Quotes ====================
    
    def get_trades(
        self,
        option_ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 1000
    ) -> Optional[List[Dict]]:
        """
        Get historical trades for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Maximum number of results
            
        Returns:
            List of trade data
        """
        try:
            logger.debug(f"Fetching trades for {option_ticker}")
            
            trades_data = self.client.list_trades(
                ticker=option_ticker,
                timestamp_gte=from_date,
                timestamp_lte=to_date,
                limit=limit
            )
            
            if not trades_data or not trades_data.results:
                return None
            
            trades = []
            for trade in trades_data.results:
                trades.append({
                    'timestamp': trade.timestamp,
                    'price': trade.price,
                    'size': trade.size,
                    'exchange': trade.exchange,
                    'conditions': trade.conditions if hasattr(trade, 'conditions') else [],
                    'participant_timestamp': trade.participant_timestamp if hasattr(trade, 'participant_timestamp') else None
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades for {option_ticker}: {e}")
            return None
    
    def get_last_trade(self, option_ticker: str) -> Optional[Dict]:
        """
        Get last trade for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            
        Returns:
            Dict with last trade data
        """
        try:
            logger.debug(f"Fetching last trade for {option_ticker}")
            
            trade_data = self.client.get_last_trade(ticker=option_ticker)
            
            if not trade_data or not trade_data.results:
                return None
            
            trade = trade_data.results
            return {
                'ticker': option_ticker,
                'timestamp': trade.timestamp,
                'price': trade.price,
                'size': trade.size,
                'exchange': trade.exchange,
                'conditions': trade.conditions if hasattr(trade, 'conditions') else []
            }
            
        except Exception as e:
            logger.error(f"Error fetching last trade for {option_ticker}: {e}")
            return None
    
    def get_quotes(
        self,
        option_ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 1000
    ) -> Optional[List[Dict]]:
        """
        Get historical quotes for an option contract
        
        Args:
            option_ticker: Option ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Maximum number of results
            
        Returns:
            List of quote data
        """
        try:
            logger.debug(f"Fetching quotes for {option_ticker}")
            
            quotes_data = self.client.list_quotes(
                ticker=option_ticker,
                timestamp_gte=from_date,
                timestamp_lte=to_date,
                limit=limit
            )
            
            if not quotes_data or not quotes_data.results:
                return None
            
            quotes = []
            for quote in quotes_data.results:
                quotes.append({
                    'timestamp': quote.timestamp,
                    'bid': quote.bid,
                    'ask': quote.ask,
                    'bid_size': quote.bid_size,
                    'ask_size': quote.ask_size,
                    'exchange': quote.exchange,
                    'conditions': quote.conditions if hasattr(quote, 'conditions') else []
                })
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error fetching quotes for {option_ticker}: {e}")
            return None
    
    # ==================== Enhanced Snapshots ====================
    
    def get_option_chain_snapshot(
        self,
        underlying: str
    ) -> Optional[Dict]:
        """
        Get option chain snapshot for an underlying asset
        
        Args:
            underlying: Underlying ticker symbol
            
        Returns:
            Dict with option chain snapshot data
        """
        try:
            logger.debug(f"Fetching option chain snapshot for {underlying}")
            
            chain_data = self.client.get_snapshot_options_chain(
                underlying_asset=underlying
            )
            
            if not chain_data or not chain_data.results:
                return None
            
            options = []
            for option in chain_data.results:
                options.append({
                    'ticker': option.ticker,
                    'underlying': option.underlying_ticker,
                    'strike': option.strike_price,
                    'expiration': option.expiration_date,
                    'contract_type': option.contract_type,
                    'last_quote': {
                        'bid': option.last_quote.bid if hasattr(option, 'last_quote') and option.last_quote else None,
                        'ask': option.last_quote.ask if hasattr(option, 'last_quote') and option.last_quote else None,
                        'last': option.last_quote.last if hasattr(option, 'last_quote') and option.last_quote else None
                    } if hasattr(option, 'last_quote') else None,
                    'day': {
                        'open': option.day.open if hasattr(option, 'day') and option.day else None,
                        'high': option.day.high if hasattr(option, 'day') and option.day else None,
                        'low': option.day.low if hasattr(option, 'day') and option.day else None,
                        'close': option.day.close if hasattr(option, 'day') and option.day else None,
                        'volume': option.day.volume if hasattr(option, 'day') and option.day else None
                    } if hasattr(option, 'day') else None,
                    'greeks': {
                        'delta': option.greeks.delta if hasattr(option, 'greeks') and option.greeks else None,
                        'gamma': option.greeks.gamma if hasattr(option, 'greeks') and option.greeks else None,
                        'theta': option.greeks.theta if hasattr(option, 'greeks') and option.greeks else None,
                        'vega': option.greeks.vega if hasattr(option, 'greeks') and option.greeks else None
                    } if hasattr(option, 'greeks') else None,
                    'implied_volatility': option.implied_volatility if hasattr(option, 'implied_volatility') else None,
                    'open_interest': option.open_interest if hasattr(option, 'open_interest') else None
                })
            
            return {
                'underlying': underlying,
                'options': options,
                'count': len(options),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching option chain snapshot for {underlying}: {e}")
            return None
    
    def get_unified_snapshot(
        self,
        tickers: List[str]
    ) -> Optional[Dict]:
        """
        Get unified snapshot for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict with unified snapshot data
        """
        try:
            logger.debug(f"Fetching unified snapshot for {len(tickers)} tickers")
            
            snapshot_data = self.client.get_snapshot_all(
                tickers=tickers
            )
            
            if not snapshot_data or not snapshot_data.results:
                return None
            
            snapshots = {}
            for ticker, data in snapshot_data.results.items():
                snapshots[ticker] = {
                    'ticker': ticker,
                    'value': data.value if hasattr(data, 'value') else None,
                    'isin': data.isin if hasattr(data, 'isin') else None,
                    'ticker_details': data.ticker_details if hasattr(data, 'ticker_details') else None,
                    'last_quote': data.last_quote if hasattr(data, 'last_quote') else None,
                    'last_trade': data.last_trade if hasattr(data, 'last_trade') else None,
                    'prev_daily_bar': data.prev_daily_bar if hasattr(data, 'prev_daily_bar') else None,
                    'min': data.min if hasattr(data, 'min') else None,
                    'max': data.max if hasattr(data, 'max') else None
                }
            
            return {
                'snapshots': snapshots,
                'count': len(snapshots),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching unified snapshot: {e}")
            return None
    
    # ==================== Enhanced Filtering ====================
    
    def search_options_contracts(
        self,
        underlying: str,
        contract_type: Optional[str] = None,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Enhanced search for options contracts with advanced filtering
        
        Args:
            underlying: Underlying ticker symbol
            contract_type: 'call' or 'put'
            expiration_date_gte: Minimum expiration date (YYYY-MM-DD)
            expiration_date_lte: Maximum expiration date (YYYY-MM-DD)
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price
            limit: Maximum number of results
            
        Returns:
            List of filtered option contracts
        """
        try:
            logger.debug(f"Searching options contracts for {underlying}")
            
            contracts = []
            
            for contract in self.client.list_options_contracts(
                underlying_ticker=underlying,
                contract_type=contract_type,
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                strike_price_gte=strike_price_gte,
                strike_price_lte=strike_price_lte,
                limit=limit
            ):
                contracts.append({
                    'ticker': contract.ticker,
                    'underlying': contract.underlying_ticker,
                    'expiration_date': contract.expiration_date,
                    'strike': float(contract.strike_price),
                    'contract_type': contract.contract_type,
                    'shares_per_contract': contract.shares_per_contract,
                    'exercise_style': contract.exercise_style if hasattr(contract, 'exercise_style') else None,
                    'primary_exchange': contract.primary_exchange if hasattr(contract, 'primary_exchange') else None
                })
            
            logger.info(f"Found {len(contracts)} contracts matching criteria")
            return contracts
            
        except Exception as e:
            logger.error(f"Error searching options contracts: {e}")
            return []
    
    def get_options_by_delta_range(
        self,
        underlying: str,
        min_delta: float,
        max_delta: float,
        contract_type: str = "put",
        dte_min: int = 20,
        dte_max: int = 60
    ) -> List[Dict]:
        """
        Get options contracts within a specific delta range
        
        Args:
            underlying: Underlying ticker symbol
            min_delta: Minimum delta value
            max_delta: Maximum delta value
            contract_type: 'call' or 'put'
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            
        Returns:
            List of options with deltas in the specified range
        """
        try:
            logger.debug(f"Searching options by delta range for {underlying}")
            
            # Calculate date range
            today = datetime.now()
            min_exp = (today + timedelta(days=dte_min)).strftime("%Y-%m-%d")
            max_exp = (today + timedelta(days=dte_max)).strftime("%Y-%m-%d")
            
            # Get contracts in date range
            contracts = self.search_options_contracts(
                underlying=underlying,
                contract_type=contract_type,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                limit=500
            )
            
            # Filter by delta range
            filtered_options = []
            
            for contract in contracts[:50]:  # Limit to avoid rate limits
                # Rate limiting
                time.sleep(0.2)
                
                snapshot = self.get_option_snapshot(contract['ticker'])
                
                if snapshot and snapshot.get('greeks', {}).get('delta'):
                    delta = abs(snapshot['greeks']['delta'])  # Use absolute value for puts
                    
                    if min_delta <= delta <= max_delta:
                        contract.update(snapshot)
                        filtered_options.append(contract)
            
            logger.info(f"Found {len(filtered_options)} options in delta range {min_delta}-{max_delta}")
            return filtered_options
            
        except Exception as e:
            logger.error(f"Error filtering options by delta range: {e}")
            return []
    
    def get_high_volume_options(
        self,
        underlying: str,
        min_volume: int = 1000,
        min_open_interest: int = 5000,
        dte_min: int = 20,
        dte_max: int = 60
    ) -> List[Dict]:
        """
        Get high volume and open interest options
        
        Args:
            underlying: Underlying ticker symbol
            min_volume: Minimum daily volume
            min_open_interest: Minimum open interest
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            
        Returns:
            List of high volume options
        """
        try:
            logger.debug(f"Searching high volume options for {underlying}")
            
            # Calculate date range
            today = datetime.now()
            min_exp = (today + timedelta(days=dte_min)).strftime("%Y-%m-%d")
            max_exp = (today + timedelta(days=dte_max)).strftime("%Y-%m-%d")
            
            # Get contracts in date range
            contracts = self.search_options_contracts(
                underlying=underlying,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                limit=500
            )
            
            # Filter by volume and open interest
            high_volume_options = []
            
            for contract in contracts[:100]:  # Limit to avoid rate limits
                # Rate limiting
                time.sleep(0.15)
                
                snapshot = self.get_option_snapshot(contract['ticker'])
                
                if snapshot:
                    volume = snapshot.get('volume', 0)
                    open_interest = snapshot.get('open_interest', 0)
                    
                    if volume >= min_volume and open_interest >= min_open_interest:
                        contract.update(snapshot)
                        high_volume_options.append(contract)
            
            # Sort by volume descending
            high_volume_options.sort(key=lambda x: x.get('volume', 0), reverse=True)
            
            logger.info(f"Found {len(high_volume_options)} high volume options")
            return high_volume_options
            
        except Exception as e:
            logger.error(f"Error filtering high volume options: {e}")
            return []

