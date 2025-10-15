"""
Polygon Options Data Client
Fetches real options data including Greeks, IV, and Open Interest
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
from polygon import RESTClient


class PolygonOptionsClient:
    """
    Client for fetching real options data from Polygon.io
    
    Features:
    - Real options chains
    - Actual Greeks (not calculated)
    - Implied Volatility
    - Open Interest
    - Options snapshots
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
                
                # Extract real market data
                last_quote = snapshot.last_quote if hasattr(snapshot, 'last_quote') else None
                day_data = snapshot.day if hasattr(snapshot, 'day') else None
                
                # Real bid/ask prices
                bid = float(last_quote.bid) if last_quote and last_quote.bid else 0.0
                ask = float(last_quote.ask) if last_quote and last_quote.ask else 0.0
                last_price = float(last_quote.last) if last_quote and last_quote.last else (bid + ask) / 2 if bid and ask else 0.0
                
                # Real volume and open interest
                volume = int(day_data.volume) if day_data and day_data.volume else 0
                open_interest = int(snapshot.open_interest) if hasattr(snapshot, 'open_interest') and snapshot.open_interest else 0
                
                # Real implied volatility
                iv = float(snapshot.implied_volatility) if hasattr(snapshot, 'implied_volatility') and snapshot.implied_volatility else None
                
                # Real Greeks from Polygon
                greeks = {}
                if hasattr(snapshot, 'greeks') and snapshot.greeks:
                    greeks = {
                        'delta': float(snapshot.greeks.delta) if snapshot.greeks.delta else None,
                        'gamma': float(snapshot.greeks.gamma) if snapshot.greeks.gamma else None,
                        'theta': float(snapshot.greeks.theta) if snapshot.greeks.theta else None,
                        'vega': float(snapshot.greeks.vega) if snapshot.greeks.vega else None,
                        'rho': float(snapshot.greeks.rho) if snapshot.greeks.rho else None,
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

