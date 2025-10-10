"""
Options Chain Collector Module
Fetch and store real-time options chain data
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import OptionsChain
from src.options.greeks import GreeksCalculator


class OptionsChainCollector:
    """
    Collect options chain data and store in database
    """
    
    def __init__(self, db_session: Session, alpaca_client=None):
        """
        Initialize options chain collector
        
        Args:
            db_session: Database session
            alpaca_client: Alpaca client for data fetching
        """
        self.db = db_session
        self.alpaca = alpaca_client
        self.greeks_calc = GreeksCalculator()
        logger.info("Options Chain Collector initialized")
    
    def collect_chain(
        self,
        symbol: str,
        target_dte: Optional[int] = None,
        min_delta: float = 0.05,
        max_delta: float = 0.95
    ) -> List[Dict]:
        """
        Collect options chain for a symbol
        
        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration (None = all)
            min_delta: Minimum delta to include
            max_delta: Maximum delta to include
            
        Returns:
            List of option contracts
        """
        try:
            if not self.alpaca:
                logger.error("Alpaca client not available")
                return []
            
            logger.info(f"Collecting options chain for {symbol}")
            
            # Get underlying price
            stock_data = self.alpaca.get_stock_data(symbol)
            if not stock_data:
                logger.error(f"Could not get stock data for {symbol}")
                return []
            
            underlying_price = stock_data['price']
            
            # Get options chain
            # Note: This is a placeholder - actual implementation depends on Alpaca API
            options = self._fetch_options_from_alpaca(symbol, target_dte)
            
            if not options:
                logger.warning(f"No options data for {symbol}")
                return []
            
            # Process and enrich options
            enriched_options = []
            
            for opt in options:
                try:
                    # Calculate time to expiry
                    dte = opt.get('dte', 0)
                    time_to_expiry = dte / 365.0
                    
                    # Get option details
                    option_type = opt.get('type', 'CALL')
                    strike = opt.get('strike', 0)
                    iv = opt.get('implied_volatility', 0.20)
                    
                    # Calculate Greeks
                    greeks = self.greeks_calc.calculate_all_greeks(
                        option_type=option_type,
                        stock_price=underlying_price,
                        strike=strike,
                        time_to_expiry=time_to_expiry,
                        volatility=iv
                    )
                    
                    # Filter by delta
                    delta = abs(greeks['delta'])
                    if delta < min_delta or delta > max_delta:
                        continue
                    
                    # Calculate intrinsic and extrinsic value
                    intrinsic = self.greeks_calc.calculate_intrinsic_value(
                        option_type, underlying_price, strike
                    )
                    
                    mid_price = opt.get('mid_price', 0)
                    extrinsic = self.greeks_calc.calculate_extrinsic_value(
                        mid_price, intrinsic
                    )
                    
                    # Determine moneyness
                    moneyness = self.greeks_calc.get_moneyness(
                        option_type, underlying_price, strike
                    )
                    
                    # Build enriched option
                    enriched_opt = {
                        'symbol': symbol,
                        'underlying_price': underlying_price,
                        'option_symbol': opt.get('option_symbol', ''),
                        'option_type': option_type,
                        'strike': strike,
                        'expiration': opt.get('expiration'),
                        'dte': dte,
                        'bid': opt.get('bid'),
                        'ask': opt.get('ask'),
                        'mid_price': mid_price,
                        'last_price': opt.get('last_price'),
                        'mark': opt.get('mark'),
                        'delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'theta': greeks['theta'],
                        'vega': greeks['vega'],
                        'rho': greeks['rho'],
                        'implied_volatility': iv,
                        'volume': opt.get('volume', 0),
                        'open_interest': opt.get('open_interest', 0),
                        'bid_ask_spread': opt.get('ask', 0) - opt.get('bid', 0),
                        'bid_ask_spread_pct': ((opt.get('ask', 0) - opt.get('bid', 0)) / 
                                               mid_price * 100) if mid_price > 0 else 0,
                        'intrinsic_value': intrinsic,
                        'extrinsic_value': extrinsic,
                        'moneyness': moneyness,
                        'timestamp': datetime.utcnow()
                    }
                    
                    enriched_options.append(enriched_opt)
                    
                except Exception as e:
                    logger.error(f"Error processing option: {e}")
                    continue
            
            logger.info(f"Collected {len(enriched_options)} options for {symbol}")
            return enriched_options
            
        except Exception as e:
            logger.error(f"Error collecting options chain: {e}")
            return []
    
    def _fetch_options_from_alpaca(
        self,
        symbol: str,
        target_dte: Optional[int]
    ) -> List[Dict]:
        """
        Fetch options from Alpaca API
        
        Note: This is a placeholder implementation.
        Actual implementation depends on Alpaca's options API.
        """
        try:
            # Placeholder: Return simulated data for testing
            # In production, this would call alpaca.get_option_chain()
            
            logger.warning("Using simulated options data (Alpaca options API limited)")
            
            # Simulate some options data
            simulated_options = []
            
            # Get current price
            stock_data = self.alpaca.get_stock_data(symbol)
            if not stock_data:
                return []
            
            price = stock_data['price']
            
            # Generate strikes around current price
            strikes = [
                price * 0.95, price * 0.97, price * 0.99,
                price, price * 1.01, price * 1.03, price * 1.05
            ]
            
            expiration = datetime.now() + timedelta(days=target_dte if target_dte else 30)
            
            for strike in strikes:
                for option_type in ['CALL', 'PUT']:
                    simulated_options.append({
                        'option_symbol': f"{symbol}{expiration.strftime('%y%m%d')}{option_type[0]}{int(strike)}",
                        'type': option_type,
                        'strike': strike,
                        'expiration': expiration,
                        'dte': target_dte if target_dte else 30,
                        'bid': max(0.01, abs(price - strike) * 0.8),
                        'ask': max(0.02, abs(price - strike) * 0.9),
                        'mid_price': max(0.015, abs(price - strike) * 0.85),
                        'last_price': max(0.015, abs(price - strike) * 0.85),
                        'mark': max(0.015, abs(price - strike) * 0.85),
                        'implied_volatility': 0.20 + (abs(price - strike) / price) * 0.1,
                        'volume': int(np.random.randint(100, 10000)),
                        'open_interest': int(np.random.randint(1000, 50000))
                    })
            
            return simulated_options
            
        except Exception as e:
            logger.error(f"Error fetching options from Alpaca: {e}")
            return []
    
    def store_chain(
        self,
        options: List[Dict]
    ) -> int:
        """
        Store options chain in database
        
        Args:
            options: List of option contracts
            
        Returns:
            Number of options stored
        """
        try:
            stored_count = 0
            
            for opt in options:
                try:
                    option_record = OptionsChain(
                        symbol=opt['symbol'],
                        underlying_price=opt['underlying_price'],
                        timestamp=opt['timestamp'],
                        option_symbol=opt['option_symbol'],
                        option_type=opt['option_type'],
                        strike=opt['strike'],
                        expiration=opt['expiration'],
                        dte=opt['dte'],
                        bid=opt.get('bid'),
                        ask=opt.get('ask'),
                        mid_price=opt.get('mid_price'),
                        last_price=opt.get('last_price'),
                        mark=opt.get('mark'),
                        delta=opt.get('delta'),
                        gamma=opt.get('gamma'),
                        theta=opt.get('theta'),
                        vega=opt.get('vega'),
                        rho=opt.get('rho'),
                        implied_volatility=opt.get('implied_volatility'),
                        volume=opt.get('volume'),
                        open_interest=opt.get('open_interest'),
                        bid_ask_spread=opt.get('bid_ask_spread'),
                        bid_ask_spread_pct=opt.get('bid_ask_spread_pct'),
                        intrinsic_value=opt.get('intrinsic_value'),
                        extrinsic_value=opt.get('extrinsic_value'),
                        moneyness=opt.get('moneyness')
                    )
                    
                    self.db.add(option_record)
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing option: {e}")
                    continue
            
            self.db.commit()
            logger.info(f"Stored {stored_count} options in database")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing options chain: {e}")
            self.db.rollback()
            return 0
    
    def collect_and_store(
        self,
        symbol: str,
        target_dte: Optional[int] = None
    ) -> int:
        """
        Collect and store options chain
        
        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            
        Returns:
            Number of options stored
        """
        options = self.collect_chain(symbol, target_dte)
        
        if not options:
            return 0
        
        return self.store_chain(options)

