"""
Options Feature Engineer
Adds real options data (Greeks, IV, OI) to ML features using Polygon
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from loguru import logger

from src.market_data.polygon_options import PolygonOptionsClient


class OptionsFeatureEngineer:
    """
    Engineer ML features from real options data
    Uses Polygon to get actual Greeks, IV, and Open Interest
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize options feature engineer
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        
        try:
            self.polygon_options = PolygonOptionsClient()
            self.enabled = True
            logger.info("Options Feature Engineer initialized with Polygon")
        except Exception as e:
            logger.warning(f"Polygon options not available: {e}")
            self.enabled = False
    
    def add_options_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        target_dte: int = 35
    ) -> pd.DataFrame:
        """
        Add options-based features to DataFrame
        
        Args:
            df: DataFrame with price data
            symbol: Symbol to add features for
            target_dte: Target days to expiration
            
        Returns:
            DataFrame with options features added
        """
        if not self.enabled:
            logger.warning("Polygon options not enabled, skipping options features")
            return self._add_default_features(df)
        
        try:
            logger.info(f"Adding options features for {symbol}")
            
            # Get current options data with Greeks
            options_data = self.polygon_options.get_options_chain_with_greeks(
                underlying=symbol,
                min_dte=target_dte - 10,
                max_dte=target_dte + 10
            )
            
            if not options_data:
                logger.warning(f"No options data for {symbol}, using defaults")
                return self._add_default_features(df)
            
            # Calculate aggregate metrics with detailed logging
            logger.debug(f"Calculating ATM IV for {symbol}...")
            atm_iv = self._get_atm_iv(options_data)
            logger.debug(f"  ATM IV: {atm_iv} (type: {type(atm_iv)})")
            
            logger.debug(f"Calculating put/call ratio for {symbol}...")
            put_call_ratio = self._calculate_put_call_ratio(options_data)
            logger.debug(f"  Put/Call Ratio: {put_call_ratio} (type: {type(put_call_ratio)})")
            
            logger.debug(f"Calculating avg delta for {symbol}...")
            avg_delta_30 = self._get_avg_delta(options_data, delta_target=0.30)
            logger.debug(f"  Avg Delta: {avg_delta_30} (type: {type(avg_delta_30)})")
            
            logger.debug(f"Calculating total OI for {symbol}...")
            total_oi = self._get_total_open_interest(options_data)
            logger.debug(f"  Total OI: {total_oi} (type: {type(total_oi)})")
            
            # Add to all rows (static snapshot) - ensure all are numeric
            logger.debug(f"Adding aggregate metrics to dataframe...")
            df['atm_iv'] = float(atm_iv)
            df['put_call_ratio'] = float(put_call_ratio)
            df['avg_delta_30'] = float(avg_delta_30)
            df['total_open_interest'] = float(total_oi)
            logger.debug(f"  Aggregate metrics added successfully")
            
            # Get ATM put/call Greeks
            logger.debug(f"Getting ATM options for {symbol}...")
            atm_put = self._get_atm_option(options_data, 'put')
            atm_call = self._get_atm_option(options_data, 'call')
            logger.debug(f"  ATM Put: {atm_put is not None}, ATM Call: {atm_call is not None}")
            
            if atm_put:
                try:
                    logger.debug(f"  Processing ATM put Greeks...")
                    logger.debug(f"    Raw delta: {atm_put['greeks'].get('delta')} (type: {type(atm_put['greeks'].get('delta'))})")
                    df['atm_put_delta'] = abs(float(atm_put['greeks'].get('delta', 0.5)))
                    df['atm_put_theta'] = float(atm_put['greeks'].get('theta', 0))
                    df['atm_put_vega'] = float(atm_put['greeks'].get('vega', 0))
                    df['atm_put_gamma'] = float(atm_put['greeks'].get('gamma', 0))
                    logger.debug(f"  ATM put Greeks added successfully")
                except (ValueError, TypeError) as e:
                    logger.warning(f"  Failed to add ATM put Greeks: {e}, using defaults")
                    df = self._add_default_greeks(df, 'put')
            else:
                logger.debug(f"  No ATM put found, using defaults")
                df = self._add_default_greeks(df, 'put')
            
            if atm_call:
                try:
                    logger.debug(f"  Processing ATM call Greeks...")
                    logger.debug(f"    Raw delta: {atm_call['greeks'].get('delta')} (type: {type(atm_call['greeks'].get('delta'))})")
                    df['atm_call_delta'] = float(atm_call['greeks'].get('delta', 0.5))
                    df['atm_call_theta'] = float(atm_call['greeks'].get('theta', 0))
                    df['atm_call_vega'] = float(atm_call['greeks'].get('vega', 0))
                    df['atm_call_gamma'] = float(atm_call['greeks'].get('gamma', 0))
                    logger.debug(f"  ATM call Greeks added successfully")
                except (ValueError, TypeError) as e:
                    logger.warning(f"  Failed to add ATM call Greeks: {e}, using defaults")
                    df = self._add_default_greeks(df, 'call')
            else:
                logger.debug(f"  No ATM call found, using defaults")
                df = self._add_default_greeks(df, 'call')
            
            # Calculate skew
            logger.debug(f"Calculating IV skew for {symbol}...")
            iv_skew = self._calculate_skew(options_data)
            logger.debug(f"  IV Skew: {iv_skew} (type: {type(iv_skew)})")
            df['iv_skew'] = float(iv_skew)
            logger.debug(f"  IV skew added successfully")
            
            # Volume metrics
            logger.debug(f"Calculating total volume for {symbol}...")
            option_volume = self._get_total_volume(options_data)
            logger.debug(f"  Option Volume: {option_volume} (type: {type(option_volume)})")
            df['option_volume'] = float(option_volume)
            logger.debug(f"  Option volume added successfully")
            
            logger.info(f"Added {8 + len([c for c in df.columns if 'atm_' in c])} options features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding options features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._add_default_features(df)
    
    def _get_atm_iv(self, options_data: List[Dict]) -> float:
        """Get ATM implied volatility"""
        ivs = [opt.get('implied_volatility', 0) 
               for opt in options_data 
               if opt.get('implied_volatility')]
        
        return np.median(ivs) if ivs else 0.25
    
    def _calculate_put_call_ratio(self, options_data: List[Dict]) -> float:
        """Calculate put/call volume ratio"""
        put_volume = sum(opt.get('volume', 0) 
                        for opt in options_data 
                        if opt.get('type') == 'put')
        
        call_volume = sum(opt.get('volume', 0) 
                         for opt in options_data 
                         if opt.get('type') == 'call')
        
        if call_volume == 0:
            return 1.0
        
        return put_volume / call_volume
    
    def _get_avg_delta(self, options_data: List[Dict], delta_target: float = 0.30) -> float:
        """Get average delta around target"""
        deltas = []
        for opt in options_data:
            if opt.get('greeks', {}).get('delta'):
                try:
                    delta = float(opt['greeks']['delta'])
                    if abs(abs(delta) - delta_target) < 0.10:
                        deltas.append(abs(delta))
                except (ValueError, TypeError):
                    continue
        
        return np.mean(deltas) if deltas else delta_target
    
    def _get_total_open_interest(self, options_data: List[Dict]) -> int:
        """Get total open interest"""
        return sum(opt.get('open_interest', 0) for opt in options_data)
    
    def _get_total_volume(self, options_data: List[Dict]) -> int:
        """Get total option volume"""
        return sum(opt.get('volume', 0) for opt in options_data)
    
    def _get_atm_option(self, options_data: List[Dict], option_type: str) -> Optional[Dict]:
        """Find ATM option of specified type"""
        candidates = [opt for opt in options_data if opt.get('type') == option_type]
        
        if not candidates:
            return None
        
        # Find closest to ATM (delta closest to 0.50 for calls, -0.50 for puts)
        target_delta = 0.50 if option_type == 'call' else -0.50
        
        for opt in candidates:
            delta = opt.get('greeks', {}).get('delta')
            if delta:
                try:
                    delta_float = float(delta)
                    opt['_delta_diff'] = abs(delta_float - target_delta)
                except (ValueError, TypeError):
                    continue
        
        candidates = [opt for opt in candidates if '_delta_diff' in opt]
        
        if candidates:
            candidates.sort(key=lambda x: x['_delta_diff'])
            return candidates[0]
        
        return None
    
    def _calculate_skew(self, options_data: List[Dict]) -> float:
        """Calculate IV skew (OTM puts vs OTM calls)"""
        try:
            # Get OTM put IVs (delta around -0.25)
            otm_put_ivs = []
            for opt in options_data:
                if (opt.get('type') == 'put' 
                    and opt.get('greeks', {}).get('delta')
                    and opt.get('implied_volatility')):
                    try:
                        delta = float(opt['greeks']['delta'])
                        if -0.35 < delta < -0.15:
                            otm_put_ivs.append(opt['implied_volatility'])
                    except (ValueError, TypeError):
                        continue
            
            # Get OTM call IVs (delta around 0.25)
            otm_call_ivs = []
            for opt in options_data:
                if (opt.get('type') == 'call'
                    and opt.get('greeks', {}).get('delta')
                    and opt.get('implied_volatility')):
                    try:
                        delta = float(opt['greeks']['delta'])
                        if 0.15 < delta < 0.35:
                            otm_call_ivs.append(opt['implied_volatility'])
                    except (ValueError, TypeError):
                        continue
            
            if otm_put_ivs and otm_call_ivs:
                avg_put_iv = np.mean(otm_put_ivs)
                avg_call_iv = np.mean(otm_call_ivs)
                return avg_put_iv - avg_call_iv  # Positive = put skew
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating skew: {e}")
            return 0.0
    
    def _add_default_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default options features when real data not available"""
        df['atm_iv'] = 0.25
        df['put_call_ratio'] = 1.0
        df['avg_delta_30'] = 0.30
        df['total_open_interest'] = 0
        df['iv_skew'] = 0.0
        df['option_volume'] = 0
        
        # Default Greeks
        df = self._add_default_greeks(df, 'put')
        df = self._add_default_greeks(df, 'call')
        
        return df
    
    def _add_default_greeks(self, df: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """Add default Greek values"""
        prefix = f'atm_{option_type}'
        
        if option_type == 'put':
            df[f'{prefix}_delta'] = 0.50
            df[f'{prefix}_theta'] = -0.05
            df[f'{prefix}_vega'] = 0.15
            df[f'{prefix}_gamma'] = 0.01
        else:  # call
            df[f'{prefix}_delta'] = 0.50
            df[f'{prefix}_theta'] = -0.05
            df[f'{prefix}_vega'] = 0.15
            df[f'{prefix}_gamma'] = 0.01
        
        return df

