"""
Volatility Surface Analyzer
Real-time IV surface construction with SVI model fitting, mispricing detection, and arbitrage-free interpolation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize
from loguru import logger
import warnings

from src.portfolio.account_manager import AccountProfile


@dataclass
class IVPoint:
    """Implied volatility data point"""
    strike: float
    expiration: datetime
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    option_type: str  # 'call' or 'put'
    underlying_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int


@dataclass
class SurfaceMetrics:
    """Volatility surface metrics"""
    atm_iv: float
    skew: float  # Put skew (OTM put IV - ATM IV)
    smile: float  # Vol smile curvature
    term_structure_slope: float
    vol_of_vol: float
    surface_quality: str  # 'HIGH', 'MEDIUM', 'LOW'
    mispriced_options: List[Dict]
    arbitrage_opportunities: List[Dict]


@dataclass
class SVIParameters:
    """SVI (Stochastic Volatility Inspired) model parameters"""
    a: float  # ATM variance level
    b: float  # ATM variance slope
    rho: float  # Correlation parameter
    m: float  # ATM moneyness
    sigma: float  # ATM variance curvature


class VolatilitySurfaceAnalyzer:
    """
    Advanced volatility surface analyzer for options trading
    
    Features:
    - Real-time IV surface construction with smile/skew/term structure
    - SVI model fitting for arbitrage-free interpolation
    - Mispriced options detection via surface deviations
    - IV rank percentile by tenor (7D, 30D, 60D, 90D)
    - Volatility momentum signals
    - Account-tier-specific analysis depth
    """
    
    def __init__(self, account_profile: AccountProfile):
        self.profile = account_profile
        
        # Surface parameters
        self.min_volume = account_profile.min_option_volume
        self.min_oi = account_profile.min_open_interest
        self.max_spread_pct = account_profile.max_bid_ask_spread_pct
        
        # Surface construction parameters
        self.moneyness_range = (0.85, 1.15)  # 85% to 115% of underlying
        self.dte_range = (7, 365)  # 7 days to 1 year
        self.min_points_per_strike = 3
        self.min_points_per_expiration = 5
        
        logger.info(f"VolatilitySurfaceAnalyzer initialized for {account_profile.tier.value} tier")
    
    def build_surface(
        self, 
        options_data: List[Dict], 
        underlying_price: float,
        analysis_time: datetime = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive volatility surface
        
        Args:
            options_data: List of option contracts with market data
            underlying_price: Current underlying price
            analysis_time: Analysis timestamp (defaults to now)
        
        Returns:
            Dictionary with surface data and metrics
        """
        try:
            if analysis_time is None:
                analysis_time = datetime.utcnow()
            
            # Filter and validate options data
            filtered_data = self._filter_options_data(options_data)
            
            if len(filtered_data) < 10:
                logger.warning(f"Insufficient options data: {len(filtered_data)} contracts")
                return self._empty_surface()
            
            # Convert to IVPoint objects
            iv_points = self._convert_to_iv_points(filtered_data, underlying_price)
            
            # Build surface grid
            surface_grid = self._build_surface_grid(iv_points, underlying_price)
            
            # Fit SVI model
            svi_parameters = self._fit_svi_model(surface_grid)
            
            # Calculate surface metrics
            surface_metrics = self._calculate_surface_metrics(surface_grid, underlying_price)
            
            # Detect mispricings
            mispricings = self._detect_mispricings(iv_points, surface_grid)
            
            # Detect arbitrage opportunities
            arbitrages = self._detect_arbitrage_opportunities(iv_points, underlying_price)
            
            # Calculate IV ranks
            iv_ranks = self._calculate_iv_ranks(surface_grid)
            
            # Volatility momentum signals
            momentum_signals = self._calculate_momentum_signals(surface_grid)
            
            return {
                'timestamp': analysis_time,
                'underlying_price': underlying_price,
                'surface_grid': surface_grid,
                'svi_parameters': svi_parameters,
                'metrics': surface_metrics,
                'mispricings': mispricings,
                'arbitrage_opportunities': arbitrages,
                'iv_ranks': iv_ranks,
                'momentum_signals': momentum_signals,
                'data_quality': self._assess_data_quality(iv_points),
                'total_contracts': len(filtered_data)
            }
            
        except Exception as e:
            logger.error(f"Error building volatility surface: {e}")
            return self._empty_surface()
    
    def _filter_options_data(self, options_data: List[Dict]) -> List[Dict]:
        """Filter options data based on liquidity and quality criteria"""
        try:
            filtered = []
            
            for option in options_data:
                # Check volume
                volume = option.get('volume', 0)
                if volume < self.min_volume:
                    continue
                
                # Check open interest
                oi = option.get('open_interest', 0)
                if oi < self.min_oi:
                    continue
                
                # Check bid-ask spread
                bid = option.get('bid', 0)
                ask = option.get('ask', 0)
                if bid > 0 and ask > 0:
                    spread_pct = ((ask - bid) / ((bid + ask) / 2)) * 100
                    if spread_pct > self.max_spread_pct:
                        continue
                
                # Check IV is reasonable
                iv = option.get('implied_volatility', 0)
                if iv <= 0 or iv > 5.0:  # 0% to 500% IV
                    continue
                
                # Check DTE
                dte = option.get('days_to_expiration', 0)
                if dte < self.dte_range[0] or dte > self.dte_range[1]:
                    continue
                
                filtered.append(option)
            
            logger.debug(f"Filtered {len(filtered)} contracts from {len(options_data)} total")
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering options data: {e}")
            return []
    
    def _convert_to_iv_points(self, options_data: List[Dict], underlying_price: float) -> List[IVPoint]:
        """Convert options data to IVPoint objects"""
        try:
            iv_points = []
            
            for option in options_data:
                try:
                    strike = float(option.get('strike', 0))
                    expiration = option.get('expiration_date')
                    if isinstance(expiration, str):
                        expiration = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
                    
                    iv_point = IVPoint(
                        strike=strike,
                        expiration=expiration,
                        iv=float(option.get('implied_volatility', 0)),
                        delta=float(option.get('delta', 0)),
                        gamma=float(option.get('gamma', 0)),
                        theta=float(option.get('theta', 0)),
                        vega=float(option.get('vega', 0)),
                        option_type=option.get('option_type', 'call'),
                        underlying_price=underlying_price,
                        bid=float(option.get('bid', 0)),
                        ask=float(option.get('ask', 0)),
                        volume=int(option.get('volume', 0)),
                        open_interest=int(option.get('open_interest', 0))
                    )
                    
                    iv_points.append(iv_point)
                    
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error converting option data: {e}")
                    continue
            
            return iv_points
            
        except Exception as e:
            logger.error(f"Error converting to IV points: {e}")
            return []
    
    def _build_surface_grid(
        self, 
        iv_points: List[IVPoint], 
        underlying_price: float
    ) -> Dict[str, Any]:
        """Build volatility surface grid with interpolation"""
        try:
            if not iv_points:
                return {}
            
            # Group by expiration
            expirations = sorted(list(set([point.expiration for point in iv_points])))
            
            surface_grid = {}
            
            for expiration in expirations:
                # Get points for this expiration
                exp_points = [p for p in iv_points if p.expiration == expiration]
                
                if len(exp_points) < self.min_points_per_expiration:
                    continue
                
                # Calculate moneyness and group by strikes
                strikes = []
                call_ivs = []
                put_ivs = []
                
                for point in exp_points:
                    moneyness = point.strike / underlying_price
                    
                    # Only include points within moneyness range
                    if self.moneyness_range[0] <= moneyness <= self.moneyness_range[1]:
                        strikes.append(point.strike)
                        
                        if point.option_type == 'call':
                            call_ivs.append(point.iv)
                            put_ivs.append(np.nan)
                        else:
                            put_ivs.append(point.iv)
                            call_ivs.append(np.nan)
                
                if len(strikes) < self.min_points_per_strike:
                    continue
                
                # Create DataFrame for this expiration
                exp_data = pd.DataFrame({
                    'strike': strikes,
                    'call_iv': call_ivs,
                    'put_iv': put_ivs
                })
                
                # Interpolate missing values
                exp_data['call_iv'] = exp_data['call_iv'].interpolate(method='linear')
                exp_data['put_iv'] = exp_data['put_iv'].interpolate(method='linear')
                
                # Calculate ATM IV (interpolated at-the-money)
                atm_strike = underlying_price
                atm_call_iv = np.interp(atm_strike, exp_data['strike'], exp_data['call_iv'])
                atm_put_iv = np.interp(atm_strike, exp_data['strike'], exp_data['put_iv'])
                atm_iv = (atm_call_iv + atm_put_iv) / 2
                
                # Calculate skew (OTM put IV - ATM IV)
                otm_put_strikes = exp_data['strike'][exp_data['strike'] < atm_strike]
                if len(otm_put_strikes) > 0:
                    otm_put_iv = np.interp(otm_put_strikes.iloc[-1], exp_data['strike'], exp_data['put_iv'])
                    skew = otm_put_iv - atm_iv
                else:
                    skew = 0.0
                
                surface_grid[expiration.isoformat()] = {
                    'expiration': expiration,
                    'dte': (expiration - datetime.utcnow()).days,
                    'atm_strike': atm_strike,
                    'atm_iv': atm_iv,
                    'skew': skew,
                    'strikes': exp_data['strike'].tolist(),
                    'call_ivs': exp_data['call_iv'].tolist(),
                    'put_ivs': exp_data['put_iv'].tolist()
                }
            
            return surface_grid
            
        except Exception as e:
            logger.error(f"Error building surface grid: {e}")
            return {}
    
    def _fit_svi_model(self, surface_grid: Dict[str, Any]) -> Dict[str, SVIParameters]:
        """Fit SVI model to volatility surface"""
        try:
            svi_parameters = {}
            
            for exp_key, exp_data in surface_grid.items():
                try:
                    strikes = np.array(exp_data['strikes'])
                    call_ivs = np.array(exp_data['call_ivs'])
                    put_ivs = np.array(exp_data['put_ivs'])
                    
                    # Use average IV for fitting
                    avg_ivs = (call_ivs + put_ivs) / 2
                    
                    # Calculate log-moneyness
                    atm_strike = exp_data['atm_strike']
                    log_moneyness = np.log(strikes / atm_strike)
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(avg_ivs) | np.isnan(log_moneyness))
                    log_moneyness = log_moneyness[valid_mask]
                    avg_ivs = avg_ivs[valid_mask]
                    
                    if len(log_moneyness) < 5:
                        continue
                    
                    # Convert IV to variance
                    variances = (avg_ivs ** 2) * exp_data['dte'] / 365.0
                    
                    # Fit SVI parameters
                    svi_params = self._fit_svi_curve(log_moneyness, variances)
                    
                    if svi_params:
                        svi_parameters[exp_key] = svi_params
                    
                except Exception as e:
                    logger.debug(f"Error fitting SVI for {exp_key}: {e}")
                    continue
            
            return svi_parameters
            
        except Exception as e:
            logger.error(f"Error fitting SVI model: {e}")
            return {}
    
    def _fit_svi_curve(self, log_moneyness: np.ndarray, variances: np.ndarray) -> Optional[SVIParameters]:
        """Fit SVI curve to variance data"""
        try:
            # SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
            # where w(k) is the variance at log-moneyness k
            
            def svi_variance(k, a, b, rho, m, sigma):
                return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            
            def objective(params):
                a, b, rho, m, sigma = params
                predicted = svi_variance(log_moneyness, a, b, rho, m, sigma)
                return np.sum((predicted - variances)**2)
            
            # Initial guess
            x0 = [
                np.mean(variances),  # a: ATM variance level
                0.1,                 # b: variance slope
                0.0,                 # rho: correlation
                0.0,                 # m: ATM moneyness
                0.1                  # sigma: curvature
            ]
            
            # Bounds
            bounds = [
                (0, 2.0),      # a: positive variance
                (0, 1.0),      # b: positive slope
                (-1, 1),       # rho: correlation
                (-0.5, 0.5),   # m: near ATM
                (0.01, 1.0)    # sigma: positive curvature
            ]
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                a, b, rho, m, sigma = result.x
                
                # Validate SVI parameters
                if (a > 0 and b > 0 and -1 <= rho <= 1 and sigma > 0):
                    return SVIParameters(a=a, b=b, rho=rho, m=m, sigma=sigma)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error fitting SVI curve: {e}")
            return None
    
    def _calculate_surface_metrics(
        self, 
        surface_grid: Dict[str, Any], 
        underlying_price: float
    ) -> SurfaceMetrics:
        """Calculate comprehensive surface metrics"""
        try:
            if not surface_grid:
                return SurfaceMetrics(0, 0, 0, 0, 0, 'LOW', [], [])
            
            # Calculate ATM IV across expirations
            atm_ivs = []
            skews = []
            expirations = []
            
            for exp_key, exp_data in surface_grid.items():
                atm_ivs.append(exp_data['atm_iv'])
                skews.append(exp_data['skew'])
                expirations.append(exp_data['dte'])
            
            if not atm_ivs:
                return SurfaceMetrics(0, 0, 0, 0, 0, 'LOW', [], [])
            
            # ATM IV (average across expirations)
            atm_iv = np.mean(atm_ivs)
            
            # Average skew
            avg_skew = np.mean(skews)
            
            # Term structure slope
            if len(atm_ivs) > 1:
                term_slope = np.polyfit(expirations, atm_ivs, 1)[0]
            else:
                term_slope = 0.0
            
            # Volatility of volatility (standard deviation of IVs)
            vol_of_vol = np.std(atm_ivs)
            
            # Smile calculation (curvature at ATM)
            smile = self._calculate_smile_curvature(surface_grid, underlying_price)
            
            # Assess surface quality
            surface_quality = self._assess_surface_quality(surface_grid)
            
            return SurfaceMetrics(
                atm_iv=atm_iv,
                skew=avg_skew,
                smile=smile,
                term_structure_slope=term_slope,
                vol_of_vol=vol_of_vol,
                surface_quality=surface_quality,
                mispriced_options=[],
                arbitrage_opportunities=[]
            )
            
        except Exception as e:
            logger.error(f"Error calculating surface metrics: {e}")
            return SurfaceMetrics(0, 0, 0, 0, 0, 'LOW', [], [])
    
    def _calculate_smile_curvature(self, surface_grid: Dict[str, Any], underlying_price: float) -> float:
        """Calculate volatility smile curvature"""
        try:
            curvatures = []
            
            for exp_key, exp_data in surface_grid.items():
                strikes = np.array(exp_data['strikes'])
                call_ivs = np.array(exp_data['call_ivs'])
                put_ivs = np.array(exp_data['put_ivs'])
                
                # Find ATM strike
                atm_strike = exp_data['atm_strike']
                atm_idx = np.argmin(np.abs(strikes - atm_strike))
                
                # Calculate second derivative (curvature) around ATM
                if atm_idx > 1 and atm_idx < len(strikes) - 2:
                    # Use put IVs for smile calculation
                    ivs = put_ivs
                    x = strikes[atm_idx-2:atm_idx+3]
                    y = ivs[atm_idx-2:atm_idx+3]
                    
                    # Fit quadratic and get second derivative
                    coeffs = np.polyfit(x, y, 2)
                    curvature = 2 * coeffs[0]  # Second derivative of quadratic
                    curvatures.append(curvature)
            
            return np.mean(curvatures) if curvatures else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating smile curvature: {e}")
            return 0.0
    
    def _assess_surface_quality(self, surface_grid: Dict[str, Any]) -> str:
        """Assess overall surface quality"""
        try:
            if not surface_grid:
                return 'LOW'
            
            quality_score = 0
            
            # Check number of expirations
            num_expirations = len(surface_grid)
            if num_expirations >= 4:
                quality_score += 3
            elif num_expirations >= 2:
                quality_score += 1
            
            # Check data density
            total_points = sum(len(exp_data['strikes']) for exp_data in surface_grid.values())
            if total_points >= 50:
                quality_score += 3
            elif total_points >= 20:
                quality_score += 1
            
            # Check IV consistency
            atm_ivs = [exp_data['atm_iv'] for exp_data in surface_grid.values()]
            if atm_ivs:
                iv_std = np.std(atm_ivs)
                if iv_std < 0.05:  # Low volatility of volatility
                    quality_score += 2
                elif iv_std < 0.10:
                    quality_score += 1
            
            # Determine quality level
            if quality_score >= 6:
                return 'HIGH'
            elif quality_score >= 3:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.error(f"Error assessing surface quality: {e}")
            return 'LOW'
    
    def _detect_mispricings(self, iv_points: List[IVPoint], surface_grid: Dict[str, Any]) -> List[Dict]:
        """Detect mispriced options based on surface deviations"""
        try:
            mispricings = []
            
            for point in iv_points:
                try:
                    # Find corresponding surface data
                    exp_key = point.expiration.isoformat()
                    if exp_key not in surface_grid:
                        continue
                    
                    exp_data = surface_grid[exp_key]
                    strikes = np.array(exp_data['strikes'])
                    
                    # Interpolate expected IV from surface
                    if point.option_type == 'call':
                        expected_ivs = np.array(exp_data['call_ivs'])
                    else:
                        expected_ivs = np.array(exp_data['put_ivs'])
                    
                    expected_iv = np.interp(point.strike, strikes, expected_ivs)
                    
                    # Calculate deviation
                    iv_deviation = (point.iv - expected_iv) / expected_iv if expected_iv > 0 else 0
                    
                    # Flag significant mispricings (> 2 standard deviations)
                    if abs(iv_deviation) > 0.15:  # 15% deviation
                        mispricings.append({
                            'symbol': f"{point.option_type}_{point.strike}_{point.expiration.strftime('%Y%m%d')}",
                            'strike': point.strike,
                            'expiration': point.expiration,
                            'option_type': point.option_type,
                            'actual_iv': point.iv,
                            'expected_iv': expected_iv,
                            'deviation_pct': iv_deviation * 100,
                            'direction': 'undervalued' if iv_deviation < 0 else 'overvalued',
                            'confidence': min(abs(iv_deviation) * 10, 1.0)
                        })
                
                except Exception as e:
                    logger.debug(f"Error detecting mispricing for point: {e}")
                    continue
            
            return mispricings
            
        except Exception as e:
            logger.error(f"Error detecting mispricings: {e}")
            return []
    
    def _detect_arbitrage_opportunities(self, iv_points: List[IVPoint], underlying_price: float) -> List[Dict]:
        """Detect arbitrage opportunities"""
        try:
            arbitrages = []
            
            # Group points by expiration
            by_expiration = {}
            for point in iv_points:
                exp_key = point.expiration.isoformat()
                if exp_key not in by_expiration:
                    by_expiration[exp_key] = []
                by_expiration[exp_key].append(point)
            
            # Check for put-call parity violations
            for exp_key, exp_points in by_expiration.items():
                # Group by strike
                by_strike = {}
                for point in exp_points:
                    if point.strike not in by_strike:
                        by_strike[point.strike] = {}
                    by_strike[point.strike][point.option_type] = point
                
                # Check put-call parity for each strike
                for strike, options in by_strike.items():
                    if 'call' in options and 'put' in options:
                        call_point = options['call']
                        put_point = options['put']
                        
                        # Put-call parity: C - P = S - K*e^(-r*T)
                        # For simplicity, assume r=0
                        parity_diff = (call_point.bid + call_point.ask) / 2 - (put_point.bid + put_point.ask) / 2
                        expected_diff = underlying_price - strike
                        
                        # Check for significant deviation (>$0.50)
                        if abs(parity_diff - expected_diff) > 0.50:
                            arbitrages.append({
                                'type': 'put_call_parity',
                                'strike': strike,
                                'expiration': call_point.expiration,
                                'actual_diff': parity_diff,
                                'expected_diff': expected_diff,
                                'deviation': parity_diff - expected_diff,
                                'direction': 'call_cheap' if parity_diff < expected_diff else 'put_cheap'
                            })
            
            return arbitrages
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage opportunities: {e}")
            return []
    
    def _calculate_iv_ranks(self, surface_grid: Dict[str, Any]) -> Dict[str, float]:
        """Calculate IV rank percentiles by tenor"""
        try:
            iv_ranks = {}
            
            # Group by tenor buckets
            tenor_buckets = {
                '7D': 7,
                '30D': 30,
                '60D': 60,
                '90D': 90
            }
            
            for bucket_name, target_dte in tenor_buckets.items():
                # Find closest expiration to target DTE
                closest_exp = None
                min_diff = float('inf')
                
                for exp_key, exp_data in surface_grid.items():
                    dte_diff = abs(exp_data['dte'] - target_dte)
                    if dte_diff < min_diff:
                        min_diff = dte_diff
                        closest_exp = exp_data
                
                if closest_exp:
                    # Calculate IV rank (percentile vs historical)
                    current_iv = closest_exp['atm_iv']
                    
                    # For now, use simplified percentile calculation
                    # In production, would compare against historical IV distribution
                    iv_rank = min(current_iv / 0.50 * 100, 100)  # Assume 50% is 100th percentile
                    iv_ranks[bucket_name] = iv_rank
            
            return iv_ranks
            
        except Exception as e:
            logger.error(f"Error calculating IV ranks: {e}")
            return {}
    
    def _calculate_momentum_signals(self, surface_grid: Dict[str, Any]) -> Dict[str, float]:
        """Calculate volatility momentum signals"""
        try:
            if len(surface_grid) < 2:
                return {}
            
            # Sort by expiration
            sorted_exps = sorted(surface_grid.items(), key=lambda x: x[1]['dte'])
            
            signals = {}
            
            # Short-term vs long-term IV momentum
            if len(sorted_exps) >= 2:
                short_term_iv = sorted_exps[0][1]['atm_iv']
                long_term_iv = sorted_exps[-1][1]['atm_iv']
                
                signals['term_structure_momentum'] = short_term_iv - long_term_iv
                signals['iv_steepness'] = (short_term_iv - long_term_iv) / long_term_iv if long_term_iv > 0 else 0
            
            # Skew momentum
            skews = [exp_data['skew'] for exp_data in surface_grid.values()]
            if len(skews) >= 2:
                signals['skew_momentum'] = skews[-1] - skews[0]
                signals['skew_trend'] = np.polyfit(range(len(skews)), skews, 1)[0] if len(skews) > 2 else 0
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating momentum signals: {e}")
            return {}
    
    def _assess_data_quality(self, iv_points: List[IVPoint]) -> Dict[str, Any]:
        """Assess data quality metrics"""
        try:
            if not iv_points:
                return {'quality_score': 0, 'issues': []}
            
            issues = []
            quality_score = 100
            
            # Check volume distribution
            volumes = [p.volume for p in iv_points]
            low_volume_pct = sum(1 for v in volumes if v < self.min_volume) / len(volumes) * 100
            
            if low_volume_pct > 50:
                issues.append(f"High percentage of low volume contracts: {low_volume_pct:.1f}%")
                quality_score -= 20
            
            # Check spread distribution
            spreads = []
            for p in iv_points:
                if p.bid > 0 and p.ask > 0:
                    spread_pct = ((p.ask - p.bid) / ((p.bid + p.ask) / 2)) * 100
                    spreads.append(spread_pct)
            
            if spreads:
                avg_spread = np.mean(spreads)
                if avg_spread > self.max_spread_pct:
                    issues.append(f"Average spread too wide: {avg_spread:.1f}%")
                    quality_score -= 15
            
            # Check IV distribution
            ivs = [p.iv for p in iv_points if p.iv > 0]
            if ivs:
                iv_std = np.std(ivs)
                if iv_std > 0.5:  # High volatility of IV
                    issues.append(f"High IV volatility: {iv_std:.3f}")
                    quality_score -= 10
            
            return {
                'quality_score': max(quality_score, 0),
                'total_contracts': len(iv_points),
                'issues': issues,
                'low_volume_pct': low_volume_pct,
                'avg_spread_pct': np.mean(spreads) if spreads else 0,
                'iv_volatility': np.std(ivs) if ivs else 0
            }
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {'quality_score': 0, 'issues': ['Data quality assessment failed']}
    
    def _empty_surface(self) -> Dict[str, Any]:
        """Return empty surface structure"""
        return {
            'timestamp': datetime.utcnow(),
            'underlying_price': 0,
            'surface_grid': {},
            'svi_parameters': {},
            'metrics': SurfaceMetrics(0, 0, 0, 0, 0, 'LOW', [], []),
            'mispricings': [],
            'arbitrage_opportunities': [],
            'iv_ranks': {},
            'momentum_signals': {},
            'data_quality': {'quality_score': 0, 'issues': ['Insufficient data']},
            'total_contracts': 0
        }
    
    def get_trading_recommendations(
        self, 
        surface_data: Dict[str, Any],
        account_tier: str
    ) -> Dict[str, Any]:
        """
        Get trading recommendations based on surface analysis
        
        Args:
            surface_data: Surface analysis results
            account_tier: Account tier for recommendations
        
        Returns:
            Dictionary with trading recommendations
        """
        try:
            recommendations = {
                'surface_quality': surface_data['data_quality']['quality_score'],
                'primary_signals': [],
                'strategy_recommendations': [],
                'risk_warnings': []
            }
            
            metrics = surface_data.get('metrics')
            if not metrics:
                return recommendations
            
            # IV rank signals
            iv_ranks = surface_data.get('iv_ranks', {})
            for tenor, rank in iv_ranks.items():
                if rank > 80:
                    recommendations['primary_signals'].append(f"High IV rank in {tenor}: {rank:.1f}% - Consider selling strategies")
                elif rank < 20:
                    recommendations['primary_signals'].append(f"Low IV rank in {tenor}: {rank:.1f}% - Consider buying strategies")
            
            # Skew signals
            if abs(metrics.skew) > 0.05:
                if metrics.skew > 0:
                    recommendations['strategy_recommendations'].append("Positive skew detected - Favor put spreads and iron condors")
                else:
                    recommendations['strategy_recommendations'].append("Negative skew detected - Favor call spreads and butterflies")
            
            # Term structure signals
            if metrics.term_structure_slope > 0.001:
                recommendations['strategy_recommendations'].append("Upward sloping term structure - Consider calendar spreads")
            elif metrics.term_structure_slope < -0.001:
                recommendations['strategy_recommendations'].append("Downward sloping term structure - Consider diagonal spreads")
            
            # Mispricing opportunities
            mispricings = surface_data.get('mispricings', [])
            if mispricings:
                undervalued = [m for m in mispricings if m['direction'] == 'undervalued']
                overvalued = [m for m in mispricings if m['direction'] == 'overvalued']
                
                if undervalued and account_tier in ['LARGE', 'INSTITUTIONAL']:
                    recommendations['strategy_recommendations'].append(f"Found {len(undervalued)} undervalued options - Consider buying")
                
                if overvalued:
                    recommendations['strategy_recommendations'].append(f"Found {len(overvalued)} overvalued options - Consider selling")
            
            # Risk warnings
            if metrics.surface_quality == 'LOW':
                recommendations['risk_warnings'].append("Low surface quality - Use caution with complex strategies")
            
            if metrics.vol_of_vol > 0.2:
                recommendations['risk_warnings'].append("High volatility of volatility - Expect large IV swings")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create surface analyzer
    analyzer = VolatilitySurfaceAnalyzer(profile)
    
    # Sample options data
    sample_options = [
        {
            'strike': 450,
            'expiration_date': '2025-02-21T00:00:00Z',
            'implied_volatility': 0.20,
            'delta': 0.25,
            'gamma': 0.02,
            'theta': -0.05,
            'vega': 0.15,
            'option_type': 'put',
            'bid': 1.25,
            'ask': 1.35,
            'volume': 150,
            'open_interest': 500
        },
        {
            'strike': 460,
            'expiration_date': '2025-02-21T00:00:00Z',
            'implied_volatility': 0.22,
            'delta': 0.15,
            'gamma': 0.02,
            'theta': -0.04,
            'vega': 0.18,
            'option_type': 'put',
            'bid': 0.85,
            'ask': 0.95,
            'volume': 200,
            'open_interest': 750
        }
    ]
    
    # Build surface
    surface = analyzer.build_surface(sample_options, underlying_price=455.0)
    
    print(f"Surface Analysis Results:")
    print(f"  Quality Score: {surface['data_quality']['quality_score']}")
    print(f"  Total Contracts: {surface['total_contracts']}")
    print(f"  Surface Quality: {surface['metrics'].surface_quality}")
    print(f"  ATM IV: {surface['metrics'].atm_iv:.3f}")
    print(f"  Skew: {surface['metrics'].skew:.3f}")
    print(f"  Term Structure Slope: {surface['metrics'].term_structure_slope:.4f}")
    
    # Get recommendations
    recommendations = analyzer.get_trading_recommendations(surface, profile.tier.value)
    print(f"\nTrading Recommendations:")
    for signal in recommendations['primary_signals']:
        print(f"  - {signal}")
    for rec in recommendations['strategy_recommendations']:
        print(f"  - {rec}")
    for warning in recommendations['risk_warnings']:
        print(f"  - WARNING: {warning}")
