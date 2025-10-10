"""
Strategy Tester Module
Test specific options strategies on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from .engine import BacktestEngine, BacktestResult
from .metrics import PerformanceMetrics
from src.database.models import IndexTickData


class StrategyTester:
    """
    Test options strategies on historical data
    Supports: Bull Put Spreads, Iron Condors, Cash-Secured Puts
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize strategy tester
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.metrics_calc = PerformanceMetrics()
        logger.info("Strategy Tester initialized")
    
    def test_bull_put_spread_strategy(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        params: Dict = None
    ) -> BacktestResult:
        """
        Test bull put spread strategy
        
        Args:
            symbol: Symbol to test
            start_date: Start date for backtest
            end_date: End date for backtest
            params: Strategy parameters
            
        Returns:
            BacktestResult
        """
        logger.info(f"Testing Bull Put Spread strategy for {symbol}")
        
        # Default parameters
        if params is None:
            params = {
                'target_delta': 0.30,
                'target_dte': 35,
                'width': 5.0,
                'min_credit': 0.25,
                'take_profit_pct': 0.50,
                'stop_loss_pct': 2.0
            }
        
        # Get historical data
        data = self._get_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            logger.error(f"No historical data for {symbol}")
            return BacktestResult()
        
        # Create strategy function
        def bull_put_spread_signal(historical_data: pd.DataFrame, params: Dict) -> Optional[Dict]:
            """Generate bull put spread signals"""
            if len(historical_data) < 50:
                return None
            
            current = historical_data.iloc[-1]
            current_price = current['price']
            
            # Check if we should enter (simplified logic)
            # In real backtest, would check IV rank, trend, etc.
            
            # Calculate moving average for trend
            sma_20 = historical_data['price'].tail(20).mean()
            
            # Only enter if price is above SMA (bullish)
            if current_price < sma_20:
                return None
            
            # Simulate finding strikes
            short_strike = current_price * 0.95  # 5% OTM
            long_strike = short_strike - params['width']
            
            # Estimate credit (simplified)
            credit = params['min_credit'] + (params['width'] * 0.05)
            
            max_profit = credit * 100
            max_loss = (params['width'] - credit) * 100
            
            return {
                'action': 'SELL',
                'symbol': symbol,
                'strategy': 'bull_put_spread',
                'entry_price': credit,
                'quantity': 1,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'metadata': {
                    'short_strike': short_strike,
                    'long_strike': long_strike,
                    'expiration': current['timestamp'] + timedelta(days=params['target_dte'])
                }
            }
        
        # Run backtest
        engine = BacktestEngine(starting_capital=10000.0)
        result = engine.run_backtest(data, bull_put_spread_signal, params)
        
        return result
    
    def test_iron_condor_strategy(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        params: Dict = None
    ) -> BacktestResult:
        """
        Test iron condor strategy
        
        Args:
            symbol: Symbol to test
            start_date: Start date for backtest
            end_date: End date for backtest
            params: Strategy parameters
            
        Returns:
            BacktestResult
        """
        logger.info(f"Testing Iron Condor strategy for {symbol}")
        
        # Default parameters
        if params is None:
            params = {
                'target_delta': 0.30,
                'target_dte': 40,
                'put_width': 5.0,
                'call_width': 5.0,
                'min_credit': 0.50,
                'take_profit_pct': 0.50
            }
        
        # Get historical data
        data = self._get_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            logger.error(f"No historical data for {symbol}")
            return BacktestResult()
        
        # Create strategy function
        def iron_condor_signal(historical_data: pd.DataFrame, params: Dict) -> Optional[Dict]:
            """Generate iron condor signals"""
            if len(historical_data) < 50:
                return None
            
            current = historical_data.iloc[-1]
            current_price = current['price']
            
            # Check for ranging market (low volatility)
            recent_prices = historical_data['price'].tail(20)
            volatility = recent_prices.std() / recent_prices.mean()
            
            # Only enter in low volatility (ranging market)
            if volatility > 0.03:  # 3% volatility threshold
                return None
            
            # Simulate finding strikes
            put_short_strike = current_price * 0.95
            put_long_strike = put_short_strike - params['put_width']
            call_short_strike = current_price * 1.05
            call_long_strike = call_short_strike + params['call_width']
            
            # Estimate credit
            credit = params['min_credit'] + (params['put_width'] * 0.04)
            
            max_profit = credit * 100
            max_loss = (max(params['put_width'], params['call_width']) - credit) * 100
            
            return {
                'action': 'SELL',
                'symbol': symbol,
                'strategy': 'iron_condor',
                'entry_price': credit,
                'quantity': 1,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'metadata': {
                    'put_short': put_short_strike,
                    'put_long': put_long_strike,
                    'call_short': call_short_strike,
                    'call_long': call_long_strike,
                    'expiration': current['timestamp'] + timedelta(days=params['target_dte'])
                }
            }
        
        # Run backtest
        engine = BacktestEngine(starting_capital=10000.0)
        result = engine.run_backtest(data, iron_condor_signal, params)
        
        return result
    
    def test_cash_secured_put_strategy(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        params: Dict = None
    ) -> BacktestResult:
        """
        Test cash-secured put strategy
        
        Args:
            symbol: Symbol to test
            start_date: Start date for backtest
            end_date: End date for backtest
            params: Strategy parameters
            
        Returns:
            BacktestResult
        """
        logger.info(f"Testing Cash-Secured Put strategy for {symbol}")
        
        # Default parameters
        if params is None:
            params = {
                'target_delta': 0.30,
                'target_dte': 30,
                'min_premium': 0.50
            }
        
        # Get historical data
        data = self._get_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            return BacktestResult()
        
        # Create strategy function
        def csp_signal(historical_data: pd.DataFrame, params: Dict) -> Optional[Dict]:
            """Generate cash-secured put signals"""
            if len(historical_data) < 50:
                return None
            
            current = historical_data.iloc[-1]
            current_price = current['price']
            
            # Check for bullish trend
            sma_20 = historical_data['price'].tail(20).mean()
            
            if current_price < sma_20:
                return None
            
            # Simulate strike selection
            strike = current_price * 0.95
            premium = params['min_premium'] + (current_price * 0.01)
            
            max_profit = premium * 100
            max_loss = (strike - premium) * 100
            
            return {
                'action': 'SELL',
                'symbol': symbol,
                'strategy': 'cash_secured_put',
                'entry_price': premium,
                'quantity': 1,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'metadata': {
                    'strike': strike,
                    'expiration': current['timestamp'] + timedelta(days=params['target_dte'])
                }
            }
        
        # Run backtest
        engine = BacktestEngine(starting_capital=10000.0)
        result = engine.run_backtest(data, csp_signal, params)
        
        return result
    
    def compare_strategies(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies
        
        Args:
            symbol: Symbol to test
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of strategy results
        """
        logger.info(f"Comparing strategies for {symbol}")
        
        results = {}
        
        # Test each strategy
        results['bull_put_spread'] = self.test_bull_put_spread_strategy(
            symbol, start_date, end_date
        )
        
        results['iron_condor'] = self.test_iron_condor_strategy(
            symbol, start_date, end_date
        )
        
        results['cash_secured_put'] = self.test_cash_secured_put_strategy(
            symbol, start_date, end_date
        )
        
        return results
    
    def _get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical tick data from database"""
        try:
            query = self.db.query(IndexTickData).filter(
                IndexTickData.symbol == symbol,
                IndexTickData.timestamp >= start_date,
                IndexTickData.timestamp <= end_date
            ).order_by(IndexTickData.timestamp.asc())
            
            data = query.all()
            
            if not data:
                logger.warning(f"No historical data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': d.timestamp,
                'price': d.price,
                'volume': d.volume or 0,
                'bid': d.bid,
                'ask': d.ask
            } for d in data])
            
            logger.info(f"Retrieved {len(df)} historical data points")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

