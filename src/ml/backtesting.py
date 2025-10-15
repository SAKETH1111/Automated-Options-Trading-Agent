"""
Advanced Backtesting Framework for Options Trading
Uses historical flat files data for comprehensive strategy testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.market_data.polygon_flat_files import PolygonFlatFilesClient


@dataclass
class BacktestResult:
    """Backtest result data structure"""
    strategy_name: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    total_return: float
    annualized_return: float
    volatility: float
    trades: List[Dict]
    daily_returns: pd.Series
    equity_curve: pd.Series


class OptionsBacktester:
    """
    Advanced backtesting framework for options trading strategies
    
    Features:
    - Historical data backtesting
    - Multiple strategy support
    - Comprehensive performance metrics
    - Risk analysis
    - Portfolio-level backtesting
    """
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data/backtesting"):
        """
        Initialize backtester
        
        Args:
            api_key: Polygon.io API key
            data_dir: Directory for backtesting data
        """
        self.api_key = api_key
        self.data_dir = data_dir
        self.flat_files_client = PolygonFlatFilesClient(api_key, data_dir)
        
        # Backtesting parameters
        self.initial_capital = 100000  # $100k default
        self.commission_per_trade = 1.0  # $1 per trade
        self.slippage = 0.001  # 0.1% slippage
        
        logger.info("OptionsBacktester initialized")
    
    def backtest_strategy(
        self,
        strategy_func,
        symbols: List[str],
        start_date: str,
        end_date: str,
        strategy_name: str = "Custom Strategy",
        **strategy_params
    ) -> BacktestResult:
        """
        Backtest a trading strategy
        
        Args:
            strategy_func: Strategy function
            symbols: List of option symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            strategy_name: Name of the strategy
            **strategy_params: Strategy parameters
            
        Returns:
            BacktestResult object
        """
        try:
            logger.info(f"Starting backtest for {strategy_name} from {start_date} to {end_date}")
            
            # Load historical data
            logger.info("Loading historical data...")
            trades_df = self.flat_files_client.get_historical_data_range(
                "trades", start_date, end_date, symbols
            )
            quotes_df = self.flat_files_client.get_historical_data_range(
                "quotes", start_date, end_date, symbols
            )
            aggregates_df = self.flat_files_client.get_historical_data_range(
                "aggregates", start_date, end_date, symbols
            )
            
            if trades_df.empty and quotes_df.empty and aggregates_df.empty:
                logger.error("No historical data found")
                return self._create_empty_result(strategy_name, start_date, end_date)
            
            # Create combined dataset
            combined_data = self._create_combined_dataset(trades_df, quotes_df, aggregates_df)
            
            # Run strategy
            logger.info("Running strategy...")
            trades = self._run_strategy(strategy_func, combined_data, **strategy_params)
            
            # Calculate performance metrics
            logger.info("Calculating performance metrics...")
            result = self._calculate_performance_metrics(
                strategy_name, start_date, end_date, trades
            )
            
            logger.info(f"Backtest completed: {result.total_trades} trades, {result.win_rate:.1f}% win rate, ${result.total_pnl:.2f} P&L")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return self._create_empty_result(strategy_name, start_date, end_date)
    
    def _create_combined_dataset(
        self,
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame,
        aggregates_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create combined dataset for backtesting"""
        try:
            # Start with aggregates as base (daily data)
            if not aggregates_df.empty:
                base_df = aggregates_df.copy()
                base_df['data_type'] = 'aggregate'
            elif not trades_df.empty:
                # If no aggregates, use trades aggregated by day
                base_df = trades_df.groupby(['sym', 'date']).agg({
                    'price': ['mean', 'min', 'max', 'std'],
                    'size': ['sum', 'mean', 'count'],
                    'timestamp': ['min', 'max']
                }).reset_index()
                base_df.columns = ['sym', 'date', 'avg_price', 'min_price', 'max_price', 'price_std',
                                 'total_volume', 'avg_size', 'trade_count', 'first_trade', 'last_trade']
                base_df['data_type'] = 'trade_aggregate'
            else:
                # If no trades, use quotes aggregated by day
                base_df = quotes_df.groupby(['sym', 'date']).agg({
                    'bid': 'mean',
                    'ask': 'mean',
                    'spread': 'mean',
                    'mid_price': 'mean',
                    'timestamp': ['min', 'max']
                }).reset_index()
                base_df.columns = ['sym', 'date', 'avg_bid', 'avg_ask', 'avg_spread', 'avg_mid_price',
                                 'first_quote', 'last_quote']
                base_df['data_type'] = 'quote_aggregate'
            
            # Add additional data
            if not trades_df.empty:
                trade_features = trades_df.groupby(['sym', 'date']).agg({
                    'price': ['mean', 'std'],
                    'size': ['sum', 'mean'],
                    'timestamp': 'count'
                }).reset_index()
                trade_features.columns = ['sym', 'date', 'trade_avg_price', 'trade_price_std',
                                        'trade_total_volume', 'trade_avg_size', 'trade_count']
                
                base_df = base_df.merge(trade_features, on=['sym', 'date'], how='left')
            
            if not quotes_df.empty:
                quote_features = quotes_df.groupby(['sym', 'date']).agg({
                    'bid': 'mean',
                    'ask': 'mean',
                    'spread': ['mean', 'std'],
                    'mid_price': 'mean',
                    'timestamp': 'count'
                }).reset_index()
                quote_features.columns = ['sym', 'date', 'quote_avg_bid', 'quote_avg_ask',
                                        'quote_avg_spread', 'quote_spread_std', 'quote_avg_mid', 'quote_count']
                
                base_df = base_df.merge(quote_features, on=['sym', 'date'], how='left')
            
            # Sort by symbol and date
            base_df = base_df.sort_values(['sym', 'date'])
            
            # Add derived features
            base_df['date'] = pd.to_datetime(base_df['date'])
            base_df['year'] = base_df['date'].dt.year
            base_df['month'] = base_df['date'].dt.month
            base_df['dayofweek'] = base_df['date'].dt.dayofweek
            
            # Add price features
            if 'avg_price' in base_df.columns:
                base_df['price'] = base_df['avg_price']
            elif 'avg_mid_price' in base_df.columns:
                base_df['price'] = base_df['avg_mid_price']
            elif 'close' in base_df.columns:
                base_df['price'] = base_df['close']
            else:
                base_df['price'] = 0
            
            # Add volume features
            if 'total_volume' in base_df.columns:
                base_df['volume'] = base_df['total_volume']
            elif 'trade_total_volume' in base_df.columns:
                base_df['volume'] = base_df['trade_total_volume']
            else:
                base_df['volume'] = 0
            
            # Add spread features
            if 'avg_spread' in base_df.columns:
                base_df['spread'] = base_df['avg_spread']
            elif 'quote_avg_spread' in base_df.columns:
                base_df['spread'] = base_df['quote_avg_spread']
            else:
                base_df['spread'] = 0
            
            logger.info(f"Created combined dataset with {len(base_df)} records")
            return base_df
            
        except Exception as e:
            logger.error(f"Error creating combined dataset: {e}")
            return pd.DataFrame()
    
    def _run_strategy(
        self,
        strategy_func,
        data: pd.DataFrame,
        **strategy_params
    ) -> List[Dict]:
        """Run trading strategy on historical data"""
        try:
            trades = []
            current_positions = {}
            capital = self.initial_capital
            
            # Group by symbol for individual symbol processing
            for symbol, symbol_data in data.groupby('sym'):
                logger.info(f"Processing strategy for {symbol}...")
                
                symbol_trades = self._run_symbol_strategy(
                    strategy_func, symbol, symbol_data, current_positions, capital, **strategy_params
                )
                
                trades.extend(symbol_trades)
            
            logger.info(f"Strategy generated {len(trades)} trades")
            return trades
            
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return []
    
    def _run_symbol_strategy(
        self,
        strategy_func,
        symbol: str,
        data: pd.DataFrame,
        current_positions: Dict,
        capital: float,
        **strategy_params
    ) -> List[Dict]:
        """Run strategy for a single symbol"""
        try:
            trades = []
            symbol_positions = current_positions.get(symbol, [])
            
            # Sort by date
            data = data.sort_values('date')
            
            for idx, row in data.iterrows():
                # Prepare market data for strategy
                market_data = {
                    'symbol': symbol,
                    'date': row['date'],
                    'price': row.get('price', 0),
                    'volume': row.get('volume', 0),
                    'spread': row.get('spread', 0),
                    'open': row.get('open', row.get('price', 0)),
                    'high': row.get('high', row.get('price', 0)),
                    'low': row.get('low', row.get('price', 0)),
                    'close': row.get('close', row.get('price', 0)),
                    'trade_count': row.get('trade_count', 0),
                    'quote_count': row.get('quote_count', 0)
                }
                
                # Add historical data
                historical_data = data[data['date'] <= row['date']].tail(20)  # Last 20 days
                market_data['historical'] = historical_data.to_dict('records')
                
                # Run strategy
                try:
                    strategy_signals = strategy_func(market_data, symbol_positions, **strategy_params)
                    
                    # Process signals
                    for signal in strategy_signals:
                        trade = self._process_signal(signal, symbol, row['date'], capital)
                        if trade:
                            trades.append(trade)
                            symbol_positions.append(trade)
                            
                            # Update capital
                            if signal['action'] == 'buy':
                                capital -= trade['cost']
                            elif signal['action'] == 'sell':
                                capital += trade['proceeds']
                
                except Exception as e:
                    logger.error(f"Error running strategy for {symbol} on {row['date']}: {e}")
                    continue
            
            # Update positions
            current_positions[symbol] = symbol_positions
            
            return trades
            
        except Exception as e:
            logger.error(f"Error running symbol strategy for {symbol}: {e}")
            return []
    
    def _process_signal(
        self,
        signal: Dict,
        symbol: str,
        date: date,
        capital: float
    ) -> Optional[Dict]:
        """Process a trading signal"""
        try:
            action = signal.get('action')
            quantity = signal.get('quantity', 1)
            price = signal.get('price', 0)
            reason = signal.get('reason', '')
            
            if action not in ['buy', 'sell'] or price <= 0:
                return None
            
            # Calculate costs
            gross_cost = quantity * price
            commission = self.commission_per_trade
            slippage_cost = gross_cost * self.slippage
            
            if action == 'buy':
                total_cost = gross_cost + commission + slippage_cost
                if total_cost > capital:
                    logger.warning(f"Insufficient capital for {symbol} buy signal")
                    return None
                
                return {
                    'symbol': symbol,
                    'date': date,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'cost': total_cost,
                    'commission': commission,
                    'slippage': slippage_cost,
                    'reason': reason,
                    'timestamp': datetime.now()
                }
            
            elif action == 'sell':
                proceeds = gross_cost - commission - slippage_cost
                
                return {
                    'symbol': symbol,
                    'date': date,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price,
                    'proceeds': proceeds,
                    'commission': commission,
                    'slippage': slippage_cost,
                    'reason': reason,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
    
    def _calculate_performance_metrics(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
        trades: List[Dict]
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        try:
            if not trades:
                return self._create_empty_result(strategy_name, start_date, end_date)
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            
            # Calculate P&L for each trade
            trades_df['pnl'] = 0.0
            trades_df['cumulative_pnl'] = 0.0
            
            # Simple P&L calculation (buy low, sell high)
            buy_trades = trades_df[trades_df['action'] == 'buy'].copy()
            sell_trades = trades_df[trades_df['action'] == 'sell'].copy()
            
            total_pnl = 0.0
            for idx, buy_trade in buy_trades.iterrows():
                # Find corresponding sell trade
                sell_trade = sell_trades[
                    (sell_trades['symbol'] == buy_trade['symbol']) &
                    (sell_trades['date'] > buy_trade['date'])
                ].iloc[0] if len(sell_trades[
                    (sell_trades['symbol'] == buy_trade['symbol']) &
                    (sell_trades['date'] > buy_trade['date'])
                ]) > 0 else None
                
                if sell_trade is not None:
                    pnl = sell_trade['proceeds'] - buy_trade['cost']
                    total_pnl += pnl
                    
                    # Update trades_df
                    trades_df.loc[trades_df['date'] == buy_trade['date'], 'pnl'] = pnl
                    trades_df.loc[trades_df['date'] == sell_trade['date'], 'pnl'] = pnl
            
            # Calculate cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # P&L metrics
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            max_win = trades_df['pnl'].max() if total_trades > 0 else 0
            max_loss = trades_df['pnl'].min() if total_trades > 0 else 0
            
            # Risk metrics
            daily_returns = trades_df.groupby('date')['pnl'].sum()
            if len(daily_returns) > 0:
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe_ratio = (daily_returns.mean() * 252) / (volatility) if volatility > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = daily_returns[daily_returns < 0]
                downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino_ratio = (daily_returns.mean() * 252) / (downside_std) if downside_std > 0 else 0
                
                # Max drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Calmar ratio
                annualized_return = daily_returns.mean() * 252
                calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # Profit factor
                gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
                gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Total return
                total_return = (total_pnl / self.initial_capital) * 100
            else:
                volatility = 0
                sharpe_ratio = 0
                sortino_ratio = 0
                max_drawdown = 0
                calmar_ratio = 0
                profit_factor = 0
                total_return = 0
            
            # Create result
            result = BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_pnl_per_trade=avg_pnl_per_trade,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_win=max_win,
                max_loss=max_loss,
                total_return=total_return,
                annualized_return=annualized_return if 'annualized_return' in locals() else 0,
                volatility=volatility,
                trades=trades,
                daily_returns=daily_returns,
                equity_curve=trades_df['cumulative_pnl'] if not trades_df.empty else pd.Series()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._create_empty_result(strategy_name, start_date, end_date)
    
    def _create_empty_result(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """Create empty result for failed backtests"""
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_pnl_per_trade=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_win=0.0,
            max_loss=0.0,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            trades=[],
            daily_returns=pd.Series(),
            equity_curve=pd.Series()
        )
    
    def compare_strategies(
        self,
        strategies: List[Tuple],
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            strategies: List of (strategy_func, strategy_name, strategy_params) tuples
            symbols: List of option symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with comparison results
        """
        try:
            results = []
            
            for strategy_func, strategy_name, strategy_params in strategies:
                logger.info(f"Backtesting {strategy_name}...")
                
                result = self.backtest_strategy(
                    strategy_func, symbols, start_date, end_date, strategy_name, **strategy_params
                )
                
                results.append({
                    'Strategy': strategy_name,
                    'Total Trades': result.total_trades,
                    'Win Rate (%)': result.win_rate,
                    'Total P&L ($)': result.total_pnl,
                    'Avg P&L per Trade ($)': result.avg_pnl_per_trade,
                    'Max Drawdown (%)': result.max_drawdown * 100,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Sortino Ratio': result.sortino_ratio,
                    'Calmar Ratio': result.calmar_ratio,
                    'Profit Factor': result.profit_factor,
                    'Total Return (%)': result.total_return,
                    'Volatility (%)': result.volatility * 100
                })
            
            comparison_df = pd.DataFrame(results)
            comparison_df = comparison_df.sort_values('Total P&L ($)', ascending=False)
            
            logger.info("Strategy comparison completed")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()
    
    def save_backtest_results(self, result: BacktestResult, filename: str):
        """Save backtest results to file"""
        try:
            # Save trades
            trades_df = pd.DataFrame(result.trades)
            trades_file = f"{filename}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            
            # Save summary
            summary = {
                'strategy_name': result.strategy_name,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'avg_pnl_per_trade': result.avg_pnl_per_trade,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'profit_factor': result.profit_factor,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
                'max_win': result.max_win,
                'max_loss': result.max_loss,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility
            }
            
            summary_df = pd.DataFrame([summary])
            summary_file = f"{filename}_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"Saved backtest results to {filename}_*.csv")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")


# Example strategy functions
def simple_momentum_strategy(market_data: Dict, positions: List[Dict], lookback_days: int = 5, threshold: float = 0.02) -> List[Dict]:
    """Simple momentum strategy example"""
    signals = []
    
    try:
        symbol = market_data['symbol']
        price = market_data['price']
        historical = market_data['historical']
        
        if len(historical) < lookback_days + 1:
            return signals
        
        # Calculate momentum
        current_price = price
        past_price = historical[-lookback_days-1]['price']
        momentum = (current_price - past_price) / past_price
        
        # Generate signals
        if momentum > threshold and len(positions) == 0:
            # Buy signal
            signals.append({
                'action': 'buy',
                'quantity': 1,
                'price': price,
                'reason': f'Momentum buy: {momentum:.3f}'
            })
        elif momentum < -threshold and len(positions) > 0:
            # Sell signal
            signals.append({
                'action': 'sell',
                'quantity': 1,
                'price': price,
                'reason': f'Momentum sell: {momentum:.3f}'
            })
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in momentum strategy: {e}")
        return []


def mean_reversion_strategy(market_data: Dict, positions: List[Dict], lookback_days: int = 10, threshold: float = 0.02) -> List[Dict]:
    """Mean reversion strategy example"""
    signals = []
    
    try:
        symbol = market_data['symbol']
        price = market_data['price']
        historical = market_data['historical']
        
        if len(historical) < lookback_days:
            return signals
        
        # Calculate mean and standard deviation
        prices = [h['price'] for h in historical[-lookback_days:]]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        # Calculate z-score
        z_score = (price - mean_price) / std_price if std_price > 0 else 0
        
        # Generate signals
        if z_score < -2 and len(positions) == 0:
            # Buy signal (oversold)
            signals.append({
                'action': 'buy',
                'quantity': 1,
                'price': price,
                'reason': f'Mean reversion buy: z-score {z_score:.3f}'
            })
        elif z_score > 2 and len(positions) > 0:
            # Sell signal (overbought)
            signals.append({
                'action': 'sell',
                'quantity': 1,
                'price': price,
                'reason': f'Mean reversion sell: z-score {z_score:.3f}'
            })
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in mean reversion strategy: {e}")
        return []
