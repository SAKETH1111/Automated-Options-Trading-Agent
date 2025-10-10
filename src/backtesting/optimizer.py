"""
Parameter Optimizer Module
Optimize strategy parameters using grid search and genetic algorithms
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Tuple
from datetime import datetime
from itertools import product
from loguru import logger

from .engine import BacktestEngine, BacktestResult


class ParameterOptimizer:
    """
    Optimize strategy parameters
    Supports: Grid search, random search
    """
    
    def __init__(self):
        """Initialize parameter optimizer"""
        logger.info("Parameter Optimizer initialized")
    
    def grid_search(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        optimization_metric: str = 'sharpe_ratio',
        starting_capital: float = 10000.0
    ) -> Tuple[Dict, List[Dict]]:
        """
        Perform grid search optimization
        
        Args:
            data: Historical data
            strategy_func: Strategy function
            param_grid: Dictionary of parameters to test
            optimization_metric: Metric to optimize
            starting_capital: Starting capital
            
        Returns:
            Tuple of (best_params, all_results)
        """
        logger.info("Starting grid search optimization...")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        best_score = -np.inf
        best_params = None
        
        for i, combo in enumerate(combinations):
            # Create parameter dict
            params = dict(zip(param_names, combo))
            
            # Run backtest
            engine = BacktestEngine(starting_capital=starting_capital)
            result = engine.run_backtest(data, strategy_func, params)
            
            # Get optimization metric
            score = getattr(result, optimization_metric, 0)
            
            # Store result
            results.append({
                'params': params,
                'result': result,
                'score': score
            })
            
            # Track best
            if score > best_score:
                best_score = score
                best_params = params
            
            if (i + 1) % 10 == 0:
                logger.info(f"Tested {i + 1}/{len(combinations)} combinations")
        
        logger.info(f"Grid search complete. Best {optimization_metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, results
    
    def random_search(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_ranges: Dict[str, Tuple],
        n_iterations: int = 50,
        optimization_metric: str = 'sharpe_ratio',
        starting_capital: float = 10000.0
    ) -> Tuple[Dict, List[Dict]]:
        """
        Perform random search optimization
        
        Args:
            data: Historical data
            strategy_func: Strategy function
            param_ranges: Dictionary of (min, max) ranges for each parameter
            n_iterations: Number of random combinations to test
            optimization_metric: Metric to optimize
            starting_capital: Starting capital
            
        Returns:
            Tuple of (best_params, all_results)
        """
        logger.info(f"Starting random search with {n_iterations} iterations...")
        
        results = []
        best_score = -np.inf
        best_params = None
        
        for i in range(n_iterations):
            # Generate random parameters
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            # Run backtest
            engine = BacktestEngine(starting_capital=starting_capital)
            result = engine.run_backtest(data, strategy_func, params)
            
            # Get optimization metric
            score = getattr(result, optimization_metric, 0)
            
            # Store result
            results.append({
                'params': params,
                'result': result,
                'score': score
            })
            
            # Track best
            if score > best_score:
                best_score = score
                best_params = params
            
            if (i + 1) % 10 == 0:
                logger.info(f"Tested {i + 1}/{n_iterations} combinations")
        
        logger.info(f"Random search complete. Best {optimization_metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, results
    
    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        train_period_days: int = 60,
        test_period_days: int = 30
    ) -> List[Dict]:
        """
        Perform walk-forward analysis
        
        Args:
            data: Historical data
            strategy_func: Strategy function
            param_grid: Parameters to optimize
            train_period_days: Training period length
            test_period_days: Testing period length
            
        Returns:
            List of walk-forward results
        """
        logger.info("Starting walk-forward analysis...")
        
        results = []
        
        # Split data into windows
        total_days = (data['timestamp'].max() - data['timestamp'].min()).days
        n_windows = total_days // (train_period_days + test_period_days)
        
        for window in range(n_windows):
            # Define train and test periods
            train_start_day = window * (train_period_days + test_period_days)
            train_end_day = train_start_day + train_period_days
            test_end_day = train_end_day + test_period_days
            
            # Get train and test data
            train_data = data.iloc[train_start_day:train_end_day]
            test_data = data.iloc[train_end_day:test_end_day]
            
            if len(train_data) < 10 or len(test_data) < 5:
                continue
            
            # Optimize on training data
            best_params, _ = self.grid_search(
                train_data,
                strategy_func,
                param_grid,
                optimization_metric='sharpe_ratio'
            )
            
            # Test on out-of-sample data
            engine = BacktestEngine()
            test_result = engine.run_backtest(test_data, strategy_func, best_params)
            
            results.append({
                'window': window,
                'train_period': (train_data['timestamp'].min(), train_data['timestamp'].max()),
                'test_period': (test_data['timestamp'].min(), test_data['timestamp'].max()),
                'best_params': best_params,
                'test_result': test_result
            })
            
            logger.info(f"Window {window + 1}/{n_windows}: "
                       f"Sharpe={test_result.sharpe_ratio:.2f}, "
                       f"Win Rate={test_result.win_rate:.1%}")
        
        logger.info("Walk-forward analysis complete")
        
        return results

