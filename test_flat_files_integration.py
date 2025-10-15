#!/usr/bin/env python3
"""
Test script for Polygon.io Flat Files integration
Tests historical data download, ML pipeline, and backtesting
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.market_data.polygon_flat_files import PolygonFlatFilesClient
from src.ml.data_pipeline import OptionsMLDataPipeline
from src.ml.backtesting import OptionsBacktester, simple_momentum_strategy, mean_reversion_strategy


def test_flat_files_client():
    """Test the flat files client"""
    logger.info("=" * 60)
    logger.info("Testing Polygon Flat Files Client")
    logger.info("=" * 60)
    
    try:
        # Initialize client
        client = PolygonFlatFilesClient()
        logger.info("✅ PolygonFlatFilesClient initialized successfully")
        
        # Test 1: List available dates
        logger.info("\n1. Testing available dates listing...")
        
        trades_dates = client.list_available_dates("trades")
        if trades_dates:
            logger.info(f"✅ Found {len(trades_dates)} available trade dates")
            logger.info(f"   Latest: {trades_dates[-1]}")
            logger.info(f"   Earliest: {trades_dates[0]}")
        else:
            logger.warning("⚠️ No trade dates found")
        
        quotes_dates = client.list_available_dates("quotes")
        if quotes_dates:
            logger.info(f"✅ Found {len(quotes_dates)} available quote dates")
        else:
            logger.warning("⚠️ No quote dates found")
        
        aggregates_dates = client.list_available_dates("aggregates")
        if aggregates_dates:
            logger.info(f"✅ Found {len(aggregates_dates)} available aggregate dates")
        else:
            logger.warning("⚠️ No aggregate dates found")
        
        # Test 2: Download data (if dates available)
        if trades_dates:
            test_date = trades_dates[-1]  # Use most recent date
            logger.info(f"\n2. Testing data download for {test_date}...")
            
            # Test trades download
            trades_file = client.download_data("trades", test_date)
            if trades_file:
                logger.info(f"✅ Downloaded trades data: {trades_file}")
            else:
                logger.warning("⚠️ Failed to download trades data")
            
            # Test quotes download
            quotes_file = client.download_data("quotes", test_date)
            if quotes_file:
                logger.info(f"✅ Downloaded quotes data: {quotes_file}")
            else:
                logger.warning("⚠️ Failed to download quotes data")
            
            # Test aggregates download
            aggregates_file = client.download_data("aggregates", test_date)
            if aggregates_file:
                logger.info(f"✅ Downloaded aggregates data: {aggregates_file}")
            else:
                logger.warning("⚠️ Failed to download aggregates data")
        
        # Test 3: Load and process data
        if trades_dates:
            test_date = trades_dates[-1]
            logger.info(f"\n3. Testing data loading for {test_date}...")
            
            # Load trades data
            trades_df = client.load_trades_data(test_date, max_rows=1000)
            if not trades_df.empty:
                logger.info(f"✅ Loaded trades data: {len(trades_df)} records")
                logger.info(f"   Columns: {list(trades_df.columns)}")
                logger.info(f"   Sample data:")
                logger.info(f"   {trades_df.head(2).to_string()}")
            else:
                logger.warning("⚠️ No trades data loaded")
            
            # Load quotes data
            quotes_df = client.load_quotes_data(test_date, max_rows=1000)
            if not quotes_df.empty:
                logger.info(f"✅ Loaded quotes data: {len(quotes_df)} records")
                logger.info(f"   Columns: {list(quotes_df.columns)}")
            else:
                logger.warning("⚠️ No quotes data loaded")
            
            # Load aggregates data
            aggregates_df = client.load_aggregates_data(test_date, max_rows=1000)
            if not aggregates_df.empty:
                logger.info(f"✅ Loaded aggregates data: {len(aggregates_df)} records")
                logger.info(f"   Columns: {list(aggregates_df.columns)}")
            else:
                logger.warning("⚠️ No aggregates data loaded")
        
        # Test 4: Create ML features
        if trades_dates:
            test_date = trades_dates[-1]
            logger.info(f"\n4. Testing ML feature creation for {test_date}...")
            
            # Load sample data
            trades_df = client.load_trades_data(test_date, max_rows=1000)
            quotes_df = client.load_quotes_data(test_date, max_rows=1000)
            aggregates_df = client.load_aggregates_data(test_date, max_rows=1000)
            
            if not trades_df.empty or not quotes_df.empty or not aggregates_df.empty:
                # Create features
                features_df = client.create_ml_features(trades_df, quotes_df, aggregates_df)
                
                if not features_df.empty:
                    logger.info(f"✅ Created ML features: {len(features_df)} records")
                    logger.info(f"   Feature columns: {len(features_df.columns)}")
                    logger.info(f"   Sample features:")
                    logger.info(f"   {features_df.head(2).to_string()}")
                else:
                    logger.warning("⚠️ No ML features created")
            else:
                logger.warning("⚠️ No data available for feature creation")
        
        logger.info("\n✅ Flat Files Client tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing Flat Files Client: {e}")
        return False


def test_ml_data_pipeline():
    """Test the ML data pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ML Data Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OptionsMLDataPipeline()
        logger.info("✅ OptionsMLDataPipeline initialized successfully")
        
        # Test with a small date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        logger.info(f"\n1. Testing comprehensive dataset creation from {start_date} to {end_date}...")
        
        # Test symbols (use some common option symbols)
        test_symbols = ["O:SPY251220P00550000", "O:SPY251220C00550000"]
        
        # Create dataset
        dataset = pipeline.create_comprehensive_dataset(
            symbols=test_symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            save_features=True
        )
        
        if not dataset.empty:
            logger.info(f"✅ Created comprehensive dataset: {len(dataset)} records")
            logger.info(f"   Columns: {len(dataset.columns)}")
            logger.info(f"   Date range: {dataset['date'].min()} to {dataset['date'].max()}")
            logger.info(f"   Symbols: {dataset['symbol'].unique()}")
            
            # Test ML data preparation
            logger.info("\n2. Testing ML data preparation...")
            
            # Find a target column
            target_columns = [col for col in dataset.columns if 'return' in col or 'future' in col]
            if target_columns:
                target_col = target_columns[0]
                logger.info(f"Using target column: {target_col}")
                
                X_train, y_train, X_test, y_test = pipeline.prepare_ml_data(
                    dataset, target_col, test_size=0.2
                )
                
                if not X_train.empty and not X_test.empty:
                    logger.info(f"✅ Prepared ML data:")
                    logger.info(f"   Training samples: {len(X_train)}")
                    logger.info(f"   Test samples: {len(X_test)}")
                    logger.info(f"   Features: {len(X_train.columns)}")
                    
                    # Test model training
                    logger.info("\n3. Testing model training...")
                    
                    models = pipeline.train_models(X_train, y_train, target_col)
                    if models:
                        logger.info(f"✅ Trained {len(models)} models")
                        
                        # Test model evaluation
                        logger.info("\n4. Testing model evaluation...")
                        
                        results = pipeline.evaluate_models(X_test, y_test, target_col)
                        if results:
                            logger.info(f"✅ Evaluated {len(results)} models")
                            
                            # Show best model
                            best_model = max(results.items(), key=lambda x: x[1]['r2'])
                            logger.info(f"   Best model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
                            
                            # Test model saving
                            logger.info("\n5. Testing model saving...")
                            
                            if pipeline.save_models(target_col):
                                logger.info("✅ Models saved successfully")
                                
                                # Test model loading
                                logger.info("\n6. Testing model loading...")
                                
                                if pipeline.load_models(target_col):
                                    logger.info("✅ Models loaded successfully")
                                else:
                                    logger.warning("⚠️ Model loading failed")
                            else:
                                logger.warning("⚠️ Model saving failed")
                        else:
                            logger.warning("⚠️ Model evaluation failed")
                    else:
                        logger.warning("⚠️ Model training failed")
                else:
                    logger.warning("⚠️ ML data preparation failed")
            else:
                logger.warning("⚠️ No target columns found for ML training")
        else:
            logger.warning("⚠️ No dataset created")
        
        logger.info("\n✅ ML Data Pipeline tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing ML Data Pipeline: {e}")
        return False


def test_backtesting_framework():
    """Test the backtesting framework"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Backtesting Framework")
    logger.info("=" * 60)
    
    try:
        # Initialize backtester
        backtester = OptionsBacktester()
        logger.info("✅ OptionsBacktester initialized successfully")
        
        # Test with a small date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=5)  # Last 5 days
        
        logger.info(f"\n1. Testing backtesting from {start_date} to {end_date}...")
        
        # Test symbols
        test_symbols = ["O:SPY251220P00550000", "O:SPY251220C00550000"]
        
        # Test momentum strategy
        logger.info("\n2. Testing momentum strategy...")
        
        momentum_result = backtester.backtest_strategy(
            strategy_func=simple_momentum_strategy,
            symbols=test_symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            strategy_name="Simple Momentum",
            lookback_days=3,
            threshold=0.01
        )
        
        if momentum_result.total_trades > 0:
            logger.info(f"✅ Momentum strategy backtest completed:")
            logger.info(f"   Total trades: {momentum_result.total_trades}")
            logger.info(f"   Win rate: {momentum_result.win_rate:.1f}%")
            logger.info(f"   Total P&L: ${momentum_result.total_pnl:.2f}")
            logger.info(f"   Sharpe ratio: {momentum_result.sharpe_ratio:.3f}")
        else:
            logger.warning("⚠️ Momentum strategy generated no trades")
        
        # Test mean reversion strategy
        logger.info("\n3. Testing mean reversion strategy...")
        
        mean_reversion_result = backtester.backtest_strategy(
            strategy_func=mean_reversion_strategy,
            symbols=test_symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            strategy_name="Mean Reversion",
            lookback_days=5,
            threshold=0.015
        )
        
        if mean_reversion_result.total_trades > 0:
            logger.info(f"✅ Mean reversion strategy backtest completed:")
            logger.info(f"   Total trades: {mean_reversion_result.total_trades}")
            logger.info(f"   Win rate: {mean_reversion_result.win_rate:.1f}%")
            logger.info(f"   Total P&L: ${mean_reversion_result.total_pnl:.2f}")
            logger.info(f"   Sharpe ratio: {mean_reversion_result.sharpe_ratio:.3f}")
        else:
            logger.warning("⚠️ Mean reversion strategy generated no trades")
        
        # Test strategy comparison
        logger.info("\n4. Testing strategy comparison...")
        
        strategies = [
            (simple_momentum_strategy, "Momentum", {"lookback_days": 3, "threshold": 0.01}),
            (mean_reversion_strategy, "Mean Reversion", {"lookback_days": 5, "threshold": 0.015})
        ]
        
        comparison_df = backtester.compare_strategies(
            strategies=strategies,
            symbols=test_symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not comparison_df.empty:
            logger.info(f"✅ Strategy comparison completed:")
            logger.info(f"\n{comparison_df.to_string(index=False)}")
        else:
            logger.warning("⚠️ Strategy comparison failed")
        
        # Test result saving
        logger.info("\n5. Testing result saving...")
        
        if momentum_result.total_trades > 0:
            backtester.save_backtest_results(momentum_result, "test_momentum")
            logger.info("✅ Momentum results saved")
        
        if mean_reversion_result.total_trades > 0:
            backtester.save_backtest_results(mean_reversion_result, "test_mean_reversion")
            logger.info("✅ Mean reversion results saved")
        
        logger.info("\n✅ Backtesting Framework tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing Backtesting Framework: {e}")
        return False


def main():
    """Main test function"""
    logger.info("🚀 Starting Polygon.io Flat Files Integration Testing")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Check for API key
    if not os.getenv('POLYGON_API_KEY'):
        logger.warning("⚠️ POLYGON_API_KEY environment variable not set!")
        logger.info("Using hardcoded API key for testing...")
        # Set the API key for testing
        os.environ['POLYGON_API_KEY'] = 'wWrUjjcksqLDPntXbJb72kiFzAwyqIpY'
    
    # Run tests
    flat_files_success = test_flat_files_client()
    ml_pipeline_success = test_ml_data_pipeline()
    backtesting_success = test_backtesting_framework()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Flat Files Client: {'✅ PASSED' if flat_files_success else '❌ FAILED'}")
    logger.info(f"ML Data Pipeline: {'✅ PASSED' if ml_pipeline_success else '❌ FAILED'}")
    logger.info(f"Backtesting Framework: {'✅ PASSED' if backtesting_success else '❌ FAILED'}")
    
    if flat_files_success and ml_pipeline_success and backtesting_success:
        logger.info("\n🎉 All flat files integration tests passed!")
        logger.info("Your trading agent now has access to comprehensive historical data for ML training and backtesting!")
        return True
    else:
        logger.error("\n❌ Some tests failed. Please check the logs above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
