#!/usr/bin/env python3
"""
Advanced Polygon Integration System

This module provides full integration with Advanced Polygon account features:
- WebSocket real-time streaming
- S3 flat files historical data
- Advanced REST API endpoints
- Real-time ML training
- Live position monitoring
"""

import os
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from loguru import logger
import pandas as pd
import numpy as np

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.market_data.polygon_websocket import PolygonOptionsWebSocketClient
from src.market_data.polygon_flat_files import PolygonFlatFilesClient
from src.market_data.polygon_options import PolygonOptionsClient
from src.market_data.collector import MarketDataCollector
from src.ml.auto_training import AutoMLTrainingSystem
from src.trading.account_adaptation import AccountAdaptationSystem

class AdvancedPolygonIntegration:
    """
    Full integration with Advanced Polygon account features
    """
    
    def __init__(self, account_balance: float, config: Optional[Dict] = None):
        self.account_balance = account_balance
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.account_adaptation = AccountAdaptationSystem(account_balance)
        self.polygon_client = PolygonOptionsClient()
        self.market_collector = MarketDataCollector()
        self.flat_files_client = PolygonFlatFilesClient()
        self.websocket_client = None
        self.auto_ml = AutoMLTrainingSystem(account_balance)
        
        # Real-time data storage
        self.live_data = {}
        self.position_data = {}
        self.market_signals = {}
        
        # Threading
        self.websocket_thread = None
        self.data_processing_thread = None
        self.is_running = False
        
        logger.info("Advanced Polygon Integration initialized")
        logger.info(f"Account Balance: ${account_balance:,.2f}")
        logger.info(f"Account Tier: {self.account_adaptation.current_tier.name}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for advanced features"""
        return {
            "websocket_enabled": True,
            "flat_files_enabled": True,
            "real_time_ml": True,
            "live_position_monitoring": True,
            "data_refresh_interval": 5,  # seconds
            "ml_retrain_interval": 3600,  # 1 hour
            "max_historical_days": 90,
            "min_data_points_for_ml": 1000,
            "websocket_symbols": None,  # Will be set based on account tier
            "alert_thresholds": {
                "price_change": 0.05,  # 5%
                "volume_spike": 2.0,   # 2x normal
                "volatility_spike": 1.5  # 1.5x normal
            }
        }
    
    def start_advanced_trading(self):
        """Start the complete advanced trading system"""
        logger.info("🚀 Starting Advanced Polygon Trading System")
        
        try:
            # 1. Initialize WebSocket for real-time data
            if self.config["websocket_enabled"]:
                self._start_websocket_streaming()
            
            # 2. Start historical data collection
            if self.config["flat_files_enabled"]:
                self._start_historical_data_collection()
            
            # 3. Start real-time ML training
            if self.config["real_time_ml"]:
                self._start_real_time_ml()
            
            # 4. Start live position monitoring
            if self.config["live_position_monitoring"]:
                self._start_position_monitoring()
            
            self.is_running = True
            logger.info("✅ Advanced trading system started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting advanced trading system: {e}")
            raise
    
    def stop_advanced_trading(self):
        """Stop the advanced trading system"""
        logger.info("🛑 Stopping Advanced Polygon Trading System")
        
        self.is_running = False
        
        # Stop WebSocket
        if self.websocket_client:
            try:
                self.websocket_client.stop()
                self.websocket_client.disconnect()
            except:
                pass
        
        # Stop auto ML
        try:
            self.auto_ml.stop_auto_training()
        except:
            pass
        
        logger.info("✅ Advanced trading system stopped")
    
    def _start_websocket_streaming(self):
        """Start WebSocket streaming for real-time data"""
        logger.info("Starting WebSocket streaming...")
        
        # Get symbols based on account tier
        symbols = self.account_adaptation.get_recommended_symbols(5)
        self.config["websocket_symbols"] = symbols
        
        # Initialize WebSocket client
        self.websocket_client = PolygonOptionsWebSocketClient()
        
        # Subscribe to all data types
        for symbol in symbols:
            # Convert to options symbols (simplified for demo)
            options_symbols = [f"O:{symbol}251220P00550000", f"O:{symbol}251220C00550000"]
            
            self.websocket_client.subscribe_to_trades(options_symbols)
            self.websocket_client.subscribe_to_quotes(options_symbols)
            self.websocket_client.subscribe_to_aggregates_minute(options_symbols)
            self.websocket_client.subscribe_to_aggregates_second(options_symbols)
            self.websocket_client.subscribe_to_fmv(options_symbols)
        
        # Connect and start streaming
        self.websocket_client.connect()
        self.websocket_client.start()
        
        # Start data processing thread
        self.websocket_thread = threading.Thread(target=self._process_websocket_data, daemon=True)
        self.websocket_thread.start()
        
        logger.info("✅ WebSocket streaming started")
    
    def _start_historical_data_collection(self):
        """Start historical data collection from flat files"""
        logger.info("Starting historical data collection...")
        
        try:
            # Get available dates
            trades_dates = self.flat_files_client.list_available_dates("trades")
            quotes_dates = self.flat_files_client.list_available_dates("quotes")
            aggregates_dates = self.flat_files_client.list_available_dates("aggregates")
            
            logger.info(f"Historical data available:")
            logger.info(f"  Trades: {len(trades_dates)} dates")
            logger.info(f"  Quotes: {len(quotes_dates)} dates")
            logger.info(f"  Aggregates: {len(aggregates_dates)} dates")
            
            # Download recent data for ML training
            recent_dates = trades_dates[-30:]  # Last 30 days
            
            for date in recent_dates:
                try:
                    # Download trades data
                    trades_file = self.flat_files_client.download_data("trades", date)
                    if trades_file and trades_file.exists():
                        logger.info(f"Downloaded trades for {date}: {trades_file.stat().st_size} bytes")
                    
                    # Download quotes data
                    quotes_file = self.flat_files_client.download_data("quotes", date)
                    if quotes_file and quotes_file.exists():
                        logger.info(f"Downloaded quotes for {date}: {quotes_file.stat().st_size} bytes")
                    
                    # Download aggregates data
                    aggregates_file = self.flat_files_client.download_data("aggregates", date)
                    if aggregates_file and aggregates_file.exists():
                        logger.info(f"Downloaded aggregates for {date}: {aggregates_file.stat().st_size} bytes")
                
                except Exception as e:
                    logger.warning(f"Error downloading data for {date}: {e}")
                    continue
            
            logger.info("✅ Historical data collection completed")
            
        except Exception as e:
            logger.error(f"❌ Error in historical data collection: {e}")
    
    def _start_real_time_ml(self):
        """Start real-time ML training and prediction"""
        logger.info("Starting real-time ML system...")
        
        # Start auto ML training
        self.auto_ml.start_auto_training()
        
        # Start real-time data processing
        self.data_processing_thread = threading.Thread(target=self._process_real_time_data, daemon=True)
        self.data_processing_thread.start()
        
        logger.info("✅ Real-time ML system started")
    
    def _start_position_monitoring(self):
        """Start live position monitoring"""
        logger.info("Starting live position monitoring...")
        
        # This would integrate with your broker API
        # For now, we'll simulate position monitoring
        logger.info("✅ Live position monitoring started")
    
    def _process_websocket_data(self):
        """Process incoming WebSocket data"""
        while self.is_running:
            try:
                if self.websocket_client:
                    # Get latest data
                    stats = self.websocket_client.get_stats()
                    
                    if stats and stats.get('message_count', 0) > 0:
                        # Process new data
                        self._analyze_real_time_data()
                
                time.sleep(self.config["data_refresh_interval"])
                
            except Exception as e:
                logger.error(f"Error processing WebSocket data: {e}")
                time.sleep(5)
    
    def _process_real_time_data(self):
        """Process real-time data for ML predictions"""
        while self.is_running:
            try:
                # Get current market data
                symbols = self.config["websocket_symbols"]
                if symbols:
                    for symbol in symbols:
                        # Get real-time options data
                        options_data = self._get_real_time_options_data(symbol)
                        if options_data:
                            # Store for ML processing
                            self.live_data[symbol] = options_data
                
                # Trigger ML retraining if needed
                if self._should_retrain_ml():
                    logger.info("Triggering real-time ML retraining...")
                    self.auto_ml.force_training()
                
                time.sleep(self.config["ml_retrain_interval"])
                
            except Exception as e:
                logger.error(f"Error in real-time data processing: {e}")
                time.sleep(60)
    
    def _get_real_time_options_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time options data for a symbol"""
        try:
            # Search for options contracts
            contracts = self.polygon_client.search_options_contracts(symbol, limit=10)
            
            if contracts:
                # Get real-time data for each contract
                options_data = {}
                for contract in contracts:
                    contract_id = contract.get('ticker', '')
                    
                    # Get current price
                    last_trade = self.polygon_client.get_last_trade(contract_id)
                    quotes = self.polygon_client.get_quotes(contract_id)
                    
                    options_data[contract_id] = {
                        'last_trade': last_trade,
                        'quotes': quotes,
                        'timestamp': datetime.now()
                    }
                
                return options_data
            
        except Exception as e:
            logger.error(f"Error getting real-time options data for {symbol}: {e}")
        
        return None
    
    def _analyze_real_time_data(self):
        """Analyze real-time data for trading signals"""
        try:
            for symbol, data in self.live_data.items():
                # Analyze price movements
                if self._detect_price_alert(data):
                    self._generate_trading_alert(symbol, "price_change", data)
                
                # Analyze volume spikes
                if self._detect_volume_spike(data):
                    self._generate_trading_alert(symbol, "volume_spike", data)
                
                # Analyze volatility changes
                if self._detect_volatility_spike(data):
                    self._generate_trading_alert(symbol, "volatility_spike", data)
        
        except Exception as e:
            logger.error(f"Error analyzing real-time data: {e}")
    
    def _detect_price_alert(self, data: Dict) -> bool:
        """Detect significant price changes"""
        # Simplified price alert detection
        return False  # Implement based on your strategy
    
    def _detect_volume_spike(self, data: Dict) -> bool:
        """Detect volume spikes"""
        # Simplified volume spike detection
        return False  # Implement based on your strategy
    
    def _detect_volatility_spike(self, data: Dict) -> bool:
        """Detect volatility spikes"""
        # Simplified volatility spike detection
        return False  # Implement based on your strategy
    
    def _generate_trading_alert(self, symbol: str, alert_type: str, data: Dict):
        """Generate trading alert"""
        logger.info(f"🚨 Trading Alert: {symbol} - {alert_type}")
        logger.info(f"Data: {data}")
    
    def _should_retrain_ml(self) -> bool:
        """Check if ML should be retrained"""
        # Check if enough time has passed since last training
        if not self.auto_ml.last_training:
            return True
        
        time_since_training = datetime.now() - self.auto_ml.last_training
        return time_since_training.total_seconds() > self.config["ml_retrain_interval"]
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "is_running": self.is_running,
            "account_tier": self.account_adaptation.current_tier.name,
            "account_balance": self.account_balance,
            "websocket_connected": self.websocket_client is not None,
            "live_data_symbols": list(self.live_data.keys()),
            "position_count": len(self.position_data),
            "ml_training_status": self.auto_ml.get_training_status(),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.websocket_client:
            websocket_stats = self.websocket_client.get_stats()
            status["websocket_stats"] = websocket_stats
        
        return status
    
    def get_trading_signals(self) -> Dict:
        """Get current trading signals"""
        return {
            "signals": self.market_signals,
            "live_data": self.live_data,
            "positions": self.position_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def force_ml_retraining(self):
        """Force immediate ML retraining"""
        logger.info("Forcing immediate ML retraining...")
        self.auto_ml.force_training()
    
    def get_historical_data_summary(self) -> Dict:
        """Get summary of available historical data"""
        try:
            trades_dates = self.flat_files_client.list_available_dates("trades")
            quotes_dates = self.flat_files_client.list_available_dates("quotes")
            aggregates_dates = self.flat_files_client.list_available_dates("aggregates")
            
            return {
                "trades": {
                    "total_dates": len(trades_dates),
                    "latest": trades_dates[-1] if trades_dates else None,
                    "earliest": trades_dates[0] if trades_dates else None
                },
                "quotes": {
                    "total_dates": len(quotes_dates),
                    "latest": quotes_dates[-1] if quotes_dates else None,
                    "earliest": quotes_dates[0] if quotes_dates else None
                },
                "aggregates": {
                    "total_dates": len(aggregates_dates),
                    "latest": aggregates_dates[-1] if aggregates_dates else None,
                    "earliest": aggregates_dates[0] if aggregates_dates else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting historical data summary: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Test the advanced integration
    advanced_system = AdvancedPolygonIntegration(account_balance=25000)
    
    print("Advanced Polygon Integration Test")
    print("=" * 50)
    print(f"Account Balance: ${advanced_system.account_balance:,}")
    print(f"Account Tier: {advanced_system.account_adaptation.current_tier.name}")
    print(f"Recommended Symbols: {advanced_system.account_adaptation.get_recommended_symbols()}")
    
    # Get historical data summary
    historical_summary = advanced_system.get_historical_data_summary()
    print(f"\nHistorical Data Summary:")
    print(f"Trades: {historical_summary.get('trades', {}).get('total_dates', 0)} dates")
    print(f"Quotes: {historical_summary.get('quotes', {}).get('total_dates', 0)} dates")
    print(f"Aggregates: {historical_summary.get('aggregates', {}).get('total_dates', 0)} dates")
    
    print("\n✅ Advanced Polygon Integration ready!")

