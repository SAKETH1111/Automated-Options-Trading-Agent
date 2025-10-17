"""
Stress Tests for Critical Scenarios
Testing system behavior under extreme conditions and edge cases

Scenarios:
- Flash crash conditions
- API failures and network issues
- High volatility periods
- PDT compliance edge cases
- Circuit breaker triggers
- Memory and resource exhaustion
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.brokers.alpaca_connector import AlpacaConnector, OrderType, OrderSide, OrderTimeInForce
from src.compliance.pdt_tracker import PDTTracker
from src.risk_management.risk_manager import RiskManager
from src.volatility.enhanced_regime_detector import EnhancedRegimeDetector, MarketRegime, VolatilityRegime

class TestFlashCrashScenarios:
    """Test system behavior during flash crash conditions"""
    
    @pytest.fixture
    def flash_crash_market_data(self):
        """Market data simulating flash crash conditions"""
        return {
            'vix': 65.0,  # Extreme volatility
            'spy_price': 380.0,  # 10% drop
            'put_call_ratio': 3.5,  # Extreme fear
            'volume': 50000000,  # 50x normal volume
            'spy_prices': [420, 415, 400, 390, 380],  # Steep decline
            'qqq_prices': [380, 375, 360, 350, 340],
            'volume_data': [1000000, 5000000, 20000000, 35000000, 50000000]
        }
    
    @pytest.mark.asyncio
    async def test_flash_crash_regime_detection(self, flash_crash_market_data):
        """Test regime detection during flash crash"""
        detector = EnhancedRegimeDetector()
        
        analysis = await detector.analyze_current_regime(flash_crash_market_data)
        
        assert analysis.current_regime == MarketRegime.FLASH_CRASH
        assert analysis.volatility_regime in [VolatilityRegime.VERY_HIGH, VolatilityRegime.EXTREME]
        assert analysis.position_size_multiplier <= 0.3  # Should reduce position sizes significantly
        assert analysis.risk_multiplier >= 1.5  # Should increase risk limits
        assert 'reduced_size_' in analysis.recommended_strategies[0]  # Should recommend reduced positions
    
    @pytest.mark.asyncio
    async def test_flash_crash_circuit_breakers(self):
        """Test circuit breaker activation during flash crash"""
        risk_manager = RiskManager()
        
        # Simulate flash crash conditions
        mock_position_data = {
            'positions': [
                {'symbol': 'SPY', 'unrealized_pnl': -5000, 'credit_received': 2000},  # 2.5x loss
                {'symbol': 'QQQ', 'unrealized_pnl': -3000, 'credit_received': 1500}   # 2x loss
            ],
            'portfolio_value': 50000,
            'daily_pnl': -8000  # 16% daily loss
        }
        
        # Test circuit breaker triggers
        position_breakers = risk_manager.check_position_circuit_breakers(mock_position_data['positions'])
        portfolio_breakers = risk_manager.check_portfolio_circuit_breakers(mock_position_data)
        
        assert len(position_breakers) > 0  # Should trigger position-level breakers
        assert len(portfolio_breakers) > 0  # Should trigger portfolio-level breakers
    
    @pytest.mark.asyncio
    async def test_flash_crash_pdt_compliance(self):
        """Test PDT compliance during flash crash"""
        tracker = PDTTracker()
        
        # Simulate day trades during flash crash
        current_time = datetime.now()
        
        # Record multiple positions opened today
        for i in range(3):
            position_id = f"FLASH_CRASH_{i}"
            tracker.record_position_open(
                position_id=position_id,
                symbol="SPY",
                contract_type="PUT",
                strike=400.0,
                expiration="2024-01-19",
                quantity=1,
                price=5.0,
                timestamp=current_time
            )
        
        # Check PDT status
        status = tracker.get_current_pdt_status(5000)  # $5K account
        
        assert status.day_trades_used == 0  # No closes yet
        assert status.can_day_trade is True  # Can still day trade
        
        # Now close all positions (day trades)
        for i in range(3):
            position_id = f"FLASH_CRASH_{i}"
            tracker.record_position_close(
                position_id=position_id,
                price=15.0,  # Emergency close at high price
                timestamp=current_time + timedelta(hours=1),
                is_emergency=True  # Emergency day trade
            )
        
        # Check final PDT status
        final_status = tracker.get_current_pdt_status(5000)
        assert final_status.day_trades_used == 3
        assert final_status.can_day_trade is False  # PDT limit reached
        assert final_status.is_pdt_violation is True

class TestAPIFailureScenarios:
    """Test system behavior during API failures"""
    
    @pytest.mark.asyncio
    async def test_alpaca_api_failure(self):
        """Test Alpaca API failure handling"""
        connector = AlpacaConnector()
        connector.session = AsyncMock()
        
        # Simulate API failure
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='Service Unavailable')
        
        connector.session.get.return_value = mock_response
        
        # Test that errors are properly raised
        with pytest.raises(Exception, match="Failed to get account info"):
            await connector.get_account_info()
    
    @pytest.mark.asyncio
    async def test_polygon_api_failure(self):
        """Test Polygon API failure handling"""
        from src.data.polygon_advanced_production import PolygonAdvancedProducer
        
        producer = PolygonAdvancedProducer()
        producer.session = AsyncMock()
        
        # Simulate API failure
        mock_response = Mock()
        mock_response.status = 429  # Rate limit exceeded
        mock_response.text = AsyncMock(return_value='Rate limit exceeded')
        
        producer.session.get.return_value = mock_response
        
        # Test rate limit handling
        await producer._check_rate_limit()
        # Should sleep and retry (tested by checking request count)
        assert producer.request_count >= 1
    
    @pytest.mark.asyncio
    async def test_websocket_disconnection(self):
        """Test WebSocket disconnection handling"""
        connector = AlpacaConnector()
        
        # Mock WebSocket that closes immediately
        mock_websocket = AsyncMock()
        mock_websocket.close = AsyncMock()
        
        with patch('websockets.connect', return_value=mock_websocket):
            with patch.object(connector, '_test_authentication', return_value={'id': 'test'}):
                await connector.initialize()
                
                # Simulate disconnection
                mock_websocket.closed = True
                
                # Test reconnection logic
                try:
                    await connector.listen_account_updates(lambda x: None)
                except Exception as e:
                    # Should handle disconnection gracefully
                    assert "Connection closed" in str(e) or "ConnectionClosed" in str(e)

class TestHighVolatilityScenarios:
    """Test system behavior during high volatility periods"""
    
    @pytest.mark.asyncio
    async def test_extreme_volatility_regime(self):
        """Test regime detection during extreme volatility"""
        detector = EnhancedRegimeDetector()
        
        extreme_vol_data = {
            'vix': 85.0,  # Extreme volatility
            'spy_price': 350.0,
            'put_call_ratio': 4.0,
            'volume': 75000000,
            'spy_prices': [400, 380, 360, 350, 340],
            'qqq_prices': [360, 340, 320, 310, 300],
            'volume_data': [2000000, 10000000, 30000000, 50000000, 75000000]
        }
        
        analysis = await detector.analyze_current_regime(extreme_vol_data)
        
        assert analysis.volatility_regime == VolatilityRegime.EXTREME
        assert analysis.position_size_multiplier <= 0.5  # Should significantly reduce positions
        assert analysis.risk_multiplier >= 2.0  # Should increase risk limits
        assert analysis.confidence >= 0.8  # Should be confident about extreme conditions
    
    @pytest.mark.asyncio
    async def test_volatility_spike_position_sizing(self):
        """Test position sizing adjustments during volatility spikes"""
        detector = EnhancedRegimeDetector()
        
        # Normal volatility
        normal_data = {'vix': 20.0, 'spy_price': 420.0, 'put_call_ratio': 1.0}
        normal_analysis = await detector.analyze_current_regime(normal_data)
        
        # High volatility
        high_vol_data = {'vix': 40.0, 'spy_price': 420.0, 'put_call_ratio': 2.0}
        high_vol_analysis = await detector.analyze_current_regime(high_vol_data)
        
        # Position size should be reduced in high volatility
        assert high_vol_analysis.position_size_multiplier < normal_analysis.position_size_multiplier
        assert high_vol_analysis.risk_multiplier > normal_analysis.risk_multiplier

class TestPDTEdgeCases:
    """Test PDT compliance edge cases"""
    
    @pytest.mark.asyncio
    async def test_pdt_rolling_window_edge_case(self):
        """Test PDT rolling window edge cases"""
        tracker = PDTTracker()
        
        # Create positions across rolling window boundary
        base_time = datetime.now()
        
        # Position 1: 6 days ago (should not count)
        old_position_id = "OLD_001"
        tracker.record_position_open(
            position_id=old_position_id,
            symbol="SPY",
            contract_type="PUT",
            strike=400.0,
            expiration="2024-01-19",
            quantity=1,
            price=5.0,
            timestamp=base_time - timedelta(days=6)
        )
        tracker.record_position_close(
            position_id=old_position_id,
            price=3.0,
            timestamp=base_time - timedelta(days=6) + timedelta(hours=2)
        )
        
        # Position 2: 4 days ago (should count)
        recent_position_id = "RECENT_001"
        tracker.record_position_open(
            position_id=recent_position_id,
            symbol="SPY",
            contract_type="PUT",
            strike=400.0,
            expiration="2024-01-19",
            quantity=1,
            price=5.0,
            timestamp=base_time - timedelta(days=4)
        )
        tracker.record_position_close(
            position_id=recent_position_id,
            price=3.0,
            timestamp=base_time - timedelta(days=4) + timedelta(hours=2)
        )
        
        # Check PDT status
        status = tracker.get_current_pdt_status(5000)
        
        assert status.day_trades_used == 1  # Only recent position should count
        assert status.can_day_trade is True  # Should still be able to day trade
    
    @pytest.mark.asyncio
    async def test_pdt_emergency_day_trades(self):
        """Test emergency day trade allowance"""
        tracker = PDTTracker()
        
        # Use up all 3 day trades
        for i in range(3):
            position_id = f"REGULAR_{i}"
            tracker.record_position_open(
                position_id=position_id,
                symbol="SPY",
                contract_type="PUT",
                strike=400.0,
                expiration="2024-01-19",
                quantity=1,
                price=5.0,
                timestamp=datetime.now()
            )
            tracker.record_position_close(
                position_id=position_id,
                price=3.0,
                timestamp=datetime.now() + timedelta(hours=1)
            )
        
        # Check regular day trade limit
        can_trade_regular, _ = tracker.can_execute_day_trade(is_emergency=False)
        assert can_trade_regular is False
        
        # Check emergency day trade allowance
        can_trade_emergency, _ = tracker.can_execute_day_trade(is_emergency=True)
        assert can_trade_emergency is True  # Should allow emergency day trades

class TestResourceExhaustion:
    """Test system behavior under resource constraints"""
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test memory pressure handling"""
        # Simulate memory pressure by creating large data structures
        large_data = []
        
        try:
            # Create large dataset
            for i in range(100000):
                large_data.append({
                    'timestamp': datetime.now(),
                    'data': np.random.randn(1000).tolist(),
                    'metadata': f'large_dataset_{i}'
                })
            
            # Test that system can still function
            detector = EnhancedRegimeDetector()
            analysis = await detector.analyze_current_regime({'vix': 20.0, 'spy_price': 420.0})
            
            assert analysis is not None
            assert analysis.current_regime is not None
            
        except MemoryError:
            # System should handle memory errors gracefully
            pytest.fail("System failed to handle memory pressure")
        finally:
            # Clean up
            del large_data
    
    @pytest.mark.asyncio
    async def test_cpu_intensive_calculations(self):
        """Test CPU-intensive calculation handling"""
        detector = EnhancedRegimeDetector()
        
        # Simulate CPU-intensive regime analysis
        start_time = time.time()
        
        for i in range(100):
            analysis = await detector.analyze_current_regime({
                'vix': 20.0 + np.random.randn() * 5,
                'spy_price': 420.0 + np.random.randn() * 10,
                'put_call_ratio': 1.0 + np.random.randn() * 0.3
            })
        
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10.0  # 10 seconds for 100 calculations
        assert analysis is not None

class TestConcurrentOperations:
    """Test concurrent operation handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_order_placement(self):
        """Test concurrent order placement"""
        connector = AlpacaConnector()
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'id': 'test_order', 'status': 'new'})
        
        connector.session = AsyncMock()
        connector.session.post.return_value = mock_response
        
        # Create multiple orders
        orders = []
        for i in range(10):
            order = OptionsOrder(
                symbol=f'STOCK_{i}',
                qty=1,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                time_in_force=OrderTimeInForce.DAY,
                limit_price=100.0
            )
            orders.append(order)
        
        # Place orders concurrently
        tasks = [connector.place_order(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all orders were processed
        assert len(results) == 10
        assert all(not isinstance(result, Exception) for result in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_regime_analysis(self):
        """Test concurrent regime analysis"""
        detector = EnhancedRegimeDetector()
        
        # Create multiple market data scenarios
        market_scenarios = []
        for i in range(20):
            scenario = {
                'vix': 15.0 + np.random.randn() * 10,
                'spy_price': 400.0 + np.random.randn() * 50,
                'put_call_ratio': 1.0 + np.random.randn() * 0.5
            }
            market_scenarios.append(scenario)
        
        # Analyze regimes concurrently
        tasks = [detector.analyze_current_regime(scenario) for scenario in market_scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all analyses completed
        assert len(results) == 20
        assert all(not isinstance(result, Exception) for result in results)
        assert all(result.current_regime is not None for result in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
